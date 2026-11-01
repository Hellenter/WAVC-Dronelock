// Дрон залетает за предмет (крону, угол дома, столб): захват не должен переходить на предмет,
// а после выхода дрона — снова на дроне. Синтетика: отметки детектора без картинки.
//  - «пятно»: пока дрон скрыт, рядом с его путём (сбоку, выше) мерцает неподвижный отклик
//    (heir_still: захват на прогнозе не передаётся стоящему треку);
//  - «кромка»: на пути дрона всё время стоит неподвижный отклик предмета, дрон в нём «тонет»
//    и выходит с другой стороны (stuck_obs: трек дрона не залипает на предмете);
//  - «зависание»: дрон плавно тормозит и висит — это не предмет, захват на нём;
//  - «второй дрон»: дрон залетел за кромку и не вылетел, а в кадре летит другой — захват на прогнозе
//    пропавшего уступает ему (stuck_yield), а не ждёт секунды; короткий заслон — захват остаётся на первом.
//    То же, когда кромка стоит с начала ролика (трек на её отклик не садится, still_reject).
//  - «вдоль кромки»: дрон летит вдоль верхушек крон, стоящих с начала ролика, и сливается с каждой —
//    трек не пересаживается на верхушку (still_reject).
#include <cmath>
#include <cstdio>
#include <random>

#include "dronolok/detector.hpp"

using namespace dronolok;

static int failures = 0;
#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            ++failures;                          \
            std::printf("FAIL: " __VA_ARGS__);   \
            std::printf("\n");                   \
        }                                        \
    } while (0)

static Candidate mk(float x, float y, float c) {
    Candidate q{};
    q.x = x;
    q.y = y;
    q.contrast = c;
    q.snr = 6;
    q.polarity = -1;
    q.energy = c * 9;
    q.margin = 2;
    return q;
}

struct Result {
    int onObject = 0;   // кадров захвата на предмете
    int back = -1;      // через сколько кадров после выхода захват снова на дроне (-1 — не вернулся)
    bool hardBefore = false;
};

// fps — частота камеры; hide — сколько секунд дрон скрыт; edge — предмет на пути (иначе пятно сбоку)
static Result run(float fps, float hide, bool edge, const TrackerConfig& base) {
    std::mt19937 rng(7);
    auto U = [&](float a, float b) { return a + (b - a) * (float)(rng() % 100000) / 100000.f; };
    Tracker t(1280, 720, tracker_for_fps(base, fps));
    const float v = 60.f / fps;   // 60 пикс./с
    const int t0 = (int)(8 * fps), occ = (int)std::lround(hide * fps);
    // предмет: на пути дрона (кромка, в которую он входит в t0) или сбоку от места пропажи
    const float ox = 100 + v * t0 + (edge ? 0.f : 70.f), oy = edge ? 300.f : 260.f;
    Result r;
    float x = 100, y = 300;
    for (int f = 0; f < t0 + occ + (int)(3 * fps); ++f) {
        x += v;
        std::vector<Candidate> d;
        const bool hidden = f >= t0 && f < t0 + occ;
        if (!hidden) d.push_back(mk(x + U(-.5f, .5f), y + U(-.5f, .5f), 30));
        // кромка видна всё время с того кадра, как дрон к ней подлетел (раньше — неотличима от неба);
        // пятно — только пока дрон скрыт, через 5 кадров после пропажи
        if (edge ? f >= t0 - (int)(0.4f * fps) : (f >= t0 + 5 && hidden)) d.push_back(mk(ox, oy, 20));
        t.update(d);
        if (f == t0 - 1) r.hardBefore = t.lockState() == LockState::Hard;
        if (f < t0 - (int)(0.4f * fps)) continue;
        float lx = 0, ly = 0;
        bool locked = false;
        for (const Track& tr : t.confirmed())
            if (tr.id == t.lockedId()) lx = tr.x, ly = tr.y, locked = true;
        if (!locked) continue;
        // на предмете: захваченный трек стоит у него, хотя дрон (или его прогноз) ушёл дальше 12 пикс.
        if (std::hypot(lx - ox, ly - oy) < 6 && std::hypot(x - ox, y - oy) > 12) ++r.onObject;
        if (f >= t0 + occ && r.back < 0 && std::hypot(lx - x, ly - y) < 12) r.back = f - (t0 + occ);
    }
    return r;
}

// Дрон плавно тормозит и зависает (за brake с), потом стоит hover с: «залипанием» это не считается,
// захват всё время на дроне
static void hover(float fps, float brake, float hover) {
    std::mt19937 rng(11);
    auto U = [&](float a, float b) { return a + (b - a) * (float)(rng() % 100000) / 100000.f; };
    Tracker t(1280, 720, tracker_for_fps(TrackerConfig(), fps));
    const float v0 = 90.f / fps;
    const int t0 = (int)(6 * fps), nb = (int)std::lround(brake * fps), nh = (int)std::lround(hover * fps);
    float x = 100, y = 300;
    int off = 0;
    for (int f = 0; f < t0 + nb + nh; ++f) {
        x += f < t0 ? v0 : f < t0 + nb ? v0 * (1.f - (float)(f - t0) / (float)nb) : 0.f;
        t.update({mk(x + U(-.5f, .5f), y + U(-.5f, .5f), 30)});
        if (f < t0) continue;
        bool on = false;
        for (const Track& tr : t.confirmed())
            on |= tr.id == t.lockedId() && std::hypot(tr.x - x, tr.y - y) < 12;
        off += !on;
    }
    std::printf("%2.0f к/с, торможение %.1f с: снят с «предмета» %d раз, кадров не на дроне %d\n", fps, brake, t.stuckEvents(), off);
    CHECK(t.stuckEvents() == 0, "%.0f к/с, торможение %.1f с: зависший дрон принят за предмет", fps, brake);
    CHECK(off == 0, "%.0f к/с, торможение %.1f с: захват не на дроне %d кадров", fps, brake, off);
}

// Отклик листвы: трек ползёт по кроне (0.8 пикс./кадр при 14 к/с), за 3 кадра перескакивает на 12 пикс. и стоит.
// «Скорость» прыжка — 4 пикс./кадр, но перед ним трек не летел (stuck_base, stuck_steady): это не дрон за
// предметом, и прогноз с такой скоростью не нужен — иначе трек листвы держал бы захват на прогнозе
static int foliage(const TrackerConfig& base) {
    const float fps = 14;
    Tracker t(1280, 720, tracker_for_fps(base, fps));
    float x = 560;
    for (int f = 0; f < 60; ++f) {
        if (f >= 20 && f < 23) x += 4.f;
        else if (f < 20) x += 0.8f;
        std::vector<Candidate> d{mk(x, 300, 30)};
        if (f % 2 == 0) d.push_back(mk(592, 290, 15));   // крона рядом мерцает (карта клаттера)
        t.update(d);
    }
    return t.stuckEvents();
}

// Первый дрон (захвачен) на t0 залетает за кромку на hide с (0 — не вылетает), второй летит в стороне с 3-й секунды.
// Кромка мерцает (каждый 4-й кадр) 3 с до того, как дрон к ней подлетел, и откликается вся, пока он за ней: трек
// дрона встаёт на её отклик и снимается с предмета (stuck_obs). Возвращает, через сколько кадров после пропажи
// захват на втором (-1 — не перешёл); back — захват в конце на первом, stuck — сколько раз трек снимался с предмета.
// standing — кромка стоит с начала ролика (видна 2 кадра из 3): трек на её отклик не садится (still_reject), и
// захват на прогнозе пропавшего должен уступать так же (мозаика 4608 m1: дрон ушёл за край четверти)
static int second(float fps, float hide, const TrackerConfig& base, bool* back = nullptr, int* stuck = nullptr,
                  bool standing = false) {
    std::mt19937 rng(5);
    auto U = [&](float a, float b) { return a + (b - a) * (float)(rng() % 100000) / 100000.f; };
    Tracker t(1280, 720, tracker_for_fps(base, fps));
    const float v = 60.f / fps, v2 = 40.f / fps;
    const int t0 = (int)(8 * fps), t2 = (int)(3 * fps), occ = hide > 0 ? (int)std::lround(hide * fps) : 1 << 20;
    const float ox = 100 + v * t0, oy = 300;
    float x = 100, y = 300, x2 = 900, y2 = 550;
    int sw = -1;
    for (int f = 0; f < t0 + (int)(6 * fps); ++f) {
        x += v;
        if (f >= t2) x2 -= v2;
        std::vector<Candidate> d;
        if (f < t0 || f >= t0 + occ) d.push_back(mk(x + U(-.5f, .5f), y + U(-.5f, .5f), 30));
        if (f >= t0 || (standing ? f % 3 != 0 : f >= t0 - (int)(3 * fps) && f % 4 == 0)) d.push_back(mk(ox + 4, oy, 20));
        if (f >= t2) d.push_back(mk(x2 + U(-.5f, .5f), y2 + U(-.5f, .5f), 25));
        t.update(d);
        if (f == t0 - 1) CHECK(t.lockState() == LockState::Hard, "%.0f к/с: до заслона нет жёсткого захвата", fps);
        if (back) *back = false;
        for (const Track& tr : t.confirmed())
            if (tr.id == t.lockedId()) {
                if (f >= t0 && sw < 0 && std::hypot(tr.x - x2, tr.y - y2) < 12) sw = f - t0;
                if (back) *back = std::hypot(tr.x - x, tr.y - y) < 12;
            }
    }
    if (stuck) *stuck = t.stuckEvents();
    return sw;
}

// Дрон летит вдоль кромки леса: на его пути верхушки крон — отклики, стоящие на месте с начала ролика
// (мерцают: видны 2 кадра из 3). Проходя верхушку ближе merge пикс., дрон с ней сливается (его отклика
// нет, есть только её). Трек дрона не должен пересаживаться на верхушку и оставаться на ней (still_reject);
// возвращает кадров захвата на верхушке, когда дрон уже дальше 12 пикс., на дроне — onDrone (после первой)
static int treeline(float fps, float merge, const TrackerConfig& base, int* onDrone = nullptr) {
    std::mt19937 rng(3);
    auto U = [&](float a, float b) { return a + (b - a) * (float)(rng() % 100000) / 100000.f; };
    Tracker t(1280, 720, tracker_for_fps(base, fps));
    const float v = 50.f / fps;
    std::vector<float> tops;
    for (float x = 420; x < 1100; x += 90) tops.push_back(x);
    float x = 100;
    const float y = 300, ty = 305;
    int onTop = 0, on = 0;
    for (int f = 0; x < 1150; ++f) {
        x += v;
        std::vector<Candidate> d;
        bool merged = false;
        for (float tx : tops) {
            const bool m = std::fabs(tx - x) < merge;
            merged |= m;
            if (f % 3 != 0 || m) d.push_back(mk(tx + U(-.3f, .3f), ty + U(-.3f, .3f), 18));
        }
        if (!merged) d.push_back(mk(x + U(-.5f, .5f), y + U(-.5f, .5f), 30));
        t.update(d);
        if (x < tops[0] - merge) continue;
        for (const Track& tr : t.confirmed())
            if (tr.id == t.lockedId()) {
                bool top = false;
                for (float tx : tops) top |= std::hypot(tr.x - tx, tr.y - ty) < 4 && std::hypot(x - tx, y - ty) > 12;
                onTop += top;
                on += std::hypot(tr.x - x, tr.y - y) < 12;
            }
    }
    if (onDrone) *onDrone = on;
    return onTop;
}

int main() {
    for (float fps : {10.f, 14.f, 25.f, 60.f})
        for (float merge : {8.f, 10.f}) {
            TrackerConfig off;
            off.still_reject = 0;
            int onA = 0, onB = 0;
            const int a = treeline(fps, merge, TrackerConfig(), &onA), b = treeline(fps, merge, off, &onB);
            std::printf("%2.0f к/с, вдоль кромки (сливается за %.0f пикс.): захват на верхушке %d кадров, на дроне %d (без правила %d и %d)\n",
                        fps, merge, a, onA, b, onB);
            CHECK(a <= (int)std::lround(0.1f * fps), "%.0f к/с, вдоль кромки %.0f: захват на верхушке %d кадров", fps, merge, a);
            CHECK(onA >= onB, "%.0f к/с, вдоль кромки %.0f: на дроне %d кадров, без правила %d", fps, merge, onA, onB);
        }
    for (float fps : {14.f, 25.f, 60.f})
        for (float brake : {0.5f, 1.f}) hover(fps, brake, 3.f);
    const TrackerConfig on;
    TrackerConfig off = on;
    off.stuck_obs = 0;
    off.heir_still = 0;
    for (float fps : {14.f, 25.f, 60.f})
        for (bool edge : {false, true})
            for (float hide : {0.8f, 1.6f}) {
                const Result a = run(fps, hide, edge, on), b = run(fps, hide, edge, off);
                std::printf("%2.0f к/с, %s, скрыт %.1f с: на предмете %d кадров (без правил %d), снова на дроне через %d (%d)\n",
                            fps, edge ? "кромка" : "пятно", hide, a.onObject, b.onObject, a.back, b.back);
                CHECK(a.hardBefore, "%.0f к/с: до заслона нет жёсткого захвата", fps);
                CHECK(a.onObject <= (int)std::lround(0.25f * fps), "%.0f к/с %s %.1f с: захват на предмете %d кадров", fps,
                      edge ? "кромка" : "пятно", hide, a.onObject);
                CHECK(a.back >= 0 && a.back <= (int)std::lround(0.5f * fps), "%.0f к/с %s %.1f с: на дрон через %d кадров", fps,
                      edge ? "кромка" : "пятно", hide, a.back);
            }
    {
        TrackerConfig loose;
        loose.stuck_steady = 0;
        const int a = foliage(TrackerConfig()), b = foliage(loose);
        std::printf("листва: снят с «предмета» %d раз (без проверки ровного полёта %d)\n", a, b);
        CHECK(a == 0, "трек листвы, перескочивший и вставший, принят за дрон за предметом (%d)", a);
    }
    for (float fps : {14.f, 25.f, 60.f}) {
        TrackerConfig hold;
        hold.stuck_yield = false;
        int stuck = 0;
        const int a = second(fps, 0, TrackerConfig(), nullptr, &stuck), b = second(fps, 0, hold);
        bool back = false;
        const int c = second(fps, 0.4f, TrackerConfig(), &back);
        std::printf("%2.0f к/с, второй дрон: захват на нём через %.1f с (без уступки %.1f с); заслон 0.4 с — %s\n", fps,
                    (float)a / fps, (float)b / fps, back ? "на первом" : c >= 0 ? "ушёл на второй" : "потерян");
        CHECK(stuck > 0, "%.0f к/с: трек первого дрона не снялся с кромки", fps);
        CHECK(a >= 0 && a <= (int)std::lround(1.5f * fps), "%.0f к/с: захват на прогнозе пропавшего не уступил второму (%d)", fps, a);
        CHECK(back, "%.0f к/с: после короткого заслона захват не на первом дроне", fps);
        const int s = second(fps, 0, TrackerConfig(), nullptr, nullptr, true);
        bool sback = false;
        second(fps, 0.4f, TrackerConfig(), &sback, nullptr, true);
        std::printf("%2.0f к/с, второй дрон, кромка стоит с начала: захват на нём через %.1f с; заслон 0.4 с — %s\n", fps,
                    (float)s / fps, sback ? "на первом" : "не на первом");
        CHECK(s >= 0 && s <= (int)std::lround(1.5f * fps),
              "%.0f к/с, кромка стоит с начала: захват на прогнозе пропавшего не уступил второму (%d)", fps, s);
        CHECK(sback, "%.0f к/с, кромка стоит с начала: после короткого заслона захват не на первом дроне", fps);
    }
    if (failures) std::printf("%d ошибок\n", failures);
    else std::printf("заслон: всё в порядке\n");
    return failures ? 1 : 0;
}
