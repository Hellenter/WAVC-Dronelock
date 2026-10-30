// Отчёт мультитеста: results.json (всё, для разбора), results.csv (строка на прогон) и
// report.html — таблицы со сравнением с базой, открывается в любом браузере без сети.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <map>

#include "dronolok/fileutil.hpp"
#include "multitest.hpp"
#include "platform.hpp"

namespace mt {

namespace {

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += (char)c;
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", c);
            o += b;
        } else o += (char)c;
    }
    return o + "\"";
}

std::string html(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '<') o += "&lt;";
        else if (c == '>') o += "&gt;";
        else if (c == '&') o += "&amp;";
        else if (c == '"') o += "&quot;";
        else o += c;
    }
    return o;
}

std::string num(double v, int digits) {
    char b[48];
    snprintf(b, sizeof(b), "%.*f", digits, v);
    return b;
}

bool has(const Metric& m, const RunResult& r, double& v) { return metric_value(m, r, v); }

size_t metric_index(const Metric& m) { return (size_t)(&m - metrics().data()); }

std::string jnum(const Metric& m, const RunResult& r) {
    double v;
    return has(m, r, v) ? num(v, std::max(2, m.digits + 1)) : "null";
}

std::string now_str() {
    const std::time_t t = std::time(nullptr);
    char b[64];
    std::strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return b;
}

bool write_file(const std::string& path, const std::string& text) {
    FILE* f = dronolok::open_file(path, "wb");
    if (!f) return false;
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    return fclose(f) == 0 && ok;
}

std::string join_path(const std::string& dir, const char* name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    return dir + (last == '/' || last == '\\' ? "" : "/") + name;
}

// среднее по видео для варианта k (у метрик по разметке — по размеченным видео)
struct Avg {
    double v = 0;
    int n = 0;
    double lo = -1, hi = -1;   // все сдвиги такта: худший и лучший сдвиг (среднее по видео на каждом сдвиге)
};
Avg avg_of(const Metric& m, const std::vector<RunResult>& res, int k) {
    Avg a;
    for (const auto& r : res) {
        double v;
        if (r.variant == k && has(m, r, v)) {
            a.v += v;
            ++a.n;
        }
    }
    if (a.n) a.v /= a.n;
    return a;
}
// то же с разбросом по сдвигам: среднее по видео на сдвиге 0, 1, ... — меньшее и большее из них.
// runs — прогоны по сдвигам; видео, у которого такого сдвига нет (другая частота), в нём не участвует
Avg avg_spread(const Metric& m, const std::vector<RunResult>& res, const std::vector<RunResult>& runs, int k) {
    Avg a = avg_of(m, res, k);
    std::map<int, Avg> byPhase;
    for (const auto& r : runs) {
        double v;
        if (r.variant == k && r.phase >= 0 && has(m, r, v)) {
            byPhase[r.phase].v += v;
            ++byPhase[r.phase].n;
        }
    }
    if (byPhase.size() < 2) return a;
    for (auto& kv : byPhase) {
        const double v = kv.second.v / kv.second.n;
        if (a.lo < 0 || v < a.lo) a.lo = v;
        if (a.hi < 0 || v > a.hi) a.hi = v;
    }
    return a;
}

const RunResult* find_run(const std::vector<RunResult>& res, int video, int variant) {
    for (const auto& r : res)
        if (r.video == video && r.variant == variant) return &r;
    return nullptr;
}

// разброс по сдвигам такта под значением: «2.0…6.5» (lo < 0 — нет)
std::string spread_html(const Metric& m, double lo, double hi) {
    if (lo < 0 || hi < 0) return "";
    return "<small class=spr title=\"худший и лучший сдвиг такта\">" + num(lo, m.digits) + "…" + num(hi, m.digits) +
           "</small>";
}

// ячейка: значение и разница с базой, цвет — лучше/хуже; extra — что дописать под числом
std::string cell(const Metric& m, bool ok, double v, bool baseOk, double b, bool isBase, bool best = false,
                 const std::string& extra = "") {
    if (!ok) return "<td class=na data-v=\"\">—</td>";
    std::string cls = best ? "best" : "", d;
    if (!isBase && baseOk) {
        const double dv = v - b;
        const double tol = std::max(std::fabs(b) * 0.02, std::pow(10.0, -m.digits) * 0.5);
        if (std::fabs(dv) > tol) {
            d = std::string("<small>") + (dv > 0 ? "+" : "−") + num(std::fabs(dv), m.digits) + "</small>";
            if (m.better) cls += (dv > 0) == (m.better > 0) ? " good" : " bad";
        }
    }
    if (!cls.empty()) cls = " class=\"" + cls + "\"";
    return "<td" + cls + " data-v=\"" + num(v, 6) + "\">" + num(v, m.digits) + d + extra + "</td>";
}

const char* CSS = R"CSS(
:root{--bg:#f6f7f9;--fg:#1b1f24;--muted:#5d6670;--card:#fff;--line:#dde1e6;--head:#eef1f4;
--good:#0a7d38;--goodbg:#e3f5ea;--bad:#b42318;--badbg:#fdeceb;--acc:#2160c4;--best:#fff4c2}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111418;--fg:#e6e9ed;--muted:#9aa3ad;
--card:#191d22;--line:#2b3138;--head:#20262d;--good:#5fd38d;--goodbg:#14301f;--bad:#ff8a80;--badbg:#3a1a18;
--acc:#6ea8ff;--best:#3a3413}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 "Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif}
main{max-width:1400px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}h3{font-size:15px;margin:18px 0 6px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--head);position:sticky;top:0;cursor:pointer;font-weight:600;user-select:none}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:220px}
td small{display:block;font-size:11px;opacity:.85}
td.good{color:var(--good);background:var(--goodbg)}td.bad{color:var(--bad);background:var(--badbg)}
td.na{color:var(--muted)}td small.spr{color:var(--muted)}tr.base td:first-child{font-weight:700}td.best{outline:2px solid var(--acc);outline-offset:-2px}
.kpis{display:flex;flex-wrap:wrap;gap:10px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:150px}
.kpi b{display:block;font-size:20px}dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0}
dt{font-weight:600}dd{margin:0;color:var(--muted)}code{background:var(--head);padding:1px 5px;border-radius:4px}
ul{margin:6px 0;padding-left:20px}
)CSS";

const char* JS = R"JS(
document.querySelectorAll('table.sort').forEach(t=>{t.querySelectorAll('th').forEach((h,i)=>{h.title=h.title||'Сортировать';
h.onclick=()=>{const b=t.tBodies[0],rows=[...b.rows],asc=h.dataset.s!=='1';t.querySelectorAll('th').forEach(x=>delete x.dataset.s);
h.dataset.s=asc?'1':'0';rows.sort((a,c)=>{const x=a.cells[i].dataset.v,y=c.cells[i].dataset.v;
if(x===undefined)return asc?a.cells[i].textContent.localeCompare(c.cells[i].textContent):c.cells[i].textContent.localeCompare(a.cells[i].textContent);
if(x==='')return 1;if(y==='')return -1;return asc?x-y:y-x});rows.forEach(r=>b.appendChild(r))}})});
)JS";

}  // namespace

bool write_report(const std::string& dir, const Config& c, const std::vector<VideoMeta>& videos,
                  const std::vector<Variant>& variants, const std::vector<RunResult>& runs, bool finished,
                  double elapsedS, std::string& err) {
    const auto& M = metrics();
    // итог на «видео x вариант»: при всех сдвигах такта — среднее по сдвигам
    const std::vector<RunResult> res = merge_phases(runs);
    bool anyGt = false, anyPhases = false;
    int done = 0;
    for (const auto& r : runs) {
        anyGt |= r.done && r.gt;
        anyPhases |= r.phase >= 0;
        done += r.done;
    }
    // ---------------- JSON
    std::string j = "{\n  \"dronolok_multitest\": 1,\n";
    j += "  \"created\": " + jstr(now_str()) + ",\n";
    j += "  \"finished\": " + std::string(finished ? "true" : "false") + ",\n";
    j += "  \"elapsed_s\": " + num(elapsedS, 1) + ",\n";
    j += "  \"cpu\": " + jstr(plat::cpu_model()) + ",\n";
    j += "  \"folder\": " + jstr(c.folder) + ",\n";
    j += "  \"mode\": " + jstr(c.mode == Mode::Grid ? "grid" : "one") + ",\n";
    j += "  \"frames_limit\": " + std::to_string(c.maxFrames) + ", \"warmup\": " + std::to_string(c.warmup) +
         ", \"radius_px\": " + num(c.radius, 1) + ", \"parallel\": " + std::to_string(c.parallel) +
         ", \"realtime\": " + (c.realtime ? "true" : "false") + ", \"emulate\": " + (c.emulate ? "true" : "false") +
         ", \"mask_file\": " + (c.maskFile ? "true" : "false") + ", \"all_phases\": " +
         (c.allPhases ? "true" : "false") + ",\n";
    j += "  \"vary\": {";
    for (size_t i = 0; i < c.choices.size(); ++i) {
        j += (i ? ", " : "") + jstr(c.choices[i].key) + ": [";
        for (size_t k = 0; k < c.choices[i].values.size(); ++k) j += (k ? ", " : "") + num(c.choices[i].values[k], 4);
        j += "]";
    }
    j += "},\n  \"base\": {";
    {
        bool first = true;
        for (const auto& p : params()) {
            if (std::string(p.key) == "profile") continue;
            j += (first ? "" : ", ") + jstr(p.key) + ": " + num(p.get(c.base), 4);
            first = false;
        }
    }
    j += "},\n  \"metrics\": {";
    for (size_t i = 0; i < M.size(); ++i)
        j += std::string(i ? ", " : "") + jstr(M[i].key) + ": {\"name\": " + jstr(M[i].name) +
             ", \"better\": " + std::to_string(M[i].better) + ", \"gt_only\": " + (M[i].gtOnly ? "true" : "false") + "}";
    j += "},\n  \"videos\": [\n";
    for (size_t i = 0; i < videos.size(); ++i) {
        const auto& v = videos[i];
        j += "    {\"index\": " + std::to_string(i) + ", \"file\": " + jstr(v.file) + ", \"w\": " + std::to_string(v.w) +
             ", \"h\": " + std::to_string(v.h) + ", \"fps\": " + num(v.fps, 3) + ", \"frames\": " +
             std::to_string(v.frames) + ", \"gt\": " + (v.gt ? "true" : "false") + ", \"no_drones\": " + (v.noDrones ? "true" : "false") + "}" +
             (i + 1 < videos.size() ? ",\n" : "\n");
    }
    j += "  ],\n  \"variants\": [\n";
    for (size_t i = 0; i < variants.size(); ++i) {
        j += "    {\"index\": " + std::to_string(i) + ", \"name\": " + jstr(variants[i].name) + ", \"changes\": {";
        for (size_t k = 0; k < variants[i].changes.size(); ++k)
            j += (k ? ", " : "") + jstr(variants[i].changes[k].first) + ": " + num(variants[i].changes[k].second, 4);
        j += "}}" + std::string(i + 1 < variants.size() ? ",\n" : "\n");
    }
    // прогон: метрики; при всех сдвигах такта — среднее по сдвигам, phases и spread {метрика: [худший, лучший]}
    auto run_json = [&](const RunResult& r) {
        std::string o = "{\"video\": " + std::to_string(r.video) + ", \"variant\": " + std::to_string(r.variant);
        if (r.phase >= 0) o += ", \"phase\": " + std::to_string(r.phase);
        if (r.phases > 1) o += ", \"phases\": " + std::to_string(r.phases);
        const int fd = r.phases > 1 ? 1 : 0;
        o += ", \"done\": " + std::string(r.done ? "true" : "false") + ", \"error\": " + jstr(r.error) +
             ", \"gt\": " + (r.gt ? "true" : "false") + ", \"wall_s\": " + num(r.wallS, 2) + ", \"eval_frames\": " +
             num(r.evalFrames, fd) + ", \"gt_drone_frames\": " + num(r.gtDroneFrames, fd);
        for (const auto& m : M) o += ", " + jstr(m.key) + ": " + jnum(m, r);
        if (r.phases > 1 && r.lo.size() == M.size()) {
            o += ", \"spread\": {";
            bool first = true;
            for (size_t i = 0; i < M.size(); ++i) {
                if (r.lo[i] < 0 || !r.done) continue;
                const int dg = std::max(2, M[i].digits + 1);
                o += std::string(first ? "" : ", ") + jstr(M[i].key) + ": [" + num(r.lo[i], dg) + ", " + num(r.hi[i], dg) + "]";
                first = false;
            }
            o += "}";
        }
        return o + "}";
    };
    j += "  ],\n  \"runs\": [\n";
    bool firstRun = true;
    for (const auto& r : res) {
        if (!r.done && r.error.empty()) continue;   // ещё не дошли
        j += std::string(firstRun ? "" : ",\n") + "    " + run_json(r);
        firstRun = false;
    }
    if (anyPhases) {
        // каждый сдвиг такта отдельно — для разбора
        j += "\n  ],\n  \"phase_runs\": [\n";
        firstRun = true;
        for (const auto& r : runs) {
            if (!r.done && r.error.empty()) continue;
            j += std::string(firstRun ? "" : ",\n") + "    " + run_json(r);
            firstRun = false;
        }
    }
    j += "\n  ],\n  \"summary\": [\n";
    for (size_t k = 0; k < variants.size(); ++k) {
        j += "    {\"variant\": " + std::to_string(k);
        std::string spr;
        for (const auto& m : M) {
            const Avg a = avg_spread(m, res, runs, (int)k);
            const int dg = std::max(2, m.digits + 1);
            j += ", " + jstr(m.key) + ": " + (a.n ? num(a.v, dg) : "null");
            if (a.n && a.lo >= 0) spr += (spr.empty() ? "" : ", ") + jstr(m.key) + ": [" + num(a.lo, dg) + ", " + num(a.hi, dg) + "]";
        }
        if (anyPhases) j += ", \"spread\": {" + spr + "}";
        j += "}" + std::string(k + 1 < variants.size() ? ",\n" : "\n");
    }
    j += "  ]\n}\n";

    // ---------------- CSV (разделитель ';' — Excel с русской локалью открывает сразу)
    std::string csv = "\xEF\xBB\xBFvideo;variant;changes;done;error";
    for (const auto& m : M) csv += std::string(";") + m.key;
    if (anyPhases) {   // все сдвиги такта: сколько сдвигов и худший/лучший по каждой метрике
        csv += ";phases";
        for (const auto& m : M) csv += std::string(";") + m.key + "_lo;" + m.key + "_hi";
    }
    csv += "\n";
    for (const auto& r : res) {
        if (!r.done && r.error.empty()) continue;
        std::string ch;
        for (const auto& kv : variants[r.variant].changes) ch += (ch.empty() ? "" : " ") + kv.first + "=" + num(kv.second, 4);
        auto q = [](std::string s) {
            for (char& x : s)
                if (x == ';' || x == '\n') x = ',';
            return s;
        };
        csv += q(videos[r.video].file) + ";" + q(variants[r.variant].name) + ";" + ch + ";" + (r.done ? "1" : "0") + ";" +
               q(r.error);
        for (const auto& m : M) {
            double v;
            csv += ";";
            if (has(m, r, v)) csv += num(v, std::max(2, m.digits + 1));
        }
        if (anyPhases) {
            csv += ";" + std::to_string(r.phases);
            for (size_t i = 0; i < M.size(); ++i) {
                const bool ok = r.done && r.phases > 1 && r.lo.size() == M.size() && r.lo[i] >= 0;
                const int dg = std::max(2, M[i].digits + 1);
                csv += ok ? ";" + num(r.lo[i], dg) + ";" + num(r.hi[i], dg) : ";;";
            }
        }
        csv += "\n";
    }

    // ---------------- HTML
    std::vector<const Metric*> cols;
    for (const auto& m : M)
        if (anyGt || !m.gtOnly) cols.push_back(&m);
    // у видео без разметки столбцы по разметке пустые — их не показываем
    std::vector<const Metric*> colsNoGt;
    for (const Metric* m : cols)
        if (!m->gtOnly) colsNoGt.push_back(m);
    auto header = [&](const std::vector<const Metric*>& cs) {
        std::string h = "<thead><tr><th>Вариант</th>";
        for (const Metric* m : cs)
            h += "<th title=\"" + html(m->help) + "\">" + html(m->name) + (m->better > 0 ? " ↑" : m->better < 0 ? " ↓" : "") + "</th>";
        return h + "</tr></thead>";
    };
    // лучшее значение в столбце сводки — рамкой
    std::map<std::string, int> bestOf;
    for (const Metric* m : cols) {
        if (!m->better) continue;
        int bk = -1;
        double bv = 0;
        for (size_t k = 0; k < variants.size(); ++k) {
            const Avg a = avg_of(*m, res, (int)k);
            if (!a.n) continue;
            if (bk < 0 || (m->better > 0 ? a.v > bv : a.v < bv)) {
                bk = (int)k;
                bv = a.v;
            }
        }
        bestOf[m->key] = bk;
    }

    std::string h = "<!doctype html><html lang=ru><head><meta charset=utf-8>"
                    "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
                    "<title>Мультитест Дронолока</title><style>";
    h += CSS;
    h += "</style></head><body><main>";
    h += "<h1>Мультитест Дронолока</h1><div class=muted>" + html(now_str()) + " · " + html(plat::cpu_model()) + " · " +
         html(c.folder) + "</div>";
    h += "<div class=kpis style=\"margin-top:14px\">";
    h += "<div class=kpi><span class=muted>Состояние</span><b>" +
         std::string(finished ? (done == (int)runs.size() ? "готово" : "остановлен") : "идёт") + "</b></div>";
    h += "<div class=kpi><span class=muted>Видео</span><b>" + std::to_string(videos.size()) + "</b></div>";
    h += "<div class=kpi><span class=muted>Вариантов</span><b>" + std::to_string(variants.size()) + "</b></div>";
    h += "<div class=kpi><span class=muted>Прогонов сделано</span><b>" + std::to_string(done) + " из " +
         std::to_string(runs.size()) + "</b></div>";
    h += "<div class=kpi><span class=muted>Время</span><b>" + num(elapsedS / 60, 1) + " мин</b></div></div>";

    // вывод
    h += "<h2>Вывод</h2><div class=card>";
    const Metric* score = nullptr;
    for (const auto& m : M)
        if (std::string(m.key) == "score") score = &m;
    if (anyGt && score) {
        const Avg base = avg_of(*score, res, 0);
        int bk = bestOf.count("score") ? bestOf["score"] : -1;
        if (bk >= 0 && base.n) {
            const Avg best = avg_of(*score, res, bk);
            if (bk == 0)
                h += "<p>Лучший результат по оценке — у <b>базы</b> (" + num(base.v, 1) +
                     "): ни одно изменение в среднем по размеченным видео не помогло.</p>";
            else
                h += "<p>Лучший вариант по оценке: <b>" + html(variants[bk].name) + "</b> — " + num(best.v, 1) +
                     " против " + num(base.v, 1) + " у базы (" + (best.v >= base.v ? "+" : "−") +
                     num(std::fabs(best.v - base.v), 1) + "), в среднем по " + std::to_string(best.n) +
                     " размеченным видео.</p>";
        }
        if (c.mode == Mode::OneAtATime) {
            // по каждой настройке — какое значение лучше базы
            std::map<std::string, std::vector<int>> byKey;
            for (size_t k = 1; k < variants.size(); ++k)
                if (variants[k].changes.size() == 1) byKey[variants[k].changes[0].first].push_back((int)k);
            std::string li;
            for (const auto& kv : byKey) {
                int bk2 = -1;
                double bv = -1e9;
                for (int k : kv.second) {
                    const Avg a = avg_of(*score, res, k);
                    if (a.n && a.v > bv) {
                        bv = a.v;
                        bk2 = k;
                    }
                }
                if (bk2 < 0 || !base.n) continue;
                const double d = bv - base.v;
                const Param* p = find_param(kv.first);
                std::string nm = p ? p->name : kv.first;
                const size_t br = nm.find(" (");
                if (br != std::string::npos) nm = nm.substr(0, br);
                const std::string bvs = p ? fmt_value(*p, p->get(c.base)) : "";
                if (d > 0.5)
                    li += "<li><b>" + html(nm) + "</b>: лучше " + html(p ? fmt_value(*p, variants[bk2].changes[0].second) : "") +
                          " (+" + num(d, 1) + " к оценке; в базе " + html(bvs) + ")</li>";
                else
                    li += "<li><b>" + html(nm) + "</b>: оставить как в базе (" + html(bvs) + "), другие значения не лучше" +
                          (d < -0.5 ? " (лучшее из них −" + num(-d, 1) + ")" : "") + "</li>";
            }
            if (!li.empty()) h += "<p>По настройкам, каждая отдельно от остальных:</p><ul>" + li + "</ul>";
        }
        h += "<p class=muted>Оценка: " + html(score->help) +
             " Сочетание лучших значений по отдельности не всегда лучшее вместе — проверьте его режимом «все сочетания».</p>";
    } else {
        h += "<p>Разметки (<code>&lt;видео&gt;.gt.json</code>) нет ни у одного видео, поэтому точность не посчитать: "
             "видно только, как варианты меняют число целей, долю кадров с захватом и смены захвата. Разметить ролик "
             "можно в приложении на вкладке «Разметка» — тогда появятся «дрон найден», «захват на дроне», «ложный "
             "захват» и оценка.</p>";
    }
    h += "<p class=muted>Для разбора пришлите файл <code>results.json</code> из этой же папки.</p></div>";

    // сводка
    h += "<h2>Сводка по вариантам</h2><div class=muted>Среднее по видео" +
         std::string(anyGt ? " (метрики по разметке — по размеченным видео)" : "") +
         ". Под числом — разница с базой: зелёный — лучше, красный — хуже. Рамка — лучшее в столбце. " +
         (anyPhases ? "Все сдвиги такта: число — среднее по сдвигам, ниже серым — худший…лучший сдвиг (среднее по "
                      "видео на каждом сдвиге). "
                    : "") +
         "Щелчок по заголовку сортирует.</div>";
    h += "<div class=wrap><table class=sort>" + header(cols) + "<tbody>";
    for (size_t k = 0; k < variants.size(); ++k) {
        h += std::string("<tr") + (k == 0 ? " class=base" : "") + "><td>" + html(variants[k].name) + "</td>";
        for (const Metric* m : cols) {
            const Avg a = avg_spread(*m, res, runs, (int)k), b = avg_of(*m, res, 0);
            const bool best = bestOf.count(m->key) && bestOf[m->key] == (int)k && variants.size() > 1;
            h += cell(*m, a.n > 0, a.v, b.n > 0, b.v, k == 0, best, spread_html(*m, a.lo, a.hi));
        }
        h += "</tr>";
    }
    h += "</tbody></table></div>";

    // по видео
    h += "<h2>По видео</h2>";
    for (size_t vi = 0; vi < videos.size(); ++vi) {
        const auto& v = videos[vi];
        h += "<h3>" + html(v.file) + "</h3><div class=muted>" + std::to_string(v.w) + "x" + std::to_string(v.h) + ", " +
             num(v.fps, 2) + " к/с, " + (v.frames ? "~" + std::to_string(v.frames) + " кадров, " : "") +
             (v.noDrones ? "разметка: дронов нет" : v.gt ? "есть разметка" : "без разметки") + "</div>";
        const RunResult* br = find_run(res, (int)vi, 0);
        const auto& vc = v.gt ? cols : colsNoGt;
        h += "<div class=wrap><table class=sort>" + header(vc) + "<tbody>";
        for (size_t k = 0; k < variants.size(); ++k) {
            const RunResult* r = find_run(res, (int)vi, (int)k);
            h += std::string("<tr") + (k == 0 ? " class=base" : "") + "><td>" + html(variants[k].name);
            if (r && !r->error.empty()) h += " <span class=muted>(" + html(r->error) + ")</span>";
            h += "</td>";
            for (const Metric* m : vc) {
                double val = 0, b = 0;
                const bool ok = r && has(*m, *r, val);
                const bool bok = br && has(*m, *br, b);
                const size_t mi = metric_index(*m);
                const bool spr = ok && r->phases > 1 && mi < r->lo.size();
                h += cell(*m, ok, val, bok, b, k == 0, false, spr ? spread_html(*m, r->lo[mi], r->hi[mi]) : "");
            }
            h += "</tr>";
        }
        h += "</tbody></table></div>";
    }

    // пояснения
    h += "<h2>Что значат столбцы</h2><div class=card><dl>";
    for (const Metric* m : cols) h += "<dt>" + html(m->name) + "</dt><dd>" + html(m->help) + "</dd>";
    h += "</dl><p class=muted>↑ — больше лучше, ↓ — меньше лучше. Цель «найдена», если трек ближе " + num(c.radius, 0) +
         " пикс. к разметке; первые " + std::to_string(c.warmup) +
         " кадров (обучение маски неба) в точность не входят. ";
    h += c.maskFile ? "Память помех между прогонами не переносится, маска неба берётся из файла."
                    : "Чистый старт: каждый прогон — без накопленного опыта, память помех и маска неба "
                      "учатся заново и в файлы рядом с видео не пишутся.";
    h += c.realtime ? " Кадры шли в темпе видео, как с камеры: что не успели — пропущено."
                    : " Кадры шли без пропусков, так быстро, как успевает ПК.";
    if (c.allPhases)
        h += " Все сдвиги такта: у камеры реже видео (шаг кадров, частота камеры) каждый вариант прогонялся с "
             "каждым сдвигом — с какого кадра файла пошёл такт (шаг 4 — 4 прогона, 14 к/с из 60 — 5); в таблицах "
             "среднее по сдвигам и разброс от худшего до лучшего.";
    if (c.parallel > 1)
        h += " Прогонов шло одновременно " + std::to_string(c.parallel) + " — время обработки выше, чем при одном.";
    h += "</p></div>";

    // настройки
    h += "<h2>Настройки</h2><div class=card><p>Перебор: " +
         std::string(c.mode == Mode::Grid ? "все сочетания" : "каждая настройка отдельно, остальные — как в базе") +
         (c.maxFrames > 0 ? ", до " + std::to_string(c.maxFrames) + " кадров с видео" : ", видео целиком") + ".</p><ul>";
    for (const auto& ch : c.choices) {
        const Param* p = find_param(ch.key);
        if (!p) continue;
        std::string vals;
        for (double v : ch.values) vals += (vals.empty() ? "" : ", ") + fmt_value(*p, v);
        h += "<li>" + html(p->name) + " <code>" + html(p->key) + "</code>: " + html(vals) + " (в базе " +
             html(std::string(p->key) == "profile" ? "свой набор" : fmt_value(*p, p->get(c.base))) + ")</li>";
    }
    h += "</ul><h3>База</h3><dl>";
    for (const auto& p : params()) {
        if (std::string(p.key) == "profile") continue;
        h += "<dt>" + html(p.name) + "</dt><dd>" + html(fmt_value(p, p.get(c.base))) + "</dd>";
    }
    h += "</dl></div><script>";
    h += JS;
    h += "</script></main></body></html>\n";

    bool ok = write_file(join_path(dir, "results.json"), j);
    ok &= write_file(join_path(dir, "results.csv"), csv);
    ok &= write_file(join_path(dir, "report.html"), h);
    if (!ok) err = "не удалось записать отчёт в " + dir;
    return ok;
}

}  // namespace mt
