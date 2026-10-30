#include "scenelink.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>

#include "platform.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef int socklen_t;
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSESOCK close
#endif

namespace fs = std::filesystem;

namespace scene {

namespace {

#if defined(_WIN32)
struct WsaInit {
    WsaInit() {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    }
};
#endif

fs::path u8(const std::string& s) { return fs::u8path(s); }
std::string s8(const fs::path& p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

}  // namespace

bool Link::start(const Options& o, const std::string& hello, std::string& err) {
    stop();
#if defined(_WIN32)
    static WsaInit wsa;
#endif
    // слушаем 127.0.0.1 на свободном порту, Godot подключается сам
    const auto ls = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    socklen_t al = sizeof(a);
    if ((intptr_t)ls < 0 || bind(ls, (sockaddr*)&a, sizeof(a)) != 0 || listen(ls, 1) != 0 ||
        getsockname(ls, (sockaddr*)&a, &al) != 0) {
        err = "не открыть порт для связи со сценой";
        if ((intptr_t)ls >= 0) CLOSESOCK(ls);
        return false;
    }
    lsock_ = (intptr_t)ls;
    const int port = ntohs(a.sin_port);
    std::vector<std::string> args = {o.godot, "--path", o.project, "--rendering-driver", "opengl3", "--audio-driver",
                                     "Dummy", "--", "--config", o.scene, "--live", std::to_string(port)};
    if (!o.sector.empty()) {
        args.push_back("--sector");
        args.push_back(o.sector);
    }
    proc_ = plat::proc_start(args, o.affinity, err);
    if (!proc_) {
        stop();
        return false;
    }
    // ждём подключения: сцена строится несколько секунд (лес, дома)
    for (int i = 0; i < 1200; ++i) {
        fd_set fs_;
        FD_ZERO(&fs_);
        FD_SET(ls, &fs_);
        timeval tv{0, 100000};
        if (select((int)ls + 1, &fs_, nullptr, nullptr, &tv) > 0) {
            const auto s = accept(ls, nullptr, nullptr);
            if ((intptr_t)s >= 0) {
                sock_ = (intptr_t)s;
                int one = 1;
                setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
                break;
            }
        }
        if (!plat::proc_alive(proc_)) {
            err = "Godot закрылся при запуске (сцена " + o.scene + ")";
            stop();
            return false;
        }
    }
    if (sock_ < 0) {
        err = "Godot не подключился за 2 минуты";
        stop();
        return false;
    }
    const std::string h = hello + "\n";
    if (!sendAll(h.data(), h.size())) {
        err = "связь со сценой оборвалась";
        stop();
        return false;
    }
    return true;
}

bool Link::sendAll(const char* p, size_t n) {
    while (n > 0) {
        const int k = send((decltype(socket(0, 0, 0)))sock_, p, (int)std::min<size_t>(n, 1 << 20), 0);
        if (k <= 0) return false;
        p += k;
        n -= k;
    }
    return true;
}

bool Link::recvLine(std::string& line) {
    for (;;) {
        const size_t e = buf_.find('\n');
        if (e != std::string::npos) {
            line = buf_.substr(0, e);
            buf_.erase(0, e + 1);
            return true;
        }
        char tmp[4096];
        const int k = recv((decltype(socket(0, 0, 0)))sock_, tmp, sizeof(tmp), 0);
        if (k <= 0) return false;
        buf_.append(tmp, k);
    }
}

bool Link::recvAll(uint8_t* p, size_t n) {
    const size_t from = std::min(n, buf_.size());   // часть кадра могла прийти вместе со строками
    memcpy(p, buf_.data(), from);
    buf_.erase(0, from);
    p += from;
    n -= from;
    while (n > 0) {
        const int k = recv((decltype(socket(0, 0, 0)))sock_, (char*)p, (int)std::min<size_t>(n, 1 << 22), 0);
        if (k <= 0) return false;
        p += k;
        n -= k;
    }
    return true;
}

bool Link::render(double t, int cam, double az, double el, std::vector<uint8_t>& px, int& w, int& h,
                  std::vector<Body>& bodies, std::string& err) {
    if (sock_ < 0) {
        err = "сцена не запущена";
        return false;
    }
    char q[200];
    snprintf(q, sizeof(q), "{\"t\": %.6f, \"cam\": %d, \"az\": %.5f, \"el\": %.5f}\n", t, cam, az, el);
    std::string line;
    long long n = 0;
    int k = 0;
    if (!sendAll(q, strlen(q)) || !recvLine(line) ||
        sscanf(line.c_str(), "FRAME %d %d %lld %d", &w, &h, &n, &k) != 4 || n != (long long)w * h) {
        err = "сцена не ответила кадром";
        return false;
    }
    bodies.clear();
    for (int i = 0; i < k; ++i) {
        if (!recvLine(line)) {
            err = "связь со сценой оборвалась";
            return false;
        }
        Body b;
        int bird = 0, hid = 0;
        if (sscanf(line.c_str(), "%d %d %d %f %lf %lf %lf %lf %lf %lf %d", &b.id, &bird, &b.shape, &b.size, &b.x, &b.y,
                   &b.z, &b.vx, &b.vy, &b.vz, &hid) == 11) {
            b.bird = bird != 0;
            b.hidden = hid != 0;
            bodies.push_back(b);
        }
    }
    px.resize((size_t)n);
    if (!recvAll(px.data(), (size_t)n)) {
        err = "связь со сценой оборвалась";
        return false;
    }
    return true;
}

void Link::stop() {
    if (sock_ >= 0) {
        sendAll("{\"quit\": 1}\n", 12);
        CLOSESOCK((decltype(socket(0, 0, 0)))sock_);
        sock_ = -1;
    }
    if (lsock_ >= 0) {
        CLOSESOCK((decltype(socket(0, 0, 0)))lsock_);
        lsock_ = -1;
    }
    if (proc_) {
        plat::proc_stop(proc_);
        proc_ = nullptr;
    }
    buf_.clear();
}

bool find_sim(const std::string& base, std::string& simDir, std::string& godot) {
    std::error_code ec;
    simDir.clear();
    const fs::path b = u8(base.empty() ? "." : base);
    for (const fs::path& c : {b / "sim", b.parent_path() / "sim", b.parent_path().parent_path() / "sim"})
        if (fs::exists(c / "godot" / "project.godot", ec) && fs::exists(c / "scenes", ec)) {
            simDir = s8(fs::absolute(c, ec));
            break;
        }
    if (simDir.empty()) return false;
    if (!godot.empty() && fs::exists(u8(godot), ec)) return true;
    godot.clear();
    if (const char* e = getenv("GODOT"))
        if (fs::exists(u8(e), ec)) godot = e;
    for (const char* n : {"Godot.exe", "godot.exe", "godot.x86_64", "godot"})
        if (godot.empty() && fs::exists(u8(simDir) / n, ec)) godot = s8(u8(simDir) / n);
#if !defined(_WIN32)
    if (godot.empty())
        for (const char* d : {"/usr/bin/godot", "/usr/local/bin/godot", "/usr/bin/godot4"})
            if (fs::exists(d, ec)) godot = d;
#endif
    return !godot.empty();
}

std::vector<std::string> list_scenes(const std::string& simDir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(u8(simDir) / "scenes", ec))
        if (e.path().extension() == ".json") out.push_back(s8(e.path().filename()));
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<Sector> scene_sectors(const std::string& path) {
    std::vector<Sector> out;
    std::ifstream f(u8(path), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    size_t p = s.find("\"sectors\"");
    if (p == std::string::npos) return out;
    p = s.find('[', p);
    const size_t e = p == std::string::npos ? p : s.find(']', p);
    if (e == std::string::npos) return out;
    const std::string arr = s.substr(p, e - p);
    static const std::regex obj("\\{[^{}]*\\}"), nm("\"name\"\\s*:\\s*\"([^\"]*)\""), ty("\"type\"\\s*:\\s*\"([^\"]*)\""),
        a0("\"az_from\"\\s*:\\s*(-?[0-9.]+)"), a1("\"az_to\"\\s*:\\s*(-?[0-9.]+)");
    for (std::sregex_iterator it(arr.begin(), arr.end(), obj), end; it != end; ++it) {
        const std::string o = it->str();
        std::smatch m;
        Sector sc;
        if (std::regex_search(o, m, nm) || std::regex_search(o, m, ty)) sc.name = m[1];
        if (std::regex_search(o, m, a0)) sc.from = (float)atof(m[1].str().c_str());
        sc.to = 360;
        if (std::regex_search(o, m, a1)) sc.to = (float)atof(m[1].str().c_str());
        float span = std::fmod(sc.to - sc.from + 720.f, 360.f);
        if (span == 0) span = 360;
        sc.mid = std::fmod(sc.from + span / 2 + 360.f, 360.f);
        out.push_back(sc);
    }
    return out;
}

}  // namespace scene
