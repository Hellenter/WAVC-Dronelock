#include "dronolok/netmsg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600   // inet_pton, inet_ntop — с Vista
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef int socklen_t;
#define DL_CLOSESOCK ::closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define DL_CLOSESOCK ::close
#endif

namespace dronolok {
namespace net {

namespace {

#if defined(__GNUC__)
void addf(std::string& s, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
#endif
void addf(std::string& s, const char* fmt, ...) {
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    if (n > 0) s.append(b, std::min<size_t>((size_t)n, sizeof(b) - 1));
}

// разбор чисел по порядку: strtod/strtol с проверкой, что что-то прочитано
struct Rd {
    const char* p;
    bool ok = true;
    double d() {
        char* e;
        const double v = std::strtod(p, &e);
        if (e == p || !std::isfinite(v)) ok = false;
        p = e;
        return v;
    }
    long l() {
        char* e;
        const long v = std::strtol(p, &e, 10);
        if (e == p) ok = false;
        p = e;
        return v;
    }
    unsigned long ul() {
        char* e;
        const unsigned long v = std::strtoul(p, &e, 10);
        if (e == p) ok = false;
        p = e;
        return v;
    }
    bool sep(char c) {   // пропустить пробелы и ожидаемый разделитель
        while (*p == ' ') ++p;
        if (*p != c) return false;
        ++p;
        return true;
    }
};

bool head(const std::string& s, const char* tag, Rd& r) {
    const size_t n = strlen(tag);
    if (s.size() < n || s.compare(0, n, tag) != 0) return false;
    r.p = s.c_str() + n;
    return true;
}

#if defined(_WIN32)
struct WsaInit {
    WsaInit() {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    }
};
#endif

}  // namespace

std::string encode(const WideMsg& m, int maxTargets) {
    const int n = std::min<int>((int)m.reps.size(), std::max(0, maxTargets));
    std::string s;
    s.reserve(64 + 72 * n);
    addf(s, "DLW1 %d %u %.6f %.6f %d", m.cam, m.seq, m.t, m.sent, n);
    for (int i = 0; i < n; ++i) {
        const TargetReport& r = m.reps[i];
        addf(s, ";%d %.5f %.5f %.4f %.4f %.5f %.4f %d %d %.4f %.3f", r.id, r.az, r.el, r.vaz, r.vel, r.sigma, r.sigma_v, r.cls,
             r.hard, r.size_deg, r.confidence);
    }
    return s;
}

std::string encode(const CueMsg& m) {
    std::string s;
    addf(s, "DLC1 %d %.6f %d", m.cam, m.t, (int)m.areas.size());
    for (const auto& a : m.areas) addf(s, ";%.5f %.5f %.5f", a.az, a.el, a.r);
    return s;
}
std::string encode(const BirdMsg& m) {
    std::string s;
    addf(s, "DLB1 %d %d", m.cam, m.local);
    return s;
}
std::string encode(const PingMsg& m) {
    std::string s;
    addf(s, "DLP1 %u %.6f", m.seq, m.t1);
    return s;
}
std::string encode(const PongMsg& m) {
    std::string s;
    addf(s, "DLQ1 %u %.6f %.6f %.6f", m.seq, m.t1, m.t2, m.t3);
    return s;
}

char kind(const std::string& s) {
    if (s.size() < 5 || s[0] != 'D' || s[1] != 'L' || s[3] != '1' || s[4] != ' ') return 0;
    const char k = s[2];
    return k == 'W' || k == 'C' || k == 'B' || k == 'P' || k == 'Q' ? k : 0;
}

bool decode(const std::string& s, WideMsg& m) {
    Rd r{nullptr};
    if (!head(s, "DLW1", r)) return false;
    m = WideMsg();
    m.cam = (int)r.l();
    m.seq = (uint32_t)r.ul();
    m.t = r.d();
    m.sent = r.d();
    const long n = r.l();
    if (!r.ok || n < 0 || n > 1000) return false;
    m.reps.resize((size_t)n);
    for (long i = 0; i < n; ++i) {
        if (!r.sep(';')) return false;
        TargetReport& t = m.reps[(size_t)i];
        t.id = (int)r.l();
        t.az = (float)r.d();
        t.el = (float)r.d();
        t.vaz = (float)r.d();
        t.vel = (float)r.d();
        t.sigma = (float)r.d();
        t.sigma_v = (float)r.d();
        t.cls = (int)r.l();
        t.hard = (int)r.l();
        t.size_deg = (float)r.d();
        t.confidence = (float)r.d();
        t.t = m.t;
        if (!r.ok) return false;
    }
    return true;
}

bool decode(const std::string& s, CueMsg& m) {
    Rd r{nullptr};
    if (!head(s, "DLC1", r)) return false;
    m = CueMsg();
    m.cam = (int)r.l();
    m.t = r.d();
    const long n = r.l();
    if (!r.ok || n < 0 || n > 1000) return false;
    for (long i = 0; i < n; ++i) {
        if (!r.sep(';')) return false;
        CueMsg::Area a;
        a.az = (float)r.d();
        a.el = (float)r.d();
        a.r = (float)r.d();
        if (!r.ok) return false;
        m.areas.push_back(a);
    }
    return true;
}

bool decode(const std::string& s, BirdMsg& m) {
    Rd r{nullptr};
    if (!head(s, "DLB1", r)) return false;
    m.cam = (int)r.l();
    m.local = (int)r.l();
    return r.ok;
}
bool decode(const std::string& s, PingMsg& m) {
    Rd r{nullptr};
    if (!head(s, "DLP1", r)) return false;
    m.seq = (uint32_t)r.ul();
    m.t1 = r.d();
    return r.ok;
}
bool decode(const std::string& s, PongMsg& m) {
    Rd r{nullptr};
    if (!head(s, "DLQ1", r)) return false;
    m.seq = (uint32_t)r.ul();
    m.t1 = r.d();
    m.t2 = r.d();
    m.t3 = r.d();
    return r.ok;
}

void ClockSync::add(double t1, double t2, double t3, double t4) {
    const double rtt = (t4 - t1) - (t3 - t2);
    const double off = ((t2 - t1) + (t3 - t4)) * 0.5;
    if (!(rtt >= 0) || !std::isfinite(off)) return;
    last_.push_back({off, rtt});
    if (last_.size() > 16) last_.erase(last_.begin());
    // лучший — самый быстрый обмен: у него меньше всего задержки в очередях, смещение точнее
    const S* b = &last_[0];
    for (const S& x : last_)
        if (x.rtt < b->rtt) b = &x;
    off_ = b->off;
    rtt_ = b->rtt;
    ++n_;
}

UdpSocket::UdpSocket() {
#if defined(_WIN32)
    static WsaInit wsa;
#endif
}
UdpSocket::~UdpSocket() { close(); }

void UdpSocket::close() {
    if (fd_ >= 0) DL_CLOSESOCK((int)fd_);
    fd_ = -1;
    port_ = 0;
}

bool UdpSocket::bind(const std::string& host, int port, std::string& err) {
    close();
    const intptr_t fd = (intptr_t)socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd < 0) {
        err = "не создать UDP-сокет";
        return false;
    }
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (host.empty() || host == "*") a.sin_addr.s_addr = htonl(INADDR_ANY);
    else if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
        DL_CLOSESOCK((int)fd);
        err = "адрес " + host + " не IPv4";
        return false;
    }
    if (::bind((int)fd, (sockaddr*)&a, sizeof(a)) != 0) {
        DL_CLOSESOCK((int)fd);
        err = "порт " + std::to_string(port) + " занят";
        return false;
    }
    socklen_t len = sizeof(a);
    getsockname((int)fd, (sockaddr*)&a, &len);
    fd_ = fd;
    port_ = ntohs(a.sin_port);
    return true;
}

bool UdpSocket::sendTo(const std::string& host, int port, const std::string& data) {
    if (fd_ < 0 || data.size() > 65000) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) return false;
    const auto n = sendto((int)fd_, data.data(), (int)data.size(), 0, (sockaddr*)&a, sizeof(a));
    return n >= 0 && (size_t)n == data.size();
}

bool UdpSocket::recv(std::string& out, int timeoutMs, std::string* fromHost, int* fromPort) {
    if (fd_ < 0) return false;
    fd_set rs;
    FD_ZERO(&rs);
    FD_SET((int)fd_, &rs);
    timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    if (select((int)fd_ + 1, &rs, nullptr, nullptr, &tv) <= 0) return false;
    char buf[65536];
    sockaddr_in a{};
    socklen_t len = sizeof(a);
    const auto n = recvfrom((int)fd_, buf, (int)sizeof(buf), 0, (sockaddr*)&a, &len);
    if (n < 0) return false;
    out.assign(buf, (size_t)n);
    if (fromHost) {
        char h[64] = "";
        inet_ntop(AF_INET, &a.sin_addr, h, sizeof(h));
        *fromHost = h;
    }
    if (fromPort) *fromPort = ntohs(a.sin_port);
    return true;
}

}  // namespace net
}  // namespace dronolok
