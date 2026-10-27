#include "platform.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#include <objbase.h>
#include <psapi.h>
#else
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace plat {

int logical_cpus() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}

#if defined(_WIN32)

static std::string to_utf8(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

double process_cpu_seconds() {
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0;
    auto t = [](FILETIME f) { return ((uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime) * 1e-7; };
    return t(k) + t(u);
}

double process_memory_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return pmc.WorkingSetSize / (1024.0 * 1024.0);
}

std::string cpu_model() {
    wchar_t buf[256];
    DWORD sz = sizeof(buf);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     L"ProcessorNameString", RRF_RT_REG_SZ, nullptr, buf, &sz) == ERROR_SUCCESS) {
        std::string s = to_utf8(buf);
        while (!s.empty() && s.back() == ' ') s.pop_back();
        return s;
    }
    return "CPU";
}

uint64_t process_affinity() {
    DWORD_PTR pm = 0, sm = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return ~0ull;
    return pm;
}

bool set_process_affinity(uint64_t mask) {
    return mask && SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)mask);
}

static HANDLE g_job = nullptr;

bool cpu_cap_supported() { return true; }

bool set_cpu_cap(int percent) {
    if (!g_job) {
        if (percent >= 100) return true;
        g_job = CreateJobObjectW(nullptr, nullptr);
        if (!g_job || !AssignProcessToJobObject(g_job, GetCurrentProcess())) return false;
    }
    JOBOBJECT_CPU_RATE_CONTROL_INFORMATION info;
    memset(&info, 0, sizeof(info));
    if (percent < 100) {
        info.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
        info.CpuRate = (DWORD)(percent < 1 ? 1 : percent) * 100;   // в сотых долях процента
    }
    return SetInformationJobObject(g_job, JobObjectCpuRateControlInformation, &info, sizeof(info)) != 0;
}

void set_process_priority(int level) {
    static const DWORD cls[] = {IDLE_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
                                ABOVE_NORMAL_PRIORITY_CLASS, HIGH_PRIORITY_CLASS};
    SetPriorityClass(GetCurrentProcess(), cls[level < -2 ? 0 : level > 2 ? 4 : level + 2]);
}

static std::atomic<DWORD> g_dlgThread{0};

static std::string dialog_blocking(void* owner) {
    g_dlgThread = GetCurrentThreadId();
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    wchar_t file[MAX_PATH * 4] = L"";
    OPENFILENAMEW ofn;
    memset(&ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    // с владельцем диалог модальный для окна приложения: всегда поверх него, сворачивается вместе
    // с ним, а щелчки по окну приложения не копятся, пока диалог открыт
    ofn.hwndOwner = (HWND)owner;
    ofn.lpstrFilter = L"Видео (*.mp4;*.mkv;*.avi;*.mov;*.ts;*.y4m;*.webm)\0*.mp4;*.mkv;*.avi;*.mov;*.ts;*.y4m;*.webm\0"
                      L"Все файлы (*.*)\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 4;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    const bool ok = GetOpenFileNameW(&ofn) != 0;
    if (SUCCEEDED(co)) CoUninitialize();
    g_dlgThread = 0;
    return ok ? to_utf8(file) : "";
}

static BOOL CALLBACK raise_window(HWND w, LPARAM) {
    if (!IsWindowVisible(w)) return TRUE;
    if (IsIconic(w)) ShowWindow(w, SW_RESTORE);
    SetForegroundWindow(w);
    return FALSE;
}

static void dialog_to_front() {
    if (const DWORD t = g_dlgThread) EnumThreadWindows(t, raise_window, 0);
}

std::vector<std::string> ui_fonts() {
    wchar_t win[MAX_PATH];
    UINT n = GetWindowsDirectoryW(win, MAX_PATH);
    std::string dir = n ? to_utf8(win) + "\\Fonts\\" : "C:\\Windows\\Fonts\\";
    return {dir + "segoeui.ttf", dir + "arial.ttf", dir + "tahoma.ttf"};
}

void attach_console() {
    // вывод уже перенаправлен в файл или канал (скрипт, CI) — его и оставляем
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out && out != INVALID_HANDLE_VALUE && GetFileType(out) != FILE_TYPE_UNKNOWN) return;
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
        SetConsoleOutputCP(CP_UTF8);
    }
}

#else  // Linux и прочие unix

double process_cpu_seconds() {
    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

double process_memory_mb() {
    long pages = 0, rss = 0;
    if (FILE* f = fopen("/proc/self/statm", "r")) {
        if (fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
        fclose(f);
    }
    return rss * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
}

std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 10, "model name") == 0 || line.compare(0, 5, "Model") == 0) {
            size_t p = line.find(':');
            if (p != std::string::npos) return line.substr(p + 2);
        }
    }
    return "CPU";
}

uint64_t process_affinity() {
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    if (sched_getaffinity(0, sizeof(s), &s) != 0) return ~0ull;
    uint64_t m = 0;
    for (int i = 0; i < 64; ++i)
        if (CPU_ISSET(i, &s)) m |= 1ull << i;
    return m;
#else
    return ~0ull;
#endif
}

bool set_process_affinity(uint64_t mask) {
#if defined(__linux__)
    if (!mask) return false;
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int i = 0; i < 64; ++i)
        if (mask >> i & 1) CPU_SET(i, &s);
    // в Linux маска у каждого потока своя: проходим по всем потокам процесса
    bool ok = true;
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            ok &= sched_setaffinity((pid_t)atoi(e->d_name), sizeof(s), &s) == 0;
        }
        closedir(d);
    }
    return ok;
#else
    (void)mask;
    return false;
#endif
}

bool cpu_cap_supported() { return false; }
bool set_cpu_cap(int percent) { return percent >= 100; }

void set_process_priority(int level) { setpriority(PRIO_PROCESS, 0, -level * 5); }

static std::string dialog_blocking(void*) {
    FILE* p = popen("zenity --file-selection --title='Видео' 2>/dev/null || kdialog --getopenfilename . 2>/dev/null", "r");
    if (!p) return "";
    char buf[4096] = {0};
    std::string s = fgets(buf, sizeof(buf), p) ? buf : "";
    pclose(p);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static void dialog_to_front() {}

std::vector<std::string> ui_fonts() {
    return {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"};
}

void attach_console() {}

#endif

// ------------------------------------------------------------ диалог выбора видео в своём потоке
// Раньше диалог открывался прямо в кадре интерфейса: пока он был открыт, окно приложения не
// перерисовывалось, а щелчки по нему копились и после закрытия открывали диалог снова и снова.

static std::mutex g_dlgMu;
static std::atomic<int> g_dlgState{0};   // 0 — нет диалога, 1 — открыт, 2 — закрыт, результат не забран
static std::string g_dlgResult;

bool open_video_dialog(void* owner) {
    int expected = 0;
    if (!g_dlgState.compare_exchange_strong(expected, 1)) {
        if (expected == 1) dialog_to_front();
        return false;
    }
    std::thread([owner] {
        std::string p = dialog_blocking(owner);
        {
            std::lock_guard<std::mutex> lk(g_dlgMu);
            g_dlgResult = std::move(p);
        }
        g_dlgState = 2;
    }).detach();
    return true;
}

bool video_dialog_open() { return g_dlgState == 1; }

bool video_dialog_result(std::string& path) {
    if (g_dlgState != 2) return false;
    {
        std::lock_guard<std::mutex> lk(g_dlgMu);
        path = std::move(g_dlgResult);
        g_dlgResult.clear();
    }
    g_dlgState = 0;
    return true;
}

}  // namespace plat
