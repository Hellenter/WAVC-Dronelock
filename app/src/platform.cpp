#include "platform.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
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
#include <shlobj.h>
#include <shellapi.h>
#include <psapi.h>
#else
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
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
        // дочерние процессы (отрисовка сцены) могут выйти из-под потолка: он только для алгоритма
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li;
        memset(&li, 0, sizeof(li));
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_BREAKAWAY_OK;
        SetInformationJobObject(g_job, JobObjectExtendedLimitInformation, &li, sizeof(li));
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
    ofn.lpstrFilter = L"Видео (*.mp4;*.mkv;*.avi;*.mov;*.ts;*.y4m;*.webm;*.h264;*.h265;*.mjpeg)\0"
                      L"*.mp4;*.mkv;*.avi;*.mov;*.ts;*.y4m;*.webm;*.m4v;*.mts;*.h264;*.264;*.h265;*.hevc;*.265;*.mjpeg;*.mjpg\0"
                      L"Все файлы (*.*)\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 4;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    const bool ok = GetOpenFileNameW(&ofn) != 0;
    if (SUCCEEDED(co)) CoUninitialize();
    g_dlgThread = 0;
    return ok ? to_utf8(file) : "";
}

static std::string folder_blocking(void* owner) {
    g_dlgThread = GetCurrentThreadId();
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    BROWSEINFOW bi;
    memset(&bi, 0, sizeof(bi));
    bi.hwndOwner = (HWND)owner;
    bi.lpszTitle = L"Папка с видео для мультитеста";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    std::string out;
    if (PIDLIST_ABSOLUTE id = SHBrowseForFolderW(&bi)) {
        wchar_t path[MAX_PATH * 4] = L"";
        if (SHGetPathFromIDListW(id, path)) out = to_utf8(path);
        CoTaskMemFree(id);
    }
    if (SUCCEEDED(co)) CoUninitialize();
    g_dlgThread = 0;
    return out;
}

bool open_in_browser(const std::string& path) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], n);
    return (INT_PTR)ShellExecuteW(nullptr, L"open", w.c_str(), nullptr, nullptr, SW_SHOWNORMAL) > 32;
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

static std::wstring widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(n > 0 ? n - 1 : 0);
    return w;
}

// аргумент в кавычках по правилам CommandLineToArgvW: обратные косые перед кавычкой удваиваются,
// кавычка внутри экранируется («D:\Godot\» или имя сектора с кавычкой не склеивают аргументы)
static void quote_arg(const std::wstring& a, std::wstring& cmd) {
    cmd += L'"';
    size_t bs = 0;
    for (wchar_t c : a) {
        if (c == L'\\') {
            ++bs;
            continue;
        }
        cmd.append(c == L'"' ? bs * 2 + 1 : bs, L'\\');
        bs = 0;
        cmd += c;
    }
    cmd.append(bs * 2, L'\\');
    cmd += L'"';
}

void* proc_start(const std::vector<std::string>& args, uint64_t affinity, std::string& err) {
    std::wstring cmd;
    for (const auto& a : args) {
        if (!cmd.empty()) cmd += L' ';
        quote_arg(widen(a), cmd);
    }
    STARTUPINFOW si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    DWORD fl = CREATE_NO_WINDOW | CREATE_SUSPENDED;
    BOOL ok = CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, fl | CREATE_BREAKAWAY_FROM_JOB, nullptr,
                             nullptr, &si, &pi);
    if (!ok) ok = CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, fl, nullptr, nullptr, &si, &pi);
    if (!ok) {
        err = "не запустился " + (args.empty() ? std::string() : args[0]) + " (ошибка " + std::to_string(GetLastError()) + ")";
        return nullptr;
    }
    if (affinity) SetProcessAffinityMask(pi.hProcess, (DWORD_PTR)affinity);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

bool proc_alive(void* p) { return p && WaitForSingleObject((HANDLE)p, 0) == WAIT_TIMEOUT; }

void proc_stop(void* p) {
    if (!p) return;
    if (proc_alive(p) && WaitForSingleObject((HANDLE)p, 1500) == WAIT_TIMEOUT) TerminateProcess((HANDLE)p, 1);
    CloseHandle((HANDLE)p);
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

static std::string folder_blocking(void*) {
    FILE* p = popen("zenity --file-selection --directory --title='Папка с видео' 2>/dev/null || "
                    "kdialog --getexistingdirectory . 2>/dev/null", "r");
    if (!p) return "";
    char buf[4096] = {0};
    std::string s = fgets(buf, sizeof(buf), p) ? buf : "";
    pclose(p);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

bool open_in_browser(const std::string& path) {
    std::string q = "'";
    for (char c : path) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    q += "'";
    return system(("xdg-open " + q + " >/dev/null 2>&1 &").c_str()) == 0;
}

static void dialog_to_front() {}

std::vector<std::string> ui_fonts() {
    return {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"};
}

void attach_console() {}

void* proc_start(const std::vector<std::string>& args, uint64_t affinity, std::string& err) {
    if (args.empty()) return nullptr;
    const pid_t pid = fork();
    if (pid < 0) {
        err = "fork не удался";
        return nullptr;
    }
    if (pid == 0) {
        if (affinity) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int i = 0; i < 64; ++i)
                if (affinity >> i & 1) CPU_SET(i, &set);
            sched_setaffinity(0, sizeof(set), &set);
        }
        std::vector<char*> av;
        for (const auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        execvp(av[0], av.data());
        _exit(127);
    }
    return new pid_t(pid);
}

bool proc_alive(void* p) {
    if (!p) return false;
    pid_t* pid = (pid_t*)p;
    if (*pid <= 0) return false;
    int st = 0;
    if (waitpid(*pid, &st, WNOHANG) == 0) return true;
    *pid = 0;
    return false;
}

void proc_stop(void* p) {
    if (!p) return;
    pid_t* pid = (pid_t*)p;
    if (proc_alive(p)) {
        kill(*pid, SIGTERM);
        for (int i = 0; i < 150 && proc_alive(p); ++i) usleep(10000);
        if (proc_alive(p)) kill(*pid, SIGKILL);
        if (*pid > 0) waitpid(*pid, nullptr, 0);
    }
    delete pid;
}

#endif

// ------------------------------------------------------------ диалог выбора видео в своём потоке
// Раньше диалог открывался прямо в кадре интерфейса: пока он был открыт, окно приложения не
// перерисовывалось, а щелчки по нему копились и после закрытия открывали диалог снова и снова.

namespace {
struct Dialog {
    std::mutex mu;
    std::atomic<int> state{0};   // 0 — нет диалога, 1 — открыт, 2 — закрыт, результат не забран
    std::string result;

    bool open(std::string (*run)(void*), void* owner) {
        int expected = 0;
        if (!state.compare_exchange_strong(expected, 1)) {
            if (expected == 1) dialog_to_front();
            return false;
        }
        std::thread([this, run, owner] {
            std::string p = run(owner);
            {
                std::lock_guard<std::mutex> lk(mu);
                result = std::move(p);
            }
            state = 2;
        }).detach();
        return true;
    }
    bool take(std::string& path) {
        if (state != 2) return false;
        {
            std::lock_guard<std::mutex> lk(mu);
            path = std::move(result);
            result.clear();
        }
        state = 0;
        return true;
    }
};
// Поток диалога отсоединён и может ещё работать, когда приложение закрывают с открытым
// диалогом: объекты не разрушаются при выходе, чтобы поток не писал в удалённый мьютекс.
Dialog& g_video = *new Dialog;
Dialog& g_folder = *new Dialog;
}  // namespace

bool open_video_dialog(void* owner) {
    if (g_folder.state == 1) {   // один диалог за раз
        dialog_to_front();
        return false;
    }
    return g_video.open(dialog_blocking, owner);
}
bool video_dialog_open() { return g_video.state == 1; }
bool video_dialog_result(std::string& path) { return g_video.take(path); }

bool open_folder_dialog(void* owner) {
    if (g_video.state == 1) {
        dialog_to_front();
        return false;
    }
    return g_folder.open(folder_blocking, owner);
}
bool folder_dialog_open() { return g_folder.state == 1; }
bool folder_dialog_result(std::string& path) { return g_folder.take(path); }

}  // namespace plat
