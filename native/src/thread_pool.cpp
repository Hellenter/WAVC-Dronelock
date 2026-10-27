#include "thread_pool.hpp"

#include <cstdio>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace dronolok {

void set_current_thread_affinity(int cpu) {
    if (cpu < 0) return;
#if defined(_WIN32)
    if (cpu < 64) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}

void set_current_thread_name(const char* name) {
#if defined(_WIN32)
    // SetThreadDescription есть с Windows 10 1607, берём динамически
    typedef HRESULT(WINAPI * Fn)(HANDLE, PCWSTR);
    static Fn fn = (Fn)(void*)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription");
    if (fn) {
        wchar_t w[32];
        int i = 0;
        for (; name[i] && i < 31; ++i) w[i] = (wchar_t)name[i];
        w[i] = 0;
        fn(GetCurrentThread(), w);
    }
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}

ThreadPool::ThreadPool(int threads, const std::vector<int>& cpus) {
    if (threads < 1) threads = 1;
    for (int i = 0; i < threads; ++i) {
        int cpu = cpus.empty() ? -1 : cpus[i % cpus.size()];
        workers_.emplace_back([this, i, cpu] { loop(i, cpu); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        ++gen_;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::run(int njobs, const std::function<void(int, int)>& fn) {
    if (njobs <= 0) return;
    std::unique_lock<std::mutex> lk(m_);
    fn_ = &fn;
    njobs_ = njobs;
    next_.store(0);
    active_ = (int)workers_.size();
    ++gen_;
    cv_.notify_all();
    done_cv_.wait(lk, [this] { return active_ == 0; });
    fn_ = nullptr;
}

void ThreadPool::loop(int idx, int cpu) {
    char name[16];
    snprintf(name, sizeof(name), "dl-worker-%d", idx);
    set_current_thread_name(name);
    set_current_thread_affinity(cpu);
    unsigned seen = 0;
    for (;;) {
        const std::function<void(int, int)>* fn;
        int njobs;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return gen_ != seen; });
            seen = gen_;
            if (stop_) return;
            fn = fn_;
            njobs = njobs_;
        }
        for (int j; (j = next_.fetch_add(1)) < njobs;) (*fn)(j, idx);
        std::lock_guard<std::mutex> lk(m_);
        if (--active_ == 0) done_cv_.notify_one();
    }
}

}  // namespace dronolok
