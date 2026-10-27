// Пул рабочих потоков с привязкой к ядрам. Задания раздаются через атомарный
// счётчик, вызывающий поток только ждёт: так вся работа детектора идёт строго
// на тех ядрах, которые разрешены (нужно для имитации слабого железа).
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace dronolok {

void set_current_thread_affinity(int cpu);   // cpu < 0 — ничего не делать
void set_current_thread_name(const char* name);

class ThreadPool {
public:
    ThreadPool(int threads, const std::vector<int>& cpus);
    ~ThreadPool();
    int size() const { return (int)workers_.size(); }
    // fn(job, worker) для job в [0, njobs); возвращается, когда всё сделано
    void run(int njobs, const std::function<void(int, int)>& fn);

private:
    void loop(int idx, int cpu);

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int, int)>* fn_ = nullptr;
    int njobs_ = 0;
    std::atomic<int> next_{0};
    int active_ = 0;
    unsigned gen_ = 0;
    bool stop_ = false;
};

}  // namespace dronolok
