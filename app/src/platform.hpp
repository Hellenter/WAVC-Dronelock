// Платформенные мелочи: загрузка CPU и память процесса, ограничение ресурсов,
// диалог выбора файла, шрифты.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace plat {

double process_cpu_seconds();          // user + system, все потоки
double process_memory_mb();            // рабочий набор (RSS)
std::string cpu_model();
int logical_cpus();

uint64_t process_affinity();           // маска логических ядер, на которых может работать процесс
bool set_process_affinity(uint64_t mask);

// Жёсткий потолок загрузки CPU всем процессом, % от всех ядер (Windows Job Object).
// percent >= 100 снимает ограничение.
bool cpu_cap_supported();
bool set_cpu_cap(int percent);

// -2 низкий, -1 ниже среднего, 0 обычный, 1 выше среднего, 2 высокий
void set_process_priority(int level);

std::string open_video_dialog();       // "" — отменено или недоступно
std::vector<std::string> ui_fonts();   // системные шрифты с кириллицей, по убыванию предпочтения
void attach_console();                 // Windows: вывод --bench в консоль, из которой запустили

}  // namespace plat
