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

// Диалог выбора видео в отдельном потоке, интерфейс продолжает работать. owner — окно приложения
// (HWND на Windows): диалог модальный для него и всегда поверх. Если диалог уже открыт, он
// поднимается поверх, второй не открывается (тогда false).
bool open_video_dialog(void* owner);
bool video_dialog_open();
bool video_dialog_result(std::string& path);   // true — диалог закрыт; path пустой, если отменили
// То же для папки (мультитест): свой диалог, независимый от выбора видео
bool open_folder_dialog(void* owner);
bool folder_dialog_open();
bool folder_dialog_result(std::string& path);
bool open_in_browser(const std::string& path);   // файл (отчёт) или папка — в программе по умолчанию
std::vector<std::string> ui_fonts();   // системные шрифты с кириллицей, по убыванию предпочтения
void attach_console();                 // Windows: вывод --bench в консоль, из которой запустили

}  // namespace plat
