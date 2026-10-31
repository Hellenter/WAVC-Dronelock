// Открытие файла по пути в UTF-8 на любой ОС. В Windows fopen понимает только
// кодовую страницу ANSI, и путь с кириллицей ("Видео/дрон.mp4") не открывался бы.
#pragma once
#include <cstdio>
#include <string>
#ifdef _WIN32
#include <cwchar>
#include <vector>
extern "C" __declspec(dllimport) int __stdcall MultiByteToWideChar(unsigned int, unsigned long, const char*, int,
                                                                   wchar_t*, int);
extern "C" __declspec(dllimport) int __stdcall MoveFileExW(const wchar_t*, const wchar_t*, unsigned long);
#endif

namespace dronolok {

#ifdef _WIN32
inline std::wstring widen_utf8(const std::string& s) {
    const int n = MultiByteToWideChar(65001 /*CP_UTF8*/, 0, s.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> w(n > 0 ? n : 1, 0);
    if (n > 0) MultiByteToWideChar(65001, 0, s.c_str(), -1, w.data(), n);
    return std::wstring(w.data());
}
#endif

inline FILE* open_file(const std::string& utf8, const char* mode) {
#ifdef _WIN32
    return _wfopen(widen_utf8(utf8).c_str(), widen_utf8(mode).c_str());
#else
    return std::fopen(utf8.c_str(), mode);
#endif
}

inline int remove_file(const std::string& utf8) {
#ifdef _WIN32
    return _wremove(widen_utf8(utf8).c_str());
#else
    return std::remove(utf8.c_str());
#endif
}

// Заменить dst файлом src (переименование; dst, если есть, перезаписывается)
inline bool replace_file(const std::string& src, const std::string& dst) {
#ifdef _WIN32
    // MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH: _wrename не заменяет существующий
    return MoveFileExW(widen_utf8(src).c_str(), widen_utf8(dst).c_str(), 0x1 | 0x8) != 0;
#else
    return std::rename(src.c_str(), dst.c_str()) == 0;
#endif
}

// Файл целиком через временный рядом (path.tmp) и замену: оборванная запись (закрыли программу,
// пропало питание) оставляет прежний файл целым, а не обрезанным. write(FILE*) пишет содержимое
template <class Write>
bool write_file_replace(const std::string& path, const char* mode, Write&& write) {
    const std::string tmp = path + ".tmp";
    FILE* f = open_file(tmp, mode);
    if (!f) return false;
    bool ok = write(f);
    ok = std::fclose(f) == 0 && ok;
    ok = ok && replace_file(tmp, path);
    if (!ok) remove_file(tmp);
    return ok;
}

}  // namespace dronolok
