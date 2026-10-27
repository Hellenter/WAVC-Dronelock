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
#endif

namespace dronolok {

inline FILE* open_file(const std::string& utf8, const char* mode) {
#ifdef _WIN32
    auto wide = [](const std::string& s) {
        const int n = MultiByteToWideChar(65001 /*CP_UTF8*/, 0, s.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> w(n > 0 ? n : 1, 0);
        if (n > 0) MultiByteToWideChar(65001, 0, s.c_str(), -1, w.data(), n);
        return std::wstring(w.data());
    };
    return _wfopen(wide(utf8).c_str(), wide(mode).c_str());
#else
    return std::fopen(utf8.c_str(), mode);
#endif
}

inline int remove_file(const std::string& utf8) {
#ifdef _WIN32
    const int n = MultiByteToWideChar(65001, 0, utf8.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> w(n > 0 ? n : 1, 0);
    if (n > 0) MultiByteToWideChar(65001, 0, utf8.c_str(), -1, w.data(), n);
    return _wremove(w.data());
#else
    return std::remove(utf8.c_str());
#endif
}

}  // namespace dronolok
