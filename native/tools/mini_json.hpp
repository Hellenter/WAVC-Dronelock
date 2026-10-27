// Минимальный JSON-парсер для файлов разметки (*.gt.json). Без зависимостей.
#pragma once
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace minijson {

struct Value {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Value> a;
    std::map<std::string, Value> o;

    const Value& operator[](const std::string& k) const {
        static const Value none;
        auto it = o.find(k);
        return it == o.end() ? none : it->second;
    }
    double num(double def = 0) const { return type == Num ? n : type == Bool ? (b ? 1 : 0) : def; }
};

class Parser {
public:
    explicit Parser(const std::string& t) : s_(t) {}
    bool parse(Value& v) {
        ws();
        if (!value(v)) return false;
        ws();
        return i_ == s_.size();
    }

private:
    void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_; }
    bool lit(const char* w) {
        size_t n = 0;
        while (w[n]) ++n;
        if (s_.compare(i_, n, w) != 0) return false;
        i_ += n;
        return true;
    }
    bool str(std::string& out) {
        if (s_[i_] != '"') return false;
        ++i_;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c == '\\' && i_ < s_.size()) {
                char e = s_[i_++];
                if (e == 'u') { i_ += 4; out += '?'; continue; }   // разметке юникод не нужен
                c = e == 'n' ? '\n' : e == 't' ? '\t' : e;
            }
            out += c;
        }
        if (i_ >= s_.size()) return false;
        ++i_;
        return true;
    }
    bool value(Value& v) {
        if (i_ >= s_.size()) return false;
        const char c = s_[i_];
        if (c == '{') {
            v.type = Value::Obj;
            ++i_;
            ws();
            if (s_[i_] == '}') { ++i_; return true; }
            for (;;) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (s_[i_++] != ':') return false;
                ws();
                if (!value(v.o[k])) return false;
                ws();
                if (s_[i_] == ',') { ++i_; continue; }
                if (s_[i_] == '}') { ++i_; return true; }
                return false;
            }
        }
        if (c == '[') {
            v.type = Value::Arr;
            ++i_;
            ws();
            if (s_[i_] == ']') { ++i_; return true; }
            for (;;) {
                ws();
                v.a.emplace_back();
                if (!value(v.a.back())) return false;
                ws();
                if (s_[i_] == ',') { ++i_; continue; }
                if (s_[i_] == ']') { ++i_; return true; }
                return false;
            }
        }
        if (c == '"') { v.type = Value::Str; return str(v.s); }
        if (lit("true")) { v.type = Value::Bool; v.b = true; return true; }
        if (lit("false")) { v.type = Value::Bool; v.b = false; return true; }
        if (lit("null")) { v.type = Value::Null; return true; }
        char* end = nullptr;
        v.n = std::strtod(s_.c_str() + i_, &end);
        if (end == s_.c_str() + i_) return false;
        v.type = Value::Num;
        i_ = end - s_.c_str();
        return true;
    }

    const std::string& s_;
    size_t i_ = 0;
};

inline bool parse(const std::string& text, Value& out) { return Parser(text).parse(out); }

}  // namespace minijson
