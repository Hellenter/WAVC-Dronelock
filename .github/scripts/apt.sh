#!/bin/bash
# Установка пакетов в CI. Зеркало Ubuntu иногда отдаёт пакеты по несколько кБ/с и apt
# висит по 10+ минут: каждую попытку ограничиваем по времени и повторяем. cmake и ninja
# на раннерах GitHub уже есть — их из списка убираем.
set -u
pkgs=()
for p in "$@"; do
  case "$p" in
    cmake) command -v cmake >/dev/null && continue ;;
    ninja-build) command -v ninja >/dev/null && continue ;;
  esac
  pkgs+=("$p")
done
echo "пакеты: ${pkgs[*]}"
opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::http::Pipeline-Depth=0)
for try in 1 2 3; do
  if timeout 420 sudo apt-get "${opts[@]}" update -q &&
     timeout 600 sudo apt-get "${opts[@]}" install -y -q --no-install-recommends "${pkgs[@]}"; then
    exit 0
  fi
  echo "попытка $try не удалась, повтор"
  sleep 10
done
exit 1
