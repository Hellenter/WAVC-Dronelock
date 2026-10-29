#!/usr/bin/env bash
# Минимальная статическая сборка FFmpeg для Windows x64 (кросс-компиляция MinGW-w64).
# Только то, что нужно для чтения видео: демуксеры mp4/mkv/avi/ts, декодеры h264/hevc/mpeg4/mjpeg/vp8/vp9,
# swscale. Статические библиотеки линкуются прямо в .exe, DLL рядом не нужны.
#
#   app/scripts/build_ffmpeg_mingw.sh <каталог исходников FFmpeg> <prefix>
set -euo pipefail
SRC=$(realpath "$1")
PREFIX=$(realpath -m "$2")
BUILD="$PREFIX.build"
mkdir -p "$BUILD" && cd "$BUILD"
"$SRC/configure" --prefix="$PREFIX" \
  --target-os=mingw32 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32- \
  --enable-static --disable-shared --disable-programs --disable-doc --disable-debug \
  --disable-everything --disable-autodetect --disable-network \
  --disable-avdevice --disable-avfilter --disable-swresample --disable-postproc \
  --enable-w32threads --enable-x86asm \
  --enable-protocol=file \
  --enable-demuxer=mov,matroska,avi,mpegts,mpegps,h264,hevc,mjpeg,yuv4mpegpipe,rawvideo,flv,asf \
  --enable-decoder=h264,hevc,mpeg4,h263,msmpeg4v3,mjpeg,vp8,vp9,mpeg1video,mpeg2video,rawvideo,ffv1,prores,wmv3,vc1 \
  --enable-parser=h264,hevc,mpeg4video,mjpeg,vp8,vp9,mpegvideo,vc1 \
  --extra-cflags="-O2"
make -j"$(nproc)"
make install
