#!/bin/bash
# Regenerate the H.264 bit-exactness fixtures in /tmp (the acceptance
# references are DEFAULT ffmpeg decodes of the same files).
# /tmp is wiped periodically — rerun this before testing.
set -e
cd /tmp
for src in testsrc2 gray black; do
  if [ "$src" = testsrc2 ]; then f="testsrc2=size=128x96:rate=10"; else f="color=$src:size=128x96:rate=10"; fi
  # CAVLC baseline (regression)
  ffmpeg -y -loglevel error -f lavfi -i "$f" -t 1.5 -c:v libx264 -profile:v baseline \
    -pix_fmt yuv420p -x264-params ref=3 /tmp/h264_$src.h264
  ffmpeg -y -loglevel error -i /tmp/h264_$src.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_$src.yuv
  # CABAC main, I/P only (bf/weightp off: B slices + weighted pred are Phase C)
  ffmpeg -y -loglevel error -f lavfi -i "$f" -t 1.5 -c:v libx264 -profile:v main -bf 0 -weightp 0 \
    -pix_fmt yuv420p -x264-params ref=3 /tmp/h264_cabac_$src.h264
  ffmpeg -y -loglevel error -i /tmp/h264_cabac_$src.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_cabac_$src.yuv
done
for name in ms cabac_ms; do
  ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=128x96:rate=10 -t 1.5 -c:v libx264 \
    $( [ $name = ms ] && echo "-profile:v baseline" || echo "-profile:v main -bf 0 -weightp 0" ) \
    -pix_fmt yuv420p -x264-params ref=3:slices=4 /tmp/h264_$name.h264
  ffmpeg -y -loglevel error -i /tmp/h264_$name.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_$name.yuv
done
ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=128x96:rate=10 -t 1.5 -c:v libx264 -profile:v main \
  -bf 0 -weightp 0 -crf 18 -pix_fmt yuv420p -x264-params ref=3 /tmp/h264_cabac_q18.h264
ffmpeg -y -loglevel error -i /tmp/h264_cabac_q18.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_cabac_q18.yuv
# Phase C: B slices + weighted prediction (x264 defaults: b-adapt + weightp)
for src in testsrc2 gray black; do
  if [ "$src" = testsrc2 ]; then f="testsrc2=size=128x96:rate=10"; else f="color=$src:size=128x96:rate=10"; fi
  ffmpeg -y -loglevel error -f lavfi -i "$f" -t 1.5 -c:v libx264 -profile:v main \
    -pix_fmt yuv420p /tmp/h264_b_$src.h264
  ffmpeg -y -loglevel error -i /tmp/h264_b_$src.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_b_$src.yuv
done
ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=128x96:rate=10 -t 1.5 -c:v libx264 -profile:v main \
  -x264-params direct=temporal -pix_fmt yuv420p /tmp/h264_b_tem.h264
ffmpeg -y -loglevel error -i /tmp/h264_b_tem.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_b_tem.yuv
ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=128x96:rate=10 -t 1.5 -c:v libx264 -profile:v main \
  -x264-params slices=4 -pix_fmt yuv420p /tmp/h264_b_ms.h264
ffmpeg -y -loglevel error -i /tmp/h264_b_ms.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_b_ms.yuv
ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=128x96:rate=10 -t 1.5 -c:v libx264 -profile:v main \
  -weightp 0 -bf 1 -pix_fmt yuv420p /tmp/h264_b_bf1.h264
ffmpeg -y -loglevel error -i /tmp/h264_b_bf1.h264 -f rawvideo -pix_fmt yuv420p /tmp/h264_ref_b_bf1.yuv
ls /tmp/h264_*.h264
