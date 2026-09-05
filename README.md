# ffmpeg_rs

A pipeline-faithful Rust port of [FFmpeg](https://github.com/ffmpeg/ffmpeg)
(the 8.0.git tree at the sibling `FFmpeg` directory), built one bounded phase
at a time, with Vulkan compute powering the swscale stage.

## Status — Phase 3a (variable-width scaling filters)

The full `demux → decode → convert → encode → mux` pipeline for **rawvideo**
and **Y4M**, driven by an ffmpeg-style CLI, now with real resampling and a
GPU engine:

```console
$ ffmpeg -f lavfi -i testsrc2=duration=1:size=128x96:rate=10 \
         -pix_fmt yuv420p -f yuv4mpegpipe in.y4m -y
$ cargo run -- -i in.y4m -f rawvideo -pix_fmt rgb24 out.raw -y
ffmpeg_rs version 0.1.0 Copyright (c) 2026 the ffmpeg_rs authors
  built with rustc (FFmpeg 8.0.git pipeline-faithful port, phase 1)
Input #0, yuv4mpegpipe, from 'in.y4m':
  Duration: 00:00:01.00, start: 0.000000, bitrate: 1475 kb/s
  Stream #0:0: Video: rawvideo (I420 / 0x30323449), yuv420p(progressive), 128x96, SAR 1:1 DAR 4:3, 10 fps, 10 tbr, 10 tbn
Stream mapping:
  Stream #0:0 -> #0:0 (rawvideo (native) -> rawvideo (native))
Output #0, rawvideo, to 'out.raw':
  Stream #0:0: Video: rawvideo (RGB[24] / 0x18424752), rgb24(pc, progressive), 128x96 [SAR 1:1 DAR 4:3], q=2-31, 2949 kb/s, 10 fps, 10 tbn
frame=   10 fps=… q=-0.0 Lsize=     360KiB time=00:00:01.00 bitrate=2949.1kbits/s speed=…x
```

Correctness is pinned against the system ffmpeg (`tests/golden.rs`):
y4m↔rawvideo conversions are **byte-exact**; the BT.601 rgb24 conversion is
within ±3 of swscale per byte; the Vulkan and CPU scale engines agree to
≤1 byte on every conversion mode × the three shader algorithms
(nearest/bilinear/bicubic). The additional libswscale kernels — area, gauss,
sinc, lanczos, spline, with C's filter-width widening on downscale
(`initFilter`, `utils.c:197-612`) — are CPU-only ports that measure ≤2 bytes
off the system binary (entirely its SIMD apply path; byte-identical to a
standalone build of FFmpeg's own C reference).

```console
$ cargo run -- -i in.y4m -s 64x48 -f rawvideo -pix_fmt rgb24 out.raw -y
# -s WxH            scale (nearest/bilinear/bicubic, default bicubic)
# -scale_algo A     nearest | bilinear | bicubic | area | gauss | sinc |
#                   lanczos | spline (the latter five CPU-only, auto-
#                   falling back from the Vulkan engine)
# -scale_engine E   auto (default) | vulkan | cpu
```

## Layout

| module | ports |
|---|---|
| `src/util` | libavutil: `Rational`, errors, log, `PixelFormat`+descriptors, imgutils, `Frame` (Arc-backed planes) |
| `src/codec` | libavcodec: `Packet`, `CodecParameters`, Decoder/Encoder traits, rawvideo dec/enc |
| `src/format` | libavformat: `IoContext` (avio), demux/mux registries, Y4M + rawvideo |
| `src/swscale` | libswscale: `ScaleContext` (unscaled converters + resampling, CPU + Vulkan engines), `filter.rs` (`initFilter` port: area/gauss/sinc/lanczos/spline) |
| `src/gpu` | libavutil/vulkan: headless `ComputeGpu` (device, queue, staging, one-shot cmdbuffers) |
| `src/shaders` | libavfilter/vulkan: GLSL compute modules (`scale.comp`) |
| `src/fftools` | the CLI: arg parsing, `av_dump_format` output, transcode loop |

Each module's docs carry the C→Rust mapping table and the list of
deliberately skipped C paths with the guard that makes them unreachable.

## Roadmap

1. ✅ Phase 1 — CPU pipeline
2. ✅ Phase 2 — Vulkan swscale: headless compute (`vulkano`), port of
   `vf_scale_vulkan.c`'s shape + the `libswscale` kernels,
   nearest/bilinear/bicubic, `-s` (colorspace matrix support is a Phase 3
   candidate)
3. ✅ Phase 3a — libswscale variable-width filters on the CPU engine:
   `initFilter` port (`utils.c:197-612`), area/gauss/sinc/lanczos/spline
   via `-scale_algo` with filter-width widening on downscale, CPU fallback
   for algorithms the Vulkan engine cannot run
4. Phase 3b — filtergraph (`libavfilter`: buffersrc/sink, `scale`/`format`)
5. Phase 4 — `swresample` + audio paths
6. Phase 5 — NUT container, more filters
7. Stretch — winit player window on the Vulkan pipeline

## Tests

```console
$ cargo test            # unit tests + golden tests vs system ffmpeg
$ cargo test --test golden
```
