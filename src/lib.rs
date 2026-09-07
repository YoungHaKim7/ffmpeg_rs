//! # ffmpeg_rs — FFmpeg, ported to Rust one phase at a time
//!
//! A pipeline-faithful Rust port of [FFmpeg](https://github.com/ffmpeg/ffmpeg)
//! 8.0.git (the sibling `FFmpeg` tree), with Vulkan compute powering the
//! swscale stage. The port proceeds in bounded phases; the work is tracked in
//! the repo plan file.
//!
//! * **Phase 1** — CPU foundation: the full
//!   demux → decode → convert → encode → mux pipeline for rawvideo and Y4M,
//!   driven by an ffmpeg-style CLI. Byte-exact with real ffmpeg on the raw
//!   paths (pinned by `tests/golden.rs`).
//! * **Phase 2** — Vulkan `swscale`: a headless compute context
//!   (`src/gpu.rs`) running `assets/scale.comp` — the `vf_scale_vulkan.c` /
//!   `libswscale/vulkan/` shape — with real resampling
//!   (nearest/bilinear/bicubic + chroma siting), `-s`, and an engine picker
//!   (`-scale_engine auto|vulkan|cpu`). The GPU shader mirrors the CPU
//!   kernels tap for tap; `tests/golden.rs` pins the two engines to ≤1 byte.
//! * **Phase 3a (this)** — libswscale's variable-width scaling filters:
//!   `area`/`gauss`/`sinc`/`lanczos`/`spline` via a bit-faithful port of
//!   `initFilter` (`utils.c:197-612`, see [`swscale::filter`]) with
//!   C's filter-width widening on downscale, exposed as `-scale_algo`
//!   values (CPU-only; `-scale_engine auto` falls back, `vulkan` rejects).
//! * Later — filtergraph (`libavfilter` buffersrc/sink + `scale`/`format`
//!   filters), `swresample`, image2/NUT containers, a winit player.
//!
//! ## Module map
//!
//! | module | ports | what it holds |
//! |---|---|---|
//! | [`util`] | `libavutil` | Rational, errors, logging, `PixelFormat` + descriptors, plane geometry, `Frame` |
//! | [`codec`] | `libavcodec` | `Packet`, `CodecParameters`, Decoder/Encoder traits, rawvideo codec |
//! | [`format`] | `libavformat` | `IoContext` (avio), demuxer/muxer registries, Y4M + rawvideo, format contexts |
//! | [`swscale`] | `libswscale` | `ScaleContext`: unscaled converters + resampling (nearest/bilinear/bicubic + the table-driven area/gauss/sinc/lanczos/spline), CPU kernels and the Vulkan engine |
//! | [`gpu`] | `libavutil/vulkan` | `ComputeGpu`: headless device + queue, one-shot command buffers, staging |
//! | [`shaders`] | `libavfilter/vulkan` | GLSL compute modules (`scale.comp`) |
//! | [`fftools`] | `fftools` | CLI parsing, `av_dump_format`-style output, the transcode loop |
//!
//! ## Data flow (Phase 1, single video stream, zero-copy where marked)
//!
//! ```text
//! file ──IoContext──▶ Demuxer (y4m|rawvideo)          libavformat
//!         read_buffer │ the one unavoidable memcpy
//!                     ▼
//!                 Packet ─ Arc<[u8]> ─────────┐       libavcodec/packet.h
//!                     │                       │ Arc clone (zero copy)
//!                     ▼                       ▼
//!                RawVideoDecoder          Frame::wrap_buffer
//!                     │                       │
//!                     │ formats equal? ──yes──┤ Frame clone (Arc share)
//!                     ▼ no                    │
//!                ScaleContext (BT.601)        │       libswscale
//!                     ▼                       ▼
//!                RawVideoEncoder ◀──── Frame ─┘
//!                     │ copy_to_buffer (row pack)
//!                     ▼
//!                 Packet ─▶ Muxer ──IoContext──▶ file  libavformat
//! ```
//!
//! ## Translation conventions
//!
//! * **Names stay FFmpeg's** (`pts`, `pict_type`, `codecpar`,
//!   `time_base`, `log2_chroma_w`) so ported code reads next to its C
//!   origin — only API *shapes* are Rustified (traits instead of function
//!   pointers, `Result` instead of negative ints).
//! * **`AVERROR_EOF`/`EAGAIN` are values, not crashes**:
//!   `Err(Error::Eof)` ends every read loop, `Err(Error::Again)` drives the
//!   decoder handshake — the C control flow, typed.
//! * **Refcounting maps to `Arc`**: `AVBufferRef`'s "writable iff one
//!   reference" becomes `Arc::strong_count == 1` (`Frame::is_writable` /
//!   `make_writable` is the copy-on-write gate).
//! * **Deliberate simplifications are documented in place**, per module,
//!   with the C guard that makes them unreachable listed (e.g. rawdec's
//!   codec-tag zoo, negative linesizes).
//!
//! ## Verification
//!
//! `cargo test` runs unit tests per module plus `tests/golden.rs`, which
//! byte-compares outputs against the system `ffmpeg` (skips politely when
//! absent):
//!
//! ```text
//! y4m → rawvideo yuv420p : byte-exact vs ffmpeg 8.1.2
//! rawvideo → y4m         : byte-exact (header writer pinned)
//! y4m → rawvideo rgb24   : max byte diff ≤ 3 (measured max 3, mean 0.66;
//!                          swscale's integer tables vs our float BT.601)
//! cpu ↔ vulkan, all 4 conversion modes × the 3 shader algorithms
//!                          (nearest, bilinear, bicubic), ½ downscale:
//!                          max byte diff ≤ 1 (unorm store vs round())
//! y4m → ½-scale yuv420p  : max 89 documented divergence — applies ONLY to
//!                          the fixed-tap float/GPU algorithms (swscale
//!                          widens the filter in source space on downscale,
//!                          utils.c:287-293; our 4-tap window under-blurs).
//!                          The table-driven kernels widen like swscale:
//! y4m → ½-scale yuv420p, area/gauss/sinc/lanczos/spline
//!                        : max ≤ 2, 0 bytes outside ±3 (measured 0/1/2/1/1
//!                          — entirely system ffmpeg's SIMD apply path
//!                          sitting ±1-2 off its own `_c` kernels; this
//!                          port is byte-identical to a standalone build of
//!                          FFmpeg's C reference, pinned by unit tests)
//! ```
//!
//! ## Example
//!
//! ```bash
//! $ ffmpeg -f lavfi -i testsrc2=duration=1:size=128x96:rate=10 \
//!          -pix_fmt yuv420p -f yuv4mpegpipe in.y4m -y
//! $ cargo run -- -i in.y4m -f rawvideo -pix_fmt rgb24 out.raw -y
//! ```

pub mod codec;
pub mod fftools;
pub mod filter;
pub mod format;
pub mod gpu;
pub mod shaders;
pub mod swscale;
pub mod util;

// Crate-root re-exports mirroring how C includes libavutil headers:
// `use crate::{Error, Result, Frame, PixelFormat, Rational}`.
pub use crate::swscale::filter::TableScaler;
pub use filter::{FilterGraph, LinkId, NodeId};
pub use util::{
    NOPTS,
    color::{ChromaLocation, ColorPrimaries, ColorRange, ColorSpace, ColorTrc},
    error::{Error, Result},
    frame::{Frame, FrameFlags, PictureType},
    imgutils, mathematics,
    pixfmt::PixelFormat,
    rational::Rational,
};
