//! # ffmpeg_rs — FFmpeg, ported to Rust one phase at a time
//!
//! A pipeline-faithful Rust port of [FFmpeg](https://github.com/ffmpeg/ffmpeg)
//! 8.0.git (the `./FFmpeg` tree in this repo), with Vulkan compute planned to
//! power the swscale/filter stages. The port proceeds in bounded phases; the
//! work is tracked in the repo plan file.
//!
//! * **Phase 1 (this)** — CPU foundation: the full
//!   demux → decode → convert → encode → mux pipeline for rawvideo and Y4M,
//!   driven by an ffmpeg-style CLI. Byte-exact with real ffmpeg on the raw
//!   paths (pinned by `tests/golden.rs`).
//! * **Phase 2 (next)** — Vulkan `swscale`: headless compute context running
//!   a port of `libavfilter/vulkan/scale.comp.glsl` + `vf_scale_vulkan.c`,
//!   real resampling (bilinear/bicubic) and `-s`.
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
//! | [`swscale`] | `libswscale` | `ScaleContext`: identity + BT.601 YUV/gray → RGB kernels |
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
//! ```
//!
//! ## Example
//!
//! ```text
//! $ ffmpeg -f lavfi -i testsrc2=duration=1:size=128x96:rate=10 \
//!          -pix_fmt yuv420p -f yuv4mpegpipe in.y4m -y
//! $ cargo run -- -i in.y4m -f rawvideo -pix_fmt rgb24 out.raw -y
//! ```

pub mod codec;
pub mod fftools;
pub mod format;
pub mod gpu;
pub mod shaders;
pub mod swscale;
pub mod util;

// Crate-root re-exports mirroring how C includes libavutil headers:
// `use crate::{Error, Result, Frame, PixelFormat, Rational}`.
pub use util::color::{ChromaLocation, ColorPrimaries, ColorRange, ColorSpace, ColorTrc};
pub use util::error::{Error, Result};
pub use util::frame::{Frame, FrameFlags, PictureType};
pub use util::imgutils;
pub use util::mathematics;
pub use util::pixfmt::PixelFormat;
pub use util::rational::Rational;
pub use util::NOPTS;
