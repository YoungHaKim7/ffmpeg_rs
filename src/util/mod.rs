//! `libavutil` — foundational utilities shared by every other layer.
//!
//! Port map (C file → Rust module):
//!
//! | FFmpeg | here | status |
//! |---|---|---|
//! | `rational.{h,c}` | [`rational`] | full |
//! | `mathematics.{h,c}` (rescale/gcd) | [`mathematics`] | used subset |
//! | `error.{h,c}` | [`error`] | used subset |
//! | `log.h` | [`log`] | default-callback behavior |
//! | `pixfmt.h` | [`pixfmt`] | ~24-format LE subset |
//! | `pixdesc.{h,c}` | [`pixdesc`] | descriptors for the subset |
//! | `imgutils.{h,c}` | [`imgutils`] | plane-geometry subset |
//! | `frame.{h,c}` | [`frame`] | video subset, Arc-backed planes |
//!
//! Not ported (out of scope for the Phase 1 chunk): `mem.h` (Rust ownership
//! replaces it), `dict.h` metadata, `avassert.h` (debug_assert!), CPU flags,
//! `hwcontext*` (arrives with the Vulkan phase), samplefmt (audio phase).

pub mod audio_frame;
pub mod channel_layout;
pub mod color;
pub mod error;
pub mod frame;
pub mod imgutils;
pub mod log;
pub mod mathematics;
pub mod pixdesc;
pub mod pixfmt;
pub mod rational;
pub mod samplefmt;

pub use error::{Error, Result};
pub use frame::{Frame, PictureType};
pub use pixfmt::PixelFormat;
pub use rational::Rational;

/// `AV_NOPTS_VALUE` (`INT64_MIN`) — the "no timestamp" sentinel. Kept as a
/// constant rather than an `Option<i64>` on frames/packets so that rescale
/// arithmetic and C-side comparisons carry over unchanged.
pub const NOPTS: i64 = i64::MIN;
