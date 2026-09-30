//! Error codes — port of `libavutil/error.h` + `error.c`.
//!
//! FFmpeg reports failures as negative `int` return values; special sentinels
//! such as `AVERROR_EOF` (= `FFERRTAG('E','O','F',' ')`) or `AVERROR(EAGAIN)`
//! are values callers are *expected* to receive and branch on, not crashes.
//!
//! The Rust translation keeps the sentinel structure but makes it explicit:
//! `Err(Error::Eof)` is the "clean end of stream" signal every demux/decode
//! loop matches on, `Err(Error::Again)` is the decoder's "feed me more input
//! before I can emit a frame" signal (`-EAGAIN` in C), and everything else is
//! a genuine failure carrying the `av_strerror()` text as its `Display` form.
//!
//! Where C guards integer overflow with `INT_MAX` checks we return
//! [`Error::OutOfRange`] instead of silently truncating.

use std::fmt;

/// The `AVERROR(...)` family as a Rust enum.
///
/// Names track the C macros (`AVERROR_EOF` → [`Error::Eof`],
/// `AVERROR_INVALIDDATA` → [`Error::InvalidData`], …) so that ported code
/// reads next to its C origin.
#[derive(Debug)]
pub enum Error {
    /// `AVERROR_EOF` — clean end of stream / drain complete.
    ///
    /// Not a failure: `read_frame`, `receive_frame` and `receive_packet`
    /// all return this to end their loops.
    Eof,
    /// `-EAGAIN` — operation would block; more input required first.
    ///
    /// The `send_packet`/`receive_frame` handshake relies on this.
    Again,
    /// `AVERROR_INVALIDDATA` — malformed input.
    InvalidData(String),
    /// `-ENOSYS` / function not implemented — also used for deliberately
    /// un-ported FFmpeg features (seeking, exotic formats, …).
    Unsupported(String),
    /// I/O failure (`AVERROR(errno)` for file reads/writes).
    Io(std::io::Error),
    /// `AVERROR_DEMUXER_NOT_FOUND` etc. — resolved name goes in the string.
    NotFound(String),
    /// `AVERROR_STREAM_NOT_FOUND`.
    StreamNotFound,
    /// `AVERROR_BUFFER_TOO_SMALL`.
    BufferTooSmall,
    /// Integer overflow / `EINVAL` on geometry (width·height, linesizes).
    OutOfRange,
    /// Bad command line or option value (`ffmpeg_opt.c` parse failures).
    InvalidArgument(String),
}

// NOTE: manual PartialEq (assert_eq! on Results in tests) — variant-aware;
// Io compares by formatted text (std::io::Error lacks PartialEq).
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Error::Eof, Error::Eof)
            | (Error::Again, Error::Again)
            | (Error::StreamNotFound, Error::StreamNotFound)
            | (Error::BufferTooSmall, Error::BufferTooSmall)
            | (Error::OutOfRange, Error::OutOfRange) => true,
            (Error::InvalidData(a), Error::InvalidData(b))
            | (Error::Unsupported(a), Error::Unsupported(b))
            | (Error::NotFound(a), Error::NotFound(b))
            | (Error::InvalidArgument(a), Error::InvalidArgument(b)) => a == b,
            (Error::Io(a), Error::Io(b)) => a.to_string() == b.to_string(),
            _ => false,
        }
    }
}
impl Eq for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Eof => write!(f, "End of file"),
            Error::Again => write!(f, "Resource temporarily unavailable"),
            Error::InvalidData(s) => write!(f, "Invalid data found when processing input: {s}"),
            Error::Unsupported(s) => write!(f, "Function not implemented: {s}"),
            Error::Io(e) => write!(f, "{e}"),
            Error::NotFound(s) => write!(f, "{s} not found"),
            Error::StreamNotFound => write!(f, "Stream not found"),
            Error::BufferTooSmall => write!(f, "Buffer too small"),
            Error::OutOfRange => write!(f, "Numerical result out of range"),
            Error::InvalidArgument(s) => write!(f, "Invalid argument: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience alias used throughout the crate — every ported C function that
/// returned `int` (0 ok / negative error) becomes `Result<T>`.
pub type Result<T> = std::result::Result<T, Error>;
