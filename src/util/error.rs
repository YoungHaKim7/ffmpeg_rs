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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eof_and_again_display_like_av_strerror() {
        // av_strerror(AVERROR_EOF) == "End of file"
        assert_eq!(Error::Eof.to_string(), "End of file");
        // av_strerror(AVERROR(EAGAIN)) starts "Resource temporarily"
        assert!(Error::Again.to_string().starts_with("Resource temporarily"));
    }

    #[test]
    fn invalid_data_carries_context() {
        let e = Error::InvalidData("YUV4MPEG stream contains an unknown pixel format.".into());
        assert!(e.to_string().contains("unknown pixel format"));
    }

    #[test]
    fn io_errors_convert() {
        let e: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "nope").into();
        assert!(matches!(e, Error::Io(_)));
    }
}
