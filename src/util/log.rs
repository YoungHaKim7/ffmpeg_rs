//! Logging — port of `libavutil/log.h` (the default-callback behavior).
//!
//! FFmpeg's `av_log(avcl, level, fmt, ...)` routes every diagnostic through a
//! level-gated global sink; the context object names the emitter (a demuxer, a
//! codec, …). The Rust port keeps the shape with plain `&'static str` context
//! names instead of `AVClass*` pointers:
//!
//! ```text
//! av_log(NULL, AV_LOG_INFO, "banner\n")      →  log_info!(None, "banner\n")
//! av_log(s,   AV_LOG_ERROR, "bad header\n")  →  log_error!(Some("y4m"), "bad header\n")
//! ```
//!
//! Output format follows the default callback: a bracketed context prefix
//! `[y4m] message` when a context is given, bare text otherwise, all to
//! stderr — ffmpeg's own banner and per-file dumps are plain-context INFO
//! messages, which is what our CLI prints.

use std::{
    fmt,
    sync::atomic::{AtomicI32, Ordering},
};

/// `av_log_set_level` — global verbosity, default `AV_LOG_INFO` like ffmpeg's.
static LEVEL: AtomicI32 = AtomicI32::new(Level::Info as i32);

/// `AV_LOG_*` levels (`log.h:192-236`). Discriminants match C so that
/// `-v/-loglevel` maps 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Level {
    Quiet = -8,
    Panic = 0,
    Fatal = 8,
    Error = 16,
    Warning = 24,
    Info = 32,
    Verbose = 40,
    Debug = 48,
    Trace = 56,
}

impl Level {
    /// Parse the names `-v`/`-loglevel` accept (`ffmpeg_opt.c` table).
    pub fn from_name(s: &str) -> Option<Level> {
        Some(match s {
            "quiet" => Level::Quiet,
            "panic" => Level::Panic,
            "fatal" => Level::Fatal,
            "error" => Level::Error,
            "warn" | "warning" => Level::Warning,
            "info" => Level::Info,
            "verbose" => Level::Verbose,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => return None,
        })
    }

    /// The `[level]` label the default C callback prefixes repeated/reduced
    /// messages with; used here when no context name is available.
    pub fn label(self) -> &'static str {
        match self {
            Level::Quiet | Level::Panic => "panic",
            Level::Fatal => "fatal",
            Level::Error => "error",
            Level::Warning => "warning",
            Level::Info => "info",
            Level::Verbose => "verbose",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }
}

/// Set the global log level (`av_log_set_level`).
pub fn set_level(level: Level) {
    LEVEL.store(level as i32, Ordering::Relaxed);
}

/// Current global log level (`av_log_get_level`).
pub fn get_level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        i32::MIN..=-8 => Level::Quiet,
        -7..=0 => Level::Panic,
        1..=8 => Level::Fatal,
        9..=16 => Level::Error,
        17..=24 => Level::Warning,
        25..=32 => Level::Info,
        33..=40 => Level::Verbose,
        41..=48 => Level::Debug,
        _ => Level::Trace,
    }
}

/// `av_log` — the single funnel. Level-gated against the global level; writes
/// to stderr. Context, when present, prefixes the message in brackets the way
/// the default callback shows `[yuv4mpegpipe @ 0x…]` without the pointer.
pub fn log(ctx: Option<&str>, level: Level, args: fmt::Arguments<'_>) {
    if level > get_level() {
        return;
    }
    match ctx {
        Some(name) => eprintln!("[{name}] {args}"),
        None => eprintln!("{args}"),
    }
}

/// `av_log(ctx, AV_LOG_ERROR, …)`.
#[macro_export]
macro_rules! log_error {
    ($ctx:expr, $($arg:tt)*) => { $crate::util::log::log($ctx, $crate::util::log::Level::Error, format_args!($($arg)*)) };
}

/// `av_log(ctx, AV_LOG_WARNING, …)`.
#[macro_export]
macro_rules! log_warning {
    ($ctx:expr, $($arg:tt)*) => { $crate::util::log::log($ctx, $crate::util::log::Level::Warning, format_args!($($arg)*)) };
}

/// `av_log(ctx, AV_LOG_INFO, …)`.
#[macro_export]
macro_rules! log_info {
    ($ctx:expr, $($arg:tt)*) => { $crate::util::log::log($ctx, $crate::util::log::Level::Info, format_args!($($arg)*)) };
}

/// `av_log(ctx, AV_LOG_VERBOSE, …)`.
#[macro_export]
macro_rules! log_verbose {
    ($ctx:expr, $($arg:tt)*) => { $crate::util::log::log($ctx, $crate::util::log::Level::Verbose, format_args!($($arg)*)) };
}

/// `av_log(ctx, AV_LOG_DEBUG, …)`.
#[macro_export]
macro_rules! log_debug {
    ($ctx:expr, $($arg:tt)*) => { $crate::util::log::log($ctx, $crate::util::log::Level::Debug, format_args!($($arg)*)) };
}
