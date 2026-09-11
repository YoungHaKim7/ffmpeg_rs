// ---------------------------------------------------------------------------
// version.c:36-44
// ---------------------------------------------------------------------------

/// `LIBSWRESAMPLE_VERSION_MAJOR` (`version_major.h:29`).
pub const LIBSWRESAMPLE_VERSION_MAJOR: u32 = 7;
/// `LIBSWRESAMPLE_VERSION_MINOR` (`version.h:23`).
pub const LIBSWRESAMPLE_VERSION_MINOR: u32 = 3;
/// `LIBSWRESAMPLE_VERSION_MICRO` (`version.h:24`).
pub const LIBSWRESAMPLE_VERSION_MICRO: u32 = 100;
/// `LIBSWRESAMPLE_VERSION_INT` = `AV_VERSION_INT(7, 3, 100)` (`version.h:26`).
pub const LIBSWRESAMPLE_VERSION_INT: u32 = (LIBSWRESAMPLE_VERSION_MAJOR << 16)
    | (LIBSWRESAMPLE_VERSION_MINOR << 8)
    | LIBSWRESAMPLE_VERSION_MICRO;

/// `swresample_version()` (`version.c:31`).
pub fn swresample_version() -> u32 {
    LIBSWRESAMPLE_VERSION_INT
}

/// `swresample_configuration()` (`version.c:36`) — C returns
/// `FFMPEG_CONFIGURATION` from the build; a cargo build has no configure
/// line, so `""`.
pub fn swresample_configuration() -> &'static str {
    ""
}

/// `swresample_license()` (`version.c:41`).
pub fn swresample_license() -> &'static str {
    "LGPL version 2.1 or later"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_accessors() {
        // version.c + version.h: AV_VERSION_INT(7, 3, 100).
        assert_eq!(LIBSWRESAMPLE_VERSION_INT, 7 << 16 | 3 << 8 | 100);
        assert_eq!(LIBSWRESAMPLE_VERSION_INT, 459620);
        assert_eq!(swresample_version(), 459620);
        assert_eq!(swresample_license(), "LGPL version 2.1 or later");
    }
}
