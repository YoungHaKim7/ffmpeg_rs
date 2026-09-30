// ---------------------------------------------------------------------
// Constants (mpegaudio.h)
// ---------------------------------------------------------------------

/// `MPA_MAX_CHANNELS` (`mpegaudio.h:42`).
pub const MPA_MAX_CHANNELS: usize = 2;
/// `SBLIMIT` — number of subbands (`mpegaudio.h:44`).
pub const SBLIMIT: usize = 32;
/// `MPA_FRAME_SIZE` — max frame size in samples (`mpegaudio.h:37`).
pub const MPA_FRAME_SIZE: usize = 1152;
/// `BACKSTEP_SIZE` (`mpegaudiodec_template.c:53`).
pub const BACKSTEP_SIZE: usize = 512;
/// `EXTRABYTES` (`mpegaudiodec_template.c:54`).
pub const EXTRABYTES: usize = 24;
/// `LAST_BUF_SIZE = 2 * BACKSTEP_SIZE + EXTRABYTES` (`:55`).
pub const LAST_BUF_SIZE: usize = 2 * BACKSTEP_SIZE + EXTRABYTES;

/// `MPA_STEREO`..`MPA_MONO` (`mpegaudio.h:46-49`).
pub const MPA_JSTEREO: i32 = 1;
pub const MPA_MONO: i32 = 3;
/// `MODE_EXT_I_STEREO` / `MODE_EXT_MS_STEREO` (`mpegaudiodata.h:37-38`).
pub const MODE_EXT_I_STEREO: i32 = 1;
pub const MODE_EXT_MS_STEREO: i32 = 2;

/// `IMDCT_SCALAR` (`mpegaudio.h:56`, `mpegaudio_tablegen.h:46`).
pub const IMDCT_SCALAR: f64 = 1.759;
/// `HEADER_SIZE` (`mpegaudiodec_template.c:101`).
pub const HEADER_SIZE: usize = 4;
/// `TABLE_4_3_SIZE` (`mpegaudiodata.h:48`).
pub const TABLE_4_3_SIZE: usize = (8191 + 16) * 4;
/// `MDCT_BUF_SIZE = FFALIGN(36, 2*4)` (`mpegaudiodsp.h:89`).
pub const MDCT_BUF_SIZE: usize = 40;
