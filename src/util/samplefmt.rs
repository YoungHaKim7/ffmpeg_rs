//! Audio sample formats — port of `libavutil/samplefmt.{h,c}`.
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `enum AVSampleFormat` (`samplefmt.h:55-80`) | [`SampleFormat`] |
//! | `sample_fmt_info[]` table (`samplefmt.c:36-50`) | [`SampleFormat`] methods (`name`/`bytes_per_sample`/`is_planar`/`alt_form`) |
//! | `av_get_sample_fmt` (`samplefmt.c:59-67`) | [`SampleFormat::from_name`] |
//! | `av_get_sample_fmt_name` (`samplefmt.c:52-57`) | [`SampleFormat::name`] |
//! | `av_get_alt_sample_fmt` (`samplefmt.c:69-76`) | [`SampleFormat::alt`] |
//! | `av_get_packed_sample_fmt` (`samplefmt.c:78-85`) | [`SampleFormat::packed`] |
//! | `av_get_planar_sample_fmt` (`samplefmt.c:87-94`) | [`SampleFormat::planar`] |
//! | `av_get_bytes_per_sample` (`samplefmt.c:109-113`) | [`SampleFormat::bytes_per_sample`] |
//! | `av_sample_fmt_is_planar` (`samplefmt.c:115-120`) | [`SampleFormat::is_planar`] |
//! | `av_samples_get_buffer_size` (`samplefmt.c:122-152`) | [`samples_get_buffer_size`] |
//! | `av_samples_fill_arrays` (`samplefmt.c:154-181`) | [`samples_fill_arrays`] / [`SampleArrays`] |
//! | `av_samples_alloc` (`samplefmt.c:183-206`) | [`samples_alloc`] |
//! | `av_samples_set_silence` (`samplefmt.c:247-269`) | [`silence_byte`] (+ `AudioFrame::set_silence`) |
//! | `av_samples_copy` (`samplefmt.c:223-245`) | `AudioFrame::copy_samples` (in [`super::audio_frame`]) |
//!
//! ## Data layout (samplefmt.h:44-52)
//!
//! Planar formats store each channel in its own plane; `linesize` is one
//! plane's byte size. Packed formats use a single plane with all channels
//! interleaved; `linesize` is the whole buffer's size.
//!
//! ## Skipped C sites (guards cited)
//!
//! * `av_get_sample_fmt_string` (`samplefmt.c:96-107`) — no caller in the
//!   ported audio pipeline (dump paths use `name()`). If fftools ever needs
//!   it, port as `fmt_string(fmt) -> String` with C's two exact formats:
//!   header `"name  " " depth"` and per-format `"%-6s" "   %2d "`.
//! * `av_samples_alloc_array_and_samples` (`samplefmt.c:208-221`) — only
//!   `calloc`s a pointer array; Rust `Vec`/`Arc` makes it meaningless.
//! * C's `AVERROR(ENOMEM)` paths (`samplefmt.c:194`) are unrepresentable
//!   (allocation failure aborts in Rust).
//! * The `AV_SAMPLE_FMT_NONE`/`NB` sentinel arms of every C function
//!   (`sample_fmt < 0 || >= AV_SAMPLE_FMT_NB`) vanish: the enum is closed,
//!   "no format yet" is `Option<SampleFormat>` at use sites.
//!
//! ## Deviations from C (documented per function)
//!
//! * C `FFALIGN` is a bitmask AND (requires power-of-two align); this port
//!   uses a `div_ceil`-based `align_up` correct for *any* `align`. Only
//!   observable for non-power-of-two `align` values, where C silently
//!   misaligns — no ported caller passes those.
//! * C's `INT_MAX` overflow guards (`samplefmt.c:134-136,142-144`) become
//!   checked `u64` arithmetic → [`Error::OutOfRange`] (`error.rs` repo
//!   convention). Results near `INT_MAX` succeed here where C fails; a
//!   semantic widening, harmless on 64-bit.

use super::error::{Error, Result};

/// `enum AVSampleFormat` (`samplefmt.h:55-80`) — all 13 formats.
///
/// Discriminants are C's exact values (declaration order = discriminant
/// order, so `PartialOrd`/`Ord` match C's integer comparisons — note
/// `S64`/`S64p` sit at 10/11, *after* `Dblp`, and `Dsd` is 12).
/// `AV_SAMPLE_FMT_NONE` (−1) is not a variant; `AV_SAMPLE_FMT_NB` (13) is
/// not a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SampleFormat {
    /// `AV_SAMPLE_FMT_U8` = 0 — unsigned 8 bits.
    U8,
    /// `AV_SAMPLE_FMT_S16` = 1 — signed 16 bits.
    S16,
    /// `AV_SAMPLE_FMT_S32` = 2 — signed 32 bits.
    S32,
    /// `AV_SAMPLE_FMT_FLT` = 3 — float.
    Flt,
    /// `AV_SAMPLE_FMT_DBL` = 4 — double.
    Dbl,
    /// `AV_SAMPLE_FMT_U8P` = 5 — unsigned 8 bits, planar.
    U8p,
    /// `AV_SAMPLE_FMT_S16P` = 6 — signed 16 bits, planar.
    S16p,
    /// `AV_SAMPLE_FMT_S32P` = 7 — signed 32 bits, planar.
    S32p,
    /// `AV_SAMPLE_FMT_FLTP` = 8 — float, planar.
    Fltp,
    /// `AV_SAMPLE_FMT_DBLP` = 9 — double, planar.
    Dblp,
    /// `AV_SAMPLE_FMT_S64` = 10 — signed 64 bits (AFTER `DBLP` in C).
    S64,
    /// `AV_SAMPLE_FMT_S64P` = 11 — signed 64 bits, planar.
    S64p,
    /// `AV_SAMPLE_FMT_DSD` = 12 — Direct Stream Digital bitstream,
    /// interleaved; one byte = 8 one-bit samples, MSB first
    /// (`samplefmt.h:71-77`). Packed, 8 bits, its own altform.
    Dsd,
}

impl SampleFormat {
    /// Every format in enum (C discriminant) order.
    pub const ALL: &'static [SampleFormat] = &[
        SampleFormat::U8,
        SampleFormat::S16,
        SampleFormat::S32,
        SampleFormat::Flt,
        SampleFormat::Dbl,
        SampleFormat::U8p,
        SampleFormat::S16p,
        SampleFormat::S32p,
        SampleFormat::Fltp,
        SampleFormat::Dblp,
        SampleFormat::S64,
        SampleFormat::S64p,
        SampleFormat::Dsd,
    ];

    /// `sample_fmt_info[].name` (`samplefmt.c:37-49`) — exact C strings.
    pub const fn name(self) -> &'static str {
        match self {
            SampleFormat::U8 => "u8",
            SampleFormat::S16 => "s16",
            SampleFormat::S32 => "s32",
            SampleFormat::Flt => "flt",
            SampleFormat::Dbl => "dbl",
            SampleFormat::U8p => "u8p",
            SampleFormat::S16p => "s16p",
            SampleFormat::S32p => "s32p",
            SampleFormat::Fltp => "fltp",
            SampleFormat::Dblp => "dblp",
            SampleFormat::S64 => "s64",
            SampleFormat::S64p => "s64p",
            SampleFormat::Dsd => "dsd",
        }
    }

    /// `av_get_sample_fmt` (`samplefmt.c:59-67`) — plain `strcmp` loop over
    /// the table names, exact match only; `None` maps to
    /// `AV_SAMPLE_FMT_NONE`. There is **no** trailing-`p` rule and no alias
    /// table: `"u8p"`/`"fltp"` are literal table entries.
    pub fn from_name(name: &str) -> Option<SampleFormat> {
        SampleFormat::ALL.iter().copied().find(|f| f.name() == name)
    }

    /// `av_get_bytes_per_sample` (`samplefmt.c:109-113`) =
    /// `sample_fmt_info[].bits >> 3`.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            SampleFormat::U8 | SampleFormat::U8p | SampleFormat::Dsd => 1,
            SampleFormat::S16 | SampleFormat::S16p => 2,
            SampleFormat::S32 | SampleFormat::Flt | SampleFormat::S32p | SampleFormat::Fltp => 4,
            SampleFormat::Dbl | SampleFormat::Dblp | SampleFormat::S64 | SampleFormat::S64p => 8,
        }
    }

    /// `av_sample_fmt_is_planar` (`samplefmt.c:115-120`) —
    /// `sample_fmt_info[].planar`. `Dsd` is packed (0).
    pub const fn is_planar(self) -> bool {
        matches!(
            self,
            SampleFormat::U8p
                | SampleFormat::S16p
                | SampleFormat::S32p
                | SampleFormat::Fltp
                | SampleFormat::Dblp
                | SampleFormat::S64p
        )
    }

    /// `sample_fmt_info[].altform` (`samplefmt.c:37-49`) — the table's
    /// planar↔packed alternative form. `Dsd` is a fixed point
    /// (`.altform = AV_SAMPLE_FMT_DSD`, `samplefmt.c:49`).
    pub const fn alt_form(self) -> SampleFormat {
        match self {
            SampleFormat::U8 => SampleFormat::U8p,
            SampleFormat::S16 => SampleFormat::S16p,
            SampleFormat::S32 => SampleFormat::S32p,
            SampleFormat::Flt => SampleFormat::Fltp,
            SampleFormat::Dbl => SampleFormat::Dblp,
            SampleFormat::U8p => SampleFormat::U8,
            SampleFormat::S16p => SampleFormat::S16,
            SampleFormat::S32p => SampleFormat::S32,
            SampleFormat::Fltp => SampleFormat::Flt,
            SampleFormat::Dblp => SampleFormat::Dbl,
            SampleFormat::S64 => SampleFormat::S64p,
            SampleFormat::S64p => SampleFormat::S64,
            SampleFormat::Dsd => SampleFormat::Dsd,
        }
    }

    /// `av_get_alt_sample_fmt` (`samplefmt.c:69-76`): the form of `self`
    /// matching the requested planarity; `self` if it already matches,
    /// else `alt_form()`. (C's `AV_SAMPLE_FMT_NONE` out-of-range arm is
    /// unreachable on the closed enum.)
    pub const fn alt(self, planar: bool) -> SampleFormat {
        if self.is_planar() == planar {
            self
        } else {
            self.alt_form()
        }
    }

    /// `av_get_packed_sample_fmt` (`samplefmt.c:78-85`).
    pub const fn packed(self) -> SampleFormat {
        self.alt(false)
    }

    /// `av_get_planar_sample_fmt` (`samplefmt.c:87-94`).
    pub const fn planar(self) -> SampleFormat {
        self.alt(true)
    }
}

impl std::fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `av_samples_get_buffer_size` (`samplefmt.c:122-152`).
///
/// C's `FFALIGN(x, a)` (`macros.h`) is a bitmask AND that requires a
/// power-of-two `a`; here it is computed as `x.div_ceil(a) * a` with every
/// multiply checked — correct for any `a ≥ 1` where C would silently
/// misalign (documented module deviation, no ported caller passes
/// non-power-of-two align).
///
/// Returns `(linesize, buffer_size)`: the per-plane linesize (for packed
/// formats the one interleaved plane's size) and the minimum total buffer
/// size (C's return value; the `linesize` out-param folds into the tuple —
/// "may be NULL" callers just ignore the element).
///
/// C algorithm, exactly (`samplefmt.c:130-151`):
///
/// 1. `nb_channels == 0 || nb_samples == 0` → `EINVAL` (C also rejects
///    `sample_size == 0`, unreachable on the closed enum).
/// 2. `align == 0` (auto): the **sample count** is rounded up to a
///    multiple of 32 (`FFALIGN(nb_samples, 32)`, c:134-139) and then *no*
///    extra byte alignment is applied (`align` becomes 1). `align == 1`
///    means exactly no padding.
/// 3. `line_size = FFALIGN(nb_samples * bps [ * nb_channels if packed ],
///    align)`; total = `line_size * nb_channels` (planar) or `line_size`
///    (packed).
///
/// C's `INT_MAX` guards (c:134-136, c:142-144) are replaced by checked
/// `u64` arithmetic → [`Error::OutOfRange`] on overflow (repo convention,
/// `error.rs:43-44`).
pub fn samples_get_buffer_size(
    nb_channels: usize,
    nb_samples: usize,
    sample_fmt: SampleFormat,
    align: usize,
) -> Result<(usize, usize)> {
    let bps = sample_fmt.bytes_per_sample();
    let planar = sample_fmt.is_planar();

    // c:130-131 — !sample_size || nb_samples <= 0 || nb_channels <= 0
    if nb_samples == 0 || nb_channels == 0 {
        return Err(Error::InvalidArgument(format!(
            "invalid audio buffer parameters: nb_channels={nb_channels}, nb_samples={nb_samples}, align={align}"
        )));
    }

    // c:134-139 — auto alignment: round the SAMPLE count up to 32.
    let mut samples = nb_samples as u64;
    let mut align = align as u64;
    if align == 0 {
        align = 1;
        samples = samples
            .div_ceil(32)
            .checked_mul(32)
            .ok_or(Error::OutOfRange)?;
    }

    let per_plane = samples.checked_mul(bps as u64).ok_or(Error::OutOfRange)?;
    let line = if planar {
        per_plane
    } else {
        per_plane
            .checked_mul(nb_channels as u64)
            .ok_or(Error::OutOfRange)?
    };
    // FFALIGN(line, align) — every multiply checked (C: INT_MAX guards at
    // c:134-136/c:142-144 -> OutOfRange here).
    let line = line
        .div_ceil(align)
        .checked_mul(align)
        .ok_or(Error::OutOfRange)?;
    let total = if planar {
        line.checked_mul(nb_channels as u64)
            .ok_or(Error::OutOfRange)?
    } else {
        line
    };

    let line = usize::try_from(line).map_err(|_| Error::OutOfRange)?;
    let total = usize::try_from(total).map_err(|_| Error::OutOfRange)?;
    Ok((line, total))
}

/// Rust shape of `av_samples_fill_arrays`' outputs (`samplefmt.c:154-181`).
///
/// `plane_offsets[ch]` is where plane `ch` starts inside the single backing
/// buffer — exactly C's `audio_data[ch] = audio_data[ch-1] + line_size`
/// (c:176-178). `plane_offsets.len()` is `nb_channels` (planar) or 1
/// (packed) — C zeroes exactly that many pointers (c:169-171); unused
/// slots simply do not exist here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleArrays {
    /// Per-plane linesize in bytes (packed: the interleaved plane's size).
    pub linesize: usize,
    /// Minimum total buffer size (C's return value, c:180).
    pub buf_size: usize,
    /// Start offset of each plane inside the buffer (`[0]` for packed).
    pub plane_offsets: Vec<usize>,
}

/// `av_samples_fill_arrays` (`samplefmt.c:154-181`) — the pointer
/// arithmetic of slicing one compact buffer into planes, without the
/// pointers. All math delegates to [`samples_get_buffer_size`] (c:161-162);
/// errors propagate unchanged. C's `buf == NULL` early return (c:173-174)
/// has no analog (no pointers to fill).
pub fn samples_fill_arrays(
    nb_channels: usize,
    nb_samples: usize,
    sample_fmt: SampleFormat,
    align: usize,
) -> Result<SampleArrays> {
    let (linesize, buf_size) = samples_get_buffer_size(nb_channels, nb_samples, sample_fmt, align)?;
    let planar = sample_fmt.is_planar();
    let nb_planes = if planar { nb_channels } else { 1 };

    let mut plane_offsets = Vec::with_capacity(nb_planes);
    for ch in 0..nb_planes {
        // c:176-178: audio_data[ch] = audio_data[ch-1] + line_size
        plane_offsets.push(ch * linesize);
    }
    Ok(SampleArrays {
        linesize,
        buf_size,
        plane_offsets,
    })
}

/// The `av_samples_set_silence` fill byte (`samplefmt.c:256-261`):
/// `0x80` for the unsigned 8-bit formats (midpoint), `0x69` for DSD
/// ("only ultrasonic tones, filtered out on playback", c:259), `0x00` for
/// every signed/float format.
pub const fn silence_byte(sample_fmt: SampleFormat) -> u8 {
    match sample_fmt {
        SampleFormat::U8 | SampleFormat::U8p => 0x80,
        SampleFormat::Dsd => 0x69,
        _ => 0x00,
    }
}

/// `av_samples_alloc` (`samplefmt.c:183-206`): one compact buffer of
/// exactly `buf_size` bytes, every byte pre-filled with the format's
/// silence (via `av_samples_set_silence`, c:203), plus the plane offsets
/// of [`samples_fill_arrays`] (c:196-197).
///
/// C's `AVERROR(ENOMEM)` path (c:194) is unrepresentable (allocation
/// failure aborts in Rust).
pub fn samples_alloc(
    nb_channels: usize,
    nb_samples: usize,
    sample_fmt: SampleFormat,
    align: usize,
) -> Result<(Vec<u8>, SampleArrays)> {
    let arrays = samples_fill_arrays(nb_channels, nb_samples, sample_fmt, align)?;
    let buffer = vec![silence_byte(sample_fmt); arrays.buf_size];
    Ok((buffer, arrays))
}
