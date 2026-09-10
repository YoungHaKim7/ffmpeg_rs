//! libswresample — the audio resampler/converter (Phase 4a).
//!
//! Port map (all files in `FFmpeg/libswresample/`):
//!
//! | file | ports | what it holds |
//! |---|---|---|
//! | [`mod`] | `swresample.c` + `swresample_internal.h` + `options.c` + `swresample_frame.c` + `version.c` | [`AudioData`], [`SwrContext`], the `swr_*` public API |
//! | [`audioconvert`] | `audioconvert.c` | the sample-format pair conversion matrix |
//! | [`rematrix`] | `rematrix.c` + `rematrix_template.c` | the channel-mix matrix build + apply |
//! | [`resample`] | `resample.c` + `resample_template.c` | the Kaiser-windowed polyphase rate converter |
//!
//! ## Driver C → Rust map (`swresample.c`, line-referenced)
//!
//! | C site | here |
//! |---|---|
//! | `swri_check_chlayout` `:33-45` | [`SwrContext::check_chlayout`] (WARNING text verbatim, log context `"SWR"` = `context_to_name`, `options.c:129-131`) |
//! | `swr_set_channel_mapping` `:47-52` | [`SwrContext::set_channel_mapping`] (the `s->in_convert` guard = `self.in_convert.is_some()`) |
//! | `swr_alloc_set_opts2` `:54-96` | [`SwrContext::alloc_set_opts2`] (option writes become typed field stores; the fail label logs `"Failed to set option"` and consumes the context) |
//! | `set_audiodata_fmt` `:98-104` | [`set_audiodata_fmt`] (mono forces planar — load-bearing downstream) |
//! | `clear_context`/`swr_free`/`swr_close` `:111-154` | [`SwrContext::clear_context`]/[`Drop`]/[`SwrContext::close`] — `outpts`, `firstpts`, `drop_output` and the option fields survive `close` (C keeps them too) |
//! | `swr_init` `:156-424` | [`SwrContext::init`] — the 42-step validation/derivation order with verbatim error texts; `goto fail` = `close()` + `Err` |
//! | `swri_realloc_audio` `:426-456` | [`AudioData::realloc_audio`] (grow-by-doubling; the 32-byte `ALIGN` stride is dropped with the SIMD it served — see [`AudioData`]) |
//! | `copy`/`fill_audiodata`/`reversefill_audiodata`/`buf_set` `:458-508` | [`copy_views`]/[`SwrContext::take_input`]/(direct)/the [`view`] sample-offset copy — the C pointer views become copies, see divergences |
//! | `resample()` `:514-607` | [`SwrContext::resample_stage`] — the invert/staging loop with `padless = 7` (x86-64 reference build, `ARCH_X86 && engine == SWR`) |
//! | `swr_convert_internal()` `:609-737` | [`SwrContext::convert_internal`] + the [`Tgt`] routing enum mirroring C's pointer-alias tree exactly (the `dither` block `:687-732` is dropped — `dither.method` is always `None` after init) |
//! | `swr_is_initialized`/`swr_convert` `:739-856` | [`SwrContext::is_initialized`]/[`SwrContext::convert`] — the drop loop (`:758-779`), flush handling (`:781-793`), the resample path (`:795-802`) and the equal-rate FIFO path (`:803-855`) |
//! | `swr_drop_output` `:858-867` | [`SwrContext::drop_output`] (`MAX_DROP_STEP` recursion-free) |
//! | `swr_inject_silence` `:869-899` | [`SwrContext::inject_silence`] (iterative chunking replaces C's self-recursion; `0x69` DSD fill byte kept — `Dsd` exists in this port's [`SampleFormat`] set) |
//! | `swr_get_delay` `:901-907` | [`SwrContext::get_delay`] (FIFO formula here, resampled formula delegated to the backend) |
//! | `swr_get_out_samples` `:909-929` | [`SwrContext::get_out_samples`] |
//! | `swr_set_compensation` `:931-949` | [`SwrContext::set_compensation`] (the forced re-init destroys buffered state — faithful) |
//! | `swr_next_pts` `:951-983` | [`SwrContext::next_pts`] — MODE 1/MODE 2, hard (`inject`/`drop`) and soft (`set_compensation`) arms |
//! | `swresample_frame.c` `:27-174` | [`SwrContext::config_frame`]/[`SwrContext::convert_frame`] (`config_changed` texts keep the input/output distinction greppable) |
//! | `options.c:40-156` | [`SwrOptions`] — the defaults verbatim (see [`SwrOptions::OPTION_NAMES`] for the canonical+alias table) |
//! | `version.c:36-44` | [`swresample_version`]/[`swresample_configuration`]/[`swresample_license`] |
//!
//! Not ported: soxr (engine 1 — [`SwrEngine::Soxr`] keeps C's exact
//! "unavailable" error), the SIMD dispatch (`*_dsp.c` x86 variants —
//! scalar only), and `dither.c` (any dither method that survives
//! `swri_dither_init`'s `scale == 0` disabling is rejected with
//! [`Error::Unsupported`]; the default `dither_method = None` path is
//! bit-identical to C).
//!
//! ## Divergences from C (driver-side, each cited)
//!
//! * **Pointer views become copies.** C's `AudioData.ch[]` pointers and
//!   `buf_set` express zero-copy offsets into one allocation; the landed
//!   [`AudioData`] (single `Arc<[u8]>`, planes derived from `count`) has no
//!   borrowed-view form, so every `buf_set`/`fill_audiodata` in the driver
//!   materializes an offset copy ([`view`], [`take_input`]) and every write
//!   splices back ([`splice`]). Byte-for-byte identical output — the C
//!   aliasing is a performance device, invisible to the backends' plane
//!   reads. The one observable corner: the resampler's documented
//!   "last tap may read one stale sample past `src_size` within capacity"
//!   (`resample.rs` divergences) — views preserve the source tail bytes C's
//!   offset pointers would expose, and caller-input staging is padded with
//!   2 zero samples so that read stays deterministic where C would read
//!   heap slack.
//! * **The drop loop propagates errors.** C's `swr_drop_output` inner-call
//!   failure sits behind `av_assert0` (unreachable in practice); the port
//!   returns the `Err` (`swresample.c:777-778`).
//! * The `ASSERT_LEVEL > 1` `max_output` cross-checks (`:754-756, 800,
//!   853`) are dropped, replaced by the `get_out_samples_upper_bound` unit
//!   test.
//! * `av_assert0/1/2` → `debug_assert!`/`debug_assert!`/dropped; the C
//!   `av_assert0(0)`-style aborts surface as `Error::Unsupported` from the
//!   backends.
//! * `swr_get_in_samples` does not exist in libswresample (a libavresample
//!   remnant; grep of the C tree confirms). It is provided here as the
//!   buffered-input-count query callers actually want, documented at the
//!   method.

pub mod audioconvert;
pub mod rematrix;
pub mod resample;

use crate::util::{
    channel_layout::{ChannelLayout, Order},
    error::{Error, Result},
    samplefmt::SampleFormat,
};
use crate::{log_debug, log_error, log_verbose, log_warning};

use audioconvert::{AudioConvert, swri_audio_convert};
use rematrix::{CustomRematrix, MatrixEncoding, RematrixContext, RematrixOptions};
use resample::{FilterType, ResampleContext};

/// `SWR_CH_MAX` (`swresample_internal.h:28`).
pub const SWR_CH_MAX: usize = 64;

/// `ALIGN` (`swresample.c:31`) — the 32-byte plane stride of
/// `swri_realloc_audio` — is dropped with the SIMD it served (see
/// [`AudioData`]'s doc); not reproduced as a constant.

/// `MAX_DROP_STEP` (`swresample.c:761`).
pub const MAX_DROP_STEP: i32 = 16384;
/// `MAX_SILENCE_STEP` (`swresample.c:876`).
pub const MAX_SILENCE_STEP: i32 = 16384;

/// `AV_NOPTS_VALUE` (`libavutil/avutil.h`) via [`crate::NOPTS`].
pub const NOPTS: i64 = crate::NOPTS;

/// The log-context name every driver `av_log` uses (`context_to_name`,
/// `options.c:129-131` — the AVClass `class_name` is `"SWResampler"` but
/// the default callback prints `item_name` = `"SWR"`).
const LOG_CTX: Option<&str> = Some("SWR");

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

/// `AudioData` from `swresample_internal.h:47-55`: one `Arc<[u8]>` backing
/// buffer replaces C's `uint8_t *data` + `ch[SWR_CH_MAX]` pointer array,
/// reproducing the aliasing C sets up in `swri_realloc_audio`
/// (`swresample.c:426-456`) and `fill_audiodata` (`swresample.c:471-483`):
///
/// * planar: channel `i` occupies `data[i*count*bps .. (i+1)*count*bps]`
///   (C's 32-byte `ALIGN` stride padding at `swresample.c:31,438` is dropped —
///   it exists only for SIMD and is invisible to every caller);
/// * packed: channel `i` is the alias starting at `data[i*bps ..]` running to
///   the end of the buffer (C `ch[i] = data + i*bps`, `swresample.c:448`).
///
/// `count` is the capacity in samples per channel (C's grow-by-doubling
/// [`swri_realloc_audio`] is [`AudioData::realloc_audio`]). Occupancy is the
/// driver's business (`in_buffer_count` etc.) — never this field.
#[derive(Clone)]
pub struct AudioData {
    /// Backing buffer shared by all channels (aliases, not copies).
    pub data: std::sync::Arc<[u8]>,
    /// Number of channels (`ch_count`).
    pub ch_count: usize,
    /// Bytes per sample (`bps`).
    pub bps: usize,
    /// Capacity in samples per channel (`count`).
    pub count: usize,
    /// 1 if planar audio, 0 otherwise (`planar`).
    pub planar: bool,
    /// Sample format (`fmt`).
    pub fmt: SampleFormat,
}

impl AudioData {
    /// Zeroed buffer of `ch_count * count * bps` bytes (the `av_calloc`
    /// analog of `swri_realloc_audio`, `swresample.c:438`).
    pub fn new(fmt: SampleFormat, ch_count: usize, count: usize) -> Self {
        let bps = fmt.bytes_per_sample();
        AudioData {
            data: vec![0u8; ch_count * count * bps].into(),
            ch_count,
            bps,
            count,
            planar: fmt.is_planar(),
            fmt,
        }
    }

    /// The `memset(a, 0, sizeof(*a))` of `free_temp` (`swresample.c:106-109`)
    /// and `clear_context` — an empty descriptor with no channels.
    pub(crate) fn empty() -> Self {
        AudioData {
            data: std::sync::Arc::from(Vec::new()),
            ch_count: 0,
            bps: 0,
            count: 0,
            planar: false,
            fmt: SampleFormat::U8,
        }
    }

    /// Channel `ch`'s plane — `None` when the channel's address falls outside
    /// the backing buffer (C's NULL `ch[]` slot, the skip case at
    /// `audioconvert.c:259-260`).
    pub fn plane(&self, ch: usize) -> Option<&[u8]> {
        let start = if self.planar {
            ch * self.count * self.bps
        } else {
            ch * self.bps
        };
        if self.planar {
            self.data.get(start..start + self.count * self.bps)
        } else {
            self.data.get(start..)
        }
    }

    /// Mutable channel `ch` plane, via a single `Arc::make_mut` borrow so
    /// packed channels (which alias one buffer) can be written at their
    /// offsets without aliasing violations.
    pub fn plane_bytes_mut(&mut self, ch: usize) -> Option<&mut [u8]> {
        let data = std::sync::Arc::make_mut(&mut self.data);
        let start = if self.planar {
            ch * self.count * self.bps
        } else {
            ch * self.bps
        };
        if self.planar {
            data.get_mut(start..start + self.count * self.bps)
        } else {
            data.get_mut(start..)
        }
    }

    /// Whole backing buffer, mutable (for the driver zone).
    pub fn data_mut(&mut self) -> &mut [u8] {
        std::sync::Arc::make_mut(&mut self.data)
    }

    /// `swri_realloc_audio` (`swresample.c:426-456`): grow the buffer to at
    /// least `count` samples per channel, doubling the request (C `:436`),
    /// preserving the old contents (planar: per-channel prefix; packed: one
    /// prefix — C `:448-451`) and zero-filling the rest (C's `av_calloc`,
    /// `:444` — the grown `in_buffer` tail is silence).
    ///
    /// Returns `true` when the buffer grew (C's `1`; the dither
    /// noise-regeneration trigger — unused once `dither.c` is dropped).
    pub(crate) fn realloc_audio(&mut self, count: usize) -> Result<bool> {
        // :441-442 — av_assert0(a->bps); av_assert0(a->ch_count).
        debug_assert!(self.bps != 0 && self.ch_count != 0);
        if self.bps == 0 || self.ch_count == 0 {
            return Err(Error::InvalidArgument(
                "swri_realloc_audio on a channel-less AudioData".into(),
            ));
        }
        // :430 — count < 0 is impossible for usize; the INT_MAX/2 clamp stays.
        if count > (i32::MAX as usize) / 2 / self.bps / self.ch_count {
            return Err(Error::InvalidArgument(
                "swri_realloc_audio: sample count out of range".into(),
            ));
        }
        if self.count >= count {
            return Ok(false);
        }
        let new_count = count * 2; // :436
        let mut data = vec![0u8; self.ch_count * new_count * self.bps];
        if self.planar {
            for ch in 0..self.ch_count {
                let from = ch * self.count * self.bps;
                let to = ch * new_count * self.bps;
                data[to..to + self.count * self.bps]
                    .copy_from_slice(&self.data[from..from + self.count * self.bps]);
            }
        } else {
            let n = self.count * self.ch_count * self.bps;
            data[..n].copy_from_slice(&self.data[..n]);
        }
        self.data = data.into();
        self.count = new_count;
        Ok(true)
    }
}

/// `set_audiodata_fmt` (`swresample.c:98-104`) — and the mono exception:
/// `if (a->ch_count == 1) a->planar = 1` (`:102-103`). Mono interleaved is
/// treated as planar everywhere downstream, which changes `fill_audiodata`
/// semantics for mono packed input (the single caller slice doubles as the
/// one plane).
fn set_audiodata_fmt(a: &mut AudioData, fmt: SampleFormat) {
    a.fmt = fmt;
    a.bps = fmt.bytes_per_sample();
    a.planar = fmt.is_planar();
    if a.ch_count == 1 {
        a.planar = true;
    }
}

/// C's `buf_set(&view, &a, off)` (`swresample.c:499-508`): a sample-offset
/// read view of `a`. The port has no borrowed pointer form, so this copies
/// each plane from `off` to the end of its capacity — the tail bytes past
/// occupancy are preserved verbatim because the resampler's kernels may
/// legally read one stale sample past `src_size` within capacity (the
/// `resample.rs` divergences note). `off` counts samples (frames).
pub(crate) fn view(a: &AudioData, off: usize) -> AudioData {
    debug_assert!(
        off <= a.count,
        "view offset {off} past capacity {}",
        a.count
    );
    let ncount = a.count - off;
    let mut out = AudioData {
        data: std::sync::Arc::from(vec![0u8; a.ch_count * ncount * a.bps]),
        ch_count: a.ch_count,
        bps: a.bps,
        count: ncount,
        planar: a.planar,
        fmt: a.fmt,
    };
    if a.planar {
        for ch in 0..a.ch_count {
            let from = ch * a.count * a.bps + off * a.bps;
            let to = ch * ncount * a.bps;
            let n = ncount * a.bps;
            out.data_mut()[to..to + n].copy_from_slice(&a.data[from..from + n]);
        }
    } else {
        let from = off * a.ch_count * a.bps;
        let n = ncount * a.ch_count * a.bps;
        out.data_mut()[..n].copy_from_slice(&a.data[from..from + n]);
    }
    out
}

/// C's `copy()` (`swresample.c:458-469`) with a destination sample offset:
/// write `count` samples of `src` (from its plane starts) into `dst` at
/// `off`. Same-geometry requirement as C's `av_assert0`s (`:460-462`).
pub(crate) fn splice(dst: &mut AudioData, off: usize, src: &AudioData, count: usize) {
    debug_assert_eq!(dst.planar, src.planar);
    debug_assert_eq!(dst.bps, src.bps);
    debug_assert_eq!(dst.ch_count, src.ch_count);
    debug_assert!(off + count <= dst.count);
    if dst.planar {
        for ch in 0..dst.ch_count {
            let to = ch * dst.count * dst.bps + off * dst.bps;
            let from = ch * src.count * dst.bps;
            let n = count * dst.bps;
            dst.data_mut()[to..to + n].copy_from_slice(&src.data[from..from + n]);
        }
    } else {
        let to = off * dst.ch_count * dst.bps;
        let n = count * dst.ch_count * dst.bps;
        dst.data_mut()[to..to + n].copy_from_slice(&src.data[..n]);
    }
}

/// The in-buffer compaction of `swresample.c:576-578, 828-830` — C's
/// overlapping `copy(&s->in_buffer, &tmp, count)` after `buf_set` to
/// `in_buffer_index`, i.e. a per-plane `memmove` of the occupancy to the
/// front (plain `copy_from_slice` on an overlapping range would be
/// UB-adjacent; C relies on glibc's forward copy).
pub(crate) fn compact(a: &mut AudioData, idx: usize, count: usize) {
    debug_assert!(idx + count <= a.count);
    let bps = a.bps;
    if a.planar {
        for ch in 0..a.ch_count {
            let base = ch * a.count * bps;
            let n = count * bps;
            let data = a.data_mut();
            data.copy_within(base + idx * bps..base + idx * bps + n, base);
        }
    } else {
        let n = count * a.ch_count * bps;
        let from = idx * a.ch_count * bps;
        a.data_mut().copy_within(from..from + n, 0);
    }
}

/// C's `copy(out, in, count)` (`swresample.c:458-469`) over two whole
/// buffers — the `swr_convert_internal:659` direct-copy path.
pub(crate) fn copy_views(out: &mut AudioData, in_: &AudioData, count: usize) {
    splice(out, 0, in_, count);
}

/// A fresh zeroed buffer with `a`'s geometry and `count` samples of
/// capacity — the write-side staging stand-in for C's caller planes.
/// `planar` is taken from `a` verbatim so the mono-forced-planar rule of
/// [`set_audiodata_fmt`] carries over.
pub(crate) fn scratch_like(a: &AudioData, count: usize) -> AudioData {
    AudioData {
        data: std::sync::Arc::from(vec![0u8; a.ch_count * count * a.bps]),
        ch_count: a.ch_count,
        bps: a.bps,
        count,
        planar: a.planar,
        fmt: a.fmt,
    }
}

// ---------------------------------------------------------------------------
// Public enums — swresample.h:143-177
// ---------------------------------------------------------------------------

/// `SWR_FLAG_RESAMPLE` (`swresample.h:143`) — the only flag: force
/// resampling even when the rates match.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SwrFlags(pub u32);

impl SwrFlags {
    /// `SWR_FLAG_RESAMPLE` = 1.
    pub const RESAMPLE: SwrFlags = SwrFlags(1);
    /// C flag containment (`flags & SWR_FLAG_RESAMPLE`).
    pub fn contains(self, other: SwrFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for SwrFlags {
    type Output = SwrFlags;
    fn bitor(self, rhs: SwrFlags) -> SwrFlags {
        SwrFlags(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for SwrFlags {
    fn bitor_assign(&mut self, rhs: SwrFlags) {
        self.0 |= rhs.0;
    }
}

/// `enum SwrEngine` (`swresample.h:166-170`). `Soxr` is accepted as an
/// option value but rejected at [`SwrContext::init`] with C's exact
/// non-soxr-build error — the soxr library is not ported.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwrEngine {
    /// `SWR_ENGINE_SWR` = 0 (the default, `options.c:93-94`).
    #[default]
    Swr,
    /// `SWR_ENGINE_SOXR` = 1 — unavailable (a C build without
    /// `CONFIG_LIBSOXR` behaves identically, `swresample.c:210-212`).
    Soxr,
}

/// `enum SwrDitherType` (`swresample.h:148-162`) — the non-noise-shaping
/// subset. The `SWR_DITHER_NS` family (64+) lives in the unported
/// `dither.c`/`noise_shaping_data.c` and is not representable; methods that
/// survive `swri_dither_init`'s `scale == 0` disabling are rejected at init
/// with [`Error::Unsupported`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwrDitherType {
    /// `SWR_DITHER_NONE` = 0 (the default).
    #[default]
    None,
    /// `SWR_DITHER_RECTANGULAR` = 1.
    Rectangular,
    /// `SWR_DITHER_TRIANGULAR` = 2.
    Triangular,
    /// `SWR_DITHER_TRIANGULAR_HIGHPASS` = 3.
    TriangularHighpass,
}

// ---------------------------------------------------------------------------
// SwrOptions — options.c:40-127 as a plain builder struct
// ---------------------------------------------------------------------------

/// The `options.c:40-127` option table as a plain builder struct (no
/// AVOption reflection; ffmpeg-CLI-shaped setters land in Phase 4b on top
/// of this). Every default is C's `.i64`/`.dbl` value verbatim; `float`
/// options store the C `AV_OPT_TYPE_FLOAT` narrowing.
///
/// Canonical + alias names (both accepted by C's table; Phase 4b CLI
/// parsing maps strings through this table without re-reading the C):
///
/// | canonical | alias | field |
/// |---|---|---|
/// | `in_sample_rate` | `isr` | [`Self::in_sample_rate`] |
/// | `out_sample_rate` | `osr` | [`Self::out_sample_rate`] |
/// | `in_sample_fmt` | `isf` | [`Self::in_sample_fmt`] |
/// | `out_sample_fmt` | `osf` | [`Self::out_sample_fmt`] |
/// | `internal_sample_fmt` | `tsf` | [`Self::internal_sample_fmt`] |
/// | `in_chlayout` | `ichl` | [`Self::in_chlayout`] |
/// | `out_chlayout` | `ochl` | [`Self::out_chlayout`] |
/// | `used_chlayout` | `uchl` | [`Self::used_chlayout`] |
/// | `center_mix_level` | `clev` | [`Self::clev`] |
/// | `surround_mix_level` | `slev` | [`Self::slev`] |
/// | `rematrix_volume` | `rmvol` | [`Self::rematrix_volume`] |
/// | `flags` | `swr_flags` | [`Self::flags`] (`res` = `SWR_FLAG_RESAMPLE`) |
/// | `resample_cutoff` | `cutoff` | [`Self::cutoff`] |
/// | `resampler` | — | [`Self::engine`] (`swr`/`soxr`) |
/// | `min_comp` | — | [`Self::min_compensation`] |
/// | `min_hard_comp` | — | [`Self::min_hard_compensation`] |
/// | `comp_duration` | — | [`Self::soft_compensation_duration`] |
/// | `max_soft_comp` | — | [`Self::max_soft_compensation`] |
/// | `async` | — | [`Self::async_]` |
/// | `first_pts` | — | [`Self::first_pts`] |
/// | `filter_type` | — | [`Self::filter_type`] (`cubic`/`blackman_nuttall`/`kaiser`) |
/// | `dither_method` | — | [`Self::dither_method`] (`rectangular`/`triangular`/`triangular_hp`) |
#[derive(Clone, Debug, PartialEq)]
pub struct SwrOptions {
    /// `isr` — default 0 (`options.c:41-42`).
    pub in_sample_rate: i32,
    /// `osr` — default 0 (`options.c:43-44`).
    pub out_sample_rate: i32,
    /// `isf` — default `AV_SAMPLE_FMT_NONE` = unset (`options.c:45-46`).
    pub in_sample_fmt: Option<SampleFormat>,
    /// `osf` — default unset (`options.c:47-48`).
    pub out_sample_fmt: Option<SampleFormat>,
    /// `tsf` (`user_int_sample_fmt`) — default unset (the ladder picks)
    /// (`options.c:49-50`).
    pub internal_sample_fmt: Option<SampleFormat>,
    /// `ichl` (`user_in_chlayout`) — default `{0}` unset (`options.c:51-52`).
    pub in_chlayout: ChannelLayout,
    /// `ochl` (`user_out_chlayout`) — default unset (`options.c:53-54`).
    pub out_chlayout: ChannelLayout,
    /// `uchl` (`user_used_chlayout`) — default unset (`options.c:55-56`).
    pub used_chlayout: ChannelLayout,
    /// `clev` — default `C_30DB = M_SQRT1_2` as float (`options.c:57-58`).
    pub clev: f32,
    /// `slev` — default `C_30DB` (`options.c:59-60`).
    pub slev: f32,
    /// `lfe_mix_level` — default 0 (`options.c:61`).
    pub lfe_mix_level: f32,
    /// `rmvol` — default 1.0 (`options.c:62-63`).
    pub rematrix_volume: f32,
    /// `rematrix_maxval` — default 0.0 = auto (`options.c:64`).
    pub rematrix_maxval: f32,
    /// `swr_flags` — default 0 (`options.c:66-67`).
    pub flags: SwrFlags,
    /// `dither_scale` (`dither.scale`) — default 1 (`options.c:70`).
    pub dither_scale: f32,
    /// `dither_method` (`user_dither_method`) — default `NONE` (`options.c:72`).
    pub dither_method: SwrDitherType,
    /// `filter_size` — default 32 (`options.c:84`).
    pub filter_size: i32,
    /// `phase_shift` — default 10 (`options.c:85`).
    pub phase_shift: i32,
    /// `linear_interp` — default 1 (`options.c:86`).
    pub linear_interp: bool,
    /// `exact_rational` — default 1 (`options.c:87`).
    pub exact_rational: bool,
    /// `cutoff` — default 0. (the engine substitutes 0.97) (`options.c:88,91`).
    pub cutoff: f64,
    /// `resampler` — default `swr` (`options.c:93`).
    pub engine: SwrEngine,
    /// `precision` (soxr-only, inert) — default 20.0 (`options.c:96-97`).
    pub precision: f64,
    /// `cheby` (soxr-only, inert) — default 0 (`options.c:98-99`).
    pub cheby: bool,
    /// `min_comp` — default `FLT_MAX` (`options.c:100-101`).
    pub min_compensation: f32,
    /// `min_hard_comp` — default 0.1 (`options.c:102-103`).
    pub min_hard_compensation: f32,
    /// `comp_duration` — default 1 (`options.c:104-105`).
    pub soft_compensation_duration: f32,
    /// `max_soft_comp` — default 0 (`options.c:106-107`).
    pub max_soft_compensation: f32,
    /// `async` — default 0 (`options.c:108-109`). `async` is a Rust keyword.
    pub async_: f32,
    /// `first_pts` (`firstpts_in_samples`) — default `AV_NOPTS_VALUE`
    /// (`options.c:110-111`).
    pub first_pts: i64,
    /// `matrix_encoding` — default `NONE` (`options.c:113`).
    pub matrix_encoding: MatrixEncoding,
    /// `filter_type` — default `kaiser` (`options.c:118`).
    pub filter_type: FilterType,
    /// `kaiser_beta` — default 9 (`options.c:123`).
    pub kaiser_beta: f64,
    /// `output_sample_bits` (`dither.output_sample_bits`) — default 0
    /// (`options.c:125`).
    pub output_sample_bits: i32,
}

impl Default for SwrOptions {
    /// `av_opt_set_defaults` over `options.c:40-127`, plus `swr_alloc`'s
    /// `firstpts = AV_NOPTS_VALUE` (`options.c:148-156` — every other field
    /// is the zeroed malloc plus these defaults).
    fn default() -> Self {
        SwrOptions {
            in_sample_rate: 0,
            out_sample_rate: 0,
            in_sample_fmt: None,
            out_sample_fmt: None,
            internal_sample_fmt: None,
            in_chlayout: ChannelLayout::default(),
            out_chlayout: ChannelLayout::default(),
            used_chlayout: ChannelLayout::default(),
            // C_30DB = M_SQRT1_2 = 0.70710678118654752440, stored to float.
            clev: 0.7071067811865476,
            slev: 0.7071067811865476,
            lfe_mix_level: 0.0,
            rematrix_volume: 1.0,
            rematrix_maxval: 0.0,
            flags: SwrFlags::default(),
            dither_scale: 1.0,
            dither_method: SwrDitherType::None,
            filter_size: 32,
            phase_shift: 10,
            linear_interp: true,
            exact_rational: true,
            cutoff: 0.0,
            engine: SwrEngine::Swr,
            precision: 20.0,
            cheby: false,
            min_compensation: f32::MAX,
            min_hard_compensation: 0.1,
            soft_compensation_duration: 1.0,
            max_soft_compensation: 0.0,
            async_: 0.0,
            first_pts: NOPTS,
            matrix_encoding: MatrixEncoding::None,
            filter_type: FilterType::Kaiser,
            kaiser_beta: 9.0,
            output_sample_bits: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SwrContext — swresample_internal.h:97-199 (driver subset)
// ---------------------------------------------------------------------------

/// The stage-routing decision tree of `swr_convert_internal`
/// (`swresample.c:638-665`) as an enum. C re-points `postin`/`midbuf`/
/// `preout` pointers at each other or at `out` to avoid copies; the port
/// walks the same tree and dispatches each backend stage over the matching
/// buffer (locals taken out of `self` for the duration — see
/// [`SwrContext::convert_internal`]). `CallerIn` is C's `postin = in`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tgt {
    CallerIn,
    Postin,
    Midbuf,
    Preout,
    Out,
}

/// `struct SwrContext` (`swresample_internal.h:97-199`) — the driver fields.
/// The rematrix matrix state (`matrix`/`matrix32`/`matrix_flt`/
/// `native_matrix`/`matrix_ch`/`native_one`) lives in the landed
/// [`RematrixContext`]; the custom matrix pre-init state is
/// [`CustomRematrix`]; `dither` shrinks to the method + the
/// `output_sample_bits` option (dither.c not ported). Dropped with cited
/// guards: `av_class`/`log_level_offset`/`log_ctx` (log macros take
/// `"SWR"`), `delayed_samples_fixup` (soxr-only), the `mix_*` SIMD slots
/// (rematrix.rs owns the scalar dispatch), and the `Resampler` vtable
/// (engine `swr` only — direct calls).
pub struct SwrContext {
    /// The user option set (`swr_alloc`'s zeroed-plus-defaults context).
    pub options: SwrOptions,

    // -- runtime state, valid between init() and close() --------------------
    /// `s->int_sample_fmt` — derived by the ladder when
    /// [`SwrOptions::internal_sample_fmt`] is `None`.
    int_sample_fmt: SampleFormat,
    /// `s->used_ch_layout` / `s->in_ch_layout` / `s->out_ch_layout`
    /// (`:104-106`).
    used_ch_layout: ChannelLayout,
    in_ch_layout: ChannelLayout,
    out_ch_layout: ChannelLayout,
    /// `s->dither.method` after `swri_dither_init`'s `scale == 0` disabling.
    dither_method: SwrDitherType,
    /// `s->rematrix` (`:145`) — rematrixing is needed.
    rematrix_needed: bool,
    /// `s->rematrix_first` (`:144`).
    resample_first: bool,
    /// `s->channel_map` (`:116`) — owned copy.
    channel_map: Option<Vec<i32>>,
    /// `swr_set_matrix` state (`rematrix_custom` + `s->matrix`).
    custom: Option<CustomRematrix>,

    /// `s->in` — format descriptor only (`count` stays 0; the caller planes
    /// are call-local staging in this port).
    in_: AudioData,
    /// `s->postin` (`:150`).
    postin: AudioData,
    /// `s->midbuf` (`:151`).
    midbuf: AudioData,
    /// `s->preout` (`:152`).
    preout: AudioData,
    /// `s->out` — format descriptor only.
    out: AudioData,
    /// `s->in_buffer` (`:153`) — the resample/FIFO staging (real data).
    in_buffer: AudioData,
    /// `s->silence` (`:154`).
    silence: AudioData,
    /// `s->drop_temp` (`:155`).
    drop_temp: AudioData,
    /// `s->in_buffer_index` / `s->in_buffer_count` (`:156-157`).
    in_buffer_index: usize,
    in_buffer_count: usize,
    /// `s->resample_in_constraint` (`:158`).
    resample_in_constraint: bool,
    /// `s->flushed` (`:159`).
    flushed: bool,
    /// `s->outpts` / `s->firstpts` (`:160-161`) — in-rate units.
    outpts: i64,
    firstpts: i64,
    /// `s->drop_output` (`:162`) — can go negative (cancellation sentinel
    /// inside the drop loop, `swresample.c:766-768`).
    drop_output: i32,

    /// `s->in_convert` / `s->out_convert` / `s->full_convert` (`:165-167`).
    in_convert: Option<AudioConvert>,
    out_convert: Option<AudioConvert>,
    full_convert: Option<AudioConvert>,
    /// `s->rematrix_ctx` — the [`RematrixContext`] built by `swri_rematrix_init`.
    rematrix_ctx: Option<RematrixContext>,
    /// `s->resample` (`:168`) — the swr engine backend.
    resample: Option<ResampleContext>,
}

impl SwrContext {
    // -- construction --------------------------------------------------------

    /// `swr_alloc` (`options.c:148-156`): a context with option defaults and
    /// `firstpts = AV_NOPTS_VALUE`. Everything else is the zeroed struct
    /// (`av_mallocz`).
    pub fn alloc() -> SwrContext {
        SwrContext {
            options: SwrOptions::default(),
            int_sample_fmt: SampleFormat::U8, // AV_SAMPLE_FMT_NONE analog; validated at init
            used_ch_layout: ChannelLayout::default(),
            in_ch_layout: ChannelLayout::default(),
            out_ch_layout: ChannelLayout::default(),
            dither_method: SwrDitherType::None,
            rematrix_needed: false,
            resample_first: false,
            channel_map: None,
            custom: None,
            in_: AudioData::empty(),
            postin: AudioData::empty(),
            midbuf: AudioData::empty(),
            preout: AudioData::empty(),
            out: AudioData::empty(),
            in_buffer: AudioData::empty(),
            silence: AudioData::empty(),
            drop_temp: AudioData::empty(),
            in_buffer_index: 0,
            in_buffer_count: 0,
            resample_in_constraint: false,
            flushed: false,
            outpts: 0,       // av_mallocz zero, like C
            firstpts: NOPTS, // swr_alloc's explicit set (options.c:154)
            drop_output: 0,
            in_convert: None,
            out_convert: None,
            full_convert: None,
            rematrix_ctx: None,
            resample: None,
        }
    }

    /// `swr_alloc_set_opts2` (`swresample.c:54-96`). An existing context is
    /// reused (`ps` in C); `log_offset`/`log_ctx` are dropped (no AVClass).
    /// The option writes are typed field stores; the only failure paths are
    /// the two `swri_check_chlayout`s, whose WARNING + `"Failed to set
    /// option"` ERROR both fire before the context is consumed (C frees
    /// `*ps`; the port returns the error and drops it).
    pub fn alloc_set_opts2(
        s: Option<SwrContext>,
        out_ch_layout: &ChannelLayout,
        out_sample_fmt: SampleFormat,
        out_sample_rate: i32,
        in_ch_layout: &ChannelLayout,
        in_sample_fmt: SampleFormat,
        in_sample_rate: i32,
    ) -> Result<SwrContext> {
        let mut s = s.unwrap_or_else(SwrContext::alloc);
        // :69-72 — ochl, then check.
        s.options.out_chlayout = *out_ch_layout;
        if let Err(e) = s.check_chlayout(out_ch_layout, "ochl") {
            log_error!(LOG_CTX, "Failed to set option");
            return Err(e);
        }
        // :74-78 — osf, osr.
        s.options.out_sample_fmt = Some(out_sample_fmt);
        s.options.out_sample_rate = out_sample_rate;
        // :80-83 — ichl, then check.
        s.options.in_chlayout = *in_ch_layout;
        if let Err(e) = s.check_chlayout(in_ch_layout, "ichl") {
            log_error!(LOG_CTX, "Failed to set option");
            return Err(e);
        }
        // :85-89 — isf, isr.
        s.options.in_sample_fmt = Some(in_sample_fmt);
        s.options.in_sample_rate = in_sample_rate;
        Ok(s)
    }

    /// `swri_check_chlayout` (`swresample.c:33-45`):
    /// `av_channel_layout_check` + the `SWR_CH_MAX` bound; on failure a
    /// WARNING naming the side and the described layout (empty string when
    /// the layout fails `check()` — C describes only when `ret` held), then
    /// `EINVAL`.
    pub(crate) fn check_chlayout(&self, chl: &ChannelLayout, name: &str) -> Result<()> {
        let ret = chl.check();
        if !ret || chl.nb_channels > SWR_CH_MAX {
            let described = if ret { chl.describe() } else { String::new() };
            log_warning!(
                LOG_CTX,
                "{name} channel layout \"{described}\" is invalid or unsupported."
            );
            return Err(Error::InvalidArgument(format!(
                "{name} channel layout \"{described}\" is invalid or unsupported."
            )));
        }
        Ok(())
    }

    /// `swr_set_channel_mapping` (`swresample.c:47-52`): only before init
    /// (C tests `s->in_convert`); the map is copied (C stores the pointer).
    /// Entries are input-channel indices, `-1` = muted (the converter
    /// substitutes its silence byte).
    pub fn set_channel_mapping(&mut self, channel_map: Option<&[i32]>) -> Result<()> {
        if self.in_convert.is_some() {
            return Err(Error::InvalidArgument(
                "swr_set_channel_mapping: the context must be allocated but not initialized".into(),
            ));
        }
        self.channel_map = channel_map.map(|m| m.to_vec());
        Ok(())
    }

    /// `swr_set_matrix` (`rematrix.c:71-91`) routed through the landed
    /// rematrix port: install a caller-provided mixing matrix (legal only
    /// before init). `matrix` is read C-style — row `out` starts at
    /// `out*stride` — and needs at least `(nb_out-1)*stride + nb_in`
    /// elements.
    pub fn set_matrix(&mut self, matrix: &[f64], stride: usize) -> Result<()> {
        let custom = rematrix::swr_set_matrix(
            &self.options.in_chlayout,
            &self.options.out_chlayout,
            matrix,
            stride,
            LOG_CTX,
            self.in_convert.is_some(),
        )?;
        self.custom = Some(custom);
        Ok(())
    }

    // -- teardown ------------------------------------------------------------

    /// `clear_context` (`swresample.c:111-135`).
    fn clear_context(&mut self) {
        self.in_buffer_index = 0;
        self.in_buffer_count = 0;
        self.resample_in_constraint = false;
        // free_temp(&postin/midbuf/preout/in_buffer/silence/drop_temp): each
        // becomes the fully-zeroed descriptor.
        self.postin = AudioData::empty();
        self.midbuf = AudioData::empty();
        self.preout = AudioData::empty();
        self.in_buffer = AudioData::empty();
        self.silence = AudioData::empty();
        self.drop_temp = AudioData::empty();
        // av_channel_layout_uninit of the three runtime layouts.
        self.in_ch_layout = ChannelLayout::default();
        self.out_ch_layout = ChannelLayout::default();
        self.used_ch_layout = ChannelLayout::default();
        self.in_convert = None;
        self.out_convert = None;
        self.full_convert = None;
        self.rematrix_ctx = None; // swri_rematrix_free (native_matrix)
        // NOT reset, faithfully: outpts, firstpts, drop_output, the resample
        // backend (swr_free-only — but Option handles its own Drop), the
        // user_* option fields, channel_map, custom (s->matrix +
        // rematrix_custom survive clear_context in C too).
        self.flushed = false;
    }

    /// `swr_close` (`swresample.c:152-154`). "Reset" = `close()` + set
    /// options + `init()`.
    pub fn close(&mut self) {
        self.clear_context();
    }

    /// `swr_is_initialized` (`swresample.c:739-741`) — exactly this
    /// predicate, not a flag: `in_buffer.ch_count` is set from `in.ch_count`
    /// at init and zeroed by close.
    pub fn is_initialized(&self) -> bool {
        self.in_buffer.ch_count != 0
    }

    fn in_rate(&self) -> i32 {
        self.options.in_sample_rate
    }
    fn out_rate(&self) -> i32 {
        self.options.out_sample_rate
    }
    fn in_fmt(&self) -> SampleFormat {
        self.options.in_sample_fmt.unwrap_or(SampleFormat::U8)
    }
    fn out_fmt(&self) -> SampleFormat {
        self.options.out_sample_fmt.unwrap_or(SampleFormat::U8)
    }

    // -- swr_init --------------------------------------------------------------

    /// `swr_init` (`swresample.c:156-424`) — the 42-step validation and
    /// derivation sequence, error texts verbatim (minus the trailing `\n`
    /// the log sink supplies). `goto fail` sites call [`Self::close`]
    /// before returning; direct `return` sites match C's direct returns
    /// (the context was cleared at the top either way).
    pub fn init(&mut self) -> Result<()> {
        // (1) clear_context.
        self.clear_context();

        // (2-3) :162-169 — format validation. C prints the raw enum int;
        // the unset default NONE = -1.
        if self.options.in_sample_fmt.is_none() {
            let msg = "Requested input sample format -1 is invalid";
            log_error!(LOG_CTX, "{msg}");
            return Err(Error::InvalidArgument(msg.into()));
        }
        if self.options.out_sample_fmt.is_none() {
            let msg = "Requested output sample format -1 is invalid";
            log_error!(LOG_CTX, "{msg}");
            return Err(Error::InvalidArgument(msg.into()));
        }
        // (4-5) :171-178 — rate validation.
        if self.in_rate() <= 0 {
            let msg = format!("Requested input sample rate {} is invalid", self.in_rate());
            log_error!(LOG_CTX, "{msg}");
            return Err(Error::InvalidArgument(msg));
        }
        if self.out_rate() <= 0 {
            let msg = format!(
                "Requested output sample rate {} is invalid",
                self.out_rate()
            );
            log_error!(LOG_CTX, "{msg}");
            return Err(Error::InvalidArgument(msg));
        }
        // (6) :180-186 — the DSD gate. Live in this port (`Dsd` exists in
        // SampleFormat): DSD passthrough at equal rates without forced
        // resampling is allowed; everything else is rejected.
        if self.out_fmt() == SampleFormat::Dsd
            && !(self.in_fmt() == SampleFormat::Dsd
                && self.in_rate() == self.out_rate()
                && !self.options.flags.contains(SwrFlags::RESAMPLE))
        {
            let msg = "Conversion to DSD is not supported";
            log_error!(LOG_CTX, "{msg}");
            return Err(Error::InvalidArgument(msg.into()));
        }

        // (7) :188-189.
        self.out.ch_count = self.options.out_chlayout.nb_channels;
        self.in_.ch_count = self.options.in_chlayout.nb_channels;

        // (8) :191-193 — input first, then output (direct EINVAL return).
        if let Err(e) = self.check_chlayout(&self.options.in_chlayout, "input") {
            return Err(e);
        }
        if let Err(e) = self.check_chlayout(&self.options.out_chlayout, "output") {
            return Err(e);
        }

        // (9) :195-199 — runtime layout copies (unrepresentable failure).
        self.in_ch_layout = self.options.in_chlayout;
        self.out_ch_layout = self.options.out_chlayout;
        self.used_ch_layout = self.options.used_chlayout;

        // (10) :201-203.
        self.int_sample_fmt = self.options.internal_sample_fmt.unwrap_or(SampleFormat::U8);
        self.dither_method = self.options.dither_method;

        // (11) :205-213 — the engine switch.
        match self.options.engine {
            SwrEngine::Swr => { /* swri_resampler — the only backend */ }
            SwrEngine::Soxr => {
                // A C build without CONFIG_LIBSOXR: the case is absent from
                // the switch and falls to the default error.
                let msg = "Requested resampling engine is unavailable";
                log_error!(LOG_CTX, "{msg}");
                return Err(Error::InvalidArgument(msg.into()));
            }
        }

        // (12) :215-216.
        if !self.used_ch_layout.check() {
            self.used_ch_layout = ChannelLayout::default_for(self.in_.ch_count);
        }
        // (13) :218-219.
        if self.used_ch_layout.nb_channels != self.in_ch_layout.nb_channels {
            self.in_ch_layout = ChannelLayout::default(); // av_channel_layout_uninit
        }
        // (14-16) :221-229.
        if self.used_ch_layout.order == Order::Unspecified {
            self.used_ch_layout = ChannelLayout::default_for(self.used_ch_layout.nb_channels);
        }
        if self.in_ch_layout.order == Order::Unspecified {
            self.in_ch_layout = self.used_ch_layout;
        }
        if self.out_ch_layout.order == Order::Unspecified {
            self.out_ch_layout = ChannelLayout::default_for(self.out.ch_count);
        }

        // (17) :231-233.
        self.rematrix_needed = self.out_ch_layout != self.in_ch_layout
            || self.options.rematrix_volume != 1.0
            || self.custom.is_some();

        // (18) :235-266 — the internal-format ladder (user set NONE).
        if self.options.internal_sample_fmt.is_none() {
            self.int_sample_fmt = if self.in_fmt() == SampleFormat::Dsd
                && self.out_fmt() != SampleFormat::Dsd
            {
                // DSD to PCM is done in floating point.
                SampleFormat::Fltp
            } else if self.in_fmt().bytes_per_sample() <= 2
                && self.out_fmt().bytes_per_sample() <= 2
                && self.out_rate() == self.in_rate()
            {
                // 16bit or less to 16bit or less with the same sample rate.
                SampleFormat::S16p
            } else if self.in_fmt().bytes_per_sample() + self.out_fmt().bytes_per_sample() <= 3 {
                // 8 -> 8, 16->8, 8->16bit.
                SampleFormat::S16p
            } else if self.in_fmt().bytes_per_sample() <= 2
                && !self.rematrix_needed
                && self.out_rate() == self.in_rate()
                && !self.options.flags.contains(SwrFlags::RESAMPLE)
            {
                SampleFormat::S16p
            } else if self.in_fmt().planar() == SampleFormat::S32p
                && self.out_fmt().planar() == SampleFormat::S32p
                && !self.rematrix_needed
                && self.out_rate() == self.in_rate()
                && !self.options.flags.contains(SwrFlags::RESAMPLE)
                && self.options.engine != SwrEngine::Soxr
            {
                SampleFormat::S32p
            } else if self.in_fmt().bytes_per_sample() <= 4 {
                SampleFormat::Fltp
            } else {
                SampleFormat::Dblp
            };
        }
        // (19) :267.
        log_debug!(
            LOG_CTX,
            "Using {} internally between filters",
            self.int_sample_fmt.name()
        );

        // (20) :269-276 — the internal whitelist (s64p kept: C accepts it
        // here and the backends reject it individually, exactly as the C
        // rematrix assert / resample fmt check do).
        if !matches!(
            self.int_sample_fmt,
            SampleFormat::S16p
                | SampleFormat::S32p
                | SampleFormat::S64p
                | SampleFormat::Fltp
                | SampleFormat::Dblp
        ) {
            let msg = format!(
                "Requested sample format {} is not supported internally, \
                 s16p/s32p/s64p/fltp/dblp are supported",
                self.int_sample_fmt.name()
            );
            log_error!(LOG_CTX, "{msg}");
            return Err(Error::InvalidArgument(msg));
        }

        // (21) :278-279.
        let in_fmt = self.in_fmt();
        let out_fmt = self.out_fmt();
        set_audiodata_fmt(&mut self.in_, in_fmt);
        set_audiodata_fmt(&mut self.out, out_fmt);

        // (22) :281-296 — firstpts/async normalization (C mutates the
        // option fields in place; re-init sees the mutated values).
        if self.options.first_pts != NOPTS {
            if self.options.async_ == 0.0 && self.options.min_compensation >= f32::MAX / 2.0 {
                self.options.async_ = 1.0;
            }
            if self.firstpts == NOPTS {
                self.firstpts = self.options.first_pts.wrapping_mul(self.out_rate() as i64);
                self.outpts = self.firstpts;
            }
        } else {
            self.firstpts = NOPTS;
        }
        if self.options.async_ != 0.0 {
            if self.options.min_compensation >= f32::MAX / 2.0 {
                self.options.min_compensation = 0.001;
            }
            if self.options.async_ > 1.0001 {
                // C: s->async / (double)in_rate, stored to the float field —
                // double division then narrowing.
                self.options.max_soft_compensation =
                    (self.options.async_ as f64 / self.in_rate() as f64) as f32;
            }
        }

        // (23) :298-305 — resampler init/free.
        if self.out_rate() != self.in_rate() || self.options.flags.contains(SwrFlags::RESAMPLE) {
            self.resample = match ResampleContext::new(
                self.out_rate(),
                self.in_rate(),
                self.options.filter_size,
                self.options.phase_shift,
                self.options.linear_interp as i32,
                self.options.cutoff,
                self.int_sample_fmt,
                self.options.filter_type,
                self.options.kaiser_beta,
                self.options.precision,
                self.options.cheby as i32,
                self.options.exact_rational as i32,
            ) {
                Ok(c) => Some(c),
                Err(_) => {
                    // C: resampler->init returned NULL -> AVERROR(ENOMEM).
                    let msg = "Failed to initialize resampler";
                    log_error!(LOG_CTX, "{msg}");
                    return Err(Error::InvalidArgument(msg.into()));
                }
            };
        } else {
            self.resample = None;
        }
        // (24) :306-314 — goto fail.
        if self.resample.is_some()
            && !matches!(
                self.int_sample_fmt,
                SampleFormat::S16p | SampleFormat::S32p | SampleFormat::Fltp | SampleFormat::Dblp
            )
        {
            let msg = "Resampling only supported with internal s16p/s32p/fltp/dblp";
            log_error!(LOG_CTX, "{msg}");
            self.close();
            return Err(Error::InvalidArgument(msg.into()));
        }

        // (25) :317-322.
        if self.in_.ch_count == 0 {
            self.in_.ch_count = self.in_ch_layout.nb_channels;
        }
        if !self.used_ch_layout.check() {
            self.used_ch_layout = ChannelLayout::default_for(self.in_.ch_count);
        }
        if self.out.ch_count == 0 {
            self.out.ch_count = self.out_ch_layout.nb_channels;
        }
        // (26) :324-329 — defensive in C too (the :191 check already rejects
        // a 0-channel user layout; kept for parity). goto fail.
        if self.in_.ch_count == 0 {
            debug_assert_eq!(self.in_ch_layout.order, Order::Unspecified);
            let msg = "Input channel count and layout are unset";
            log_error!(LOG_CTX, "{msg}");
            self.close();
            return Err(Error::InvalidArgument(msg.into()));
        }

        // (27) :331-332.
        let l2 = self.out_ch_layout.describe();
        let l1 = self.in_ch_layout.describe();
        // (28) :333-337 — defensive in C (the :218-227 normalization keeps
        // in.nb == used.nb); goto fail on mismatch.
        if self.in_ch_layout.order != Order::Unspecified
            && self.used_ch_layout.nb_channels != self.in_ch_layout.nb_channels
        {
            let msg = format!(
                "Input channel layout {} mismatches specified channel count {}",
                l1, self.used_ch_layout.nb_channels
            );
            log_error!(LOG_CTX, "{msg}");
            self.close();
            return Err(Error::InvalidArgument(msg));
        }
        // (29) :339-345 — goto fail.
        if (self.out_ch_layout.order == Order::Unspecified
            || self.in_ch_layout.order == Order::Unspecified)
            && self.used_ch_layout.nb_channels != self.out.ch_count
            && self.custom.is_none()
        {
            let msg = format!(
                "Rematrix is needed between {} and {} but there is not enough information to do it",
                l1, l2
            );
            log_error!(LOG_CTX, "{msg}");
            self.close();
            return Err(Error::InvalidArgument(msg));
        }

        // (30) :347-348.
        debug_assert!(self.used_ch_layout.nb_channels != 0);
        debug_assert!(self.out.ch_count != 0);

        // (31) :349 — RSC = 1; INTEGER division on the left (x86-64 int);
        // the right side divides in float, subtracts the double 1.0, and
        // the comparison promotes the int directly to double.
        let left =
            (self.out.ch_count as i32 / self.used_ch_layout.nb_channels as i32 - 1) as f64;
        let right = (self.out_rate() as f32 / self.in_rate() as f32) as f64 - 1.0;
        self.resample_first = left < right;

        // (32) :351-353 — descriptor clones (count 0, no data).
        self.in_buffer = self.in_.clone();
        self.silence = self.in_.clone();
        self.drop_temp = self.out.clone();

        // (33) :355-356 — swri_dither_init (the scale computation of
        // dither.c:85-103; everything past it needs dither.c). goto fail.
        if let Err(e) = self.dither_init() {
            self.close();
            return Err(e);
        }

        // (34) :358-365 — the single-conversion fast path.
        if self.resample.is_none()
            && !self.rematrix_needed
            && self.channel_map.is_none()
            && self.dither_method == SwrDitherType::None
        {
            if let Ok(fc) =
                AudioConvert::new(self.out_fmt(), self.in_fmt(), self.in_.ch_count, None)
            {
                self.full_convert = Some(fc);
                return Ok(());
            }
            // else: fall through to the generic path for conversions with no
            // direct implementation (e.g. DSD input to non-float output).
        }

        // (35) :367-378 — goto fail on either converter.
        self.in_convert = AudioConvert::new(
            self.int_sample_fmt,
            self.in_fmt(),
            self.used_ch_layout.nb_channels,
            self.channel_map.as_deref(),
        )
        .ok();
        self.out_convert =
            AudioConvert::new(self.out_fmt(), self.int_sample_fmt, self.out.ch_count, None).ok();
        if self.in_convert.is_none() || self.out_convert.is_none() {
            let msg = format!(
                "Cannot convert {} sample format to {} sample format",
                if self.in_convert.is_none() {
                    self.in_fmt().name()
                } else {
                    self.int_sample_fmt.name()
                },
                if self.in_convert.is_none() {
                    self.int_sample_fmt.name()
                } else {
                    self.out_fmt().name()
                }
            );
            log_error!(LOG_CTX, "{msg}");
            self.close();
            return Err(Error::InvalidArgument(msg));
        }

        // (36) :380-382.
        self.postin = self.in_.clone();
        self.preout = self.out.clone();
        self.midbuf = self.in_.clone();

        // (37-39) :384-402 — channel-count adjustments + int formats.
        if self.channel_map.is_some() {
            self.postin.ch_count = self.used_ch_layout.nb_channels;
            self.midbuf.ch_count = self.used_ch_layout.nb_channels;
            if self.resample.is_some() {
                self.in_buffer.ch_count = self.used_ch_layout.nb_channels;
            }
        }
        if !self.resample_first {
            self.midbuf.ch_count = self.out.ch_count;
            if self.resample.is_some() {
                self.in_buffer.ch_count = self.out.ch_count;
            }
        }
        set_audiodata_fmt(&mut self.postin, self.int_sample_fmt);
        set_audiodata_fmt(&mut self.midbuf, self.int_sample_fmt);
        set_audiodata_fmt(&mut self.preout, self.int_sample_fmt);
        if self.resample.is_some() {
            set_audiodata_fmt(&mut self.in_buffer, self.int_sample_fmt);
        }

        // (40) :404-411 — av_assert0(!preout.count) + the dither noise/temp
        // clones: dropped with dither.c.
        debug_assert_eq!(self.preout.count, 0);

        // (41) :413-417 — swri_rematrix_init. goto fail.
        if self.rematrix_needed || self.dither_method != SwrDitherType::None {
            let rm_opts = RematrixOptions {
                clev: self.options.clev as f64,
                slev: self.options.slev as f64,
                lfe_mix_level: self.options.lfe_mix_level as f64,
                rematrix_volume: self.options.rematrix_volume as f64,
                rematrix_maxval: self.options.rematrix_maxval as f64,
                matrix_encoding: self.options.matrix_encoding,
            };
            match RematrixContext::init(
                &self.in_ch_layout,
                self.used_ch_layout.nb_channels,
                &self.out_ch_layout,
                self.out.ch_count,
                self.out_fmt(),
                self.int_sample_fmt,
                &rm_opts,
                self.custom.as_ref(),
                LOG_CTX,
            ) {
                Ok(rm) => self.rematrix_ctx = Some(rm),
                Err(e) => {
                    self.close();
                    return Err(e);
                }
            }
        }

        // (42).
        Ok(())
    }

    /// `swri_dither_init` (`dither.c:83-103`) — only the `scale` computation
    /// and the `scale == 0` disabling; everything past `:106` (noise
    /// buffers, noise-shaping filters) needs the unported `dither.c`.
    ///
    /// The C `method > SWR_DITHER_TRIANGULAR_HIGHPASS && method <= NS`
    /// `EINVAL` (`:85-86`) is unrepresentable — the NS family is not in
    /// [`SwrDitherType`].
    fn dither_init(&mut self) -> Result<()> {
        let out_fmt = self.out.fmt.packed();
        let in_fmt = self.in_.fmt.packed();
        let mut scale = 0.0f64;
        // :87-92.
        if in_fmt == SampleFormat::Flt || in_fmt == SampleFormat::Dbl {
            if out_fmt == SampleFormat::S32 {
                scale = 1.0 / (1u64 << 31) as f64;
            }
            if out_fmt == SampleFormat::S16 {
                scale = 1.0 / (1i64 << 15) as f64;
            }
            if out_fmt == SampleFormat::U8 {
                scale = 1.0 / (1i64 << 7) as f64;
            }
        }
        // :93-97.
        if in_fmt == SampleFormat::S32
            && out_fmt == SampleFormat::S32
            && (self.options.output_sample_bits & 31) != 0
        {
            scale = 1.0;
        }
        if in_fmt == SampleFormat::S32 && out_fmt == SampleFormat::S16 {
            scale = (1i64 << 16) as f64;
        }
        if in_fmt == SampleFormat::S32 && out_fmt == SampleFormat::U8 {
            scale = (1i64 << 24) as f64;
        }
        if in_fmt == SampleFormat::S16 && out_fmt == SampleFormat::U8 {
            scale = (1i64 << 8) as f64;
        }
        // :99 — dither.scale is the float option.
        scale *= self.options.dither_scale as f64;
        // :101-102 — `1<<(32-bits)` is C UB for bits > 32 (the option range
        // is 0..64, options.c:125); x86 masks the shift count, replicated.
        if out_fmt == SampleFormat::S32 && self.options.output_sample_bits != 0 {
            let sh = ((32 - self.options.output_sample_bits) as u32) & 31;
            scale *= (1u32 << sh) as f64;
        }
        // :100-103 — no precision loss: dithering disabled, C-identical.
        // A method of NONE stays inert in C regardless of `scale`
        // (swr_convert_internal gates on `if (s->dither.method)`), so only
        // a real method past dither.c:106 is unportable.
        if scale == 0.0 || self.dither_method == SwrDitherType::None {
            self.dither_method = SwrDitherType::None;
            return Ok(());
        }
        // Past dither.c:106 the method needs the unported noise machinery.
        Err(Error::Unsupported(format!(
            "dither method {} requires dither.c (not ported)",
            self.dither_method as i32
        )))
    }

    // -- swr_convert ------------------------------------------------------------

    /// `swr_convert` (`swresample.c:743-856`) — the public entry.
    ///
    /// * `in_ = None` is C's NULL `in_arg` = **flush** (drain); `Some` with
    ///   `in_count = 0` is a non-flush empty call (`swr_drop_output` relies
    ///   on the distinction).
    /// * Interleaved callers pass one plane; planar one per channel.
    /// * Returns the number of samples written per channel.
    pub fn convert(
        &mut self,
        out: Option<&mut [&mut [u8]]>,
        out_count: usize,
        in_: Option<&[&[u8]]>,
        in_count: usize,
    ) -> Result<usize> {
        // (A) :750-753.
        if !self.is_initialized() {
            log_error!(LOG_CTX, "Context has not been initialized");
            return Err(Error::InvalidArgument(
                "Context has not been initialized".into(),
            ));
        }
        // The ASSERT_LEVEL>1 max_output cross-check (:754-756, 800, 853) is
        // dropped — the get_out_samples bound is pinned by a unit test.

        // (B) :758-779 — the drop loop.
        let mut in_count = in_count;
        while self.drop_output > 0 {
            // :762 — swri_realloc_audio(&s->drop_temp, FFMIN(drop, STEP)).
            let step = (self.drop_output as usize).min(MAX_DROP_STEP as usize);
            // reversefill_audiodata: drop_temp's planes are the out target.
            let mut drop_temp = std::mem::replace(&mut self.drop_temp, AudioData::empty());
            let res = drop_temp.realloc_audio(step).and_then(|_| {
                // :766-768 — the sentinel flip makes the recursive call skip
                // this loop and keeps outpts from advancing.
                self.drop_output = -self.drop_output;
                let r = self.convert_body(&mut drop_temp, step, in_, in_count);
                self.drop_output = -self.drop_output;
                r
            });
            self.drop_temp = drop_temp;
            // C swallows the inner failure behind av_assert0; the port
            // propagates it (documented divergence).
            let ret = res?;
            in_count = 0; // :769.
            if ret > 0 {
                self.drop_output -= ret as i32;
                if self.drop_output == 0 && out.is_none() {
                    return Ok(0); // :772-773.
                }
                continue;
            }
            // :777-778 — ret == 0: assert drop_output, return 0.
            debug_assert!(self.drop_output != 0);
            return Ok(0);
        }

        // (D) — out staging (fill_audiodata + reversefill). Reaching the body
        // with no out planes and out_count > 0 is C UB (NULL ch[] writes);
        // the debug assert catches it, release clamps.
        let out_count = if out.is_none() {
            debug_assert_eq!(out_count, 0, "swr_convert: NULL out with out_count > 0");
            0
        } else {
            out_count
        };

        match out {
            Some(planes) => {
                let mut out_ad = scratch_like(&self.out, out_count);
                let ret = self.convert_body(&mut out_ad, out_count, in_, in_count)?;
                self.copy_out_planes(planes, &out_ad, ret)?;
                Ok(ret)
            }
            None => {
                let mut out_ad = scratch_like(&self.out, 0);
                self.convert_body(&mut out_ad, 0, in_, in_count)
            }
        }
    }

    /// The body of `swr_convert` past the drop loop, operating on an owned
    /// out staging buffer (C's `out` AudioData with caller planes).
    fn convert_body(
        &mut self,
        out_ad: &mut AudioData,
        out_count: usize,
        in_: Option<&[&[u8]]>,
        in_count: usize,
    ) -> Result<usize> {
        // (C) :781-792 — flush/empty-input handling + input staging.
        let in_ad: AudioData;
        match in_ {
            Some(planes) => {
                in_ad = self.take_input(planes, in_count)?;
            }
            None => {
                if self.resample.is_some() {
                    if !self.flushed {
                        // s->resampler->flush(s) — resample.c:437-454.
                        let resample = self.resample.as_ref().expect("checked above");
                        resample.resample_flush(
                            &mut self.in_buffer,
                            self.in_buffer_index,
                            &mut self.in_buffer_count,
                        )?;
                    }
                    self.resample_in_constraint = false;
                    self.flushed = true;
                } else if self.in_buffer_count == 0 {
                    return Ok(0); // :787-789.
                }
                in_ad = scratch_like(&self.in_, 0); // empty descriptor
            }
        }

        // (E) :795-802 — the resample path.
        if self.resample.is_some() {
            let ret = self.convert_internal(out_ad, out_count, &in_ad, in_count)?;
            if ret > 0 && self.drop_output == 0 {
                // outpts advances in IN-rate units (swr_next_pts rescales).
                self.outpts = self.outpts.wrapping_add(ret as i64 * self.in_rate() as i64);
            }
            return Ok(ret);
        }

        // (F) :803-855 — the equal-rate FIFO path.
        let mut ret2 = 0usize;
        let mut out_off = 0usize;
        let mut out_count = out_count;
        let mut in_off = 0usize;
        let mut in_count = in_count;

        // :807-820 — drain the buffer first.
        let size = out_count.min(self.in_buffer_count);
        if size > 0 {
            let tmp = view(&self.in_buffer, self.in_buffer_index);
            let ret = self.convert_internal(out_ad, size, &tmp, size)?;
            ret2 = ret;
            self.in_buffer_count -= ret;
            self.in_buffer_index += ret;
            out_off += ret;
            out_count -= ret;
            if self.in_buffer_count == 0 {
                self.in_buffer_index = 0;
            }
        }

        if in_count > 0 {
            // :823 — C computes `size` as a (possibly negative) int here; it
            // is only read inside the in_count > out_count branch, where it
            // is guaranteed non-negative — computed there for usize.
            // :825-834.
            if in_count > out_count {
                let size = self.in_buffer_index + self.in_buffer_count + in_count - out_count;
                if size > self.in_buffer.count
                    && self.in_buffer_count + in_count - out_count <= self.in_buffer_index
                {
                    compact(
                        &mut self.in_buffer,
                        self.in_buffer_index,
                        self.in_buffer_count,
                    );
                    self.in_buffer_index = 0;
                } else {
                    self.in_buffer.realloc_audio(size)?;
                }
            }
            // :836-844 — convert directly from the caller input when space
            // remains.
            if out_count > 0 {
                let size2 = in_count.min(out_count);
                let inv = view(&in_ad, in_off);
                // The second call writes at out_off: stage a scratch and
                // splice (C advances its out ch[] pointers instead).
                let mut o2 = scratch_like(&self.out, size2);
                let ret = self.convert_internal(&mut o2, size2, &inv, size2)?;
                splice(out_ad, out_off, &o2, ret);
                in_off += ret;
                in_count -= ret;
                ret2 += ret;
            }
            // :845-849 — buffer the leftovers.
            if in_count > 0 {
                let inv = view(&in_ad, in_off);
                splice(
                    &mut self.in_buffer,
                    self.in_buffer_index + self.in_buffer_count,
                    &inv,
                    in_count,
                );
                self.in_buffer_count += in_count;
            }
        }
        if ret2 > 0 && self.drop_output == 0 {
            self.outpts = self
                .outpts
                .wrapping_add(ret2 as i64 * self.in_rate() as i64);
        }
        Ok(ret2)
    }

    /// `fill_audiodata` (`swresample.c:471-483`) into an owned staging
    /// buffer: planar — one plane copy per channel; packed — the single
    /// interleaved plane. Two zero samples of tail padding keep the
    /// resampler's documented one-past-`src_size` read deterministic where
    /// C would read caller heap slack (module divergences).
    fn take_input(&self, planes: &[&[u8]], count: usize) -> Result<AudioData> {
        let bps = self.in_.bps;
        let ch = self.in_.ch_count;
        if self.in_.planar {
            let cap = count + 2;
            let mut ad = AudioData::new(self.in_.fmt, ch, cap);
            ad.planar = true;
            let need = count * bps;
            for i in 0..ch {
                let src = planes.get(i).ok_or(Error::BufferTooSmall)?;
                if src.len() < need {
                    return Err(Error::BufferTooSmall);
                }
                let to = i * cap * bps;
                ad.data_mut()[to..to + need].copy_from_slice(&src[..need]);
            }
            Ok(ad)
        } else {
            let need = count * ch * bps;
            let src = planes.first().ok_or(Error::BufferTooSmall)?;
            if src.len() < need {
                return Err(Error::BufferTooSmall);
            }
            Ok(AudioData {
                data: src[..need].to_vec().into(),
                ch_count: ch,
                bps,
                count,
                planar: false,
                fmt: self.in_.fmt,
            })
        }
    }

    /// `reversefill_audiodata`'s copy-back: the out staging to the caller's
    /// planes, `ret` samples per channel.
    fn copy_out_planes(
        &self,
        planes: &mut [&mut [u8]],
        out_ad: &AudioData,
        ret: usize,
    ) -> Result<()> {
        let bps = self.out.bps;
        if self.out.planar {
            let need = ret * bps;
            for i in 0..self.out.ch_count {
                let src = out_ad.plane(i).ok_or(Error::BufferTooSmall)?;
                let dst = planes.get_mut(i).ok_or(Error::BufferTooSmall)?;
                if dst.len() < need || src.len() < need {
                    return Err(Error::BufferTooSmall);
                }
                dst[..need].copy_from_slice(&src[..need]);
            }
        } else {
            let need = ret * self.out.ch_count * bps;
            let src = out_ad.plane(0).ok_or(Error::BufferTooSmall)?;
            let dst = planes.first_mut().ok_or(Error::BufferTooSmall)?;
            if dst.len() < need || src.len() < need {
                return Err(Error::BufferTooSmall);
            }
            dst[..need].copy_from_slice(&src[..need]);
        }
        Ok(())
    }

    // -- swr_convert_internal (:609-737) ----------------------------------------

    /// The stage pipeline: buffering → in_convert → rematrix/resample (per
    /// `resample_first`) → out_convert, with C's pointer-alias routing
    /// (`:638-665`) walked as a [`Tgt`] tree. The three stage buffers are
    /// taken out of `self` for the duration so the backends can borrow them
    /// disjointly; every path restores them.
    fn convert_internal(
        &mut self,
        out: &mut AudioData,
        out_count: usize,
        in_: &AudioData,
        in_count: usize,
    ) -> Result<usize> {
        // :615-619 — the single-conversion fast path.
        if let Some(fc) = &self.full_convert {
            debug_assert!(self.resample.is_none());
            swri_audio_convert(fc, out, in_, in_count)?;
            return Ok(out_count);
        }

        let mut postin = std::mem::replace(&mut self.postin, AudioData::empty());
        let mut midbuf = std::mem::replace(&mut self.midbuf, AudioData::empty());
        let mut preout = std::mem::replace(&mut self.preout, AudioData::empty());
        let res = self.convert_stages(
            &mut postin,
            &mut midbuf,
            &mut preout,
            out,
            out_count,
            in_,
            in_count,
        );
        self.postin = postin;
        self.midbuf = midbuf;
        self.preout = preout;
        res
    }

    /// The body of [`Self::convert_internal`] past the `full_convert` check,
    /// with the stage buffers as locals.
    fn convert_stages(
        &mut self,
        postin: &mut AudioData,
        midbuf: &mut AudioData,
        preout: &mut AudioData,
        out: &mut AudioData,
        mut out_count: usize,
        in_: &AudioData,
        in_count: usize,
    ) -> Result<usize> {
        // :624-636 — the capacity reallocs, before any routing (they size
        // preout for the stages; C runs them even when the buffer is later
        // unused — order kept).
        postin.realloc_audio(in_count)?;
        if self.resample_first {
            debug_assert_eq!(midbuf.ch_count, self.used_ch_layout.nb_channels);
            midbuf.realloc_audio(out_count)?;
        } else {
            debug_assert_eq!(midbuf.ch_count, self.out.ch_count);
            midbuf.realloc_audio(in_count)?;
        }
        preout.realloc_audio(out_count)?;

        // :645-646 — a1: postin aliases the caller input.
        let a1 =
            self.int_sample_fmt == self.in_.fmt && self.in_.planar && self.channel_map.is_none();
        // :648-649 — a2: midbuf aliases postin.
        let a2 = if self.resample_first {
            self.resample.is_none()
        } else {
            !self.rematrix_needed
        };
        // :651-652 — a3: preout aliases midbuf.
        let a3 = if self.resample_first {
            !self.rematrix_needed
        } else {
            self.resample.is_none()
        };
        // :654-655 — b: preout may alias the caller output (the dither
        // S32P carve-out reads the output_sample_bits option directly —
        // options.c:125 stores it in DitherContext).
        let b = self.int_sample_fmt == self.out.fmt
            && self.out.planar
            && !(self.out.fmt == SampleFormat::S32p && (self.options.output_sample_bits & 31) != 0);

        let pt0 = if a1 { Tgt::CallerIn } else { Tgt::Postin };
        let mt0 = if a2 { pt0 } else { Tgt::Midbuf };

        let (postin_t, midbuf_t, preout_t) = if b {
            if a1 && a2 && a3 {
                // :656-661 — the whole chain aliases the input; pure copy.
                let n = out_count.min(in_count);
                debug_assert!(in_.planar, "swresample.c:658");
                copy_views(out, in_, n);
                return Ok(n);
            } else if a2 && a3 {
                (Tgt::Out, Tgt::Out, Tgt::Out) // :662 — preout==postin
            } else if a3 {
                (pt0, Tgt::Out, Tgt::Out) // :663 — preout==midbuf
            } else {
                (pt0, mt0, Tgt::Out) // :664 — preout=out
            }
        } else {
            (pt0, mt0, if a3 { mt0 } else { Tgt::Preout })
        };

        // :667-669 — the input conversion.
        if postin_t != Tgt::CallerIn {
            let ctx = self.in_convert.as_ref().expect("swr_init set in_convert");
            match postin_t {
                Tgt::Postin => swri_audio_convert(ctx, postin, in_, in_count)?,
                Tgt::Out => swri_audio_convert(ctx, out, in_, in_count)?,
                _ => unreachable!("postin routes to CallerIn/Postin/Out only"),
            }
        }

        // :671-683 — the two middle stages in C's order.
        if self.resample_first {
            if midbuf_t != postin_t {
                let src = src_of(postin_t, in_, postin, midbuf);
                match midbuf_t {
                    Tgt::Midbuf => {
                        out_count = self.resample_stage(midbuf, out_count, &src, in_count)?
                    }
                    Tgt::Out => out_count = self.resample_stage(out, out_count, &src, in_count)?,
                    _ => unreachable!("resample writes Midbuf or Out"),
                }
            }
            if midbuf_t != preout_t {
                let rm = self.rematrix_ctx.as_ref().expect("swr_init built it");
                let src = src_of(midbuf_t, in_, postin, midbuf);
                let mustcopy = preout_t == Tgt::Out;
                match preout_t {
                    Tgt::Preout => rm.rematrix(preout, &src, out_count, mustcopy)?,
                    Tgt::Out => rm.rematrix(out, &src, out_count, mustcopy)?,
                    _ => unreachable!("rematrix writes Preout or Out"),
                }
            }
        } else {
            if postin_t != midbuf_t {
                let rm = self.rematrix_ctx.as_ref().expect("swr_init built it");
                let src = src_of(postin_t, in_, postin, midbuf);
                let mustcopy = midbuf_t == Tgt::Out;
                match midbuf_t {
                    Tgt::Midbuf => rm.rematrix(midbuf, &src, in_count, mustcopy)?,
                    Tgt::Out => rm.rematrix(out, &src, in_count, mustcopy)?,
                    _ => unreachable!("rematrix writes Midbuf or Out"),
                }
            }
            if midbuf_t != preout_t {
                let src = src_of(midbuf_t, in_, postin, midbuf);
                match preout_t {
                    Tgt::Preout => {
                        out_count = self.resample_stage(preout, out_count, &src, in_count)?
                    }
                    Tgt::Out => out_count = self.resample_stage(out, out_count, &src, in_count)?,
                    _ => unreachable!("resample writes Preout or Out"),
                }
            }
        }

        // :685-735 — the output conversion (the dither block :687-732 is
        // dropped: dither_method is always None after init, so
        // conv_src == preout).
        if preout_t != Tgt::Out && out_count > 0 {
            let ctx = self.out_convert.as_ref().expect("swr_init set out_convert");
            let src = match preout_t {
                Tgt::CallerIn => in_.clone(),
                Tgt::Postin => postin.clone(),
                Tgt::Midbuf => midbuf.clone(),
                Tgt::Preout => preout.clone(),
                Tgt::Out => unreachable!("guarded above"),
            };
            swri_audio_convert(ctx, out, &src, out_count)?;
        }
        Ok(out_count)
    }

    // -- resample() (:514-607) ---------------------------------------------------

    /// The driver-side resample loop: invert-priming, the buffered/direct
    /// two-site `multiple_resample` dance with the `padless` x86 window,
    /// the in-buffer compaction/realloc, and the copy-avoidance staging.
    /// Writes `ret_sum` samples into `dst[0..ret_sum)`; returns the count.
    fn resample_stage(
        &mut self,
        dst: &mut AudioData,
        mut out_count: usize,
        src: &AudioData,
        in_count0: usize,
    ) -> Result<usize> {
        let mut ret_sum = 0usize;
        let mut out_written = 0usize;
        let mut in_off: isize = 0;
        let mut in_count = in_count0;
        // :519 — ARCH_X86 && engine == SWR: 7 on the x86-64 reference
        // platform (chunking-only difference from padless=0 builds; byte
        // stream identical). Applies to the FIRST pass only (:596-599).
        let mut padless: usize = 7;

        // :521-523.
        debug_assert_eq!(self.in_buffer.ch_count, src.ch_count);
        debug_assert_eq!(self.in_buffer.planar, src.planar);
        debug_assert_eq!(self.in_buffer.fmt, src.fmt);

        // PHASE A (:528-538) — the startup priming.
        let border = self
            .resample
            .as_mut()
            .expect("resample_stage called with a resampler")
            .invert_initial_buffer(
                &mut self.in_buffer,
                src,
                in_count,
                &mut self.in_buffer_index,
                &mut self.in_buffer_count,
            )?;
        let mut border: usize = match border {
            i32::MAX => return Ok(0), // :530-531 — wait for more input.
            b if b > 0 => {
                in_off += b as isize;
                in_count -= b as usize;
                self.resample_in_constraint = false;
                b as usize
            }
            _ => 0,
        };

        // The do-while (:540-602).
        loop {
            // (1) :542-560 — resample from the buffer.
            if !self.resample_in_constraint && self.in_buffer_count > 0 {
                let tmp = view(&self.in_buffer, self.in_buffer_index);
                let mut consumed: i32 = 0;
                let mut mdst = scratch_like(dst, out_count);
                let ret = self.resample.as_mut().expect("checked").multiple_resample(
                    &mut mdst,
                    out_count as i32,
                    &tmp,
                    self.in_buffer_count as i32,
                    &mut consumed,
                )? as usize;
                splice(dst, out_written, &mdst, ret);
                out_count -= ret;
                ret_sum += ret;
                out_written += ret;
                self.in_buffer_count -= consumed as usize;
                self.in_buffer_index += consumed as usize;

                if in_count == 0 {
                    break; // :551-552.
                }
                if self.in_buffer_count <= border {
                    // :553-559 — un-consume the buffered tail back into the
                    // direct-input stream (rewind; count may go negative in
                    // C's buf_set).
                    in_off -= self.in_buffer_count as isize;
                    in_count += self.in_buffer_count;
                    self.in_buffer_count = 0;
                    self.in_buffer_index = 0;
                    border = 0;
                }
            }

            // (2) :562-570 — resample directly from the input.
            if (self.flushed || in_count > padless) && self.in_buffer_count == 0 {
                self.in_buffer_index = 0;
                let direct = in_count.saturating_sub(padless); // FFMAX(in_count-padless, 0)
                let src_v = view(src, in_off.max(0) as usize);
                let mut consumed: i32 = 0;
                let mut mdst = scratch_like(dst, out_count);
                let ret = self.resample.as_mut().expect("checked").multiple_resample(
                    &mut mdst,
                    out_count as i32,
                    &src_v,
                    direct as i32,
                    &mut consumed,
                )? as usize;
                splice(dst, out_written, &mdst, ret);
                out_count -= ret;
                ret_sum += ret;
                out_written += ret;
                in_count -= consumed as usize;
                in_off += consumed as isize;
            }

            // (3) :572-581 — compaction or growth.
            let size = self.in_buffer_index + self.in_buffer_count + in_count;
            if size > self.in_buffer.count
                && self.in_buffer_count + in_count <= self.in_buffer_index
            {
                compact(
                    &mut self.in_buffer,
                    self.in_buffer_index,
                    self.in_buffer_count,
                );
                self.in_buffer_index = 0;
            } else {
                self.in_buffer.realloc_audio(size)?;
            }

            // (4) :583-600 — stage the input into the buffer.
            if in_count > 0 {
                let mut count = in_count;
                if self.in_buffer_count > 0 && self.in_buffer_count + 2 < count && out_count > 0 {
                    count = self.in_buffer_count + 2;
                }
                let src_v = view(src, in_off.max(0) as usize);
                splice(
                    &mut self.in_buffer,
                    self.in_buffer_index + self.in_buffer_count,
                    &src_v,
                    count,
                );
                self.in_buffer_count += count;
                in_count -= count;
                border += count;
                in_off += count as isize;
                self.resample_in_constraint = false;
                if self.in_buffer_count != count || in_count > 0 {
                    continue; // :594-595.
                }
                if padless != 0 {
                    padless = 0; // :596-599 — one-shot.
                    continue;
                }
            }
            break; // :601.
        }

        // :604.
        self.resample_in_constraint = out_count != 0;
        Ok(ret_sum)
    }

    // -- the small public queries --------------------------------------------------

    /// `swr_get_delay` (`swresample.c:901-907`) — samples "inside" the
    /// context, expressed in `base` units. The resampled formula
    /// (resample.c:408-416) is delegated to the backend; the FIFO formula
    /// is integer `(count*base + in_rate/2) / in_rate`.
    pub fn get_delay(&self, base: i64) -> i64 {
        if let Some(r) = &self.resample {
            r.get_delay(base, self.in_buffer_count as i64, self.in_rate())
        } else {
            (self.in_buffer_count as i64 * base + (self.in_rate() as i64 >> 1))
                / self.in_rate() as i64
        }
    }

    /// `swr_get_out_samples` (`swresample.c:909-929`) — an upper bound on
    /// the output obtainable from `in_samples` more input. Contract (C's
    /// ASSERT_LEVEL>1 invariant, pinned by test): calling `convert` with
    /// `out_count = get_out_samples(n)` never buffers for lack of space.
    pub fn get_out_samples(&self, in_samples: i32) -> Result<usize> {
        if in_samples < 0 {
            return Err(Error::InvalidArgument(
                "swr_get_out_samples: negative in_samples".into(),
            ));
        }
        let out_samples = if let Some(r) = &self.resample {
            // The !resampler->get_out_samples ENOSYS arm (:917-918) is
            // unreachable — the Rust backend contract makes it a method.
            r.get_out_samples(
                in_samples,
                self.in_buffer_count as i64,
                self.in_rate(),
                self.out_rate(),
            )?
        } else {
            debug_assert_eq!(self.out_rate(), self.in_rate());
            self.in_buffer_count as i64 + in_samples as i64
        };
        if out_samples > i32::MAX as i64 {
            return Err(Error::InvalidArgument(
                "swr_get_out_samples: count overflow".into(),
            ));
        }
        Ok(out_samples as usize)
    }

    /// Buffered input sample count. **Not a libswresample function** — it is
    /// a libavresample remnant (grep of the C tree finds no
    /// `swr_get_in_samples`); provided as the query callers of that name
    /// actually want: the delay at `in_sample_rate` is the buffered count on
    /// the FIFO path, and the resampler's in-flight window on the resampled
    /// path.
    pub fn get_in_samples(&self) -> usize {
        self.get_delay(self.in_rate() as i64).max(0) as usize
    }

    /// `swr_set_compensation` (`swresample.c:931-949`). Activating
    /// compensation on a non-resampling context forces the resampler
    /// (`SWR_FLAG_RESAMPLE` + re-init) — which destroys all buffered state;
    /// faithful C behavior, kept.
    pub fn set_compensation(
        &mut self,
        sample_delta: i32,
        compensation_distance: i32,
    ) -> Result<()> {
        if compensation_distance < 0 {
            return Err(Error::InvalidArgument(
                "swr_set_compensation: negative compensation_distance".into(),
            ));
        }
        if compensation_distance == 0 && sample_delta != 0 {
            return Err(Error::InvalidArgument(
                "swr_set_compensation: sample_delta without compensation_distance".into(),
            ));
        }
        if self.resample.is_none() {
            self.options.flags |= SwrFlags::RESAMPLE;
            self.init()?;
        }
        // The !resampler->set_compensation EINVAL (:944-945) is unreachable.
        self.resample
            .as_mut()
            .expect("init above")
            .set_compensation(sample_delta, compensation_distance)
    }

    /// `swr_drop_output` (`swresample.c:858-867`).
    pub fn drop_output(&mut self, count: i32) -> Result<usize> {
        self.drop_output += count;
        if self.drop_output <= 0 {
            return Ok(0);
        }
        log_verbose!(LOG_CTX, "discarding {} audio samples", count);
        // C passes a garbage non-NULL pointer with count 0 — a non-flush
        // zero-input call that enters the drop loop. An empty plane slice
        // is the same shape.
        let planes: [&[u8]; 1] = [&[]];
        self.convert(None, self.drop_output as usize, Some(&planes), 0)
    }

    /// `swr_inject_silence` (`swresample.c:869-899`).
    pub fn inject_silence(&mut self, mut count: i32) -> Result<usize> {
        if count <= 0 {
            return Ok(0);
        }
        // :877-881 — iterative chunking replaces C's self-recursion.
        while count > MAX_SILENCE_STEP {
            self.inject_silence(MAX_SILENCE_STEP)?;
            count -= MAX_SILENCE_STEP;
        }
        let count = count as usize;
        // The silence buffer is INPUT-format (inited from s->in, :352) — the
        // injected samples traverse the full pipeline.
        let mut silence = std::mem::replace(&mut self.silence, AudioData::empty());
        let res = silence.realloc_audio(count).and_then(|_| {
            // :887-888 — the fill byte (DSD 0x69 kept: Dsd is in this port's
            // SampleFormat).
            let fill = if silence.fmt == SampleFormat::Dsd {
                0x69
            } else if silence.bps == 1 {
                0x80
            } else {
                0
            };
            let bps = silence.bps;
            if silence.planar {
                for i in 0..silence.ch_count {
                    let plane = silence.plane_bytes_mut(i).ok_or(Error::BufferTooSmall)?;
                    plane[..count * bps].fill(fill);
                }
            } else {
                let n = count * silence.bps * silence.ch_count;
                silence.data_mut()[..n].fill(fill);
            }
            log_verbose!(LOG_CTX, "adding {} samples of silence", count);
            // reversefill_audiodata + swr_convert(NULL, 0, tmp, count).
            let planes: Vec<&[u8]> = if silence.planar {
                (0..silence.ch_count)
                    .map(|i| silence.plane(i).unwrap_or(&[]))
                    .collect()
            } else {
                vec![silence.plane(0).unwrap_or(&[])]
            };
            self.convert(None, 0, Some(&planes), count)
        });
        self.silence = silence;
        res
    }

    /// `swr_next_pts` (`swresample.c:951-983`) — the timestamp governor.
    /// MODE 1 (default `min_compensation = FLT_MAX`): pure passthrough with
    /// the delay subtracted. MODE 2: hard compensation (silence inject /
    /// output drop) or soft (rate skew via [`Self::set_compensation`]).
    pub fn next_pts(&mut self, pts: i64) -> i64 {
        // :952-953 — INT64_MIN is "no pts".
        if pts == i64::MIN {
            return self.outpts;
        }
        // :955-956 — latch the first pts.
        if self.firstpts == NOPTS {
            self.outpts = pts;
            self.firstpts = pts;
        }

        if self.options.min_compensation >= f32::MAX {
            // :958-959 — MODE 1.
            self.outpts = pts - self.get_delay(self.in_rate() as i64 * self.out_rate() as i64);
            return self.outpts;
        }
        // :961-962 — MODE 2.
        let delta =
            pts - self.get_delay(self.in_rate() as i64 * self.out_rate() as i64) - self.outpts
                + self.drop_output as i64 * self.in_rate() as i64;
        let fdelta = delta as f64 / (self.in_rate() as i64 * self.out_rate() as i64) as f64;

        if fdelta.abs() > self.options.min_compensation as f64 {
            if self.outpts == self.firstpts
                || fdelta.abs() > self.options.min_hard_compensation as f64
            {
                // :967-968 — hard compensation (i64 truncating division).
                let ret = if delta > 0 {
                    self.inject_silence((delta / self.out_rate() as i64) as i32)
                } else {
                    self.drop_output(((-delta) / self.in_rate() as i64) as i32)
                };
                if ret.is_err() {
                    // :969-971 — logged, not propagated.
                    log_error!(
                        LOG_CTX,
                        "Failed to compensate for timestamp delta of {:.6}",
                        fdelta
                    );
                }
            } else if self.options.soft_compensation_duration != 0.0
                && self.options.max_soft_compensation != 0.0
            {
                // :973-977 — soft compensation. C's av_clipf narrows to
                // float; duration and the division keep C's promotion shape.
                let duration =
                    (self.out_rate() as f32 * self.options.soft_compensation_duration) as i32;
                let max_soft_compensation = self.options.max_soft_compensation
                    / if self.options.max_soft_compensation < 0.0 {
                        -(self.in_rate() as f32)
                    } else {
                        1.0
                    };
                let comp = ((fdelta as f32).clamp(-max_soft_compensation, max_soft_compensation)
                    * duration as f32) as i32;
                log_verbose!(
                    LOG_CTX,
                    "compensating audio timestamp drift:{:.6} compensation:{} in:{}",
                    fdelta,
                    comp,
                    duration
                );
                let _ = self.set_compensation(comp, duration);
            }
        }
        self.outpts
    }

    // -- swresample_frame.c ---------------------------------------------------

    /// `swr_config_frame` (`swresample_frame.c:27-62`): reset + adopt the
    /// frames' configurations. Always closes first — C's documented
    /// contract, even on failure. The C failure path (`av_opt_set_*` +
    /// `"Failed to set option"`) is unrepresentable: typed field stores
    /// cannot fail.
    pub fn config_frame(
        &mut self,
        out: Option<&crate::util::audio_frame::AudioFrame>,
        in_: Option<&crate::util::audio_frame::AudioFrame>,
    ) -> Result<()> {
        self.close();
        if let Some(f) = in_ {
            self.options.in_chlayout = f.ch_layout;
            self.options.in_sample_fmt = Some(f.format);
            self.options.in_sample_rate = f.sample_rate;
        }
        if let Some(f) = out {
            self.options.out_chlayout = f.ch_layout;
            self.options.out_sample_fmt = Some(f.format);
            self.options.out_sample_rate = f.sample_rate;
        }
        Ok(())
    }

    /// `config_changed` (`swresample_frame.c:64-92`). C ORs
    /// `AVERROR_INPUT_CHANGED|AVERROR_OUTPUT_CHANGED`; no consumer branches
    /// on the codes, so the texts carry the distinction.
    fn config_changed(
        &self,
        out: Option<&crate::util::audio_frame::AudioFrame>,
        in_: Option<&crate::util::audio_frame::AudioFrame>,
    ) -> Result<()> {
        let mut text = String::new();
        if let Some(f) = in_ {
            if self.in_ch_layout != f.ch_layout
                || self.in_rate() != f.sample_rate
                || self.in_fmt() != f.format
            {
                text = "input configuration changed".into();
            }
        }
        if let Some(f) = out {
            if self.out_ch_layout != f.ch_layout
                || self.out_rate() != f.sample_rate
                || self.out_fmt() != f.format
            {
                text = if text.is_empty() {
                    "output configuration changed".into()
                } else {
                    "input and output configuration changed".into()
                };
            }
        }
        if text.is_empty() {
            Ok(())
        } else {
            Err(Error::InvalidArgument(text))
        }
    }

    /// `swr_convert_frame` (`swresample_frame.c:139-174`).
    pub fn convert_frame(
        &mut self,
        mut out: Option<&mut crate::util::audio_frame::AudioFrame>,
        in_: Option<&crate::util::audio_frame::AudioFrame>,
    ) -> Result<usize> {
        let mut setup = false;
        if !self.is_initialized() {
            // :144-149.
            let (o, i) = split_frame_ref(out.as_deref_mut(), in_);
            self.config_frame(o, i)?;
            self.init()?;
            setup = true;
        } else {
            // :150-153.
            let (o, i) = split_frame_ref(out.as_deref_mut(), in_);
            if let Err(e) = self.config_changed(o, i) {
                return Err(e);
            }
        }

        if let Some(out_f) = out.as_deref_mut() {
            // :156-171 — the output frame sizing.
            if out_f.planes.is_empty() || out_f.planes[0].linesize == 0 {
                let mut nb = self.get_delay(self.out_rate() as i64) as usize + 3;
                if let Some(in_f) = in_ {
                    // Truncating i64 division (C's in_nb*out/in cast).
                    nb = (nb as i64
                        + in_f.nb_samples as i64 * self.out_rate() as i64 / self.in_rate() as i64)
                        as usize;
                }
                out_f.nb_samples = nb;
                // av_frame_get_buffer keeps the frame's metadata; the port
                // preserves pts/time_base (the only fields this path sets).
                let pts = out_f.pts;
                let time_base = out_f.time_base;
                let format = out_f.format;
                let ch_layout = out_f.ch_layout;
                match crate::util::audio_frame::AudioFrame::alloc(format, ch_layout, nb) {
                    Ok(mut f) => {
                        f.pts = pts;
                        f.time_base = time_base;
                        *out_f = f;
                    }
                    Err(e) => {
                        if setup {
                            self.close(); // :163-165.
                        }
                        return Err(e);
                    }
                }
            } else if out_f.nb_samples == 0 {
                // available_samples (swresample_frame.c:126-137).
                let bps = out_f.format.bytes_per_sample();
                let samples = out_f.planes[0].linesize / bps;
                out_f.nb_samples = if out_f.format.is_planar() {
                    samples
                } else {
                    samples / out_f.ch_layout.nb_channels
                };
            }
        }

        // convert_frame (:94-124).
        let in_nb = in_.map_or(0, |f| f.nb_samples);
        let iplanes: Vec<&[u8]> = match in_ {
            Some(in_f) => (0..in_f.nb_planes())
                .map(|i| in_f.plane(i))
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        let in_ref: Option<&[&[u8]]> = if in_.is_some() { Some(&iplanes) } else { None };
        let res = match out.as_deref_mut() {
            Some(out_f) => {
                let nb_samples = out_f.nb_samples;
                // iter_mut yields disjoint plane borrows; Plane::data_mut is
                // the frame's CoW buffer accessor.
                let mut planes: Vec<&mut [u8]> =
                    out_f.planes.iter_mut().map(|p| p.data_mut()).collect();
                self.convert(Some(&mut planes), nb_samples, in_ref, in_nb)
            }
            None => self.convert(None, 0, in_ref, 0),
        };
        match res {
            Ok(ret) => {
                if let Some(out_f) = out.as_deref_mut() {
                    out_f.nb_samples = ret;
                }
                Ok(ret)
            }
            Err(e) => {
                if let Some(out_f) = out.as_deref_mut() {
                    out_f.nb_samples = 0; // :115-117.
                }
                Err(e)
            }
        }
    }
}

/// Split an `Option<&mut AudioFrame>` + `Option<&AudioFrame>` pair into two
/// shared refs for the config queries (both only read). Returns
/// `(out_shared, in_shared)`.
fn split_frame_ref<'a>(
    out: Option<&'a mut crate::util::audio_frame::AudioFrame>,
    in_: Option<&'a crate::util::audio_frame::AudioFrame>,
) -> (
    Option<&'a crate::util::audio_frame::AudioFrame>,
    Option<&'a crate::util::audio_frame::AudioFrame>,
) {
    (out.map(|f| &*f), in_)
}

/// The stage-2/3 source buffer by [`Tgt`] — an `Arc`-shared clone (cheap;
/// never the same target as the stage's destination, which the C
/// `postin != midbuf` / `midbuf != preout` guards guarantee).
fn src_of(t: Tgt, in_: &AudioData, postin: &AudioData, midbuf: &AudioData) -> AudioData {
    match t {
        Tgt::CallerIn => in_.clone(),
        Tgt::Postin => postin.clone(),
        Tgt::Midbuf => midbuf.clone(),
        _ => unreachable!("preout/out are never stage sources"),
    }
}

/// `swr_free` (`swresample.c:137-150`) — plain ownership teardown; the
/// resample backend drops with the field.
impl Drop for SwrContext {
    fn drop(&mut self) {
        self.clear_context();
    }
}

// ---------------------------------------------------------------------------
// Tests — the driver contracts (swresample.c + options.c + swresample_frame.c)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::audio_frame::AudioFrame;
    use crate::util::mathematics::{Rounding, rescale_rnd};

    // -- helpers --------------------------------------------------------------

    /// A context with the six `swr_alloc_set_opts2` parameters set, pre-init.
    fn swr_ctx(
        isr: i32,
        osr: i32,
        isf: SampleFormat,
        osf: SampleFormat,
        ichl: ChannelLayout,
        ochl: ChannelLayout,
    ) -> SwrContext {
        let mut s = SwrContext::alloc();
        s.options.in_sample_rate = isr;
        s.options.out_sample_rate = osr;
        s.options.in_sample_fmt = Some(isf);
        s.options.out_sample_fmt = Some(osf);
        s.options.in_chlayout = ichl;
        s.options.out_chlayout = ochl;
        s
    }

    fn err_text<T: std::fmt::Debug>(r: Result<T>) -> String {
        match r {
            Err(Error::InvalidArgument(t)) => t,
            Err(Error::Unsupported(t)) => t,
            other => panic!("expected InvalidArgument/Unsupported, got {other:?}"),
        }
    }

    /// `err_text` for an already-extracted error value.
    fn err_str(e: Error) -> String {
        match e {
            Error::InvalidArgument(t) | Error::Unsupported(t) => t,
            other => panic!("expected InvalidArgument/Unsupported, got {other:?}"),
        }
    }

    fn s16_bytes(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }
    fn as_i16(b: &[u8]) -> Vec<i16> {
        b.chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// One packed (interleaved) convert call.
    fn conv_packed(
        s: &mut SwrContext,
        out: &mut [u8],
        out_count: usize,
        input: &[u8],
        in_count: usize,
    ) -> Result<usize> {
        let mut o = [out];
        let i = [input];
        s.convert(Some(&mut o), out_count, Some(&i), in_count)
    }

    // -- swr_alloc / options.c defaults ----------------------------------------

    #[test]
    fn alloc_defaults() {
        // options.c:40-127 field by field.
        let o = SwrOptions::default();
        assert_eq!(o.in_sample_rate, 0);
        assert_eq!(o.out_sample_rate, 0);
        assert_eq!(o.in_sample_fmt, None);
        assert_eq!(o.out_sample_fmt, None);
        assert_eq!(o.internal_sample_fmt, None);
        assert_eq!(o.in_chlayout, ChannelLayout::default());
        assert_eq!(o.out_chlayout, ChannelLayout::default());
        assert_eq!(o.used_chlayout, ChannelLayout::default());
        // C_30DB = M_SQRT1_2 stored to float.
        assert_eq!(o.clev.to_bits(), 0.7071067811865476f32.to_bits());
        assert_eq!(o.slev.to_bits(), 0.7071067811865476f32.to_bits());
        assert_eq!(o.lfe_mix_level, 0.0);
        assert_eq!(o.rematrix_volume, 1.0);
        assert_eq!(o.rematrix_maxval, 0.0);
        assert_eq!(o.flags, SwrFlags::default());
        assert_eq!(o.dither_scale, 1.0);
        assert_eq!(o.dither_method, SwrDitherType::None);
        assert_eq!(o.filter_size, 32);
        assert_eq!(o.phase_shift, 10);
        assert!(o.linear_interp);
        assert!(o.exact_rational);
        assert_eq!(o.cutoff, 0.0);
        assert_eq!(o.engine, SwrEngine::Swr);
        assert_eq!(o.precision, 20.0);
        assert!(!o.cheby);
        assert_eq!(o.min_compensation, f32::MAX);
        assert_eq!(o.min_hard_compensation, 0.1);
        assert_eq!(o.soft_compensation_duration, 1.0);
        assert_eq!(o.max_soft_compensation, 0.0);
        assert_eq!(o.async_, 0.0);
        assert_eq!(o.first_pts, NOPTS);
        assert_eq!(o.matrix_encoding, MatrixEncoding::None);
        assert_eq!(o.filter_type, FilterType::Kaiser);
        assert_eq!(o.kaiser_beta, 9.0);
        assert_eq!(o.output_sample_bits, 0);
        // swr_alloc's runtime set: firstpts NOPTS, everything else zeroed.
        let s = SwrContext::alloc();
        assert_eq!(s.firstpts, NOPTS);
        assert_eq!(s.outpts, 0);
        assert!(!s.is_initialized());
    }

    #[test]
    fn version_accessors() {
        // version.c + version.h: AV_VERSION_INT(7, 3, 100).
        assert_eq!(LIBSWRESAMPLE_VERSION_INT, 7 << 16 | 3 << 8 | 100);
        assert_eq!(LIBSWRESAMPLE_VERSION_INT, 459620);
        assert_eq!(swresample_version(), 459620);
        assert_eq!(swresample_license(), "LGPL version 2.1 or later");
    }

    // -- swr_init: validation order + verbatim texts ---------------------------

    #[test]
    fn init_error_order() {
        // (2) formats first (unset = -1).
        let mut s = SwrContext::alloc();
        assert_eq!(
            err_text(s.init()),
            "Requested input sample format -1 is invalid"
        );
        s.options.in_sample_fmt = Some(SampleFormat::S16);
        assert_eq!(
            err_text(s.init()),
            "Requested output sample format -1 is invalid"
        );
        // (4-5) rates.
        s.options.out_sample_fmt = Some(SampleFormat::S16);
        assert_eq!(
            err_text(s.init()),
            "Requested input sample rate 0 is invalid"
        );
        s.options.in_sample_rate = 48000;
        assert_eq!(
            err_text(s.init()),
            "Requested output sample rate 0 is invalid"
        );
        // (8) layouts — check_chlayout text, input side first.
        s.options.out_sample_rate = 48000;
        assert_eq!(
            err_text(s.init()),
            "input channel layout \"\" is invalid or unsupported."
        );
        s.options.in_chlayout = ChannelLayout::STEREO;
        assert_eq!(
            err_text(s.init()),
            "output channel layout \"\" is invalid or unsupported."
        );
        // SWR_CH_MAX over-wide but check()-valid: described layout quoted.
        s.options.out_chlayout = ChannelLayout::unspecified(65);
        assert_eq!(
            err_text(s.init()),
            "output channel layout \"65 channels\" is invalid or unsupported."
        );
        s.options.out_chlayout = ChannelLayout::STEREO;
        assert!(s.init().is_ok(), "minimal stereo ctx inits");
    }

    #[test]
    fn init_dsd_gate() {
        // swresample.c:180-186 — conversion TO DSD rejected; dsd->dsd at
        // equal rates without forced resampling is the only legal DSD.
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::Dsd,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        assert_eq!(err_text(s.init()), "Conversion to DSD is not supported");
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::Dsd,
            SampleFormat::Dsd,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        assert!(s.init().is_ok());
    }

    #[test]
    fn engine_soxr_unavailable() {
        // swresample.c:210-212 — the non-soxr-build error, verbatim.
        let mut s = swr_ctx(
            48000,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.engine = SwrEngine::Soxr;
        assert_eq!(
            err_text(s.init()),
            "Requested resampling engine is unavailable"
        );
    }

    #[test]
    fn internal_fmt_whitelist() {
        // swresample.c:269-276 — text verbatim incl. s64p.
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.internal_sample_fmt = Some(SampleFormat::S16);
        assert_eq!(
            err_text(s.init()),
            "Requested sample format s16 is not supported internally, \
             s16p/s32p/s64p/fltp/dblp are supported"
        );
        // s64p passes the whitelist (C accepts it; the backends reject it
        // individually) — a passthrough s64p->s64p ctx inits.
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S64p,
            SampleFormat::S64p,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.internal_sample_fmt = Some(SampleFormat::S64p);
        assert!(s.init().is_ok());
    }

    #[test]
    fn rematrix_needed_information_error() {
        // swresample.c:339-345 — the out layout stays UNSPEC (no 9-channel
        // native default exists), used.nb != out.ch_count -> not enough
        // information.
        let mut s = SwrContext::alloc();
        s.options.in_sample_rate = 48000;
        s.options.out_sample_rate = 48000;
        s.options.in_sample_fmt = Some(SampleFormat::S16);
        s.options.out_sample_fmt = Some(SampleFormat::S16);
        s.options.in_chlayout = ChannelLayout::unspecified(2);
        s.options.out_chlayout = ChannelLayout::unspecified(9);
        assert_eq!(
            err_text(s.init()),
            "Rematrix is needed between stereo and 9 channels but there is not enough information to do it"
        );
    }

    #[test]
    fn cannot_convert_pair_error() {
        // swresample.c:372-378 — in_convert pair text (source first).
        let mut s = swr_ctx(
            48000,
            44100,
            SampleFormat::Dsd,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        assert_eq!(
            err_text(s.init()),
            "Cannot convert dsd sample format to fltp sample format"
        );
    }

    #[test]
    fn dither_scale_zero_disables_unsupported_rejects() {
        // dither.c:87-103 — scale 0 disables the method (C-identical path);
        // a real precision loss needs dither.c -> Unsupported.
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S16,
            SampleFormat::Fltp,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.dither_method = SwrDitherType::Rectangular;
        assert!(s.init().is_ok(), "s16->fltp loses no bits: disabled");
        assert_eq!(s.dither_method, SwrDitherType::None);

        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::Fltp,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.dither_method = SwrDitherType::Triangular;
        match s.init() {
            Err(Error::Unsupported(m)) => assert!(m.contains("dither.c"), "{m}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // -- the internal-format ladder (swresample.c:235-266) ---------------------

    #[test]
    fn int_fmt_ladder() {
        let stereo = ChannelLayout::STEREO;
        #[rustfmt::skip]
        let cases: Vec<(i32, i32, SampleFormat, SampleFormat, ChannelLayout, ChannelLayout, SampleFormat)> = vec![
            // b: <=16bit to <=16bit at the same rate.
            (48000, 48000, SampleFormat::U8,  SampleFormat::U8,  stereo, stereo, SampleFormat::S16p),
            // c: bps sum <= 3.
            (48000, 48000, SampleFormat::U8,  SampleFormat::S16, stereo, stereo, SampleFormat::S16p),
            (48000, 48000, SampleFormat::S16, SampleFormat::Fltp, stereo, stereo, SampleFormat::S16p),
            // d: <=16bit, no rematrix, same rate, no forced resample.
            (48000, 48000, SampleFormat::S16, SampleFormat::S16p, stereo, stereo, SampleFormat::S16p),
            // e: s32p->s32p planar pair, no rematrix, same rate.
            (48000, 48000, SampleFormat::S32p, SampleFormat::S32p, stereo, stereo, SampleFormat::S32p),
            // stereo->mono rematrix does NOT gate branch b.
            (48000, 48000, SampleFormat::S16, SampleFormat::S16, stereo, ChannelLayout::MONO, SampleFormat::S16p),
            // f: <=4 bytes.
            (48000, 48000, SampleFormat::Flt, SampleFormat::Dbl, stereo, stereo, SampleFormat::Fltp),
            (44100, 48000, SampleFormat::S16, SampleFormat::S16, stereo, stereo, SampleFormat::Fltp),
            // g: dbl only.
            (48000, 48000, SampleFormat::Dbl, SampleFormat::Dblp, stereo, stereo, SampleFormat::Dblp),
        ];
        for (isr, osr, isf, osf, ichl, ochl, want) in cases {
            let mut s = swr_ctx(isr, osr, isf, osf, ichl, ochl);
            s.init()
                .unwrap_or_else(|e| panic!("{isf:?}->{osf:?}: {e:?}"));
            assert_eq!(s.int_sample_fmt, want, "{isf:?}->{osf:?} @{isr}->{osr}");
        }
    }

    #[test]
    fn resample_first_formula() {
        // swresample.c:349 — INTEGER division on the left, RSC = 1.
        let stereo = ChannelLayout::STEREO;
        let cases = vec![
            // (in, out, in_rate, out_rate, want)
            (stereo, ChannelLayout::FivePointOneBack, 48000, 48000, false), // 6/2-1 = 2 < 0? no
            (ChannelLayout::FivePointOneBack, stereo, 48000, 48000, true),  // 2/6 = 0; -1 < 0
            (stereo, stereo, 24000, 48000, true),                           // 2x rate: 0 < 1.0
            (
                stereo,
                ChannelLayout::FivePointZeroBack,
                48000,
                48000,
                false,
            ), // 5/2 = 2; 1 < 0? no
        ];
        for (ichl, ochl, isr, osr, want) in cases {
            let mut s = swr_ctx(isr, osr, SampleFormat::S16, SampleFormat::S16, ichl, ochl);
            s.init().unwrap();
            assert_eq!(s.resample_first, want, "{ichl:?} -> {ochl:?} @{isr}->{osr}");
        }
    }

    // -- lifecycle ---------------------------------------------------------------

    #[test]
    fn is_initialized_lifecycle() {
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        assert!(!s.is_initialized());
        s.init().unwrap();
        assert!(s.is_initialized());
        s.close();
        assert!(!s.is_initialized());
        // Re-init with changed options.
        s.options.out_sample_fmt = Some(SampleFormat::Fltp);
        s.init().unwrap();
        assert!(s.is_initialized());
        s.close();
        assert_eq!(s.outpts, 0, "outpts survives close (C keeps it)");
    }

    #[test]
    fn alloc_set_opts2_paths() {
        // The happy path sets the six fields.
        let s = SwrContext::alloc_set_opts2(
            None,
            &ChannelLayout::STEREO,
            SampleFormat::S16,
            44100,
            &ChannelLayout::FivePointOneBack,
            SampleFormat::Fltp,
            48000,
        )
        .unwrap();
        assert_eq!(s.options.out_chlayout, ChannelLayout::STEREO);
        assert_eq!(s.options.out_sample_fmt, Some(SampleFormat::S16));
        assert_eq!(s.options.out_sample_rate, 44100);
        assert_eq!(s.options.in_chlayout, ChannelLayout::FivePointOneBack);
        assert_eq!(s.options.in_sample_fmt, Some(SampleFormat::Fltp));
        assert_eq!(s.options.in_sample_rate, 48000);
        // The fail path: over-wide out layout -> the check_chlayout text
        // (after its WARNING; the fail label's ERROR is log-only).
        let e = match SwrContext::alloc_set_opts2(
            None,
            &ChannelLayout::unspecified(65),
            SampleFormat::S16,
            44100,
            &ChannelLayout::STEREO,
            SampleFormat::S16,
            48000,
        ) {
            Err(e) => e,
            Ok(_) => panic!("expected the over-wide layout to fail"),
        };
        assert_eq!(
            err_str(e),
            "ochl channel layout \"65 channels\" is invalid or unsupported."
        );
    }

    // -- the equal-rate paths ------------------------------------------------------

    #[test]
    fn identity_full_convert() {
        // swresample.c:358-365 — the single-conversion fast path: bytes out
        // == bytes in.
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        assert!(s.full_convert.is_some());
        let input: Vec<i16> = (0..200).map(|i| (i as i16).wrapping_mul(73)).collect();
        let mut out = vec![0u8; 400];
        let ret = conv_packed(&mut s, &mut out, 100, &s16_bytes(&input), 100).unwrap();
        assert_eq!(ret, 100);
        assert_eq!(out, s16_bytes(&input));
    }

    #[test]
    fn fmt_conversion_eq_rate() {
        // fltp -> s16: no rematrix/resample/channel-map, so the
        // single-conversion fast path takes it (one AudioConvert over the
        // caller planes); lrint ties-to-even lands in the output.
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::Fltp,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        assert!(s.full_convert.is_some(), "the fast path covers the pair");
        let ramp = [-1.0f32, -0.5, 0.0, 0.5, 0.9, -0.9, 1.0, -1.0, 0.25, -0.25];
        // Stereo: L = ramp, R = -ramp/2. Planar in, PACKED out (one plane).
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for &v in &ramp {
            l.extend_from_slice(&v.to_le_bytes());
            r.extend_from_slice(&(-v / 2.0).to_le_bytes());
        }
        let mut out = vec![0u8; 40]; // 10 frames interleaved
        let mut o = [&mut out[..]];
        let i = [&l[..], &r[..]];
        let ret = s.convert(Some(&mut o), 10, Some(&i), 10).unwrap();
        assert_eq!(ret, 10);
        let want_l: Vec<i16> = ramp
            .iter()
            .map(|&v| (v * 32768.0f32).round_ties_even().clamp(-32768.0, 32767.0) as i16)
            .collect();
        let want_r: Vec<i16> = ramp
            .iter()
            .map(|&v| {
                (-v / 2.0 * 32768.0f32)
                    .round_ties_even()
                    .clamp(-32768.0, 32767.0) as i16
            })
            .collect();
        let got = as_i16(&out);
        for i in 0..10 {
            assert_eq!(got[2 * i], want_l[i], "L[{i}]");
            assert_eq!(got[2 * i + 1], want_r[i], "R[{i}]");
        }
    }

    #[test]
    fn mono_u8_passthrough() {
        // ch_count == 1 forces planar internally (set_audiodata_fmt:102-103).
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::U8,
            SampleFormat::S16,
            ChannelLayout::MONO,
            ChannelLayout::MONO,
        );
        s.init().unwrap();
        let mut out = vec![0u8; 6];
        let ret = conv_packed(&mut s, &mut out, 3, &[0x00, 0x80, 0xFF], 3).unwrap();
        assert_eq!(ret, 3);
        assert_eq!(as_i16(&out[..6]), vec![-32768i16, 0, 32512]);
    }

    #[test]
    fn fifo_buffering() {
        // Feed 100 with 40 of out space -> 40; drains 40+20; flush -> 0.
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        let input: Vec<i16> = (0..200).map(|i| (i as i16).wrapping_mul(11)).collect();
        let inb = s16_bytes(&input);
        let mut collected = Vec::<u8>::new();
        let mut out = vec![0u8; 160];

        let ret = conv_packed(&mut s, &mut out, 40, &inb, 100).unwrap();
        assert_eq!(ret, 40);
        collected.extend_from_slice(&out[..160]);
        // Non-flush empty calls drain the FIFO (Some, 0 — not NULL).
        let empty: Vec<u8> = Vec::new();
        for expect in [40usize, 20] {
            out.iter_mut().for_each(|b| *b = 0);
            let ret = conv_packed(&mut s, &mut out, 40, &empty, 0).unwrap();
            assert_eq!(ret, expect);
            collected.extend_from_slice(&out[..expect * 4]);
        }
        // Flush on a non-resampling context with an empty buffer: 0.
        let n = s.convert(Some(&mut [&mut out]), 40, None, 0).unwrap();
        assert_eq!(n, 0);
        assert_eq!(collected, inb, "FIFO is order-preserving");
        // outpts advanced by all 100 samples (in-rate units).
        assert_eq!(s.outpts, 100 * 44100);
    }

    #[test]
    fn get_delay_fifo() {
        // swresample.c:905 — integer formula, three bases.
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        let input = vec![0u8; 400]; // 100 stereo frames
        let mut out = vec![0u8; 0];
        // Zero out space: everything buffers.
        let ret = conv_packed(&mut s, &mut out, 0, &input, 100).unwrap();
        assert_eq!(ret, 0);
        assert_eq!(s.in_buffer_count, 100);
        assert_eq!(s.get_delay(48000), 100);
        assert_eq!(s.get_delay(1000), 2); // (100*1000 + 24000)/48000
        assert_eq!(s.get_delay(1), 0); // (100 + 24000)/48000
        assert_eq!(s.get_in_samples(), 100);
    }

    #[test]
    fn custom_matrix_through_driver() {
        // swr_set_matrix routing: the matrix reaches rematrix init.
        let mut s = SwrContext::alloc();
        s.options.in_sample_rate = 44100;
        s.options.out_sample_rate = 44100;
        s.options.in_sample_fmt = Some(SampleFormat::S16);
        s.options.out_sample_fmt = Some(SampleFormat::S16);
        s.options.in_chlayout = ChannelLayout::STEREO;
        s.options.out_chlayout = ChannelLayout::STEREO;
        s.set_matrix(&[0.5, 0.0, 0.0, 0.5], 2).unwrap();
        s.init().unwrap();
        assert!(s.rematrix_needed, "rematrix_custom forces rematrixing");
        assert!(s.rematrix_ctx.is_some());
        assert!(s.full_convert.is_none());
        // 0.5 in 17.15 is exact (16384): 1000 -> 500.
        let input = s16_bytes(&[1000i16, -1000, 1000, -1000]);
        let mut out = vec![0u8; 8];
        let ret = conv_packed(&mut s, &mut out, 2, &input, 2).unwrap();
        assert_eq!(ret, 2);
        assert_eq!(as_i16(&out), vec![500i16, -500, 500, -500]);
        // set_matrix after init is rejected (rematrix.c:75).
        let e = s.set_matrix(&[1.0, 0.0, 0.0, 1.0], 2).unwrap_err();
        assert!(err_str(e).contains("allocated but not initialized"));
    }

    #[test]
    fn rematrix_5p1_to_stereo() {
        // The downmix through the full driver: FC-only input lands on both
        // outs at clev/(1+clev+slev) (row L1-normalized for the int stage).
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::FivePointOneBack,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        assert!(s.rematrix_needed);
        assert!(s.rematrix_ctx.is_some());
        assert_eq!(s.int_sample_fmt, SampleFormat::S16p);
        // 10 frames, FC = 1000, everything else 0.
        let mut frame = [0i16; 6];
        frame[2] = 1000; // FC
        let mut input = Vec::new();
        for _ in 0..10 {
            input.extend_from_slice(&s16_bytes(&frame));
        }
        let mut out = vec![0u8; 40];
        let ret = conv_packed(&mut s, &mut out, 10, &input, 10).unwrap();
        assert_eq!(ret, 10, "rematrix does not change the count");
        let vals = as_i16(&out);
        for i in 0..10 {
            let (l, r) = (vals[2 * i], vals[2 * i + 1]);
            assert_eq!(l, r, "symmetric downmix");
            // 1000 * 0.7071/(1+2*0.7071) = 292.9 -> ~293.
            assert!((250..=340).contains(&l), "FC downmix ~293, got {l}");
        }
    }

    // -- drop / inject ------------------------------------------------------------

    #[test]
    fn drop_output_consumes_without_emitting() {
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        // Queue the drop first (no input yet: returns 0, stays pending).
        let r = s.drop_output(50).unwrap();
        assert_eq!(r, 0);
        assert_eq!(s.drop_output, 50);
        // The next real convert discards the first 50 frames and emits from
        // frame 50 (interleaved: 50 frames = 100 i16 values at [100..200]).
        let input: Vec<i16> = (0..200).map(|i| (i as i16).wrapping_mul(3)).collect();
        let inb = s16_bytes(&input);
        let mut out = vec![0u8; 400];
        let ret = conv_packed(&mut s, &mut out, 100, &inb, 100).unwrap();
        assert_eq!(ret, 50);
        assert_eq!(as_i16(&out[..200]), input[100..200].to_vec());
        assert_eq!(s.drop_output, 0);
        // The main body's out_count (100) covered the whole remainder: the
        // buffered 50 drained in the same call, the buffer is empty.
        let empty: Vec<u8> = Vec::new();
        let ret = conv_packed(&mut s, &mut out, 50, &empty, 0).unwrap();
        assert_eq!(ret, 0);
        // outpts counts every surfaced sample (in-rate units).
        assert_eq!(s.outpts, 50 * 44100);
        // Cancellation: a negative drop zeroes a pending drop.
        s.drop_output(10).unwrap();
        s.drop_output(-10).unwrap();
        assert_eq!(s.drop_output, 0);
    }

    #[test]
    fn inject_silence_traverses_pipeline() {
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        let r = s.inject_silence(100).unwrap();
        assert_eq!(r, 0);
        assert_eq!(s.in_buffer_count, 100);
        // The 0x80 fill byte is for bps == 1 (u8/DSD); s16 silence is 0x00.
        let mut out = vec![0xEEu8; 400];
        let ret = conv_packed(&mut s, &mut out, 100, &[], 0).unwrap();
        assert_eq!(ret, 100);
        assert!(out[..400].iter().all(|&b| b == 0), "s16 silence is zeros");
        assert_eq!(s.outpts, 100 * 44100);
    }

    // -- next_pts -------------------------------------------------------------------

    #[test]
    fn next_pts_passthrough_mode1() {
        // Default options: min_compensation = FLT_MAX -> MODE 1.
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        let input = vec![0u8; 800]; // 200 frames
        let mut out = vec![0u8; 800];
        conv_packed(&mut s, &mut out, 200, &input, 200).unwrap();
        assert_eq!(s.outpts, 200 * 44100);
        // First call latches firstpts = pts, then outpts = pts - delay(0).
        let p = 1000 * 44100;
        assert_eq!(s.next_pts(p), p);
        assert_eq!(s.firstpts, p);
        assert_eq!(s.outpts, p);
        // INT64_MIN is "no pts": outpts returned unchanged.
        assert_eq!(s.next_pts(i64::MIN), p);
        // With buffered input the delay subtracts (in_rate*out_rate base).
        conv_packed(&mut s, &mut out, 0, &input, 200).unwrap(); // buffer 200
        let p2 = 5000 * 44100;
        let got = s.next_pts(p2);
        assert_eq!(got, p2 - s.get_delay((44100 * 44100) as i64));
    }

    #[test]
    fn next_pts_firstpts_and_hard_compensation() {
        // first_pts switches the context to MODE 2 (async forced to 1,
        // min_compensation 0.001) and seeds outpts.
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.first_pts = 1000;
        s.init().unwrap();
        assert_eq!(s.firstpts, 1000 * 44100);
        assert_eq!(s.outpts, 1000 * 44100);
        assert_eq!(s.options.async_, 1.0);
        assert_eq!(s.options.min_compensation, 0.001);

        // Hard inject: a pts far ahead (delta > min_hard_comp seconds).
        let input = vec![0u8; 400];
        let mut out = vec![0u8; 400];
        conv_packed(&mut s, &mut out, 100, &input, 100).unwrap(); // outpts 1100*rate
        let ret = s.next_pts(11_000 * 44100);
        assert_eq!(ret, 1100 * 44100, "next_pts returns outpts");
        // delta = 9900 samples > 0.1s: silence injected.
        assert!(
            s.in_buffer_count >= 9900 - 2,
            "injected {}",
            s.in_buffer_count
        );
    }

    #[test]
    fn next_pts_hard_drop() {
        let mut s = swr_ctx(
            44100,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.options.first_pts = 0;
        s.init().unwrap();
        let input = vec![0u8; 80_000]; // 20_000 frames
        let mut out = vec![0u8; 80_000];
        conv_packed(&mut s, &mut out, 20_000, &input, 20_000).unwrap();
        assert_eq!(s.outpts, 20_000 * 44100);
        // A pts 19_900 samples behind -> drop_output(19900) queued.
        let ret = s.next_pts(100 * 44100);
        assert_eq!(ret, 20_000 * 44100);
        assert_eq!(s.drop_output, 19_900);
    }

    // -- the resample path ------------------------------------------------------------

    #[test]
    fn constant_through_2x_upsample() {
        // THE scalar: 2x upsample of a constant stays constant through the
        // full driver (the raw resample zone couldn't run this).
        let mut s = swr_ctx(
            24000,
            48000,
            SampleFormat::Fltp,
            SampleFormat::Fltp,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        assert!(s.resample.is_some());
        assert!(s.resample_first);
        const N: usize = 256;
        let l = f32_bytes(&vec![0.25f32; N]);
        let r = f32_bytes(&vec![-0.25f32; N]);
        let mut outl = vec![0u8; 16 * N]; // 4*N samples of f32 headroom
        let mut outr = vec![0u8; 16 * N];
        let mut got = 0usize;
        let bound = s.get_out_samples(N as i32).unwrap();
        assert!(bound >= 2 * N);
        {
            let mut o = [&mut outl[..], &mut outr[..]];
            let i = [&l[..], &r[..]];
            let ret = s.convert(Some(&mut o), bound, Some(&i), N).unwrap();
            assert!(ret <= bound, "convert respects the out bound");
            got += ret;
        }
        // EOF flush drains the filter (in = None).
        loop {
            let room = (outl.len() / 4) - got;
            let mut o = [&mut outl[got * 4..], &mut outr[got * 4..]];
            let ret = s.convert(Some(&mut o), room, None, 0).unwrap();
            if ret == 0 {
                break;
            }
            got += ret;
        }
        assert!(
            (got as i64 - 2 * N as i64).abs() <= 2,
            "total {got} vs {}",
            2 * N
        );
        let lv: Vec<f32> = outl[..got * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let rv: Vec<f32> = outr[..got * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (i, (&a, &b)) in lv.iter().zip(rv.iter()).enumerate() {
            assert!(
                (a - 0.25).abs() < 1e-4 && (b + 0.25).abs() < 1e-4,
                "sample {i}: {a} {b}"
            );
        }
    }

    #[test]
    fn count_bookkeeping_at_rate_change() {
        // Feed N at a rate change -> out ~= N*out/in after drain; the EOF
        // flush yields the delay (get_delay -> 0 once drained).
        let (inr, outr) = (48000u64, 44100u64);
        let mut s = swr_ctx(
            inr as i32,
            outr as i32,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        assert_eq!(s.int_sample_fmt, SampleFormat::Fltp);
        assert!(s.get_delay(inr as i64) >= 0);

        const CHUNK: usize = 1024;
        const NCHUNKS: usize = 10;
        let input = s16_bytes(&vec![1234i16; 2 * CHUNK]);
        let mut out = vec![0u8; 8 * CHUNK + 64];
        let mut total = 0usize;
        for _ in 0..NCHUNKS {
            let bound = s.get_out_samples(CHUNK as i32).unwrap();
            assert!(bound > 0);
            let ret = conv_packed(&mut s, &mut out, bound, &input, CHUNK).unwrap();
            assert!(ret <= bound, "the get_out_samples contract");
            total += ret;
        }
        // Not everything is out yet: the filter holds the delay.
        let delayed = s.get_delay(inr as i64);
        assert!(delayed > 0, "pre-flush delay {delayed}");
        // EOF flush (in = None).
        loop {
            let oc = out.len() / 4;
            let mut o = [&mut out[..]];
            let ret = s.convert(Some(&mut o), oc, None, 0).unwrap();
            if ret == 0 {
                break;
            }
            total += ret;
        }
        let expect = rescale_rnd(
            (NCHUNKS * CHUNK) as i64,
            outr as i64,
            inr as i64,
            Rounding::Up,
            false,
        );
        assert!(
            (total as i64 - expect).abs() <= 4,
            "total {total} vs {expect}"
        );
        // The residual delay after a full drain is the filter's inherent
        // group-delay report (bounded by (filter_length+1)/2 = 18 for the
        // default 32-tap filter at this ratio), not a data backlog.
        assert!(
            s.get_delay(inr as i64) <= 36,
            "post-drain delay {}",
            s.get_delay(inr as i64)
        );
        // Idempotent flush.
        assert_eq!(conv_packed(&mut s, &mut out, 64, &[], 0).unwrap(), 0);
        assert_eq!(s.convert(Some(&mut [&mut out]), 64, None, 0).unwrap(), 0);
        // outpts advances by OUT samples in IN-rate units (C :798).
        assert_eq!(s.outpts, total as i64 * inr as i64);
    }

    #[test]
    fn get_out_samples_upper_bound() {
        // The C ASSERT_LEVEL>1 invariant, as a test: converting with
        // out_count = get_out_samples(n) never buffers for lack of space.
        let mut s = swr_ctx(
            48000,
            44100,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        let input = vec![0u8; 2 * 2 * 512];
        let mut out = vec![0u8; 8192];
        for chunk in [7usize, 512, 1, 100] {
            let bound = s.get_out_samples(chunk as i32).unwrap();
            let ret = conv_packed(&mut s, &mut out, bound, &input, chunk).unwrap();
            assert!(ret <= bound, "chunk {chunk}: {ret} > {bound}");
        }
        // FIFO bound: buffered + in; negative rejected.
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        let input = vec![0u8; 400];
        let mut out = vec![0u8; 0];
        conv_packed(&mut s, &mut out, 0, &input, 100).unwrap();
        assert_eq!(s.get_out_samples(32).unwrap(), 132);
        assert!(s.get_out_samples(-1).is_err());
    }

    #[test]
    fn roundtrip_48k_44k1_48k() {
        // The C self-test shape: fwd 48000->44100 fltp stereo, back, N=9600.
        let mut fwd = swr_ctx(
            48000,
            44100,
            SampleFormat::Fltp,
            SampleFormat::Fltp,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        fwd.init().unwrap();
        let mut bwd = swr_ctx(
            44100,
            48000,
            SampleFormat::Fltp,
            SampleFormat::Fltp,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        bwd.init().unwrap();

        const N: usize = 9600;
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for i in 0..N {
            let t = i as f32 / 48000.0;
            let v = 0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            l.extend_from_slice(&v.to_le_bytes());
            r.extend_from_slice(&(-v).to_le_bytes());
        }

        // Push one planar buffer through ctx, then flush-drain it.
        fn run(
            ctx: &mut SwrContext,
            il: &[u8],
            ir: &[u8],
            in_count: usize,
        ) -> (Vec<f32>, Vec<f32>) {
            let (mut outl, mut outr) = (Vec::new(), Vec::new());
            let cap = 2 * in_count + 8192;
            let mut bl = vec![0u8; 4 * cap];
            let mut br = vec![0u8; 4 * cap];
            let ret = {
                let mut o = [&mut bl[..], &mut br[..]];
                let i = [il, ir];
                ctx.convert(Some(&mut o), cap, Some(&i), in_count).unwrap()
            };
            outl.extend_from_slice(&bl[..ret * 4]);
            outr.extend_from_slice(&br[..ret * 4]);
            loop {
                let mut o = [&mut bl[..], &mut br[..]];
                let ret = ctx.convert(Some(&mut o), cap, None, 0).unwrap();
                if ret == 0 {
                    break;
                }
                outl.extend_from_slice(&bl[..ret * 4]);
                outr.extend_from_slice(&br[..ret * 4]);
            }
            (
                outl.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                outr.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }

        let (midl, midr) = run(&mut fwd, &l, &r, N);
        let (backl, backr) = run(&mut bwd, &f32_bytes(&midl), &f32_bytes(&midr), midl.len());
        assert_eq!(backl.len(), backr.len());
        assert!(
            (backl.len() as i64 - N as i64).abs() <= 4,
            "length {} vs {N}",
            backl.len()
        );
        // Best-lag alignment, then maxdiff + correlation on the L channel.
        let src: Vec<f32> = l
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut best = (f32::INFINITY, 0i64);
        for lag in -64i64..=64i64 {
            let mut maxd = 0f32;
            for i in 100..(N - 100) {
                let j = i as i64 + lag;
                if j < 0 || j >= backl.len() as i64 {
                    continue;
                }
                maxd = maxd.max((src[i] - backl[j as usize]).abs());
            }
            if maxd < best.0 {
                best = (maxd, lag);
            }
        }
        assert!(best.0 < 0.05, "maxdiff {} at lag {}", best.0, best.1);
        let lag = best.1 as usize;
        let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let n = (N - 200) as f64;
        for i in 100..(N - 100) {
            let a = src[i] as f64;
            let b = backl[i + lag] as f64;
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
        let cov = sab - sa * sb / n;
        let var = ((saa - sa * sa / n) * (sbb - sb * sb / n)).sqrt();
        assert!(cov / var > 0.99, "correlation {}", cov / var);
    }

    // -- swresample_frame.c --------------------------------------------------------

    #[test]
    fn convert_frame_lifecycle() {
        // Config via frames (the swr_config_frame path), then convert with
        // an unallocated out frame. The OUT side must already be configured
        // (an out-less convert_frame cannot set it — same as C).
        let mut s = SwrContext::alloc();
        s.options.out_sample_rate = 48000;
        s.options.out_sample_fmt = Some(SampleFormat::S16);
        s.options.out_chlayout = ChannelLayout::STEREO;
        let in_f = AudioFrame::alloc(SampleFormat::Fltp, ChannelLayout::STEREO, 100).unwrap();
        // av_frame carries the sample rate from the producer; alloc's
        // default is 0, so set it like a decoded frame would.
        let mut in_f = in_f;
        in_f.sample_rate = 48000;
        assert_eq!(in_f.nb_samples, 100);
        let ret = s.convert_frame(None, Some(&in_f)).unwrap();
        assert_eq!(ret, 0, "no out frame: nothing emitted");
        assert!(s.is_initialized());

        // A mismatched input frame after init: configuration changed.
        let other =
            AudioFrame::alloc(SampleFormat::Fltp, ChannelLayout::FivePointOneBack, 100).unwrap();
        let e = s.convert_frame(None, Some(&other)).unwrap_err();
        assert_eq!(err_str(e), "input configuration changed");

        // The out frame is allocated to delay + 3 + in*out/in samples.
        let mut out_f = AudioFrame::default();
        out_f.format = SampleFormat::S16;
        out_f.ch_layout = ChannelLayout::STEREO;
        out_f.sample_rate = 48000;
        let ret = s.convert_frame(Some(&mut out_f), Some(&in_f)).unwrap();
        assert_eq!(ret, 100);
        assert_eq!(out_f.nb_samples, 100);
        assert_eq!(out_f.nb_planes(), 1, "packed s16 out");
        assert!(out_f.plane(0).len() >= 100 * 2 * 2);
        // An already-allocated empty out frame sizes from its linesize.
        let mut sized = AudioFrame::alloc(SampleFormat::S16, ChannelLayout::STEREO, 50).unwrap();
        sized.sample_rate = 48000; // a real frame carries its rate
        sized.nb_samples = 0;
        let ret = s.convert_frame(Some(&mut sized), None).unwrap();
        assert!(ret <= 50);
    }

    // -- channel mapping ------------------------------------------------------------

    #[test]
    fn channel_mapping_routes() {
        // swresample.c:47-52 — only before init; postin/midbuf take
        // used_chlayout.nb_channels (swresample.c:384-389).
        let base = || {
            let mut s = SwrContext::alloc();
            s.options.in_sample_rate = 44100;
            s.options.out_sample_rate = 44100;
            s.options.in_sample_fmt = Some(SampleFormat::S16);
            s.options.out_sample_fmt = Some(SampleFormat::S16);
            s.options.in_chlayout = ChannelLayout::STEREO;
            s.options.out_chlayout = ChannelLayout::STEREO;
            s
        };
        let mut s = base();
        // Swap channels: [1, 0].
        s.set_channel_mapping(Some(&[1, 0])).unwrap();
        s.init().unwrap();
        assert!(s.in_convert.is_some());
        assert_eq!(s.postin.ch_count, 2);
        assert!(s.full_convert.is_none(), "channel_map blocks the fast path");
        let input = s16_bytes(&[100i16, -100, 200, -200]);
        let mut out = vec![0u8; 8];
        let ret = conv_packed(&mut s, &mut out, 2, &input, 2).unwrap();
        assert_eq!(ret, 2);
        assert_eq!(as_i16(&out), vec![-100i16, 100, -200, 200]);
        // After init: rejected.
        let e = s.set_channel_mapping(None).unwrap_err();
        assert!(err_str(e).contains("allocated but not initialized"));
        // Muted channel (-1) substitutes the s16 silence byte (zeros).
        let mut s = base();
        s.set_channel_mapping(Some(&[-1, 0])).unwrap();
        s.init().unwrap();
        let mut out = vec![0u8; 8];
        conv_packed(&mut s, &mut out, 2, &input, 2).unwrap();
        assert_eq!(as_i16(&out), vec![0i16, 100, 0, 200]);
    }

    // -- set_compensation ------------------------------------------------------------

    #[test]
    fn set_compensation_forces_resampler() {
        // swresample.c:938-943 — an equal-rate context gains the resampler
        // (and is re-initialized, destroying buffered state).
        let mut s = swr_ctx(
            48000,
            48000,
            SampleFormat::S16,
            SampleFormat::S16,
            ChannelLayout::STEREO,
            ChannelLayout::STEREO,
        );
        s.init().unwrap();
        assert!(s.resample.is_none());
        let input = vec![0u8; 400];
        let mut out = vec![0u8; 0];
        conv_packed(&mut s, &mut out, 0, &input, 100).unwrap();
        assert_eq!(s.in_buffer_count, 100);
        s.set_compensation(10, 1000).unwrap();
        assert!(s.resample.is_some());
        assert_eq!(s.in_buffer_count, 0, "re-init cleared the buffer");
        assert!(s.options.flags.contains(SwrFlags::RESAMPLE));
        // The EINVAL guards.
        assert!(s.set_compensation(1, -1).is_err());
        assert!(s.set_compensation(1, 0).is_err());
        assert!(s.set_compensation(0, 0).is_ok());
    }
}
