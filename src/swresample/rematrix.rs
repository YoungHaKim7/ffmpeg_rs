//! Channel rematrixing — port of `libswresample/rematrix.c` (879 lines) plus
//! the five instantiations of `rematrix_template.c`.
//!
//! ## C → Rust map
//!
//! | C site | here |
//! |---|---|
//! | channel-id defines `rematrix.c:45-69` | private `const FRONT_LEFT … BOTTOM_FRONT_RIGHT` (values = [`Channel`] discriminants) |
//! | `even()` `rematrix.c:93-97` | [`even`] |
//! | `clean_layout()` `rematrix.c:99-112` | [`clean_layout`] (the `av_channel_layout_copy` failure arm is unrepresentable — layouts are `Copy`) |
//! | `sane_layout()` `rematrix.c:114-149` | [`sane_layout`] (`pub`: the future swresample-core guard at `swresample.c:340-346` reuses it) |
//! | `build_matrix()` `rematrix.c:151-570` | [`build_matrix`] — the per-unaccounted-channel decision table, max-row-L1 normalization, `rematrix_volume` scaling |
//! | `swr_build_matrix2()` `rematrix.c:572-652` (public API `swresample.h:380-404`) | [`swr_build_matrix2`] |
//! | `swr_set_matrix()` `rematrix.c:71-91` | [`swr_set_matrix`] → [`CustomRematrix`] (the `s->in_convert` guard at `:75` becomes the `ctx_already_initialized` flag owned by the core context) |
//! | `swri_check_chlayout()` `swresample.c:33-44` (only caller of the above) | [`check_chlayout`] |
//! | `auto_matrix()` `rematrix.c:654-671` | private `auto_matrix` (inside [`RematrixContext::init`]) |
//! | `swri_rematrix_init()` `rematrix.c:673-793` | [`RematrixContext::init`] |
//! | 17.15 quantization with error diffusion `rematrix.c:699-715` (s16p), `:748-760` (s32p) | [`RematrixContext::init`] using [`lrintf`] (`f64→f32` conversion first, then round-half-to-even = C `lrintf` under the default `FE_TONEAREST`) |
//! | mix-function selection `rematrix.c:716-725,726-736,737-747,748-765` | `clip_s16` + [`AnyFastPath`] + [`NativeMatrix`] fields of [`RematrixContext`] |
//! | `matrix_ch`/`matrix32`/`matrix_flt` fill `rematrix.c:768-786` | same fields (the C union `swresample_internal.h:172-177` becomes two arrays; reading the invalid one is C UB, here it is stale zeros) |
//! | `swri_rematrix_free()` `rematrix.c:795-798` | nothing — Rust ownership replaces `av_freep` |
//! | `swri_rematrix()` `rematrix.c:800-879` | [`RematrixContext::rematrix`] |
//! | `copy`/`sum2`/`mix6to2`/`mix8to2` × {s16, s16+clip, s32, float, double} `rematrix_template.c:52-127` | [`copy_s16`] … [`mix8to2_double`] — kernels over byte slices, little-endian (crate LE policy, cf. `util::pixfmt`) |
//! | `enum AVMatrixEncoding` `channel_layout.h:273-282` | [`MatrixEncoding`] — defined here, `util::channel_layout` explicitly left it to the rematrix phase (its module doc, "Not ported" list) |
//!
//! ## Fixed-point format (the load-bearing numbers)
//!
//! * Integer internal formats store the matrix in **17.15** fixed point:
//!   `coeff = lrintf(x * 32768)` with the f32 conversion noted above, and the
//!   apply rounds `(v + 16384) >> 15` (arithmetic shift; `rematrix_template.c:38,41,45`).
//! * `native_matrix` (s16p `rematrix.c:699-715`, s32p `:748-760`) quantizes
//!   with **per-row error diffusion** (`rem += target - quantized`); there is
//!   no `REMATRIXED_SIGMA` in this tree (grep over `FFmpeg/` finds nothing —
//!   that is a libavresample-era name; the real options are `rematrix_volume`
//!   and `rematrix_maxval`, `options.c:57-64`). `matrix32` (`:781`) is the
//!   **same scale without diffusion**. `native_one.i = 32768`, float/double
//!   `one = 1.0`.
//! * When an s16p row's quantized |coeff| sum exceeds 32768
//!   (`rematrix.c:717`) the clip variants are selected (`av_clip_int16`
//!   after the shift); float paths never clip (`sum2_float`/`mix6to2_float`
//!   have no clamp — overflow to `inf` is possible with a custom matrix,
//!   C behaves the same, `rematrix_template.c:21-32`).
//! * S32P accumulates in `int64_t` (`INTER` `rematrix_template.c:48`) and
//!   truncates the shifted `i64` back to `i32` on store (C implementation-
//!   defined truncation; Rust `as i32` wraps identically).
//!
//! ## Not ported (guards cited)
//!
//! * **x86 SIMD** (`swri_rematrix_init_x86` `rematrix.c:788-790`,
//!   `native_simd_matrix`/`native_simd_one`, `mix_1_1_simd`/`mix_2_1_simd`
//!   and their `len&~15`/`off` split at `:810-813`, `:827-830`, `:840-845`):
//!   a C build without x86asm selects exactly the C functions ported here;
//!   the SIMD split degenerates to `len1 = 0` ⇒ one full-length scalar call,
//!   which is what [`RematrixContext::rematrix`] does.
//! * **`AV_CHANNEL_ORDER_CUSTOM`** (`rematrix.c:163-183`, the `AV_CHAN_UNUSED`
//!   row/column clearing): unrepresentable — `util::channel_layout` did not
//!   port custom orders (its module doc). Unreachable, not merely skipped:
//!   `sane_layout` (`:117-125`) accepts only native (or coerced mono) layouts
//!   by the time `build_matrix` runs, and `swr_set_matrix` zero-fills first
//!   (`:80`).
//! * **S64P internal format**: `swr_init` accepts it (`swresample.c:279-283`)
//!   but `swri_rematrix_init` aborts (`av_assert0(0)` `rematrix.c:766`) —
//!   [`Error::Unsupported`] here. Any other non-{s16p,fltp,dblp,s32p}
//!   internal format hits the same assert.
//! * **`rematrix_dither`/`maxrematrix`**: do not exist in this tree (grep
//!   over `FFmpeg/` returns nothing). Dither is a separate `DitherContext`
//!   applied *after* rematrix in `swr_convert`, reusing this module's
//!   [`sum2_s16`]/[`sum2_s32`]/[`sum2_float`]/[`sum2_double`] with
//!   `native_one` (`swresample.c:708-728`) — hence those kernels and the
//!   [`NativeMatrix`] accessors are `pub(crate)`.
//!
//! ## Replicated upstream bug (documented, tested)
//!
//! The generic ≥3-tap **integer** branch (`rematrix.c:866-875`) casts the
//! planes to `int16_t*` regardless of the actual sample width: for **s32p**
//! buffers it therefore reads only the low 16 bits of every `i32` sample and
//! writes only the low 16 bits of the output `i32`, leaving its upper 16
//! bits stale. Present since the 2011 original commit; unreachable for
//! 5.1/7.1→stereo (the `mix6to2`/`mix8to2` fast paths) but reachable for
//! e.g. `s32p` 5.1→mono. This port replicates the arithmetic exactly (low
//! halves read via `(sample as i16)` on LE) and writes the result
//! **sign-extended into the full `i32`** — C's upper half is indeterminate
//! buffer garbage, ours is deterministic. Pinned by
//! `apply_generic_s32p_upstream_bug`.
//!
//! ## Divergences from C (each pinned by tests or cited at its guard)
//!
//! * Plane **steal** (`out->ch[out_i] = in->ch[in_i]`, `rematrix.c:834`):
//!   C re-points one channel plane; [`AudioData`] has a single `Arc<[u8]>`
//!   backing store, so the steal is performed **whole-buffer** (`Arc` clone,
//!   copy-on-write on later writes) and only when every output row is a
//!   1.0-coefficient identity copy — otherwise a per-plane byte copy runs.
//!   Observably identical whenever `out` and `in` are distinct buffers
//!   (the only legal Rust call shape; C's in-place aliasing is unportable).
//! * C `av_assert0(0)` else-arms in `build_matrix` are unreachable after
//!   `sane_layout` (every sane output has FL+FR, or FC, or both); they become
//!   `unreachable!()` with the C line cited.
//! * `av_assert0`s at `rematrix.c:815-816` become `debug_assert!`s; the
//!   plane-capacity UB of the C kernels becomes [`Error::BufferTooSmall`].
//! * The debug matrix dump (`rematrix.c:635-644`, `:684-698`) composes each
//!   C output *line* into one `log_debug!` call (the log sink is line-based);
//!   the rendered text is identical, including `%f` → `{:.6}`.

use crate::{
    util::{
        channel_layout::{Channel, ChannelLayout, Order},
        error::{Error, Result},
        samplefmt::SampleFormat,
    },
    {log_debug, log_error, log_verbose, log_warning},
};

use super::{AudioData, SWR_CH_MAX};

// ---------------------------------------------------------------------------
// Constants — rematrix.c:45-69 + swresample_internal.h:28-32
// ---------------------------------------------------------------------------

/// `NUM_NAMED_CHANNELS` (`rematrix.c:69`): channel ids `0..=40` have mixing
/// rules; ids 41..63 (`SIDE_SURROUND` … `BINAURAL`) get identity passthrough
/// only (`rematrix.c:552-555`).
const NUM_NAMED_CHANNELS: usize = 41;

/// `M_SQRT1_2` (`libavutil/mathematics.h`) — the **literal** C value
/// (`0x3FE6A09E667F3BCD`). Must not be computed as `1.0 / SQRT_2`:
/// that is 1 ulp lower (`0x3FE6A09E667F3BCC`) and would poison every
/// default coefficient. Rust has no `SQRT_1_2` const.
pub const M_SQRT1_2: f64 = 0.70710678118654752440;

/// `SQRT1_3` (`swresample_internal.h:30`) — `sqrt(1/3)`, the C literal.
const SQRT1_3: f64 = 0.57735026918962576451;

/// `SQRT2_3` (`swresample_internal.h:31`) — `sqrt(2/3)`, the C literal.
const SQRT2_3: f64 = 0.81649658092772603273;

/// `SQRT3_2` (`swresample_internal.h:32`) — `sqrt(3/2)`, the C literal.
const SQRT3_2: f64 = 1.22474487139158904909;

// `rematrix.c:45-68` — channel-id defines. Values are the `Channel`
// discriminants (`channel_layout.h:47-112`); kept as plain constants so the
// decision table below reads like the C.
const FRONT_LEFT: usize = Channel::FrontLeft as i32 as usize; // 0
const FRONT_RIGHT: usize = Channel::FrontRight as i32 as usize; // 1
const FRONT_CENTER: usize = Channel::FrontCenter as i32 as usize; // 2
const LOW_FREQUENCY: usize = Channel::LowFrequency as i32 as usize; // 3
const BACK_LEFT: usize = Channel::BackLeft as i32 as usize; // 4
const BACK_RIGHT: usize = Channel::BackRight as i32 as usize; // 5
const FRONT_LEFT_OF_CENTER: usize = Channel::FrontLeftOfCenter as i32 as usize; // 6
const FRONT_RIGHT_OF_CENTER: usize = Channel::FrontRightOfCenter as i32 as usize; // 7
const BACK_CENTER: usize = Channel::BackCenter as i32 as usize; // 8
const SIDE_LEFT: usize = Channel::SideLeft as i32 as usize; // 9
const SIDE_RIGHT: usize = Channel::SideRight as i32 as usize; // 10
const TOP_CENTER: usize = Channel::TopCenter as i32 as usize; // 11
const TOP_FRONT_LEFT: usize = Channel::TopFrontLeft as i32 as usize; // 12
const TOP_FRONT_CENTER: usize = Channel::TopFrontCenter as i32 as usize; // 13
const TOP_FRONT_RIGHT: usize = Channel::TopFrontRight as i32 as usize; // 14
const TOP_BACK_LEFT: usize = Channel::TopBackLeft as i32 as usize; // 15
const TOP_BACK_CENTER: usize = Channel::TopBackCenter as i32 as usize; // 16
const TOP_BACK_RIGHT: usize = Channel::TopBackRight as i32 as usize; // 17
const LOW_FREQUENCY_2: usize = Channel::LowFrequency2 as i32 as usize; // 35
const TOP_SIDE_LEFT: usize = Channel::TopSideLeft as i32 as usize; // 36
const TOP_SIDE_RIGHT: usize = Channel::TopSideRight as i32 as usize; // 37
const BOTTOM_FRONT_CENTER: usize = Channel::BottomFrontCenter as i32 as usize; // 38
const BOTTOM_FRONT_LEFT: usize = Channel::BottomFrontLeft as i32 as usize; // 39
const BOTTOM_FRONT_RIGHT: usize = Channel::BottomFrontRight as i32 as usize; // 40

/// `AV_CH_LAYOUT_STEREO` = `FL|FR` (`channel_layout.h:218`).
const CH_LAYOUT_STEREO: u64 = ch_bit(Channel::FrontLeft) | ch_bit(Channel::FrontRight);

/// `AV_CH_LAYOUT_SURROUND` = `STEREO|FC` (`channel_layout.h:221`).
const CH_LAYOUT_SURROUND: u64 = CH_LAYOUT_STEREO | ch_bit(Channel::FrontCenter);

/// `AV_CH_LAYOUT_STEREO_DOWNMIX` = `DL|DR` (`channel_layout.h:255`).
const CH_LAYOUT_STEREO_DOWNMIX: u64 = ch_bit(Channel::StereoLeft) | ch_bit(Channel::StereoRight);

/// `INT_MAX` as the double C compares `maxcoef` against (`rematrix.c:664`).
const INT_MAX_F64: f64 = 2147483647.0;

/// `AV_CH_FOO` bit of a channel (`channel_layout.h:175-210`).
const fn ch_bit(c: Channel) -> u64 {
    1u64 << (c as i32)
}

// ---------------------------------------------------------------------------
// RematrixContext — swri_rematrix_init (rematrix.c:673-793) + swri_rematrix
// (rematrix.c:800-879)
// ---------------------------------------------------------------------------

/// `s->native_matrix` + `s->native_one` (`swresample_internal.h:180-186`):
/// the quantized matrix, row-major `[out][in]` with row stride
/// `used_ch_layout.nb_channels` (`rematrix.c:675`), plus the unit
/// coefficient of the format (int: 32768, float/double: 1.0,
/// `rematrix.c:716,733,744,761`).
///
/// The typed accessors are `pub(crate)` for the dither port, which reuses
/// this module's [`sum2_*`] kernels with `one` (`swresample.c:708-728`).
#[derive(Clone, Debug)]
pub enum NativeMatrix {
    /// `int` flavor (s16p/s32p), 17.15 with error diffusion.
    Int {
        /// `(int*)s->native_matrix`.
        coeffs: Vec<i32>,
        /// `s->native_one.i` = 32768.
        one: i32,
    },
    /// `float` flavor (fltp).
    Float {
        /// `(float*)s->native_matrix`.
        coeffs: Vec<f32>,
        /// `s->native_one.f` = 1.0.
        one: f32,
    },
    /// `double` flavor (dblp).
    Double {
        /// `(double*)s->native_matrix`.
        coeffs: Vec<f64>,
        /// `s->native_one.d` = 1.0.
        one: f64,
    },
}

// The typed accessors are exercised by this module's tests today and by
// the dither port (swresample.c:708-728) once it lands.
#[allow(dead_code)]
impl NativeMatrix {
    /// The `Int` flavor's coefficients and `native_one.i`, if this is it.
    pub(crate) fn as_int(&self) -> Option<(&[i32], i32)> {
        if let NativeMatrix::Int { coeffs, one } = self {
            Some((coeffs, *one))
        } else {
            None
        }
    }

    /// The `Float` flavor's coefficients and `native_one.f`.
    pub(crate) fn as_float(&self) -> Option<(&[f32], f32)> {
        if let NativeMatrix::Float { coeffs, one } = self {
            Some((coeffs, *one))
        } else {
            None
        }
    }

    /// The `Double` flavor's coefficients and `native_one.d`.
    pub(crate) fn as_double(&self) -> Option<(&[f64], f64)> {
        if let NativeMatrix::Double { coeffs, one } = self {
            Some((coeffs, *one))
        } else {
            None
        }
    }
}

/// `s->mix_any_f` (`rematrix_template.c:108-127`) — the whole-buffer fast
/// path; the C per-format function-pointer duplication collapses to this
/// tag dispatched on `int_fmt` (+`clip_s16` for the s16 pair).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnyFastPath {
    /// `mix6to2` — 5.1 (side or back) → stereo.
    Mix6to2,
    /// `mix8to2` — 7.1 → stereo.
    Mix8to2,
}

/// Every rematrix field of C's `SwrContext` (`swresample_internal.h:144-196`
/// subset). Built by [`RematrixContext::init`] where C calls
/// `swri_rematrix_init` (`swresample.c:413-415`); the future swresample-core
/// `SwrContext` embeds it and calls [`RematrixContext::rematrix`] at
/// `swresample.c:676/679`. `swri_rematrix_free` (`rematrix.c:795-798`) is
/// plain ownership — no `Drop` work.
#[derive(Clone, Debug)]
pub struct RematrixContext {
    /// `s->matrix` — floating-point coefficients, row-major `[out][in]`.
    matrix: [[f64; SWR_CH_MAX]; SWR_CH_MAX],
    /// `s->matrix_flt` — valid iff `int_fmt == Fltp` (C union,
    /// `swresample_internal.h:172-177`).
    matrix_flt: [[f32; SWR_CH_MAX]; SWR_CH_MAX],
    /// `s->matrix32` — 17.15, valid iff `int_fmt` ∉ {Fltp, Dblp} (no
    /// diffusion, `rematrix.c:781`).
    matrix32: [[i32; SWR_CH_MAX]; SWR_CH_MAX],
    /// `s->native_matrix` + `s->native_one`.
    native_matrix: NativeMatrix,
    /// `s->matrix_ch` (`rematrix.c:768-786`): `[out][0]` = tap count,
    /// `[out][1..=count]` = input channel indices with `matrix != 0.0`,
    /// increasing.
    matrix_ch: [[u8; SWR_CH_MAX + 1]; SWR_CH_MAX],
    /// `s->midbuf.fmt == s->int_sample_fmt`.
    int_fmt: SampleFormat,
    /// s16p with a quantized row |coeff| sum > 32768 (`rematrix.c:717`) —
    /// selects the `av_clip_int16` kernel pair.
    clip_s16: bool,
    /// `s->mix_any_f`.
    mix_any: Option<AnyFastPath>,
    /// `used_ch_layout.nb_channels` — the row stride of `native_matrix`
    /// (`rematrix.c:675`). C indexes it with `in->ch_count` at `:830/:841`;
    /// `swr_init` keeps the two equal (`swresample.c:335-337`).
    in_nb: usize,
}

impl RematrixContext {
    /// `swri_rematrix_init` (`rematrix.c:673-793`). Field mapping of the C
    /// context the body reads:
    ///
    /// | parameter | C field |
    /// |---|---|
    /// | `in_ch_layout` | `s->in_ch_layout` (auto_matrix + fast-path checks) |
    /// | `used_in_nb_channels` | `used_ch_layout.nb_channels` — `nb_in`, the `native_matrix` row stride (`:675`) |
    /// | `out_ch_layout` | `s->out_ch_layout` |
    /// | `out_nb_channels` | `s->out.ch_count` — `nb_out` (`:676`) |
    /// | `out_sample_fmt` | `s->out_sample_fmt` (auto_matrix `maxval`) |
    /// | `int_sample_fmt` | `s->int_sample_fmt` == `s->midbuf.fmt` (quantization switch `:699,726,737,748`; `matrix32` switch `:774`) |
    /// | `options` | `clev`/`slev`/`lfe_mix_level`/`rematrix_volume`/`rematrix_maxval`/`matrix_encoding` |
    /// | `custom` | `s->rematrix_custom` + `s->matrix` (set earlier by [`swr_set_matrix`]) |
    /// | `log_ctx` | `s` (av_log context) |
    ///
    /// The final x86 dispatch (`:788-790`) is not ported (module doc).
    pub fn init(
        in_ch_layout: &ChannelLayout,
        used_in_nb_channels: usize,
        out_ch_layout: &ChannelLayout,
        out_nb_channels: usize,
        out_sample_fmt: SampleFormat,
        int_sample_fmt: SampleFormat,
        options: &RematrixOptions,
        custom: Option<&CustomRematrix>,
        log_ctx: Option<&str>,
    ) -> Result<Self> {
        let nb_in = used_in_nb_channels;
        let nb_out = out_nb_channels;

        // rematrix.c:678 — s->mix_any_f = NULL; every quantization branch
        // below re-selects it (get_mix_any_func returns NULL -> None).
        let mix_any;

        let mut matrix = [[0.0f64; SWR_CH_MAX]; SWR_CH_MAX];
        if let Some(custom) = custom {
            // rematrix.c:684-698 — custom matrix used verbatim (no auto
            // build, no maxval/volume handling), DEBUG dump.
            log_debug!(log_ctx, "Custom matrix coefficients:");
            for i in 0..out_ch_layout.nb_channels {
                let mut line = format!(
                    "{}: ",
                    out_ch_layout
                        .channel_from_index(i)
                        .unwrap_or(Channel::None)
                        .name()
                );
                for j in 0..in_ch_layout.nb_channels {
                    line.push_str(&format!(
                        "{}:{:.6} ",
                        in_ch_layout
                            .channel_from_index(j)
                            .unwrap_or(Channel::None)
                            .name(),
                        custom.matrix[i][j]
                    ));
                }
                log_debug!(log_ctx, "{line}");
            }
            matrix = custom.matrix;
        } else {
            // rematrix.c:680-683.
            auto_matrix(
                &mut matrix,
                in_ch_layout,
                out_ch_layout,
                out_sample_fmt,
                int_sample_fmt,
                options,
                log_ctx,
            )?;
        }

        // rematrix.c:699-766 — quantize into native_matrix, select the mix
        // function set (and the s16 clip variants on maxsum > 32768).
        let native_matrix;
        let mut clip_s16 = false;
        match int_sample_fmt {
            SampleFormat::S16p => {
                // rematrix.c:699-725 — 17.15 with per-row error diffusion.
                let mut coeffs = vec![0i32; nb_in * nb_out];
                let mut maxsum = 0i32;
                for i in 0..nb_out {
                    let mut rem = 0.0f64;
                    let mut sum = 0i32;
                    for j in 0..nb_in {
                        let target = matrix[i][j] * 32768.0 + rem;
                        let q = lrintf(target);
                        coeffs[i * nb_in + j] = q;
                        rem += target - f64::from(q);
                        sum = sum.wrapping_add(q.abs());
                    }
                    maxsum = maxsum.max(sum);
                }
                clip_s16 = maxsum > 32768;
                mix_any = get_mix_any_func(&matrix, in_ch_layout, out_ch_layout);
                native_matrix = NativeMatrix::Int {
                    coeffs,
                    one: 32768, // rematrix.c:716
                };
            }
            SampleFormat::Fltp => {
                // rematrix.c:726-736.
                let mut coeffs = vec![0f32; nb_in * nb_out];
                for i in 0..nb_out {
                    for j in 0..nb_in {
                        coeffs[i * nb_in + j] = matrix[i][j] as f32;
                    }
                }
                mix_any = get_mix_any_func(&matrix, in_ch_layout, out_ch_layout);
                native_matrix = NativeMatrix::Float {
                    coeffs,
                    one: 1.0, // rematrix.c:733
                };
            }
            SampleFormat::Dblp => {
                // rematrix.c:737-747.
                let mut coeffs = vec![0f64; nb_in * nb_out];
                for i in 0..nb_out {
                    for j in 0..nb_in {
                        coeffs[i * nb_in + j] = matrix[i][j];
                    }
                }
                mix_any = get_mix_any_func(&matrix, in_ch_layout, out_ch_layout);
                native_matrix = NativeMatrix::Double {
                    coeffs,
                    one: 1.0, // rematrix.c:744
                };
            }
            SampleFormat::S32p => {
                // rematrix.c:748-764 — same diffusion as s16, no clip check.
                let mut coeffs = vec![0i32; nb_in * nb_out];
                for i in 0..nb_out {
                    let mut rem = 0.0f64;
                    for j in 0..nb_in {
                        let target = matrix[i][j] * 32768.0 + rem;
                        let q = lrintf(target);
                        coeffs[i * nb_in + j] = q;
                        rem += target - f64::from(q);
                    }
                }
                mix_any = get_mix_any_func(&matrix, in_ch_layout, out_ch_layout);
                native_matrix = NativeMatrix::Int {
                    coeffs,
                    one: 32768, // rematrix.c:761
                };
            }
            // rematrix.c:765-766 — av_assert0(0). swr_init accepts s64p
            // (swresample.c:279-283) but rematrixing aborts here.
            _ => {
                return Err(Error::Unsupported(format!(
                    "rematrixing with internal sample format {} (C: av_assert0(0), rematrix.c:766)",
                    int_sample_fmt.name()
                )));
            }
        }

        // rematrix.c:768-786 — matrix_ch sparsity + matrix_flt/matrix32.
        let mut matrix_ch = [[0u8; SWR_CH_MAX + 1]; SWR_CH_MAX];
        let mut matrix_flt = [[0f32; SWR_CH_MAX]; SWR_CH_MAX];
        let mut matrix32 = [[0i32; SWR_CH_MAX]; SWR_CH_MAX];
        for i in 0..SWR_CH_MAX {
            let mut ch_in = 0usize;
            for j in 0..SWR_CH_MAX {
                let coeff = matrix[i][j];
                if coeff != 0.0 {
                    ch_in += 1;
                    matrix_ch[i][ch_in] = j as u8;
                }
                match int_sample_fmt {
                    SampleFormat::Fltp => matrix_flt[i][j] = coeff as f32,
                    SampleFormat::Dblp => {}
                    _ => matrix32[i][j] = lrintf(coeff * 32768.0),
                }
            }
            matrix_ch[i][0] = ch_in as u8;
        }

        Ok(RematrixContext {
            matrix,
            matrix_flt,
            matrix32,
            native_matrix,
            matrix_ch,
            int_fmt: int_sample_fmt,
            clip_s16,
            mix_any,
            in_nb: nb_in,
        })
    }

    /// `s->int_sample_fmt`.
    pub fn int_fmt(&self) -> SampleFormat {
        self.int_fmt
    }

    /// `s->native_matrix` + `native_one` (for the dither port,
    /// `swresample.c:708-728`; exercised by tests until then).
    #[allow(dead_code)]
    pub(crate) fn native_matrix(&self) -> &NativeMatrix {
        &self.native_matrix
    }

    /// `swri_rematrix` (`rematrix.c:800-879`): remix `len` samples per
    /// channel from `input` into `out`.
    ///
    /// `mustcopy` is C's parameter (`preout==out` / `midbuf==out` at
    /// `swresample.c:676,679`): it forbids the plane steal and turns empty
    /// output rows into explicit zeroing (`:821-822`).
    ///
    /// Divergences (module doc): whole-buffer `Arc` steal instead of
    /// C's per-channel pointer steal; [`Error::BufferTooSmall`] instead of
    /// OOB UB; the `av_assert0`s at `:815-816` are `debug_assert!`s. The
    /// buffers must be planar — C only ever passes the planar internal
    /// formats (`swresample.c:405-411`).
    pub fn rematrix(
        &self,
        out: &mut AudioData,
        input: &AudioData,
        len: usize,
        mustcopy: bool,
    ) -> Result<()> {
        if !input.planar || !out.planar {
            return Err(Error::InvalidArgument(
                "swri_rematrix requires planar internal-format buffers (swresample.c:405-411)"
                    .into(),
            ));
        }
        // rematrix.c:815-816, plus the stride identity of rematrix.c:675
        // (used_ch_layout.nb == in.ch_count, enforced by swr_init).
        debug_assert_eq!(input.ch_count, self.in_nb);

        // rematrix.c:805-808 — the whole-buffer fast path covers every
        // output channel (get_mix_any_func guarantees a stereo output).
        if let Some(fast) = self.mix_any {
            return self.apply_mix_any(fast, out, input, len);
        }

        // rematrix.c:834 — out->ch[out_i] = in->ch[in_i]. C steals one plane
        // per identity row; the single-`Arc` AudioData can only steal the
        // whole buffer, so that happens iff EVERY row is an identity 1.0
        // copy (and the geometries match). Otherwise identity rows take the
        // byte-copy road — observably identical for distinct buffers.
        let steal_all = !mustcopy
            && out.ch_count == input.ch_count
            && out.count == input.count
            && out.bps == input.bps
            && out.fmt == input.fmt
            && (0..out.ch_count).all(|i| {
                self.matrix_ch[i][0] == 1
                    && self.matrix_ch[i][1] as usize == i
                    && self.matrix[i][i] == 1.0
            });
        if steal_all {
            out.data = input.data.clone();
            return Ok(());
        }

        let bps = self.int_fmt.bytes_per_sample();
        for out_i in 0..out.ch_count {
            match self.matrix_ch[out_i][0] {
                // rematrix.c:820-823 — no inputs: zero on mustcopy, else
                // leave the plane untouched.
                0 => {
                    if mustcopy {
                        let plane = out.plane_bytes_mut(out_i).ok_or(Error::BufferTooSmall)?;
                        let n = len * bps;
                        if plane.len() < n {
                            return Err(Error::BufferTooSmall);
                        }
                        plane[..n].fill(0);
                    }
                }
                // rematrix.c:824-836 — one input.
                1 => {
                    let in_i = self.matrix_ch[out_i][1] as usize;
                    if self.matrix[out_i][in_i] != 1.0 {
                        // rematrix.c:827-830 — mix_1_1_f (copy with coeff).
                        let idx = input.ch_count * out_i + in_i;
                        let src = input.plane(in_i).ok_or(Error::BufferTooSmall)?;
                        let dst = out.plane_bytes_mut(out_i).ok_or(Error::BufferTooSmall)?;
                        if src.len() < len * bps || dst.len() < len * bps {
                            return Err(Error::BufferTooSmall);
                        }
                        match (&self.native_matrix, self.int_fmt, self.clip_s16) {
                            (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, false) => {
                                copy_s16(dst, src, coeffs[idx], len)
                            }
                            (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, true) => {
                                copy_clip_s16(dst, src, coeffs[idx], len)
                            }
                            (NativeMatrix::Int { coeffs, .. }, SampleFormat::S32p, _) => {
                                copy_s32(dst, src, coeffs[idx], len)
                            }
                            (NativeMatrix::Float { coeffs, .. }, SampleFormat::Fltp, _) => {
                                copy_float(dst, src, coeffs[idx], len)
                            }
                            (NativeMatrix::Double { coeffs, .. }, SampleFormat::Dblp, _) => {
                                copy_double(dst, src, coeffs[idx], len)
                            }
                            _ => unreachable!("init only yields these format/coeff pairs"),
                        }
                    } else {
                        // rematrix.c:831-835 — memcpy on mustcopy, steal
                        // otherwise (see steal_all above).
                        let n = len * out.bps;
                        let src = input.plane(in_i).ok_or(Error::BufferTooSmall)?;
                        let dst = out.plane_bytes_mut(out_i).ok_or(Error::BufferTooSmall)?;
                        if src.len() < n || dst.len() < n {
                            return Err(Error::BufferTooSmall);
                        }
                        dst[..n].copy_from_slice(&src[..n]);
                    }
                }
                // rematrix.c:837-846 — two inputs: mix_2_1_f.
                2 => {
                    let in_i1 = self.matrix_ch[out_i][1] as usize;
                    let in_i2 = self.matrix_ch[out_i][2] as usize;
                    let idx1 = input.ch_count * out_i + in_i1;
                    let idx2 = input.ch_count * out_i + in_i2;
                    let src1 = input.plane(in_i1).ok_or(Error::BufferTooSmall)?;
                    let src2 = input.plane(in_i2).ok_or(Error::BufferTooSmall)?;
                    let dst = out.plane_bytes_mut(out_i).ok_or(Error::BufferTooSmall)?;
                    if src1.len() < len * bps || src2.len() < len * bps || dst.len() < len * bps {
                        return Err(Error::BufferTooSmall);
                    }
                    match (&self.native_matrix, self.int_fmt, self.clip_s16) {
                        (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, false) => {
                            sum2_s16(dst, src1, src2, coeffs[idx1], coeffs[idx2], len)
                        }
                        (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, true) => {
                            sum2_clip_s16(dst, src1, src2, coeffs[idx1], coeffs[idx2], len)
                        }
                        (NativeMatrix::Int { coeffs, .. }, SampleFormat::S32p, _) => {
                            sum2_s32(dst, src1, src2, coeffs[idx1], coeffs[idx2], len)
                        }
                        (NativeMatrix::Float { coeffs, .. }, SampleFormat::Fltp, _) => {
                            sum2_float(dst, src1, src2, coeffs[idx1], coeffs[idx2], len)
                        }
                        (NativeMatrix::Double { coeffs, .. }, SampleFormat::Dblp, _) => {
                            sum2_double(dst, src1, src2, coeffs[idx1], coeffs[idx2], len)
                        }
                        _ => unreachable!("init only yields these format/coeff pairs"),
                    }
                }
                // rematrix.c:847-875 — generic ≥3-tap path.
                _ => {
                    self.apply_generic(out, out_i, input, len, bps)?;
                }
            }
        }
        Ok(())
    }

    /// The `s->mix_any_f` call (`rematrix.c:805-808` +
    /// `rematrix_template.c:78-106`). `out` must be stereo (guaranteed by
    /// `get_mix_any_func`).
    fn apply_mix_any(
        &self,
        fast: AnyFastPath,
        out: &mut AudioData,
        input: &AudioData,
        len: usize,
    ) -> Result<()> {
        debug_assert_eq!(out.ch_count, 2);
        let need = len * self.int_fmt.bytes_per_sample();
        if input.ch_count < 6 || input.plane(0).map_or(true, |p| p.len() < need) {
            return Err(Error::BufferTooSmall);
        }
        let (o0, o1) = two_planes_mut(out, 0, 1).ok_or(Error::BufferTooSmall)?;
        if o0.len() < need || o1.len() < need {
            return Err(Error::BufferTooSmall);
        }
        match fast {
            AnyFastPath::Mix6to2 => {
                debug_assert_eq!(
                    input.ch_count, 6,
                    "coeff stride is 6 (rematrix_template.c:87)"
                );
                match (&self.native_matrix, self.int_fmt, self.clip_s16) {
                    (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, false) => {
                        mix6to2_s16(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, true) => {
                        mix6to2_clip_s16(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Int { coeffs, .. }, SampleFormat::S32p, _) => {
                        mix6to2_s32(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Float { coeffs, .. }, SampleFormat::Fltp, _) => {
                        mix6to2_float(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Double { coeffs, .. }, SampleFormat::Dblp, _) => {
                        mix6to2_double(o0, o1, input, coeffs, len)
                    }
                    _ => unreachable!("init only yields these format/coeff pairs"),
                }
            }
            AnyFastPath::Mix8to2 => {
                debug_assert_eq!(
                    input.ch_count, 8,
                    "coeff stride is 8 (rematrix_template.c:102)"
                );
                if input.ch_count < 8 {
                    return Err(Error::BufferTooSmall);
                }
                match (&self.native_matrix, self.int_fmt, self.clip_s16) {
                    (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, false) => {
                        mix8to2_s16(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Int { coeffs, .. }, SampleFormat::S16p, true) => {
                        mix8to2_clip_s16(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Int { coeffs, .. }, SampleFormat::S32p, _) => {
                        mix8to2_s32(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Float { coeffs, .. }, SampleFormat::Fltp, _) => {
                        mix8to2_float(o0, o1, input, coeffs, len)
                    }
                    (NativeMatrix::Double { coeffs, .. }, SampleFormat::Dblp, _) => {
                        mix8to2_double(o0, o1, input, coeffs, len)
                    }
                    _ => unreachable!("init only yields these format/coeff pairs"),
                }
            }
        }
        Ok(())
    }

    /// The generic ≥3-tap branch (`rematrix.c:847-875`): per-sample
    /// multiply-accumulate over `matrix_ch[out_i]`, format by format. The
    /// integer arm replicates the documented upstream bug (module doc):
    /// samples are read as **i16** (the low half of s32 samples on LE) and
    /// the `(v+16384)>>15` result is stored sign-extended — C stores only
    /// the low 16 bits of the output slot, leaving the upper half stale.
    fn apply_generic(
        &self,
        out: &mut AudioData,
        out_i: usize,
        input: &AudioData,
        len: usize,
        bps: usize,
    ) -> Result<()> {
        let taps = self.matrix_ch[out_i][0] as usize;
        let need = len * bps;
        let dst = out.plane_bytes_mut(out_i).ok_or(Error::BufferTooSmall)?;
        if dst.len() < need {
            return Err(Error::BufferTooSmall);
        }
        let srcs: Vec<&[u8]> = (0..taps)
            .map(|j| input.plane(self.matrix_ch[out_i][1 + j] as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::BufferTooSmall)?;
        if srcs.iter().any(|p| p.len() < need) {
            return Err(Error::BufferTooSmall);
        }

        match self.int_fmt {
            // rematrix.c:848-856 — float, matrix_flt, f32 accumulation.
            SampleFormat::Fltp => {
                for i in 0..len {
                    let mut v = 0.0f32;
                    for j in 0..taps {
                        let in_i = self.matrix_ch[out_i][1 + j] as usize;
                        v += ld_f32(srcs[j], i) * self.matrix_flt[out_i][in_i];
                    }
                    st_f32(dst, i, v);
                }
            }
            // rematrix.c:857-865 — double, matrix, f64 accumulation.
            SampleFormat::Dblp => {
                for i in 0..len {
                    let mut v = 0.0f64;
                    for j in 0..taps {
                        let in_i = self.matrix_ch[out_i][1 + j] as usize;
                        v += ld_f64(srcs[j], i) * self.matrix[out_i][in_i];
                    }
                    st_f64(dst, i, v);
                }
            }
            // rematrix.c:866-875 — integer: matrix32, i32 accumulation of
            // i16-READ samples (the upstream bug), (v+16384)>>15 store.
            _ => {
                for i in 0..len {
                    let mut v = 0i32;
                    for j in 0..taps {
                        let in_i = self.matrix_ch[out_i][1 + j] as usize;
                        v = v.wrapping_add(
                            (ld_i16(srcs[j], i) as i32).wrapping_mul(self.matrix32[out_i][in_i]),
                        );
                    }
                    let r = (v.wrapping_add(16384) >> 15) as i16;
                    if bps == 2 {
                        st_i16(dst, i, r);
                    } else {
                        // s32p: C writes ((int16_t*)out)[i] — the low half
                        // only, upper half indeterminate. Sign-extend here
                        // (deterministic; module doc "Replicated upstream
                        // bug").
                        st_i32(dst, i, r as i32);
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MatrixEncoding — channel_layout.h:273-282 (owned here; the channel_layout
// module left it to the rematrix phase)
// ---------------------------------------------------------------------------

/// `enum AVMatrixEncoding` (`channel_layout.h:273-282`) — exact C
/// discriminants. Only `None`/`Dolby`/`Dplii` are ever tested by rematrix
/// code (`rematrix.c:223-224,254,259,290,295`); the rest fall through to the
/// plain (else) branches exactly as in C.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MatrixEncoding {
    /// `AV_MATRIX_ENCODING_NONE` = 0.
    #[default]
    None,
    /// `AV_MATRIX_ENCODING_DOLBY` = 1.
    Dolby,
    /// `AV_MATRIX_ENCODING_DPLII` = 2.
    Dplii,
    /// `AV_MATRIX_ENCODING_DPLIIX` = 3 (untested by rematrix.c).
    Dpliix,
    /// `AV_MATRIX_ENCODING_DPLIIZ` = 4 (untested by rematrix.c).
    Dpliiz,
    /// `AV_MATRIX_ENCODING_DOLBYEX` = 5 (untested by rematrix.c).
    DolbyEx,
    /// `AV_MATRIX_ENCODING_DOLBYHEADPHONE` = 6 (untested by rematrix.c).
    DolbyHeadphone,
}

// ---------------------------------------------------------------------------
// RematrixOptions — the SwrContext option subset (swresample_internal.h
// 110-115), defaults from options.c:32,57-64,113
// ---------------------------------------------------------------------------

/// The rematrixing option fields of C's `SwrContext`. C stores them as
/// `float`; the option layer (AVOption `AV_OPT_TYPE_FLOAT`, `.dbl` defaults)
/// converts to `f32` on set — this port keeps `f64` and lets the future
/// options layer narrow, so the C defaults' `f64` literals survive exactly.
///
/// Range enforcement stays with the CLI/options layer (`options.c:57-64`):
/// `clev`/`slev`/`lfe_mix_level` ∈ [−32, 32], `rematrix_volume` ∈
/// [−1000, 1000], `rematrix_maxval` ∈ [0, 1000].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RematrixOptions {
    /// `s->clev` — center mix level. Default `C_30DB = M_SQRT1_2`
    /// (`options.c:32,57-58`).
    pub clev: f64,
    /// `s->slev` — surround mix level. Default `C_30DB = M_SQRT1_2`
    /// (`options.c:59-60`).
    pub slev: f64,
    /// `s->lfe_mix_level` — LFE mix level. Default 0 (`options.c:61`).
    pub lfe_mix_level: f64,
    /// `s->rematrix_volume` — output scaling. Default 1.0 (`options.c:62-63`).
    pub rematrix_volume: f64,
    /// `s->rematrix_maxval` — row L1 clamp for the double matrix; 0 = auto
    /// (1.0 for any integer stage, `INT_MAX` for all-float, `auto_matrix`
    /// `rematrix.c:658-664`). Default 0.0 (`options.c:64`).
    pub rematrix_maxval: f64,
    /// `s->matrix_encoding` (`options.c:113-116`).
    pub matrix_encoding: MatrixEncoding,
}

impl Default for RematrixOptions {
    /// C option defaults (`options.c:32,57-64,113`).
    fn default() -> Self {
        RematrixOptions {
            clev: M_SQRT1_2, // C_30DB
            slev: M_SQRT1_2, // C_30DB
            lfe_mix_level: 0.0,
            rematrix_volume: 1.0,
            rematrix_maxval: 0.0,
            matrix_encoding: MatrixEncoding::None,
        }
    }
}

// ---------------------------------------------------------------------------
// swr_set_matrix — rematrix.c:71-91
// ---------------------------------------------------------------------------

/// The state `swr_set_matrix` leaves in an allocated-but-uninitialized C
/// context: `s->matrix` (`memset` then filled, `rematrix.c:80-88`) plus the
/// user layout channel counts (`s->rematrix_custom = 1`, `:89`).
#[derive(Clone, Debug, PartialEq)]
pub struct CustomRematrix {
    /// `s->matrix` — row-major `[out][in]`, everything outside the user
    /// rectangle zero (the `memset` at `rematrix.c:80`).
    pub matrix: [[f64; SWR_CH_MAX]; SWR_CH_MAX],
    /// `user_in_chlayout.nb_channels`.
    pub in_nb_channels: usize,
    /// `user_out_chlayout.nb_channels`.
    pub out_nb_channels: usize,
}

/// `swri_check_chlayout` (`swresample.c:33-44`): `av_channel_layout_check`
/// plus the `SWR_CH_MAX` bound; on failure a WARNING naming the side and the
/// described layout (empty when the layout fails `check()`), then `EINVAL`.
fn check_chlayout(chl: &ChannelLayout, name: &str, log_ctx: Option<&str>) -> Result<()> {
    let ret = chl.check();
    if !ret || chl.nb_channels > SWR_CH_MAX {
        let described = if ret { chl.describe() } else { String::new() };
        log_warning!(
            log_ctx,
            "{name} channel layout \"{described}\" is invalid or unsupported."
        );
        return Err(Error::InvalidArgument(format!(
            "{name} channel layout \"{described}\" is invalid or unsupported."
        )));
    }
    Ok(())
}

/// `swr_set_matrix` (`rematrix.c:71-91`) — install a caller-provided matrix.
///
/// C's first guard `!s || s->in_convert` ("allocated but not initialized",
/// `:75`) becomes the [`ctx_already_initialized`] flag the future core
/// context owns (`swr_set_matrix` is legal only before `swr_init`). The
/// layout checks are `swri_check_chlayout` (`swresample.c:33-44`).
///
/// The `matrix` slice is read C-style: row `out` starts at
/// `matrix + out*stride` (`:84-88`); requires at least
/// `(nb_out−1)*stride + nb_in` elements.
pub fn swr_set_matrix(
    user_in_chlayout: &ChannelLayout,
    user_out_chlayout: &ChannelLayout,
    matrix: &[f64],
    stride: usize,
    log_ctx: Option<&str>,
    ctx_already_initialized: bool,
) -> Result<CustomRematrix> {
    // rematrix.c:75 — s needs to be allocated but not initialized.
    if ctx_already_initialized {
        return Err(Error::InvalidArgument(
            "swr_set_matrix: the context must be allocated but not initialized".into(),
        ));
    }
    check_chlayout(user_in_chlayout, "input", log_ctx)?;
    check_chlayout(user_out_chlayout, "output", log_ctx)?;

    let nb_in = user_in_chlayout.nb_channels;
    let nb_out = user_out_chlayout.nb_channels;
    if nb_out == 0 || matrix.len() < (nb_out - 1) * stride + nb_in {
        return Err(Error::BufferTooSmall);
    }

    // rematrix.c:80 — memset(s->matrix, 0, sizeof(s->matrix)).
    let mut m = [[0.0f64; SWR_CH_MAX]; SWR_CH_MAX];
    for out in 0..nb_out {
        for in_ in 0..nb_in {
            m[out][in_] = matrix[out * stride + in_];
        }
    }
    Ok(CustomRematrix {
        matrix: m,
        in_nb_channels: nb_in,
        out_nb_channels: nb_out,
    })
}

// ---------------------------------------------------------------------------
// even / clean_layout / sane_layout — rematrix.c:93-149
// ---------------------------------------------------------------------------

/// `even()` (`rematrix.c:93-97`). The name is misleading: it returns `true`
/// for 0 bits **and** for ≥2 bits — `false` only for exactly one set bit,
/// i.e. "no asymmetric pair" (`!even(mask & (L|R))` at `:131-146` rejects
/// layouts with one side of a pair).
const fn even(layout_mask: u64) -> bool {
    // if(!layout) return 1; if(layout&(layout-1)) return 1; return 0;
    layout_mask == 0 || (layout_mask & (layout_mask.wrapping_sub(1))) != 0
}

/// `clean_layout` (`rematrix.c:99-112`): a single-channel layout without a
/// front-center channel is treated as mono (VERBOSE log, `:106`); everything
/// else is copied verbatim. C's `av_channel_layout_copy` failure
/// (`:109`, allocation only) is unrepresentable — the type is `Copy`.
fn clean_layout(in_: &ChannelLayout, log_ctx: Option<&str>) -> ChannelLayout {
    if in_.index_from_channel(Channel::FrontCenter).is_err() && in_.nb_channels == 1 {
        log_verbose!(log_ctx, "Treating {} as mono", in_.describe());
        ChannelLayout::MONO
    } else {
        *in_
    }
}

/// `sane_layout` (`rematrix.c:114-149`), exact order:
///
/// 1. `nb_channels >= SWR_CH_MAX` → false (`:115-116`);
/// 2. custom orders: every id must be `AV_CHAN_UNUSED` or `< 64`
///    (`:117-125`) — unrepresentable here (no custom orders), and any
///    non-native order (UNSPEC, AMBISONIC) → false (`:126-127`);
/// 3. `mask = subset(~0)` (`:128`);
/// 4. `mask & AV_CH_LAYOUT_SURROUND == 0` → false — at least one front
///    speaker (`:129-130`);
/// 5. each of the 8 L/R pairs (front, side, back, front-of-center,
///    top-front, top-back, top-side, bottom-front) must not be asymmetric
///    (`:131-146`).
///
/// `pub` because the swresample-core port reuses it for the same guard at
/// `swresample.c:340-346`. Note UNSPEC never passes — but
/// [`swr_build_matrix2`] first coerces UNSPEC-1 to MONO
/// ([`clean_layout`]), so only UNSPEC with ≥2 channels reaches the
/// "'%s' is not supported" error.
pub fn sane_layout(ch_layout: &ChannelLayout) -> bool {
    if ch_layout.nb_channels >= SWR_CH_MAX {
        return false;
    }
    if ch_layout.order != Order::Native {
        return false;
    }
    let mask = ch_layout.subset(u64::MAX);
    if mask & CH_LAYOUT_SURROUND == 0 {
        // at least 1 front speaker
        return false;
    }
    let pairs = [
        (Channel::FrontLeft, Channel::FrontRight),
        (Channel::SideLeft, Channel::SideRight),
        (Channel::BackLeft, Channel::BackRight),
        (Channel::FrontLeftOfCenter, Channel::FrontRightOfCenter),
        (Channel::TopFrontLeft, Channel::TopFrontRight),
        (Channel::TopBackLeft, Channel::TopBackRight),
        (Channel::TopSideLeft, Channel::TopSideRight),
        (Channel::BottomFrontLeft, Channel::BottomFrontRight),
    ];
    for (l, r) in pairs {
        if !even(mask & (ch_bit(l) | ch_bit(r))) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// build_matrix — rematrix.c:151-570
// ---------------------------------------------------------------------------

/// `av_channel_layout_index_from_channel` on a raw channel *id* 0..63
/// (`channel_layout.c:715-747`, native arm `:736-742`): UNSPEC → `None`;
/// bit not set (or no such channel) → `None`; else the popcount below the
/// bit. Used because the C loops pass plain `int`s including phantom ids.
fn index_from_channel_id(layout: &ChannelLayout, id: u32) -> Option<usize> {
    match layout.order {
        Order::Unspecified => None,
        Order::Native => {
            let bit = 1u64 << id;
            if layout.mask & bit == 0 {
                None
            } else {
                Some((layout.mask & (bit - 1)).count_ones() as usize)
            }
        }
    }
}

/// `build_matrix` (`rematrix.c:151-570`) — synthesize the `[out][in]`
/// coefficient table for every input channel the output does not have
/// (`unaccounted = in_mask & !out_mask`), then normalize.
///
/// Branch order is C's exactly; all contributions `+=` onto the local
/// 41×41 table **except** the three ASSIGNMENTS that overwrite the 1.0
/// diagonal: `matrix[FC][FC] = clev*sqrt(2)` (`:210`),
/// `matrix[TFC][TFC]` (`:327`), `matrix[BFC][BFC]` (`:506`) — they fire
/// when the unaccounted stereo/top-front/bottom-front pair folds into an
/// output center that also exists in the input.
///
/// `matrix_param` is written at `stride*out_i + in_i` for every channel pair
/// present (`:553-555`); entries whose (i, j) are not both present are left
/// untouched, and the normalization at `:563-569` divides the **entire**
/// 64×64 strided array (rows beyond `nb_channels` included). Requires
/// `matrix_param.len() >= stride*63 + 64` (C's unchecked bound).
fn build_matrix(
    in_ch_layout: &ChannelLayout,
    out_ch_layout: &ChannelLayout,
    center_mix_level: f64,
    surround_mix_level: f64,
    lfe_mix_level: f64,
    maxval: f64,
    rematrix_volume: f64,
    matrix_param: &mut [f64],
    stride: usize,
    matrix_encoding: MatrixEncoding,
) {
    debug_assert!(matrix_param.len() >= stride * (SWR_CH_MAX - 1) + SWR_CH_MAX);

    let mut matrix = [[0.0f64; NUM_NAMED_CHANNELS]; NUM_NAMED_CHANNELS];
    let in_mask = in_ch_layout.subset(u64::MAX);
    let out_mask = out_ch_layout.subset(u64::MAX);
    let unaccounted = in_mask & !out_mask;
    let mut maxcoef = 0.0f64;
    let clev = center_mix_level;
    let slev = surround_mix_level;
    let lfe = lfe_mix_level;

    // rematrix.c:163-183 — AV_CHAN_UNUSED row/column clearing for CUSTOM
    // orders: unreachable here, `util::channel_layout` does not represent
    // AV_CHANNEL_ORDER_CUSTOM (module doc there); `swr_build_matrix2` /
    // `swr_set_matrix` callers always pass zeroed matrices.

    // rematrix.c:185-188 — diagonal passthrough.
    for i in 0..NUM_NAMED_CHANNELS {
        if in_mask & out_mask & (1u64 << i) != 0 {
            matrix[i][i] = 1.0;
        }
    }

    // The unaccounted-channel decision table, in C order. `m[o][i] += v`
    // mirrors `matrix[o][i] += v` at the cited lines.
    let mut m = matrix;

    if unaccounted & ch_bit(Channel::FrontCenter) != 0 {
        // rematrix.c:193-204
        if out_mask & CH_LAYOUT_STEREO == CH_LAYOUT_STEREO {
            if in_mask & CH_LAYOUT_STEREO != 0 {
                m[FRONT_LEFT][FRONT_CENTER] += clev;
                m[FRONT_RIGHT][FRONT_CENTER] += clev;
            } else {
                m[FRONT_LEFT][FRONT_CENTER] += M_SQRT1_2;
                m[FRONT_RIGHT][FRONT_CENTER] += M_SQRT1_2;
            }
        } else {
            unreachable!("rematrix.c:203: sane_layout guarantees a front-pair output");
        }
    }
    if unaccounted & CH_LAYOUT_STEREO != 0 {
        // rematrix.c:205-213
        if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][FRONT_LEFT] += M_SQRT1_2;
            m[FRONT_CENTER][FRONT_RIGHT] += M_SQRT1_2;
            if in_mask & ch_bit(Channel::FrontCenter) != 0 {
                // ASSIGNMENT (rematrix.c:210) — replaces the diagonal 1.0.
                m[FRONT_CENTER][FRONT_CENTER] = clev * std::f64::consts::SQRT_2;
            }
        } else {
            unreachable!("rematrix.c:212: sane_layout guarantees FL+FR or FC output");
        }
    }

    if unaccounted & ch_bit(Channel::BackCenter) != 0 {
        // rematrix.c:215-240
        if out_mask & ch_bit(Channel::BackLeft) != 0 {
            m[BACK_LEFT][BACK_CENTER] += M_SQRT1_2;
            m[BACK_RIGHT][BACK_CENTER] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
            m[SIDE_LEFT][BACK_CENTER] += M_SQRT1_2;
            m[SIDE_RIGHT][BACK_CENTER] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            if matches!(
                matrix_encoding,
                MatrixEncoding::Dolby | MatrixEncoding::Dplii
            ) {
                if unaccounted & (ch_bit(Channel::BackLeft) | ch_bit(Channel::SideLeft)) != 0 {
                    m[FRONT_LEFT][BACK_CENTER] -= slev * M_SQRT1_2;
                    m[FRONT_RIGHT][BACK_CENTER] += slev * M_SQRT1_2;
                } else {
                    m[FRONT_LEFT][BACK_CENTER] -= slev;
                    m[FRONT_RIGHT][BACK_CENTER] += slev;
                }
            } else {
                m[FRONT_LEFT][BACK_CENTER] += slev * M_SQRT1_2;
                m[FRONT_RIGHT][BACK_CENTER] += slev * M_SQRT1_2;
            }
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][BACK_CENTER] += slev * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:239: sane_layout guarantees a front output");
        }
    }
    if unaccounted & ch_bit(Channel::BackLeft) != 0 {
        // rematrix.c:241-273
        if out_mask & ch_bit(Channel::BackCenter) != 0 {
            m[BACK_CENTER][BACK_LEFT] += M_SQRT1_2;
            m[BACK_CENTER][BACK_RIGHT] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
            if in_mask & ch_bit(Channel::SideLeft) != 0 {
                m[SIDE_LEFT][BACK_LEFT] += M_SQRT1_2;
                m[SIDE_RIGHT][BACK_RIGHT] += M_SQRT1_2;
            } else {
                m[SIDE_LEFT][BACK_LEFT] += 1.0;
                m[SIDE_RIGHT][BACK_RIGHT] += 1.0;
            }
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            if matrix_encoding == MatrixEncoding::Dolby {
                m[FRONT_LEFT][BACK_LEFT] -= slev * M_SQRT1_2;
                m[FRONT_LEFT][BACK_RIGHT] -= slev * M_SQRT1_2;
                m[FRONT_RIGHT][BACK_LEFT] += slev * M_SQRT1_2;
                m[FRONT_RIGHT][BACK_RIGHT] += slev * M_SQRT1_2;
            } else if matrix_encoding == MatrixEncoding::Dplii {
                m[FRONT_LEFT][BACK_LEFT] -= slev * SQRT3_2;
                m[FRONT_LEFT][BACK_RIGHT] -= slev * M_SQRT1_2;
                m[FRONT_RIGHT][BACK_LEFT] += slev * M_SQRT1_2;
                m[FRONT_RIGHT][BACK_RIGHT] += slev * SQRT3_2;
            } else {
                m[FRONT_LEFT][BACK_LEFT] += slev;
                m[FRONT_RIGHT][BACK_RIGHT] += slev;
            }
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][BACK_LEFT] += slev * M_SQRT1_2;
            m[FRONT_CENTER][BACK_RIGHT] += slev * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:272: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::SideLeft) != 0 {
        // rematrix.c:275-309
        if out_mask & ch_bit(Channel::BackLeft) != 0 {
            // if back channels do not exist in the input, just copy side
            // channels to back channels, otherwise mix side into back
            if in_mask & ch_bit(Channel::BackLeft) != 0 {
                m[BACK_LEFT][SIDE_LEFT] += M_SQRT1_2;
                m[BACK_RIGHT][SIDE_RIGHT] += M_SQRT1_2;
            } else {
                m[BACK_LEFT][SIDE_LEFT] += 1.0;
                m[BACK_RIGHT][SIDE_RIGHT] += 1.0;
            }
        } else if out_mask & ch_bit(Channel::BackCenter) != 0 {
            m[BACK_CENTER][SIDE_LEFT] += M_SQRT1_2;
            m[BACK_CENTER][SIDE_RIGHT] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            if matrix_encoding == MatrixEncoding::Dolby {
                m[FRONT_LEFT][SIDE_LEFT] -= slev * M_SQRT1_2;
                m[FRONT_LEFT][SIDE_RIGHT] -= slev * M_SQRT1_2;
                m[FRONT_RIGHT][SIDE_LEFT] += slev * M_SQRT1_2;
                m[FRONT_RIGHT][SIDE_RIGHT] += slev * M_SQRT1_2;
            } else if matrix_encoding == MatrixEncoding::Dplii {
                m[FRONT_LEFT][SIDE_LEFT] -= slev * SQRT3_2;
                m[FRONT_LEFT][SIDE_RIGHT] -= slev * M_SQRT1_2;
                m[FRONT_RIGHT][SIDE_LEFT] += slev * M_SQRT1_2;
                m[FRONT_RIGHT][SIDE_RIGHT] += slev * SQRT3_2;
            } else {
                m[FRONT_LEFT][SIDE_LEFT] += slev;
                m[FRONT_RIGHT][SIDE_RIGHT] += slev;
            }
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][SIDE_LEFT] += slev * M_SQRT1_2;
            m[FRONT_CENTER][SIDE_RIGHT] += slev * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:308: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::FrontLeftOfCenter) != 0 {
        // rematrix.c:311-320
        if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            m[FRONT_LEFT][FRONT_LEFT_OF_CENTER] += 1.0;
            m[FRONT_RIGHT][FRONT_RIGHT_OF_CENTER] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][FRONT_LEFT_OF_CENTER] += M_SQRT1_2;
            m[FRONT_CENTER][FRONT_RIGHT_OF_CENTER] += M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:319: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::TopFrontLeft) != 0 {
        // rematrix.c:322-337
        if out_mask & ch_bit(Channel::TopFrontCenter) != 0 {
            m[TOP_FRONT_CENTER][TOP_FRONT_LEFT] += M_SQRT1_2;
            m[TOP_FRONT_CENTER][TOP_FRONT_RIGHT] += M_SQRT1_2;
            if in_mask & ch_bit(Channel::TopFrontCenter) != 0 {
                // ASSIGNMENT (rematrix.c:327).
                m[TOP_FRONT_CENTER][TOP_FRONT_CENTER] = clev * std::f64::consts::SQRT_2;
            }
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            // U+030 -> M+030 in ITU-R BS.2127-1, Table 16.
            m[FRONT_LEFT][TOP_FRONT_LEFT] += 1.0;
            m[FRONT_RIGHT][TOP_FRONT_RIGHT] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][TOP_FRONT_LEFT] += M_SQRT1_2;
            m[FRONT_CENTER][TOP_FRONT_RIGHT] += M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:336: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::TopFrontCenter) != 0 {
        // rematrix.c:339-353
        if out_mask & ch_bit(Channel::TopFrontLeft) != 0 {
            // U+030 = U-030 = sqrt(1/2)
            m[TOP_FRONT_LEFT][TOP_FRONT_CENTER] += M_SQRT1_2;
            m[TOP_FRONT_RIGHT][TOP_FRONT_CENTER] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            // M+000 = 1
            m[FRONT_CENTER][TOP_FRONT_CENTER] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            // M+030 = M-030 = sqrt(1/2)
            m[FRONT_LEFT][TOP_FRONT_CENTER] += clev;
            m[FRONT_RIGHT][TOP_FRONT_CENTER] += clev;
        } else {
            unreachable!("rematrix.c:352: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::TopBackLeft) != 0 {
        // rematrix.c:355-377
        if out_mask & ch_bit(Channel::TopBackCenter) != 0 {
            m[TOP_BACK_CENTER][TOP_BACK_LEFT] += M_SQRT1_2;
            m[TOP_BACK_CENTER][TOP_BACK_RIGHT] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::TopFrontLeft) != 0 {
            // IAMF v1.1.0, Section 7.3.2.1.1.
            m[TOP_FRONT_LEFT][TOP_BACK_LEFT] += M_SQRT1_2;
            m[TOP_FRONT_RIGHT][TOP_BACK_RIGHT] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::BackLeft) != 0 {
            m[BACK_LEFT][TOP_BACK_LEFT] += 1.0;
            m[BACK_RIGHT][TOP_BACK_RIGHT] += 1.0;
        } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
            m[SIDE_LEFT][TOP_BACK_LEFT] += 1.0;
            m[SIDE_RIGHT][TOP_BACK_RIGHT] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            m[FRONT_LEFT][TOP_BACK_LEFT] += slev;
            m[FRONT_RIGHT][TOP_BACK_RIGHT] += slev;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][TOP_BACK_LEFT] += slev * M_SQRT1_2;
            m[FRONT_CENTER][TOP_BACK_RIGHT] += slev * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:376: sane_layout guarantees a front output");
        }
    }

    // BS.2127-1 maps U+180 to rear outputs before front outputs.
    if unaccounted & ch_bit(Channel::TopBackCenter) != 0 {
        // rematrix.c:380-397
        if out_mask & ch_bit(Channel::TopBackLeft) != 0 {
            m[TOP_BACK_LEFT][TOP_BACK_CENTER] += M_SQRT1_2;
            m[TOP_BACK_RIGHT][TOP_BACK_CENTER] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::BackLeft) != 0 {
            m[BACK_LEFT][TOP_BACK_CENTER] += M_SQRT1_2;
            m[BACK_RIGHT][TOP_BACK_CENTER] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
            m[SIDE_LEFT][TOP_BACK_CENTER] += M_SQRT1_2;
            m[SIDE_RIGHT][TOP_BACK_CENTER] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            m[FRONT_LEFT][TOP_BACK_CENTER] += 0.5;
            m[FRONT_RIGHT][TOP_BACK_CENTER] += 0.5;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][TOP_BACK_CENTER] += 0.5;
        } else {
            unreachable!("rematrix.c:396: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::TopSideLeft) != 0 {
        // rematrix.c:400-444
        let tfl_tbc = ch_bit(Channel::TopFrontLeft) | ch_bit(Channel::TopBackCenter);
        let tfl_tbl = ch_bit(Channel::TopFrontLeft) | ch_bit(Channel::TopBackLeft);
        if out_mask & tfl_tbc == tfl_tbc {
            // UH+180 = sqrt(1/3); U±045 = sqrt(2/3)
            m[TOP_FRONT_LEFT][TOP_SIDE_LEFT] += SQRT2_3;
            m[TOP_FRONT_RIGHT][TOP_SIDE_RIGHT] += SQRT2_3;
            m[TOP_BACK_CENTER][TOP_SIDE_LEFT] += SQRT1_3;
            m[TOP_BACK_CENTER][TOP_SIDE_RIGHT] += SQRT1_3;
        } else if out_mask & tfl_tbl == tfl_tbl {
            // U±030 = U±110 = sqrt(1/2)
            m[TOP_FRONT_LEFT][TOP_SIDE_LEFT] += M_SQRT1_2;
            m[TOP_FRONT_RIGHT][TOP_SIDE_RIGHT] += M_SQRT1_2;
            m[TOP_BACK_LEFT][TOP_SIDE_LEFT] += M_SQRT1_2;
            m[TOP_BACK_RIGHT][TOP_SIDE_RIGHT] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::TopFrontLeft) != 0
            && out_mask & (ch_bit(Channel::BackLeft) | ch_bit(Channel::SideLeft)) != 0
        {
            // U±030 = M±110 = sqrt(1/2)
            m[TOP_FRONT_LEFT][TOP_SIDE_LEFT] += M_SQRT1_2;
            m[TOP_FRONT_RIGHT][TOP_SIDE_RIGHT] += M_SQRT1_2;
            if out_mask & ch_bit(Channel::BackLeft) != 0 {
                m[BACK_LEFT][TOP_SIDE_LEFT] += M_SQRT1_2;
                m[BACK_RIGHT][TOP_SIDE_RIGHT] += M_SQRT1_2;
            } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
                m[SIDE_LEFT][TOP_SIDE_LEFT] += M_SQRT1_2;
                m[SIDE_RIGHT][TOP_SIDE_RIGHT] += M_SQRT1_2;
            }
        } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
            // M±090 = 1
            m[SIDE_LEFT][TOP_SIDE_LEFT] += 1.0;
            m[SIDE_RIGHT][TOP_SIDE_RIGHT] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            // M±030 = M±110 = sqrt(1/2)
            m[FRONT_LEFT][TOP_SIDE_LEFT] += slev;
            m[FRONT_RIGHT][TOP_SIDE_RIGHT] += slev;
            if out_mask & ch_bit(Channel::BackLeft) != 0 {
                m[BACK_LEFT][TOP_SIDE_LEFT] += M_SQRT1_2;
                m[BACK_RIGHT][TOP_SIDE_RIGHT] += M_SQRT1_2;
            }
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][TOP_SIDE_LEFT] += slev * M_SQRT1_2;
            m[FRONT_CENTER][TOP_SIDE_RIGHT] += slev * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:443: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::TopCenter) != 0 {
        // rematrix.c:446-489
        let tfl_tbl = ch_bit(Channel::TopFrontLeft) | ch_bit(Channel::TopBackLeft);
        let tfl_tbc = ch_bit(Channel::TopFrontLeft) | ch_bit(Channel::TopBackCenter);
        if out_mask & tfl_tbl == tfl_tbl {
            // U+045 = U-045 = U+135 = U-135 = sqrt(1/4)
            m[TOP_FRONT_LEFT][TOP_CENTER] += 0.5;
            m[TOP_FRONT_RIGHT][TOP_CENTER] += 0.5;
            m[TOP_BACK_LEFT][TOP_CENTER] += 0.5;
            m[TOP_BACK_RIGHT][TOP_CENTER] += 0.5;
        } else if out_mask & tfl_tbc == tfl_tbc {
            // U+045 = U-045 = UH+180 = sqrt(1/3)
            m[TOP_FRONT_LEFT][TOP_CENTER] += SQRT1_3;
            m[TOP_FRONT_RIGHT][TOP_CENTER] += SQRT1_3;
            m[TOP_BACK_CENTER][TOP_CENTER] += SQRT1_3;
        } else if out_mask & ch_bit(Channel::TopFrontLeft) != 0
            && out_mask & (ch_bit(Channel::BackLeft) | ch_bit(Channel::SideLeft)) != 0
        {
            // U+045 = U-045 = M+135 = M-135 = sqrt(1/4)
            // U+030 = U-030 = M+110 = M-110 = sqrt(1/4)
            m[TOP_FRONT_LEFT][TOP_CENTER] += 0.5;
            m[TOP_FRONT_RIGHT][TOP_CENTER] += 0.5;
            if out_mask & ch_bit(Channel::BackLeft) != 0 {
                m[BACK_LEFT][TOP_CENTER] += 0.5;
                m[BACK_RIGHT][TOP_CENTER] += 0.5;
            } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
                m[SIDE_LEFT][TOP_CENTER] += 0.5;
                m[SIDE_RIGHT][TOP_CENTER] += 0.5;
            }
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            // M+030 = M-030 = M+135 = M-135 = sqrt(1/4)
            m[FRONT_LEFT][TOP_CENTER] += 0.5;
            m[FRONT_RIGHT][TOP_CENTER] += 0.5;
            if out_mask & ch_bit(Channel::BackLeft) != 0 {
                m[BACK_LEFT][TOP_CENTER] += 0.5;
                m[BACK_RIGHT][TOP_CENTER] += 0.5;
            } else if out_mask & ch_bit(Channel::SideLeft) != 0 {
                m[SIDE_LEFT][TOP_CENTER] += 0.5;
                m[SIDE_RIGHT][TOP_CENTER] += 0.5;
            }
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][TOP_CENTER] += 0.5;
        } else {
            unreachable!("rematrix.c:488: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::BottomFrontCenter) != 0 {
        // rematrix.c:491-499
        if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][BOTTOM_FRONT_CENTER] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            m[FRONT_LEFT][BOTTOM_FRONT_CENTER] += clev;
            m[FRONT_RIGHT][BOTTOM_FRONT_CENTER] += clev;
        } else {
            unreachable!("rematrix.c:498: sane_layout guarantees a front output");
        }
    }

    if unaccounted & ch_bit(Channel::BottomFrontLeft) != 0 {
        // rematrix.c:501-516
        if out_mask & ch_bit(Channel::BottomFrontCenter) != 0 {
            m[BOTTOM_FRONT_CENTER][BOTTOM_FRONT_LEFT] += M_SQRT1_2;
            m[BOTTOM_FRONT_CENTER][BOTTOM_FRONT_RIGHT] += M_SQRT1_2;
            if in_mask & ch_bit(Channel::BottomFrontCenter) != 0 {
                // ASSIGNMENT (rematrix.c:506).
                m[BOTTOM_FRONT_CENTER][BOTTOM_FRONT_CENTER] = clev * std::f64::consts::SQRT_2;
            }
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            // M±030 = 1
            m[FRONT_LEFT][BOTTOM_FRONT_LEFT] += 1.0;
            m[FRONT_RIGHT][BOTTOM_FRONT_RIGHT] += 1.0;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][BOTTOM_FRONT_LEFT] += M_SQRT1_2;
            m[FRONT_CENTER][BOTTOM_FRONT_RIGHT] += M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:515: sane_layout guarantees a front output");
        }
    }

    // mix LFE into front left/right or center (rematrix.c:519-527).
    if unaccounted & ch_bit(Channel::LowFrequency) != 0 {
        if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][LOW_FREQUENCY] += lfe;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            m[FRONT_LEFT][LOW_FREQUENCY] += lfe * M_SQRT1_2;
            m[FRONT_RIGHT][LOW_FREQUENCY] += lfe * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:526: sane_layout guarantees a front output");
        }
    }

    // mix LFE2 into LFE, front left/right or center (rematrix.c:530-540).
    if unaccounted & ch_bit(Channel::LowFrequency2) != 0 {
        if out_mask & ch_bit(Channel::LowFrequency) != 0 {
            m[LOW_FREQUENCY][LOW_FREQUENCY_2] += M_SQRT1_2;
        } else if out_mask & ch_bit(Channel::FrontCenter) != 0 {
            m[FRONT_CENTER][LOW_FREQUENCY_2] += lfe;
        } else if out_mask & ch_bit(Channel::FrontLeft) != 0 {
            m[FRONT_LEFT][LOW_FREQUENCY_2] += lfe * M_SQRT1_2;
            m[FRONT_RIGHT][LOW_FREQUENCY_2] += lfe * M_SQRT1_2;
        } else {
            unreachable!("rematrix.c:539: sane_layout guarantees a front output");
        }
    }

    // rematrix.c:543-559 — scatter into matrix_param, track max row L1.
    for i in 0..64u32 {
        let Some(out_i) = index_from_channel_id(out_ch_layout, i) else {
            continue;
        };
        let mut sum = 0.0f64;
        for j in 0..64u32 {
            let Some(in_i) = index_from_channel_id(in_ch_layout, j) else {
                continue;
            };
            let v = if (i as usize) < NUM_NAMED_CHANNELS && (j as usize) < NUM_NAMED_CHANNELS {
                m[i as usize][j as usize]
            } else {
                // rematrix.c:555 — identity passthrough for ids >= 41.
                ((i == j && in_mask & out_mask & (1u64 << i) != 0) as u8) as f64
            };
            matrix_param[stride * out_i + in_i] = v;
            sum += v.abs();
        }
        maxcoef = maxcoef.max(sum);
    }

    // rematrix.c:560-569 — normalization over the FULL 64x64 strided array.
    if rematrix_volume < 0.0 {
        maxcoef = -rematrix_volume;
    }
    if maxcoef > maxval || rematrix_volume < 0.0 {
        maxcoef /= maxval;
        for i in 0..SWR_CH_MAX {
            for j in 0..SWR_CH_MAX {
                matrix_param[stride * i + j] /= maxcoef;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// swr_build_matrix2 — rematrix.c:572-652 (public API, swresample.h:380-404)
// ---------------------------------------------------------------------------

/// `swr_build_matrix2` (`rematrix.c:572-652`).
///
/// 1. `clean_layout` both sides (`:582-583`; C's `ret |=` with
///    `av_channel_layout_copy` failures is unrepresentable).
/// 2. STEREO_DOWNMIX translation (`:587-598`): a downmix output becomes
///    STEREO unless the input already carries DL/DR (and symmetrically for
///    the input).
/// 3.-6. `check` + `sane_layout` for input then output, with C's exact
///    ERROR texts (`:601,607,613,619`).
/// 7. [`build_matrix`], 8. `rematrix_volume > 0` scaling over the full
///    64×64 (`:628-633`), 9. the DEBUG coefficient dump (`:635-644`).
///
/// `matrix` is the caller's `double*` with `stride` — it must hold
/// `stride*63 + 64` elements ([`Error::BufferTooSmall`] where C would write
/// out of bounds).
pub fn swr_build_matrix2(
    in_layout: &ChannelLayout,
    out_layout: &ChannelLayout,
    center_mix_level: f64,
    surround_mix_level: f64,
    lfe_mix_level: f64,
    maxval: f64,
    rematrix_volume: f64,
    matrix: &mut [f64],
    stride: usize,
    matrix_encoding: MatrixEncoding,
    log_ctx: Option<&str>,
) -> Result<()> {
    if matrix.len() < stride * (SWR_CH_MAX - 1) + SWR_CH_MAX {
        return Err(Error::BufferTooSmall);
    }

    let mut in_ch_layout = clean_layout(in_layout, log_ctx);
    let mut out_ch_layout = clean_layout(out_layout, log_ctx);

    // rematrix.c:587-598.
    if out_ch_layout == ChannelLayout::STEREO_DOWNMIX
        && in_ch_layout.subset(CH_LAYOUT_STEREO_DOWNMIX) == 0
    {
        out_ch_layout = ChannelLayout::STEREO;
    }
    if in_ch_layout == ChannelLayout::STEREO_DOWNMIX
        && out_ch_layout.subset(CH_LAYOUT_STEREO_DOWNMIX) == 0
    {
        in_ch_layout = ChannelLayout::STEREO;
    }

    // rematrix.c:600-622 — layout checks, input first (C's order).
    if !in_ch_layout.check() {
        log_error!(log_ctx, "Input channel layout is invalid");
        return Err(Error::InvalidArgument(
            "Input channel layout is invalid".into(),
        ));
    }
    if !sane_layout(&in_ch_layout) {
        let described = in_ch_layout.describe();
        log_error!(
            log_ctx,
            "Input channel layout '{described}' is not supported"
        );
        return Err(Error::InvalidArgument(format!(
            "Input channel layout '{described}' is not supported"
        )));
    }
    if !out_ch_layout.check() {
        log_error!(log_ctx, "Output channel layout is invalid");
        return Err(Error::InvalidArgument(
            "Output channel layout is invalid".into(),
        ));
    }
    if !sane_layout(&out_ch_layout) {
        let described = out_ch_layout.describe();
        log_error!(
            log_ctx,
            "Output channel layout '{described}' is not supported"
        );
        return Err(Error::InvalidArgument(format!(
            "Output channel layout '{described}' is not supported"
        )));
    }

    build_matrix(
        &in_ch_layout,
        &out_ch_layout,
        center_mix_level,
        surround_mix_level,
        lfe_mix_level,
        maxval,
        rematrix_volume,
        matrix,
        stride,
        matrix_encoding,
    );

    // rematrix.c:628-633.
    if rematrix_volume > 0.0 {
        for i in 0..SWR_CH_MAX {
            for j in 0..SWR_CH_MAX {
                matrix[stride * i + j] *= rematrix_volume;
            }
        }
    }

    // rematrix.c:635-644 — DEBUG dump. One log line per C output line.
    log_debug!(log_ctx, "Matrix coefficients:");
    for i in 0..out_ch_layout.nb_channels {
        let mut line = format!(
            "{}: ",
            out_ch_layout
                .channel_from_index(i)
                .unwrap_or(Channel::None)
                .name()
        );
        for j in 0..in_ch_layout.nb_channels {
            line.push_str(&format!(
                "{}:{:.6} ",
                in_ch_layout
                    .channel_from_index(j)
                    .unwrap_or(Channel::None)
                    .name(),
                matrix[stride * i + j]
            ));
        }
        log_debug!(log_ctx, "{line}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// rematrix_template.c — the kernels
// ---------------------------------------------------------------------------
//
// Each kernel mirrors one RENAME() instantiation of rematrix_template.c:52-127.
// All operate on byte slices (a channel plane of an AudioData), sample i at
// byte offset i*bps, little-endian (crate LE policy). Contract: the caller
// guarantees `buf.len() >= len*bps` — RematrixContext::rematrix checks and
// returns Error::BufferTooSmall where C's pointer arithmetic would be UB.
//
// INTER types (rematrix_template.c:22-49):
//   float  -> f32   (R(x) = x)
//   double -> f64   (R(x) = x)
//   s16    -> i32   (R(x) = ((x + 16384) >> 15) as i16, wrapping int math)
//   s16+clip -> i32 (R(x) = clamp(-32768, 32767)((x + 16384) >> 15))
//   s32    -> i64   (R(x) = ((x + 16384) >> 15) as i32 — the i64 store
//                    truncates exactly like C's int64_t -> int32_t assignment)
// C int overflow is UB that every real compiler turns into wrapping; the
// port uses wrapping ops so the behavior is defined and identical.

/// C `lrintf(target)` where `target: double` (`rematrix.c:710,757,781`):
/// implicit `double -> float` conversion **first** (round-to-nearest-even),
/// then round-half-to-even to `int` (default `FE_TONEAREST`). Out-of-range
/// results are C UB (x86-64 returns the "integer indefinite" value); Rust's
/// `as` saturates — unreachable for 17.15 coefficients in |x| < 2^31/32768.
#[inline]
fn lrintf(target: f64) -> i32 {
    (target as f32).round_ties_even() as i32
}

/// `av_clip_int16_c` (`libavutil/common.h:243-247`) — `clamp(-32768, 32767)`
/// for in-range-typed inputs.
#[inline]
const fn clip_i16(v: i32) -> i32 {
    if v < -32768 {
        -32768
    } else if v > 32767 {
        32767
    } else {
        v
    }
}

// --- little-endian sample access -------------------------------------------

#[inline]
fn ld_i16(b: &[u8], i: usize) -> i16 {
    i16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}
#[inline]
fn st_i16(b: &mut [u8], i: usize, v: i16) {
    b[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn ld_i32(b: &[u8], i: usize) -> i32 {
    i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}
#[inline]
fn st_i32(b: &mut [u8], i: usize, v: i32) {
    b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn ld_f32(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}
#[inline]
fn st_f32(b: &mut [u8], i: usize, v: f32) {
    b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn ld_f64(b: &[u8], i: usize) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[8 * i..8 * i + 8]);
    f64::from_le_bytes(a)
}
#[inline]
fn st_f64(b: &mut [u8], i: usize, v: f64) {
    b[8 * i..8 * i + 8].copy_from_slice(&v.to_le_bytes());
}

// --- float (TEMPLATE_REMATRIX_FLT, rematrix_template.c:21-32) --------------

/// `copy_float` (`rematrix_template.c:66-76`).
pub(crate) fn copy_float(out: &mut [u8], in_: &[u8], coeff: f32, len: usize) {
    for i in 0..len {
        st_f32(out, i, coeff * ld_f32(in_, i));
    }
}

/// `sum2_float` (`rematrix_template.c:52-64`).
pub(crate) fn sum2_float(
    out: &mut [u8],
    in1: &[u8],
    in2: &[u8],
    coeff1: f32,
    coeff2: f32,
    len: usize,
) {
    for i in 0..len {
        // R(coeff1*in1[i] + coeff2*in2[i]) — f32 math, left-assoc.
        st_f32(out, i, coeff1 * ld_f32(in1, i) + coeff2 * ld_f32(in2, i));
    }
}

/// `mix6to2_float` (`rematrix_template.c:78-91`).
fn mix6to2_float(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[f32], len: usize) {
    let (i0, i1, i2, i3, i4, i5) = six_planes(in_);
    for i in 0..len {
        let t = ld_f32(i2, i) * c[0 * 6 + 2] + ld_f32(i3, i) * c[0 * 6 + 3];
        st_f32(
            out0,
            i,
            t + ld_f32(i0, i) * c[0 * 6 + 0] + ld_f32(i4, i) * c[0 * 6 + 4],
        );
        st_f32(
            out1,
            i,
            t + ld_f32(i1, i) * c[1 * 6 + 1] + ld_f32(i5, i) * c[1 * 6 + 5],
        );
    }
}

/// `mix8to2_float` (`rematrix_template.c:93-106`).
fn mix8to2_float(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[f32], len: usize) {
    let (i0, i1, i2, i3, i4, i5, i6, i7) = eight_planes(in_);
    for i in 0..len {
        let t = ld_f32(i2, i) * c[0 * 8 + 2] + ld_f32(i3, i) * c[0 * 8 + 3];
        st_f32(
            out0,
            i,
            t + ld_f32(i0, i) * c[0 * 8 + 0]
                + ld_f32(i4, i) * c[0 * 8 + 4]
                + ld_f32(i6, i) * c[0 * 8 + 6],
        );
        st_f32(
            out1,
            i,
            t + ld_f32(i1, i) * c[1 * 8 + 1]
                + ld_f32(i5, i) * c[1 * 8 + 5]
                + ld_f32(i7, i) * c[1 * 8 + 7],
        );
    }
}

// --- double (TEMPLATE_REMATRIX_DBL, rematrix_template.c:27-32) -------------

/// `copy_double` (`rematrix_template.c:66-76`).
pub(crate) fn copy_double(out: &mut [u8], in_: &[u8], coeff: f64, len: usize) {
    for i in 0..len {
        st_f64(out, i, coeff * ld_f64(in_, i));
    }
}

/// `sum2_double` (`rematrix_template.c:52-64`).
pub(crate) fn sum2_double(
    out: &mut [u8],
    in1: &[u8],
    in2: &[u8],
    coeff1: f64,
    coeff2: f64,
    len: usize,
) {
    for i in 0..len {
        st_f64(out, i, coeff1 * ld_f64(in1, i) + coeff2 * ld_f64(in2, i));
    }
}

/// `mix6to2_double` (`rematrix_template.c:78-91`).
fn mix6to2_double(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[f64], len: usize) {
    let (i0, i1, i2, i3, i4, i5) = six_planes(in_);
    for i in 0..len {
        let t = ld_f64(i2, i) * c[0 * 6 + 2] + ld_f64(i3, i) * c[0 * 6 + 3];
        st_f64(
            out0,
            i,
            t + ld_f64(i0, i) * c[0 * 6 + 0] + ld_f64(i4, i) * c[0 * 6 + 4],
        );
        st_f64(
            out1,
            i,
            t + ld_f64(i1, i) * c[1 * 6 + 1] + ld_f64(i5, i) * c[1 * 6 + 5],
        );
    }
}

/// `mix8to2_double` (`rematrix_template.c:93-106`).
fn mix8to2_double(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[f64], len: usize) {
    let (i0, i1, i2, i3, i4, i5, i6, i7) = eight_planes(in_);
    for i in 0..len {
        let t = ld_f64(i2, i) * c[0 * 8 + 2] + ld_f64(i3, i) * c[0 * 8 + 3];
        st_f64(
            out0,
            i,
            t + ld_f64(i0, i) * c[0 * 8 + 0]
                + ld_f64(i4, i) * c[0 * 8 + 4]
                + ld_f64(i6, i) * c[0 * 8 + 6],
        );
        st_f64(
            out1,
            i,
            t + ld_f64(i1, i) * c[1 * 8 + 1]
                + ld_f64(i5, i) * c[1 * 8 + 5]
                + ld_f64(i7, i) * c[1 * 8 + 7],
        );
    }
}

// --- s16 (TEMPLATE_REMATRIX_S16, rematrix_template.c:33-43) ----------------

/// `copy_s16` (`rematrix_template.c:66-76`, `R(x) = ((x+16384)>>15)`).
pub(crate) fn copy_s16(out: &mut [u8], in_: &[u8], coeff: i32, len: usize) {
    for i in 0..len {
        let v = coeff.wrapping_mul(ld_i16(in_, i) as i32);
        st_i16(out, i, (v.wrapping_add(16384) >> 15) as i16);
    }
}

/// `copy_clip_s16` (`rematrix_template.c:36-39`,
/// `R(x) = av_clip_int16(((x+16384)>>15))`).
pub(crate) fn copy_clip_s16(out: &mut [u8], in_: &[u8], coeff: i32, len: usize) {
    for i in 0..len {
        let v = coeff.wrapping_mul(ld_i16(in_, i) as i32);
        st_i16(out, i, clip_i16(v.wrapping_add(16384) >> 15) as i16);
    }
}

/// `sum2_s16` (`rematrix_template.c:52-64`).
pub(crate) fn sum2_s16(
    out: &mut [u8],
    in1: &[u8],
    in2: &[u8],
    coeff1: i32,
    coeff2: i32,
    len: usize,
) {
    for i in 0..len {
        let v = coeff1
            .wrapping_mul(ld_i16(in1, i) as i32)
            .wrapping_add(coeff2.wrapping_mul(ld_i16(in2, i) as i32));
        st_i16(out, i, (v.wrapping_add(16384) >> 15) as i16);
    }
}

/// `sum2_clip_s16` (`rematrix_template.c:36-39` + `:52-64`).
pub(crate) fn sum2_clip_s16(
    out: &mut [u8],
    in1: &[u8],
    in2: &[u8],
    coeff1: i32,
    coeff2: i32,
    len: usize,
) {
    for i in 0..len {
        let v = coeff1
            .wrapping_mul(ld_i16(in1, i) as i32)
            .wrapping_add(coeff2.wrapping_mul(ld_i16(in2, i) as i32));
        st_i16(out, i, clip_i16(v.wrapping_add(16384) >> 15) as i16);
    }
}

/// `mix6to2_s16` (`rematrix_template.c:78-91`).
fn mix6to2_s16(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[i32], len: usize) {
    let (i0, i1, i2, i3, i4, i5) = six_planes(in_);
    for i in 0..len {
        let t = c[0 * 6 + 2]
            .wrapping_mul(ld_i16(i2, i) as i32)
            .wrapping_add(c[0 * 6 + 3].wrapping_mul(ld_i16(i3, i) as i32));
        let l = t
            .wrapping_add(c[0 * 6 + 0].wrapping_mul(ld_i16(i0, i) as i32))
            .wrapping_add(c[0 * 6 + 4].wrapping_mul(ld_i16(i4, i) as i32));
        let r = t
            .wrapping_add(c[1 * 6 + 1].wrapping_mul(ld_i16(i1, i) as i32))
            .wrapping_add(c[1 * 6 + 5].wrapping_mul(ld_i16(i5, i) as i32));
        st_i16(out0, i, (l.wrapping_add(16384) >> 15) as i16);
        st_i16(out1, i, (r.wrapping_add(16384) >> 15) as i16);
    }
}

/// `mix6to2_clip_s16` (`rematrix_template.c:36-39` + `:78-91`).
fn mix6to2_clip_s16(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[i32], len: usize) {
    let (i0, i1, i2, i3, i4, i5) = six_planes(in_);
    for i in 0..len {
        let t = c[0 * 6 + 2]
            .wrapping_mul(ld_i16(i2, i) as i32)
            .wrapping_add(c[0 * 6 + 3].wrapping_mul(ld_i16(i3, i) as i32));
        let l = t
            .wrapping_add(c[0 * 6 + 0].wrapping_mul(ld_i16(i0, i) as i32))
            .wrapping_add(c[0 * 6 + 4].wrapping_mul(ld_i16(i4, i) as i32));
        let r = t
            .wrapping_add(c[1 * 6 + 1].wrapping_mul(ld_i16(i1, i) as i32))
            .wrapping_add(c[1 * 6 + 5].wrapping_mul(ld_i16(i5, i) as i32));
        st_i16(out0, i, clip_i16(l.wrapping_add(16384) >> 15) as i16);
        st_i16(out1, i, clip_i16(r.wrapping_add(16384) >> 15) as i16);
    }
}

/// `mix8to2_s16` (`rematrix_template.c:93-106`).
fn mix8to2_s16(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[i32], len: usize) {
    let (i0, i1, i2, i3, i4, i5, i6, i7) = eight_planes(in_);
    for i in 0..len {
        let t = c[0 * 8 + 2]
            .wrapping_mul(ld_i16(i2, i) as i32)
            .wrapping_add(c[0 * 8 + 3].wrapping_mul(ld_i16(i3, i) as i32));
        let l = t
            .wrapping_add(c[0 * 8 + 0].wrapping_mul(ld_i16(i0, i) as i32))
            .wrapping_add(c[0 * 8 + 4].wrapping_mul(ld_i16(i4, i) as i32))
            .wrapping_add(c[0 * 8 + 6].wrapping_mul(ld_i16(i6, i) as i32));
        let r = t
            .wrapping_add(c[1 * 8 + 1].wrapping_mul(ld_i16(i1, i) as i32))
            .wrapping_add(c[1 * 8 + 5].wrapping_mul(ld_i16(i5, i) as i32))
            .wrapping_add(c[1 * 8 + 7].wrapping_mul(ld_i16(i7, i) as i32));
        st_i16(out0, i, (l.wrapping_add(16384) >> 15) as i16);
        st_i16(out1, i, (r.wrapping_add(16384) >> 15) as i16);
    }
}

/// `mix8to2_clip_s16` (`rematrix_template.c:36-39` + `:93-106`).
fn mix8to2_clip_s16(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[i32], len: usize) {
    let (i0, i1, i2, i3, i4, i5, i6, i7) = eight_planes(in_);
    for i in 0..len {
        let t = c[0 * 8 + 2]
            .wrapping_mul(ld_i16(i2, i) as i32)
            .wrapping_add(c[0 * 8 + 3].wrapping_mul(ld_i16(i3, i) as i32));
        let l = t
            .wrapping_add(c[0 * 8 + 0].wrapping_mul(ld_i16(i0, i) as i32))
            .wrapping_add(c[0 * 8 + 4].wrapping_mul(ld_i16(i4, i) as i32))
            .wrapping_add(c[0 * 8 + 6].wrapping_mul(ld_i16(i6, i) as i32));
        let r = t
            .wrapping_add(c[1 * 8 + 1].wrapping_mul(ld_i16(i1, i) as i32))
            .wrapping_add(c[1 * 8 + 5].wrapping_mul(ld_i16(i5, i) as i32))
            .wrapping_add(c[1 * 8 + 7].wrapping_mul(ld_i16(i7, i) as i32));
        st_i16(out0, i, clip_i16(l.wrapping_add(16384) >> 15) as i16);
        st_i16(out1, i, clip_i16(r.wrapping_add(16384) >> 15) as i16);
    }
}

// --- s32 (TEMPLATE_REMATRIX_S32, rematrix_template.c:44-49) ----------------

/// `copy_s32` (`rematrix_template.c:66-76`, INTER = `int64_t`).
pub(crate) fn copy_s32(out: &mut [u8], in_: &[u8], coeff: i32, len: usize) {
    for i in 0..len {
        let v = (coeff as i64).wrapping_mul(ld_i32(in_, i) as i64);
        st_i32(out, i, (v.wrapping_add(16384) >> 15) as i32);
    }
}

/// `sum2_s32` (`rematrix_template.c:52-64`, INTER = `int64_t`).
pub(crate) fn sum2_s32(
    out: &mut [u8],
    in1: &[u8],
    in2: &[u8],
    coeff1: i32,
    coeff2: i32,
    len: usize,
) {
    for i in 0..len {
        let v = (coeff1 as i64)
            .wrapping_mul(ld_i32(in1, i) as i64)
            .wrapping_add((coeff2 as i64).wrapping_mul(ld_i32(in2, i) as i64));
        st_i32(out, i, (v.wrapping_add(16384) >> 15) as i32);
    }
}

/// `mix6to2_s32` (`rematrix_template.c:78-91`).
fn mix6to2_s32(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[i32], len: usize) {
    let (i0, i1, i2, i3, i4, i5) = six_planes(in_);
    for i in 0..len {
        let cm = |k: usize, p: &[u8]| (c[k] as i64).wrapping_mul(ld_i32(p, i) as i64);
        let t = cm(0 * 6 + 2, i2).wrapping_add(cm(0 * 6 + 3, i3));
        let l = t
            .wrapping_add(cm(0 * 6 + 0, i0))
            .wrapping_add(cm(0 * 6 + 4, i4));
        let r = t
            .wrapping_add(cm(1 * 6 + 1, i1))
            .wrapping_add(cm(1 * 6 + 5, i5));
        st_i32(out0, i, (l.wrapping_add(16384) >> 15) as i32);
        st_i32(out1, i, (r.wrapping_add(16384) >> 15) as i32);
    }
}

/// `mix8to2_s32` (`rematrix_template.c:93-106`).
fn mix8to2_s32(out0: &mut [u8], out1: &mut [u8], in_: &AudioData, c: &[i32], len: usize) {
    let (i0, i1, i2, i3, i4, i5, i6, i7) = eight_planes(in_);
    for i in 0..len {
        let cm = |k: usize, p: &[u8]| (c[k] as i64).wrapping_mul(ld_i32(p, i) as i64);
        let t = cm(0 * 8 + 2, i2).wrapping_add(cm(0 * 8 + 3, i3));
        let l = t
            .wrapping_add(cm(0 * 8 + 0, i0))
            .wrapping_add(cm(0 * 8 + 4, i4))
            .wrapping_add(cm(0 * 8 + 6, i6));
        let r = t
            .wrapping_add(cm(1 * 8 + 1, i1))
            .wrapping_add(cm(1 * 8 + 5, i5))
            .wrapping_add(cm(1 * 8 + 7, i7));
        st_i32(out0, i, (l.wrapping_add(16384) >> 15) as i32);
        st_i32(out1, i, (r.wrapping_add(16384) >> 15) as i32);
    }
}

/// The six input planes of a 5.1 buffer (`in->ch[0..6]`, `rematrix.c:806`).
/// Planar only — the internal formats are planar by construction
/// (`swresample.c:405-411`); the caller checks.
fn six_planes(in_: &AudioData) -> (&[u8], &[u8], &[u8], &[u8], &[u8], &[u8]) {
    debug_assert!(in_.ch_count >= 6);
    (
        in_.plane(0).unwrap(),
        in_.plane(1).unwrap(),
        in_.plane(2).unwrap(),
        in_.plane(3).unwrap(),
        in_.plane(4).unwrap(),
        in_.plane(5).unwrap(),
    )
}

/// The eight input planes of a 7.1 buffer (`in->ch[0..8]`).
fn eight_planes(in_: &AudioData) -> (&[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8]) {
    debug_assert!(in_.ch_count >= 8);
    (
        in_.plane(0).unwrap(),
        in_.plane(1).unwrap(),
        in_.plane(2).unwrap(),
        in_.plane(3).unwrap(),
        in_.plane(4).unwrap(),
        in_.plane(5).unwrap(),
        in_.plane(6).unwrap(),
        in_.plane(7).unwrap(),
    )
}

/// `get_mix_any_func_*` (`rematrix_template.c:108-127`) — same conditions
/// for every format: out STEREO, in 5.1 (side/back → `mix6to2`) or 7.1
/// (`mix8to2`), shared FC/LFE coefficients, and the specific cross-zero
/// pattern. `av_channel_layout_compare(...) == 0` is `==` on the native
/// layouts; `!s->matrix[a][b]` is `== 0.0` (note `-0.0 == 0.0`, as in C).
fn get_mix_any_func(
    matrix: &[[f64; SWR_CH_MAX]; SWR_CH_MAX],
    in_ch_layout: &ChannelLayout,
    out_ch_layout: &ChannelLayout,
) -> Option<AnyFastPath> {
    if *out_ch_layout == ChannelLayout::STEREO
        && (*in_ch_layout == ChannelLayout::FivePointOne
            || *in_ch_layout == ChannelLayout::FivePointOneBack)
        && matrix[0][2] == matrix[1][2]
        && matrix[0][3] == matrix[1][3]
        && matrix[0][1] == 0.0
        && matrix[0][5] == 0.0
        && matrix[1][0] == 0.0
        && matrix[1][4] == 0.0
    {
        return Some(AnyFastPath::Mix6to2);
    }

    if *out_ch_layout == ChannelLayout::STEREO
        && *in_ch_layout == ChannelLayout::SevenPointOne
        && matrix[0][2] == matrix[1][2]
        && matrix[0][3] == matrix[1][3]
        && matrix[0][1] == 0.0
        && matrix[0][5] == 0.0
        && matrix[1][0] == 0.0
        && matrix[1][4] == 0.0
        && matrix[0][7] == 0.0
        && matrix[1][6] == 0.0
    {
        return Some(AnyFastPath::Mix8to2);
    }

    None
}

/// `auto_matrix` (`rematrix.c:654-671`): pick `maxval` (explicit
/// `rematrix_maxval`, else 1.0 whenever the output or internal format has an
/// integer stage — `packed(fmt) < AV_SAMPLE_FMT_FLT` on C's enum values —
/// else `INT_MAX`), zero the matrix, delegate to [`swr_build_matrix2`].
fn auto_matrix(
    matrix: &mut [[f64; SWR_CH_MAX]; SWR_CH_MAX],
    in_ch_layout: &ChannelLayout,
    out_ch_layout: &ChannelLayout,
    out_sample_fmt: SampleFormat,
    int_sample_fmt: SampleFormat,
    options: &RematrixOptions,
    log_ctx: Option<&str>,
) -> Result<()> {
    let maxval = if options.rematrix_maxval > 0.0 {
        options.rematrix_maxval
    } else if out_sample_fmt.packed() < SampleFormat::Flt
        || int_sample_fmt.packed() < SampleFormat::Flt
    {
        1.0
    } else {
        INT_MAX_F64
    };

    // rematrix.c:666 — memset(s->matrix, 0, sizeof(s->matrix)).
    *matrix = [[0.0; SWR_CH_MAX]; SWR_CH_MAX];
    // C passes stride = s->matrix[1] - s->matrix[0] = SWR_CH_MAX.
    swr_build_matrix2(
        in_ch_layout,
        out_ch_layout,
        options.clev,
        options.slev,
        options.lfe_mix_level,
        maxval,
        options.rematrix_volume,
        matrix.as_flattened_mut(),
        SWR_CH_MAX,
        options.matrix_encoding,
        log_ctx,
    )
}

/// Two disjoint mutable planes `a < b` of a planar `AudioData`, via one
/// `data_mut()` borrow split at `b`'s start (the Rust analog of C's
/// independent `out->ch[]` pointers).
fn two_planes_mut(ad: &mut AudioData, a: usize, b: usize) -> Option<(&mut [u8], &mut [u8])> {
    debug_assert!(ad.planar);
    let plane_len = ad.count * ad.bps;
    let a_start = a * plane_len;
    let b_start = b * plane_len;
    let buf = ad.data_mut();
    if b_start + plane_len > buf.len() {
        return None;
    }
    let (lo, hi) = buf.split_at_mut(b_start);
    Some((&mut lo[a_start..a_start + plane_len], &mut hi[..plane_len]))
}

// ---------------------------------------------------------------------------
// tests — every value pinned against the C arithmetic (double-emulated;
// the s16 5.1→stereo apply additionally verified byte-exact against system
// ffmpeg 8.1.2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::channel_layout::ChannelLayout as CL;

    /// f64 bit pattern helper (byte-exactness of the default coefficients).
    fn b(x: f64) -> u64 {
        x.to_bits()
    }
    /// f32 bit pattern helper.
    fn b32(x: f32) -> u32 {
        x.to_bits()
    }

    /// `swr_build_matrix2` with the default mix levels into a 64×64 buffer.
    fn build(in_l: &CL, out_l: &CL, maxval: f64) -> Vec<f64> {
        build_opts(in_l, out_l, maxval, 1.0, &RematrixOptions::default())
    }

    fn build_opts(
        in_l: &CL,
        out_l: &CL,
        maxval: f64,
        rematrix_volume: f64,
        opts: &RematrixOptions,
    ) -> Vec<f64> {
        let mut m = vec![0.0; SWR_CH_MAX * SWR_CH_MAX];
        swr_build_matrix2(
            in_l,
            out_l,
            opts.clev,
            opts.slev,
            opts.lfe_mix_level,
            maxval,
            rematrix_volume,
            &mut m,
            SWR_CH_MAX,
            opts.matrix_encoding,
            None,
        )
        .unwrap();
        m
    }

    fn init_ctx(
        in_l: &CL,
        out_l: &CL,
        out_fmt: SampleFormat,
        int_fmt: SampleFormat,
        opts: &RematrixOptions,
    ) -> RematrixContext {
        RematrixContext::init(
            in_l,
            in_l.nb_channels,
            out_l,
            out_l.nb_channels,
            out_fmt,
            int_fmt,
            opts,
            None,
            None,
        )
        .unwrap()
    }

    fn put_i16(ad: &mut AudioData, ch: usize, samples: &[i16]) {
        let plane = ad.plane_bytes_mut(ch).unwrap();
        for (i, s) in samples.iter().enumerate() {
            st_i16(plane, i, *s);
        }
    }
    fn get_i16(ad: &AudioData, ch: usize, i: usize) -> i16 {
        ld_i16(ad.plane(ch).unwrap(), i)
    }
    fn put_i32(ad: &mut AudioData, ch: usize, samples: &[i32]) {
        let plane = ad.plane_bytes_mut(ch).unwrap();
        for (i, s) in samples.iter().enumerate() {
            st_i32(plane, i, *s);
        }
    }
    fn get_i32(ad: &AudioData, ch: usize, i: usize) -> i32 {
        ld_i32(ad.plane(ch).unwrap(), i)
    }
    fn put_f32(ad: &mut AudioData, ch: usize, samples: &[f32]) {
        let plane = ad.plane_bytes_mut(ch).unwrap();
        for (i, s) in samples.iter().enumerate() {
            st_f32(plane, i, *s);
        }
    }
    fn get_f32(ad: &AudioData, ch: usize, i: usize) -> f32 {
        ld_f32(ad.plane(ch).unwrap(), i)
    }

    // ---- constants (swresample_internal.h:28-32) --------------------------

    #[test]
    fn constants_match_c_literals() {
        assert_eq!(b(M_SQRT1_2), 0x3FE6A09E667F3BCD);
        assert_eq!(b(SQRT1_3), 0x3FE279A74590331C);
        assert_eq!(b(SQRT2_3), 0x3FEA20BD700C2C3E);
        assert_eq!(b(SQRT3_2), 0x3FF3988E1409212E);
        // The hazard the spec pins: 1.0/SQRT_2 is 1 ulp LOWER.
        assert_ne!(b(1.0 / std::f64::consts::SQRT_2), b(M_SQRT1_2));
        assert_eq!(NUM_NAMED_CHANNELS, 41);
    }

    // ---- build_matrix: defaults, normalization, volume ---------------------

    #[test]
    fn matrix_identity_stereo_to_stereo() {
        let m = build(&CL::STEREO, &CL::STEREO, 1.0);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[SWR_CH_MAX + 0], 0.0);
        assert_eq!(m[SWR_CH_MAX + 1], 1.0);
        // No cross-contamination beyond nb_channels.
        assert_eq!(m[2 * SWR_CH_MAX + 2], 0.0);
    }

    #[test]
    fn stereo_to_mono_exact_halves() {
        let m = build(&CL::STEREO, &CL::MONO, 1.0);
        assert_eq!(b(m[0]), 0x3FE0_0000_0000_0000); // 0.5 exactly
        assert_eq!(b(m[1]), 0x3FE0_0000_0000_0000);
        // No normalization on the float path: stays [sqrt(1/2), sqrt(1/2)].
        let m = build(&CL::STEREO, &CL::MONO, INT_MAX_F64);
        assert_eq!(b(m[0]), 0x3FE6A09E667F3BCD);
        assert_eq!(b(m[1]), 0x3FE6A09E667F3BCD);
    }

    #[test]
    fn mono_to_stereo_unnormalized() {
        // maxcoef = sqrt(1/2) < 1: NO normalization under either maxval —
        // the mirror image of stereo->mono.
        for maxval in [1.0, INT_MAX_F64] {
            let m = build(&CL::MONO, &CL::STEREO, maxval);
            assert_eq!(b(m[0]), 0x3FE6A09E667F3BCD, "maxval {maxval}");
            assert_eq!(b(m[SWR_CH_MAX]), 0x3FE6A09E667F3BCD);
        }
    }

    #[test]
    fn five_point_one_to_stereo_exact() {
        for in_l in [&CL::FivePointOne, &CL::FivePointOneBack] {
            let m = build(in_l, &CL::STEREO, 1.0);
            // L row = [1, 0, clev, 0, slev, 0] / (1 + 2*sqrt(1/2)).
            assert_eq!(b(m[0]), 0x3FDA827999FCEF33, "{:?}", in_l.describe());
            assert_eq!(m[1], 0.0);
            assert_eq!(b(m[2]), 0x3FD2BEC333018868);
            assert_eq!(m[3], 0.0);
            assert_eq!(b(m[4]), 0x3FD2BEC333018868);
            assert_eq!(m[5], 0.0);
            // R row mirrored.
            assert_eq!(m[SWR_CH_MAX + 0], 0.0);
            assert_eq!(b(m[SWR_CH_MAX + 1]), 0x3FDA827999FCEF33);
            assert_eq!(b(m[SWR_CH_MAX + 2]), 0x3FD2BEC333018868);
            assert_eq!(b(m[SWR_CH_MAX + 5]), 0x3FD2BEC333018868);
            // Float path (maxval = INT_MAX): the raw [1, 0, sqrt(1/2), ...].
            let m = build(in_l, &CL::STEREO, INT_MAX_F64);
            assert_eq!(m[0], 1.0);
            assert_eq!(b(m[2]), 0x3FE6A09E667F3BCD);
            assert_eq!(b(m[4]), 0x3FE6A09E667F3BCD);
        }
    }

    #[test]
    fn lfe_mix_level_one() {
        let opts = RematrixOptions {
            lfe_mix_level: 1.0,
            ..RematrixOptions::default()
        };
        // Unnormalized: LFE coefficient sqrt(1/2) on both outputs.
        let m = build_opts(&CL::FivePointOne, &CL::STEREO, INT_MAX_F64, 1.0, &opts);
        assert_eq!(b(m[3]), 0x3FE6A09E667F3BCD);
        assert_eq!(b(m[SWR_CH_MAX + 3]), 0x3FE6A09E667F3BCD);
        // Normalized: maxcoef = 1 + 3*sqrt(1/2) (FC + LFE + SL all sqrt(1/2)).
        let m = build_opts(&CL::FivePointOne, &CL::STEREO, 1.0, 1.0, &opts);
        assert_eq!(b(m[0]), b(1.0 / 3.121_320_343_559_642_4));
        assert_eq!(b(m[3]), b(M_SQRT1_2 / 3.121_320_343_559_642_4));
        assert_eq!(b(m[3]), 0x3FCCFF4AF8932960);
    }

    #[test]
    fn stereo_to_mono_volume_and_maxval() {
        // volume 0.5: normalized 0.5 then *0.5 -> 0.25 (both entries).
        let m = build_opts(
            &CL::STEREO,
            &CL::MONO,
            1.0,
            0.5,
            &RematrixOptions::default(),
        );
        assert_eq!(b(m[0]), 0x3FD0_0000_0000_0000);
        assert_eq!(b(m[1]), 0x3FD0_0000_0000_0000);
        // Negative volume: maxcoef := -volume = 1, so the row is divided by
        // maxcoef/1 = 1 — it keeps its PRE-normalization values sqrt(1/2)
        // (not the normalized halves!) and the *volume multiply is skipped.
        let m = build_opts(
            &CL::STEREO,
            &CL::MONO,
            1.0,
            -1.0,
            &RematrixOptions::default(),
        );
        assert_eq!(b(m[0]), 0x3FE6A09E667F3BCD);
        assert_eq!(b(m[1]), 0x3FE6A09E667F3BCD);
        // rematrix_maxval = 0.5: maxcoef := 0.5 -> divide by (1.414../0.5).
        let opts = RematrixOptions {
            rematrix_maxval: 0.5,
            ..RematrixOptions::default()
        };
        let m = build_opts(&CL::STEREO, &CL::MONO, 0.5, 1.0, &opts);
        assert_eq!(b(m[0]), 0x3FD0_0000_0000_0000);
        assert_eq!(b(m[1]), 0x3FD0_0000_0000_0000);
    }

    #[test]
    fn dolby_dplii_surround() {
        // Pre-normalization doubles (maxval = INT_MAX), 5.1(back) -> stereo:
        // unaccounted BL/BR fold into FL/FR with the encoding's signs.
        let dolby = RematrixOptions {
            matrix_encoding: MatrixEncoding::Dolby,
            ..RematrixOptions::default()
        };
        let m = build_opts(&CL::FivePointOneBack, &CL::STEREO, INT_MAX_F64, 1.0, &dolby);
        assert_eq!(b(m[4]), b(-M_SQRT1_2 * M_SQRT1_2)); // FL <- BL negative
        assert_eq!(b(m[SWR_CH_MAX + 4]), b(M_SQRT1_2 * M_SQRT1_2)); // FR <- BL
        assert_eq!(b(m[5]), b(-M_SQRT1_2 * M_SQRT1_2));
        assert_eq!(b(m[SWR_CH_MAX + 5]), b(M_SQRT1_2 * M_SQRT1_2));
        // FC still clev on both rows; LFE 0.
        assert_eq!(b(m[2]), b(M_SQRT1_2));
        assert_eq!(m[3], 0.0);

        let dplii = RematrixOptions {
            matrix_encoding: MatrixEncoding::Dplii,
            ..RematrixOptions::default()
        };
        let m = build_opts(&CL::FivePointOneBack, &CL::STEREO, INT_MAX_F64, 1.0, &dplii);
        assert_eq!(b(m[4]), b(-M_SQRT1_2 * SQRT3_2)); // FL,BL = -slev*sqrt(3/2)
        assert_eq!(b(m[5]), b(-M_SQRT1_2 * M_SQRT1_2)); // FL,BR = -slev*sqrt(1/2)
        assert_eq!(b(m[SWR_CH_MAX + 4]), b(M_SQRT1_2 * M_SQRT1_2)); // FR,BL
        assert_eq!(b(m[SWR_CH_MAX + 5]), b(M_SQRT1_2 * SQRT3_2)); // FR,BR
    }

    #[test]
    fn top_channel_constants() {
        // TSL/TSR with both TFL and TBC present (ITU-R BS.2127-1):
        // UH+180 = sqrt(1/3); U±045 = sqrt(2/3).
        let in_l = CL::from_mask(
            ch_bit(Channel::FrontLeft)
                | ch_bit(Channel::FrontRight)
                | ch_bit(Channel::TopSideLeft)
                | ch_bit(Channel::TopSideRight),
        )
        .unwrap();
        let out_l = CL::from_mask(
            ch_bit(Channel::FrontLeft)
                | ch_bit(Channel::FrontRight)
                | ch_bit(Channel::TopFrontLeft)
                | ch_bit(Channel::TopFrontRight)
                | ch_bit(Channel::TopBackCenter),
        )
        .unwrap();
        let m = build(&in_l, &out_l, INT_MAX_F64);
        // out indices: FL=0 FR=1 TFL=2 TFR=3 TBC=4; in: FL=0 FR=1 TSL=2 TSR=3.
        assert_eq!(m[2 * SWR_CH_MAX + 2], SQRT2_3); // TFL <- TSL
        assert_eq!(m[3 * SWR_CH_MAX + 3], SQRT2_3); // TFR <- TSR
        assert_eq!(m[4 * SWR_CH_MAX + 2], SQRT1_3); // TBC <- TSL
        assert_eq!(m[4 * SWR_CH_MAX + 3], SQRT1_3); // TBC <- TSR

        // TC with both TFL and TBL: everything 0.5 (sqrt(1/4)).
        let in_l = CL::from_mask(CL::STEREO.mask | ch_bit(Channel::TopCenter)).unwrap();
        let out_l = CL::from_mask(
            CL::STEREO.mask
                | ch_bit(Channel::TopFrontLeft)
                | ch_bit(Channel::TopFrontRight)
                | ch_bit(Channel::TopBackLeft)
                | ch_bit(Channel::TopBackRight),
        )
        .unwrap();
        let m = build(&in_l, &out_l, INT_MAX_F64);
        // in: FL=0 FR=1 TC=2; out: FL=0 FR=1 TFL=2 TFR=3 TBL=4 TBR=5.
        assert_eq!(m[2 * SWR_CH_MAX + 2], 0.5);
        assert_eq!(m[3 * SWR_CH_MAX + 2], 0.5);
        assert_eq!(m[4 * SWR_CH_MAX + 2], 0.5);
        assert_eq!(m[5 * SWR_CH_MAX + 2], 0.5);
    }

    #[test]
    fn stereo_downmix_translation() {
        // out STEREO_DOWNMIX + in without DL/DR -> out becomes STEREO
        // (rematrix.c:587-592): identical to a plain STEREO output.
        let a = build(&CL::SURROUND, &CL::STEREO_DOWNMIX, 1.0);
        let c = build(&CL::SURROUND, &CL::STEREO, 1.0);
        assert_eq!(a, c);
        // in STEREO_DOWNMIX + out without DL/DR -> in becomes STEREO
        // (rematrix.c:593-598): identity matrix.
        let m = build(&CL::STEREO_DOWNMIX, &CL::STEREO, 1.0);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[SWR_CH_MAX + 1], 1.0);
        // BOTH downmix: no translation fires (each side's subset carries
        // DL|DR), and STEREO_DOWNMIX then fails sane_layout (no front
        // speaker) — exactly C's behavior.
        let err = swr_build_matrix2(
            &CL::STEREO_DOWNMIX,
            &CL::STEREO_DOWNMIX,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument("Input channel layout 'downmix' is not supported".into())
        );
    }

    // ---- layout checks / errors --------------------------------------------

    #[test]
    fn sane_layout_rejections() {
        // Asymmetric side pair (nb != 1, so clean_layout does not coerce).
        let asym = CL::from_mask(CL::STEREO.mask | ch_bit(Channel::SideLeft)).unwrap();
        let err = swr_build_matrix2(
            &asym,
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument(
                "Input channel layout '3 channels (FL+FR+SL)' is not supported".into()
            )
        );

        // A single FL channel IS coerced to mono by clean_layout
        // (rematrix.c:103-108) — it builds fine (mirror of UNSPEC-1).
        swr_build_matrix2(
            &CL::from_mask(ch_bit(Channel::FrontLeft)).unwrap(),
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap();

        let back_only =
            CL::from_mask(ch_bit(Channel::BackLeft) | ch_bit(Channel::BackRight)).unwrap();
        let err = swr_build_matrix2(
            &back_only,
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument(
                "Input channel layout '2 channels (BL+BR)' is not supported".into()
            )
        );

        // UNSPEC never passes sane_layout (rematrix.c:126-127).
        let err = swr_build_matrix2(
            &CL::unspecified(2),
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument("Input channel layout '2 channels' is not supported".into())
        );

        // >= SWR_CH_MAX channels (rematrix.c:115-116).
        let err = swr_build_matrix2(
            &CL::unspecified(64),
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument("Input channel layout '64 channels' is not supported".into())
        );

        // The output side, with its own message (rematrix.c:617-619).
        let err = swr_build_matrix2(
            &CL::STEREO,
            &asym,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut vec![0.0; SWR_CH_MAX * SWR_CH_MAX],
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument(
                "Output channel layout '3 channels (FL+FR+SL)' is not supported".into()
            )
        );
    }

    #[test]
    fn invalid_layout_is_invalid() {
        // check() fails: popcount 2 != nb 3 (rematrix.c:600-604).
        let bad = CL {
            order: Order::Native,
            nb_channels: 3,
            mask: CL::STEREO.mask,
        };
        let mut m = vec![0.0; SWR_CH_MAX * SWR_CH_MAX];
        let err = swr_build_matrix2(
            &bad,
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut m,
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument("Input channel layout is invalid".into())
        );
        let err = swr_build_matrix2(
            &CL::STEREO,
            &bad,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut m,
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument("Output channel layout is invalid".into())
        );
    }

    #[test]
    fn unspecified_mono_treated_as_mono() {
        // clean_layout coerces UNSPEC-1 to MONO (rematrix.c:103-108): the
        // matrices are identical.
        let a = build(&CL::unspecified(1), &CL::STEREO, 1.0);
        let c = build(&CL::MONO, &CL::STEREO, 1.0);
        assert_eq!(a, c);
    }

    #[test]
    fn sane_layout_direct() {
        assert!(sane_layout(&CL::MONO));
        assert!(sane_layout(&CL::STEREO));
        assert!(sane_layout(&CL::SURROUND));
        assert!(sane_layout(&CL::FivePointOne));
        assert!(sane_layout(&CL::FivePointOneBack));
        assert!(sane_layout(&CL::SevenPointOne));
        assert!(sane_layout(&CL::QUAD));
        assert!(sane_layout(&CL::TwentyTwoTwo)); // 24 channels, all pairs even
        // No front speaker (DL/DR only), asymmetric side, too many channels,
        // and UNSPEC all fail (a lone FL also fails — but clean_layout
        // coerces it to mono before sane_layout ever sees it).
        assert!(!sane_layout(&CL::STEREO_DOWNMIX));
        assert!(!sane_layout(
            &CL::from_mask(CL::STEREO.mask | ch_bit(Channel::SideLeft)).unwrap()
        ));
        assert!(!sane_layout(&CL::unspecified(64)));
        assert!(!sane_layout(&CL::unspecified(2)));
        // even(): true for 0 or >= 2 bits, false for exactly one.
        assert!(even(0));
        assert!(!even(1));
        assert!(even(3));
        assert!(even(0b111));
    }

    #[test]
    fn matrix_buffer_too_small() {
        let mut m = vec![0.0; 10];
        let err = swr_build_matrix2(
            &CL::STEREO,
            &CL::STEREO,
            M_SQRT1_2,
            M_SQRT1_2,
            0.0,
            1.0,
            1.0,
            &mut m,
            SWR_CH_MAX,
            MatrixEncoding::None,
            None,
        )
        .unwrap_err();
        assert_eq!(err, Error::BufferTooSmall);
    }

    // ---- swr_set_matrix -----------------------------------------------------

    #[test]
    fn swr_set_matrix_guards_and_copy() {
        // Already-initialized guard (rematrix.c:75).
        let err = swr_set_matrix(&CL::STEREO, &CL::STEREO, &[1.0; 4], 2, None, true).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        // swri_check_chlayout: nb_channels == 0 fails check (swresample.c:35-43).
        let err =
            swr_set_matrix(&CL::default(), &CL::STEREO, &[1.0; 4], 2, None, false).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidArgument("input channel layout \"\" is invalid or unsupported.".into())
        );
        // Valid 2x2 custom matrix, stride 2.
        let c = swr_set_matrix(
            &CL::STEREO,
            &CL::STEREO,
            &[0.25, 0.75, 0.75, 0.25],
            2,
            None,
            false,
        )
        .unwrap();
        assert_eq!(c.in_nb_channels, 2);
        assert_eq!(c.out_nb_channels, 2);
        assert_eq!(c.matrix[0][..2], [0.25, 0.75]);
        assert_eq!(c.matrix[1][..2], [0.75, 0.25]);
        // Everything outside the user rectangle is zero (the :80 memset).
        assert_eq!(c.matrix[0][2], 0.0);
        assert_eq!(c.matrix[2][0], 0.0);
    }

    // ---- swri_rematrix_init: quantization -----------------------------------

    #[test]
    fn native_s16_error_diffusion() {
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::STEREO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        let (coeffs, one) = ctx.native_matrix().as_int().unwrap();
        assert_eq!(one, 32768);
        // Row L: the diffusion pushes the -0.9/-0.2 residuals forward —
        // the zero LFE column even receives a 1.
        assert_eq!(&coeffs[0..6], &[13573, 0, 9597, 0, 9598, 1]);
        assert_eq!(&coeffs[6..12], &[0, 13573, 9597, 0, 1, 9598]);
        // maxsum = 13573+9597+1+9598 = 32769 > 32768 -> clip variants
        // (rematrix.c:717) and the 5.1->stereo fast path.
        assert!(ctx.clip_s16);
        assert_eq!(ctx.mix_any, Some(AnyFastPath::Mix6to2));
    }

    #[test]
    fn matrix32_no_diffusion_and_sparsity() {
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::STEREO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        // matrix32 = lrintf(coeff*32768) per entry, no residual carry.
        assert_eq!(&ctx.matrix32[0][..6], &[13573, 0, 9598, 0, 9598, 0]);
        assert_eq!(&ctx.matrix32[1][..6], &[0, 13573, 9598, 0, 0, 9598]);
        // matrix_ch counts DOUBLE-matrix non-zeros in increasing input order
        // — the +-1 diffused taps of native_matrix are NOT counted.
        assert_eq!(ctx.matrix_ch[0][0], 3);
        assert_eq!(ctx.matrix_ch[0][1], 0); // FL
        assert_eq!(ctx.matrix_ch[0][2], 2); // FC
        assert_eq!(ctx.matrix_ch[0][3], 4); // SL
        assert_eq!(ctx.matrix_ch[0][4], 0); // rest zero
        assert_eq!(ctx.matrix_ch[1][0], 3);
        assert_eq!(ctx.matrix_ch[1][1], 1); // FR
        assert_eq!(ctx.matrix_ch[1][2], 2); // FC
        assert_eq!(ctx.matrix_ch[1][3], 5); // SR
    }

    #[test]
    fn custom_matrix_no_normalization() {
        // The swr_set_matrix path uses the values verbatim: no maxval, no
        // rematrix_volume, native quantized straight from them.
        let custom = swr_set_matrix(
            &CL::STEREO,
            &CL::STEREO,
            &[0.25, 0.75, 0.75, 0.25],
            2,
            None,
            false,
        )
        .unwrap();
        let ctx = RematrixContext::init(
            &CL::STEREO,
            2,
            &CL::STEREO,
            2,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
            Some(&custom),
            None,
        )
        .unwrap();
        assert_eq!(ctx.matrix[0][..2], [0.25, 0.75]);
        let (coeffs, _) = ctx.native_matrix().as_int().unwrap();
        assert_eq!(&coeffs[0..2], &[8192, 24576]);
        assert_eq!(&coeffs[2..4], &[24576, 8192]);
        // No fast path for custom matrices through sane stereo (zeros at
        // [0][1]/[1][0] hold, but the clev-sharing conditions fail) —
        // matrix_ch rows both have 2 taps.
        assert_eq!(ctx.matrix_ch[0][0], 2);
        assert_eq!(ctx.mix_any, None);
    }

    #[test]
    fn lrintf_ties_to_even() {
        // Round-half-to-even on the f32-converted target.
        assert_eq!(lrintf(0.5), 0);
        assert_eq!(lrintf(1.5), 2);
        assert_eq!(lrintf(2.5), 2);
        assert_eq!(lrintf(3.5), 4);
        assert_eq!(lrintf(-2.5), -2);
        // The conversion to f32 happens FIRST: 2.5000001 is not a tie in
        // double (rounds to 3) but its f32 value is exactly 2.5 -> 2.
        assert_eq!(lrintf(2.5000001), 2);
        assert_eq!(lrintf(2.5000001e0), 2);
        // Through the quantizer (custom 1x1 mono->mono matrices).
        for (coeff, want) in [
            (3.0 / 65536.0, 2i32),    // target 1.5 -> 2
            (5.0 / 65536.0, 2),       // target 2.5 -> 2
            (7.0 / 65536.0, 4),       // target 3.5 -> 4
            (2.5000001 / 32768.0, 2), // f32(2.5000001) = 2.5 -> 2
        ] {
            let custom = swr_set_matrix(&CL::MONO, &CL::MONO, &[coeff], 1, None, false).unwrap();
            let ctx = RematrixContext::init(
                &CL::MONO,
                1,
                &CL::MONO,
                1,
                SampleFormat::S16,
                SampleFormat::S16p,
                &RematrixOptions::default(),
                Some(&custom),
                None,
            )
            .unwrap();
            let (c, _) = ctx.native_matrix().as_int().unwrap();
            assert_eq!(c[0], want, "coeff {coeff}");
            // matrix32 (no diffusion) quantizes identically for 1 tap.
            assert_eq!(ctx.matrix32[0][0], want);
        }
    }

    #[test]
    fn s64p_unsupported() {
        let err = RematrixContext::init(
            &CL::MONO,
            1,
            &CL::MONO,
            1,
            SampleFormat::S64,
            SampleFormat::S64p,
            &RematrixOptions::default(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            Error::Unsupported(msg) => {
                assert!(msg.contains("rematrix.c:766"), "{msg}");
                assert!(msg.contains("s64p"), "{msg}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn mix8to2_selection_and_quantization() {
        // 7.1 -> stereo: L row = [1, 0, clev, 0, slev, 0, slev, 0]
        // (SL and BL both fold in), maxcoef = 1 + 3*sqrt(1/2).
        let ctx = init_ctx(
            &CL::SevenPointOne,
            &CL::STEREO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        assert_eq!(ctx.mix_any, Some(AnyFastPath::Mix8to2));
        let (coeffs, _) = ctx.native_matrix().as_int().unwrap();
        assert_eq!(&coeffs[0..8], &[10498, 0, 7424, 0, 7423, -1, 7423, 0]);
        assert_eq!(&coeffs[8..16], &[0, 10498, 7423, 1, 0, 7423, 1, 7423]);
        assert!(ctx.clip_s16);
        // Apply the fast path: t = FC*c + LFE*c shared, 3 front/side terms
        // each (rematrix_template.c:102-105).
        let mut in_ad = AudioData::new(SampleFormat::S16p, 8, 1);
        for (ch, s) in [
            (0, 10000i16),
            (1, -20000),
            (2, 3000),
            (3, 4000),
            (4, -5000),
            (5, 6000),
            (6, -7000),
            (7, 8000),
        ] {
            put_i16(&mut in_ad, ch, &[s]);
        }
        let mut out_ad = AudioData::new(SampleFormat::S16p, 2, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert_eq!(get_i16(&out_ad, 0, 0), 1165);
        assert_eq!(get_i16(&out_ad, 1, 0), -2556);
    }

    #[test]
    fn fltp_quantization() {
        // All-float path (maxval = INT_MAX): native = f32 of the doubles,
        // one = 1.0, and matrix_flt carries the same values.
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::STEREO,
            SampleFormat::Flt,
            SampleFormat::Fltp,
            &RematrixOptions::default(),
        );
        let (coeffs, one) = ctx.native_matrix().as_float().unwrap();
        assert_eq!(one, 1.0);
        assert_eq!(b32(coeffs[0]), 0x3F800000); // 1.0
        assert_eq!(b32(coeffs[2]), 0x3F3504F3); // f32(sqrt(1/2))
        assert_eq!(b32(coeffs[4]), 0x3F3504F3);
        assert_eq!(b32(ctx.matrix_flt[0][4]), 0x3F3504F3);
        assert_eq!(ctx.mix_any, Some(AnyFastPath::Mix6to2));
        assert!(!ctx.clip_s16);
    }

    // ---- swri_rematrix: apply -----------------------------------------------

    #[test]
    fn apply_s16_mix6to2_clip_matches_ffmpeg() {
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::STEREO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        let mut in_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        for (ch, s) in [
            (0, 10000i16),
            (1, -20000),
            (2, 3000),
            (3, 4000),
            (4, -5000),
            (5, 6000),
        ] {
            put_i16(&mut in_ad, ch, &[s]);
        }
        let mut out_ad = AudioData::new(SampleFormat::S16p, 2, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        // Byte-exact against system ffmpeg 8.1.2 (s16le 5.1 -> stereo).
        assert_eq!(get_i16(&out_ad, 0, 0), 3556);
        assert_eq!(get_i16(&out_ad, 1, 0), -5648);
    }

    #[test]
    fn apply_fltp_mix6to2() {
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::STEREO,
            SampleFormat::Flt,
            SampleFormat::Fltp,
            &RematrixOptions::default(),
        );
        let mut in_ad = AudioData::new(SampleFormat::Fltp, 6, 1);
        for (ch, s) in [
            (0, 0.5f32),
            (1, -0.25),
            (2, 0.125),
            (3, 10.0),
            (4, -0.75),
            (5, 0.375),
        ] {
            put_f32(&mut in_ad, ch, &[s]);
        }
        let mut out_ad = AudioData::new(SampleFormat::Fltp, 2, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        // t = 0.125*sqrt(1/2) + 10*0 (the LFE coefficient is 0 by default!)
        // = 0.088388346; L = ((t + 0.5*1.0) + (-0.75)*sqrt(1/2)) in f32 at
        // every step, left-associative.
        assert_eq!(b32(get_f32(&out_ad, 0, 0)), 0x3D6DCE80); // 0.058058262
        assert_eq!(b32(get_f32(&out_ad, 1, 0)), 0x3DD413CC); // 0.103553385
    }

    #[test]
    fn apply_generic_mono_s16() {
        // 5.1 -> mono defeats every fast path (out != stereo) and lands in
        // the generic >= 3-tap branch: row = [sqrt(1/2), sqrt(1/2),
        // clev*sqrt(2), 0, slev*sqrt(1/2), slev*sqrt(1/2)] / 3.4142...,
        // matrix32 = [6786, 6786, 9598, 0, 4799, 4799].
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::MONO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        assert_eq!(ctx.mix_any, None);
        assert_eq!(ctx.matrix_ch[0][0], 5);
        assert_eq!(&ctx.matrix32[0][..6], &[6786, 6786, 9598, 0, 4799, 4799]);
        let mut in_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        for (ch, s) in [
            (0, 10000i16),
            (1, -20000),
            (2, 3000),
            (3, 4000),
            (4, -5000),
            (5, 6000),
        ] {
            put_i16(&mut in_ad, ch, &[s]);
        }
        let mut out_ad = AudioData::new(SampleFormat::S16p, 1, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert_eq!(get_i16(&out_ad, 0, 0), -1046);
    }

    #[test]
    fn apply_generic_mono_fltp() {
        // Same layout on the float path: matrix_flt = f32 of the
        // UNNORMALIZED row (maxval INT_MAX), f32 accumulation.
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::MONO,
            SampleFormat::Flt,
            SampleFormat::Fltp,
            &RematrixOptions::default(),
        );
        assert_eq!(b32(ctx.matrix_flt[0][0]), 0x3F3504F3); // f32(sqrt(1/2))
        assert_eq!(b32(ctx.matrix_flt[0][2]), 0x3F800000); // f32(clev*sqrt(2)) = exactly 1.0
        assert_eq!(b32(ctx.matrix_flt[0][4]), 0x3F000000); // f32(slev*sqrt(1/2)) = exactly 0.5
        let mut in_ad = AudioData::new(SampleFormat::Fltp, 6, 1);
        for (ch, s) in [
            (0, 10000f32),
            (1, -20000.0),
            (2, 3000.0),
            (3, 4000.0),
            (4, -5000.0),
            (5, 6000.0),
        ] {
            put_f32(&mut in_ad, ch, &[s]);
        }
        let mut out_ad = AudioData::new(SampleFormat::Fltp, 1, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert_eq!(b32(get_f32(&out_ad, 0, 0)), 0xC55F3116); // -3571.06787109375
    }

    #[test]
    fn apply_generic_s32p_upstream_bug() {
        // rematrix.c:866-875 replicated: the generic integer branch reads
        // int16_t samples — the LOW 16 bits of each i32 — and stores the
        // shifted result sign-extended (C leaves the upper half stale).
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::MONO,
            SampleFormat::S32,
            SampleFormat::S32p,
            &RematrixOptions::default(),
        );
        assert_eq!(ctx.matrix_ch[0][0], 5);
        let mut in_ad = AudioData::new(SampleFormat::S32p, 6, 1);
        for (ch, s) in [
            (0, 100000i32),
            (1, -200000),
            (2, 30000),
            (3, 40000),
            (4, -50000),
            (5, 60000),
        ] {
            put_i32(&mut in_ad, ch, &[s]);
        }
        // Low halves (LE): [-31072, -3392, 30000, (skipped), 15536, -5536]
        // -> v = 102057296 -> (v+16384)>>15 = 3115.
        let mut out_ad = AudioData::new(SampleFormat::S32p, 1, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert_eq!(get_i32(&out_ad, 0, 0), 3115);
    }

    #[test]
    fn apply_quad_sum2_two_tap() {
        // 5.1(side) -> quad: FL/FR rows have 2 taps (sum2 path, no
        // diffusion: [19195, 0, 13573, ...]), BL/BR one tap of 19195.
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::QUAD,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        assert_eq!(ctx.mix_any, None);
        assert_eq!(ctx.matrix_ch[0][0], 2);
        assert_eq!(&ctx.matrix32[0][..6], &[19195, 0, 13573, 0, 0, 0]);
        assert_eq!(&ctx.matrix32[2][..6], &[0, 0, 0, 0, 19195, 0]);
        let mut in_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        for (ch, s) in [
            (0, 10000i16),
            (1, -20000),
            (2, 3000),
            (3, 4000),
            (4, -5000),
            (5, 6000),
        ] {
            put_i16(&mut in_ad, ch, &[s]);
        }
        let mut out_ad = AudioData::new(SampleFormat::S16p, 4, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert_eq!(get_i16(&out_ad, 0, 0), 7100); // FL
        assert_eq!(get_i16(&out_ad, 1, 0), -10473); // FR
        assert_eq!(get_i16(&out_ad, 2, 0), -2929); // BL <- SL
        assert_eq!(get_i16(&out_ad, 3, 0), 3515); // BR <- SR
    }

    #[test]
    fn mustcopy_zero_and_steal() {
        // (a) Custom matrix with an empty row: mustcopy zeroes the output
        // plane, otherwise it is left untouched (rematrix.c:820-823).
        let custom = swr_set_matrix(
            &CL::STEREO,
            &CL::STEREO,
            &[0.0, 0.0, 1.0, 1.0],
            2,
            None,
            false,
        )
        .unwrap();
        let ctx = RematrixContext::init(
            &CL::STEREO,
            2,
            &CL::STEREO,
            2,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
            Some(&custom),
            None,
        )
        .unwrap();
        assert_eq!(ctx.matrix_ch[0][0], 0);
        assert_eq!(ctx.matrix_ch[1][0], 2);
        let mut in_ad = AudioData::new(SampleFormat::S16p, 2, 1);
        put_i16(&mut in_ad, 0, &[100]);
        put_i16(&mut in_ad, 1, &[-60]);

        let mut out_ad = AudioData::new(SampleFormat::S16p, 2, 1);
        out_ad.data_mut().iter_mut().for_each(|x| *x = 0xAA);
        ctx.rematrix(&mut out_ad, &in_ad, 1, true).unwrap();
        assert_eq!(get_i16(&out_ad, 0, 0), 0); // zeroed
        assert_eq!(get_i16(&out_ad, 1, 0), 40); // sum2 with coeffs 32768/32768
        // (identity sum): (100*32768 - 60*32768 + 16384)>>15 = 1327104>>15
        // = 40 — the +16384 rounding keeps the exact 40.

        let mut out_ad = AudioData::new(SampleFormat::S16p, 2, 1);
        out_ad.data_mut().iter_mut().for_each(|x| *x = 0xAA);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert_eq!(get_i16(&out_ad, 0, 0), i16::from_le_bytes([0xAA, 0xAA])); // untouched
        assert_eq!(get_i16(&out_ad, 1, 0), 40);

        // (b) Identity matrix (5.1 -> 5.1) without mustcopy: the whole
        // backing Arc is stolen (C rematrix.c:834 per-channel steal;
        // divergence documented in the module doc).
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::FivePointOne,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        let mut in_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        put_i16(&mut in_ad, 0, &[42]);
        let mut out_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, false).unwrap();
        assert!(std::sync::Arc::ptr_eq(&out_ad.data, &in_ad.data));
        assert_eq!(get_i16(&out_ad, 0, 0), 42);
        // With mustcopy the planes are copied instead.
        let mut out_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        ctx.rematrix(&mut out_ad, &in_ad, 1, true).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&out_ad.data, &in_ad.data));
        assert_eq!(get_i16(&out_ad, 0, 0), 42);

        // (c) copy-with-coefficient path (matrix != 1.0): mono -> stereo
        // keeps sqrt(1/2) (maxcoef < 1 -> no normalization), native coeff
        // lrintf(sqrt(1/2)*32768) = 23170.
        let ctx = init_ctx(
            &CL::MONO,
            &CL::STEREO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        let (coeffs, _) = ctx.native_matrix().as_int().unwrap();
        assert_eq!(coeffs, [23170, 23170]);
        let mut in_ad = AudioData::new(SampleFormat::S16p, 1, 2);
        put_i16(&mut in_ad, 0, &[1000, -3]);
        let mut out_ad = AudioData::new(SampleFormat::S16p, 2, 2);
        ctx.rematrix(&mut out_ad, &in_ad, 2, false).unwrap();
        assert_eq!(get_i16(&out_ad, 0, 0), 707); // (1000*23170+16384)>>15
        assert_eq!(get_i16(&out_ad, 0, 1), -2); // (-3*23170+16384)>>15
        assert_eq!(get_i16(&out_ad, 1, 0), 707);
    }

    #[test]
    fn case2_sum2_rounding() {
        // Direct kernel: stereo->mono native [16384, 16384] — the +16384
        // rounds halves up (arithmetic shift of negative sums toward zero).
        let mut out = vec![0u8; 2];
        let s = |v: i16| v.to_le_bytes().to_vec();
        sum2_s16(&mut out, &s(1), &s(2), 16384, 16384, 1);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 2); // 1.5 -> 2
        sum2_s16(&mut out, &s(1), &s(0), 16384, 16384, 1);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 1); // 0.5 -> 1
        sum2_s16(&mut out, &s(-3), &s(0), 16384, 16384, 1);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), -1); // -1.5 -> -1
        sum2_s16(&mut out, &s(32767), &s(32767), 16384, 16384, 1);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        // Clip variant overdrives: coeffs 32768/32768 on full-scale input
        // exceed int16 after the shift -> av_clip_int16.
        sum2_clip_s16(&mut out, &s(32767), &s(32767), 32768, 32768, 1);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        sum2_clip_s16(&mut out, &s(-32768), &s(-32768), 32768, 32768, 1);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), -32768);
        // s32 kernel: i64 accumulate, truncated store.
        let mut out = vec![0u8; 4];
        let s32b = |v: i32| v.to_le_bytes().to_vec();
        sum2_s32(&mut out, &s32b(1), &s32b(2), 16384, 16384, 1);
        assert_eq!(i32::from_le_bytes([out[0], out[1], out[2], out[3]]), 2);
    }

    #[test]
    fn apply_buffer_too_small() {
        let ctx = init_ctx(
            &CL::FivePointOne,
            &CL::STEREO,
            SampleFormat::S16,
            SampleFormat::S16p,
            &RematrixOptions::default(),
        );
        let in_ad = AudioData::new(SampleFormat::S16p, 6, 1);
        let mut out_ad = AudioData::new(SampleFormat::S16p, 2, 1);
        // len 2 into count-1 planes: 4 bytes needed, 2 present.
        assert_eq!(
            ctx.rematrix(&mut out_ad, &in_ad, 2, false),
            Err(Error::BufferTooSmall)
        );
        // Non-planar buffers are rejected outright (C would only ever pass
        // the planar internal formats).
        let mut packed = AudioData::new(SampleFormat::S16, 2, 4);
        assert!(matches!(
            ctx.rematrix(&mut packed, &in_ad, 1, false),
            Err(Error::InvalidArgument(_))
        ));
    }
}
