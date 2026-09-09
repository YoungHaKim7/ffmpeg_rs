//! Audio sample-format conversion — port of `libswresample/audioconvert.c`
//! (265 lines) plus the `AudioConvert`/`swri_audio_convert` contract from
//! `libswresample/audioconvert.h:36-77`.
//!
//! ## C → Rust map
//!
//! | C site | port |
//! |---|---|
//! | `CONV_FUNC` macro `audioconvert.c:35-51` (FIXME "rounding?" at `:37` settled as C libm `lrint`/`lrintf` = round-half-to-even) | one `for i in 0..len` loop per kernel; C's 4× unroll plus tail is pure optimization with identical iteration order and count (`len = (end-po)/os`) |
//! | 36 numeric kernels `audioconvert.c:54-89` + DSD byte copy `:90` | [`conv_u8_to_u8`] … [`conv_dsd_to_dsd`] (C names kept: `CONV_FUNC_NAME(ofmt, ifmt)` expands to `conv_<SRC>_to_<DST>`, `audioconvert.c:35`) |
//! | DSD→FLT `audioconvert.c:92-95` → `dsd2pcm.c` FIR (48 taps) | **not ported** — `AudioConvert::new(in=Dsd, out=Flt)` returns `Error::Unsupported`; DSD→DSD pass-through stays working |
//! | `fmt_pair_to_conv_functions[]` table `audioconvert.c:97-138`, lookup `[out + AV_SAMPLE_FMT_NB*in]` on packed formats at `:159` | [`conv_pair`] — a `match` over the legal pairs. **This table IS the legality constraint**: there is no `AV_SAMPLE_FMT_PAIR_CONVERSION` macro anywhere in this FFmpeg tree (grep over `FFmpeg/` returns nothing). The full 6×6 matrix over packed {u8,s16,s32,flt,dbl,s64} exists (so all 12×12 = 144 packed/planar combinations are legal), plus `dsd→flt` (blocked here, see above) and `dsd→dsd`; every other DSD combination is a `NULL` hole |
//! | `cpy1/2/4/8` `audioconvert.c:140-151`, selection at `:184-191`, dispatch at `:235-252` | flattened into `simd_bps: Option<usize>` (set iff `out_fmt == in_fmt && ch_map.is_none()`): a whole-plane copy inside [`swri_audio_convert`]. C's `off = len&!15` 16-sample chunking plus scalar remainder is output-identical to one full copy because the same-format kernels are per-sample byte copies. The alignment/misalignment logic `:218-231` is dropped — `in/out_simd_align_mask` are only ever set by the skipped arch initializers, hence 0 in the scalar build. Note: x86-64 asm builds actually NULL the cpy path anyway (`x86/audio_convert_init.c:43`, "FIXME add memcpy case"), so release C runs the same scalar kernels |
//! | `swri_audio_convert_alloc` `audioconvert.c:153-202` | [`AudioConvert::new`]. Mono normalization `:167-170` (channels == 1 → both formats planar) happens after the kernel pick and before the copy-path decision, exactly as in C. Silence `:175-182`: `[0x80; 8]` for u8 input, `[0x69; 8]` for DSD input; `swri_dsd2pcm_init()`/`dsd_state[]` skipped (unported, only reachable for dsd→dsd after the pair check). Arch init `:193-199` (`swri_audio_convert_init_x86/arm/aarch64`, guarded `#if ARCH_X86 && HAVE_X86ASM` etc.) skipped — SIMD is flattened to scalar. The `flags` (`AV_CPU_FLAG_*`) parameter is dropped: unused in the scalar body. `swri_audio_convert_free` `:204-207` needs no port (Rust `Drop`) |
//! | `swri_audio_convert` `audioconvert.c:209-265` | [`swri_audio_convert`] (C param `in` renamed `input` — `in` is a Rust keyword). `os` from OUT only `:213`; `av_assert0(channels == out->ch_count)` `:216` degraded to `debug_assert_eq!` + `Error::InvalidArgument` in release (repo convention for C aborts); scalar channel loop `:254-263` with `ch_map` reorder/mute `:255-257`, `is=0`+`silence` for muted channels, and the `if(!po) continue` skip `:259-260` |
//! | `av_clip_uint8_c` / `av_clip_int16_c` / `av_clipl_int32_c` `libavutil/common.h:210-214,243-247,254-258` | [`clip_u8`]/[`clip_i16`]/[`clip_i32`] — for in-range-typed inputs each C bit trick is exactly `clamp` |
//! | C libm `lrintf`/`lrint`/`llrintf`/`llrint` (default `FE_TONEAREST`) | [`lrint_f32`]/[`lrint_f64`] = `round_ties_even()` + Rust's saturating `as i64` |
//!
//! ## Stride / plane geometry (the load-bearing part)
//!
//! Packed `AudioData` channels are strided aliases `ch[i] = data + i*bps` into
//! ONE interleaved buffer (`swresample.c:448`, `:479-481`), so the same kernels
//! handle packed↔packed, packed↔planar and planar↔planar with no special
//! casing: the input stride is `is = (input.planar ? 1 : input.ch_count) *
//! input.bps`, the output stride `os = (out.planar ? 1 : out.ch_count) *
//! out.bps`, and each kernel walks `po[i*os .. i*os+out_bps]` reading
//! `pi[i*is .. i*is+in_bps]` (all integers little-endian — the crate is
//! LE-only by policy, like `util::pixfmt`).
//!
//! ## Divergences from C (each pinned by tests or documented at its guard)
//!
//! * Float→int kernels: the C clips take `int` (`av_clip_uint8`/
//!   `av_clip_int16`) so the `long` `lrintf` result is truncated (wrapping,
//!   mod 2^32) BEFORE clipping — kept exactly, including overdrive wrap
//!   (`f = 2^28` → `u8` output 128, `s16` output 0). `av_clipl_int32` and the
//!   `s64` targets take `int64_t` — no truncation. `NaN`/`|x| ≥ 2^64` are C
//!   UB (x86-64 yields the "integer indefinite" `i64::MIN`); the port
//!   saturates (`f32/f64 → i64` Rust semantics) — deterministic, documented.
//!   The `f → s64` kernel has no clip in C either, so `f = +1.0` (product
//!   exactly 2^63, out of `i64` range) gives `i64::MIN` on x86-64 C but
//!   `i64::MAX` here; U8/S16/S32 targets clip first and agree.
//! * C UB sites degraded: unchecked `ch_map` values (`:255-257`) →
//!   `debug_assert` + `Error::BufferTooSmall` when the plane address falls
//!   outside the backing buffer (never UB); unchecked plane capacities →
//!   `Error::BufferTooSmall` instead of an OOB write (note: an error may
//!   follow partial writes — C has no such failure mode); `av_assert1/2`
//!   `:237-239` dropped (compiled out at C's default ASSERT_LEVEL too).
//! * In-place conversion (`out` aliasing `input`) is legal C but unexpressible
//!   with `&mut`/`&`; no FFmpeg caller does it through `swri_audio_convert`
//!   (`swresample.c:617,668,734` all pass distinct `AudioData`), so the
//!   restriction is invisible.
//! * The exact C bound for a channel's last write is `(len-1)*stride + bps`
//!   bytes past the channel's base — the port guards with exactly that (for
//!   planar planes, where `stride == bps`, this coincides with `stride*len`).

use crate::util::{
    error::{Error, Result},
    samplefmt::SampleFormat,
};

use super::AudioData;

/// C `conv_func_type` (`audioconvert.h:36`) minus the `DSDContext *st`
/// parameter (only the unported DSD→FLT kernel used it), with C's `end`
/// pointer replaced by the sample count `len = (end - po) / os`.
///
/// Contract (enforced by the caller [`swri_audio_convert`], not the kernels):
/// `po.len() >= (len-1)*os + out_bps` and `pi.len() >= (len-1)*is + in_bps`.
type ConvFunc = fn(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize);

// --- source U8: audioconvert.c:54-59 ---------------------------------------

/// `audioconvert.c:54` — identity byte copy.
fn conv_u8_to_u8(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os] = pi[i * is];
    }
}

/// `audioconvert.c:55` — `(*(const uint8_t*)pi - 0x80U)<<8` stored to
/// `int16_t`. The `0x80U` makes the subtraction unsigned, but the `int16_t`
/// store keeps only the low 16 bits, which equals the signed form.
fn conv_u8_to_s16(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let s = ((pi[i * is] as i32 - 0x80) << 8) as i16;
        po[i * os..i * os + 2].copy_from_slice(&s.to_le_bytes());
    }
}

/// `audioconvert.c:56` — same unsigned/signed equivalence on 32 bits.
fn conv_u8_to_s32(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let v = ((pi[i * is] as i32 - 0x80) << 24) as i32;
        po[i * os..i * os + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// `audioconvert.c:57` — `(uint64_t)(u - 0x80U)<<56`: the zero-extension
/// keeps only the low 8 bits under `<<56`, giving exactly the signed shift.
/// `0x00 -> i64::MIN`.
fn conv_u8_to_s64(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let w = (pi[i * is] as i64 - 0x80) << 56;
        po[i * os..i * os + 8].copy_from_slice(&w.to_le_bytes());
    }
}

/// `audioconvert.c:58` — note `0x80` has NO `U` suffix here: the subtraction
/// is signed int in [-128,127], so `u8 = 0x00` maps to −1.0, not +255/128.
fn conv_u8_to_flt(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let f = (pi[i * is] as i32 - 0x80) as f32 * (1.0f32 / 128.0);
        po[i * os..i * os + 4].copy_from_slice(&f.to_le_bytes());
    }
}

/// `audioconvert.c:59` — signed 0x80 subtraction, `1.0/(1<<7)` in f64.
fn conv_u8_to_dbl(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let d = (pi[i * is] as i32 - 0x80) as f64 * (1.0 / 128.0);
        po[i * os..i * os + 8].copy_from_slice(&d.to_le_bytes());
    }
}

// --- source S16: audioconvert.c:60-65 --------------------------------------

/// `audioconvert.c:60` — arithmetic `>>8` lands in [-128,127], `+0x80` in
/// [0,255]: the `uint8_t` store is always exact, no clip.
fn conv_s16_to_u8(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let s = i16::from_le_bytes(pi[i * is..i * is + 2].try_into().unwrap());
        po[i * os] = ((s >> 8) + 0x80) as u8;
    }
}

/// `audioconvert.c:61` — bit copy.
fn conv_s16_to_s16(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os..i * os + 2].copy_from_slice(&pi[i * is..i * is + 2]);
    }
}

/// `audioconvert.c:62` — the `int16_t` promotes to `int`, `* (1<<16)` never
/// overflows 32 bits (−32768·65536 = −2^31 exactly).
fn conv_s16_to_s32(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let s = i16::from_le_bytes(pi[i * is..i * is + 2].try_into().unwrap());
        let v = (s as i32) * (1 << 16);
        po[i * os..i * os + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// `audioconvert.c:63` — sign-extends through `int` to the 64-bit shift.
/// `-32768 -> i64::MIN` exactly.
fn conv_s16_to_s64(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let s = i16::from_le_bytes(pi[i * is..i * is + 2].try_into().unwrap());
        let w = (s as i64) << 48;
        po[i * os..i * os + 8].copy_from_slice(&w.to_le_bytes());
    }
}

/// `audioconvert.c:64` — `*(1.0f/(1<<15))`, 2^-15 exact.
fn conv_s16_to_flt(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let s = i16::from_le_bytes(pi[i * is..i * is + 2].try_into().unwrap());
        let f = s as f32 * (1.0f32 / 32768.0);
        po[i * os..i * os + 4].copy_from_slice(&f.to_le_bytes());
    }
}

/// `audioconvert.c:65`.
fn conv_s16_to_dbl(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let s = i16::from_le_bytes(pi[i * is..i * is + 2].try_into().unwrap());
        let d = s as f64 * (1.0 / 32768.0);
        po[i * os..i * os + 8].copy_from_slice(&d.to_le_bytes());
    }
}

// --- source S32: audioconvert.c:66-71 --------------------------------------

/// `audioconvert.c:66` — arithmetic `>>24` + 0x80, always in [0,255].
fn conv_s32_to_u8(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let v = i32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        po[i * os] = ((v >> 24) + 0x80) as u8;
    }
}

/// `audioconvert.c:67` — `>>16` always lands in int16 range.
fn conv_s32_to_s16(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let v = i32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        po[i * os..i * os + 2].copy_from_slice(&((v >> 16) as i16).to_le_bytes());
    }
}

/// `audioconvert.c:68` — bit copy.
fn conv_s32_to_s32(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os..i * os + 4].copy_from_slice(&pi[i * is..i * is + 4]);
    }
}

/// `audioconvert.c:69`.
fn conv_s32_to_s64(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let v = i32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        po[i * os..i * os + 8].copy_from_slice(&((v as i64) << 32).to_le_bytes());
    }
}

/// `audioconvert.c:70` — the int→float conversion rounds to nearest-even
/// exactly like C's implicit conversion, then the 2^-31 scale is exact.
fn conv_s32_to_flt(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let v = i32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        let f = v as f32 * (1.0f32 / 2147483648.0);
        po[i * os..i * os + 4].copy_from_slice(&f.to_le_bytes());
    }
}

/// `audioconvert.c:71`.
fn conv_s32_to_dbl(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let v = i32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        let d = v as f64 * (1.0 / 2147483648.0);
        po[i * os..i * os + 8].copy_from_slice(&d.to_le_bytes());
    }
}

// --- source S64: audioconvert.c:72-77 --------------------------------------

/// `audioconvert.c:72` — arithmetic `>>56` + 0x80: `i64::MIN -> 0`, `-1 -> 127`.
fn conv_s64_to_u8(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let w = i64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        po[i * os] = ((w >> 56) + 0x80) as u8;
    }
}

/// `audioconvert.c:73`.
fn conv_s64_to_s16(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let w = i64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        po[i * os..i * os + 2].copy_from_slice(&((w >> 48) as i16).to_le_bytes());
    }
}

/// `audioconvert.c:74`.
fn conv_s64_to_s32(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let w = i64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        po[i * os..i * os + 4].copy_from_slice(&((w >> 32) as i32).to_le_bytes());
    }
}

/// `audioconvert.c:75` — bit copy.
fn conv_s64_to_s64(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os..i * os + 8].copy_from_slice(&pi[i * is..i * is + 8]);
    }
}

/// `audioconvert.c:76` — 2^63 and 2^-63 both exact in f32.
fn conv_s64_to_flt(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let w = i64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        let f = w as f32 * (1.0f32 / 9223372036854775808.0);
        po[i * os..i * os + 4].copy_from_slice(&f.to_le_bytes());
    }
}

/// `audioconvert.c:77`.
fn conv_s64_to_dbl(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let w = i64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        let d = w as f64 * (1.0 / 9223372036854775808.0);
        po[i * os..i * os + 8].copy_from_slice(&d.to_le_bytes());
    }
}

// --- source FLT: audioconvert.c:78-83 --------------------------------------

/// `audioconvert.c:78` — `av_clip_uint8(lrintf(f*(1<<7)) + 0x80)`.
///
/// `lrintf` returns `long`; the `+0x80` is `long` arithmetic; the clip takes
/// an `int`, so the SUM is truncated mod 2^32 first. `(r as i32)` in Rust is
/// the same wrapping truncation, and `wrapping_add(0x80)` keeps the modular
/// equality `((r mod 2^32) + 128) mod 2^32 == (r + 128) mod 2^32`.
fn conv_flt_to_u8(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let f = f32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        let a = (lrint_f32(f * 128.0f32) as i32).wrapping_add(0x80);
        po[i * os] = clip_u8(a);
    }
}

/// `audioconvert.c:79` — `av_clip_int16(lrintf(f*(1<<15)))`, int-taking clip
/// ⇒ `i32` wrapping truncation before the clamp.
fn conv_flt_to_s16(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let f = f32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        let a = lrint_f32(f * 32768.0f32) as i32;
        po[i * os..i * os + 2].copy_from_slice(&clip_i16(a).to_le_bytes());
    }
}

/// `audioconvert.c:80` — `av_clipl_int32(llrintf(f*(1U<<31)))`: the clip
/// takes `int64_t`, so NO `i32` truncation happens.
fn conv_flt_to_s32(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let f = f32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        let a = lrint_f32(f * 2147483648.0f32);
        po[i * os..i * os + 4].copy_from_slice(&clip_i32(a).to_le_bytes());
    }
}

/// `audioconvert.c:81` — `llrintf(f*(UINT64_C(1)<<63))`, no clip. C UB (x86
/// `i64::MIN`) for `f = +1.0`; the port saturates to `i64::MAX` (documented
/// divergence, module map).
fn conv_flt_to_s64(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let f = f32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        let w = lrint_f32(f * 9223372036854775808.0f32);
        po[i * os..i * os + 8].copy_from_slice(&w.to_le_bytes());
    }
}

/// `audioconvert.c:82` — bit copy.
fn conv_flt_to_flt(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os..i * os + 4].copy_from_slice(&pi[i * is..i * is + 4]);
    }
}

/// `audioconvert.c:83` — exact widening.
fn conv_flt_to_dbl(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let f = f32::from_le_bytes(pi[i * is..i * is + 4].try_into().unwrap());
        po[i * os..i * os + 8].copy_from_slice(&(f as f64).to_le_bytes());
    }
}

// --- source DBL: audioconvert.c:84-89 --------------------------------------

/// `audioconvert.c:84` — as [`conv_flt_to_u8`] with f64 arithmetic.
fn conv_dbl_to_u8(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let d = f64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        let a = (lrint_f64(d * 128.0) as i32).wrapping_add(0x80);
        po[i * os] = clip_u8(a);
    }
}

/// `audioconvert.c:85`.
fn conv_dbl_to_s16(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let d = f64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        let a = lrint_f64(d * 32768.0) as i32;
        po[i * os..i * os + 2].copy_from_slice(&clip_i16(a).to_le_bytes());
    }
}

/// `audioconvert.c:86` — `int64_t`-taking clip, no `i32` truncation.
fn conv_dbl_to_s32(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let d = f64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        let a = lrint_f64(d * 2147483648.0);
        po[i * os..i * os + 4].copy_from_slice(&clip_i32(a).to_le_bytes());
    }
}

/// `audioconvert.c:87` — no clip; same saturation divergence as
/// [`conv_flt_to_s64`] at/above full scale.
fn conv_dbl_to_s64(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let d = f64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        let w = lrint_f64(d * 9223372036854775808.0);
        po[i * os..i * os + 8].copy_from_slice(&w.to_le_bytes());
    }
}

/// `audioconvert.c:88` — round-to-nearest narrowing (C's implicit
/// double→float conversion).
fn conv_dbl_to_flt(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        let d = f64::from_le_bytes(pi[i * is..i * is + 8].try_into().unwrap());
        po[i * os..i * os + 4].copy_from_slice(&(d as f32).to_le_bytes());
    }
}

/// `audioconvert.c:89` — bit copy.
fn conv_dbl_to_dbl(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os..i * os + 8].copy_from_slice(&pi[i * is..i * is + 8]);
    }
}

// --- source DSD: audioconvert.c:90-95 --------------------------------------

/// `audioconvert.c:90` — byte copy (`CONV_FUNC(DSD, DSD)`).
///
/// `audioconvert.c:92-95` — DSD→FLT needs dsd2pcm.c (unported): rejected at
/// alloc with a dedicated `Error::Unsupported`, see [`AudioConvert::new`].
fn conv_dsd_to_dsd(po: &mut [u8], pi: &[u8], is: usize, os: usize, len: usize) {
    for i in 0..len {
        po[i * os] = pi[i * is];
    }
}

// --- clips + lrint: libavutil/common.h:210-258, C libm ----------------------

/// `av_clip_uint8_c` (`libavutil/common.h:210-214`) — for an `i32` input the
/// bit trick is exactly `clamp(0, 255)`.
fn clip_u8(a: i32) -> u8 {
    a.clamp(0, 255) as u8
}

/// `av_clip_int16_c` (`libavutil/common.h:243-247`) — exactly
/// `clamp(-32768, 32767)` for an `i32` input.
fn clip_i16(a: i32) -> i16 {
    a.clamp(-32768, 32767) as i16
}

/// `av_clipl_int32_c` (`libavutil/common.h:254-258`) — takes `int64_t`, so
/// no 32-bit truncation happens before the clamp.
fn clip_i32(a: i64) -> i32 {
    a.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// C libm `lrintf`/`llrintf` under the default `FE_TONEAREST`: round to
/// nearest, ties to even (what the C FIXME "rounding?" at `audioconvert.c:37`
/// settled on), then the Rust `as i64` saturating cast.
fn lrint_f32(x: f32) -> i64 {
    x.round_ties_even() as i64
}

/// C libm `lrint`/`llrint` (double) — same ties-to-even + saturating cast.
fn lrint_f64(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The `fmt_pair_to_conv_functions` table (`audioconvert.c:97-138`) looked up
/// as `[av_get_packed_sample_fmt(out) + AV_SAMPLE_FMT_NB *
/// av_get_packed_sample_fmt(in)]` (`audioconvert.c:159`) — `None` is C's
/// `NULL` hole. This table IS the format-pair legality constraint; there is
/// no `AV_SAMPLE_FMT_PAIR_CONVERSION` macro in this tree.
///
/// `(Dsd, Flt)` is a C-legal pair (`audioconvert.c:136`) but its kernel is
/// the unported dsd2pcm FIR, so it returns `None` here and
/// [`AudioConvert::new`] gives it the dedicated unsupported message.
fn conv_pair(in_packed: SampleFormat, out_packed: SampleFormat) -> Option<ConvFunc> {
    use SampleFormat as F;
    let f: ConvFunc = match (in_packed, out_packed) {
        // in U8 — audioconvert.c:100-105
        (F::U8, F::U8) => conv_u8_to_u8,
        (F::U8, F::S16) => conv_u8_to_s16,
        (F::U8, F::S32) => conv_u8_to_s32,
        (F::U8, F::Flt) => conv_u8_to_flt,
        (F::U8, F::Dbl) => conv_u8_to_dbl,
        (F::U8, F::S64) => conv_u8_to_s64,
        // in S16 — audioconvert.c:106-111
        (F::S16, F::U8) => conv_s16_to_u8,
        (F::S16, F::S16) => conv_s16_to_s16,
        (F::S16, F::S32) => conv_s16_to_s32,
        (F::S16, F::Flt) => conv_s16_to_flt,
        (F::S16, F::Dbl) => conv_s16_to_dbl,
        (F::S16, F::S64) => conv_s16_to_s64,
        // in S32 — audioconvert.c:112-117
        (F::S32, F::U8) => conv_s32_to_u8,
        (F::S32, F::S16) => conv_s32_to_s16,
        (F::S32, F::S32) => conv_s32_to_s32,
        (F::S32, F::Flt) => conv_s32_to_flt,
        (F::S32, F::Dbl) => conv_s32_to_dbl,
        (F::S32, F::S64) => conv_s32_to_s64,
        // in FLT — audioconvert.c:118-123
        (F::Flt, F::U8) => conv_flt_to_u8,
        (F::Flt, F::S16) => conv_flt_to_s16,
        (F::Flt, F::S32) => conv_flt_to_s32,
        (F::Flt, F::Flt) => conv_flt_to_flt,
        (F::Flt, F::Dbl) => conv_flt_to_dbl,
        (F::Flt, F::S64) => conv_flt_to_s64,
        // in DBL — audioconvert.c:124-129
        (F::Dbl, F::U8) => conv_dbl_to_u8,
        (F::Dbl, F::S16) => conv_dbl_to_s16,
        (F::Dbl, F::S32) => conv_dbl_to_s32,
        (F::Dbl, F::Flt) => conv_dbl_to_flt,
        (F::Dbl, F::Dbl) => conv_dbl_to_dbl,
        (F::Dbl, F::S64) => conv_dbl_to_s64,
        // in S64 — audioconvert.c:130-135
        (F::S64, F::U8) => conv_s64_to_u8,
        (F::S64, F::S16) => conv_s64_to_s16,
        (F::S64, F::S32) => conv_s64_to_s32,
        (F::S64, F::Flt) => conv_s64_to_flt,
        (F::S64, F::Dbl) => conv_s64_to_dbl,
        (F::S64, F::S64) => conv_s64_to_s64,
        // in DSD — audioconvert.c:136-137 (DSD→FLT unported, see above)
        (F::Dsd, F::Dsd) => conv_dsd_to_dsd,
        _ => return None,
    };
    Some(f)
}

/// `struct AudioConvert` (`audioconvert.h:39-48`).
///
/// Field mapping vs C: `channels`, `conv_f`, `silence[8]` keep their names;
/// `simd_bps` replaces `simd_f` + the `cpy1/2/4/8` selection (bytes per
/// sample 1/2/4/8, set at alloc when `out_fmt == in_fmt && ch_map.is_none()`,
/// consumed as a whole-plane copy); `ch_map` is an owned clone of C's
/// borrowed `const int *ch_map` (entries `>= 0` = source channel index,
/// `-1` = muted channel, `audioconvert.h:56-58`).
///
/// Dropped with documented guards: `in/out_simd_align_mask` (only the skipped
/// arch initializers set them — always 0 in scalar builds, and the misalign
/// accumulation at `audioconvert.c:218-231` is dead code) and
/// `dsd_state[SWR_CH_MAX]` (only the unported DSD→FLT kernel reads it).
pub struct AudioConvert {
    channels: usize,
    conv_f: ConvFunc,
    simd_bps: Option<usize>,
    ch_map: Option<Vec<i32>>,
    silence: [u8; 8],
}

impl std::fmt::Debug for AudioConvert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // fn-pointer Debug is address noise — list the C-visible state instead.
        f.debug_struct("AudioConvert")
            .field("channels", &self.channels)
            .field("ch_map", &self.ch_map)
            .field("simd_bps", &self.simd_bps)
            .field("silence", &self.silence)
            .finish_non_exhaustive()
    }
}

impl AudioConvert {
    /// `swri_audio_convert_alloc` (`audioconvert.c:153-202`; doc at
    /// `audioconvert.h:50-63`). C returns `NULL` on a pair-table hole; the
    /// port returns `Err` — the generic message mirrors the caller's log text
    /// at `swresample.c:373-375` ("Cannot convert %s sample format to %s
    /// sample format", source first, destination second). The C `flags`
    /// (`AV_CPU_FLAG_*`) parameter is dropped: unused in the scalar body.
    pub fn new(
        out_fmt: SampleFormat,
        in_fmt: SampleFormat,
        channels: usize,
        ch_map: Option<&[i32]>,
    ) -> Result<Self> {
        // C-legal pair at audioconvert.c:136 whose kernel delegates to
        // dsd2pcm.c (unported) — dedicated message rather than the generic
        // hole text, so the missing piece is identifiable.
        if in_fmt.packed() == SampleFormat::Dsd && out_fmt.packed() == SampleFormat::Flt {
            return Err(Error::Unsupported(
                "DSD to float conversion requires dsd2pcm (libswresample/dsd2pcm.c), which is not ported"
                    .into(),
            ));
        }

        // audioconvert.c:159-162 — table lookup on PACKED forms (the kernel
        // does not care about planarity; strides carry that).
        let Some(conv_f) = conv_pair(in_fmt.packed(), out_fmt.packed()) else {
            return Err(Error::Unsupported(format!(
                "cannot convert sample format {} to {}",
                in_fmt.name(),
                out_fmt.name()
            )));
        };

        // audioconvert.c:167-170 — LOCAL normalization: mono always converts
        // planar->planar. Affects only the silence + copy-path decisions
        // below (the kernel was already chosen from the packed forms).
        let mut in_fmt = in_fmt;
        let mut out_fmt = out_fmt;
        if channels == 1 {
            in_fmt = in_fmt.planar();
            out_fmt = out_fmt.planar();
        }

        // audioconvert.c:172-174 — the port owns a clone of the borrowed map.
        let ch_map = ch_map.map(|m| m.to_vec());

        // audioconvert.c:175-182 — silence input sample.
        let mut silence = [0u8; 8];
        if in_fmt == SampleFormat::U8 || in_fmt == SampleFormat::U8p {
            silence = [0x80; 8];
        }
        if in_fmt == SampleFormat::Dsd {
            // swri_dsd2pcm_init() + dsd_state[] init skipped (unported; only
            // reachable for dsd->dsd after the pair check above).
            silence = [0x69; 8];
        }

        // audioconvert.c:184-191 — the cpy1/2/4/8 selection, flattened to
        // "bytes per sample of an identical-format, unmapped converter".
        // AFTER the mono normalization, exactly as in C (an unequal pair can
        // become equal here, e.g. out=S16P in=S16 mono).
        let simd_bps = if out_fmt == in_fmt && ch_map.is_none() {
            Some(in_fmt.bytes_per_sample())
        } else {
            None
        };

        // audioconvert.c:193-199 — swri_audio_convert_init_x86/arm/aarch64
        // skipped (guards `#if ARCH_X86 && HAVE_X86ASM` / ARCH_ARM /
        // ARCH_AARCH64): SIMD macros flattened to scalar, output-identical.

        Ok(AudioConvert {
            channels,
            conv_f,
            simd_bps,
            ch_map,
            silence,
        })
    }
}

/// `swri_audio_convert` (`audioconvert.c:209-265`; doc at
/// `audioconvert.h:72-77`): convert `len` samples per channel between the
/// formats the context was built for.
///
/// C's `int` return (0 ok) becomes `Result<()>`; C's pointer trust becomes
/// bounds guards (`Error::BufferTooSmall`) and the `av_assert0` at `:216`
/// becomes a debug assert plus `Error::InvalidArgument` in release.
pub fn swri_audio_convert(
    ctx: &AudioConvert,
    out: &mut AudioData,
    input: &AudioData,
    len: usize,
) -> Result<()> {
    // audioconvert.c:213 — output stride, computed from OUT only.
    let os = (if out.planar { 1 } else { out.ch_count }) * out.bps;

    // audioconvert.c:216 — av_assert0(ctx->channels == out->ch_count).
    debug_assert_eq!(ctx.channels, out.ch_count);
    if ctx.channels != out.ch_count {
        return Err(Error::InvalidArgument(
            "swri_audioconvert: out channel count does not match the converter context".into(),
        ));
    }

    // len = 0: C's fast path returns immediately (off = 0&~15 = 0 == len,
    // audioconvert.c:250-251) and the scalar kernels write nothing — a
    // complete no-op that never touches the planes.
    if len == 0 {
        return Ok(());
    }

    // audioconvert.c:218-231 — misalignment accumulation dropped: the align
    // masks are only set by the skipped arch initializers, so `!misaligned`
    // is always true in the scalar build.
    //
    // audioconvert.c:235-252 — fast path (simd_bps is Some only when
    // ch_map.is_none(), and out_fmt == in_fmt post-normalization, hence
    // out.planar == input.planar — C's mismatch else-arm at :246-248 is
    // defensive dead code, asserted here). C copies `off = len&!15` samples
    // then lets the scalar identity kernels finish the remainder; since the
    // identity kernels are per-sample byte copies, one full copy is
    // output-identical — flattened.
    if let Some(bps) = ctx.simd_bps {
        debug_assert!(ctx.ch_map.is_none());
        // audioconvert.c:184-191: simd_bps implies out_fmt == in_fmt post-
        // normalization. For mono the packed/planar AudioData shapes are
        // memory-identical (one plane either way), so only multi-channel
        // pairs must agree.
        debug_assert!(
            out.planar == input.planar || out.ch_count == 1,
            "simd_bps implies out_fmt == in_fmt (audioconvert.c:184-191)"
        );
        let planes = if out.planar { out.ch_count } else { 1 };
        let out_per_plane = bps * len * (if out.planar { 1 } else { out.ch_count });
        let in_per_plane = bps * len * (if input.planar { 1 } else { input.ch_count });
        for ch in 0..planes {
            let src = input.plane(ch).ok_or(Error::BufferTooSmall)?;
            if src.len() < in_per_plane {
                return Err(Error::BufferTooSmall);
            }
            let dst = out.plane_bytes_mut(ch).ok_or(Error::BufferTooSmall)?;
            if dst.len() < out_per_plane {
                return Err(Error::BufferTooSmall);
            }
            dst[..out_per_plane].copy_from_slice(&src[..out_per_plane]);
        }
        return Ok(());
    }

    // audioconvert.c:254-263 — scalar channel loop.
    for ch in 0..ctx.channels {
        // C trusts ch_map length and values (audioconvert.h:56-58); debug
        // builds assert, release builds surface an out-of-buffer plane as
        // BufferTooSmall instead of C's uninitialized-pointer read.
        let ich = match ctx.ch_map.as_ref() {
            Some(m) => m[ch],
            None => ch as i32,
        };
        // audioconvert.c:256 — input stride; muted channels read is=0.
        let is = if ich < 0 {
            0
        } else {
            (if input.planar { 1 } else { input.ch_count }) * input.bps
        };
        // audioconvert.c:257 — muted channels read ctx->silence.
        let pi: &[u8] = if ich < 0 {
            &ctx.silence[..]
        } else {
            debug_assert!(
                (ich as usize) < input.ch_count,
                "ch_map[{ch}] = {ich} out of range (input.ch_count = {})",
                input.ch_count
            );
            input.plane(ich as usize).ok_or(Error::BufferTooSmall)?
        };
        // Read bound the kernels rely on: last sample spans
        // (len-1)*is .. +in_bps from the channel base. A muted channel reads
        // the same silence bytes (is = 0, 8 bytes >= any bps).
        if pi.len() < (len - 1) * is + input.bps {
            return Err(Error::BufferTooSmall);
        }
        // Write bound: last sample spans (len-1)*os .. +out_bps.
        let out_need = (len - 1) * os + out.bps;
        // audioconvert.c:258-260 — absent output channels are skipped
        // (audioconvert.h:73: "set to NULL to ignore processing").
        let po = match out.plane_bytes_mut(ch) {
            Some(p) => p,
            None => continue,
        };
        if po.len() < out_need {
            return Err(Error::BufferTooSmall);
        }
        // audioconvert.c:262 — the kernel call (off = 0 after the flatten).
        (ctx.conv_f)(po, pi, is, os, len);
    }
    // audioconvert.c:264 — return 0.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- byte/value helpers ---------------------------------------------------

    fn s16_bytes(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }
    fn s32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }
    fn s64_bytes(v: &[i64]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }
    fn as_i16(b: &[u8]) -> Vec<i16> {
        b.chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn as_i32(b: &[u8]) -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn as_i64(b: &[u8]) -> Vec<i64> {
        b.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn as_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn as_f64(b: &[u8]) -> Vec<f64> {
        b.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    /// Run one kernel over `len` samples, mono strides, sized exactly.
    fn run(f: ConvFunc, pi: &[u8], is: usize, os: usize, out_bps: usize, len: usize) -> Vec<u8> {
        let mut po = vec![0xEEu8; (len - 1) * os + out_bps];
        f(&mut po, pi, is, os, len);
        po
    }
    /// AudioData from raw bytes with explicit geometry.
    fn ad(data: Vec<u8>, fmt: SampleFormat, ch_count: usize, count: usize) -> AudioData {
        AudioData {
            bps: fmt.bytes_per_sample(),
            data: data.into(),
            ch_count,
            count,
            planar: fmt.is_planar(),
            fmt,
        }
    }

    // -- pair table ------------------------------------------------------------

    #[test]
    fn pair_table_completeness() {
        // audioconvert.c:97-138,159 — the designated-initializer table IS
        // the constraint: full 6x6 over packed {u8,s16,s32,flt,dbl,s64}
        // (all 144 packed/planar combos legal) + dsd->{flt,dsd}; FLT blocked
        // here by the missing dsd2pcm.
        let six = [
            SampleFormat::U8,
            SampleFormat::S16,
            SampleFormat::S32,
            SampleFormat::Flt,
            SampleFormat::Dbl,
            SampleFormat::S64,
        ];
        let (mut ok, mut err) = (0, 0);
        for &i in SampleFormat::ALL {
            for &o in SampleFormat::ALL {
                let res = AudioConvert::new(o, i, 2, None);
                let legal = six.contains(&i.packed()) && six.contains(&o.packed())
                    || (i == SampleFormat::Dsd && o == SampleFormat::Dsd);
                if legal {
                    assert!(res.is_ok(), "expected ok: {} -> {}", i.name(), o.name());
                    ok += 1;
                } else {
                    let msg = match res {
                        Err(Error::Unsupported(m)) => m,
                        other => panic!(
                            "expected Err for {} -> {}, got {other:?}",
                            i.name(),
                            o.name()
                        ),
                    };
                    if i == SampleFormat::Dsd && matches!(o, SampleFormat::Flt | SampleFormat::Fltp)
                    {
                        assert_eq!(
                            msg,
                            "DSD to float conversion requires dsd2pcm (libswresample/dsd2pcm.c), which is not ported"
                        );
                    } else {
                        assert_eq!(
                            msg,
                            format!("cannot convert sample format {} to {}", i.name(), o.name()),
                            "source name first, destination second (swresample.c:373-375)"
                        );
                    }
                    err += 1;
                }
            }
        }
        assert_eq!(ok, 145); // 144 + dsd->dsd
        assert_eq!(err, 24); // 169 total - 145
    }

    // -- source u8 kernels (audioconvert.c:54-59) -------------------------------

    #[test]
    fn u8_source_kernels() {
        let pi = [0x00u8, 0x80, 0xFF];
        // :54 identity
        assert_eq!(run(conv_u8_to_u8, &pi, 1, 1, 1, 3), pi.to_vec());
        // :55 — signed-form equivalence of the 0x80U subtraction
        assert_eq!(
            as_i16(&run(conv_u8_to_s16, &pi, 1, 2, 2, 3)),
            vec![-32768, 0, 32512]
        );
        // :56 — 0xFF - 0x80 = 127, 127<<24 = 0x7F000000 (C truth; the
        // 0x7FFF0000 value would be the S16-source row)
        assert_eq!(
            as_i32(&run(conv_u8_to_s32, &pi, 1, 4, 4, 3)),
            vec![-2147483648, 0, 2130706432]
        );
        // :57 — 0x00 -> i64::MIN, 0xFF -> 127<<56
        assert_eq!(
            as_i64(&run(conv_u8_to_s64, &pi, 1, 8, 8, 3)),
            vec![i64::MIN, 0, 127i64 << 56]
        );
        // :58 — SIGNED 0x80 subtraction: 0x00 -> -1.0 (not +255/128)
        assert_eq!(
            as_f32(&run(conv_u8_to_flt, &pi, 1, 4, 4, 3)),
            vec![-1.0, 0.0, 127.0 / 128.0]
        );
        // :59
        assert_eq!(
            as_f64(&run(conv_u8_to_dbl, &pi, 1, 8, 8, 3)),
            vec![-1.0, 0.0, 127.0 / 128.0]
        );
    }

    // -- source s16 kernels (audioconvert.c:60-65) ------------------------------

    #[test]
    fn s16_source_kernels() {
        let src = [-32768i16, 0, 32767];
        let pi = s16_bytes(&src);
        // :60 — arithmetic >>8 + 0x80, no clip needed
        assert_eq!(run(conv_s16_to_u8, &pi, 2, 1, 1, 3), vec![0, 128, 255]);
        // :61 identity bits
        assert_eq!(run(conv_s16_to_s16, &pi, 2, 2, 2, 3), pi.clone());
        // :62 — -32768*65536 = -2^31 exactly, 32767*65536 = 2147418112
        assert_eq!(
            as_i32(&run(conv_s16_to_s32, &pi, 2, 4, 4, 3)),
            vec![-2147483648, 0, 2147418112]
        );
        // :63 — -32768 -> i64::MIN, 1 -> 1<<48, 32767 -> 32767<<48
        let pi1 = s16_bytes(&[-32768i16, 1, 32767]);
        assert_eq!(
            as_i64(&run(conv_s16_to_s64, &pi1, 2, 8, 8, 3)),
            vec![i64::MIN, 1i64 << 48, 32767i64 << 48]
        );
        // :64
        assert_eq!(
            as_f32(&run(conv_s16_to_flt, &pi, 2, 4, 4, 3)),
            vec![-1.0, 0.0, 32767.0 / 32768.0]
        );
        // :65
        assert_eq!(
            as_f64(&run(conv_s16_to_dbl, &pi, 2, 8, 8, 3)),
            vec![-1.0, 0.0, 32767.0 / 32768.0]
        );
    }

    // -- arithmetic shifts, s32/s64 sources (audioconvert.c:66-74) ---------------

    #[test]
    fn shift_truncation_s32_s64() {
        // :67 — 0x80000000 >> 16 = 0xFFFF8000 (arithmetic), stored to i16
        let pi = s32_bytes(&[i32::MIN, -1]);
        assert_eq!(
            as_i16(&run(conv_s32_to_s16, &pi, 4, 2, 2, 2)),
            vec![-32768, -1]
        );
        // :66 — i32::MIN >> 24 = -128, + 0x80 = 0; -1 >> 24 = -1, + 0x80 = 127
        assert_eq!(run(conv_s32_to_u8, &pi, 4, 1, 1, 2), vec![0, 127]);
        // :74/:73/:72 — -1 shifts to -1 everywhere; i64::MIN >> 56 = -128
        let pi = s64_bytes(&[-1i64, i64::MIN]);
        // i64::MIN >> 32 = 0xFFFFFFFF80000000 -> low 32 = -2147483648;
        // >> 48 -> low 16 = -32768 (C's arithmetic shifts, :72-73).
        assert_eq!(
            as_i32(&run(conv_s64_to_s32, &pi, 8, 4, 4, 2)),
            vec![-1, -2147483648]
        );
        assert_eq!(
            as_i16(&run(conv_s64_to_s16, &pi, 8, 2, 2, 2)),
            vec![-1, -32768]
        );
        assert_eq!(run(conv_s64_to_u8, &pi, 8, 1, 1, 2), vec![127, 0]);
    }

    // -- float clipping (audioconvert.c:78-87) ----------------------------------

    #[test]
    fn float_clipping() {
        let src = [0.0f32, 0.5, 1.0, -1.0, 2.0];
        let pi = src.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>();
        // :79
        assert_eq!(
            as_i16(&run(conv_flt_to_s16, &pi, 4, 2, 2, 5)),
            vec![0, 16384, 32767, -32768, 32767]
        );
        // :78
        assert_eq!(
            run(conv_flt_to_u8, &pi, 4, 1, 1, 5),
            vec![128, 192, 255, 0, 255]
        );
        // :80 — -1.0 * 2^31 exact, clip to i32::MIN
        assert_eq!(
            as_i32(&run(conv_flt_to_s32, &pi, 4, 4, 4, 5)),
            vec![0, 16384 * 65536, 2147483647, -2147483648, 2147483647]
        );
        // :81 — no clip in C; -1.0 -> i64::MIN exact. +1.0 (2^63) is C UB
        // (x86 i64::MIN) vs port saturation i64::MAX — documented divergence.
        assert_eq!(
            as_i64(&run(conv_flt_to_s64, &pi, 4, 8, 8, 5)),
            vec![0, 1i64 << 62, i64::MAX, i64::MIN, i64::MAX]
        );
        // :84/:85/:86 — dbl row, same shape. 1e10 * 2^31 > 2^64: C UB, the
        // port saturates then clips to i32::MAX (documented divergence).
        let src = [0.0f64, -1.0, 1e10];
        let pi = src.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(
            as_i32(&run(conv_dbl_to_s32, &pi, 8, 4, 4, 3)),
            vec![0, -2147483648, 2147483647]
        );
    }

    #[test]
    fn float_overdrive_i32_truncation_before_clip() {
        // The int-taking clips (audioconvert.c:78-79) truncate the long
        // lrintf result mod 2^32 BEFORE clipping — load-bearing for
        // overdrive parity. f = 2^28: f*(1<<7) = 2^35, low 32 bits = 0.
        let pi = 2.0f32.powi(28).to_le_bytes();
        assert_eq!(run(conv_flt_to_u8, &pi, 4, 1, 1, 1), vec![128]); // 0 + 0x80
        assert_eq!(as_i16(&run(conv_flt_to_s16, &pi, 4, 2, 2, 1)), vec![0]);
        // negative side wraps identically: -2^35 + 0x80 -> 0x80
        let pi = (-2.0f32.powi(28)).to_le_bytes();
        assert_eq!(run(conv_flt_to_u8, &pi, 4, 1, 1, 1), vec![128]);
    }

    #[test]
    fn ties_to_even_rounding() {
        // audioconvert.c:85-87 — lrint ties-to-even (FE_TONEAREST).
        let pi = (2.5f64 / 2147483648.0).to_le_bytes();
        assert_eq!(as_i32(&run(conv_dbl_to_s32, &pi, 8, 4, 4, 1)), vec![2]);
        let pi = (3.5f64 / 2147483648.0).to_le_bytes();
        assert_eq!(as_i32(&run(conv_dbl_to_s32, &pi, 8, 4, 4, 1)), vec![4]);
        // :79 — f32 path: 0.5 -> 0, 1.5 -> 2 (products exact).
        let pi = [0.5f32 / 32768.0, 1.5 / 32768.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(as_i16(&run(conv_flt_to_s16, &pi, 4, 2, 2, 2)), vec![0, 2]);
    }

    #[test]
    fn int_to_float_scales() {
        // audioconvert.c:58-59,64-65,70-71,76-77 — bit-exact scales.
        let pi = s16_bytes(&[-32768i16]);
        assert_eq!(
            as_f32(&run(conv_s16_to_flt, &pi, 2, 4, 4, 1)),
            vec![-1.0f32]
        );
        // s32 max: int->f32 rounds to nearest-even first (C implicit
        // conversion), then the exact 2^-31 scale.
        let pi = s32_bytes(&[2147483647i32]);
        assert_eq!(
            as_f32(&run(conv_s32_to_flt, &pi, 4, 4, 4, 1)),
            vec![2147483647i32 as f32 * (1.0f32 / 2147483648.0)]
        );
        // s64 1<<62 * 2^-63 = 0.5 exactly.
        let pi = s64_bytes(&[1i64 << 62]);
        assert_eq!(as_f32(&run(conv_s64_to_flt, &pi, 8, 4, 4, 1)), vec![0.5f32]);
        // u8 0x00 -> -1.0 (the signed 0x80 subtraction).
        assert_eq!(
            as_f32(&run(conv_u8_to_flt, &[0x00], 1, 4, 4, 1)),
            vec![-1.0f32]
        );
    }

    // -- identity + width kernels -----------------------------------------------

    #[test]
    fn identity_kernels_bitexact() {
        // audioconvert.c:54,61,68,75,82,89,90 — byte-exact round-trips.
        let pat: Vec<u8> = (0..40u8)
            .map(|i| i.wrapping_mul(7).wrapping_add(3))
            .collect();
        for (f, bps) in [
            (conv_u8_to_u8 as ConvFunc, 1usize),
            (conv_s16_to_s16 as ConvFunc, 2),
            (conv_s32_to_s32 as ConvFunc, 4),
            (conv_s64_to_s64 as ConvFunc, 8),
            (conv_flt_to_flt as ConvFunc, 4),
            (conv_dbl_to_dbl as ConvFunc, 8),
            (conv_dsd_to_dsd as ConvFunc, 1),
        ] {
            let len = pat.len() / bps;
            assert_eq!(run(f, &pat, bps, bps, bps, len), pat[..len * bps].to_vec());
        }
        // :83 — exact widening.
        let pi = 0.5f32.to_le_bytes();
        assert_eq!(as_f64(&run(conv_flt_to_dbl, &pi, 4, 8, 8, 1)), vec![0.5f64]);
        // :88 — round-to-nearest narrowing: 1.0 + 2^-25 -> 1.0f32.
        let pi = (1.0f64 + 2.0f64.powi(-25)).to_le_bytes();
        assert_eq!(as_f32(&run(conv_dbl_to_flt, &pi, 8, 4, 4, 1)), vec![1.0f32]);
        assert_eq!(
            as_f32(&run(conv_dbl_to_flt, &0.5f64.to_le_bytes(), 8, 4, 4, 1)),
            vec![0.5f32]
        );
    }

    // -- AudioData addressing (swresample.c:448,479-481) ------------------------

    #[test]
    fn audio_data_plane_addressing() {
        // Packed: channel i aliases data[i*bps..] to the END of the buffer.
        let d = ad(vec![1, 2, 3, 4, 5, 6, 7, 8], SampleFormat::S16, 2, 2);
        assert_eq!(d.plane(0), Some(&[1u8, 2, 3, 4, 5, 6, 7, 8][..]));
        assert_eq!(d.plane(1), Some(&[3u8, 4, 5, 6, 7, 8][..]));
        // Planar: channel i is the bounded slice [i*count*bps .. +count*bps].
        let d = ad(vec![0; 16], SampleFormat::S16p, 2, 4);
        assert_eq!(d.plane(1).unwrap().len(), 8);
        assert_eq!(d.plane(1), Some(&d.data[8..16]));
        // Zero-initialized by new().
        let d = AudioData::new(SampleFormat::S16p, 2, 4);
        assert!(d.data.iter().all(|&b| b == 0));
        assert_eq!(d.bps, 2);
        // Clone shares the Arc; plane_bytes_mut copies on write.
        let mut a = AudioData::new(SampleFormat::S16, 1, 4);
        let b = a.clone();
        assert!(std::sync::Arc::ptr_eq(&a.data, &b.data));
        a.data_mut().copy_from_slice(&[9u8; 8]);
        assert!(b.data.iter().all(|&x| x == 0));
        assert!(!std::sync::Arc::ptr_eq(&a.data, &b.data));
    }

    // -- stride geometry through swri_audio_convert ------------------------------

    #[test]
    fn packed_interleaved_to_planar_and_back() {
        // Stereo packed s16 [L0,R0,L1,R1,L2,R2] -> s16p: is = ch_count*bps =
        // 4, os = bps = 2 via the ch[i] = base + i*bps aliases.
        let l = [100i16, -200, 300];
        let r = [-400i16, 500, -600];
        let mut packed = Vec::new();
        for i in 0..3 {
            packed.extend_from_slice(&l[i].to_le_bytes());
            packed.extend_from_slice(&r[i].to_le_bytes());
        }
        let input = ad(packed.clone(), SampleFormat::S16, 2, 3);
        let mut out = AudioData::new(SampleFormat::S16p, 2, 3);
        let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut out, &input, 3).unwrap();
        assert_eq!(as_i16(out.plane(0).unwrap()), l.to_vec());
        assert_eq!(as_i16(out.plane(1).unwrap()), r.to_vec());
        // Reverse: planar -> packed reproduces the interleaved order.
        let mut back = AudioData::new(SampleFormat::S16, 2, 3);
        let ctx = AudioConvert::new(SampleFormat::S16, SampleFormat::S16p, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut back, &out, 3).unwrap();
        assert_eq!(back.data.to_vec(), packed);
    }

    #[test]
    fn packed_to_packed_channel_stride() {
        // Stereo packed u8 [0,128,255,128] -> packed s16: proves
        // is = ch_count*bps and the per-channel base offsets (audioconvert.c:256-257).
        let input = ad(vec![0u8, 128, 255, 128], SampleFormat::U8, 2, 2);
        let mut out = AudioData::new(SampleFormat::S16, 2, 2);
        let ctx = AudioConvert::new(SampleFormat::S16, SampleFormat::U8, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        // ch0 reads offsets 0,2 (values 0, 255) -> [-32768, 32512];
        // ch1 reads offsets 1,3 (values 128, 128) -> [0, 0]; interleaved out.
        assert_eq!(out.data.to_vec(), s16_bytes(&[-32768i16, 0, 32512, 0]));
    }

    #[test]
    fn ch_map_reorder_and_mute() {
        // audioconvert.c:255-257 — ch_map reorder + mute; ch_map present
        // disables the copy fast path (scalar even for identical formats).
        let packed = s16_bytes(&[100i16, -400, -200, 500]);
        let input = ad(packed.clone(), SampleFormat::S16, 2, 2);
        // Swap: ch_map = [1, 0].
        let mut out = AudioData::new(SampleFormat::S16, 2, 2);
        let ctx =
            AudioConvert::new(SampleFormat::S16, SampleFormat::S16, 2, Some(&[1, 0])).unwrap();
        assert_eq!(ctx.simd_bps, None);
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(out.data.to_vec(), s16_bytes(&[-400i16, 100, 500, -200]));
        // Mute channel 0: s16 input -> default silence [0;8] -> zeros.
        let mut out = AudioData::new(SampleFormat::S16, 2, 2);
        let ctx =
            AudioConvert::new(SampleFormat::S16, SampleFormat::S16, 2, Some(&[-1, 0])).unwrap();
        assert_eq!(ctx.silence, [0u8; 8]);
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(out.data.to_vec(), s16_bytes(&[0i16, 100, 0, -200]));
        // Mute with u8 input: silence = [0x80; 8] (audioconvert.c:175-176),
        // every len sample reads the SAME silence bytes (is = 0).
        let input = ad(vec![10u8, 20, 30, 40], SampleFormat::U8, 2, 2);
        let mut out = AudioData::new(SampleFormat::U8, 2, 2);
        let ctx = AudioConvert::new(SampleFormat::U8, SampleFormat::U8, 2, Some(&[-1, 0])).unwrap();
        assert_eq!(ctx.silence, [0x80; 8]);
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(out.data.to_vec(), vec![0x80u8, 10, 0x80, 30]);
    }

    #[test]
    fn silence_bytes_per_input_format() {
        // audioconvert.c:175-182.
        let ctx = AudioConvert::new(SampleFormat::S16, SampleFormat::U8p, 2, None).unwrap();
        assert_eq!(ctx.silence, [0x80; 8]);
        let ctx = AudioConvert::new(SampleFormat::Dsd, SampleFormat::Dsd, 2, None).unwrap();
        assert_eq!(ctx.silence, [0x69; 8]);
        let ctx = AudioConvert::new(SampleFormat::S16, SampleFormat::Flt, 2, None).unwrap();
        assert_eq!(ctx.silence, [0u8; 8]);
    }

    #[test]
    fn single_channel_planar_normalization() {
        // audioconvert.c:167-170 — channels == 1 normalizes both formats to
        // planar AFTER the kernel pick; an unequal pair becomes equal, so the
        // copy path activates (simd_bps = Some(2)).
        let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 1, None).unwrap();
        assert_eq!(ctx.simd_bps, Some(2));
        let input = ad(s16_bytes(&[111i16, -222]), SampleFormat::S16, 1, 2);
        let mut out = AudioData::new(SampleFormat::S16p, 1, 2);
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(out.data.to_vec(), s16_bytes(&[111i16, -222]));
        // Equal + 2 channels (no normalization needed) also takes the copy
        // path; a ch_map forces the scalar path even when equal.
        assert_eq!(
            AudioConvert::new(SampleFormat::S16, SampleFormat::S16, 2, None)
                .unwrap()
                .simd_bps,
            Some(2)
        );
        assert_eq!(
            AudioConvert::new(SampleFormat::S16, SampleFormat::S16, 2, Some(&[0, 1]))
                .unwrap()
                .simd_bps,
            None
        );
        assert_eq!(
            AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 2, None)
                .unwrap()
                .simd_bps,
            None
        );
    }

    #[test]
    fn simd_copy_fast_path_len_split() {
        // audioconvert.c:235-252 — C would split off = len&!15 + remainder;
        // the flattened full copy is output-identical for any len.
        for len in [64usize, 5] {
            // s16p -> s16p stereo (C split at off=48 / off=0).
            let pat: Vec<i16> = (0..(2 * len))
                .map(|i| (i as i16).wrapping_mul(2654435761u32 as i16))
                .collect();
            let mut data = Vec::new();
            for i in 0..len {
                data.extend_from_slice(&pat[i].to_le_bytes());
                data.extend_from_slice(&pat[len + i].to_le_bytes());
            }
            let input = ad(data.clone(), SampleFormat::S16p, 2, len);
            let mut out = AudioData::new(SampleFormat::S16p, 2, len);
            let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16p, 2, None).unwrap();
            swri_audio_convert(&ctx, &mut out, &input, len).unwrap();
            assert_eq!(out.data.to_vec(), data, "s16p len={len}");
            // fltp -> fltp.
            let fpat: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 - 8.5).collect();
            let bytes: Vec<u8> = fpat.iter().flat_map(|f| f.to_le_bytes()).collect();
            let input = ad(bytes.clone(), SampleFormat::Fltp, 1, len);
            let mut out = AudioData::new(SampleFormat::Fltp, 1, len);
            let ctx = AudioConvert::new(SampleFormat::Fltp, SampleFormat::Fltp, 1, None).unwrap();
            swri_audio_convert(&ctx, &mut out, &input, len).unwrap();
            assert_eq!(out.data.to_vec(), bytes, "fltp len={len}");
        }
        // Packed s16 -> s16 stereo copies len*ch_count samples in one plane.
        let data = s16_bytes(&(0..16i16).collect::<Vec<_>>());
        let input = ad(data.clone(), SampleFormat::S16, 2, 8);
        let mut out = AudioData::new(SampleFormat::S16, 2, 8);
        let ctx = AudioConvert::new(SampleFormat::S16, SampleFormat::S16, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut out, &input, 8).unwrap();
        assert_eq!(out.data.to_vec(), data);
    }

    #[test]
    fn dsd_to_dsd_passthrough() {
        // audioconvert.c:90 + :177-179 — DSD pass-through works; only
        // DSD->FLT is blocked (dsd2pcm unported).
        let data = vec![0x69u8, 0x96, 0xAA, 0x55];
        let input = ad(data.clone(), SampleFormat::Dsd, 2, 2);
        let mut out = AudioData::new(SampleFormat::Dsd, 2, 2);
        let ctx = AudioConvert::new(SampleFormat::Dsd, SampleFormat::Dsd, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(out.data.to_vec(), data);
    }

    // -- skip / guard / no-op semantics ------------------------------------------

    #[test]
    fn skipped_out_channel() {
        // audioconvert.c:259-260 — an absent out channel (empty backing
        // alias) is skipped without panic while other channels convert.
        // Scalar path (s16 -> s16p, formats differ).
        let input = ad(
            s16_bytes(&[100i16, -400, -200, 500]),
            SampleFormat::S16,
            2,
            2,
        );
        // Backing sized for plane 0 only: plane(1) -> None -> continue.
        let mut out = ad(vec![0xEE; 4], SampleFormat::S16p, 2, 2);
        let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(as_i16(&out.data), vec![100, -200]); // L0, L1 de-interleaved
    }

    #[test]
    fn bounds_guards() {
        // Out plane too small for len samples -> BufferTooSmall.
        let input = ad(s16_bytes(&[1i16, 2, 3, 4]), SampleFormat::S16, 2, 2);
        let mut out = AudioData::new(SampleFormat::S16p, 2, 1); // 1 sample capacity
        let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 2, None).unwrap();
        assert_eq!(
            swri_audio_convert(&ctx, &mut out, &input, 2),
            Err(Error::BufferTooSmall)
        );
        // Input plane shorter than (len-1)*is + bps -> BufferTooSmall.
        let input = ad(s16_bytes(&[1i16, 2]), SampleFormat::S16, 2, 1); // 1 frame only
        let mut out = AudioData::new(SampleFormat::S16p, 2, 2);
        assert_eq!(
            swri_audio_convert(&ctx, &mut out, &input, 2),
            Err(Error::BufferTooSmall)
        );
        // ctx.channels != out.ch_count: C's av_assert0 at :216 ABORTS in
        // debug builds — the port's debug_assert does the same, so the
        // graceful Error::InvalidArgument path is release-only and cannot
        // be asserted in this suite.
    }

    #[test]
    fn len_zero_no_op() {
        // C returns before touching any plane (off = 0 == len, :250-251).
        let input = ad(vec![], SampleFormat::S16, 2, 0);
        let mut out = AudioData::new(SampleFormat::S16p, 2, 0);
        let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 2, None).unwrap();
        assert_eq!(swri_audio_convert(&ctx, &mut out, &input, 0), Ok(()));
        // And with real buffers, nothing is written.
        let input = ad(s16_bytes(&[7i16, 8]), SampleFormat::S16, 2, 1);
        let mut out = AudioData::new(SampleFormat::S16p, 2, 1);
        assert_eq!(swri_audio_convert(&ctx, &mut out, &input, 0), Ok(()));
        assert!(out.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn conversion_counts_exact_len() {
        // Exactly len samples converted: sentinel tail untouched.
        let input = ad(
            s16_bytes(&[100i16, -400, -200, 500]),
            SampleFormat::S16,
            2,
            2,
        );
        let mut out = ad(vec![0xAA; 16], SampleFormat::S16p, 2, 4); // capacity 4
        let ctx = AudioConvert::new(SampleFormat::S16p, SampleFormat::S16, 2, None).unwrap();
        swri_audio_convert(&ctx, &mut out, &input, 2).unwrap();
        assert_eq!(as_i16(&out.data[0..4]), vec![100, -200]);
        // unwritten capacity KEEPS the ad() sentinel (0xAAAA): the kernels
        // write exactly len samples and nothing else.
        assert_eq!(as_i16(&out.data[4..8]), vec![-21846, -21846]);
        assert_eq!(as_i16(&out.data[8..12]), vec![-400, 500]);
        assert_eq!(&out.data[12..16], &[0xAA, 0xAA, 0xAA, 0xAA]); // sentinel
    }
}
