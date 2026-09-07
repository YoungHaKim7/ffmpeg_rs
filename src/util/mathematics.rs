//! Integer math — port of `libavutil/mathematics.{h,c}` (the parts the
//! pipeline needs: `av_gcd`, the `av_rescale` family).
//!
//! `av_rescale_q` is the single most-used timestamp helper in FFmpeg: it moves
//! a PTS between timebases (`frame.pts` in stream timebase → encoder timebase
//! → output timebase) with exact round-to-nearest arithmetic. Getting its
//! rounding and overflow behavior right matters, so this is a faithful port:
//!
//! | C | Rust |
//! |---|---|
//! | `av_gcd` | [`gcd`] (Euclid on magnitudes; same result as Stein's) |
//! | `av_rescale_rnd` | [`rescale_rnd`] |
//! | `av_rescale` | [`rescale`] |
//! | `av_rescale_q` / `av_rescale_q_rnd` | [`rescale_q`] / [`rescale_q_rnd`] |
//!
//! The C 128-bit path (manual 32-bit limb multiply in `mathematics.c`) is
//! expressed here with Rust's native `u128`, which computes the identical
//! `(a·b + r) / c` quotient. Overflow still reports the C way: return
//! `i64::MIN` as a sentinel (`INT64_MIN`), which callers treat as "invalid
//! timestamp" — notably `AV_NOPTS_VALUE` *is* `INT64_MIN`, so an overflowed
//! rescale naturally decays into "no timestamp".

use super::rational::Rational;

/// `enum AVRounding` (`mathematics.h:60`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// `AV_ROUND_ZERO` — toward zero.
    Zero,
    /// `AV_ROUND_INF` — away from zero.
    Inf,
    /// `AV_ROUND_DOWN` — toward −∞.
    Down,
    /// `AV_ROUND_UP` — toward +∞.
    Up,
    /// `AV_ROUND_NEAR_INF` — round to nearest, halves away from zero.
    /// This is what plain `av_rescale_q` uses.
    NearInf,
}

/// `av_gcd` — greatest common divisor; never negative, `gcd(0, x) == |x|`.
pub fn gcd(a: i64, b: i64) -> i64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    // Magnitudes via u64: |i64::MIN| == 2^63 does not fit i64 but fits u64.
    let mut u = a.unsigned_abs();
    let mut v = b.unsigned_abs();
    while v != 0 {
        let t = u % v;
        u = v;
        v = t;
    }
    u as i64 // divides min(|a|,|b|) ≤ i64::MAX, so the cast is lossless.
}

/// `av_rescale_rnd(a, b, c, rnd)` — computes `a·b/c` with the given rounding.
///
/// Contract (asserted in C, enforced here by returning `i64::MIN`):
/// `c > 0`, `b >= 0`. Negative `a` rounds symmetrically (the C code flips the
/// rounding mode and negates).
///
/// `pass_minmax` is C's `AV_ROUND_PASS_MINMAX` flag: with it, `i64::MIN`
/// (`AV_NOPTS_VALUE`) and `i64::MAX` pass through unchanged instead of
/// computing — this is how "no timestamp" survives rescaling. Without it,
/// those inputs compute normally (and typically overflow to the `i64::MIN`
/// error sentinel).
pub fn rescale_rnd(a: i64, b: i64, c: i64, rnd: Rounding, pass_minmax: bool) -> i64 {
    if c <= 0 || b < 0 {
        return i64::MIN;
    }
    if pass_minmax && (a == i64::MIN || a == i64::MAX) {
        return a;
    }

    if a < 0 {
        // Negate with saturation (i64::MIN clamps to i64::MAX like C's
        // FFMAX(a, -INT64_MAX)) and flip the rounding direction.
        let flipped = match rnd {
            Rounding::Down => Rounding::Up,
            Rounding::Up => Rounding::Down,
            other => other,
        };
        let inner = rescale_rnd(-a.max(i64::MIN + 1), b, c, flipped, false);
        // -(uint64_t)x in C wraps; wrapping_neg reproduces it exactly.
        return inner.wrapping_neg();
    }

    // Rounding bias `r`: NEAR_INF adds half of c; INF/UP add c-1.
    let r: u64 = match rnd {
        Rounding::NearInf => (c / 2) as u64,
        Rounding::Inf | Rounding::Up => (c - 1) as u64,
        Rounding::Zero | Rounding::Down => 0,
    };

    // Exact (a·b + r)/c in 128-bit — equivalent to the C limb arithmetic.
    let num = (a as u128) * (b as u128) + r as u128;
    let q = num / (c as u128);
    if q > i64::MAX as u128 {
        return i64::MIN;
    }
    q as i64
}

/// `av_rescale(a, b, c)` — round-to-nearest `a·b/c`.
pub fn rescale(a: i64, b: i64, c: i64) -> i64 {
    rescale_rnd(a, b, c, Rounding::NearInf, false)
}

/// `av_rescale_q(a, bq, cq)` — rescale `a` from unit `bq` to unit `cq`,
/// rounding to nearest. E.g. PTS 25 at 1/25s → 1/90000s units.
pub fn rescale_q(a: i64, bq: Rational, cq: Rational) -> i64 {
    rescale_q_rnd(a, bq, cq, Rounding::NearInf, false)
}

/// `av_rescale_q_rnd` — same, with explicit rounding and PASS_MINMAX flag.
/// This is the shape FFmpeg's own callers use for timestamps, so `pts`
/// values flowing through use `pass_minmax = true`.
pub fn rescale_q_rnd(a: i64, bq: Rational, cq: Rational, rnd: Rounding, pass_minmax: bool) -> i64 {
    let b = bq.num as i64 * cq.den as i64;
    let c = cq.num as i64 * bq.den as i64;
    rescale_rnd(a, b, c, rnd, pass_minmax)
}

/// Timestamp-rescaling convenience: `av_rescale_q_rnd(…, NEAR_INF |
/// PASS_MINMAX)` — the exact call fftools makes for every PTS hop.
pub fn rescale_ts(a: i64, bq: Rational, cq: Rational) -> i64 {
    rescale_q_rnd(a, bq, cq, Rounding::NearInf, true)
}
