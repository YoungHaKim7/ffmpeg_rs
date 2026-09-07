//! Rational numbers — port of `libavutil/rational.{h,c}`.
//!
//! FFmpeg carries *every* timestamp, framerate, aspect ratio and timebase as an
//! exact rational (`AVRational { num, den }`) instead of a float, so that PTS
//! arithmetic never drifts. This module ports that type near 1:1.
//!
//! Mapping notes:
//!
//! | C | Rust |
//! |---|---|
//! | `AVRational` | [`Rational`] |
//! | `av_make_q(n, d)` | [`Rational::new`](crate::util::rational::Rational::new) |
//! | `av_reduce(&n, &d, N, D, max)` | [`Rational::reduce`](crate::util::rational::Rational::reduce) |
//! | `av_add_q` / `av_sub_q` / `av_mul_q` / `av_div_q` | [`Rational::add`]… via `ops` |
//! | `av_cmp_q` | [`Rational::cmp_q`](crate::util::rational::Rational::cmp_q) (returns `Option`) |
//! | `av_q2d` | [`Rational::to_f64`](crate::util::rational::Rational::to_f64) |
//! | `av_d2q(d, max)` | [`Rational::from_f64`](crate::util::rational::Rational::from_f64) |
//! | `av_inv_q` | [`Rational::inv`](crate::util::rational::Rational::inv) |
//!
//! `0/0` is FFmpeg's "unknown" rational (SAR of unspecified streams etc.) and
//! is representable here too; `Rational::default()` is `0/0`, matching C's
//! zero-initialization of the struct.

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::mathematics::gcd;

/// Exact rational number (`AVRational`).
///
/// Invariant: after `reduce`/arithmetic, `|num|` and `den` fit in `i32` and
/// `den >= 0` — except the special `0/0` ("unknown") and `±1/0` (±∞) forms,
/// which are legal in FFmpeg and preserved here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    /// `av_make_q` — construct without any normalization.
    pub const fn new(num: i32, den: i32) -> Self {
        Rational { num, den }
    }

    /// The `0/1` zero value (distinct from `0/0` = unknown).
    pub const ZERO: Rational = Rational { num: 0, den: 1 };

    /// The `1/1` identity.
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    /// `0/0` — FFmpeg's "unknown" sentinel; also the zero-initialized value.
    pub const UNKNOWN: Rational = Rational { num: 0, den: 0 };

    /// `av_reduce(num, den, max)` — reduce `num/den` to lowest terms with both
    /// parts ≤ `max`, using the continued-fraction ("Stern–Brocot") walk from
    /// `rational.c:35`. Returns the reduced value plus whether the result is
    /// *exact* (C returns this as the `int` function result).
    ///
    /// This is the workhorse behind every arithmetic op; it must handle
    /// `i64`-sized inputs (products of two `i32` rationals) without overflow.
    pub fn reduce(num: i64, den: i64, max: i64) -> (Rational, bool) {
        // a0/a1 are successive continued-fraction convergents; `den` doubles as
        // the "still reducing" flag once the value fits (mirrors the C exactly).
        let mut a0 = Rational::new(0, 1);
        let mut a1 = Rational::new(1, 0);
        let sign = (num < 0) ^ (den < 0);
        let mut num = num.abs();
        let mut den = den.abs();

        let gcd = gcd(num, den);
        if gcd != 0 {
            num /= gcd;
            den /= gcd;
        }
        if num <= max && den <= max {
            a1 = Rational::new(num as i32, den as i32);
            den = 0;
        }

        while den != 0 {
            let x = num / den;
            let next_den = num - den * x;
            let a2n = x * a1.num as i64 + a0.num as i64;
            let a2d = x * a1.den as i64 + a0.den as i64;

            if a2n > max || a2d > max {
                // Convergent overflowed `max`: back off to the largest
                // approximant that fits, preferring the closer one.
                let mut x = i64::MAX;
                if a1.num != 0 {
                    x = (max - a0.num as i64) / a1.num as i64;
                }
                if a1.den != 0 {
                    x = x.min((max - a0.den as i64) / a1.den as i64);
                }
                if den * (2 * x * a1.den as i64 + a0.den as i64) > num * a1.den as i64 {
                    a1 = Rational::new(
                        (x * a1.num as i64 + a0.num as i64) as i32,
                        (x * a1.den as i64 + a0.den as i64) as i32,
                    );
                }
                break;
            }

            a0 = a1;
            a1 = Rational::new(a2n as i32, a2d as i32);
            num = den;
            den = next_den;
        }
        // "Exact" means the reduction terminated without hitting `max` —
        // the C function's return value.
        let exact = den == 0;

        let out = Rational::new(if sign { -a1.num } else { a1.num }, a1.den);
        (out, exact)
    }

    /// `av_cmp_q` (rational.h:89) — total order with escape hatches:
    ///
    /// * different values → `Less`/`Greater` (sign corrected for negative
    ///   denominators — the C `(tmp ^ a.den ^ b.den) >> 63` trick)
    /// * equal values with both denominators non-zero → `Equal`
    /// * two infinities (`±1/0`) → compared by sign
    /// * anything else (`0/0` unknown vs. anything, zero vs. infinity) →
    ///   `None` (C returns `INT_MIN`)
    pub fn cmp_q(self, other: Rational) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        let tmp = self.num as i64 * other.den as i64 - other.num as i64 * self.den as i64;
        if tmp != 0 {
            let negative = (tmp < 0) ^ (self.den < 0) ^ (other.den < 0);
            Some(if negative {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        } else if self.den != 0 && other.den != 0 {
            Some(Ordering::Equal)
        } else if self.num != 0 && other.num != 0 {
            // Both infinite: +∞ > −∞ (C: (a.num>>31) - (b.num>>31)).
            match (self.num > 0, other.num > 0) {
                (true, false) => Some(Ordering::Greater),
                (false, true) => Some(Ordering::Less),
                _ => Some(Ordering::Equal),
            }
        } else {
            None
        }
    }

    /// `av_q2d` — lossy conversion; only for display and probe scoring.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// `av_inv_q` — `{den, num}`. `0/0` and infinities swap into themselves
    /// and their negations respectively, exactly as in C.
    pub const fn inv(self) -> Rational {
        Rational {
            num: self.den,
            den: self.num,
        }
    }

    /// `av_d2q(d, max)` — approximate a double with a rational whose parts
    /// are ≤ `max`. Ports the `frexp`-based scaling from `rational.c:110`:
    /// normalize `d` to a mantissa in `[0.5, 1)`, scale by `2^(62-e)` so the
    /// product is integral, round, and reduce.
    pub fn from_f64(d: f64, max: i32) -> Rational {
        if d.is_nan() {
            return Rational { num: 0, den: 0 };
        }
        if d.abs() > i32::MAX as f64 + 3.0 {
            return Rational {
                num: if d < 0.0 { -1 } else { 1 },
                den: 0,
            };
        }
        let (mantissa, exponent) = frexp(d);
        let _ = mantissa; // C's av_d2q only uses the exponent for scaling
        let exponent = exponent.saturating_sub(1).max(0);
        let den = 1i64 << (62 - exponent);
        // floor(d * den + 0.5) — C's rint-free workaround, kept bit-for-bit.
        let num = (d * den as f64 + 0.5).floor();
        Rational::reduce(num as i64, den, max as i64).0
    }
}

/// C `frexp(d, &e)`: returns `(m, e)` with `d = m·2^e` and `0.5 ≤ |m| < 1`.
/// Implemented via the IEEE-754 exponent field so it is exact (Rust std has
/// no stable `frexp`).
fn frexp(d: f64) -> (f64, i32) {
    if d == 0.0 || !d.is_finite() {
        return (d, 0);
    }
    let bits = d.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    if raw_exp == 0 {
        // Subnormal: renormalize by doubling — subnormals never occur for
        // framerates/SARs, so a loop is fine here.
        let mut m = d;
        let mut e = 0;
        while m.abs() < 0.5 {
            m *= 2.0;
            e -= 1;
        }
        return (m, e);
    }
    // Re-bias to exponent 1022 (= mantissa in [0.5, 1)); value scaled by 2^-1.
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, raw_exp - 1022)
}

impl Add for Rational {
    type Output = Rational;
    /// `av_add_q` — cross-multiply in i64, reduce back into i32 range.
    fn add(self, rhs: Rational) -> Rational {
        let num = self.num as i64 * rhs.den as i64 + rhs.num as i64 * self.den as i64;
        let den = self.den as i64 * rhs.den as i64;
        Rational::reduce(num, den, i32::MAX as i64).0
    }
}

impl Sub for Rational {
    type Output = Rational;
    /// `av_sub_q`.
    fn sub(self, rhs: Rational) -> Rational {
        let num = self.num as i64 * rhs.den as i64 - rhs.num as i64 * self.den as i64;
        let den = self.den as i64 * rhs.den as i64;
        Rational::reduce(num, den, i32::MAX as i64).0
    }
}

impl Mul for Rational {
    type Output = Rational;
    /// `av_mul_q`.
    fn mul(self, rhs: Rational) -> Rational {
        let num = self.num as i64 * rhs.num as i64;
        let den = self.den as i64 * rhs.den as i64;
        Rational::reduce(num, den, i32::MAX as i64).0
    }
}

impl Div for Rational {
    type Output = Rational;
    /// `av_div_q` — multiply by the inverse.
    fn div(self, rhs: Rational) -> Rational {
        self * rhs.inv()
    }
}

impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational {
            num: -self.num,
            den: self.den,
        }
    }
}
