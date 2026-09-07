use ffmpeg_rs::Rational;

/// Port of the exhaustive small-value sweep in `libavutil/tests/rational.c`:
/// every arithmetic op is cross-checked against exact f64 math.
#[test]
fn arithmetic_matches_float_reference() {
    for a_num in -2..=2i32 {
        for a_den in -2..=2i32 {
            for b_num in -2..=2i32 {
                for b_den in -2..=2i32 {
                    let a = Rational::new(a_num, a_den);
                    let b = Rational::new(b_num, b_den);
                    if a.den == 0 || b.den == 0 || b.to_f64() == 0.0 {
                        continue; // infinities / div-by-zero out of scope
                    }
                    let fa = a.to_f64();
                    let fb = b.to_f64();
                    let close = |q: Rational, f: f64| (q.to_f64() - f).abs() < 1e-9;
                    assert!(close(a + b, fa + fb), "{a:?} + {b:?}");
                    assert!(close(a - b, fa - fb), "{a:?} - {b:?}");
                    assert!(close(a * b, fa * fb), "{a:?} * {b:?}");
                    assert!(close(a / b, fa / fb), "{a:?} / {b:?}");
                }
            }
        }
    }
}

#[test]
fn reduce_finds_lowest_terms() {
    let (r, exact) = Rational::reduce(4, 8, i32::MAX as i64);
    assert_eq!(r, Rational::new(1, 2));
    assert!(exact);
    let (r, _) = Rational::reduce(-6, 9, i32::MAX as i64);
    assert_eq!(r, Rational::new(-2, 3));
    let (r, _) = Rational::reduce(0, 5, i32::MAX as i64);
    assert_eq!(r, Rational::new(0, 1));
    let (r, _) = Rational::reduce(7, 0, i32::MAX as i64); // infinity
    assert_eq!(r, Rational::new(1, 0));
}

#[test]
fn reduce_respects_max_and_reports_inexact() {
    // 1,000,000,000,001 / 1,000,000,000,000 cannot fit max=1000 exactly.
    let (r, exact) = Rational::reduce(1_000_000_000_001, 1_000_000_000_000, 1000);
    assert!(!exact);
    assert!(r.den <= 1000 && r.num <= 1000);
    assert!((r.to_f64() - 1.000000000001).abs() < 1e-6, "got {r:?}");
}

#[test]
fn cmp_q_orders_and_rejects_unknown() {
    use std::cmp::Ordering;
    assert_eq!(
        Rational::new(1, 2).cmp_q(Rational::new(2, 3)),
        Some(Ordering::Less)
    );
    assert_eq!(
        Rational::new(-1, 2).cmp_q(Rational::new(1, -2)),
        Some(Ordering::Equal)
    );
    // Negative denominator flips the visible sign.
    assert_eq!(
        Rational::new(1, 2).cmp_q(Rational::new(1, -2)),
        Some(Ordering::Greater)
    );
    // Infinities compare by sign.
    assert_eq!(
        Rational::new(1, 0).cmp_q(Rational::new(1, 0)),
        Some(Ordering::Equal)
    );
    assert_eq!(
        Rational::new(1, 0).cmp_q(Rational::new(-1, 0)),
        Some(Ordering::Greater)
    );
    // Unknown (0/0) vs a value is incomparable (INT_MIN in C)…
    assert_eq!(Rational::UNKNOWN.cmp_q(Rational::ONE), None);
    assert_eq!(Rational::UNKNOWN.cmp_q(Rational::UNKNOWN), None);
    // …but zero vs infinity is a normal comparison (tmp != 0).
    assert_eq!(
        Rational::ZERO.cmp_q(Rational::new(1, 0)),
        Some(Ordering::Less)
    );
}

#[test]
fn d2q_round_trips_common_rates() {
    for d in [1.0, 25.0, 29.97, 23.976, 0.5, 59.94] {
        let q = Rational::from_f64(d, 4096);
        assert!((q.to_f64() - d).abs() < 1e-9, "{d} -> {q:?}");
    }
    assert_eq!(Rational::from_f64(f64::NAN, 4096), Rational::UNKNOWN);
    let inf = Rational::from_f64(1e18, 4096);
    assert_eq!(inf, Rational::new(1, 0));
}

#[test]
fn inv_swaps_and_preserves_specials() {
    assert_eq!(Rational::new(3, 4).inv(), Rational::new(4, 3));
    assert_eq!(Rational::UNKNOWN.inv(), Rational::UNKNOWN);
}
