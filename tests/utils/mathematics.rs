use ffmpeg_rs::{
    Rational,
    fftools::dump::rescale_ts,
    mathematics::{gcd, rescale, rescale_q, rescale_rnd},
    util::mathematics::Rounding,
};

#[test]
fn gcd_basics() {
    assert_eq!(gcd(0, 5), 5);
    assert_eq!(gcd(5, 0), 5);
    assert_eq!(gcd(12, 18), 6);
    assert_eq!(gcd(-12, 18), 6);
    assert_eq!(gcd(i64::MIN, 0), i64::MIN); // matches C shortcut a==0? no: b==0 → a
}

#[test]
fn rescale_moves_between_timebases() {
    // 1 second worth of ticks: 25 fps → 90 kHz.
    assert_eq!(
        rescale_q(25, Rational::new(1, 25), Rational::new(1, 90000)),
        90000
    );
    // And back.
    assert_eq!(
        rescale_q(90000, Rational::new(1, 90000), Rational::new(1, 25)),
        25
    );
}

#[test]
fn rescale_rounds_half_away_from_zero() {
    assert_eq!(rescale_rnd(1, 1, 2, Rounding::NearInf, false), 1);
    assert_eq!(rescale_rnd(3, 1, 2, Rounding::NearInf, false), 2); // 1.5 → 2
    assert_eq!(rescale_rnd(-3, 1, 2, Rounding::NearInf, false), -2); // −1.5 → −2
    assert_eq!(rescale_rnd(3, 1, 2, Rounding::Zero, false), 1);
    assert_eq!(rescale_rnd(3, 1, 2, Rounding::Up, false), 2);
    assert_eq!(rescale_rnd(3, 1, 2, Rounding::Down, false), 1);
    assert_eq!(rescale_rnd(-3, 1, 2, Rounding::Down, false), -2);
}

#[test]
fn rescale_handles_big_values_with_u128_path() {
    // b, c both beyond INT_MAX forces the 128-bit path in C.
    let b = 1i64 << 40;
    let c = 3i64 << 40;
    assert_eq!(rescale(7, b, c), 2); // 7/3 = 2.33 → 2
    assert_eq!(rescale(9, b, c), 3);
    // Without PASS_MINMAX, INT64_MAX computes and overflows to the
    // error sentinel.
    assert_eq!(rescale(i64::MAX, i64::MAX, 1), i64::MIN);
}

#[test]
fn pass_minmax_flag_guards_nopts() {
    assert_eq!(
        rescale_rnd(i64::MIN, 1, 1, Rounding::NearInf, true),
        i64::MIN
    );
    assert_eq!(
        rescale_rnd(i64::MAX, 1, 1, Rounding::NearInf, true),
        i64::MAX
    );
    assert_eq!(
        rescale_ts(i64::MIN, Rational::new(1, 25), Rational::new(1, 90000)),
        i64::MIN
    );
    // Without the flag, INT64_MIN computes: C negates through
    // -(uint64_t), yielding exactly -INT64_MAX (not the sentinel).
    assert_eq!(rescale(i64::MIN, 1, 1), -i64::MAX);
}

#[test]
fn rescale_rejects_bad_denominators() {
    assert_eq!(rescale_rnd(5, 1, 0, Rounding::NearInf, false), i64::MIN);
    assert_eq!(rescale_rnd(5, -1, 1, Rounding::NearInf, false), i64::MIN);
}
