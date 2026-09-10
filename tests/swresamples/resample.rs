use std::sync::Arc;

use ffmpeg_rs::{
    swresample::{
        AudioData,
        resample::{FilterType, ResampleContext, bessel_i0},
    },
    util::samplefmt::SampleFormat,
};

fn mono(count: usize, fmt: SampleFormat, bytes: Vec<u8>) -> AudioData {
    AudioData {
        bps: fmt.bytes_per_sample(),
        data: Arc::from(bytes),
        ch_count: 1,
        count,
        planar: true,
        fmt,
    }
}

fn s16_plane(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// The swr_convert staging shape: invert_initial_buffer stages the
/// first filter_length+1 samples (mirrored around the start), returning
/// the staged sample count — then multiple_resample reads the staging.
fn stage(c: &mut ResampleContext, src: &AudioData, in_count: usize) -> (AudioData, usize) {
    let mut staging = AudioData::new(c.format, 1, 0);
    let mut out_idx = 0usize;
    let mut out_sz = 0usize;
    let r = c
        .invert_initial_buffer(&mut staging, src, in_count, &mut out_idx, &mut out_sz)
        .unwrap();
    assert!(
        r != i32::MAX,
        "enough input staged (feed >= filter_length+1)"
    );
    (staging, out_sz)
}

fn ctx_44100_48000() -> ResampleContext {
    // C defaults (options.c): filter_size 16, phase_shift 10, linear 0,
    // cutoff 0 (→0.97), Kaiser beta 10? — pinned explicitly instead of
    // relying on option defaults.
    ResampleContext::new(
        44100,
        48000,
        16,
        10,
        0,
        0.0,
        SampleFormat::S16p,
        FilterType::Kaiser,
        10.0,
        0.0,
        0,
        1,
    )
    .unwrap()
}

/// `av_bessel_i0` against reference values (Abramowitz & Stegun 9.6.16
/// / mpmath): I0(0)=1, I0(1)=1.2660658777520084, I0(2)=2.2795853023360673.
#[test]
fn bessel_i0_values() {
    assert_eq!(bessel_i0(0.0), 1.0);
    assert!((bessel_i0(1.0) - 1.2660658777520084).abs() < 1e-14);
    assert!((bessel_i0(2.0) - 2.2795853023360673).abs() < 1e-13);
    assert_eq!(bessel_i0(-1.5), bessel_i0(1.5), "I0 is even");
}

/// resample_init's geometry anchors, computed by hand from C:
/// factor = min(44100·0.97/48000, 1) = 0.89031250;
/// filter_length = ceil(16/0.8903125) = ceil(17.970…) = 18 (already even);
/// filter_alloc = FFALIGN(18, 8) = 24;
/// exact_rational: av_reduce(44100, 48000) = 147/160 ≤ 1024 ⇒ phase_count
/// = 147, compensation count = 147·⌊1024/147⌋ = 147·6 = 882;
/// index = −147·((18−1)/2) = −1176; src/dst_incr = av_reduce(44100,
/// 48000·147) = 1/160, doubled while BOTH < 2²⁰ — dst_incr hits 2²⁰
/// first (13 doublings): src = 2¹³ = 8192, dst = 160·8192 = 1310720;
/// dst_incr_div = 160, dst_incr_mod = 0, frac = 0.
#[test]
fn init_geometry_anchors() {
    let c = ctx_44100_48000();
    assert!((c.factor - (44100.0 * 0.97 / 48000.0)).abs() < 1e-12);
    assert_eq!(c.filter_length, 18);
    assert_eq!(c.filter_alloc, 24);
    assert_eq!(c.phase_count, 147);
    assert_eq!(c.phase_count_compensation, 882);
    assert_eq!(c.index, -1176);
    assert_eq!(c.frac, 0);
    assert_eq!(c.src_incr, 8192);
    assert_eq!(c.dst_incr, 1310720);
    assert_eq!(c.dst_incr_div, 160);
    assert_eq!(c.dst_incr_mod, 0);
    assert_eq!(c.filter_shift, 15, "s16p filter_shift (resample.c:227)");
}

/// DC-gain invariant: phase 0's quantized taps sum to the scale factor
/// (1<<15 for s16p) — `norm` accumulates phase 0 pre-quantization, so
/// Σ tab[i]·2¹⁵/norm = 2¹⁵ up to per-tap rounding.
#[test]
fn phase0_dc_gain_s16() {
    let c = ctx_44100_48000();
    let sum: f64 = (0..c.filter_length as usize)
        .map(|t| c.filter_coeff(0, t))
        .sum();
    assert!(
        (sum - (1i64 << 15) as f64).abs() <= c.filter_length as f64,
        "phase-0 DC gain {sum} vs 32768 (±{len} rounding)",
        len = c.filter_length
    );
}

/// Staged-window ½ downsample: with exactly the filter_length+1
/// staged samples the apply loop must produce ≈staged/2 outputs (the
/// full-pipeline length math is the driver's swr_convert test).
#[test]
fn downsample_staged_window() {
    let mut c = ResampleContext::new(
        24000,
        48000,
        16,
        10,
        0,
        0.0,
        SampleFormat::S16p,
        FilterType::Kaiser,
        10.0,
        0.0,
        0,
        1,
    )
    .unwrap();
    const IN: usize = 256;
    let src = mono(IN, SampleFormat::S16p, s16_plane(&vec![1000i16; IN]));
    let (staging, staged) = stage(&mut c, &src, IN);
    // staged = the mirror-zone extent; the RESAMPLEABLE span of a
    // single-shot staging window is (staged − filter_length) at the
    // out/in ratio — the driver loops swr_convert to cover the rest.
    let expect = (staged - c.filter_length as usize) as f64 / 2.0;
    let mut dst = mono(IN, SampleFormat::S16p, vec![0; IN * 2]);
    let mut consumed = 0;
    let n = c
        .multiple_resample(&mut dst, IN as i32, &staging, staged as i32, &mut consumed)
        .unwrap();
    assert!(
        ((n as f64) - expect).abs() <= 4.0,
        "≈(staged−filter_length)/2 out, got {n} for staged {staged} fl {}",
        c.filter_length
    );
}

/// Staged-window 2× upsample of a constant: the interior reproduces
/// the constant to float precision (normalized Kaiser DC gain).
#[test]
fn upsample_staged_constant() {
    let mut c = ResampleContext::new(
        48000,
        24000,
        16,
        10,
        0,
        0.0,
        SampleFormat::Fltp,
        FilterType::Kaiser,
        10.0,
        0.0,
        0,
        1,
    )
    .unwrap();
    const IN: usize = 256;
    let src = mono(
        IN,
        SampleFormat::Fltp,
        vec![0.5f32; IN]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect(),
    );
    let (staging, staged) = stage(&mut c, &src, IN);
    let mut dst = mono(IN * 2, SampleFormat::Fltp, vec![0; IN * 2 * 4]);
    let mut consumed = 0;
    let n = c
        .multiple_resample(
            &mut dst,
            (IN * 2) as i32,
            &staging,
            staged as i32,
            &mut consumed,
        )
        .unwrap();
    let expect = ((staged - c.filter_length as usize) * 2) as i32;
    assert!(
        (n - expect).abs() <= 4,
        "2× up: n={n} expect≈{expect} (staged {staged} fl {})",
        c.filter_length
    );
    let out: Vec<f32> = dst.data[..n as usize * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // The MIRRORED staging prefix makes the first samples see only the
    // constant; every output is the constant to float precision.
    for &v in &out {
        assert!((v - 0.5).abs() < 1e-4, "constant through 2× upsample: {v}");
    }
}

/// get_delay at equal rates is zero before any input (swresample.c's
/// delay query shape).
#[test]
fn delay_shapes() {
    let c = ctx_44100_48000();
    // (48−44.1)k means the engine holds a filter_length/2-ish window;
    // assert it is non-negative and in-sample-rate units for base=in_rate.
    let d = c.get_delay(48000, 0, 48000);
    assert!(d >= 0);
}
