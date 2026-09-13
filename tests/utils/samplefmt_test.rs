use ffmpeg_rs::{
    Error,
    util::samplefmt::{
        SampleFormat, samples_alloc, samples_fill_arrays, samples_get_buffer_size, silence_byte,
    },
};

// ---- bytes_per_sample: all 13 variants (sample_fmt_info[].bits >> 3) ----
#[test]
fn bytes_per_sample_table() {
    let expected = [
        (SampleFormat::U8, 1usize),
        (SampleFormat::S16, 2),
        (SampleFormat::S32, 4),
        (SampleFormat::Flt, 4),
        (SampleFormat::Dbl, 8),
        (SampleFormat::U8p, 1),
        (SampleFormat::S16p, 2),
        (SampleFormat::S32p, 4),
        (SampleFormat::Fltp, 4),
        (SampleFormat::Dblp, 8),
        (SampleFormat::S64, 8),
        (SampleFormat::S64p, 8),
        (SampleFormat::Dsd, 1),
    ];
    assert_eq!(expected.len(), SampleFormat::ALL.len());
    for (fmt, bps) in expected {
        assert_eq!(fmt.bytes_per_sample(), bps, "{fmt}");
    }
}

// ---- name/from_name round-trip; no suffix rule, case-sensitive ----
#[test]
fn name_round_trip() {
    for fmt in SampleFormat::ALL {
        assert_eq!(SampleFormat::from_name(fmt.name()), Some(*fmt));
    }
    // Table names, not a trailing-'p' rule (samplefmt.c:59-67 is a
    // plain strcmp loop).
    assert_eq!(SampleFormat::from_name("s16p"), Some(SampleFormat::S16p));
    assert_eq!(SampleFormat::from_name("u8p"), Some(SampleFormat::U8p));
    assert_eq!(SampleFormat::from_name("fltp"), Some(SampleFormat::Fltp));
    assert_eq!(SampleFormat::from_name("s64"), Some(SampleFormat::S64));
    assert_eq!(SampleFormat::from_name("dsd"), Some(SampleFormat::Dsd));
    // No such C entries.
    assert_eq!(SampleFormat::from_name("s24"), None);
    assert_eq!(SampleFormat::from_name("s16le"), None);
    // Case-sensitive.
    assert_eq!(SampleFormat::from_name("S16"), None);
    assert_eq!(SampleFormat::from_name(""), None);
}

// ---- enum order = C discriminants (S64/S64p AFTER Dblp) ----
#[test]
fn discriminant_order_matches_c() {
    let all = SampleFormat::ALL;
    assert_eq!(all.iter().position(|f| *f == SampleFormat::Dblp), Some(9));
    assert_eq!(all.iter().position(|f| *f == SampleFormat::S64), Some(10));
    assert_eq!(all.iter().position(|f| *f == SampleFormat::S64p), Some(11));
    assert_eq!(all.iter().position(|f| *f == SampleFormat::Dsd), Some(12));
    assert_eq!(all.len(), 13);
    // Ord follows declaration order, like C integer comparisons.
    assert!(SampleFormat::Dblp < SampleFormat::S64);
    assert!(SampleFormat::S64 < SampleFormat::S64p);
    assert!(SampleFormat::S64p < SampleFormat::Dsd);
}

// ---- packed()/planar()/alt() (samplefmt.c:69-94) ----
#[test]
fn planar_packed_alt() {
    assert_eq!(SampleFormat::S16.planar(), SampleFormat::S16p);
    assert_eq!(SampleFormat::S16p.packed(), SampleFormat::S16);
    assert_eq!(SampleFormat::U8.planar(), SampleFormat::U8p);
    assert_eq!(SampleFormat::Fltp.packed(), SampleFormat::Flt);
    assert_eq!(SampleFormat::Dblp.packed(), SampleFormat::Dbl);
    assert_eq!(SampleFormat::S64.planar(), SampleFormat::S64p);
    // Identity when already in the requested form.
    assert_eq!(SampleFormat::S16p.planar(), SampleFormat::S16p);
    assert_eq!(SampleFormat::S16.packed(), SampleFormat::S16);
    assert_eq!(SampleFormat::Flt.alt(false), SampleFormat::Flt);
    assert_eq!(SampleFormat::Fltp.alt(true), SampleFormat::Fltp);
    // Dsd is a fixed point for all three (c:49).
    assert_eq!(SampleFormat::Dsd.planar(), SampleFormat::Dsd);
    assert_eq!(SampleFormat::Dsd.packed(), SampleFormat::Dsd);
    assert_eq!(SampleFormat::Dsd.alt(true), SampleFormat::Dsd);
    assert_eq!(SampleFormat::Dsd.alt(false), SampleFormat::Dsd);
    assert!(!SampleFormat::Dsd.is_planar());
}

// ---- samples_get_buffer_size worked examples (spec-pinned) ----
#[test]
fn buffer_size_examples() {
    // s16 packed 2ch 10 samples, align=0: samples -> 32, line 32*2*2.
    assert_eq!(
        samples_get_buffer_size(2, 10, SampleFormat::S16, 0),
        Ok((128, 128))
    );
    // s16 planar 2ch 10, align=0: line 32*2, total *2.
    assert_eq!(
        samples_get_buffer_size(2, 10, SampleFormat::S16p, 0),
        Ok((64, 128))
    );
    // align=1: no 32-rounding.
    assert_eq!(
        samples_get_buffer_size(2, 10, SampleFormat::S16, 1),
        Ok((40, 40))
    );
    assert_eq!(
        samples_get_buffer_size(2, 10, SampleFormat::S16p, 1),
        Ok((20, 40))
    );
    // u8 mono 5, align=0 -> 32-sample rounding.
    assert_eq!(
        samples_get_buffer_size(1, 5, SampleFormat::U8, 0),
        Ok((32, 32))
    );
    // dsd 1ch 8, align=1: compact.
    assert_eq!(
        samples_get_buffer_size(1, 8, SampleFormat::Dsd, 1),
        Ok((8, 8))
    );
    // Already a multiple of 32: unchanged.
    assert_eq!(
        samples_get_buffer_size(1, 32, SampleFormat::S16, 0),
        Ok((64, 64))
    );
}

#[test]
fn buffer_size_errors() {
    assert!(matches!(
        samples_get_buffer_size(2, 0, SampleFormat::S16, 0),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        samples_get_buffer_size(0, 10, SampleFormat::S16, 0),
        Err(Error::InvalidArgument(_))
    ));
    // Checked math: usize::MAX samples overflows the byte computation
    // (C: INT_MAX guards at c:134-136/142-144 -> OutOfRange here).
    assert!(matches!(
        samples_get_buffer_size(2, usize::MAX, SampleFormat::S16, 1),
        Err(Error::OutOfRange)
    ));
    // 32-rounding overflow.
    assert!(matches!(
        samples_get_buffer_size(1, usize::MAX - 5, SampleFormat::S16, 0),
        Err(Error::OutOfRange)
    ));
}

// ---- samples_fill_arrays offsets (c:169-178) ----
#[test]
fn fill_arrays_offsets() {
    // Planar 3ch s16, align=0: samples -> 32, line 64.
    let a = samples_fill_arrays(3, 10, SampleFormat::S16p, 0).unwrap();
    assert_eq!(a.linesize, 64);
    assert_eq!(a.plane_offsets, vec![0, 64, 128]);
    assert_eq!(a.buf_size, 192);

    // Packed: exactly one plane, offset [0].
    let b = samples_fill_arrays(3, 10, SampleFormat::S16, 0).unwrap();
    assert_eq!(b.plane_offsets, vec![0]);
    assert_eq!(b.buf_size, 192); // 32 samples * 2 bytes * 3 ch

    // buf_size always equals samples_get_buffer_size's total.
    let (_, total) = samples_get_buffer_size(3, 10, SampleFormat::S16p, 0).unwrap();
    assert_eq!(a.buf_size, total);
    let (_, total_packed) = samples_get_buffer_size(3, 10, SampleFormat::S16, 0).unwrap();
    assert_eq!(b.buf_size, total_packed);
}

// ---- samples_alloc silence (c:203 + c:256-261) ----
#[test]
fn alloc_silence_bytes() {
    let (buf_u8, a) = samples_alloc(1, 4, SampleFormat::U8, 1).unwrap();
    assert_eq!(buf_u8, vec![0x80; 4]);
    assert_eq!(buf_u8.len(), a.buf_size);

    let (buf_s16, _) = samples_alloc(1, 4, SampleFormat::S16, 1).unwrap();
    assert_eq!(buf_s16, vec![0x00; 8]);

    let (buf_u8p, _) = samples_alloc(2, 4, SampleFormat::U8p, 1).unwrap();
    assert_eq!(buf_u8p, vec![0x80; 8]);

    // DSD silence is 0x69 ("only ultrasonic tones, filtered out on
    // playback", samplefmt.c:259).
    let (buf_dsd, _) = samples_alloc(1, 4, SampleFormat::Dsd, 1).unwrap();
    assert_eq!(buf_dsd, vec![0x69; 4]);
    assert_eq!(silence_byte(SampleFormat::Dsd), 0x69);
    assert_eq!(silence_byte(SampleFormat::S16), 0x00);
    assert_eq!(silence_byte(SampleFormat::U8), 0x80);
}
