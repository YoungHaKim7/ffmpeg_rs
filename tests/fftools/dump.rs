#[test]
fn fourcc_formatting_matches_ffmpeg() {
    assert_eq!(tag_string(tag(b"I420")), "I420");
    assert_eq!(tag_string(tag(b"RGB\x18")), "RGB[24]");
    assert_eq!(tag_string(tag(b"RGB\x10")), "RGB[16]");
    assert_eq!(tag(b"RGB\x18"), 0x18424752);
    assert_eq!(tag(b"I420"), 0x30323449);
}

#[test]
fn fps_strings() {
    assert_eq!(fps_string(Rational::new(10, 1)), "10");
    assert_eq!(fps_string(Rational::new(25, 1)), "25");
    // 30000/1001 → 29.97
    let r = Rational::reduce(30000, 1001, i32::MAX as i64).0;
    assert_eq!(fps_string(r), "29.97");
}

#[test]
fn time_strings() {
    assert_eq!(time_string(1.0), "00:00:01.00");
    assert_eq!(time_string(61.23), "00:01:01.23");
    assert_eq!(time_string(3661.0), "01:01:01.00");
}
