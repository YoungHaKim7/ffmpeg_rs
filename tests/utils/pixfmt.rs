use ffmpeg_rs::PixelFormat;

#[test]
fn names_round_trip() {
    for &fmt in PixelFormat::ALL {
        assert_eq!(
            PixelFormat::from_name(fmt.name()),
            Some(fmt),
            "{}",
            fmt.name()
        );
    }
}

#[test]
fn aliases_resolve() {
    assert_eq!(PixelFormat::from_name("gray8"), Some(PixelFormat::Gray8));
    assert_eq!(PixelFormat::from_name("y8"), Some(PixelFormat::Gray8));
    assert_eq!(
        PixelFormat::from_name("yuv420p10"),
        Some(PixelFormat::Yuv420p10le)
    );
    assert_eq!(PixelFormat::from_name("nosuch"), None);
}

#[test]
fn canonical_gray_spelling() {
    // pixdesc.c names GRAY8 "gray", not "gray8".
    assert_eq!(PixelFormat::Gray8.name(), "gray");
}
