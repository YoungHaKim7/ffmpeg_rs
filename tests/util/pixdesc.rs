use ffmpeg_rs::util::{
    pixdesc::{PixFmtFlags, bits_per_pixel, count_planes, descriptor},
    pixfmt::PixelFormat,
};

#[test]
fn plane_counts() {
    assert_eq!(count_planes(PixelFormat::Yuv420p), 3);
    assert_eq!(count_planes(PixelFormat::Nv12), 2);
    assert_eq!(count_planes(PixelFormat::Yuyv422), 1);
    assert_eq!(count_planes(PixelFormat::Rgb24), 1);
    assert_eq!(count_planes(PixelFormat::Gbrap), 4);
    assert_eq!(count_planes(PixelFormat::Gray8), 1);
}

#[test]
fn bits_per_pixel_matches_c() {
    assert_eq!(bits_per_pixel(PixelFormat::Yuv420p), 12);
    assert_eq!(bits_per_pixel(PixelFormat::Yuyv422), 16);
    assert_eq!(bits_per_pixel(PixelFormat::Rgb24), 24);
    assert_eq!(bits_per_pixel(PixelFormat::Rgba), 32);
    assert_eq!(bits_per_pixel(PixelFormat::Gray8), 8);
    assert_eq!(bits_per_pixel(PixelFormat::Yuv420p10le), 15); // (10·4+10+10)/4
    assert_eq!(bits_per_pixel(PixelFormat::Rgb565le), 16);
}

#[test]
fn flags() {
    assert!(
        descriptor(PixelFormat::Yuv420p)
            .flags
            .contains(PixFmtFlags::PLANAR)
    );
    assert!(
        !descriptor(PixelFormat::Yuv420p)
            .flags
            .contains(PixFmtFlags::RGB)
    );
    assert!(
        descriptor(PixelFormat::Gbrp)
            .flags
            .contains(PixFmtFlags::PLANAR)
    );
    assert!(
        descriptor(PixelFormat::Gbrp)
            .flags
            .contains(PixFmtFlags::RGB)
    );
    assert!(
        descriptor(PixelFormat::Rgba)
            .flags
            .contains(PixFmtFlags::ALPHA)
    );
    assert!(
        !descriptor(PixelFormat::Yuyv422)
            .flags
            .contains(PixFmtFlags::PLANAR)
    );
}
