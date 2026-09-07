use std::sync::Arc;

use ffmpeg_rs::{
    ChromaLocation, ColorRange, ColorSpace, Frame, FrameFlags, NOPTS, PictureType, PixelFormat,
    Rational,
};

#[test]
fn alloc_geometry_yuv420p() {
    let f = Frame::alloc(PixelFormat::Yuv420p, 128, 96).unwrap();
    assert_eq!(f.planes.len(), 3);
    assert_eq!(f.plane(0).len(), 128 * 96);
    assert_eq!(f.plane(1).len(), 64 * 48);
    assert_eq!(f.plane(2).len(), 64 * 48);
    assert_eq!(f.linesize(0), 128);
    assert_eq!(f.linesize(1), 64);
    // Freshly allocated frames are writable (per-plane buffers).
    assert!(f.is_writable());
}

#[test]
fn wrap_buffer_rejects_short_buffers() {
    let short: Arc<[u8]> = Arc::from(vec![0u8; 100]);
    assert!(Frame::wrap_buffer(short.clone(), PixelFormat::Yuv420p, 128, 96).is_err());
    let ok: Arc<[u8]> = Arc::from(vec![0u8; 128 * 96 * 3 / 2]);
    let f = Frame::wrap_buffer(ok, PixelFormat::Yuv420p, 128, 96).unwrap();
    assert_eq!(f.plane(0).len(), 128 * 96);
    // A wrapped (packet-backed) frame is not writable until made so —
    // all its planes share the incoming buffer.
    assert!(!f.is_writable());
}

#[test]
fn writability_follows_refcount() {
    let f = Frame::alloc(PixelFormat::Gray8, 16, 16).unwrap();
    assert!(f.is_writable());
    let clone = f.clone();
    assert!(!f.is_writable()); // clone holds references
    drop(clone);
    assert!(f.is_writable()); // back to unique after the clone drops

    let mut shared = f.clone();
    assert!(shared.make_writable().is_ok());
    assert!(shared.is_writable());
    shared.plane_mut(0)[0] = 42;
    assert_eq!(shared.plane(0)[0], 42);
    assert_eq!(f.plane(0)[0], 0); // original untouched (copy-on-write)
}

#[test]
fn default_is_unallocated() {
    let f = Frame::default();
    assert_eq!(f.pts, NOPTS);
    assert_eq!(f.time_base, Rational::UNKNOWN);
    assert_eq!(f.sample_aspect_ratio, Rational::UNKNOWN);
    assert!(f.planes.is_empty());
}

#[test]
fn copy_props_copies_metadata_not_geometry() {
    let mut src = Frame::alloc(PixelFormat::Yuv420p, 32, 24).unwrap();
    src.pts = 900;
    src.duration = 100;
    src.time_base = Rational::new(1, 90000);
    src.pict_type = PictureType::I;
    src.flags = FrameFlags::KEY;
    src.sample_aspect_ratio = Rational::new(16, 9);
    src.crop_top = 2;
    src.color_range = ColorRange::Mpeg;
    src.color_space = ColorSpace::Bt709;
    src.chroma_location = ChromaLocation::Left;
    src.plane_mut(0)[0] = 0xAB;

    let mut dst = Frame::alloc(PixelFormat::Rgb24, 8, 8).unwrap();
    dst.copy_props(&src);
    assert_eq!(dst.pts, 900);
    assert_eq!(dst.duration, 100);
    assert_eq!(dst.time_base, Rational::new(1, 90000));
    assert_eq!(dst.pict_type, PictureType::I);
    assert!(dst.flags.contains(FrameFlags::KEY));
    assert_eq!(dst.sample_aspect_ratio, Rational::new(16, 9));
    assert_eq!(dst.crop_top, 2);
    assert_eq!(dst.color_range, ColorRange::Mpeg);
    assert_eq!(dst.color_space, ColorSpace::Bt709);
    assert_eq!(dst.chroma_location, ChromaLocation::Left);
    // Geometry and pixels are NOT copied — dst keeps its own.
    assert_eq!(dst.format, PixelFormat::Rgb24);
    assert_eq!(dst.width, 8);
    assert_eq!(dst.height, 8);
    assert_eq!(dst.plane(0)[0], 0);
}
