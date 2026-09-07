use ffmpeg_rs::{
    PixelFormat,
    imgutils::{check_size, copy_to_buffer, fill_linesizes, fill_plane_sizes, get_buffer_size},
};

#[test]
fn linesizes_at_odd_and_even_widths() {
    // (format, width, expected [4] linesizes)
    let cases: &[(PixelFormat, u32, [usize; 4])] = &[
        (PixelFormat::Yuv420p, 7, [7, 4, 4, 0]),
        (PixelFormat::Yuv420p, 8, [8, 4, 4, 0]),
        (PixelFormat::Yuv420p, 128, [128, 64, 64, 0]),
        (PixelFormat::Yuv422p, 7, [7, 4, 4, 0]),
        (PixelFormat::Yuv422p, 8, [8, 4, 4, 0]),
        (PixelFormat::Yuv444p, 7, [7, 7, 7, 0]),
        (PixelFormat::Yuv420p10le, 7, [14, 8, 8, 0]),
        (PixelFormat::Yuv420p10le, 8, [16, 8, 8, 0]),
        (PixelFormat::Yuv444p10le, 7, [14, 14, 14, 0]),
        (PixelFormat::Yuv420p16le, 7, [14, 8, 8, 0]),
        (PixelFormat::Nv12, 7, [7, 8, 0, 0]), // UV plane = 2*ceil(7/2)
        (PixelFormat::Nv12, 8, [8, 8, 0, 0]),
        (PixelFormat::Nv21, 7, [7, 8, 0, 0]),
        (PixelFormat::Yuyv422, 7, [16, 0, 0, 0]), // 4 * ceil(7/2)
        (PixelFormat::Yuyv422, 8, [16, 0, 0, 0]),
        (PixelFormat::Uyvy422, 7, [16, 0, 0, 0]),
        (PixelFormat::Gray8, 7, [7, 0, 0, 0]),
        (PixelFormat::Gray16le, 7, [14, 0, 0, 0]),
        (PixelFormat::Rgb24, 7, [21, 0, 0, 0]),
        (PixelFormat::Rgb24, 8, [24, 0, 0, 0]),
        (PixelFormat::Bgr24, 7, [21, 0, 0, 0]),
        (PixelFormat::Rgba, 7, [28, 0, 0, 0]),
        (PixelFormat::Rgb565le, 7, [14, 0, 0, 0]),
        (PixelFormat::Gbrp, 7, [7, 7, 7, 0]),
        (PixelFormat::Gbrap, 7, [7, 7, 7, 7]),
    ];
    for &(fmt, w, want) in cases {
        assert_eq!(fill_linesizes(fmt, w).unwrap(), want, "{fmt} @ {w}");
    }
    // Every format in the enum gets a sane linesize (monotone in width,
    // non-zero on plane 0).
    for &fmt in PixelFormat::ALL {
        let l7 = fill_linesizes(fmt, 7).unwrap();
        let l8 = fill_linesizes(fmt, 8).unwrap();
        assert!(l7[0] > 0 && l8[0] >= l7[0], "{fmt}: {l7:?} {l8:?}");
    }
}

#[test]
fn buffer_size_yuv420p_128x96() {
    // 128*96 + 2 * (64*48) = 12288 + 6144 = 18432 — matches a real Y4M
    // frame payload exactly.
    assert_eq!(
        get_buffer_size(PixelFormat::Yuv420p, 128, 96, 1).unwrap(),
        18432
    );
}

#[test]
fn buffer_size_matches_y4m_frame_layout() {
    for &(fmt, w, h) in &[
        (PixelFormat::Yuv444p, 64u32, 48u32),
        (PixelFormat::Gray8, 100, 100),
        (PixelFormat::Yuyv422, 64, 48),
        (PixelFormat::Nv12, 64, 48),
    ] {
        let s = get_buffer_size(fmt, w, h, 1).unwrap();
        let ls = fill_linesizes(fmt, w).unwrap();
        let sizes = fill_plane_sizes(fmt, h, &ls).unwrap();
        assert_eq!(s, sizes.iter().sum::<usize>(), "{fmt}");
    }
}

#[test]
fn zero_size_rejected() {
    assert!(check_size(0, 16).is_err());
    assert!(check_size(16, 0).is_err());
    assert!(check_size(16, 16).is_ok());
}

#[test]
fn copy_to_buffer_packs_rows() {
    // 4x2 yuv420p with distinct source linesizes (stride 8) — the copy
    // must compact rows. Linesizes: Y 4, U/V 2. The Y plane spans
    // 2 rows at stride 8, so row 1 lives at byte 8; source buffers must
    // cover stride*rows bytes even though only `linesize` per row is read.
    let y = [1u8, 2, 3, 4, 99, 99, 99, 99, 5, 6, 7, 8];
    let u = [9u8, 10, 11, 12]; // 1 row, stride 4, only 2 used
    let v = [13u8, 14, 15, 16];
    let planes: [&[u8]; 3] = [&y, &u, &v];
    let src_ls = [8, 4, 4, 0];
    let out = copy_to_buffer(PixelFormat::Yuv420p, 4, 2, 1, &planes, &src_ls).unwrap();
    assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]);
    assert_eq!(
        out.len(),
        get_buffer_size(PixelFormat::Yuv420p, 4, 2, 1).unwrap()
    );
}
