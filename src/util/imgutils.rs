//! Image geometry utilities — port of `libavutil/imgutils.{h,c}`.
//!
//! These are the functions every layer uses to answer "how large is a plane
//! row / a plane / a whole frame in format X at size WxH". The C semantics
//! that matter and are preserved exactly:
//!
//! * A plane's linesize is `max_step_of_plane * ceil(width >> chroma_shift)`,
//!   where the chroma shift applies only when the plane's *widest-step
//!   component* is U or V (`max_step_comp == 1 || == 2`) — this is what makes
//!   NV12's UV plane `2 * ceil(w/2)` while a hypothetical 16-bit Y plane in a
//!   subsampled format keeps its own width.
//! * Plane heights: planes 1 and 2 use `ceil(height >> log2_chroma_h)`.
//! * With `align = 1` (the only alignment Phase 1 uses — FFmpeg pads to 32 in
//!   some paths, we allocate compactly and document it) the buffer size is the
//!   plain sum of plane sizes.

use super::{
    error::{Error, Result},
    pixdesc::{PixFmtDescriptor, descriptor},
    pixfmt::PixelFormat,
};

/// `AV_CEIL_RSHIFT(a, b)` — ceil(a / 2^b).
#[inline]
fn ceil_rshift(a: u32, b: u32) -> u32 {
    (a + (1 << b) - 1) >> b
}

/// `av_image_fill_max_pixsteps` — for each plane, the largest component step
/// and which component (index) provided it. `max_step_comp` drives the
/// chroma-shift decision in [`get_linesize`].
fn fill_max_pixsteps(desc: &PixFmtDescriptor) -> ([u8; 4], [u8; 4]) {
    let mut max_step = [0u8; 4];
    let mut max_step_comp = [0u8; 4];
    for i in 0..4 {
        let comp = &desc.comp[i];
        if comp.step > max_step[comp.plane as usize] {
            max_step[comp.plane as usize] = comp.step;
            max_step_comp[comp.plane as usize] = i as u8;
        }
    }
    (max_step, max_step_comp)
}

/// `image_get_linesize` (imgutils.c:54) — linesize of one plane.
fn image_get_linesize(
    width: u32,
    _plane: usize,
    max_step: u8,
    max_step_comp: u8,
    desc: &PixFmtDescriptor,
) -> Result<usize> {
    // Subsampling applies only when the plane's widest component is chroma.
    let s = if max_step_comp == 1 || max_step_comp == 2 {
        desc.log2_chroma_w
    } else {
        0
    };
    let shifted_w = ceil_rshift(width, s as u32);
    Ok(max_step as usize * shifted_w as usize)
}

/// `av_image_get_linesize`.
pub fn get_linesize(fmt: PixelFormat, width: u32, plane: usize) -> Result<usize> {
    let desc = descriptor(fmt);
    let (max_step, max_step_comp) = fill_max_pixsteps(desc);
    image_get_linesize(width, plane, max_step[plane], max_step_comp[plane], desc)
}

/// `av_image_fill_linesizes` — linesizes for all 4 planes (trailing unused
/// planes are 0).
pub fn fill_linesizes(fmt: PixelFormat, width: u32) -> Result<[usize; 4]> {
    let desc = descriptor(fmt);
    let (max_step, max_step_comp) = fill_max_pixsteps(desc);
    let mut linesizes = [0usize; 4];
    for i in 0..4 {
        linesizes[i] = image_get_linesize(width, i, max_step[i], max_step_comp[i], desc)?;
    }
    Ok(linesizes)
}

/// `av_image_fill_plane_sizes` — byte size of each plane given its linesize
/// and the image height. Planes past `count_planes` stay 0.
pub fn fill_plane_sizes(
    fmt: PixelFormat,
    height: u32,
    linesizes: &[usize; 4],
) -> Result<[usize; 4]> {
    let desc = descriptor(fmt);
    let mut sizes = [0usize; 4];

    sizes[0] = linesizes[0]
        .checked_mul(height as usize)
        .ok_or(Error::OutOfRange)?;

    let mut has_plane = [false; 4];
    for i in 0..desc.nb_components as usize {
        has_plane[desc.comp[i].plane as usize] = true;
    }
    for i in 1..4 {
        if !has_plane[i] {
            break; // C loop stops at the first absent plane
        }
        let s = if i == 1 || i == 2 {
            desc.log2_chroma_h
        } else {
            0
        };
        let h = ceil_rshift(height, s as u32);
        sizes[i] = linesizes[i]
            .checked_mul(h as usize)
            .ok_or(Error::OutOfRange)?;
    }
    Ok(sizes)
}

/// `av_image_get_buffer_size` — total bytes of a compact frame
/// (`FFALIGN(linesize, align)` per row; Phase 1 always calls with `align=1`).
pub fn get_buffer_size(fmt: PixelFormat, width: u32, height: u32, align: usize) -> Result<usize> {
    check_size(width, height)?;
    let linesizes = fill_linesizes(fmt, width)?;
    let mut aligned = [0usize; 4];
    for i in 0..4 {
        // FFALIGN
        aligned[i] = linesizes[i].div_ceil(align) * align;
    }
    let sizes = fill_plane_sizes(fmt, height, &aligned)?;
    let total: usize = sizes
        .iter()
        .try_fold(0usize, |acc, &s| acc.checked_add(s))
        .ok_or(Error::OutOfRange)?;
    Ok(total)
}

/// `av_image_check_size` (via `check_size2` with default max_pixels): rejects
/// empty or absurdly large frames before any allocation math.
pub fn check_size(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 || w > i32::MAX as u32 || h > i32::MAX as u32 {
        crate::log_error!(Some("imgutils"), "Picture size {w}x{h} is invalid");
        return Err(Error::OutOfRange);
    }
    Ok(())
}

/// `av_image_copy_to_buffer` — pack frame planes (each with its own
/// `src_linesize`) into one contiguous buffer, row by row, honoring `align`
/// row padding. Returns the packed buffer; the C version writes into a
/// caller buffer and returns the size, the Rust version allocates.
pub fn copy_to_buffer(
    fmt: PixelFormat,
    width: u32,
    height: u32,
    align: usize,
    planes: &[&[u8]],
    src_linesizes: &[usize; 4],
) -> Result<Vec<u8>> {
    let size = get_buffer_size(fmt, width, height, align)?;
    let desc = descriptor(fmt);
    let linesizes = fill_linesizes(fmt, width)?;
    let nb_planes = {
        let mut n = 0usize;
        for i in 0..desc.nb_components as usize {
            n = n.max(desc.comp[i].plane as usize + 1);
        }
        n
    };

    let mut out = vec![0u8; size];
    let mut dst = out.as_mut_slice();
    for i in 0..nb_planes {
        let shift = if i == 1 || i == 2 {
            desc.log2_chroma_h
        } else {
            0
        };
        let h = ceil_rshift(height, shift as u32) as usize;
        let src = planes.get(i).copied().unwrap_or(&[]);
        for j in 0..h {
            let row = linesizes[i];
            let padded = row.div_ceil(align) * align;
            if dst.len() < padded || src.len() < j * src_linesizes[i] + row {
                return Err(Error::BufferTooSmall);
            }
            dst[..row].copy_from_slice(&src[j * src_linesizes[i]..j * src_linesizes[i] + row]);
            dst = &mut dst[padded..];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sweep every format at an odd and an even width, comparing against
    /// hand-derived expectations from the C table. This is the guard against
    /// descriptor transcription errors.
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
}
