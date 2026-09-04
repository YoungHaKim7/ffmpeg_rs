//! `libswscale` — pixel-format conversion and scaling.
//!
//! ## Phase 1 scope (CPU kernels)
//!
//! `ScaleContext` is the `sws_getContext` analog; `scale` the `sws_scale`
//! one. Phase 1 implements:
//!
//! * **identity** — same format, same size (`libswscale/swscale_unscaled.c`'s
//!   direct-copy shortcut; the transcode loop usually avoids even this by
//!   keeping the frame as-is),
//! * **yuv420p → packed RGB** (rgb24/bgr24/rgba/bgra/argb/abgr) at the same
//!   size, BT.601 limited-range integer math — the flagship demo conversion
//!   (`libswscale/yuv2rgb.c`'s 601 tables, float precision here),
//! * **gray8 → packed RGB** (R=G=B=Y with range handling).
//!
//! Everything else fails with `Unsupported`, and the CLI reports it with
//! ffmpeg's "Impossible to convert between the formats" shape.
//!
//! ## Phase 2 (planned)
//!
//! The `libswscale/vulkan/` + `vf_scale_vulkan.c` model: a headless compute
//! context (vulkano, the `HeadlessGpu` pattern) running a scale.comp.glsl
//! port over storage images, behind this same constructor — plus real
//! resampling (bilinear/bicubic) for `-s`.

use crate::util::color::ColorRange;
use crate::util::error::{Error, Result};
use crate::util::frame::Frame;
use crate::util::pixdesc;
use crate::util::pixfmt::PixelFormat;

/// The packed-RGB outputs `scale` can emit in Phase 1.
const RGB_OUTPUTS: &[PixelFormat] = &[
    PixelFormat::Rgb24,
    PixelFormat::Bgr24,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
];

/// `sws_isSupportedInput`/`Output` rolled into one query for Phase 1.
fn supported_pair(src: PixelFormat, dst: PixelFormat) -> bool {
    src == dst
        || (src == PixelFormat::Yuv420p && RGB_OUTPUTS.contains(&dst))
        || (src == PixelFormat::Gray8 && RGB_OUTPUTS.contains(&dst))
}

/// `SwsContext` — conversion configuration (`sws_getContext`).
#[derive(Debug, Clone)]
pub struct ScaleContext {
    src: (PixelFormat, u32, u32),
    dst: (PixelFormat, u32, u32),
}

impl ScaleContext {
    /// `sws_getContext(srcW, srcH, srcFormat, dstW, dstH, dstFormat, …)`.
    /// Phase 1 additionally requires equal dimensions (no resampling yet).
    pub fn new(src: (PixelFormat, u32, u32), dst: (PixelFormat, u32, u32)) -> Result<Self> {
        if !supported_pair(src.0, dst.0) {
            return Err(Error::Unsupported(format!(
                "cannot convert {} to {}",
                src.0.name(),
                dst.0.name()
            )));
        }
        if src.1 != dst.1 || src.2 != dst.2 {
            return Err(Error::Unsupported(
                "resampling arrives with the Vulkan phase; sizes must match for now".into(),
            ));
        }
        Ok(ScaleContext { src, dst })
    }

    /// `sws_scale_frame` — convert `src` into a caller-allocated `dst`.
    /// The frames' geometry must match the context's configuration.
    pub fn scale(&self, src: &Frame, dst: &mut Frame) -> Result<()> {
        if (src.format, src.width, src.height) != self.src
            || (dst.format, dst.width, dst.height) != self.dst
        {
            return Err(Error::InvalidArgument(
                "frame geometry does not match ScaleContext".into(),
            ));
        }
        if src.format == dst.format {
            // Unscaled direct copy (swscale_unscaled.c): row-by-row memcpy
            // through the descriptor's plane layout.
            for p in 0..pixdesc::count_planes(src.format) {
                let s = src.plane(p);
                let ls = src.linesize(p);
                debug_assert_eq!(ls, dst.linesize(p));
                let d = dst.plane_mut(p);
                for (i, row) in s.chunks_exact(ls).enumerate() {
                    d[i * ls..i * ls + ls].copy_from_slice(row);
                }
            }
            return Ok(());
        }

        match src.format {
            PixelFormat::Yuv420p => self.yuv420p_to_rgb(src, dst),
            PixelFormat::Gray8 => self.gray8_to_rgb(src, dst),
            other => Err(Error::Unsupported(format!("input {other} not wired up yet"))),
        }
    }

    /// BT.601 YUV→RGB, the `yuv2rgb.c` coefficients:
    ///
    /// ```text
    /// R = 1.16438·(Y−16)              + 1.59603·(V−128)
    /// G = 1.16438·(Y−16) − 0.39176·(U−128) − 0.81297·(V−128)
    /// B = 1.16438·(Y−16) + 2.01723·(U−128)
    /// ```
    ///
    /// For full-range (JPEG) input the luma scaling collapses to
    /// `1·(Y−0)` — the same branch `sws_setColorspaceDetails` takes.
    fn yuv420p_to_rgb(&self, src: &Frame, dst: &mut Frame) -> Result<()> {
        let full = src.color_range == ColorRange::Jpeg;
        let y_scale = if full { 1.0 } else { 255.0 / 219.0 };
        let y_offset: f32 = if full { 0.0 } else { 16.0 };

        let y_plane = src.plane(0);
        let u_plane = src.plane(1);
        let v_plane = src.plane(2);
        let y_ls = src.linesize(0);
        let u_ls = src.linesize(1);
        let v_ls = src.linesize(2);

        let (r_off, g_off, b_off) = rgb_offsets(dst.format);
        let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
        let out_ls = dst.linesize(0);
        let out = dst.plane_mut(0);

        for yy in 0..src.height as usize {
            let y_row = &y_plane[yy * y_ls..];
            // LEFT chroma siting (ffmpeg's default for yuv420p): each chroma
            // sample covers the 2×2 block at its top-left.
            let u_row = &u_plane[(yy >> 1) * u_ls..];
            let v_row = &v_plane[(yy >> 1) * v_ls..];
            let out_row = &mut out[yy * out_ls..];
            for xx in 0..src.width as usize {
                let y = y_row[xx] as f32;
                let u = u_row[xx >> 1] as f32 - 128.0;
                let v = v_row[xx >> 1] as f32 - 128.0;
                let fy = (y - y_offset) * y_scale;
                let r = (fy + 1.59603 * v).round().clamp(0.0, 255.0) as u8;
                let g = (fy - 0.39176 * u - 0.81297 * v).round().clamp(0.0, 255.0) as u8;
                let b = (fy + 2.01723 * u).round().clamp(0.0, 255.0) as u8;
                let px = xx * step;
                out_row[px + r_off] = r;
                out_row[px + g_off] = g;
                out_row[px + b_off] = b;
            }
        }
        Ok(())
    }

    /// gray → packed RGB: luma replicated to all three components.
    fn gray8_to_rgb(&self, src: &Frame, dst: &mut Frame) -> Result<()> {
        let full = src.color_range == ColorRange::Jpeg;
        let y_scale = if full { 1.0 } else { 255.0 / 219.0 };
        let y_offset: f32 = if full { 0.0 } else { 16.0 };

        let g_plane = src.plane(0);
        let g_ls = src.linesize(0);
        let (r_off, g_off, b_off) = rgb_offsets(dst.format);
        let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
        let out_ls = dst.linesize(0);
        let out = dst.plane_mut(0);

        for yy in 0..src.height as usize {
            let src_row = &g_plane[yy * g_ls..];
            let out_row = &mut out[yy * out_ls..];
            for xx in 0..src.width as usize {
                let c = ((src_row[xx] as f32 - y_offset) * y_scale).round().clamp(0.0, 255.0) as u8;
                let px = xx * step;
                out_row[px + r_off] = c;
                out_row[px + g_off] = c;
                out_row[px + b_off] = c;
            }
        }
        Ok(())
    }
}

/// Component byte offsets of R/G/B inside one pixel of a packed RGB format
/// (`desc->comp[0..3].offset`).
fn rgb_offsets(fmt: PixelFormat) -> (usize, usize, usize) {
    let d = pixdesc::descriptor(fmt);
    (d.comp[0].offset as usize, d.comp[1].offset as usize, d.comp[2].offset as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::color::ColorRange;
    use crate::util::frame::Frame;

    /// 2×1 yuv420p, black then white (limited range): known RGB results.
    #[test]
    fn bt601_limited_range_endpoints() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 1).unwrap();
        src.color_range = ColorRange::Mpeg;
        src.plane_mut(0)[..2].copy_from_slice(&[16, 235]); // black, white
        src.plane_mut(1)[..1].copy_from_slice(&[128]); // neutral chroma
        src.plane_mut(2)[..1].copy_from_slice(&[128]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 1).unwrap();
        let ctx = ScaleContext::new(
            (PixelFormat::Yuv420p, 2, 1),
            (PixelFormat::Rgb24, 2, 1),
        )
        .unwrap();
        ctx.scale(&src, &mut dst).unwrap();

        let out = dst.plane(0);
        assert_eq!(&out[..3], &[0, 0, 0]); // Y=16 → black
        assert_eq!(&out[3..6], &[255, 255, 255]); // Y=235 → white
    }

    #[test]
    fn bt601_full_range_endpoints() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 1).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0)[..2].copy_from_slice(&[0, 255]);
        src.plane_mut(1)[..1].copy_from_slice(&[128]);
        src.plane_mut(2)[..1].copy_from_slice(&[128]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 1).unwrap();
        ScaleContext::new((PixelFormat::Yuv420p, 2, 1), (PixelFormat::Rgb24, 2, 1))
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
        let out = dst.plane(0);
        assert_eq!(&out[..3], &[0, 0, 0]);
        assert_eq!(&out[3..6], &[255, 255, 255]);
    }

    /// Hand-computed mid-gray + colored chroma (limited range):
    /// Y=126, U=64 (blue-ish), V=128 →
    /// fy = (126-16)·(255/219) ≈ 128.0
    /// R = 128.0, G = 128.0 − 0.39176·(−64) ≈ 153.1, B = 128.0 + 2.01723·(−64) ≈ −1.1 → 0
    #[test]
    fn bt601_colored_pixel_hand_computed() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 2).unwrap();
        src.color_range = ColorRange::Mpeg;
        for row in src.plane_mut(0).chunks_exact_mut(2) {
            row.copy_from_slice(&[126, 126]);
        }
        src.plane_mut(1)[..1].copy_from_slice(&[64]);
        src.plane_mut(2)[..1].copy_from_slice(&[128]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 2).unwrap();
        ScaleContext::new((PixelFormat::Yuv420p, 2, 2), (PixelFormat::Rgb24, 2, 2))
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
        let out = dst.plane(0);
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 128, "R");
            assert_eq!(px[1], 153, "G");
            assert_eq!(px[2], 0, "B");
        }
    }

    #[test]
    fn bgr24_swaps_component_order() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 1).unwrap();
        src.plane_mut(0)[..2].copy_from_slice(&[16, 235]);
        src.plane_mut(1)[..1].copy_from_slice(&[255]); // max U → blue push
        src.plane_mut(2)[..1].copy_from_slice(&[128]);
        let mut dst = Frame::alloc(PixelFormat::Bgr24, 2, 1).unwrap();
        ScaleContext::new((PixelFormat::Yuv420p, 2, 1), (PixelFormat::Bgr24, 2, 1))
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
        let out = dst.plane(0);
        // Pixel is stored B, G, R: first byte is blue (pushed by max U).
        assert_eq!(out[0], 255, "B first in bgr24");
    }

    #[test]
    fn identity_copies_planes() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 4, 2).unwrap();
        src.plane_mut(0).iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
        let mut dst = Frame::alloc(PixelFormat::Yuv420p, 4, 2).unwrap();
        ScaleContext::new((PixelFormat::Yuv420p, 4, 2), (PixelFormat::Yuv420p, 4, 2))
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
        assert_eq!(src.plane(0), dst.plane(0));
    }

    #[test]
    fn gray8_to_rgb_replicates() {
        let mut src = Frame::alloc(PixelFormat::Gray8, 2, 1).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0)[..2].copy_from_slice(&[0, 200]);
        let mut dst = Frame::alloc(PixelFormat::Rgba, 2, 1).unwrap();
        ScaleContext::new((PixelFormat::Gray8, 2, 1), (PixelFormat::Rgba, 2, 1))
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
        let out = dst.plane(0);
        assert_eq!(&out[..4], &[0, 0, 0, 0]); // alpha zeroed by alloc
        assert_eq!(&out[4..8], &[200, 200, 200, 0]);
    }

    #[test]
    fn unsupported_pairs_rejected() {
        assert!(ScaleContext::new(
            (PixelFormat::Yuv422p, 8, 8),
            (PixelFormat::Rgb24, 8, 8)
        )
        .is_err());
        assert!(ScaleContext::new(
            (PixelFormat::Yuv420p, 8, 8),
            (PixelFormat::Rgb24, 16, 16)
        )
        .is_err());
    }
}
