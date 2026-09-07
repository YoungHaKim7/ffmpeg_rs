//! Pixel formats — port of `libavutil/pixfmt.h` (`enum AVPixelFormat` subset).
//!
//! FFmpeg enumerates ~400 pixel formats; this port carries the ~24 that the
//! Phase 1 pipeline (rawvideo + Y4M + basic swscale) exercises. The variants
//! keep FFmpeg's enum names upper-camel-cased (`AV_PIX_FMT_YUV420P` →
//! [`PixelFormat::Yuv420p`]) and `name()` returns the exact C string
//! (`"yuv420p"`, `"gray"`…) so CLI flags and file dumps interoperate with
//! real ffmpeg byte-for-byte.
//!
//! `AV_PIX_FMT_NONE` (−1) is not a variant: Rust models "no format yet" as
//! `Option<PixelFormat>` at the use sites.
//!
//! Deliberately out of subset: big-endian variants (except none — Y4M's mono
//! tables were skipped accordingly), palette (`pal8`), bitstream (`monow/b`),
//! Bayer, float/XYZ, hardware formats (`vulkan` arrives with Phase 2 as a
//! *frame* concept, not a swizzle-able pixel layout).

/// `enum AVPixelFormat` — supported subset, LE formats only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PixelFormat {
    /// Planar YUV 4:2:0, 8-bit — THE canonical video format.
    Yuv420p,
    /// Planar YUV 4:2:2, 8-bit.
    Yuv422p,
    /// Planar YUV 4:4:4, 8-bit.
    Yuv444p,
    /// Planar YUV 4:2:0, 10-bit little-endian.
    Yuv420p10le,
    /// Planar YUV 4:2:2, 10-bit little-endian.
    Yuv422p10le,
    /// Planar YUV 4:4:4, 10-bit little-endian.
    Yuv444p10le,
    /// Planar YUV 4:2:0, 16-bit little-endian.
    Yuv420p16le,
    /// Planar YUV 4:4:4, 16-bit little-endian.
    Yuv444p16le,
    /// Bi-planar YUV 4:2:0 (Y plane + interleaved UV).
    Nv12,
    /// Bi-planar YUV 4:2:0 (Y plane + interleaved VU).
    Nv21,
    /// Packed YUV 4:2:2 (Y0 U Y1 V).
    Yuyv422,
    /// Packed YUV 4:2:2 (U Y0 V Y1).
    Uyvy422,
    /// Grayscale, 8-bit. Canonical name is "gray".
    Gray8,
    /// Grayscale, 16-bit little-endian.
    Gray16le,
    /// Packed RGB 8:8:8 (byte order R, G, B).
    Rgb24,
    /// Packed RGB 8:8:8 (byte order B, G, R).
    Bgr24,
    /// Packed RGBA 8:8:8:8.
    Rgba,
    /// Packed BGRA 8:8:8:8.
    Bgra,
    /// Packed ARGB 8:8:8:8.
    Argb,
    /// Packed ABGR 8:8:8:8.
    Abgr,
    /// Packed RGB 5:6:5, 16-bit little-endian.
    Rgb565le,
    /// Planar GBR 4:4:4 8-bit (plane order in memory: G, B, R).
    Gbrp,
    /// Planar GBR 4:4:4 8-bit plus alpha plane.
    Gbrap,
}

impl PixelFormat {
    /// Every supported format, in enum order (`av_pix_fmt_desc_next` analog).
    pub const ALL: &'static [PixelFormat] = &[
        PixelFormat::Yuv420p,
        PixelFormat::Yuv422p,
        PixelFormat::Yuv444p,
        PixelFormat::Yuv420p10le,
        PixelFormat::Yuv422p10le,
        PixelFormat::Yuv444p10le,
        PixelFormat::Yuv420p16le,
        PixelFormat::Yuv444p16le,
        PixelFormat::Nv12,
        PixelFormat::Nv21,
        PixelFormat::Yuyv422,
        PixelFormat::Uyvy422,
        PixelFormat::Gray8,
        PixelFormat::Gray16le,
        PixelFormat::Rgb24,
        PixelFormat::Bgr24,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
        PixelFormat::Rgb565le,
        PixelFormat::Gbrp,
        PixelFormat::Gbrap,
    ];

    /// Canonical C name (`desc->name`) — what `-pix_fmt` accepts and
    /// `av_dump_format` prints.
    pub const fn name(self) -> &'static str {
        match self {
            PixelFormat::Yuv420p => "yuv420p",
            PixelFormat::Yuv422p => "yuv422p",
            PixelFormat::Yuv444p => "yuv444p",
            PixelFormat::Yuv420p10le => "yuv420p10le",
            PixelFormat::Yuv422p10le => "yuv422p10le",
            PixelFormat::Yuv444p10le => "yuv444p10le",
            PixelFormat::Yuv420p16le => "yuv420p16le",
            PixelFormat::Yuv444p16le => "yuv444p16le",
            PixelFormat::Nv12 => "nv12",
            PixelFormat::Nv21 => "nv21",
            PixelFormat::Yuyv422 => "yuyv422",
            PixelFormat::Uyvy422 => "uyvy422",
            PixelFormat::Gray8 => "gray",
            PixelFormat::Gray16le => "gray16le",
            PixelFormat::Rgb24 => "rgb24",
            PixelFormat::Bgr24 => "bgr24",
            PixelFormat::Rgba => "rgba",
            PixelFormat::Bgra => "bgra",
            PixelFormat::Argb => "argb",
            PixelFormat::Abgr => "abgr",
            PixelFormat::Rgb565le => "rgb565le",
            PixelFormat::Gbrp => "gbrp",
            PixelFormat::Gbrap => "gbrap",
        }
    }

    /// `av_get_pix_fmt(name)` — canonical name or known alias; `None` maps to
    /// `AV_PIX_FMT_NONE`. Aliases come from the `alias` field in `pixdesc.c`.
    pub fn from_name(name: &str) -> Option<PixelFormat> {
        Some(match name {
            "yuv420p" => PixelFormat::Yuv420p,
            "yuv422p" => PixelFormat::Yuv422p,
            "yuv444p" => PixelFormat::Yuv444p,
            "yuv420p10le" | "yuv420p10" => PixelFormat::Yuv420p10le,
            "yuv422p10le" | "yuv422p10" => PixelFormat::Yuv422p10le,
            "yuv444p10le" | "yuv444p10" => PixelFormat::Yuv444p10le,
            "yuv420p16le" | "yuv420p16" => PixelFormat::Yuv420p16le,
            "yuv444p16le" | "yuv444p16" => PixelFormat::Yuv444p16le,
            "nv12" => PixelFormat::Nv12,
            "nv21" => PixelFormat::Nv21,
            "yuyv422" | "yuyv422p" => PixelFormat::Yuyv422,
            "uyvy422" | "uyvy422p" => PixelFormat::Uyvy422,
            "gray" | "gray8" | "y8" => PixelFormat::Gray8,
            "gray16le" | "y16le" => PixelFormat::Gray16le,
            "rgb24" => PixelFormat::Rgb24,
            "bgr24" => PixelFormat::Bgr24,
            "rgba" => PixelFormat::Rgba,
            "bgra" => PixelFormat::Bgra,
            "argb" => PixelFormat::Argb,
            "abgr" => PixelFormat::Abgr,
            "rgb565le" => PixelFormat::Rgb565le,
            "gbrp" | "gbrp8" => PixelFormat::Gbrp,
            "gbrap" => PixelFormat::Gbrap,
            _ => return None,
        })
    }
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
