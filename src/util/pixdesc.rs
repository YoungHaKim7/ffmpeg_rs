//! Pixel-format descriptors — port of `libavutil/pixdesc.{h,c}`.
//!
//! A descriptor describes *memory layout only* (which plane, byte step,
//! offset, bit depth each component lives at) — not colorimetric meaning.
//! Everything that needs to know "how many bytes is a row of plane 1" walks
//! this table, exactly like the C:
//!
//! ```text
//! packed RGB24:  1 plane, comp R {plane 0, step 3, offset 0}
//!                Y plane row = width * 3 bytes
//! planar YUV420: 3 planes, log2_chroma_w/h = 1
//!                U/V planes are ceil(w/2) x ceil(h/2)
//! NV12:          2 planes; U and V share plane 1 at step 2, offsets 0/1
//! ```
//!
//! The table below is a literal transcription of the corresponding entries in
//! `pixdesc.c` (`av_pix_fmt_descriptors[]`) — component tuples are
//! `{plane, step, offset, shift, depth}` exactly as in C. The per-format
//! linesize tests in [`crate::util::imgutils`] are the safety net against
//! transcription slips.

use super::pixfmt::PixelFormat;

/// `AVComponentDescriptor` (`pixdesc.h:33`) — where one component lives.
#[derive(Clone, Copy, Debug)]
pub struct ComponentDescriptor {
    /// Which of the (up to 4) image planes the component is stored in.
    pub plane: u8,
    /// Bytes between horizontally adjacent pixels of this component
    /// (bits for bitstream formats, which we do not support).
    pub step: u8,
    /// Byte offset of the component inside `step` (e.g. U in YUYV: step 4,
    /// offset 1).
    pub offset: u8,
    /// Bit offset inside the step's first byte (only non-zero for sub-byte
    /// formats like rgb565: G sits at bit 5).
    pub shift: u8,
    /// Component bit depth (8, 10, 12, 16…).
    pub depth: u8,
}

/// `AV_PIX_FMT_FLAG_*` bitset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PixFmtFlags(pub u32);

impl PixFmtFlags {
    pub const PLANAR: PixFmtFlags = PixFmtFlags(1 << 4);
    pub const RGB: PixFmtFlags = PixFmtFlags(1 << 5);
    pub const ALPHA: PixFmtFlags = PixFmtFlags(1 << 7);
    pub const BE: PixFmtFlags = PixFmtFlags(1 << 2);

    pub const fn contains(self, other: PixFmtFlags) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn union(self, other: PixFmtFlags) -> PixFmtFlags {
        PixFmtFlags(self.0 | other.0)
    }
}

/// `AVPixFmtDescriptor` (`pixdesc.h:88`) — layout description of one format.
#[derive(Clone, Copy, Debug)]
pub struct PixFmtDescriptor {
    pub name: &'static str,
    pub nb_components: u8,
    /// Chroma subsampling: U/V are stored at
    /// `ceil(w >> log2_chroma_w) x ceil(h >> log2_chroma_h)`.
    pub log2_chroma_w: u8,
    pub log2_chroma_h: u8,
    pub flags: PixFmtFlags,
    /// Components in YUV order, or R/G/B(/A) for RGB formats.
    pub comp: [ComponentDescriptor; 4],
}

/// `av_pix_fmt_desc_get` — total over [`PixelFormat::ALL`].
pub fn descriptor(fmt: PixelFormat) -> &'static PixFmtDescriptor {
    match fmt {
        // ---- planar YUV ------------------------------------------------
        PixelFormat::Yuv420p => &DESC_YUV420P,
        PixelFormat::Yuv422p => &DESC_YUV422P,
        PixelFormat::Yuv444p => &DESC_YUV444P,
        PixelFormat::Yuv420p10le => &DESC_YUV420P10LE,
        PixelFormat::Yuv422p10le => &DESC_YUV422P10LE,
        PixelFormat::Yuv444p10le => &DESC_YUV444P10LE,
        PixelFormat::Yuv420p16le => &DESC_YUV420P16LE,
        PixelFormat::Yuv444p16le => &DESC_YUV444P16LE,
        // ---- bi-planar / packed YUV ------------------------------------
        PixelFormat::Nv12 => &DESC_NV12,
        PixelFormat::Nv21 => &DESC_NV21,
        PixelFormat::Yuyv422 => &DESC_YUYV422,
        PixelFormat::Uyvy422 => &DESC_UYVY422,
        // ---- gray ------------------------------------------------------
        PixelFormat::Gray8 => &DESC_GRAY8,
        PixelFormat::Gray16le => &DESC_GRAY16LE,
        // ---- packed RGB ------------------------------------------------
        PixelFormat::Rgb24 => &DESC_RGB24,
        PixelFormat::Bgr24 => &DESC_BGR24,
        PixelFormat::Rgba => &DESC_RGBA,
        PixelFormat::Bgra => &DESC_BGRA,
        PixelFormat::Argb => &DESC_ARGB,
        PixelFormat::Abgr => &DESC_ABGR,
        PixelFormat::Rgb565le => &DESC_RGB565LE,
        // ---- planar RGB ------------------------------------------------
        PixelFormat::Gbrp => &DESC_GBRP,
        PixelFormat::Gbrap => &DESC_GBRAP,
    }
}

// Component shorthand: (plane, step, offset, shift, depth).
const fn c(plane: u8, step: u8, offset: u8, shift: u8, depth: u8) -> ComponentDescriptor {
    ComponentDescriptor { plane, step, offset, shift, depth }
}

const DESC_YUV420P: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv420p",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 1,
    comp: [c(0, 1, 0, 0, 8), c(1, 1, 0, 0, 8), c(2, 1, 0, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV422P: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv422p",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 0,
    comp: [c(0, 1, 0, 0, 8), c(1, 1, 0, 0, 8), c(2, 1, 0, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV444P: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv444p",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 1, 0, 0, 8), c(1, 1, 0, 0, 8), c(2, 1, 0, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV420P10LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv420p10le",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 1,
    comp: [c(0, 2, 0, 0, 10), c(1, 2, 0, 0, 10), c(2, 2, 0, 0, 10), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV422P10LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv422p10le",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 0,
    comp: [c(0, 2, 0, 0, 10), c(1, 2, 0, 0, 10), c(2, 2, 0, 0, 10), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV444P10LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv444p10le",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 2, 0, 0, 10), c(1, 2, 0, 0, 10), c(2, 2, 0, 0, 10), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV420P16LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv420p16le",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 1,
    comp: [c(0, 2, 0, 0, 16), c(1, 2, 0, 0, 16), c(2, 2, 0, 0, 16), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUV444P16LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuv444p16le",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 2, 0, 0, 16), c(1, 2, 0, 0, 16), c(2, 2, 0, 0, 16), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_NV12: PixFmtDescriptor = PixFmtDescriptor {
    name: "nv12",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 1,
    // U and V share plane 1, interleaved at 2-byte steps.
    comp: [c(0, 1, 0, 0, 8), c(1, 2, 0, 0, 8), c(1, 2, 1, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_NV21: PixFmtDescriptor = PixFmtDescriptor {
    name: "nv21",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 1,
    comp: [c(0, 1, 0, 0, 8), c(1, 2, 1, 0, 8), c(1, 2, 0, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR,
};

const DESC_YUYV422: PixFmtDescriptor = PixFmtDescriptor {
    name: "yuyv422",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 0,
    comp: [c(0, 2, 0, 0, 8), c(0, 4, 1, 0, 8), c(0, 4, 3, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags(0),
};

const DESC_UYVY422: PixFmtDescriptor = PixFmtDescriptor {
    name: "uyvy422",
    nb_components: 3,
    log2_chroma_w: 1,
    log2_chroma_h: 0,
    comp: [c(0, 2, 1, 0, 8), c(0, 4, 0, 0, 8), c(0, 4, 2, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags(0),
};

const DESC_GRAY8: PixFmtDescriptor = PixFmtDescriptor {
    name: "gray",
    nb_components: 1,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 1, 0, 0, 8), c(0, 0, 0, 0, 0), c(0, 0, 0, 0, 0), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags(0),
};

const DESC_GRAY16LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "gray16le",
    nb_components: 1,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 2, 0, 0, 16), c(0, 0, 0, 0, 0), c(0, 0, 0, 0, 0), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags(0),
};

const DESC_RGB24: PixFmtDescriptor = PixFmtDescriptor {
    name: "rgb24",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 3, 0, 0, 8), c(0, 3, 1, 0, 8), c(0, 3, 2, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::RGB,
};

const DESC_BGR24: PixFmtDescriptor = PixFmtDescriptor {
    name: "bgr24",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [c(0, 3, 2, 0, 8), c(0, 3, 1, 0, 8), c(0, 3, 0, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::RGB,
};

const DESC_RGBA: PixFmtDescriptor = PixFmtDescriptor {
    name: "rgba",
    nb_components: 4,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [
        c(0, 4, 0, 0, 8),
        c(0, 4, 1, 0, 8),
        c(0, 4, 2, 0, 8),
        c(0, 4, 3, 0, 8),
    ],
    flags: PixFmtFlags::RGB.union(PixFmtFlags::ALPHA),
};

const DESC_BGRA: PixFmtDescriptor = PixFmtDescriptor {
    name: "bgra",
    nb_components: 4,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [
        c(0, 4, 2, 0, 8),
        c(0, 4, 1, 0, 8),
        c(0, 4, 0, 0, 8),
        c(0, 4, 3, 0, 8),
    ],
    flags: PixFmtFlags::RGB.union(PixFmtFlags::ALPHA),
};

const DESC_ARGB: PixFmtDescriptor = PixFmtDescriptor {
    name: "argb",
    nb_components: 4,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [
        c(0, 4, 1, 0, 8),
        c(0, 4, 2, 0, 8),
        c(0, 4, 3, 0, 8),
        c(0, 4, 0, 0, 8),
    ],
    flags: PixFmtFlags::RGB.union(PixFmtFlags::ALPHA),
};

const DESC_ABGR: PixFmtDescriptor = PixFmtDescriptor {
    name: "abgr",
    nb_components: 4,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [
        c(0, 4, 3, 0, 8),
        c(0, 4, 2, 0, 8),
        c(0, 4, 1, 0, 8),
        c(0, 4, 0, 0, 8),
    ],
    flags: PixFmtFlags::RGB.union(PixFmtFlags::ALPHA),
};

const DESC_RGB565LE: PixFmtDescriptor = PixFmtDescriptor {
    name: "rgb565le",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    // Sub-byte layout: R at bit 3 of byte 1 (5 bits), G at bit 5 of byte 0
    // (6 bits), B at bit 0 of byte 0 (5 bits).
    comp: [c(0, 2, 1, 3, 5), c(0, 2, 0, 5, 6), c(0, 2, 0, 0, 5), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::RGB,
};

const DESC_GBRP: PixFmtDescriptor = PixFmtDescriptor {
    name: "gbrp",
    nb_components: 3,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    // Memory plane order is G, B, R — components name planes 2/0/1.
    comp: [c(2, 1, 0, 0, 8), c(0, 1, 0, 0, 8), c(1, 1, 0, 0, 8), c(0, 0, 0, 0, 0)],
    flags: PixFmtFlags::PLANAR.union(PixFmtFlags::RGB),
};

const DESC_GBRAP: PixFmtDescriptor = PixFmtDescriptor {
    name: "gbrap",
    nb_components: 4,
    log2_chroma_w: 0,
    log2_chroma_h: 0,
    comp: [
        c(2, 1, 0, 0, 8),
        c(0, 1, 0, 0, 8),
        c(1, 1, 0, 0, 8),
        c(3, 1, 0, 0, 8),
    ],
    flags: PixFmtFlags::PLANAR.union(PixFmtFlags::RGB).union(PixFmtFlags::ALPHA),
};

/// `av_pix_fmt_count_planes` — highest plane index in use, +1.
pub fn count_planes(fmt: PixelFormat) -> usize {
    let desc = descriptor(fmt);
    let mut max_plane = 0;
    for i in 0..desc.nb_components as usize {
        max_plane = max_plane.max(desc.comp[i].plane as usize);
    }
    max_plane + 1
}

/// `av_get_bits_per_pixel` — average bits per *pixel* (not per sample):
/// luma components are weighted by the chroma subsampling factor, so
/// yuv420p is 12, yuyv422 is 16.
pub fn bits_per_pixel(fmt: PixelFormat) -> usize {
    let desc = descriptor(fmt);
    let log2_pixels = (desc.log2_chroma_w + desc.log2_chroma_h) as usize;
    let mut bits = 0usize;
    for c in 0..desc.nb_components as usize {
        // Chroma components are shared between 2^log2_pixels pixels.
        let s = if c == 1 || c == 2 { 0 } else { log2_pixels };
        bits += (desc.comp[c].depth as usize) << s;
    }
    bits >> log2_pixels
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(descriptor(PixelFormat::Yuv420p).flags.contains(PixFmtFlags::PLANAR));
        assert!(!descriptor(PixelFormat::Yuv420p).flags.contains(PixFmtFlags::RGB));
        assert!(descriptor(PixelFormat::Gbrp).flags.contains(PixFmtFlags::PLANAR));
        assert!(descriptor(PixelFormat::Gbrp).flags.contains(PixFmtFlags::RGB));
        assert!(descriptor(PixelFormat::Rgba).flags.contains(PixFmtFlags::ALPHA));
        assert!(!descriptor(PixelFormat::Yuyv422).flags.contains(PixFmtFlags::PLANAR));
    }
}
