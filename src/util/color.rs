//! Colorimetry enums — port of the `AVColor*` enums in `libavutil/pixfmt.h`
//! (lines 640–810).
//!
//! Phase 1 only *carries* these values between layers (Y4M headers set
//! `color_range`/`chroma_location`, the dump prints `(pc)` for full range);
//! real color management arrives with the swscale/Vulkan phase. Discriminants
//! match C exactly because they are what stream descriptors exchange.

/// `enum AVColorRange` — MPEG (limited, 16–235) vs JPEG (full, 0–255) luma
/// excursion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ColorRange {
    #[default]
    Unspecified = 0,
    /// Narrow range: Y 16–235 (8-bit), U/V 16–240.
    Mpeg = 1,
    /// Full range: Y/U/V 0–255.
    Jpeg = 2,
}

/// `enum AVColorPrimaries` (subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ColorPrimaries {
    Reserved0 = 0,
    /// Also ITU-R BT1361 / IEC 61966-2-4 / SMPTE RP 177 Annex B.
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Reserved = 3,
    Bt470m = 4,
    /// Also ITU-R BT601-6 625 PAL/SECAM.
    Bt470bg = 5,
    /// Also ITU-R BT601-6 525 NTSC.
    Smpte170m = 6,
    Smpte240m = 7,
    Film = 8,
    Bt2020 = 9,
}

/// `enum AVColorTransferCharacteristic` (subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ColorTrc {
    Reserved0 = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Reserved = 3,
    Gamma22 = 4,
    Gamma28 = 5,
    Smpte170m = 6,
    Smpte240m = 7,
    Linear = 8,
    /// IEC 61966-2-1 (sRGB).
    Iec61966_2_1 = 13,
    Bt2020_10 = 14,
    Bt2020_12 = 15,
    Smpte2084 = 16,
    AribStdB67 = 18,
}

/// `enum AVColorSpace` — YUV matrix (subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ColorSpace {
    /// Order of coefficients is actually GBR, also IEC 61966-2-1 sRGB.
    Rgb = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Reserved = 3,
    Fcc = 4,
    /// ITU-R BT601-6 625 (PAL).
    Bt470bg = 5,
    /// ITU-R BT601-6 525 (NTSC) — functionally identical to BT470BG.
    Smpte170m = 6,
    Smpte240m = 7,
    Bt2020Ncl = 9,
}

/// `enum AVChromaLocation` — where chroma samples sit relative to luma.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ChromaLocation {
    #[default]
    Unspecified = 0,
    /// MPEG-2/4 4:2:0, H.264 default — chroma on the left column.
    Left = 1,
    /// MPEG-1 4:2:0, JPEG 4:2:0 — chroma between columns.
    Center = 2,
    TopLeft = 3,
    Top = 4,
    BottomLeft = 5,
    Bottom = 6,
}
