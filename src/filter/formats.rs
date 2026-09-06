//! Format negotiation primitives — port of `libavfilter/formats.{h,c}`
//! (merge/can-merge + list helpers) plus `av_find_best_pix_fmt_of_2` and its
//! scoring from `libavutil/pixdesc.c:3586-3775`, and
//! `ff_fmt_is_regular_yuv`/`ff_fmt_is_forced_full_range` from
//! `libavfilter/avfiltergraph.c:774-799`.
//!
//! ## The C trick and its Rust translation
//!
//! C's `AVFilterFormats` is a refcounted list that stores **back-pointers to
//! every slot pointing at it** (`refs[]`), so a merge (`MERGE_REF`,
//! formats.c:36-54) can retarget every holder of the losing list to the
//! survivor. Here the lists live in graph-owned arenas (`fmt_lists` etc.) and
//! links hold [`ListIdx`] handles: identity ("already merged") is index
//! equality, and "retarget every holder" is a graph sweep replacing the loser
//! index with the winner's (same O(links) cost as C's refs walk).
//!
//! Not ported: the audio axes (`AVFilterChannelLayouts`, samplerates with
//! their empty-means-any wildcard, `mergers_audio`, formats.c:426-461) —
//! audio out of scope; the alpha-mode merger (formats.c:416-423) — the
//! alpha-mode negotiation axis is out of scope; the debug `PRINT_NAME`
//! dump helpers (formats.c:350-381).

use crate::util::{
    color::{ColorRange, ColorSpace},
    pixdesc::{PixFmtDescriptor, PixFmtFlags, descriptor},
    pixfmt::PixelFormat,
};

use super::{
    graph::FilterGraph,
    link::{FormatsConfig, ListIdx},
};

// ---------------------------------------------------------------------------
// "all" lists (formats.c:603-734)
// ---------------------------------------------------------------------------

/// `ff_all_formats(AVMEDIA_TYPE_VIDEO)` == `ff_formats_pixdesc_filter(0, 0)`:
/// every pixel format with a descriptor, hwaccel ones included in C
/// (formats.c:608, gotcha: C's "all" is broader than a swscale-supported
/// set). Ours is [`PixelFormat::ALL`] — the enum's full variant list, which
/// is narrower than C's (~400 formats incl. hwaccel), the practical
/// consequence being that negotiation can never pick a format we cannot
/// represent.
pub fn all_pix_fmts() -> Vec<PixelFormat> {
    PixelFormat::ALL.to_vec()
}

/// `ff_all_color_spaces` (formats.c:698-709): `UNSPECIFIED` first, then every
/// value in enum order skipping `RESERVED` and the duplicate `UNSPECIFIED`.
/// Order matters — `pick_format` takes `color_spaces[0]` on ties.
pub fn all_color_spaces() -> Vec<ColorSpace> {
    use ColorSpace::*;
    vec![
        Unspecified, // pushed first (formats.c:701)
        Rgb,         // 0
        Bt709,       // 1
        // Unspecified (2) skipped — already pushed
        // Reserved (3) skipped
        Fcc,       // 4
        Bt470bg,   // 5
        Smpte170m, // 6
        Smpte240m, // 7
        Bt2020Ncl, // 9
    ]
}

/// `ff_all_color_ranges` (formats.c:711-716): every value in enum order.
pub fn all_color_ranges() -> Vec<ColorRange> {
    use ColorRange::*;
    vec![Unspecified, Mpeg, Jpeg]
}

/// `ff_pixfmt_is_in` (formats.c:472-477) — membership test against a static
/// list (C's sentinel-terminated arrays become slices).
pub fn pixfmt_is_in(fmt: PixelFormat, fmts: &[PixelFormat]) -> bool {
    fmts.contains(&fmt)
}

// ---------------------------------------------------------------------------
// Merge core (formats.c:36-136, 326-348)
// ---------------------------------------------------------------------------

/// `MERGE_FORMATS(a, b, check=0, empty_allowed=0)` for pixel formats:
/// intersect `b` into `a` **in place preserving a's order** (writes happen
/// only on matches with `k <= i`, so an empty intersection leaves `a`
/// untouched — formats.c:85-90). Returns false when the intersection is
/// empty (both lists unchanged, exactly like C).
fn intersect_pix_in_place(a: &mut Vec<PixelFormat>, b: &[PixelFormat]) -> bool {
    let mut k = 0;
    for i in 0..a.len() {
        if b.contains(&a[i]) {
            a[k] = a[i];
            k += 1;
        }
    }
    if k == 0 {
        return false;
    }
    a.truncate(k);
    true
}

/// The chroma/alpha-loss guard of `merge_formats_internal`
/// (formats.c:108-131), run BEFORE intersection and in can-merge mode too.
///
/// Over the full cross product: `chroma2`/`alpha2` = *each* side has a
/// format with the property ("chroma" = `nb_components > 1` — RGB counts);
/// `chroma1`/`alpha1` = some COMMON format has it. Refuse the merge when the
/// sides have a property the common formats lack — e.g. `{yuv420p, gray8}`
/// vs `{rgb24, gray8}` would collapse to gray, silently losing chroma, so a
/// conversion filter must be inserted instead (merge returns false).
fn merge_guard_pix(a: &[PixelFormat], b: &[PixelFormat]) -> bool {
    let mut alpha1 = false;
    let mut alpha2 = false;
    let mut chroma1 = false;
    let mut chroma2 = false;
    for &fa in a {
        let adesc = descriptor(fa);
        for &fb in b {
            let bdesc = descriptor(fb);
            alpha2 |= adesc.flags.contains(PixFmtFlags::ALPHA)
                && bdesc.flags.contains(PixFmtFlags::ALPHA);
            chroma2 |= adesc.nb_components > 1 && bdesc.nb_components > 1;
            if fa == fb {
                alpha1 |= adesc.flags.contains(PixFmtFlags::ALPHA);
                chroma1 |= adesc.nb_components > 1;
            }
        }
    }
    !(alpha2 && !alpha1) && !(chroma2 && !chroma1)
}

/// `can_merge_pix_fmts` (formats.c:145-149, check=1 mode): predicate subject
/// to the chroma/alpha guard — true if the lists are the same object or have
/// a common format the guard allows. Returns at the FIRST common format; it
/// is not a full intersection.
pub fn can_merge_pix_fmts(g: &FilterGraph, a: ListIdx, b: ListIdx) -> bool {
    if a == b {
        return true; // pointer identity in C (formats.c:105-106)
    }
    let (av, bv) = (&g.fmt_lists[a as usize], &g.fmt_lists[b as usize]);
    merge_guard_pix(av, bv) && av.iter().any(|f| bv.contains(f))
}

/// `merge_pix_fmts` (formats.c:163-166, check=0 mode): guard, then intersect
/// in place into `a`'s arena slot preserving `a`'s order (a = the incfg side
/// — the survivor keeps the SRC filter's preference order), then sweep every
/// link half replacing the loser index `b` with `a` (`MERGE_REF`).
///
/// Returns true on merge, false when incompatible (nothing modified).
pub fn merge_pix_fmts(g: &mut FilterGraph, a: ListIdx, b: ListIdx) -> bool {
    if a == b {
        return true;
    }
    if !can_merge_pix_fmts(g, a, b) {
        return false;
    }
    let merged = {
        let (av, bv) = two_mut(&mut g.fmt_lists, a as usize, b as usize);
        intersect_pix_in_place(av, bv)
    };
    if !merged {
        return false; // empty intersection — both lists unchanged
    }
    sweep_list_idx(g, super::graph::Axis::Formats, b, a);
    true
}

/// `can_merge_generic` for the color-space axis (formats.c:326-343).
pub fn can_merge_csp(g: &FilterGraph, a: ListIdx, b: ListIdx) -> bool {
    if a == b {
        return true;
    }
    let (av, bv) = (&g.csp_lists[a as usize], &g.csp_lists[b as usize]);
    av.iter().any(|f| bv.contains(f))
}

/// `merge_generic` for the color-space axis (formats.c:345-348): plain
/// intersection, a-order, then sweep.
pub fn merge_csp(g: &mut FilterGraph, a: ListIdx, b: ListIdx) -> bool {
    if a == b {
        return true;
    }
    let b_set = g.csp_lists[b as usize].clone();
    if !intersect_in_place(&mut g.csp_lists[a as usize], &b_set) {
        return false;
    }
    sweep_list_idx(g, super::graph::Axis::ColorSpaces, b, a);
    true
}

/// `can_merge_generic` for the color-range axis.
pub fn can_merge_rng(g: &FilterGraph, a: ListIdx, b: ListIdx) -> bool {
    if a == b {
        return true;
    }
    let (av, bv) = (&g.rng_lists[a as usize], &g.rng_lists[b as usize]);
    av.iter().any(|f| bv.contains(f))
}

/// `merge_generic` for the color-range axis.
pub fn merge_rng(g: &mut FilterGraph, a: ListIdx, b: ListIdx) -> bool {
    if a == b {
        return true;
    }
    let b_set = g.rng_lists[b as usize].clone();
    if !intersect_in_place(&mut g.rng_lists[a as usize], &b_set) {
        return false;
    }
    sweep_list_idx(g, super::graph::Axis::ColorRanges, b, a);
    true
}

/// `MERGE_FORMATS(a, b, check=0, empty_allowed=0)` generic body: intersect
/// `b` into `a` in place preserving a's order; false (both unchanged) on an
/// empty intersection.
fn intersect_in_place<T: PartialEq>(a: &mut Vec<T>, b: &[T]) -> bool {
    let mut k = 0;
    for i in 0..a.len() {
        if b.contains(&a[i]) {
            a.swap(k, i);
            k += 1;
        }
    }
    if k == 0 {
        return false;
    }
    a.truncate(k);
    true
}

/// `MERGE_REF` (formats.c:36-54): every link slot holding the loser index is
/// retargeted to the winner — the graph-wide sweep that replaces C's refs[]
/// back-pointer registry. Covers both halves (`incfg`/`outcfg`) of every
/// link on the given axis.
fn sweep_list_idx(g: &mut FilterGraph, axis: super::graph::Axis, loser: ListIdx, winner: ListIdx) {
    for link in &mut g.links {
        let halves = [&mut link.incfg, &mut link.outcfg];
        for half in halves {
            let slot = half.slot_mut(axis);
            if *slot == Some(loser) {
                *slot = Some(winner);
            }
        }
    }
}

/// Two mutable borrows of distinct indices of one arena.
fn two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(i, j);
    if i < j {
        let (lo, hi) = v.split_at_mut(j);
        (&mut lo[i], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(i);
        (&mut hi[0], &mut lo[j])
    }
}

impl FormatsConfig {
    /// The list index of one negotiation axis on this half.
    pub(crate) fn slot(&self, axis: super::graph::Axis) -> Option<ListIdx> {
        match axis {
            super::graph::Axis::Formats => self.formats,
            super::graph::Axis::ColorSpaces => self.color_spaces,
            super::graph::Axis::ColorRanges => self.color_ranges,
        }
    }

    /// `&mut` to one axis slot (set/clear).
    pub(crate) fn slot_mut(&mut self, axis: super::graph::Axis) -> &mut Option<ListIdx> {
        match axis {
            super::graph::Axis::Formats => &mut self.formats,
            super::graph::Axis::ColorSpaces => &mut self.color_spaces,
            super::graph::Axis::ColorRanges => &mut self.color_ranges,
        }
    }
}

// ---------------------------------------------------------------------------
// ff_fmt_is_regular_yuv / ff_fmt_is_forced_full_range (avfiltergraph.c:774-799)
// ---------------------------------------------------------------------------

/// `ff_fmt_is_regular_yuv` — true for ≥3-component formats without the
/// RGB/PAL/XYZ/FLOAT flags; decides whether colorspace/range negotiation
/// applies in `pick_format`. Grayscale (`nb_components < 3`) is explicitly
/// full-range by swscale convention.
///
/// Not ported: the PAL/XYZ/FLOAT flag tests (C: `avfiltergraph.c:782-783`) —
/// our descriptor table has no formats carrying them (no pal8/xyz/float
/// variants in the enum), so only the RGB flag test remains live; the
/// HWACCEL assert is unreachable for the same reason.
pub fn regular_yuv(fmt: PixelFormat) -> bool {
    let desc = descriptor(fmt);
    if desc.nb_components < 3 {
        return false; // grayscale is explicitly full-range in swscale
    }
    !desc.flags.contains(PixFmtFlags::RGB)
}

/// `ff_fmt_is_forced_full_range` — the deprecated `YUVJ*` formats are
/// hard-wired to full (JPEG) range. Our `PixelFormat` enum carries no YUVJ
/// variants, so this always returns false; ported anyway (wave 2's
/// `pick_format` branches on it).
pub fn forced_full_range(_fmt: PixelFormat) -> bool {
    // C: matches YUVJ420P | YUVJ422P | YUVJ444P | YUVJ440P | YUVJ411P.
    false
}

// ---------------------------------------------------------------------------
// av_find_best_pix_fmt_of_2 (libavutil/pixdesc.c:3586-3775)
// ---------------------------------------------------------------------------

/// `FF_LOSS_*` bitset (`pixdesc.h:397-404`) — the static loss table.
pub struct Loss;

impl Loss {
    pub const RESOLUTION: u32 = 0x0001;
    pub const DEPTH: u32 = 0x0002;
    pub const COLORSPACE: u32 = 0x0004;
    pub const ALPHA: u32 = 0x0008;
    pub const COLORQUANT: u32 = 0x0010;
    pub const CHROMA: u32 = 0x0020;
    pub const EXCESS_RESOLUTION: u32 = 0x0040;
    pub const EXCESS_DEPTH: u32 = 0x0080;
}

/// `FF_COLOR_*` classes (`pixdesc.c:3533-3537`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorType {
    Na = -1,
    Rgb = 0,
    Gray = 1,
    Yuv = 2,
    YuvJpeg = 3,
    Xyz = 4,
}

/// `get_color_type` (`pixdesc.c:3544-3566`). Not ported: the PAL→RGB branch
/// (no PAL flag/formats in the subset); the XYZ flag test is kept for shape
/// but no format carries it.
fn get_color_type(desc: &PixFmtDescriptor) -> ColorType {
    if desc.nb_components == 1 || desc.nb_components == 2 {
        return ColorType::Gray;
    }
    // C checks av_strstart(desc->name, "yuvj") — deprecated full-range YUV.
    if desc.name.starts_with("yuvj") {
        return ColorType::YuvJpeg;
    }
    if desc.flags.contains(PixFmtFlags::RGB) {
        return ColorType::Rgb;
    }
    // AV_PIX_FMT_FLAG_XYZ: not present in our descriptor table.
    if desc.nb_components == 0 {
        return ColorType::Na;
    }
    ColorType::Yuv
}

/// `get_pix_fmt_depth` (`pixdesc.c:3557-3584`): min/max component depth.
fn pix_fmt_depth(desc: &PixFmtDescriptor) -> (u8, u8) {
    let mut min = u8::MAX;
    let mut max = 0u8;
    for i in 0..desc.nb_components as usize {
        min = min.min(desc.comp[i].depth);
        max = max.max(desc.comp[i].depth);
    }
    (min, max)
}

/// `av_get_padded_bits_per_pixel` (`pixdesc.c:3425-3440`): max step per plane
/// (scaled by subsampling), summed, ×8 (no BITSTREAM formats in the subset).
fn padded_bits_per_pixel(desc: &PixFmtDescriptor) -> i32 {
    let log2_pixels = (desc.log2_chroma_w + desc.log2_chroma_h) as i32;
    let mut steps = [0i32; 4];
    for c in 0..desc.nb_components as usize {
        let comp = &desc.comp[c];
        let s = if c == 1 || c == 2 { 0 } else { log2_pixels };
        steps[comp.plane as usize] = (comp.step as i32) << s;
    }
    let bits: i32 = steps.iter().sum();
    (bits * 8) >> log2_pixels
}

/// `get_pix_fmt_score` (`pixdesc.c:3586-3726`) — THE similarity heuristic
/// behind format picking. Verbatim port of the loss scoring; see the C
/// comments for the rationale of each term.
///
/// Not ported: the HWACCEL early-outs (3600-3606 — no hw formats here); the
/// two `AV_PIX_FMT_PAL8` special cases (3621-3622, 3627, 3718-3722 — no
/// palette formats here).
fn get_pix_fmt_score(dst: PixelFormat, src: PixelFormat, loss: &mut u32, consider: u32) -> i32 {
    let src_desc = descriptor(src);
    let dst_desc = descriptor(dst);
    let mut score: i32 = i32::MAX - 1;

    *loss = 0;

    if dst == src {
        return i32::MAX;
    }

    let (_src_min, _src_max) = pix_fmt_depth(src_desc);
    let (_dst_min, _dst_max) = pix_fmt_depth(dst_desc);

    let src_color = get_color_type(src_desc);
    let dst_color = get_color_type(dst_desc);
    let nb_components = src_desc.nb_components.min(dst_desc.nb_components) as usize;

    // Depth per component (3626-3639): shallower dst loses DEPTH (big
    // penalty), deeper dst loses EXCESS_DEPTH (tiny preference for exact).
    for i in 0..nb_components {
        let depth_minus1 = (dst_desc.comp[i].depth - 1) as i32;
        let depth_delta = (src_desc.comp[i].depth - 1) as i32 - depth_minus1;
        if depth_delta > 0 && (consider & Loss::DEPTH) != 0 {
            *loss |= Loss::DEPTH;
            score -= 65536 >> depth_minus1;
        } else if depth_delta < 0 && (consider & Loss::EXCESS_DEPTH) != 0 {
            *loss |= Loss::EXCESS_DEPTH;
            score += depth_delta;
        }
    }

    // Resolution (3641-3655): dst subsampling coarser than src loses; 420 is
    // bonused over 444 when downsampling either way (decoder support).
    if (consider & Loss::RESOLUTION) != 0 {
        if dst_desc.log2_chroma_w > src_desc.log2_chroma_w {
            *loss |= Loss::RESOLUTION;
            score -= 256 << dst_desc.log2_chroma_w;
        }
        if dst_desc.log2_chroma_h > src_desc.log2_chroma_h {
            *loss |= Loss::RESOLUTION;
            score -= 256 << dst_desc.log2_chroma_h;
        }
        if dst_desc.log2_chroma_w == 1
            && src_desc.log2_chroma_w == 0
            && dst_desc.log2_chroma_h == 1
            && src_desc.log2_chroma_h == 0
        {
            score += 512;
        }
    }

    // Excess resolution (3657-3677): dst finer than src loses slightly; 420
    // bonused over 411.
    if (consider & Loss::EXCESS_RESOLUTION) != 0 {
        if dst_desc.log2_chroma_w < src_desc.log2_chroma_w {
            *loss |= Loss::EXCESS_RESOLUTION;
            score -= 1 << (src_desc.log2_chroma_w - dst_desc.log2_chroma_w);
        }
        if dst_desc.log2_chroma_h < src_desc.log2_chroma_h {
            *loss |= Loss::EXCESS_RESOLUTION;
            score -= 1 << (src_desc.log2_chroma_h - dst_desc.log2_chroma_h);
        }
        if dst_desc.log2_chroma_w == 1
            && src_desc.log2_chroma_w == 2
            && dst_desc.log2_chroma_h == 1
            && src_desc.log2_chroma_h == 2
        {
            score += 4;
        }
    }

    // Colorspace class mismatch (3679-3707): the per-class acceptance table.
    if (consider & Loss::COLORSPACE) != 0 {
        match dst_color {
            ColorType::Rgb => {
                if src_color != ColorType::Rgb && src_color != ColorType::Gray {
                    *loss |= Loss::COLORSPACE;
                }
            }
            ColorType::Gray => {
                if src_color != ColorType::Gray {
                    *loss |= Loss::COLORSPACE;
                }
            }
            ColorType::Yuv => {
                if src_color != ColorType::Yuv {
                    *loss |= Loss::COLORSPACE;
                }
            }
            ColorType::YuvJpeg => {
                if src_color != ColorType::YuvJpeg
                    && src_color != ColorType::Yuv
                    && src_color != ColorType::Gray
                {
                    *loss |= Loss::COLORSPACE;
                }
            }
            _ => {
                if src_color != dst_color {
                    *loss |= Loss::COLORSPACE;
                }
            }
        }
    }
    if (*loss & Loss::COLORSPACE) != 0 {
        let min_depth = (dst_desc.comp[0].depth - 1).min(src_desc.comp[0].depth - 1) as i32;
        score -= (nb_components as i32 * 65536) >> min_depth;
    }

    // Chroma loss: dst gray, src colored (3709-3713).
    if dst_color == ColorType::Gray
        && src_color != ColorType::Gray
        && (consider & Loss::CHROMA) != 0
    {
        *loss |= Loss::CHROMA;
        score -= 2 * 65536;
    }

    // Alpha loss (3714-3717): dst lacks alpha, src has it.
    let has_alpha = |d: &PixFmtDescriptor| d.flags.contains(PixFmtFlags::ALPHA);
    if !has_alpha(dst_desc) && has_alpha(src_desc) && (consider & Loss::ALPHA) != 0 {
        *loss |= Loss::ALPHA;
        score -= 65536;
    }

    // COLORQUANT (3718-3722): PAL8-only, not ported (no palette formats).

    score
}

/// `av_find_best_pix_fmt_of_2` (`pixdesc.c:3739-3775`): which of two
/// candidate destination formats converts from `src` least lossily.
///
/// `dst1 == None` is C's `AV_PIX_FMT_NONE` seeding of `pick_format`'s fold —
/// the first candidate is returned unscored.
pub fn find_best_pix_fmt_of_2(
    dst1: Option<PixelFormat>,
    dst2: PixelFormat,
    src: PixelFormat,
    has_alpha: bool,
) -> PixelFormat {
    let d1 = match dst1 {
        None => return dst2, // !desc1 (pixdesc.c:3748-3749)
        Some(d) => d,
    };
    let desc1 = descriptor(d1);
    let desc2 = descriptor(dst2);

    let mut loss_mask: u32 = u32::MAX; // loss_ptr == NULL in pick_format
    if !has_alpha {
        loss_mask &= !Loss::ALPHA;
    }

    let mut loss1 = 0u32;
    let mut loss2 = 0u32;
    let score1 = get_pix_fmt_score(d1, src, &mut loss1, loss_mask);
    let score2 = get_pix_fmt_score(dst2, src, &mut loss2, loss_mask);

    if score1 == score2 {
        // Tie-breaks (3760-3765): fewer padded bits, then fewer components;
        // the incumbent (dst1) wins exact ties.
        let pb1 = padded_bits_per_pixel(desc1);
        let pb2 = padded_bits_per_pixel(desc2);
        if pb2 != pb1 {
            if pb2 < pb1 { dst2 } else { d1 }
        } else if desc2.nb_components < desc1.nb_components {
            dst2
        } else {
            d1
        }
    } else if score1 < score2 {
        dst2
    } else {
        d1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::pixfmt::PixelFormat::*;

    // ---- all-list builders ------------------------------------------------

    #[test]
    fn all_csp_order_unspecified_first() {
        let cs = all_color_spaces();
        assert_eq!(cs[0], ColorSpace::Unspecified);
        assert!(!cs[..1].contains(&ColorSpace::Reserved));
        assert_eq!(cs.len(), 8);
        // no duplicates
        for i in 0..cs.len() {
            for j in i + 1..cs.len() {
                assert_ne!(cs[i], cs[j]);
            }
        }
    }

    #[test]
    fn all_ranges_in_enum_order() {
        assert_eq!(
            all_color_ranges(),
            vec![ColorRange::Unspecified, ColorRange::Mpeg, ColorRange::Jpeg]
        );
    }

    // ---- chroma/alpha guard (formats.c:108-131) ----------------------------

    /// Build a graph with two links whose halves carry the two lists.
    fn graph_with_links(
        a: Vec<PixelFormat>,
        b: Vec<PixelFormat>,
    ) -> (FilterGraph, ListIdx, ListIdx) {
        let mut g = FilterGraph::new();
        let ia = g.alloc_pix_list(a);
        let ib = g.alloc_pix_list(b);
        g.links.push(super::super::link::Link::default());
        g.links.push(super::super::link::Link::default());
        g.links[0].incfg.formats = Some(ia);
        g.links[0].outcfg.formats = Some(ib);
        (g, ia, ib)
    }

    #[test]
    fn guard_refuses_chroma_losing_merge() {
        // {rgb24, gray8} vs {yuv420p, gray8}: the only common format is gray
        // (chroma1 false) while both sides carry chroma formats (chroma2
        // true) — merging would silently drop chroma, so it must refuse and
        // leave both lists untouched.
        let (mut g, ia, ib) = graph_with_links(vec![Rgb24, Gray8], vec![Yuv420p, Gray8]);
        assert!(!can_merge_pix_fmts(&g, ia, ib));
        assert!(!merge_pix_fmts(&mut g, ia, ib));
        assert_eq!(g.fmt_lists[ia as usize], vec![Rgb24, Gray8]);
        assert_eq!(g.fmt_lists[ib as usize], vec![Yuv420p, Gray8]);
        // No sweep happened either.
        assert_eq!(g.links[0].incfg.formats, Some(ia));
        assert_eq!(g.links[0].outcfg.formats, Some(ib));
    }

    #[test]
    fn merge_intersects_in_a_order_and_sweeps() {
        // a = incfg = src's list; survivor keeps a's order.
        let (mut g, ia, ib) = graph_with_links(vec![Rgb24, Yuv420p, Gray8], vec![Yuv420p, Gray8]);
        assert!(can_merge_pix_fmts(&g, ia, ib));
        assert!(merge_pix_fmts(&mut g, ia, ib));
        // Intersection preserving a's element order.
        assert_eq!(g.fmt_lists[ia as usize], vec![Yuv420p, Gray8]);
        // MERGE_REF: every holder of b now points at a.
        assert_eq!(g.links[0].incfg.formats, Some(ia));
        assert_eq!(g.links[0].outcfg.formats, Some(ia));
    }

    #[test]
    fn merge_empty_intersection_refused_unchanged() {
        let (mut g, ia, ib) = graph_with_links(vec![Rgb24], vec![Yuv420p, Gray8]);
        // chroma2: rgb24(3 comp) vs yuv420p(3 comp) → true; no common →
        // can_merge false (no common), merge false, lists unchanged.
        assert!(!can_merge_pix_fmts(&g, ia, ib));
        assert!(!merge_pix_fmts(&mut g, ia, ib));
        assert_eq!(g.fmt_lists[ia as usize], vec![Rgb24]);
        assert_eq!(g.fmt_lists[ib as usize], vec![Yuv420p, Gray8]);
    }

    #[test]
    fn same_list_idx_is_already_merged() {
        let (g, ia, _ib) = graph_with_links(vec![Yuv420p], vec![]);
        assert!(can_merge_pix_fmts(&g, ia, ia));
    }

    #[test]
    fn generic_merge_csp_sweeps() {
        let mut g = FilterGraph::new();
        let ia = g.alloc_csp_list(vec![ColorSpace::Unspecified, ColorSpace::Bt709]);
        let ib = g.alloc_csp_list(vec![ColorSpace::Bt709, ColorSpace::Bt470bg]);
        g.links.push(super::super::link::Link::default());
        g.links[0].incfg.color_spaces = Some(ia);
        g.links[0].outcfg.color_spaces = Some(ib);
        assert!(can_merge_csp(&g, ia, ib));
        assert!(merge_csp(&mut g, ia, ib));
        assert_eq!(g.csp_lists[ia as usize], vec![ColorSpace::Bt709]);
        assert_eq!(g.links[0].incfg.color_spaces, Some(ia));
        assert_eq!(g.links[0].outcfg.color_spaces, Some(ia));
    }

    // ---- regular_yuv / forced_full_range ------------------------------------

    #[test]
    fn regular_yuv_classification() {
        assert!(regular_yuv(Yuv420p));
        assert!(regular_yuv(Yuv422p));
        assert!(regular_yuv(Yuv444p));
        assert!(regular_yuv(Yuv420p10le));
        assert!(!regular_yuv(Gray8)); // <3 components
        assert!(!regular_yuv(Rgb24)); // RGB flag
        assert!(!regular_yuv(Gbrp)); // RGB flag
        assert!(!forced_full_range(Yuv420p)); // no YUVJ variants in the enum
    }

    // ---- av_find_best_pix_fmt_of_2 ------------------------------------------

    #[test]
    fn best_of_2_seeding_none_returns_dst2() {
        assert_eq!(find_best_pix_fmt_of_2(None, Rgb24, Yuv420p, false), Rgb24);
    }

    #[test]
    fn best_of_2_ref_yuv420p_prefers_yuv420p_over_rgb24() {
        // Identity scores INT_MAX; rgb24 loses COLORSPACE big.
        assert_eq!(
            find_best_pix_fmt_of_2(Some(Rgb24), Yuv420p, Yuv420p, false),
            Yuv420p
        );
        // Fold over a whole list: [rgb24, yuv420p] with ref yuv420p →
        // yuv420p (pick_format's exact usage).
        let list = [Rgb24, Yuv420p];
        let mut best: Option<PixelFormat> = None;
        for &p in &list {
            best = Some(find_best_pix_fmt_of_2(best, p, Yuv420p, false));
        }
        assert_eq!(best, Some(Yuv420p));
    }

    #[test]
    fn best_of_2_ref_yuv420p_prefers_yuv422p_over_yuv444p() {
        // 422 keeps one subsampling axis (small excess-resolution penalty
        // -2); 444 keeps neither (-2 -2 = -4).
        let list = [Yuv444p, Yuv422p];
        let mut best: Option<PixelFormat> = None;
        for &p in &list {
            best = Some(find_best_pix_fmt_of_2(best, p, Yuv420p, false));
        }
        assert_eq!(best, Some(Yuv422p));
    }

    #[test]
    fn best_of_2_same_score_prefers_fewer_bits() {
        // rgb24 vs bgr24 are symmetric under any YUV ref — same score, same
        // padded bits, same component count → incumbent (dst1) wins.
        assert_eq!(
            find_best_pix_fmt_of_2(Some(Rgb24), Bgr24, Yuv420p, false),
            Rgb24
        );
    }

    #[test]
    fn pixfmt_is_in_membership() {
        assert!(pixfmt_is_in(Yuv420p, &[Yuv420p, Gray8]));
        assert!(!pixfmt_is_in(Rgb24, &[Yuv420p, Gray8]));
    }
}
