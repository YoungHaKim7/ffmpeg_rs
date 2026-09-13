//! `hflip` and `vflip` video filters — port of `libavfilter/vf_hflip.c`
//! (157 lines), `vf_hflip_init.h` (113 lines), `hflip.h` (39 lines) and
//! `vf_vflip.c` (138 lines). Two filters, one file — they share nothing in C
//! but the theme (and C's own `vf_flip.c` was renamed to `vf_vflip.c`; there
//! is no combined file upstream anymore).
//!
//! * **hflip** mirrors every row: `config_props` (INPUT pad,
//!   vf_hflip.c:60-78) computes the per-plane geometry +
//!   `max_step`,`ff_hflip_init` (vf_hflip_init.h:90-110) picks one of six
//!   `flip_line` functions by `step`, and `filter_slice` (84-112) walks
//!   rows copying pixels from the row END to its start:
//!   `out(j) = in(width-1-j)` (the C inner loop reads `src[-j]` off a
//!   pointer parked on the LAST pixel, vf_hflip_init.h:33-37).
//! * **vflip** mirrors the row order. C does it with NEGATIVE linesizes —
//!   `get_video_buffer` (vf_vflip.c:48-69) pre-flips the allocation so
//!   `filter_frame` (100-119) is the same one-line pointer surgery, zero
//!   copy. The port FORBIDS negative linesizes (frame.rs module doc), so
//!   vflip becomes a real copy: a fresh frame with the rows reversed. Same
//!   pixels out, different mechanism — the one semantic that changes is
//!   that the port's vflip output is always a private buffer (C may forward
//!   a buffer whose stride is negative, which downstream C code accepts).

use crate::util::{error::Result, frame::Frame, pixdesc, pixfmt::PixelFormat};

use super::vf_crop::{ceil_rshift, fill_max_pixsteps};
use super::{
    filter::{FilterDef, FilterFlags, FilterImpl, PadDef, PadRef, filter_frame},
    graph::FilterGraph,
    link::NodeId,
};

// ---------------------------------------------------------------------------
// hflip — private context (hflip.h:27-34)
// ---------------------------------------------------------------------------

/// `FlipContext` for hflip (hflip.h:27-34). `void (*flip_line[4])(...)`
/// collapses into the generic `step`-byte loop — every C variant
/// (`hflip_byte/short/dword/b24/b48/qword_c`, vf_hflip_init.h:33-88) is a
/// byte-for-byte copy of `step` bytes per pixel (the `AV_RB24/RB48` dance
/// included), so one loop reproduces all six; `ff_hflip_init`'s
/// `AVERROR_BUG` default arm has no reachable format in the port's universe
/// (steps there are 1/2/3/4 ⊂ {1,2,3,4,6,8}).
pub struct HFlipContext {
    /// `max_step[4]` — max pixel step per plane in bytes.
    pub max_step: [usize; 4],
    /// `bayer_plus1` — 1 for every non-Bayer format; the port's universe
    /// has no `AV_PIX_FMT_FLAG_BAYER` format, so this is always 1 (kept for
    /// the C shape; the `planewidth / bayer_plus1` sites still apply it).
    pub bayer_plus1: usize,
    /// `planewidth[4]` — pixel width of each plane (chroma-shifted).
    pub planewidth: [usize; 4],
    /// `planeheight[4]` — row count of each plane.
    pub planeheight: [usize; 4],
}

impl Default for HFlipContext {
    fn default() -> Self {
        HFlipContext {
            max_step: [0; 4],
            bayer_plus1: 1,
            planewidth: [0; 4],
            planeheight: [0; 4],
        }
    }
}

// ---------------------------------------------------------------------------
// vflip — private context (vf_vflip.c:32-35)
// ---------------------------------------------------------------------------

/// `FlipContext` for vflip (vf_vflip.c:32-35): `vsub` (stored by
/// `config_input`, 42) and `bayer` (43). In the port `vsub` is unused by
/// the copy path — the per-plane `rows` already carry the chroma height —
/// and the Bayer path (`flip_bayer`, 71-98) has no format to fire on; both
/// fields survive for C shape only.
pub struct VFlipContext {
    pub vsub: u32,
    pub bayer: bool,
}

impl Default for VFlipContext {
    fn default() -> Self {
        VFlipContext {
            vsub: 0,
            bayer: false,
        }
    }
}

// ---------------------------------------------------------------------------
// hflip — FilterImpl
// ---------------------------------------------------------------------------

impl FilterImpl for HFlipContext {
    // NOTE: vf_hflip has NO options (no AVOption table, vf_hflip.c:149-157)
    // — the trait's default `init` leaves every entry a leftover, and
    // `init_filter`'s check yields C's own unknown-option error shape.

    /// `query_formats` (vf_hflip.c:40-58): reject HWACCEL, bitstream and
    /// the PACKED 4:2:2 family (`log2_chroma_w != log2_chroma_h` while all
    /// components share plane 0 — a packed row cannot be h-flipped at the
    /// chroma grid, vf_hflip.c:49-53). The port's universe keeps everything
    /// except YUYV422/UYVY422; planar 4:2:2 survives (each plane flips
    /// independently).
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let list: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&f| {
                let d = pixdesc::descriptor(f);
                !(d.log2_chroma_w != d.log2_chroma_h && d.comp[0].plane == d.comp[1].plane)
            })
            .collect();
        let list = g.alloc_pix_list(list);
        g.set_common_formats(node, list)?;
        Ok(())
    }

    /// `config_props` on the INPUT pad (vf_hflip.c:60-78, wired at 140-147).
    fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
        let PadRef::In(0) = pad else {
            return Ok(());
        };
        let inlink = g.inlink(node, 0);
        let l = &g.links[inlink.0];
        let fmt = l
            .format
            .expect("formats picked before config_props (query_formats round)");
        let pix_desc = pixdesc::descriptor(fmt);
        let hsub = pix_desc.log2_chroma_w as u32;
        let vsub = pix_desc.log2_chroma_h as u32;

        // (68-73) the plane geometry. planewidth[0]=[3]=w, [1]=[2]=ceil(w>>hsub);
        // planeheight similarly. bayer_plus1 = !!(BAYER)+1 == 1 here.
        self.max_step = fill_max_pixsteps(pix_desc);
        self.planewidth[0] = l.w as usize;
        self.planewidth[3] = l.w as usize;
        self.planewidth[1] = ceil_rshift(l.w, hsub) as usize;
        self.planewidth[2] = self.planewidth[1];
        self.planeheight[0] = l.h as usize;
        self.planeheight[3] = l.h as usize;
        self.planeheight[1] = ceil_rshift(l.h, vsub) as usize;
        self.planeheight[2] = self.planeheight[1];
        // (73) AV_PIX_FMT_FLAG_BAYER does not exist in the port → 1.
        self.bayer_plus1 = 1;

        // (75-77) ff_hflip_init: the per-plane step dispatch — the generic
        // loop stands in for all six flip_line variants; the AVERROR_BUG
        // default arm is unreachable over the port's formats.
        Ok(())
    }

    /// `filter_frame` (vf_hflip.c:114-138).
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        _pad: usize,
        frame: Frame,
    ) -> Result<()> {
        let outlink = g.outlink(node, 0);
        // (121) a fresh buffer at the output link's geometry (hflip changes
        // nothing — the graph defaults keep the input's).
        let fmt = g.links[outlink.0]
            .format
            .expect("formats picked before config_props");
        let mut out = Frame::alloc(fmt, g.links[outlink.0].w, g.links[outlink.0].h)?;

        // (126) copy props. (128-130) the PAL palette copy — no PAL formats
        // in the port's universe, dropped.
        out.copy_props(&frame);

        // (84-112) filter_slice, one slice (start=0, end=height): for every
        // EXISTING plane (C's `data[plane] && linesize[plane]` walk), flip
        // `width` pixels of `step` bytes per row: out[j] = in[width-1-j].
        for p in 0..frame.planes.len() {
            let width = self.planewidth[p] / self.bayer_plus1;
            let step = self.max_step[p] * self.bayer_plus1;
            let ils = frame.planes[p].linesize;
            let ols = out.planes[p].linesize;
            for row in 0..self.planeheight[p] {
                let src = &frame.plane(p)[row * ils..(row + 1) * ils];
                let dst = &mut out.plane_mut(p)[row * ols..(row + 1) * ols];
                for j in 0..width {
                    dst[j * step..(j + 1) * step]
                        .copy_from_slice(&src[(width - 1 - j) * step..(width - j) * step]);
                }
            }
        }

        filter_frame(g, outlink, out)
    }
}

// ---------------------------------------------------------------------------
// vflip — FilterImpl
// ---------------------------------------------------------------------------

impl FilterImpl for VFlipContext {
    /// `config_input` (vf_vflip.c:37-46) on the INPUT pad (wired at
    /// 120-128): store `vsub` and `bayer` — both unused by the port's copy
    /// path (see the context doc).
    fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
        let PadRef::In(0) = pad else {
            return Ok(());
        };
        let inlink = g.inlink(node, 0);
        let fmt = g.links[inlink.0]
            .format
            .expect("formats picked before config_props (query_formats round)");
        let desc = pixdesc::descriptor(fmt);
        self.vsub = desc.log2_chroma_h as u32;
        self.bayer = false; // !!(AV_PIX_FMT_FLAG_BAYER): no bayer formats
        Ok(())
    }

    /// `filter_frame` (vf_vflip.c:100-119). C re-points `data[i]` to the
    /// last row and negates `linesize[i]` (108-116) — zero copy, but the
    /// port's Plane invariant forbids negative strides, so this is a REAL
    /// copy with the rows reversed (the port-visible difference: the output
    /// is always a private buffer). The Bayer branch (105-106, 71-98) has
    /// no format to fire on.
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        _pad: usize,
        frame: Frame,
    ) -> Result<()> {
        let outlink = g.outlink(node, 0);
        let fmt = g.links[outlink.0]
            .format
            .expect("formats picked before config_props");
        let mut out = Frame::alloc(fmt, g.links[outlink.0].w, g.links[outlink.0].h)?;

        // C forwards the SAME frame (metadata rides along for free); the
        // port re-buffers, so copy the props explicitly.
        out.copy_props(&frame);

        for p in 0..frame.planes.len() {
            let rows = frame.planes[p].rows;
            let ls = frame.planes[p].linesize;
            let ols = out.planes[p].linesize;
            for i in 0..rows {
                out.plane_mut(p)[i * ols..(i + 1) * ols]
                    .copy_from_slice(&frame.plane(p)[(rows - 1 - i) * ls..(rows - i) * ls]);
            }
        }

        filter_frame(g, outlink, out)
    }
}

// ---------------------------------------------------------------------------
// Filter descriptors (vf_hflip.c:140-157, vf_vflip.c:120-138)
// ---------------------------------------------------------------------------

/// The shared single video pad.
static DEFAULT_PAD: PadDef = PadDef {
    name: "default",
    needs_writable: false,
};

/// `ff_vf_hflip` (vf_hflip.c:149-157): "Horizontally flip the input video."
///
/// * flags: hflip is NOT in C's `ff_filter_frame` validation skip list
///   (avfilter.c:1075-1082) → `FilterFlags(0)`. C's
///   `AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC | AVFILTER_FLAG_SLICE_THREADS`
///   (152) are not modeled (no timeline / threading in the port).
/// * shorthand: EMPTY — vf_hflip has no AVOption table, so every option
///   needs an explicit key (and then fails as unknown, like C).
pub static HFLIP_DEF: FilterDef = FilterDef {
    name: "hflip",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags(0),
    shorthand: &[],
    make: || Box::new(HFlipContext::default()),
};

/// `ff_vf_vflip` (vf_vflip.c:130-138): "Flip the input video vertically."
/// Same shape as [`HFLIP_DEF`]: no options, no validation-skip flag (C's
/// `AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC` not modeled). vflip defines NO
/// `query_formats` in C — the default all-lists query applies (the trait
/// default + the engine's `default_query_formats`).
pub static VFLIP_DEF: FilterDef = FilterDef {
    name: "vflip",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags(0),
    shorthand: &[],
    make: || Box::new(VFlipContext::default()),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::filter_def;
    use crate::filter::link::LinkId;
    use crate::util::error::Error;
    use crate::util::rational::Rational;

    /// buffer(pix_fmt WxH, tb 1/25) → filter(args) → buffersink.
    fn chain(
        filter: &str,
        fmt: PixelFormat,
        size: (u32, u32),
    ) -> (FilterGraph, NodeId, NodeId, LinkId, LinkId) {
        let mut g = FilterGraph::new();
        let src = g
            .create_filter(
                "buffer",
                &format!(
                    "video_size={}x{}:pix_fmt={}:time_base=1/25:sar=1/1",
                    size.0,
                    size.1,
                    fmt.name()
                ),
            )
            .expect("buffer init");
        let f = g.create_filter(filter, "").expect("filter init");
        let sink = g.create_filter("buffersink", "").unwrap();
        let lin = g.link(src, 0, f, 0).unwrap();
        let lout = g.link(f, 0, sink, 0).unwrap();
        g.config().expect("graph config");
        (g, src, sink, lin, lout)
    }

    /// The shared pixel generator (same as vf_crop's tests).
    fn px(plane: usize, row: usize, col: usize, byte: usize) -> u8 {
        (row.wrapping_mul(31) + col.wrapping_mul(7) + plane * 13 + byte * 101 + 1) as u8
    }

    fn frame_at(fmt: PixelFormat, w: u32, h: u32, pts: i64) -> Frame {
        let mut f = Frame::alloc(fmt, w, h).unwrap();
        f.pts = pts;
        f.duration = 1;
        f.time_base = Rational::new(1, 25);
        let desc = pixdesc::descriptor(fmt);
        let steps = fill_max_pixsteps(desc);
        for p in 0..f.planes.len() {
            let shift = if p == 1 || p == 2 {
                desc.log2_chroma_w as u32
            } else {
                0
            };
            let cols = ceil_rshift(w, shift) as usize;
            let ls = f.planes[p].linesize;
            for r in 0..f.planes[p].rows {
                for c in 0..cols {
                    for b in 0..steps[p] {
                        f.planes[p].data_mut()[r * ls + c * steps[p] + b] = px(p, r, c, b);
                    }
                }
            }
        }
        f
    }

    fn run(g: &mut FilterGraph, src: NodeId, sink: NodeId, frames: Vec<Frame>) -> Vec<Frame> {
        for f in frames {
            g.add_frame(src, &f).unwrap();
        }
        g.close_source(src).unwrap();
        let mut out = Vec::new();
        while let Ok(f) = g.get_frame(sink) {
            out.push(f);
        }
        out
    }

    /// Per-plane pixel dims (width, height) of a frame.
    fn plane_dims(fmt: PixelFormat, w: u32, h: u32, p: usize) -> (usize, usize) {
        let d = pixdesc::descriptor(fmt);
        let hs = if p == 1 || p == 2 {
            d.log2_chroma_w as u32
        } else {
            0
        };
        let vs = if p == 1 || p == 2 {
            d.log2_chroma_h as u32
        } else {
            0
        };
        (ceil_rshift(w, hs) as usize, ceil_rshift(h, vs) as usize)
    }

    // ---- hflip ----------------------------------------------------------------

    #[test]
    fn hflip_per_step_and_chroma() {
        // Every step class the port's universe has: gray8 (1), gray16le (2),
        // rgb24 (3 — the AV_RB24 variant), rgba (4); yuv420p exercises the
        // half-width chroma; gbrap the 4-plane walk.
        for fmt in [
            PixelFormat::Gray8,
            PixelFormat::Gray16le,
            PixelFormat::Rgb24,
            PixelFormat::Rgba,
            PixelFormat::Yuv420p,
            PixelFormat::Gbrap,
            PixelFormat::Nv12,
        ] {
            let (w, h) = (7u32, 5u32); // odd: chroma ceil arithmetic
            let src = frame_at(fmt, w, h, 11);
            let (mut g, s, k, _i, lout) = chain("hflip", fmt, (w, h));
            assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (w, h));
            let out = run(&mut g, s, k, vec![src.clone()]);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].pts, 11);
            let steps = fill_max_pixsteps(pixdesc::descriptor(fmt));
            for p in 0..src.planes.len() {
                let (cols, rows) = plane_dims(fmt, w, h, p);
                let step = steps[p];
                let sls = src.planes[p].linesize;
                let ols = out[0].planes[p].linesize;
                for r in 0..rows {
                    for c in 0..cols {
                        for b in 0..step {
                            assert_eq!(
                                out[0].planes[p].data()[r * ols + c * step + b],
                                src.planes[p].data()[r * sls + (cols - 1 - c) * step + b],
                                "{:?} plane {p} ({r},{c},{b})",
                                fmt
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hflip_is_an_involution() {
        // hflip twice = identity (a cheap whole-surface check on 16-bit).
        let mut g = FilterGraph::new();
        let src = g
            .create_filter(
                "buffer",
                "video_size=6x4:pix_fmt=gray16le:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let f1 = g.create_filter("hflip", "").unwrap();
        let f2 = g.create_filter("hflip", "").unwrap();
        let sink = g.create_filter("buffersink", "").unwrap();
        g.link(src, 0, f1, 0).unwrap();
        g.link(f1, 0, f2, 0).unwrap();
        g.link(f2, 0, sink, 0).unwrap();
        g.config().unwrap();
        let orig = frame_at(PixelFormat::Gray16le, 6, 4, 3);
        let out = run(&mut g, src, sink, vec![orig.clone()]);
        assert_eq!(out[0].planes[0].data(), orig.planes[0].data());
    }

    #[test]
    fn hflip_query_formats_excludes_packed_422() {
        // vf_hflip.c:49-53: log2 mismatch AND all comps in plane 0 — the
        // packed YUYV/UYVY pair. Planar yuv422p SURVIVES (comp planes
        // differ).
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let f = g.create_filter("hflip", "").unwrap();
        let sink = g.alloc_test_sink();
        let lin = g.link(src, 0, f, 0).unwrap();
        let _lout = g.link(f, 0, sink, 0).unwrap();
        let mut imp = g.nodes[f.0].imp.take().expect("imp present");
        imp.query_formats(&mut g, f).unwrap();
        g.nodes[f.0].imp = Some(imp);
        let idx = g.links[lin.0].outcfg.formats.expect("declared");
        let expected: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&p| {
                let d = pixdesc::descriptor(p);
                !(d.log2_chroma_w != d.log2_chroma_h && d.comp[0].plane == d.comp[1].plane)
            })
            .collect();
        assert_eq!(g.fmt_lists[idx as usize], expected);
        assert!(!g.fmt_lists[idx as usize].contains(&PixelFormat::Yuyv422));
        assert!(!g.fmt_lists[idx as usize].contains(&PixelFormat::Uyvy422));
        assert!(g.fmt_lists[idx as usize].contains(&PixelFormat::Yuv422p));
    }

    // ---- vflip ----------------------------------------------------------------

    #[test]
    fn vflip_reverses_rows_per_plane() {
        for fmt in [PixelFormat::Yuv420p, PixelFormat::Rgb24, PixelFormat::Gbrap] {
            let (w, h) = (7u32, 5u32);
            let src = frame_at(fmt, w, h, 11);
            let (mut g, s, k, _i, lout) = chain("vflip", fmt, (w, h));
            assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (w, h));
            let out = run(&mut g, s, k, vec![src.clone()]);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].pts, 11);
            for p in 0..src.planes.len() {
                let rows = src.planes[p].rows;
                let ls = src.planes[p].linesize;
                for i in 0..rows {
                    assert_eq!(
                        &out[0].planes[p].data()[i * ls..(i + 1) * ls],
                        &src.planes[p].data()[(rows - 1 - i) * ls..(rows - i) * ls],
                        "{:?} plane {p} row {i}",
                        fmt
                    );
                }
            }
        }
    }

    #[test]
    fn vflip_is_an_involution() {
        let mut g = FilterGraph::new();
        let src = g
            .create_filter(
                "buffer",
                "video_size=6x4:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let f1 = g.create_filter("vflip", "").unwrap();
        let f2 = g.create_filter("vflip", "").unwrap();
        let sink = g.create_filter("buffersink", "").unwrap();
        g.link(src, 0, f1, 0).unwrap();
        g.link(f1, 0, f2, 0).unwrap();
        g.link(f2, 0, sink, 0).unwrap();
        g.config().unwrap();
        let orig = frame_at(PixelFormat::Yuv420p, 6, 4, 3);
        let out = run(&mut g, src, sink, vec![orig.clone()]);
        for p in 0..orig.planes.len() {
            assert_eq!(out[0].planes[p].data(), orig.planes[p].data(), "plane {p}");
        }
    }

    #[test]
    fn vflip_odd_height_chroma_ceil() {
        // 5 rows → chroma 3 rows (ceil): all three chroma rows reverse.
        let src = frame_at(PixelFormat::Yuv420p, 8, 5, 0);
        let (mut g, s, k, _i, _l) = chain("vflip", PixelFormat::Yuv420p, (8, 5));
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(out[0].planes[1].rows, 3);
        let rows = src.planes[1].rows;
        let ls = src.planes[1].linesize;
        for i in 0..rows {
            assert_eq!(
                &out[0].planes[1].data()[i * ls..(i + 1) * ls],
                &src.planes[1].data()[(rows - 1 - i) * ls..(rows - i) * ls]
            );
        }
    }

    // ---- both: metadata, options, defs -----------------------------------------

    #[test]
    fn flips_preserve_geometry_and_metadata() {
        for name in ["hflip", "vflip"] {
            let mut src = frame_at(PixelFormat::Yuv420p, 6, 4, 42);
            src.sample_aspect_ratio = Rational::ONE;
            let (mut g, s, k, _i, _l) = chain(name, PixelFormat::Yuv420p, (6, 4));
            let out = run(&mut g, s, k, vec![src]);
            assert_eq!((out[0].width, out[0].height), (6, 4), "{name}");
            assert_eq!(out[0].format, PixelFormat::Yuv420p, "{name}");
            assert_eq!(out[0].pts, 42, "{name}");
            assert_eq!(out[0].duration, 1, "{name}");
            assert_eq!(out[0].time_base, Rational::new(1, 25), "{name}");
            assert_eq!(out[0].sample_aspect_ratio, Rational::ONE, "{name}");
        }
    }

    #[test]
    fn flips_reject_any_option() {
        // No AVOption table: every option is unknown (C's own shape via
        // the leftover check).
        for name in ["hflip", "vflip"] {
            let mut g = FilterGraph::new();
            match g.create_filter(name, "x=1").unwrap_err() {
                Error::NotFound(m) => assert_eq!(m, "No such option: x", "{name}"),
                other => panic!("unexpected error for {name}: {other}"),
            }
            // A positional value has no shorthand slot either.
            let mut g = FilterGraph::new();
            match g.create_filter(name, "1").unwrap_err() {
                Error::InvalidArgument(m) => {
                    assert_eq!(m, "No option name near '1'", "{name}")
                }
                other => panic!("unexpected error for {name}: {other}"),
            }
        }
    }

    #[test]
    fn flip_def_shapes() {
        let h = filter_def("hflip").expect("hflip registered");
        let v = filter_def("vflip").expect("vflip registered");
        assert!(std::ptr::eq(h, &HFLIP_DEF));
        assert!(std::ptr::eq(v, &VFLIP_DEF));
        for def in [h, v] {
            assert_eq!(def.inputs.len(), 1);
            assert_eq!(def.outputs.len(), 1);
            assert_eq!(def.inputs[0].name, "default");
            assert_eq!(def.flags, FilterFlags(0));
            assert!(def.shorthand.is_empty());
        }
    }
}
