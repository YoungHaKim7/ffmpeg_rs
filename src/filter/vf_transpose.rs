//! `transpose` video filter — port of `libavfilter/vf_transpose.c` (419
//! lines) + `transpose.h` (the enums).
//!
//! Transposes (optionally with flips) every frame: the output link swaps
//! `w`/`h` (`config_props_output`, vf_transpose.c:213-214), the SAR becomes
//! its reciprocal (216-220), and each frame is re-buffered with the pixel
//! grid mirrored along the main diagonal plus the `dir` flips
//! (`filter_slice`, 269-329).
//!
//! ## The transposition mapping
//!
//! C's `filter_slice` walks 8×8 blocks with per-plane function tables
//! (`transpose_8x8_*`, 76-177) selected by `pixstep`, and encodes the
//! direction as two sign tricks: `dir&1` starts the source at the LAST row
//! with a negated `src_linesize` (297-300), `dir&2` starts the destination
//! at the last row with a negated `dst_linesize` (302-305). Unrolling the
//! block loop (pure performance structure; the port is single-threaded, one
//! slice) the per-pixel mapping for plane `p`, output row `r`, column `c`:
//!
//! ```text
//! src_row = dir & 1 ? inh - 1 - c : c      // the source row this out col reads
//! dst_row = dir & 2 ? outh - 1 - r : r     // where the out row lands
//! out[dst_row][c] = in[src_row][r]         // `step` bytes per pixel
//! ```
//!
//! i.e. `dir=0` is the plain transpose `out(r,c) = in(c,r)`, `dir=1`
//! (`clock`) rotates clockwise, `dir=2` (`cclock`) counterclockwise, `dir=3`
//! (`clock_flip`) clockwise + vertical flip. The port computes exactly this
//! with byte copies — every C variant (`_8/_16/_24/_32/_48/_64`, including
//! the big-endian `AV_RB24/RB48` loads) is a byte-for-byte copy of `step`
//! bytes, so one generic loop reproduces all six; the negative-stride
//! pointer tricks become the index arithmetic above (the port forbids
//! negative linesizes, frame.rs module doc).
//!
//! ## Dropped from C (each also noted at its site)
//!
//! * `get_video_buffer` (256-263): pad buffer callbacks do not exist (no
//!   frame pools); a fresh output frame is allocated per `filter_frame`.
//! * slice threading (`ff_filter_execute`, the job split 285-286) and the
//!   x86 SIMD dispatch (240-246): single-threaded scalar loop.
//! * `av_assert0(desc_in->nb_components == desc_out->nb_components)`
//!   (208): transpose does not convert formats — both links carry the same
//!   negotiated format, so the assert is a tautology here.
//! * `TRANSPOSE_REVERSAL/HFLIP/VFLIP` (dir 4-6) only reach the deprecated
//!   `dir&4` path (187-192) which rewrites them to 0-3 — the enum values
//!   are kept for the option table, exactly like C.
//!
//! ## Format restriction
//!
//! `query_formats` (55-74) rejects formats whose chroma is subsampled
//! differently on the two axes (`log2_chroma_w != log2_chroma_h`,
//! 64-67 — 4:2:2 in all packings) plus PAL/HW/bitstream (none exist in the
//! port's universe): a transposed 4:2:2 grid is not a 4:2:2 grid. The
//! port's list = `PixelFormat::ALL` filtered by the log2 equality.

use crate::{
    log_error, log_verbose, log_warning,
    util::{
        error::{Error, Result},
        frame::Frame,
        pixdesc,
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::vf_crop::{ceil_rshift, fill_max_pixsteps};
use super::{
    filter::{FilterDef, FilterFlags, FilterImpl, PadDef, PadRef, filter_frame},
    graph::FilterGraph,
    link::NodeId,
};

// ---------------------------------------------------------------------------
// transpose.h (24-38)
// ---------------------------------------------------------------------------

/// `enum PassthroughType` (transpose.h:24-28).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PassthroughType {
    /// `TRANSPOSE_PT_TYPE_NONE` — always apply transposition.
    #[default]
    None,
    /// `TRANSPOSE_PT_TYPE_LANDSCAPE` — preserve landscape geometry.
    Landscape,
    /// `TRANSPOSE_PT_TYPE_PORTRAIT` — preserve portrait geometry.
    Portrait,
}

/// `enum TransposeDir` (transpose.h:30-38). Values 4-6 (`REVERSAL`, `HFLIP`,
/// `VFLIP`) are deprecated spellings that `config_props_output` rewrites via
/// `dir&4` (vf_transpose.c:187-192) — kept so the numeric option round-trips
/// like C's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum TransposeDir {
    /// `TRANSPOSE_CCLOCK_FLIP` — rotate counterclockwise with vertical flip.
    #[default]
    CclockFlip = 0,
    /// `TRANSPOSE_CLOCK` — rotate clockwise.
    Clock = 1,
    /// `TRANSPOSE_CCLOCK` — rotate counterclockwise.
    Cclock = 2,
    /// `TRANSPOSE_CLOCK_FLIP` — rotate clockwise with vertical flip.
    ClockFlip = 3,
    /// `TRANSPOSE_REVERSAL` — rotate by half-turn (deprecated alias of dir 0).
    Reversal = 4,
    /// `TRANSPOSE_HFLIP` (deprecated alias of dir 1).
    Hflip = 5,
    /// `TRANSPOSE_VFLIP` (deprecated alias of dir 2).
    Vflip = 6,
}

// ---------------------------------------------------------------------------
// Private context (vf_transpose.c:43-53)
// ---------------------------------------------------------------------------

/// `TransContext` (vf_transpose.c:43-53). `TransVtable vtables[4]`
/// (42-46 of transpose.h) collapses into the per-plane `pixsteps` walk —
/// the generic byte-copy loop IS all six C function variants.
pub struct TransContext {
    /// `hsub`/`vsub` — chroma subsampling log2s (204-205).
    pub hsub: u32,
    pub vsub: u32,
    /// `planes` — `av_pix_fmt_count_planes(outlink->format)` (206).
    pub planes: usize,
    /// `pixsteps[4]` — `av_image_fill_max_pixsteps(desc_out)` (211).
    pub pixsteps: [usize; 4],
    /// `passthrough` — the geometry-preservation mode.
    pub passthrough: PassthroughType,
    /// `dir` — kept as the RAW u32 (0..=7): config_props_output rewrites
    /// `dir&4` values in place (187-192).
    pub dir: u32,
}

impl Default for TransContext {
    /// C zero-init + the AVOption defaults (vf_transpose.c:375-389):
    /// `dir = TRANSPOSE_CCLOCK_FLIP` (0), `passthrough = NONE` (0).
    fn default() -> Self {
        TransContext {
            hsub: 0,
            vsub: 0,
            planes: 0,
            pixsteps: [0; 4],
            passthrough: PassthroughType::None,
            dir: 0,
        }
    }
}

/// C `strtol(s, &p, 10)` core (the set_string_int numeric arm) — same shape
/// as vf_scale.rs's private twin.
fn strtol_i32(s: &str) -> (i32, usize) {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digits = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits {
        return (0, 0);
    }
    let text = std::str::from_utf8(&b[start..i]).expect("ascii scan produced ascii slice");
    let n: i64 = text
        .parse()
        .unwrap_or(if b[start] == b'-' { i64::MIN } else { i64::MAX });
    (n as i32, i)
}

/// A fully-consumed decimal i64, or None.
fn strtol_i64_full(val: &str) -> Option<i64> {
    let (n, consumed) = strtol_i32(val);
    if consumed == val.len() && !val.is_empty() {
        Some(n as i64)
    } else {
        None
    }
}

/// The buffersrc-style generic parse failure (opt.c:500) — the same text the
/// port's other INT+CONST option parsers produce.
fn option_parse_failed(name: &str, key: &str, val: &str) -> Error {
    log_error!(
        Some(name),
        "Unable to parse \"{key}\" option value \"{val}\"\n"
    );
    Error::InvalidArgument(format!("Unable to parse \"{key}\" option value \"{val}\""))
}

/// `dir` (vf_transpose.c:376-380): CONST names, then the numeric fallback —
/// `AV_OPT_TYPE_INT` with range 0..=7.
fn parse_dir(name: &str, key: &str, val: &str) -> Result<u32> {
    let n = match val {
        "cclock_flip" => 0,
        "clock" => 1,
        "cclock" => 2,
        "clock_flip" => 3,
        _ => {
            let Some(n) = strtol_i64_full(val) else {
                return Err(option_parse_failed(name, key, val));
            };
            if !(0..=7).contains(&n) {
                return Err(Error::InvalidArgument(format!(
                    "Value {n} for parameter {key} out of range"
                )));
            }
            n
        }
    };
    Ok(n as u32)
}

/// `passthrough` (vf_transpose.c:382-386): CONST names, then the numeric
/// fallback (C's INT range is 0..INT_MAX; values other than 0/1/2 behave
/// like NONE after `config_props_output` normalizes them, 201).
fn parse_passthrough(name: &str, key: &str, val: &str) -> Result<PassthroughType> {
    let n = match val {
        "none" => 0,
        "portrait" => 1,
        "landscape" => 2,
        _ => {
            let Some(n) = strtol_i64_full(val) else {
                return Err(option_parse_failed(name, key, val));
            };
            if !(0..=i32::MAX as i64).contains(&n) {
                return Err(Error::InvalidArgument(format!(
                    "Value {n} for parameter {key} out of range"
                )));
            }
            n
        }
    };
    Ok(match n {
        1 => PassthroughType::Portrait,
        2 => PassthroughType::Landscape,
        _ => PassthroughType::None,
    })
}

// ---------------------------------------------------------------------------
// FilterImpl — init / query_formats / config_props / filter_frame
// ---------------------------------------------------------------------------

impl FilterImpl for TransContext {
    /// Option intake (the AVOption table vf_transpose.c:375-389): `dir`
    /// (INT 0..=7 + the four CONST names) and `passthrough` (INT + three
    /// CONST names). Last occurrence wins; unknown keys pre-scanned first,
    /// same as vf_scale.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        const KNOWN_KEYS: &[&str] = &["dir", "passthrough"];
        if let Some((key, _)) = g.nodes[node.0]
            .opts
            .entries
            .iter()
            .find(|(k, _)| !KNOWN_KEYS.contains(&k.as_str()))
        {
            return Err(Error::NotFound(format!("No such option: {key}")));
        }

        let name = g.nodes[node.0].name.clone();
        let entries = std::mem::take(&mut g.nodes[node.0].opts.entries);
        let mut leftovers = Vec::new();
        for (key, value) in entries {
            match key.as_str() {
                "dir" => self.dir = parse_dir(&name, &key, &value)?,
                "passthrough" => self.passthrough = parse_passthrough(&name, &key, &value)?,
                _ => leftovers.push((key, value)),
            }
        }
        g.nodes[node.0].opts.entries = leftovers;
        Ok(())
    }

    /// `query_formats` (vf_transpose.c:55-74): every format with
    /// `log2_chroma_w == log2_chroma_h` (the 4:2:2 family is excluded —
    /// PAL/HW/bitstream flags do not exist in the port's universe), one list
    /// handle on both pads, `PixelFormat::ALL` order (C's enum-order
    /// `ff_add_format` walk).
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let list: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&f| {
                let d = pixdesc::descriptor(f);
                d.log2_chroma_w == d.log2_chroma_h
            })
            .collect();
        let list = g.alloc_pix_list(list);
        g.set_common_formats(node, list)?;
        Ok(())
    }

    /// `config_props` — wired on the OUTPUT pad only
    /// (vf_transpose.c:402-408).
    fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
        match pad {
            PadRef::Out(0) => self.config_props_output(g, node),
            _ => Ok(()),
        }
    }

    /// `filter_frame` (vf_transpose.c:331-370).
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        _pad: usize,
        frame: Frame,
    ) -> Result<()> {
        let outlink = g.outlink(node, 0);

        // (340-341) passthrough forwards the frame VERBATIM.
        if self.passthrough != PassthroughType::None {
            return filter_frame(g, outlink, frame);
        }

        // (343-347) the output buffer at the LINK's (swapped) geometry.
        let fmt = g.links[outlink.0]
            .format
            .expect("formats picked before config_props");
        let mut out = Frame::alloc(fmt, g.links[outlink.0].w, g.links[outlink.0].h)?;

        // (349-358) copy props, then the frame-level SAR: a raw num/den swap
        // when num != 0 (NOT the reduced av_div_q the link uses — C really
        // has both shapes).
        out.copy_props(&frame);
        out.sample_aspect_ratio = if frame.sample_aspect_ratio.num == 0 {
            frame.sample_aspect_ratio
        } else {
            Rational::new(frame.sample_aspect_ratio.den, frame.sample_aspect_ratio.num)
        };

        // (349-364 + filter_slice 269-329) the transposition — see the
        // module doc mapping. Single slice: start = 0, end = outh.
        for p in 0..self.planes {
            let step = self.pixsteps[p];
            let hsub = if p == 1 || p == 2 { self.hsub } else { 0 };
            let vsub = if p == 1 || p == 2 { self.vsub } else { 0 };
            // filter_slice 282-284.
            let inh = ceil_rshift(frame.height, vsub) as usize;
            let outw = ceil_rshift(out.width, hsub) as usize;
            let outh = ceil_rshift(out.height, vsub) as usize;
            let ils = frame.planes[p].linesize;
            let ols = out.planes[p].linesize;
            for r in 0..outh {
                let dst_row = if self.dir & 2 != 0 { outh - 1 - r } else { r };
                for c in 0..outw {
                    let src_row = if self.dir & 1 != 0 { inh - 1 - c } else { c };
                    let so = src_row * ils + r * step;
                    let doff = dst_row * ols + c * step;
                    out.plane_mut(p)[doff..doff + step]
                        .copy_from_slice(&frame.plane(p)[so..so + step]);
                }
            }
        }

        filter_frame(g, outlink, out)
    }
}

impl TransContext {
    /// `config_props_output` (vf_transpose.c:179-254) — the OUTPUT pad's
    /// config_props.
    fn config_props_output(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let inlink = g.inlink(node, 0);
        let outlink = g.outlink(node, 0);
        let fmt = g.links[inlink.0]
            .format
            .expect("formats picked before config_props (query_formats round)");

        // (187-192) the deprecated dir 4-7: warn, mask to 0-3, force the
        // landscape passthrough.
        if self.dir & 4 != 0 {
            log_warning!(
                Some("transpose"),
                "dir values greater than 3 are deprecated, use the passthrough option instead\n"
            );
            self.dir &= 3;
            self.passthrough = PassthroughType::Landscape;
        }

        let (iw, ih) = (g.links[inlink.0].w, g.links[inlink.0].h);

        // (194-202) the geometry-preservation test — note >= / <=: a SQUARE
        // input matches BOTH modes.
        let keep = match self.passthrough {
            PassthroughType::Landscape => iw >= ih,
            PassthroughType::Portrait => iw <= ih,
            PassthroughType::None => false,
        };
        if keep {
            // (196-199) passthrough mode: the output link keeps the input
            // geometry (the swap below never runs).
            log_verbose!(
                Some("transpose"),
                "w:{iw} h:{ih} -> w:{iw} h:{ih} (passthrough mode)\n"
            );
            return Ok(());
        }
        self.passthrough = PassthroughType::None;

        // (204-211) the plane tables. desc_in == desc_out (no conversion);
        // C's nb_components assert (208) is a tautology, dropped.
        let desc = pixdesc::descriptor(fmt);
        self.hsub = desc.log2_chroma_w as u32;
        self.vsub = desc.log2_chroma_h as u32;
        self.planes = pixdesc::count_planes(fmt);
        self.pixsteps = fill_max_pixsteps(desc);

        // (213-214) the geometry swap.
        g.links[outlink.0].w = ih;
        g.links[outlink.0].h = iw;

        // (216-220) the link SAR: reciprocal when set, verbatim when not.
        let sar = g.links[inlink.0].sample_aspect_ratio;
        g.links[outlink.0].sample_aspect_ratio = if sar.num != 0 {
            Rational::ONE / sar
        } else {
            sar
        };

        // (248-252) the verbose tail (rotation/vflip summary) and the x86
        // dispatch (240-246) — the latter is not ported.
        log_verbose!(
            Some("transpose"),
            "w:{iw} h:{ih} dir:{} -> w:{} h:{} rotation:{} vflip:{}\n",
            self.dir,
            g.links[outlink.0].w,
            g.links[outlink.0].h,
            if self.dir == 1 || self.dir == 3 {
                "clockwise"
            } else {
                "counterclockwise"
            },
            (self.dir == 0 || self.dir == 3) as i32
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Filter descriptor (vf_transpose.c:393-419)
// ---------------------------------------------------------------------------

/// The shared single video pad (vf_transpose.c:393-408).
static DEFAULT_PAD: PadDef = PadDef {
    name: "default",
    needs_writable: false,
};

/// `ff_vf_transpose` (vf_transpose.c:410-419): "Transpose input video."
///
/// * flags: transpose is NOT in C's `ff_filter_frame` validation skip list
///   (avfilter.c:1075-1082) → `FilterFlags(0)`; `AVFILTER_FLAG_SLICE_THREADS`
///   (414) is not modeled.
/// * shorthand: the option-table walk (avfilter.c:855-902) — `dir`, then
///   `passthrough` (the CONST entries are skipped), so `transpose=1` and
///   `transpose=cclock` bind exactly like C.
pub static TRANSPOSE_DEF: FilterDef = FilterDef {
    name: "transpose",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags(0),
    shorthand: &["dir", "passthrough"],
    make: || Box::new(TransContext::default()),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::filter_def;
    use crate::filter::link::LinkId;

    /// buffer(pix_fmt WxH, tb 1/25, sar) → transpose(args) → buffersink.
    /// `sar == ""` omits the option — the link keeps the 0/0 UNKNOWN
    /// default (buffersrc's own range check rejects an explicit `sar=0/0`,
    /// same as C's AVOption range check).
    fn tchain(
        args: &str,
        fmt: PixelFormat,
        size: (u32, u32),
        sar: &str,
    ) -> (FilterGraph, NodeId, NodeId, LinkId, LinkId) {
        let mut g = FilterGraph::new();
        let sar_part = if sar.is_empty() {
            String::new()
        } else {
            format!(":sar={sar}")
        };
        let src = g
            .create_filter(
                "buffer",
                &format!(
                    "video_size={}x{}:pix_fmt={}:time_base=1/25{sar_part}",
                    size.0,
                    size.1,
                    fmt.name()
                ),
            )
            .expect("buffer init");
        let f = g.create_filter("transpose", args).expect("transpose init");
        let sink = g.create_filter("buffersink", "").unwrap();
        let lin = g.link(src, 0, f, 0).unwrap();
        let lout = g.link(f, 0, sink, 0).unwrap();
        g.config().expect("graph config");
        (g, src, sink, lin, lout)
    }

    /// The same pixel generator as vf_crop's tests — byte `b` of the pixel
    /// at (row, col) of plane `p`.
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

    /// Verify the transposition mapping (module doc): for every plane,
    /// `out[dst_row][c] == in[src_row][r]` with the dir flips.
    fn assert_transpose(out: &Frame, src: &Frame, dir: u32) {
        assert_eq!(out.format, src.format);
        assert_eq!(out.width, src.height, "geometry swapped");
        assert_eq!(out.height, src.width);
        let desc = pixdesc::descriptor(src.format);
        let steps = fill_max_pixsteps(desc);
        for p in 0..out.planes.len() {
            let (hs, vs) = if p == 1 || p == 2 {
                (desc.log2_chroma_w as u32, desc.log2_chroma_h as u32)
            } else {
                (0, 0)
            };
            let inh = ceil_rshift(src.height, vs) as usize;
            let outw = ceil_rshift(out.width, hs) as usize;
            let outh = ceil_rshift(out.height, vs) as usize;
            assert_eq!(outh, ceil_rshift(src.width, vs) as usize);
            assert_eq!(outw, ceil_rshift(src.height, hs) as usize);
            let ils = src.planes[p].linesize;
            let ols = out.planes[p].linesize;
            for r in 0..outh {
                let dst_row = if dir & 2 != 0 { outh - 1 - r } else { r };
                for c in 0..outw {
                    let src_row = if dir & 1 != 0 { inh - 1 - c } else { c };
                    for b in 0..steps[p] {
                        assert_eq!(
                            out.planes[p].data()[dst_row * ols + c * steps[p] + b],
                            src.planes[p].data()[src_row * ils + r * steps[p] + b],
                            "plane {p} out({dst_row},{c}) byte {b}"
                        );
                    }
                }
            }
        }
    }

    // ---- the four directions ------------------------------------------------

    #[test]
    fn transpose_four_dirs_yuv420p() {
        // 5x3 (odd): chroma 3x2 → out 3x5 with chroma 2x3 — the ceil
        // arithmetic of AV_CEIL_RSHIFT.
        for (args, dir) in [
            ("dir=cclock_flip", 0u32),
            ("dir=clock", 1),
            ("dir=cclock", 2),
            ("dir=clock_flip", 3),
            ("0", 0), // numeric spellings
            ("1", 1),
            ("2", 2),
            ("3", 3),
        ] {
            let src = frame_at(PixelFormat::Yuv420p, 5, 3, 7);
            let (mut g, s, k, _i, lout) = tchain(args, PixelFormat::Yuv420p, (5, 3), "1/1");
            assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (3, 5), "{args}");
            let out = run(&mut g, s, k, vec![src.clone()]);
            assert_eq!(out.len(), 1, "{args}");
            assert_eq!(out[0].pts, 7, "pts survives copy_props");
            assert_transpose(&out[0], &src, dir);
        }
    }

    #[test]
    fn transpose_packed_rgb_steps() {
        // rgb24 (step 3 — the AV_RB24 variant), rgba (step 4), gray16le
        // (step 2): the generic byte-copy loop covers every C bpp variant.
        for fmt in [PixelFormat::Rgb24, PixelFormat::Rgba, PixelFormat::Gray16le] {
            for (args, dir) in [("0", 0u32), ("1", 1), ("2", 2), ("3", 3)] {
                let src = frame_at(fmt, 6, 4, 0);
                let (mut g, s, k, _i, _l) = tchain(args, fmt, (6, 4), "1/1");
                let out = run(&mut g, s, k, vec![src.clone()]);
                assert_transpose(&out[0], &src, dir);
            }
        }
    }

    #[test]
    fn transpose_planar_four_planes() {
        // gbrap: 4 planes, all full resolution.
        let src = frame_at(PixelFormat::Gbrap, 6, 4, 0);
        let (mut g, s, k, _i, _l) = tchain("1", PixelFormat::Gbrap, (6, 4), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(out[0].planes.len(), 4);
        assert_transpose(&out[0], &src, 1);
    }

    #[test]
    fn transpose_dir_corner_semantics() {
        // Pin the named semantics by the output's corner pixels (gray8 4x3).
        let src = frame_at(PixelFormat::Gray8, 4, 3, 0);
        let corner =
            |f: &Frame, r: usize, c: usize| f.planes[0].data()[r * f.planes[0].linesize + c];
        for (args, (dr, dc), (sr, sc), what) in [
            (
                "dir=cclock_flip",
                (0usize, 0usize),
                (0, 0),
                "ccw+flip: top-left <- top-left",
            ),
            (
                "dir=clock",
                (0, 0),
                (2, 0),
                "clockwise: top-left <- bottom-left",
            ),
            ("dir=cclock", (0, 0), (0, 3), "ccw: top-left <- top-right"),
            (
                "dir=clock_flip",
                (0, 0),
                (2, 3),
                "cw+flip: top-left <- bottom-right",
            ),
        ] {
            let (mut g, s, k, _i, _l) = tchain(args, PixelFormat::Gray8, (4, 3), "1/1");
            let out = run(&mut g, s, k, vec![src.clone()]);
            assert_eq!(corner(&out[0], dr, dc), corner(&src, sr, sc), "{what}");
        }
    }

    // ---- SAR -----------------------------------------------------------------

    #[test]
    fn transpose_sar_link_and_frame() {
        // Link: reciprocal (av_div_q); frame: raw num/den swap of the
        // FRAME's own sar (the buffer's sar option sets the LINK, not the
        // pushed frames — stamp the frame explicitly).
        let mut src0 = frame_at(PixelFormat::Gray8, 4, 3, 0);
        src0.sample_aspect_ratio = Rational::new(2, 3);
        let (mut g, s, k, _i, lout) = tchain("0", PixelFormat::Gray8, (4, 3), "2/3");
        assert_eq!(g.links[lout.0].sample_aspect_ratio, Rational::new(3, 2));
        let out = run(&mut g, s, k, vec![src0]);
        assert_eq!(out[0].sample_aspect_ratio, Rational::new(3, 2));

        // An unknown FRAME SAR (num == 0) is copied verbatim. The LINK sar
        // never stays unknown here: the graph defaults pass normalizes an
        // unset source SAR to 1/1 (avfilter.c:405-430) BEFORE transpose's
        // config runs, so the reciprocal is 1/1 too — same as C.
        let mut f = frame_at(PixelFormat::Gray8, 4, 3, 0);
        f.sample_aspect_ratio = Rational::UNKNOWN;
        let (mut g, s, k, _i, lout) = tchain("0", PixelFormat::Gray8, (4, 3), "");
        assert_eq!(g.links[lout.0].sample_aspect_ratio, Rational::ONE);
        let out = run(&mut g, s, k, vec![f]);
        assert_eq!(out[0].sample_aspect_ratio, Rational::UNKNOWN);
    }

    // ---- passthrough -----------------------------------------------------------

    #[test]
    fn transpose_passthrough_landscape() {
        // 8x4 landscape input + landscape mode: forwarded VERBATIM (same
        // frame, same geometry — vf_transpose.c:194-199 + 340-341).
        let src = frame_at(PixelFormat::Yuv420p, 8, 4, 9);
        let (mut g, s, k, _i, lout) =
            tchain("passthrough=landscape", PixelFormat::Yuv420p, (8, 4), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (8, 4));
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!((out[0].width, out[0].height), (8, 4));
        assert_eq!(out[0].planes[0].data(), src.planes[0].data(), "verbatim");

        // A PORTRAIT input under landscape mode IS transposed (4x8 → 8x4).
        let src = frame_at(PixelFormat::Yuv420p, 4, 8, 9);
        let (mut g, s, k, _i, lout) =
            tchain("passthrough=landscape", PixelFormat::Yuv420p, (4, 8), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (8, 4));
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_transpose(&out[0], &src, 0);
    }

    #[test]
    fn transpose_passthrough_portrait_and_square() {
        // Portrait mode: portrait input passes, landscape transposes.
        let src = frame_at(PixelFormat::Gray8, 4, 8, 9);
        let (mut g, s, k, _i, lout) =
            tchain("passthrough=portrait", PixelFormat::Gray8, (4, 8), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (4, 8));
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(out[0].planes[0].data(), src.planes[0].data(), "verbatim");

        let src = frame_at(PixelFormat::Gray8, 8, 4, 9);
        let (mut g, s, k, _i, lout) =
            tchain("passthrough=portrait", PixelFormat::Gray8, (8, 4), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (4, 8));
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_transpose(&out[0], &src, 0);

        // A SQUARE input matches BOTH modes (>= / <=, vf_transpose.c:194-195).
        for mode in ["landscape", "portrait"] {
            let src = frame_at(PixelFormat::Gray8, 6, 6, 9);
            let (mut g, s, k, _i, lout) = tchain(
                &format!("passthrough={mode}"),
                PixelFormat::Gray8,
                (6, 6),
                "1/1",
            );
            assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (6, 6));
            let out = run(&mut g, s, k, vec![src.clone()]);
            assert_eq!(out[0].planes[0].data(), src.planes[0].data(), "{mode}");
        }
    }

    #[test]
    fn transpose_deprecated_dirs_rewrite() {
        // dir 4-7: warn + dir &= 3 + passthrough = LANDSCAPE
        // (vf_transpose.c:187-192). dir=5 behaves as dir=1 + landscape.
        let src = frame_at(PixelFormat::Gray8, 8, 4, 9);
        let (mut g, s, k, _i, lout) = tchain("dir=5", PixelFormat::Gray8, (8, 4), "1/1");
        assert_eq!(
            (g.links[lout.0].w, g.links[lout.0].h),
            (8, 4),
            "landscape kept"
        );
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(out[0].planes[0].data(), src.planes[0].data(), "passthrough");

        let src = frame_at(PixelFormat::Gray8, 4, 8, 9);
        let (mut g, s, k, _i, lout) = tchain("dir=5", PixelFormat::Gray8, (4, 8), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (8, 4), "transposed");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_transpose(&out[0], &src, 1);

        // dir=4 → 0, dir=6 → 2, dir=7 → 3 (all with landscape mode).
        for (raw, eff) in [(4u32, 0u32), (6, 2), (7, 3)] {
            let src = frame_at(PixelFormat::Gray8, 4, 8, 9);
            let (mut g, s, k, _i, _l) =
                tchain(&format!("dir={raw}"), PixelFormat::Gray8, (4, 8), "1/1");
            let out = run(&mut g, s, k, vec![src.clone()]);
            assert_transpose(&out[0], &src, eff);
        }
    }

    // ---- format restriction -----------------------------------------------------

    #[test]
    fn transpose_query_formats_excludes_422() {
        // log2_chroma_w != log2_chroma_h → excluded (yuv422p, yuyv422,
        // uyvy422, yuv422p10le); everything else in ALL order.
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let f = g.create_filter("transpose", "").unwrap();
        let sink = g.alloc_test_sink();
        let lin = g.link(src, 0, f, 0).unwrap();
        let lout = g.link(f, 0, sink, 0).unwrap();
        let mut imp = g.nodes[f.0].imp.take().expect("imp present");
        imp.query_formats(&mut g, f).unwrap();
        g.nodes[f.0].imp = Some(imp);
        let idx = g.links[lin.0].outcfg.formats.expect("declared");
        assert_eq!(idx, g.links[lout.0].incfg.formats.unwrap());
        let expected: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&p| {
                let d = pixdesc::descriptor(p);
                d.log2_chroma_w == d.log2_chroma_h
            })
            .collect();
        assert_eq!(g.fmt_lists[idx as usize], expected);
        for excluded in [
            PixelFormat::Yuv422p,
            PixelFormat::Yuv422p10le,
            PixelFormat::Yuyv422,
            PixelFormat::Uyvy422,
        ] {
            assert!(!g.fmt_lists[idx as usize].contains(&excluded));
        }
    }

    // ---- options / def -----------------------------------------------------------

    #[test]
    fn transpose_option_parsing() {
        // Positional shorthand: transpose=2 == dir=2; the second slot is
        // passthrough.
        let src = frame_at(PixelFormat::Gray8, 4, 3, 0);
        let (mut g, s, k, _i, _l) = tchain("1", PixelFormat::Gray8, (4, 3), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_transpose(&out[0], &src, 1);

        let src = frame_at(PixelFormat::Gray8, 4, 8, 0);
        let (mut g, s, k, _i, _l) = tchain("0:landscape", PixelFormat::Gray8, (4, 8), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(
            (out[0].width, out[0].height),
            (8, 4),
            "transposed (portrait in)"
        );

        // Unknown / bad values.
        let mut g = FilterGraph::new();
        match g.create_filter("transpose", "bogus=1").unwrap_err() {
            Error::NotFound(m) => assert_eq!(m, "No such option: bogus"),
            other => panic!("unexpected error: {other}"),
        }
        let mut g = FilterGraph::new();
        match g.create_filter("transpose", "dir=sideways").unwrap_err() {
            Error::InvalidArgument(m) => {
                assert_eq!(m, "Unable to parse \"dir\" option value \"sideways\"")
            }
            other => panic!("unexpected error: {other}"),
        }
        let mut g = FilterGraph::new();
        match g.create_filter("transpose", "dir=8").unwrap_err() {
            Error::InvalidArgument(m) => assert_eq!(m, "Value 8 for parameter dir out of range"),
            other => panic!("unexpected error: {other}"),
        }
        let mut g = FilterGraph::new();
        match g
            .create_filter("transpose", "passthrough=diagonal")
            .unwrap_err()
        {
            Error::InvalidArgument(m) => {
                assert_eq!(
                    m,
                    "Unable to parse \"passthrough\" option value \"diagonal\""
                )
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn transpose_def_shape() {
        let def = filter_def("transpose").expect("transpose registered");
        assert!(std::ptr::eq(def, &TRANSPOSE_DEF));
        assert_eq!(def.name, "transpose");
        assert_eq!(def.inputs.len(), 1);
        assert_eq!(def.outputs.len(), 1);
        assert_eq!(def.flags, FilterFlags(0));
        assert_eq!(def.shorthand, &["dir", "passthrough"][..]);
    }
}
