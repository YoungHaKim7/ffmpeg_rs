//! `null` video filter — port of `libavfilter/vf_null.c` (the whole file is
//! a 6-line static descriptor: no callbacks whatsoever).
//!
//! C relies on the framework defaults for everything: the default formats
//! query (`ff_default_query_formats` — vf_null is PASSTHROUGH) and
//! `default_filter_frame` (avfilter.c:1007-1010 — forward the frame to the
//! filter's first output unchanged). The Rust trait makes `filter_frame`
//! required, so the forward is stated explicitly; it is still pure engine —
//! the frame moves untouched (cheap `Frame` pass-by-value, Arc refs intact).
//!
//! C's `AVFILTER_FLAG_METADATA_ONLY` (frame data passes through untouched)
//! has no Rust counterpart flag — ownership transfer IS the metadata-only
//! semantics here.

use crate::util::error::Result;
use crate::util::frame::Frame;

use super::filter::{filter_frame, FilterDef, FilterFlags, FilterImpl, PadDef};
use super::graph::FilterGraph;
use super::link::NodeId;

/// Zero state — vf_null's private context is empty.
pub struct Null;

impl FilterImpl for Null {
    /// `default_filter_frame` (avfilter.c:1007-1010): forward to outputs[0].
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        _pad: usize,
        frame: Frame,
    ) -> Result<()> {
        let out = g.outlink(node, 0);
        filter_frame(g, out, frame)
    }
}

const DEFAULT_PAD: PadDef = PadDef { name: "default", needs_writable: false };

/// `ff_vf_null` (vf_null.c:29-35).
pub static NULL_DEF: FilterDef = FilterDef {
    name: "null",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    // AVFILTER_FLAG_METADATA_ONLY — see module doc.
    flags: FilterFlags(0),
    shorthand: &[],
    make: || Box::new(Null),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::link::Link;
    use crate::util::pixfmt::PixelFormat;

    #[test]
    fn null_forwards_frame_untouched() {
        let mut g = FilterGraph::new();
        let a = NodeId(0);
        let b = NodeId(1);
        for def in [&NULL_DEF, &NULL_DEF] {
            g.nodes.push(super::super::filter::FilterNode {
                def,
                name: def.name.to_string(),
                inputs: vec![None],
                outputs: vec![None],
                imp: Some((def.make)()),
                ready: 0,
                initialized: true,
                opts: super::Options::default(),
            });
        }
        let l = g.link(a, 0, b, 0).unwrap();

        let mut frame = Frame::alloc(PixelFormat::Yuv420p, 16, 16).unwrap();
        frame.pts = 33;
        let pixels_before: Vec<u8> = frame.plane(0).to_vec();
        filter_frame(&mut g, l, frame).unwrap();

        // The queued frame on the far side is the SAME data (Arc share, no
        // copy) with identical metadata.
        let out = g.links[l.0].fifo.front().expect("frame queued on dst side");
        assert_eq!(out.pts, 33);
        assert_eq!(out.width, 16);
        assert_eq!(out.format, PixelFormat::Yuv420p);
        assert_eq!(out.plane(0), &pixels_before[..]);
        assert_eq!(g.links[l.0].frame_count_in, 1);
        assert_eq!(g.nodes[b.0].ready, 300, "dst woken at priority 300");
    }
}
