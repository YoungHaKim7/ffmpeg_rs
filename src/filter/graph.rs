//! The filtergraph — port of `libavfilter/avfiltergraph.c` (graph lifecycle,
//! validity checks, format-list plumbing, `insert_filter`) plus
//! `avfilter_link` (`avfilter.c:149-196`) and `avfilter_insert_filter`
//! (`avfilter.c:282-326`).
//!
//! Wave 1 scope: `config()` is a stub that runs `graph_check_validity` only
//! (avfiltergraph.c:203-240) — the formats negotiation round
//! (`avfiltergraph.c:522-737`), `reduce_formats`, `pick_formats`,
//! `graph_config_links` and `graph_check_links` arrive with wave 2, as do the
//! buffersrc/buffersink runtime entry points (`add_frame`/`close_source`/
//! `get_frame`) and `parse_ptr` (graphparser.c).
//!
//! Not ported (audio/threads/misc, documented per C anchor): the sink-link
//! age heap (`avfiltergraph.c:1394-1414, 1516-1565` — multi-sink scheduling),
//! command queueing (1454-1514), `swap_sample_fmts`-family audio reorderings
//! (1057-1309), `ff_filter_graph_remove_filter` (101 — nothing removes nodes
//! after config), the AVOption table on AVFilterGraph (44-70 — only
//! `scale_sws_opts` survives as a plain field).

use crate::swscale::{ScaleAlgorithm, ScaleEngine};
use crate::util::color::{ColorRange, ColorSpace};
use crate::util::error::{Error, Result};
use crate::util::frame::Frame;
use crate::util::pixfmt::PixelFormat;
use crate::{log_error, log_verbose};

use super::filter::{FilterDef, FilterNode};
use super::formats;
use super::link::{Link, LinkId, ListIdx, NodeId};
use super::Options;

/// One negotiation axis — C's `AVFilterFormatsMerger.offset` field selection
/// (`formats.h:562-580`) becomes an enum. The merger order for video is
/// **formats → color_spaces → color_ranges** (`mergers_video`,
/// formats.c:391-424); the alpha-modes axis is dropped (out of scope).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Formats,
    ColorSpaces,
    ColorRanges,
}

/// `AVFilterInOut` (`avfilter.h:718-728`) — one open pad returned by graph
/// parsing. `name: None` is an unnamed pad (never matches by label).
#[derive(Clone, Debug)]
pub struct InOut {
    pub name: Option<String>,
    pub node: NodeId,
    pub pad: usize,
}

/// `AVFilterGraph` + `FFFilterGraph` flattened (`avfilter.h:561-619`,
/// `avfilter_internal.h:138-152`): the node/link arenas, the negotiation
/// list arenas, and the auto-conversion configuration.
pub struct FilterGraph {
    pub(crate) nodes: Vec<FilterNode>,
    pub(crate) links: Vec<Link>,
    /// Negotiation list arenas — `AVFilterFormats` refcounted lists; a
    /// `ListIdx` in a link half is a "reference" (identity = index equality,
    /// merges sweep losers — see `filter::formats`).
    pub(crate) fmt_lists: Vec<Vec<PixelFormat>>,
    pub(crate) csp_lists: Vec<Vec<ColorSpace>>,
    pub(crate) rng_lists: Vec<Vec<ColorRange>>,
    /// Options for auto-inserted scale filters; also set by the CLI's
    /// `-vf "sws_flags=..."` prefix (wave 3).
    pub scale_sws_opts: String,
    /// Default engine for every ScaleContext the graph makes.
    pub scale_engine: ScaleEngine,
    /// C default: bicubic.
    pub scale_algorithm: ScaleAlgorithm,
    /// `fffiltergraph->disable_auto_convert` (avfiltergraph.c:162): when set,
    /// format mismatches are hard errors instead of auto scale insertion.
    pub disable_auto_convert: bool,
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterGraph {
    /// `avfilter_graph_alloc` (avfiltergraph.c:85-99). With threads and the
    /// framequeue global dropped, construction is field defaults; the
    /// swscale defaults mirror C's (bicubic algorithm; engine choice is our
    /// extension — `Auto`, matching `ScaleOptions::default()`).
    pub fn new() -> Self {
        FilterGraph {
            nodes: Vec::new(),
            links: Vec::new(),
            fmt_lists: Vec::new(),
            csp_lists: Vec::new(),
            rng_lists: Vec::new(),
            scale_sws_opts: String::new(),
            scale_engine: ScaleEngine::default(),
            scale_algorithm: ScaleAlgorithm::default(),
            disable_auto_convert: false,
        }
    }

    // -----------------------------------------------------------------------
    // Node lifecycle (avfiltergraph.c:140-201, avfilter.c:701-986)
    // -----------------------------------------------------------------------

    /// `avfilter_graph_alloc_filter` (avfiltergraph.c:167-201) +
    /// `ff_filter_alloc` (avfilter.c:701-798): allocate an UNINITIALIZED
    /// instance from the registry. Thread init (174-184) dropped.
    pub fn alloc_filter(&mut self, name: &str) -> Result<NodeId> {
        let def: &'static FilterDef = super::filter_def(name).ok_or_else(|| {
            Error::NotFound(format!("No such filter: '{name}'"))
        })?;
        self.nodes.push(FilterNode {
            def,
            name: def.name.to_string(),
            inputs: vec![None; def.inputs.len()],
            outputs: vec![None; def.outputs.len()],
            imp: Some((def.make)()),
            ready: 0,
            initialized: false,
            opts: Options::default(),
        });
        Ok(NodeId(self.nodes.len() - 1))
    }

    /// `avfilter_graph_create_filter` (avfiltergraph.c:140-159) = alloc +
    /// `avfilter_init_str` (parse + init + leftover check).
    pub fn create_filter(&mut self, name: &str, args: &str) -> Result<NodeId> {
        let node = self.alloc_filter(name)?;
        let def = self.nodes[node.0].def;
        let opts = parse_args_simple(args, def)?;
        self.init_filter(node, opts)?;
        Ok(node)
    }

    /// `avfilter_init_dict` (avfilter.c:919-986): apply the option dict to
    /// the filter's impl, run its init, mark INITIALIZED (the gate for
    /// `link()`). Slice-thread wiring (935-942) and enable strings (949-953)
    /// dropped.
    ///
    /// The dict reaches the impl through `nodes[node].opts` (an extension of
    /// the C struct — there the dict is a function argument; the parser
    /// module owns the full lifecycle once it lands). Leftover entries after
    /// a successful init are the caller's "No such option" error, exactly
    /// like `avfilter_init_str` (avfilter.c:976-980) — checked here so both
    /// call shapes get it.
    pub fn init_filter(&mut self, node: NodeId, opts: Options) -> Result<()> {
        if self.nodes[node.0].initialized {
            return Err(Error::InvalidArgument(format!(
                "filter '{}' already initialized",
                self.nodes[node.0].name
            )));
        }
        self.nodes[node.0].opts = opts;
        let mut imp = self.nodes[node.0]
            .imp
            .take()
            .expect("init_filter: node imp already taken");
        let ret = imp.init(self, node);
        debug_assert!(
            self.nodes[node.0].imp.is_none(),
            "init: imp was restored while taken"
        );
        self.nodes[node.0].imp = Some(imp);
        ret?;
        self.nodes[node.0].initialized = true;
        // avfilter_init_str's leftover check (avfilter.c:976-980): the filter
        // IS initialized at this point in C too; the caller discards it.
        if let Some((key, _)) = self.nodes[node.0].opts.entries.first() {
            return Err(Error::NotFound(format!("No such option: {key}")));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Linking (avfilter.c:149-196, 282-326)
    // -----------------------------------------------------------------------

    /// `avfilter_link` (avfilter.c:149-196): connect src's output pad to
    /// dst's input pad. Pad-range/occupied/initialized checks per C; the
    /// media-type check (169-175) is dropped — the port is video-only.
    pub fn link(
        &mut self,
        src: NodeId,
        srcpad: usize,
        dst: NodeId,
        dstpad: usize,
    ) -> Result<LinkId> {
        let (src_out_len, src_taken) = {
            let n = &self.nodes[src.0];
            (n.outputs.len(), n.outputs.get(srcpad).copied().flatten())
        };
        let (dst_in_len, dst_taken) = {
            let n = &self.nodes[dst.0];
            (n.inputs.len(), n.inputs.get(dstpad).copied().flatten())
        };
        if srcpad >= src_out_len || dstpad >= dst_in_len || src_taken.is_some() || dst_taken.is_some()
        {
            return Err(Error::InvalidArgument(format!(
                "cannot link {}:{} -> {}:{} (pad out of range or already linked)",
                self.nodes[src.0].name, srcpad, self.nodes[dst.0].name, dstpad
            )));
        }
        if !self.nodes[src.0].initialized || !self.nodes[dst.0].initialized {
            log_error!(
                Some("graph"),
                "Filters must be initialized before linking.\n"
            );
            return Err(Error::InvalidArgument(
                "Filters must be initialized before linking.".into(),
            ));
        }

        let mut link = Link::default();
        link.src = src;
        link.srcpad = srcpad;
        link.dst = dst;
        link.dstpad = dstpad;
        self.links.push(link);
        let lid = LinkId(self.links.len() - 1);
        self.nodes[src.0].outputs[srcpad] = Some(lid);
        self.nodes[dst.0].inputs[dstpad] = Some(lid);
        Ok(lid)
    }

    /// `avfilter_insert_filter` (avfilter.c:282-326): splice `filt` (pads
    /// 0/0 — conversion filters are single-pad) into an existing link.
    ///
    /// The **outcfg half-move** (306-323, `ff_formats_changeref`): per axis,
    /// the OLD link's outcfg ListIdx (the downstream filter's declarations)
    /// moves into the NEW (filt→old_dst) link's outcfg, leaving the old
    /// link's half None — this keeps the downstream constraints attached to
    /// the downstream link so the post-insertion settle-merge has both
    /// halves. Losing this breaks `format=rgb24` graphs.
    pub fn insert_filter(&mut self, link: LinkId, filt: NodeId) -> Result<()> {
        let (old_dst, dstpad_idx, src_name, filt_name, dst_name) = {
            let l = &self.links[link.0];
            (
                l.dst,
                l.dstpad,
                self.nodes[l.src.0].name.clone(),
                self.nodes[filt.0].name.clone(),
                self.nodes[l.dst.0].name.clone(),
            )
        };
        log_verbose!(
            Some("graph"),
            "auto-inserting filter '{filt_name}' between the filter '{src_name}' and the filter '{dst_name}'\n"
        );

        self.nodes[old_dst.0].inputs[dstpad_idx] = None;
        let newlink = match self.link(filt, 0, old_dst, dstpad_idx) {
            Ok(nl) => nl,
            Err(e) => {
                // failed to link the inserted filter's output — restore
                self.nodes[old_dst.0].inputs[dstpad_idx] = Some(link);
                return Err(e);
            }
        };

        // Re-hook the original link to the inserted filter.
        self.links[link.0].dst = filt;
        self.links[link.0].dstpad = 0;
        self.nodes[filt.0].inputs[0] = Some(link);

        // Preserve already-declared dst-side formats: outcfg half-move.
        for axis in [Axis::Formats, Axis::ColorSpaces, Axis::ColorRanges] {
            if let Some(x) = self.links[link.0].outcfg.slot_mut(axis).take() {
                *self.links[newlink.0].outcfg.slot_mut(axis) = Some(x);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Configuration (avfiltergraph.c:203-285, 1434-1449) — WAVE 1 STUB
    // -----------------------------------------------------------------------

    /// `avfilter_graph_config` (avfiltergraph.c:1434-1449) — in C the strict
    /// order validity → formats → links → check → pointers. Wave 1 ports the
    /// validity stage only; the formats negotiation round, link config and
    /// size checks arrive with wave 2 (this method grows them in order).
    pub fn config(&mut self) -> Result<()> {
        self.check_validity()
    }

    /// `graph_check_validity` (avfiltergraph.c:203-240): every pad of every
    /// filter must be connected. Error texts per C (media type string is
    /// always "video" in this port).
    pub(crate) fn check_validity(&self) -> Result<()> {
        for (i, filt) in self.nodes.iter().enumerate() {
            let node = NodeId(i);
            for (j, pad) in filt.def.inputs.iter().enumerate() {
                let connected = filt
                    .inputs
                    .get(j)
                    .copied()
                    .flatten()
                    .is_some_and(|l| self.links[l.0].src == node);
                if !connected {
                    let msg = format!(
                        "Input pad \"{}\" with type video of the filter instance \"{}\" of {} not connected to any source",
                        pad.name, filt.name, filt.def.name
                    );
                    log_error!(None, "{msg}\n");
                    return Err(Error::InvalidArgument(msg));
                }
            }
            for (j, pad) in filt.def.outputs.iter().enumerate() {
                let connected = filt
                    .outputs
                    .get(j)
                    .copied()
                    .flatten()
                    .is_some_and(|l| self.links[l.0].dst == node);
                if !connected {
                    let msg = format!(
                        "Output pad \"{}\" with type video of the filter instance \"{}\" of {} not connected to any destination",
                        pad.name, filt.name, filt.def.name
                    );
                    log_error!(None, "{msg}\n");
                    return Err(Error::InvalidArgument(msg));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Negotiation list plumbing (formats.c:481-601, 837-860, 1171-1232)
    // -----------------------------------------------------------------------

    /// `ff_make_format_list` / `ff_add_format` — push a fresh list into the
    /// arena, return its handle.
    pub fn alloc_pix_list(&mut self, fmts: Vec<PixelFormat>) -> ListIdx {
        self.fmt_lists.push(fmts);
        (self.fmt_lists.len() - 1) as ListIdx
    }

    /// Color-space list arena.
    pub fn alloc_csp_list(&mut self, csps: Vec<ColorSpace>) -> ListIdx {
        self.csp_lists.push(csps);
        (self.csp_lists.len() - 1) as ListIdx
    }

    /// Color-range list arena.
    pub fn alloc_rng_list(&mut self, ranges: Vec<ColorRange>) -> ListIdx {
        self.rng_lists.push(ranges);
        (self.rng_lists.len() - 1) as ListIdx
    }

    /// The universe list — `ff_all_formats(VIDEO)`; see
    /// `formats::all_pix_fmts` for the narrower-than-C caveat.
    pub fn all_pix_list(&mut self) -> ListIdx {
        self.alloc_pix_list(formats::all_pix_fmts())
    }

    /// `ff_set_common_formats` (formats.c:837-860): hand the SAME list
    /// handle to every still-unset pad of the filter — inputs' **outcfg** +
    /// outputs' **incfg** (the crossed sides; see `link::FormatsConfig`).
    /// First query wins: an already-set pad is never overwritten. Identity
    /// sharing is the point — one merge collapses all the filter's pads.
    pub fn set_common_formats(&mut self, node: NodeId, list: ListIdx) -> Result<()> {
        let inputs = self.nodes[node.0].inputs.clone();
        for l in inputs.iter().flatten() {
            if self.links[l.0].outcfg.formats.is_none() {
                self.links[l.0].outcfg.formats = Some(list);
            }
        }
        let outputs = self.nodes[node.0].outputs.clone();
        for l in outputs.iter().flatten() {
            if self.links[l.0].incfg.formats.is_none() {
                self.links[l.0].incfg.formats = Some(list);
            }
        }
        Ok(())
    }

    /// `ff_set_common_color_spaces` — same crossing on the csp axis.
    pub fn set_common_color_spaces(&mut self, node: NodeId, list: ListIdx) -> Result<()> {
        let inputs = self.nodes[node.0].inputs.clone();
        for l in inputs.iter().flatten() {
            if self.links[l.0].outcfg.color_spaces.is_none() {
                self.links[l.0].outcfg.color_spaces = Some(list);
            }
        }
        let outputs = self.nodes[node.0].outputs.clone();
        for l in outputs.iter().flatten() {
            if self.links[l.0].incfg.color_spaces.is_none() {
                self.links[l.0].incfg.color_spaces = Some(list);
            }
        }
        Ok(())
    }

    /// `ff_set_common_color_ranges` — same crossing on the range axis.
    pub fn set_common_color_ranges(&mut self, node: NodeId, list: ListIdx) -> Result<()> {
        let inputs = self.nodes[node.0].inputs.clone();
        for l in inputs.iter().flatten() {
            if self.links[l.0].outcfg.color_ranges.is_none() {
                self.links[l.0].outcfg.color_ranges = Some(list);
            }
        }
        let outputs = self.nodes[node.0].outputs.clone();
        for l in outputs.iter().flatten() {
            if self.links[l.0].incfg.color_ranges.is_none() {
                self.links[l.0].incfg.color_ranges = Some(list);
            }
        }
        Ok(())
    }

    /// `ff_default_query_formats` (formats.c:1171-1232) — the fill-everything
    /// default the engine runs AFTER every filter's `query_formats`
    /// (avfiltergraph.c:410): all-lists into every still-unset slot
    /// (fill-if-unset). For vf_null (PASSTHROUGH) this is the only query.
    pub fn default_query_formats(&mut self, node: NodeId) -> Result<()> {
        let pix = self.all_pix_list();
        self.set_common_formats(node, pix)?;
        let csp = self.alloc_csp_list(formats::all_color_spaces());
        self.set_common_color_spaces(node, csp)?;
        let rng = self.alloc_rng_list(formats::all_color_ranges());
        self.set_common_color_ranges(node, rng)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Runtime API (buffersrc/buffersink wrappers) — WAVE 2
    // -----------------------------------------------------------------------

    /// `av_buffersrc_add_frame_flags(KEEP_REF)` — push one decoded frame into
    /// the source. **Wave 2** (buffersrc.c:220-287): param-change checks,
    /// KEEP_REF clone, watchdog. Stub.
    pub fn add_frame(&mut self, src: NodeId, frame: &Frame) -> Result<()> {
        let _ = (src, frame);
        Err(Error::Unsupported(
            "FilterGraph::add_frame arrives with the buffersrc module (wave 2)".into(),
        ))
    }

    /// `av_buffersrc_close` — announce EOF on the source. **Wave 2**. Stub.
    pub fn close_source(&mut self, src: NodeId) -> Result<()> {
        let _ = src;
        Err(Error::Unsupported(
            "FilterGraph::close_source arrives with the buffersrc module (wave 2)".into(),
        ))
    }

    /// `av_buffersink_get_frame` — pull one frame from the sink.
    /// `Err(Again)` = starved, `Err(Eof)` = drained. **Wave 2** (the
    /// `get_frame_internal` loop, buffersink.c:93-130). Stub.
    pub fn get_frame(&mut self, sink: NodeId) -> Result<Frame> {
        let _ = sink;
        Err(Error::Unsupported(
            "FilterGraph::get_frame arrives with the buffersink module (wave 2)".into(),
        ))
    }

    /// `avfilter_graph_parse_ptr` — parse a graph description into open pad
    /// lists. **Wave 2** (graphparser.c). Stub.
    pub fn parse_ptr(&mut self, desc: &str) -> Result<(Vec<InOut>, Vec<InOut>)> {
        let _ = desc;
        Err(Error::Unsupported(
            "graph parsing arrives with the parser module (wave 2)".into(),
        ))
    }

    // -----------------------------------------------------------------------
    // Accessors used by filter impls
    // -----------------------------------------------------------------------

    /// The link on input pad `pad` (panics while unlinked — filters read
    /// these only after the graph is wired).
    pub fn inlink(&self, node: NodeId, pad: usize) -> LinkId {
        self.nodes[node.0].inputs[pad]
            .expect("input pad not linked (filters read links only post-config)")
    }

    /// The link on output pad `pad`.
    pub fn outlink(&self, node: NodeId, pad: usize) -> LinkId {
        self.nodes[node.0].outputs[pad]
            .expect("output pad not linked (filters read links only post-config)")
    }
}

/// Wave-1 stand-in for `ff_filter_opt_parse` (avfilter.c:853-902): split an
/// args string into ordered `(key, value)` pairs on ':' / '=', assigning
/// positional values to `def.shorthand` slots in order; an explicit
/// `key=value` disables all remaining positional slots (avfilter.c:889-892).
/// No quoting/escaping — the parser module (wave 2) replaces this wholesale
/// with the real `av_get_token`-based parser.
fn parse_args_simple(args: &str, def: &FilterDef) -> Result<Options> {
    let mut opts = Options::default();
    if args.is_empty() {
        return Ok(opts);
    }
    let mut shorthand_pos = 0usize;
    let mut explicit_seen = false;
    for tok in args.split(':') {
        if let Some(eq) = tok.find('=') {
            explicit_seen = true;
            opts.entries
                .push((tok[..eq].to_string(), tok[eq + 1..].to_string()));
        } else {
            if explicit_seen || shorthand_pos >= def.shorthand.len() {
                return Err(Error::InvalidArgument(format!(
                    "No option name near '{tok}'"
                )));
            }
            opts.entries
                .push((def.shorthand[shorthand_pos].to_string(), tok.to_string()));
            shorthand_pos += 1;
        }
    }
    Ok(opts)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::graph::engine_test_helpers::*;
    use crate::filter::filter;

    fn null_pair() -> (FilterGraph, NodeId, NodeId, LinkId) {
        let mut g = FilterGraph::new();
        let a = g.create_filter("null", "").unwrap();
        let b = g.create_filter("null", "").unwrap();
        let l = g.link(a, 0, b, 0).unwrap();
        (g, a, b, l)
    }

    #[test]
    fn link_enforces_range_occupancy_and_init() {
        let (mut g, a, b, _l) = null_pair();
        // pad out of range
        assert!(g.link(a, 1, b, 0).is_err());
        // pad occupied
        assert!(g.link(a, 0, b, 0).is_err());
        // uninitialized filter refused (avfilter.c:163-167) — with both pads
        // free so the earlier checks pass and the init gate is what fires.
        let c = g.alloc_filter("null").unwrap();
        let d = g.create_filter("null", "").unwrap();
        assert!(g.link(c, 0, d, 0).is_err());
        g.init_filter(c, Options::default()).unwrap();
        assert!(g.link(c, 0, d, 0).is_ok());
    }

    #[test]
    fn create_filter_unknown_name_and_option() {
        let mut g = FilterGraph::new();
        assert!(matches!(
            g.create_filter("nosuchfilter", ""),
            Err(Error::NotFound(_))
        ));
        assert_eq!(g.nodes.len(), 0);
        // null has no options and no shorthand → any pair is a leftover.
        assert!(matches!(
            g.create_filter("null", "w=800"),
            Err(Error::NotFound(_))
        ));
        // the node is still allocated and initialized (C semantics: the
        // leftover is found after init), the CALLER discards the graph.
        assert_eq!(g.nodes.len(), 1);
        // malformed: positional with no shorthand slot available.
        assert!(matches!(
            g.create_filter("null", "800"),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn check_validity_texts() {
        let (mut g, a, b, _l) = null_pair();
        // a's input and b's output are open pads: must fail.
        let err = g.check_validity().unwrap_err();
        assert!(err.to_string().contains("not connected to any source"));
        // close the chain: c -> a -> b -> d
        let c = g.create_filter("null", "").unwrap();
        let d = g.create_filter("null", "").unwrap();
        g.link(c, 0, a, 0).unwrap();
        g.link(b, 0, d, 0).unwrap();
        g.check_validity().unwrap();
    }

    #[test]
    fn set_common_fill_if_unset_and_identity_sharing() {
        let (mut g, a, _b, l) = null_pair();
        // An explicitly declared half survives the default query.
        let explicit = g.alloc_pix_list(vec![PixelFormat::Yuv420p]);
        g.links[l.0].outcfg.formats = Some(explicit);
        g.default_query_formats(a).unwrap();
        // fill-if-unset: the explicit list was NOT overwritten...
        assert_eq!(g.links[l.0].outcfg.formats, Some(explicit));
        // ...the unset half got the ALL list...
        let all_idx = g.links[l.0].incfg.formats.expect("filled by default query");
        assert_eq!(g.fmt_lists[all_idx as usize].as_slice(), PixelFormat::ALL);
        // ...and csp/range axes are filled too (same crossing).
        assert!(g.links[l.0].incfg.color_spaces.is_some());
        assert!(g.links[l.0].outcfg.color_spaces.is_some());
        assert!(g.links[l.0].incfg.color_ranges.is_some());
    }

    #[test]
    fn insert_filter_rewires_and_moves_outcfg() {
        let (mut g, a, b, l) = null_pair();
        // Downstream declarations on the old link's outcfg (what negotiation
        // would have put there)...
        let fmts = g.alloc_pix_list(vec![PixelFormat::Rgb24]);
        let csps = g.alloc_csp_list(vec![ColorSpace::Bt709]);
        g.links[l.0].outcfg.formats = Some(fmts);
        g.links[l.0].outcfg.color_spaces = Some(csps);
        // ...splice a third filter in between.
        let c = g.create_filter("null", "").unwrap();
        g.insert_filter(l, c).unwrap();
        // Old link now ends at the inserted filter; a NEW link c->b exists.
        assert_eq!(g.links[l.0].dst, c);
        assert_eq!(g.nodes[c.0].inputs[0], Some(l));
        let newl = g.nodes[c.0].outputs[0].expect("new link created");
        assert_eq!(g.links[newl.0].dst, b);
        assert_eq!(g.nodes[b.0].inputs[0], Some(newl));
        // The outcfg halves MOVED to the new link (ff_formats_changeref);
        // the old link's halves are None so the inserted filter's query
        // fills them.
        assert_eq!(g.links[newl.0].outcfg.formats, Some(fmts));
        assert_eq!(g.links[newl.0].outcfg.color_spaces, Some(csps));
        assert_eq!(g.links[l.0].outcfg.formats, None);
        assert_eq!(g.links[l.0].outcfg.color_spaces, None);
    }

    /// THE engine proof: src → null → null → sink, one frame through, then
    /// EOF propagation end to end — before buffersrc/buffersink exist.
    #[test]
    fn engine_end_to_end_frame_then_eof() {
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let n0 = g.create_filter("null", "").unwrap();
        let n1 = g.create_filter("null", "").unwrap();
        let sink = g.alloc_test_sink();
        let la = g.link(src, 0, n0, 0).unwrap();
        let lb = g.link(n0, 0, n1, 0).unwrap();
        let lc = g.link(n1, 0, sink, 0).unwrap();

        // Wave-1 config stub: validity only (formats/links arrive wave 2).
        g.config().unwrap();

        // Push a frame through the source's output link (what buffersrc's
        // add_frame will do internally).
        let mut frame = Frame::alloc(PixelFormat::Gray8, 8, 8).unwrap();
        frame.pts = 5;
        filter::filter_frame(&mut g, la, frame).unwrap();
        assert_eq!(g.links[la.0].fifo.len(), 1);
        assert_eq!(g.nodes[n0.0].ready, 300);

        // Run to quiescence: both nulls forward, the sink consumes.
        run_to_quiescence(&mut g);
        assert_eq!(g.links[lc.0].fifo.len(), 0, "sink consumed the frame");
        for (name, l) in [("la", la), ("lb", lb), ("lc", lc)] {
            assert_eq!(g.links[l.0].frame_count_in, 1, "{name} in");
            assert_eq!(g.links[l.0].frame_count_out, 1, "{name} out");
        }

        // EOF: the source announces it on its output link (what
        // buffersrc_close does); the status must propagate to the sink-side
        // link after everything drains.
        filter::set_in_status(&mut g, la, Error::Eof, 100);
        run_to_quiescence(&mut g);
        assert!(
            matches!(g.links[lc.0].status_in, Some(Error::Eof)),
            "EOF must reach the sink link"
        );
        for (name, l) in [("la", la), ("lb", lb), ("lc", lc)] {
            assert!(g.links[l.0].fifo.is_empty(), "{name} drained");
        }
        // And the graph is quiescent afterwards.
        assert!(matches!(g.run_once(), Err(Error::Again)));
    }
}

/// Hand-rolled endpoint filters for the engine tests — buffersrc/buffersink
/// arrive in wave 2; these provide the minimum a source/sink must do.
#[cfg(test)]
pub(crate) mod engine_test_helpers {
    use super::super::filter::{FilterDef, FilterImpl, FilterFlags, FilterNode, PadDef};
    use super::super::link::NodeId;
    use super::*;

    /// A source that produces nothing itself (the test pushes frames into
    /// its output link directly); activate = the BUFFERSRC_EMPTY shape
    /// (Ok(()) with nothing to do — see filter.rs `run_once` doc).
    struct TestSrc;
    impl FilterImpl for TestSrc {
        fn filter_frame(
            &mut self,
            _g: &mut FilterGraph,
            _node: NodeId,
            _pad: usize,
            _frame: Frame,
        ) -> Result<()> {
            Err(Error::Unsupported("test src has no input".into()))
        }
        fn activate(&mut self, _g: &mut FilterGraph, _node: NodeId) -> Result<()> {
            Ok(()) // nothing queued upstream: would-be BUFFERSRC_EMPTY
        }
    }

    /// A sink that consumes (drops) frames.
    struct TestSink;
    impl FilterImpl for TestSink {
        fn filter_frame(
            &mut self,
            _g: &mut FilterGraph,
            _node: NodeId,
            _pad: usize,
            _frame: Frame,
        ) -> Result<()> {
            Ok(())
        }
    }

    const PAD: PadDef = PadDef { name: "default", needs_writable: false };

    static TEST_SRC_DEF: FilterDef = FilterDef {
        name: "testsrc_impl",
        inputs: &[],
        outputs: &[PAD],
        flags: FilterFlags(0),
        shorthand: &[],
        make: || Box::new(TestSrc),
    };

    static TEST_SINK_DEF: FilterDef = FilterDef {
        name: "testsink_impl",
        inputs: &[PAD],
        outputs: &[],
        flags: FilterFlags(0),
        shorthand: &[],
        make: || Box::new(TestSink),
    };

    impl FilterGraph {
        pub(crate) fn alloc_test_src(&mut self) -> NodeId {
            self.push_bare(&TEST_SRC_DEF)
        }
        pub(crate) fn alloc_test_sink(&mut self) -> NodeId {
            self.push_bare(&TEST_SINK_DEF)
        }
        fn push_bare(&mut self, def: &'static FilterDef) -> NodeId {
            self.nodes.push(FilterNode {
                def,
                name: def.name.to_string(),
                inputs: vec![None; def.inputs.len()],
                outputs: vec![None; def.outputs.len()],
                imp: Some((def.make)()),
                ready: 0,
                initialized: true,
                opts: Options::default(),
            });
            NodeId(self.nodes.len() - 1)
        }
    }

    /// Drive `run_once` until it reports idle, panicking on real errors.
    pub(crate) fn run_to_quiescence(g: &mut FilterGraph) {
        for _ in 0..1000 {
            match g.run_once() {
                Ok(()) => {}
                Err(Error::Again) => return,
                Err(e) => panic!("run_once failed: {e}"),
            }
        }
        panic!("graph did not reach quiescence in 1000 activations");
    }
}
