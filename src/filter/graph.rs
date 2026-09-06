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

use crate::{
    log_error, log_verbose,
    swscale::{ScaleAlgorithm, ScaleEngine},
    util::{
        color::{ColorRange, ColorSpace},
        error::{Error, Result},
        frame::Frame,
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::{
    Options,
    filter::{FilterDef, FilterNode, PadRef},
    formats,
    link::{Link, LinkId, LinkInitState, ListIdx, NodeId},
};

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

/// One typed list entry — `REDUCE_FORMATS`'s `fmt_type fmt` generic, which
/// C instantiates per axis (avfiltergraph.c:943-978).
#[derive(Clone, Copy, Debug)]
enum ListValue {
    Pix(PixelFormat),
    Csp(ColorSpace),
    Rng(ColorRange),
}

/// `REDUCE_FORMATS`'s per-axis body (avfiltergraph.c:962-974): collapse
/// `list` onto the singleton when it contains it; an empty list BECOMES the
/// singleton (C's `ff_add_format` onto an empty list). Returns whether a
/// reduction happened; a no-op for lists already of length 1.
fn reduce_in_place<T: PartialEq + Copy>(list: &mut Vec<T>, fmt: T) -> bool {
    if list.len() == 1 {
        return false; // C: `out_link->incfg.list->nb == 1` skips
    }
    if list.is_empty() {
        list.push(fmt);
        return true;
    }
    match list.iter().position(|f| *f == fmt) {
        Some(idx) => {
            list.swap(0, idx);
            list.truncate(1);
            true
        }
        None => false,
    }
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
        let def: &'static FilterDef = super::filter_def(name)
            .ok_or_else(|| Error::NotFound(format!("No such filter: '{name}'")))?;
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
        let opts = super::parser::filter_opt_parse(args, def)?;
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
        if srcpad >= src_out_len
            || dstpad >= dst_in_len
            || src_taken.is_some()
            || dst_taken.is_some()
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
        // `avfilter_graph_config` order (avfiltergraph.c:1426-1452);
        // `graph_config_pointers` (sink-link age heap) is not ported.
        self.check_validity()?;
        // graph_config_formats (1366-1398): loop while the round reports
        // progress-pending; any OTHER error is fatal (C returns it).
        let mut ret = self.query_formats_round();
        while matches!(ret, Err(Error::Again)) {
            log_verbose!(None, "query_formats not finished\n"); // avfiltergraph.c:1374
            ret = self.query_formats_round();
        }
        ret?;
        self.graph_config_formats_tail()?;
        self.graph_config_links()?;
        self.graph_check_links()
    }

    // -----------------------------------------------------------------------
    // Format negotiation (avfiltergraph.c:522-737, 943-1060, 1311-1400)
    // -----------------------------------------------------------------------

    /// `query_formats` (avfiltergraph.c:522-737) — ONE round: query every
    /// under-declared filter, then merge lists link by link, auto-inserting
    /// `scale` converters where halves cannot merge. `Err(Again)` = progress
    /// was made, caller re-runs; the stuck case errors with C's text.
    pub(crate) fn query_formats_round(&mut self) -> Result<()> {
        // (527-547) query every filter whose declarations are incomplete.
        let mut count_queried = 0;
        let mut count_merged = 0;
        let mut count_already_merged = 0;
        let mut count_delayed = 0;
        let mut converter_count = 0usize;
        for i in 0..self.nodes.len() {
            let node = NodeId(i);
            if self.formats_declared(node) {
                continue;
            }
            match self.query_formats_filter(node) {
                // EAGAIN may indicate partial success — not counted yet.
                Err(Error::Again) => continue,
                other => other?,
            }
            count_queried += 1;
        }

        // (549-709) merge as many lists as possible; `retry:` re-enters here
        // after converter insertion.
        'retry: loop {
            for i in 0..self.nodes.len() {
                let filter = NodeId(i);
                let inputs = self.nodes[filter.0].inputs.clone();
                for link in inputs.iter().flatten() {
                    // Pass 1 (565-584): an axis whose halves cannot merge
                    // registers a converter. Every video merger's conversion
                    // filter is "scale" (formats.c:384-419), so at most one
                    // converter is ever registered per link (C dedupes by
                    // name); alpha modes ("premultiply_dynamic") dropped.
                    let mut need_conv = false;
                    for axis in [Axis::Formats, Axis::ColorSpaces, Axis::ColorRanges] {
                        let (a, b) = self.link_halves(*link, axis);
                        if let (Some(a), Some(b)) = (a, b) {
                            if a != b && !self.can_merge_axis(axis, a, b) {
                                need_conv = true;
                            }
                        }
                    }
                    // Pass 2 (586-605): classify each axis.
                    for axis in [Axis::Formats, Axis::ColorSpaces, Axis::ColorRanges] {
                        let (a, b) = self.link_halves(*link, axis);
                        match (a, b) {
                            (None, _) | (_, None) => count_delayed += 1,
                            (Some(a), Some(b)) if a == b => count_already_merged += 1,
                            (Some(a), Some(b)) if !need_conv => {
                                count_merged += 1;
                                if !self.merge_axis(axis, a, b) {
                                    need_conv = true;
                                }
                            }
                            _ => {} // mergeable but a converter is pending
                        }
                    }

                    if !need_conv {
                        continue;
                    }
                    // (611-621) automatic conversion disabled → hard error.
                    if self.disable_auto_convert {
                        let (src_name, dst_name) = self.link_endpoints(*link);
                        log_error!(
                            None,
                            "The filters '{src_name}' and '{dst_name}' do not have a common format and automatic conversion is disabled.\n"
                        );
                        return Err(Error::InvalidArgument(
                            "filters do not have a common format and automatic conversion is disabled".into(),
                        ));
                    }
                    // (622-641) auto-insert the converter.
                    if super::filter_def("scale").is_none() {
                        log_error!(
                            None,
                            "'scale' filter not present, cannot convert formats.\n"
                        );
                        return Err(Error::NotFound(
                            "'scale' filter not present, cannot convert formats.".into(),
                        ));
                    }
                    let inst_name = format!("auto_scale_{converter_count}");
                    converter_count += 1;
                    // C: `avfilter_graph_create_filter(.., inst_name, conv_opts..)`
                    // — `scale_sws_opts` is not ported (empty options); the
                    // node keeps C's `auto_scale_N` instance name.
                    let conv = self.create_filter("scale", "")?;
                    self.nodes[conv.0].name = inst_name;
                    self.insert_filter(*link, conv)?;
                    self.query_formats_filter(conv)?;

                    // (649-701) preemptively settle the converter's own
                    // links on every scale-bearing axis.
                    let inlink =
                        self.nodes[conv.0].inputs[0].expect("inserted converter has an input link");
                    let outlink = self.nodes[conv.0].outputs[0]
                        .expect("inserted converter has an output link");
                    let (src_name, dst_name) = {
                        let l = &self.links[link.0];
                        (
                            self.nodes[l.src.0].name.clone(),
                            self.nodes[l.dst.0].name.clone(),
                        )
                    };
                    for axis in [Axis::Formats, Axis::ColorSpaces, Axis::ColorRanges] {
                        for l in [inlink, outlink] {
                            let (a, b) = self.link_halves(l, axis);
                            match (a, b) {
                                (Some(a), Some(b)) if self.merge_axis(axis, a, b) => {
                                    count_merged += 1;
                                }
                                _ => {
                                    log_error!(
                                        None,
                                        "Impossible to convert between the formats supported by the filter '{}' and the filter '{}'\n",
                                        src_name,
                                        dst_name
                                    );
                                    return Err(Error::Unsupported(
                                        "Impossible to convert between the formats supported by the filter".into(),
                                    ));
                                }
                            }
                        }
                    }
                    // (704-706) converters may cross-interact — start over.
                    continue 'retry;
                }
            }
            break;
        }

        // (709-712) C's debug summary — kept as a verbose line.
        log_verbose!(
            None,
            "query_formats: {count_queried} queried, {count_merged} merged, {count_already_merged} already done, {count_delayed} delayed\n"
        );
        // (714-736) delayed halves with no progress = stuck negotiation.
        if count_delayed > 0 {
            if count_queried > 0 || count_merged > 0 {
                return Err(Error::Again);
            }
            let mut names = String::new();
            for i in 0..self.nodes.len() {
                let node = NodeId(i);
                if !self.formats_declared(node) {
                    if !names.is_empty() {
                        names += ", ";
                    }
                    names += &self.nodes[i].name;
                }
            }
            log_error!(
                None,
                "The following filters could not choose their formats: {names}\nConsider inserting the (a)format filter near their input or output.\n"
            );
            return Err(Error::InvalidData(
                "The following filters could not choose their formats".into(),
            ));
        }
        Ok(())
    }

    /// `filter_query_formats` (avfiltergraph.c:336-410): the filter's own
    /// query, then the always-run default fill. `Err(Again)` from the impl
    /// short-circuits BEFORE the fill (C returns early at 395). After a
    /// successful own query, declared lists are checked for redundancy
    /// (`filter_check_formats`, avfiltergraph.c:329-345).
    fn query_formats_filter(&mut self, node: NodeId) -> Result<()> {
        let mut imp = self.nodes[node.0]
            .imp
            .take()
            .expect("query_formats_filter: imp already taken");
        let ret = imp.query_formats(self, node);
        self.nodes[node.0].imp = Some(imp);
        ret?;
        self.filter_check_formats(node)?;
        self.default_query_formats(node)
    }

    /// `filter_check_formats` + `check_list` (avfiltergraph.c:296-345,
    /// formats.c:1236-1252): every list the filter declared (inputs'
    /// `outcfg` + outputs' `incfg`) must be duplicate-free — "Duplicated
    /// pixel format\n" & friends.
    fn filter_check_formats(&self, node: NodeId) -> Result<()> {
        let mut halves = Vec::new();
        for l in self.nodes[node.0].inputs.iter().flatten() {
            halves.push((*l, false)); // false = outcfg side
        }
        for l in self.nodes[node.0].outputs.iter().flatten() {
            halves.push((*l, true)); // true = incfg side
        }
        for (l, incfg) in halves {
            let cfg = if incfg {
                &self.links[l.0].incfg
            } else {
                &self.links[l.0].outcfg
            };
            if let Some(idx) = cfg.formats {
                self.check_list("pixel format", &self.fmt_lists[idx as usize])?;
            }
            if let Some(idx) = cfg.color_spaces {
                self.check_list("color space", &self.csp_lists[idx as usize])?;
            }
            if let Some(idx) = cfg.color_ranges {
                self.check_list("color range", &self.rng_lists[idx as usize])?;
            }
        }
        Ok(())
    }

    fn check_list<T: PartialEq>(&self, name: &str, list: &[T]) -> Result<()> {
        for i in 0..list.len() {
            if list[i + 1..].contains(&list[i]) {
                log_error!(None, "Duplicated {name}\n");
                return Err(Error::InvalidArgument(format!("Duplicated {name}")));
            }
        }
        Ok(())
    }

    /// `formats_declared` (avfiltergraph.c:413-449) — video halves only
    /// (audio axes and alpha modes are out of scope).
    fn formats_declared(&self, node: NodeId) -> bool {
        for l in self.nodes[node.0].inputs.iter().flatten() {
            let h = &self.links[l.0].outcfg;
            if h.formats.is_none() || h.color_spaces.is_none() || h.color_ranges.is_none() {
                return false;
            }
        }
        for l in self.nodes[node.0].outputs.iter().flatten() {
            let h = &self.links[l.0].incfg;
            if h.formats.is_none() || h.color_spaces.is_none() || h.color_ranges.is_none() {
                return false;
            }
        }
        true
    }

    /// The two negotiation halves of one axis on one link (`incfg`, `outcfg`
    /// — C reads them via the merger's struct offset).
    fn link_halves(&self, link: LinkId, axis: Axis) -> (Option<ListIdx>, Option<ListIdx>) {
        let l = &self.links[link.0];
        match axis {
            Axis::Formats => (l.incfg.formats, l.outcfg.formats),
            Axis::ColorSpaces => (l.incfg.color_spaces, l.outcfg.color_spaces),
            Axis::ColorRanges => (l.incfg.color_ranges, l.outcfg.color_ranges),
        }
    }

    fn can_merge_axis(&self, axis: Axis, a: ListIdx, b: ListIdx) -> bool {
        match axis {
            Axis::Formats => formats::can_merge_pix_fmts(self, a, b),
            Axis::ColorSpaces => formats::can_merge_csp(self, a, b),
            Axis::ColorRanges => formats::can_merge_rng(self, a, b),
        }
    }

    fn merge_axis(&mut self, axis: Axis, a: ListIdx, b: ListIdx) -> bool {
        match axis {
            Axis::Formats => formats::merge_pix_fmts(self, a, b),
            Axis::ColorSpaces => formats::merge_csp(self, a, b),
            Axis::ColorRanges => formats::merge_rng(self, a, b),
        }
    }

    /// The post-query tail of `graph_config_formats` (avfiltergraph.c:1379-1398):
    /// reduce, then pick. (`swap_sample_fmts`/`swap_samplerates`/
    /// `swap_channel_layouts`, 1057-1309, are audio-only.)
    fn graph_config_formats_tail(&mut self) -> Result<()> {
        self.reduce_formats()?;
        self.pick_formats()
    }

    /// `reduce_formats` (avfiltergraph.c:1040-1057): fixpoint loop over
    /// `reduce_formats_on_filter`.
    fn reduce_formats(&mut self) -> Result<()> {
        loop {
            let mut reduced = false;
            for i in 0..self.nodes.len() {
                if self.reduce_formats_on_filter(NodeId(i))? {
                    reduced = true;
                }
            }
            if !reduced {
                return Ok(());
            }
        }
    }

    /// `REDUCE_FORMATS` over the three video axes (avfiltergraph.c:943-978):
    /// an input's singleton `outcfg` list collapses each output's `incfg`
    /// list — empty becomes the singleton, non-singleton containing it
    /// collapses onto it. Lists are mutated IN PLACE in the arena (C mutates
    /// the shared object; every sharer sees the reduction).
    fn reduce_formats_on_filter(&mut self, filter: NodeId) -> Result<bool> {
        let mut ret = false;
        for axis in [Axis::Formats, Axis::ColorSpaces, Axis::ColorRanges] {
            let inputs = self.nodes[filter.0].inputs.clone();
            for l in inputs.iter().flatten() {
                let Some(singleton) = self.links[l.0].outcfg.slot(axis) else {
                    continue;
                };
                let fmt = match axis {
                    Axis::Formats => match self.fmt_lists[singleton as usize].as_slice() {
                        [f] => ListValue::Pix(*f),
                        _ => continue, // not a singleton
                    },
                    Axis::ColorSpaces => match self.csp_lists[singleton as usize].as_slice() {
                        [c] => ListValue::Csp(*c),
                        _ => continue,
                    },
                    Axis::ColorRanges => match self.rng_lists[singleton as usize].as_slice() {
                        [r] => ListValue::Rng(*r),
                        _ => continue,
                    },
                };
                let outputs = self.nodes[filter.0].outputs.clone();
                for o in outputs.iter().flatten() {
                    let Some(list_idx) = self.links[o.0].incfg.slot(axis) else {
                        continue;
                    };
                    let collapse = match (axis, fmt) {
                        (Axis::Formats, ListValue::Pix(f)) => {
                            reduce_in_place(&mut self.fmt_lists[list_idx as usize], f)
                        }
                        (Axis::ColorSpaces, ListValue::Csp(c)) => {
                            reduce_in_place(&mut self.csp_lists[list_idx as usize], c)
                        }
                        (Axis::ColorRanges, ListValue::Rng(r)) => {
                            reduce_in_place(&mut self.rng_lists[list_idx as usize], r)
                        }
                        _ => unreachable!("axis/value kind mismatch"),
                    };
                    if collapse {
                        ret = true;
                        break; // C breaks the output loop per axis/input
                    }
                }
            }
        }
        Ok(ret)
    }

    /// `pick_formats` (avfiltergraph.c:1311-1364): change-loop — a
    /// singleton list picks its link NOW, and a filter with a picked input
    /// picks its outputs against that reference; final sweep picks the rest.
    fn pick_formats(&mut self) -> Result<()> {
        loop {
            let mut change = false;
            for i in 0..self.nodes.len() {
                let filter = NodeId(i);
                let inputs = self.nodes[filter.0].inputs.clone();
                for l in inputs.iter().flatten() {
                    if self.incfg_singleton(*l) {
                        self.pick_format(*l, None)?;
                        change = true;
                    }
                }
                let outputs = self.nodes[filter.0].outputs.clone();
                for l in outputs.iter().flatten() {
                    if self.incfg_singleton(*l) {
                        self.pick_format(*l, None)?;
                        change = true;
                    }
                }
                // (1345-1352) forward the picked input format to unpicked
                // outputs via best-of-2.
                if let Some(&in0) = inputs.first().and_then(|l| l.as_ref()) {
                    if self.links[in0.0].format.is_some() {
                        for l in outputs.iter().flatten() {
                            if self.links[l.0].format.is_none() {
                                self.pick_format(*l, Some(in0))?;
                                change = true;
                            }
                        }
                    }
                }
            }
            if !change {
                break;
            }
        }
        // (1354-1362) final sweep: pick everything still unpicked.
        for i in 0..self.nodes.len() {
            let filter = NodeId(i);
            let inputs = self.nodes[filter.0].inputs.clone();
            for l in inputs.iter().flatten() {
                self.pick_format(*l, None)?;
            }
            let outputs = self.nodes[filter.0].outputs.clone();
            for l in outputs.iter().flatten() {
                self.pick_format(*l, None)?;
            }
        }
        Ok(())
    }

    /// incfg formats list exists and has exactly one entry (the C loop
    /// guards at avfiltergraph.c:1323/1333).
    fn incfg_singleton(&self, link: LinkId) -> bool {
        self.links[link.0]
            .incfg
            .formats
            .is_some_and(|l| self.fmt_lists[l as usize].len() == 1)
    }

    /// `pick_format` (avfiltergraph.c:793-941): choose the link's format
    /// (best-of-2 against `ref` when given), then color space/range, then
    /// drop every list half on the link.
    fn pick_format(&mut self, link: LinkId, refr: Option<LinkId>) -> Result<()> {
        let Some(list) = self.links[link.0].incfg.formats else {
            return Ok(()); // nothing declared — already final
        };
        if let Some(refl) = refr {
            // (801-813) fold best-of-2 over the whole list against the
            // reference format. has_alpha: C's FIXME keeps the
            // nb_components-even approximation.
            let ref_fmt = self.links[refl.0].format.expect("reference link picked");
            let has_alpha = crate::util::pixdesc::descriptor(ref_fmt).nb_components % 2 == 0;
            let mut best: Option<PixelFormat> = None;
            for &p in &self.fmt_lists[list as usize] {
                best = Some(formats::find_best_pix_fmt_of_2(best, p, ref_fmt, has_alpha));
            }
            if let Some(best) = best {
                self.fmt_lists[list as usize] = vec![best];
            }
        }
        let fmt = self.fmt_lists[list as usize][0];
        self.links[link.0].format = Some(fmt);

        // (843-891) color space / range selection.
        if !formats::regular_yuv(fmt) {
            // Explicitly YUV-only fields get sane values otherwise.
            let desc = crate::util::pixdesc::descriptor(fmt);
            self.links[link.0].color_range = if desc
                .flags
                .contains(crate::util::pixdesc::PixFmtFlags::FLOAT)
            {
                ColorRange::Unspecified
            } else {
                ColorRange::Jpeg
            };
            self.links[link.0].colorspace = if desc.flags.intersects(
                crate::util::pixdesc::PixFmtFlags::RGB
                    .union(crate::util::pixdesc::PixFmtFlags::XYZ),
            ) {
                ColorSpace::Rgb
            } else {
                ColorSpace::Unspecified
            };
        } else {
            let csp_list = self.links[link.0]
                .incfg
                .color_spaces
                .expect("query filled color spaces");
            if self.csp_lists[csp_list as usize].is_empty() {
                let (src_name, dst_name) = self.link_endpoints(link);
                log_error!(
                    None,
                    "Cannot select color space for the link between filters {src_name} and {dst_name}.\n"
                );
                return Err(Error::InvalidArgument(
                    "Cannot select color space for the link".into(),
                ));
            }
            self.links[link.0].colorspace = self.csp_lists[csp_list as usize][0];

            if formats::forced_full_range(fmt) {
                self.links[link.0].color_range = ColorRange::Jpeg;
            } else {
                let rng_list = self.links[link.0]
                    .incfg
                    .color_ranges
                    .expect("query filled color ranges");
                if self.rng_lists[rng_list as usize].is_empty() {
                    let (src_name, dst_name) = self.link_endpoints(link);
                    log_error!(
                        None,
                        "Cannot select color range for the link between filters {src_name} and {dst_name}.\n"
                    );
                    return Err(Error::InvalidArgument(
                        "Cannot select color range for the link".into(),
                    ));
                }
                self.links[link.0].color_range = self.rng_lists[rng_list as usize][0];
            }
        }

        // (920-935) unref both halves of every axis on this link.
        let l = &mut self.links[link.0];
        l.incfg.formats = None;
        l.outcfg.formats = None;
        l.incfg.color_spaces = None;
        l.outcfg.color_spaces = None;
        l.incfg.color_ranges = None;
        l.outcfg.color_ranges = None;
        Ok(())
    }

    fn link_endpoints(&self, link: LinkId) -> (String, String) {
        let l = &self.links[link.0];
        (
            self.nodes[l.src.0].name.clone(),
            self.nodes[l.dst.0].name.clone(),
        )
    }

    // -----------------------------------------------------------------------
    // Link configuration (avfiltergraph.c:248-260 + avfilter.c:327-452)
    // -----------------------------------------------------------------------

    /// `graph_config_links` (avfiltergraph.c:248-260): configure from the
    /// sinks inward (filters with no outputs first — recursion walks
    /// upstream).
    fn graph_config_links(&mut self) -> Result<()> {
        for i in 0..self.nodes.len() {
            if self.nodes[i].def.outputs.is_empty() {
                self.config_links_for(NodeId(i))?;
            }
        }
        Ok(())
    }

    /// `ff_filter_config_links` (avfilter.c:327-452): per input link —
    /// recurse upstream, run the SOURCE's output-pad `config_props` first,
    /// apply the video defaults, then the DEST's input-pad `config_props`.
    fn config_links_for(&mut self, filter: NodeId) -> Result<()> {
        let n_inputs = self.nodes[filter.0].def.inputs.len();
        for j in 0..n_inputs {
            let Some(link) = self.nodes[filter.0].inputs[j] else {
                log_error!(
                    None,
                    "Not all input and output are properly linked ({j}).\n"
                );
                return Err(Error::InvalidArgument(
                    "Not all input and output are properly linked".into(),
                ));
            };
            match self.links[link.0].init_state {
                LinkInitState::Init => continue,
                LinkInitState::StartInit => {
                    log_verbose!(None, "circular filter chain detected\n"); // avfilter.c:352
                    return Ok(());
                }
                LinkInitState::Uninit => {}
            }
            self.links[link.0].init_state = LinkInitState::StartInit;

            let src = self.links[link.0].src;
            let srcpad = self.links[link.0].srcpad;
            let dst = self.links[link.0].dst;
            let dstpad = self.links[link.0].dstpad;
            self.config_links_for(src)?;

            // (364-373) the source's output-pad config_props. C errors when
            // a source/multi-input filter lacks the callback; FilterImpl has
            // a default no-op, so the check is not portable — noted divergence.
            {
                let mut imp = self.nodes[src.0]
                    .imp
                    .take()
                    .expect("config_links: imp already taken");
                let ret = imp.config_props(self, src, PadRef::Out(srcpad));
                self.nodes[src.0].imp = Some(imp);
                if let Err(e) = ret {
                    log_error!(
                        None,
                        "Failed to configure output pad on {}\n",
                        self.nodes[src.0].name
                    );
                    return Err(e);
                }
            }

            // (405-430) video defaults: inherit time base / SAR / frame
            // rate / geometry from the source's first input link.
            let inlink = self.nodes[src.0].inputs.first().copied().flatten();
            {
                // (405-430) inherit from the source's first input link. C's
                // "unset" test is `!num && !den` (the 0/0 zero-init — our
                // `Rational::UNKNOWN`), NOT 0/1.
                let upstream = inlink.map(|il| {
                    let i = &self.links[il.0];
                    (i.time_base, i.sample_aspect_ratio, i.frame_rate, i.w, i.h)
                });
                let l = &mut self.links[link.0];
                if l.time_base == Rational::UNKNOWN {
                    l.time_base = upstream.map(|u| u.0).unwrap_or(Rational::new(1, 1_000_000)); // AV_TIME_BASE_Q
                }
                if l.sample_aspect_ratio == Rational::UNKNOWN {
                    l.sample_aspect_ratio = upstream.map(|u| u.1).unwrap_or(Rational::ONE);
                }
                if let Some((_, _, fr, w, h)) = upstream {
                    if l.frame_rate == Rational::UNKNOWN {
                        l.frame_rate = fr;
                    }
                    if l.w == 0 {
                        l.w = w;
                    }
                    if l.h == 0 {
                        l.h = h;
                    }
                } else if l.w == 0 || l.h == 0 {
                    log_error!(
                        None,
                        "Video source filters must set their output link's width and height\n"
                    );
                    return Err(Error::InvalidArgument(
                        "Video source filters must set their output link's width and height".into(),
                    ));
                }
            }

            // (445-452) the destination's input-pad config_props, LAST.
            {
                let mut imp = self.nodes[dst.0]
                    .imp
                    .take()
                    .expect("config_links: imp already taken");
                let ret = imp.config_props(self, dst, PadRef::In(dstpad));
                self.nodes[dst.0].imp = Some(imp);
                if let Err(e) = ret {
                    log_error!(
                        None,
                        "Failed to configure input pad on {}\n",
                        self.nodes[dst.0].name
                    );
                    return Err(e);
                }
            }

            self.links[link.0].init_state = LinkInitState::Init;
        }
        Ok(())
    }

    /// `graph_check_links` (avfiltergraph.c:263-282): every output link must
    /// carry a sane geometry for its negotiated format.
    fn graph_check_links(&mut self) -> Result<()> {
        for i in 0..self.nodes.len() {
            let outputs = self.nodes[i].outputs.clone();
            for l in outputs.iter().flatten() {
                let link = &self.links[l.0];
                let Some(fmt) = link.format else {
                    continue; // unconfigured links only exist pre-config
                };
                crate::util::imgutils::check_size(link.w, link.h)?;
                let _ = fmt;
            }
        }
        Ok(())
    }

    /// `graph_check_validity` (avfiltergraph.c:203-240): every pad of every
    /// filter must be connected. Error texts per C (media type string is
    /// always "video" in this port).
    pub(crate) fn check_validity(&self) -> Result<()> {
        for (i, filt) in self.nodes.iter().enumerate() {
            let node = NodeId(i);
            for (j, pad) in filt.def.inputs.iter().enumerate() {
                // C checks `filt->inputs[j] && filt->inputs[j]->src` — the
                // stored link must be attached to THIS node on its dst side.
                let connected =
                    filt.inputs.get(j).copied().flatten().is_some_and(|l| {
                        self.links[l.0].dst == node && self.links[l.0].dstpad == j
                    });
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
                let connected =
                    filt.outputs.get(j).copied().flatten().is_some_and(|l| {
                        self.links[l.0].src == node && self.links[l.0].srcpad == j
                    });
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
    /// the source (buffersrc.c:220-287): param-change checks, KEEP_REF
    /// semantics (our Arc-backed `&Frame` clone is exactly that), watchdog.
    pub fn add_frame(&mut self, src: NodeId, frame: &Frame) -> Result<()> {
        super::buffersrc::buffersrc_add_frame(self, src, Some(frame))
    }

    /// `av_buffersrc_close` — announce EOF on the source.
    pub fn close_source(&mut self, src: NodeId) -> Result<()> {
        super::buffersrc::buffersrc_close(self, src, crate::NOPTS)
    }

    /// `av_buffersink_get_frame` — pull one frame from the sink.
    /// `Err(Again)` = starved, `Err(Eof)` = drained (`get_frame_internal`
    /// loop, buffersink.c:93-133).
    pub fn get_frame(&mut self, sink: NodeId) -> Result<Frame> {
        super::buffersink::buffersink_get_frame(self, sink)
    }

    /// `avfilter_graph_parse_ptr` — parse a graph description into open pad
    /// lists. **Wave 2** (graphparser.c). Stub.
    pub fn parse_ptr(&mut self, desc: &str) -> Result<(Vec<InOut>, Vec<InOut>)> {
        // `avfilter_graph_parse_ptr` (graphparser.c:920-1042) — no
        // caller-supplied open lists in this shape (the full C-signature
        // port is parser::graph_parse_ptr).
        super::parser::parse_ptr(self, desc)
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
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterImpl;
    use crate::filter::filter;
    use crate::filter::graph::engine_test_helpers::*;

    fn null_pair() -> (FilterGraph, NodeId, NodeId, LinkId) {
        let mut g = FilterGraph::new();
        let a = g.create_filter("null", "").unwrap();
        let b = g.create_filter("null", "").unwrap();
        let l = g.link(a, 0, b, 0).unwrap();
        (g, a, b, l)
    }

    // -----------------------------------------------------------------------
    // Negotiation driver (avfiltergraph.c:522-1060, 1311-1400)
    // -----------------------------------------------------------------------

    #[test]
    fn query_round_picks_formats_and_drops_halves() {
        // src → null → sink with default (all-list) queries: one round
        // merges everything, pick_formats picks, halves disappear.
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let n = g.create_filter("null", "").unwrap();
        let sink = g.alloc_test_sink();
        let la = g.link(src, 0, n, 0).unwrap();
        let lb = g.link(n, 0, sink, 0).unwrap();
        g.query_formats_round().unwrap();
        g.graph_config_formats_tail().unwrap();
        for l in [la, lb] {
            assert!(g.links[l.0].format.is_some(), "format picked");
            // pick_format unrefs both halves (avfiltergraph.c:920-935).
            assert_eq!(g.links[l.0].incfg.formats, None);
            assert_eq!(g.links[l.0].outcfg.formats, None);
            assert!(matches!(
                g.links[l.0].colorspace,
                ColorSpace::Unspecified | ColorSpace::Bt709 | ColorSpace::Bt470bg
            ));
        }
    }

    #[test]
    fn reduce_formats_collapses_output_lists() {
        // src --la--> a --lb--> b: a declares its INPUT accepts only yuv420p
        // (la's outcfg singleton); its OUTPUT list contains it among others
        // — REDUCE_FORMATS collapses lb's incfg onto it.
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let a = g.create_filter("null", "").unwrap();
        let b = g.create_filter("null", "").unwrap();
        let la = g.link(src, 0, a, 0).unwrap();
        let lb = g.link(a, 0, b, 0).unwrap();
        let single = g.alloc_pix_list(vec![PixelFormat::Yuv420p]);
        let many = g.alloc_pix_list(vec![PixelFormat::Rgb24, PixelFormat::Yuv420p]);
        g.links[la.0].outcfg.formats = Some(single);
        g.links[lb.0].incfg.formats = Some(many);
        assert!(g.reduce_formats_on_filter(a).unwrap());
        assert_eq!(
            g.fmt_lists[many as usize],
            vec![PixelFormat::Yuv420p],
            "reduced in place, sharers see it"
        );
        // Fixpoint: a second round is a no-op.
        assert!(!g.reduce_formats_on_filter(a).unwrap());
    }

    #[test]
    fn pick_format_non_yuv_gets_jpeg_range_rgb_space() {
        // avfiltergraph.c:845-861: rgb24 is not regular YUV → range JPEG,
        // space RGB — WITHOUT consulting the (unset) color lists.
        let (mut g, _a, _b, l) = null_pair();
        let list = g.alloc_pix_list(vec![PixelFormat::Rgb24]);
        g.links[l.0].incfg.formats = Some(list);
        g.pick_format(l, None).unwrap();
        assert_eq!(g.links[l.0].format, Some(PixelFormat::Rgb24));
        assert_eq!(g.links[l.0].color_range, ColorRange::Jpeg);
        assert_eq!(g.links[l.0].colorspace, ColorSpace::Rgb);
        assert_eq!(g.links[l.0].incfg.color_spaces, None, "halves dropped");
    }

    #[test]
    fn pick_format_yuv_picks_declared_color_lists() {
        let (mut g, _a, _b, l) = null_pair();
        let fmts = g.alloc_pix_list(vec![PixelFormat::Yuv420p]);
        let csps = g.alloc_csp_list(vec![ColorSpace::Bt709]);
        let rngs = g.alloc_rng_list(vec![ColorRange::Mpeg]);
        g.links[l.0].incfg.formats = Some(fmts);
        g.links[l.0].incfg.color_spaces = Some(csps);
        g.links[l.0].incfg.color_ranges = Some(rngs);
        g.pick_format(l, None).unwrap();
        assert_eq!(g.links[l.0].colorspace, ColorSpace::Bt709);
        assert_eq!(g.links[l.0].color_range, ColorRange::Mpeg);
    }

    #[test]
    fn negotiation_stuck_errors_with_filter_names() {
        // A filter whose query always defers (Err(Again)) with nothing else
        // to merge: C's "could not choose their formats" EIO path
        // (avfiltergraph.c:723-736).
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let sink = g.alloc_test_sink();
        let l = g.link(src, 0, sink, 0).unwrap();
        let _ = l;
        // Make BOTH endpoints defer: swap src's impl for a deferring one.
        g.nodes[src.0].imp = Some(Box::new(DeferQuery));
        g.nodes[sink.0].imp = Some(Box::new(DeferQuery));
        let err = g.query_formats_round().unwrap_err();
        assert!(err.to_string().contains("could not choose their formats"));
    }

    /// query_formats that never declares anything (C: filters returning
    /// AVERROR(EAGAIN) from their query).
    struct DeferQuery;
    impl FilterImpl for DeferQuery {
        fn filter_frame(
            &mut self,
            _g: &mut FilterGraph,
            _node: NodeId,
            _pad: usize,
            _frame: Frame,
        ) -> Result<()> {
            Err(Error::Unsupported("not used".into()))
        }
        fn query_formats(&mut self, _g: &mut FilterGraph, _node: NodeId) -> Result<()> {
            Err(Error::Again)
        }
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
        // Close the chain with bare endpoint filters: src -> a -> b -> sink.
        // (Extra null filters would leave their outer pads open — C checks
        // EVERY pad, no exceptions.)
        let src = g.alloc_test_src();
        let sink = g.alloc_test_sink();
        g.link(src, 0, a, 0).unwrap();
        g.link(b, 0, sink, 0).unwrap();
        g.check_validity().unwrap();
    }

    #[test]
    fn set_common_fill_if_unset_and_identity_sharing() {
        let (mut g, a, b, l) = null_pair();
        // An explicitly declared half survives the default query.
        let explicit = g.alloc_pix_list(vec![PixelFormat::Yuv420p]);
        g.links[l.0].outcfg.formats = Some(explicit);
        g.default_query_formats(a).unwrap();
        // fill-if-unset: the explicit list was NOT overwritten...
        assert_eq!(g.links[l.0].outcfg.formats, Some(explicit));
        // ...the unset half got the ALL list (a's output → link incfg)...
        let all_idx = g.links[l.0].incfg.formats.expect("filled by default query");
        assert_eq!(g.fmt_lists[all_idx as usize].as_slice(), PixelFormat::ALL);
        // The far end's query fills the opposite halves (SET_COMMON_FORMATS2
        // crossing: inputs→outcfg, outputs→incfg — C formats.c:985-1010).
        g.default_query_formats(b).unwrap();
        assert!(g.links[l.0].incfg.color_spaces.is_some());
        assert!(g.links[l.0].outcfg.color_spaces.is_some());
        assert!(g.links[l.0].incfg.color_ranges.is_some());
        assert!(g.links[l.0].outcfg.color_ranges.is_some());
    }

    #[test]
    fn insert_filter_rewires_and_moves_outcfg() {
        let (mut g, _a, b, l) = null_pair();
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

        // Full config: validity → query/merge/pick formats → link config.
        // TestSrc sets w/h=8 in config_props (as buffersrc does); formats
        // are negotiated from the default all-lists.
        g.config().unwrap();
        for (name, l) in [("la", la), ("lb", lb), ("lc", lc)] {
            assert_eq!(g.links[l.0].w, 8, "{name} width from config_props");
            assert!(g.links[l.0].format.is_some(), "{name} format picked");
        }

        // Push a frame through the source's output link (what buffersrc's
        // add_frame will do internally) — in the NEGOTIATED format.
        let negotiated = g.links[la.0].format.unwrap();
        let mut frame = Frame::alloc(negotiated, 8, 8).unwrap();
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
    use super::super::filter::{FilterDef, FilterFlags, FilterImpl, FilterNode, PadDef};
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
        /// What buffersrc does: a source sets its output geometry in
        /// config_props (avfilter.c:424-429 requires it).
        fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
            if let PadRef::Out(_) = pad {
                let l = g.outlink(node, 0);
                g.links[l.0].w = 8;
                g.links[l.0].h = 8;
            }
            Ok(())
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

    const PAD: PadDef = PadDef {
        name: "default",
        needs_writable: false,
    };

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
