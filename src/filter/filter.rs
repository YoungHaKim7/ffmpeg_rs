//! Filter definitions + the activation engine — port of
//! `libavfilter/avfilter.c` (static defs `avfilter.c:631-986`, engine
//! `avfilter.c:216-560` and `avfilter.c:1018-1700`).
//!
//! The engine is the scheduler heart: the push path (`filter_frame` → fifo →
//! ready 300 → `default_activate` drains → `FilterImpl::filter_frame`), the
//! pull path (`request_frame_to_filter` → `ff_request_frame` →
//! `frame_wanted_out` + ready 100 upstream), and EOF/error propagation
//! (`set_in_status` from the source side, `link_set_out_status` from the
//! destination side, `forward_status_change` in the default activation).
//! The big scheduling comment at `avfilter.c:1308-1449` is the spec for the
//! `status_in`/`status_out`/`frame_wanted_out`/`frame_blocked_in` protocol —
//! ported verbatim in stage order.
//!
//! Not ported (documented at each site): audio sample machinery
//! (`take_samples`/`min_samples`, avfilter.c:1127-1193), timeline/enable
//! expressions (`evaluate_timeline_at_frame`, 1018-1034), the command queue
//! (`ff_inlink_process_commands`, 1607-1621), threading (`ff_filter_execute`),
//! `NEEDS_WRITABLE` copy-on-write (`ff_inlink_make_frame_writable`, 1567-1605
//! — the port's frames are cheap-`clone` Arc holders; filters that may
//! mutate call `Frame::make_writable` themselves), frame pools/hwframes, and
//! the graph-wide frame-queue cap (C's `FFFrameQueueGlobal` backpressure,
//! avfilter.c:1111-1115 — default `SIZE_MAX`, i.e. unbounded, in C too
//! unless `max_buffered_frames` is set).

use crate::util::error::Error;
use crate::util::frame::Frame;
use crate::util::rational::Rational;
use crate::util::{mathematics, NOPTS};
use crate::log_warning;

use super::graph::FilterGraph;
use super::link::{clone_status, status_eq, LinkId, NodeId};
use super::Options;

// ---------------------------------------------------------------------------
// Static filter definitions (avfilter.c:631-986, filters.h:267-462)
// ---------------------------------------------------------------------------

/// `AVFILTER_FLAG_*` subset (`avfilter.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterFlags(pub u32);

impl FilterFlags {
    /// `AVFILTER_FLAG_DYNAMIC_INPUTS/OUTPUTS` etc. are not needed by the
    /// ported set; the one flag that survives is the reconfigure whitelist.
    ///
    /// `ff_filter_frame` skips frame-vs-link validation for these filters
    /// (avfilter.c:1076-1082: buffersink, format, idet, null, scale,
    /// libplacebo, hqdn3d). Ported subset: buffersink, format, null, scale.
    /// NOT noformat (not in C's list).
    pub const ALLOWS_RECONFIGURE: FilterFlags = FilterFlags(1 << 0);

    pub const fn contains(self, other: FilterFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

/// `AVFilterPad` subset (`filters.h:40-121`): name + input-only
/// `NEEDS_WRITABLE`. Not ported: the media-type field (video-only port),
/// `FREE_NAME` (owned strings), the `get_buffer`/`filter_frame`/
/// `request_frame`/`config_props` pad callbacks — those live on
/// [`FilterImpl`] instead, so a pad is pure data.
pub struct PadDef {
    pub name: &'static str,
    pub needs_writable: bool,
}

/// `AVFilter` + `FFilter` flattened (`filters.h:267-462`). The AVOption
/// `priv_class` machinery disappears: per-filter state is the
/// [`FilterImpl`] the `make` constructor builds, and option strings reach it
/// through [`super::Options`].
pub struct FilterDef {
    pub name: &'static str,
    pub inputs: &'static [PadDef],
    pub outputs: &'static [PadDef],
    pub flags: FilterFlags,
    /// Positional-option names in AVOption declaration order (C derives this
    /// in `ff_filter_opt_parse`, avfilter.c:855-902 — an explicit `key=value`
    /// disables the REST of the chain).
    /// scale: `["w","h","flags","interl","size"]`; format/noformat:
    /// `["pix_fmts"]`; others: `[]`.
    pub shorthand: &'static [&'static str],
    /// `ff_filter_alloc`'s priv allocation (avfilter.c:719-723).
    pub make: fn() -> Box<dyn FilterImpl>,
}

/// Pad reference: one pad of one node, either side (`PadRef::In(i)` is the
/// i-th input pad). C passes `AVFilterPad*` + direction; we pass the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadRef {
    In(usize),
    Out(usize),
}

/// `AVFilterContext` + `FFFilterContext` flattened (`avfilter.h:273-353`,
/// `avfilter_internal.h:99-123`). Owned by `FilterGraph::nodes`.
///
/// Not ported: av_class/AVOption reflection, thread_type/nb_threads/execute
/// (single-threaded), enable_str/is_disabled (timeline), command_queue,
/// hw_device_ctx/extra_hw_frames, the graph back-pointer (the graph is the
/// enclosing arena). `state_flags & AV_CLASS_STATE_INITIALIZED` becomes
/// `initialized`.
pub struct FilterNode {
    pub def: &'static FilterDef,
    /// Instance name; auto-inserted converters are `"auto_scale_N"` exactly.
    pub name: String,
    pub inputs: Vec<Option<LinkId>>,
    pub outputs: Vec<Option<LinkId>>,
    /// The filter's private state. `None` only while `run_once` holds it.
    pub imp: Option<Box<dyn FilterImpl>>,
    /// 0 = idle; `set_ready` keeps the max; `run_once` CLEARS to 0 before
    /// activate (avfilter.c:229-233, 1460).
    pub ready: u32,
    pub initialized: bool,
    /// Options parsed but not yet consumed by `init` — the port of the
    /// option dict `avfilter_init_dict` applies (avfilter.c:929-933). Not in
    /// the C struct (the dict is a function argument there); the parser
    /// module (wave 2) owns the full lifecycle. `init_filter` sets it,
    /// `FilterImpl::init` drains it, leftovers are the "No such option"
    /// error.
    pub(crate) opts: Options,
}

/// The filter callbacks. Engine functions that must call back INTO a node's
/// impl take the impl as a parameter (`imp: &mut dyn FilterImpl`) —
/// `run_once` takes the node's imp out ONCE and threads it through
/// `activate` → `default_activate` → `filter_frame` dispatch. A node never
/// re-enters its own callbacks except through that threaded reference.
pub trait FilterImpl {
    /// The filter's `init`: consume recognized entries from
    /// `g.nodes[node].opts` (leftovers are the caller's "No such option"
    /// error). May not assume links exist yet.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<(), Error> {
        let _ = (g, node);
        Ok(())
    }

    /// Set format-list halves on OWN pads (inputs' `outcfg` / outputs'
    /// `incfg` — via `g.set_common_*` which fill-if-unset). The ENGINE always
    /// runs `g.default_query_formats(node)` AFTER this
    /// (avfiltergraph.c:410) — do not call it from the impl. May return
    /// `Err(Error::Again)` for "partial, re-ask later".
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<(), Error> {
        let _ = (g, node);
        Ok(())
    }

    /// `config_props` for one pad. Called by graph_config_links in C's order:
    /// src's `Out(pad)` FIRST → defaults pass → dst's `In(pad)` LAST
    /// (avfilter.c:364-448).
    fn config_props(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        pad: PadRef,
    ) -> Result<(), Error> {
        let _ = (g, node, pad);
        Ok(())
    }

    /// Push path: frame arrived on input `pad` (the engine already drained
    /// the fifo into this call). The default C pad behavior (forward to the
    /// first output, `default_filter_frame` avfilter.c:1007-1010) has no
    /// trait default — each impl states it (null/format forward verbatim).
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        pad: usize,
        frame: Frame,
    ) -> Result<(), Error>;

    /// Sources/sinks override. Default = the engine's default activation.
    fn activate(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<(), Error> {
        crate::filter::default_activate(self, g, node)
    }

}

// ---------------------------------------------------------------------------
// Scheduling primitives (avfilter.c:216-280)
// ---------------------------------------------------------------------------

/// `update_link_current_pts` (avfilter.c:216-227). `current_pts_us` and the
/// age-heap update are not ported (no `avfilter_graph_request_oldest`).
fn update_link_current_pts(g: &mut FilterGraph, link: LinkId, pts: i64) {
    if pts == NOPTS {
        return;
    }
    g.links[link.0].current_pts = pts;
}

/// `ff_filter_set_ready` (avfilter.c:229-233): keep the max. Priorities in
/// practice: 100 frame requested, 200 status change, 300 frame queued /
/// run-again.
pub fn set_ready(g: &mut FilterGraph, node: NodeId, priority: u32) {
    let n = &mut g.nodes[node.0];
    n.ready = n.ready.max(priority);
}

/// `filter_unblock` (avfilter.c:239-247): clear `frame_blocked_in` on all of
/// `filter`'s outputs — "necessary whenever something changes on input".
pub(crate) fn filter_unblock(g: &mut FilterGraph, node: NodeId) {
    for out in &g.nodes[node.0].outputs {
        if let Some(l) = out {
            g.links[l.0].frame_blocked_in = false;
        }
    }
}

/// `ff_avfilter_link_set_in_status` (avfilter.c:250-263) — the ONLY way a
/// source announces EOF/error on its output link. The status is HELD behind
/// queued frames until the destination acknowledges it; one-way (setting a
/// second status is a bug, debug-asserted).
pub fn set_in_status(g: &mut FilterGraph, link: LinkId, status: Error, pts: i64) {
    if let Some(cur) = &g.links[link.0].status_in {
        if status_eq(cur, &status) {
            return;
        }
    }
    debug_assert!(
        g.links[link.0].status_in.is_none(),
        "in-status set a second time"
    );
    let dst = g.links[link.0].dst;
    {
        let l = &mut g.links[link.0];
        l.status_in = Some(status);
        l.status_in_pts = pts;
        l.frame_wanted_out = false;
        l.frame_blocked_in = false;
    }
    filter_unblock(g, dst);
    set_ready(g, dst, 200);
}

/// `ff_outlink_set_status` — an ALIAS of [`set_in_status`] (`filters.h`:
/// same semantics, called from the source filter on its own output link).
/// NOT `link_set_out_status`.
pub fn outlink_set_status(g: &mut FilterGraph, link: LinkId, status: Error, pts: i64) {
    set_in_status(g, link, status, pts);
}

/// `link_set_out_status` (avfilter.c:269-280) — engine-private ack from the
/// destination side: the status change is "already happened" as of now.
/// Keeps the fifo intact (contrast [`inlink_set_status`]).
pub(crate) fn link_set_out_status(g: &mut FilterGraph, link: LinkId, status: Error, pts: i64) {
    debug_assert!(
        !g.links[link.0].frame_wanted_out,
        "link_set_out_status with a pending frame request"
    );
    debug_assert!(g.links[link.0].status_out.is_none(), "out-status set twice");
    let (dst, src) = {
        let l = &g.links[link.0];
        (l.dst, l.src)
    };
    g.links[link.0].status_out = Some(status);
    if pts != NOPTS {
        update_link_current_pts(g, link, pts);
    }
    filter_unblock(g, dst);
    set_ready(g, src, 200);
}

// ---------------------------------------------------------------------------
// Push path (avfilter.c:1007-1112, 1195-1226)
// ---------------------------------------------------------------------------

/// `ff_filter_frame` (avfilter.c:1068-1125) — push one frame over an output
/// link: validate against the link, queue it into the destination's fifo,
/// mark the destination ready(300).
///
/// The video consistency checks (1075-1088) are C `av_assert1`s — debug
/// asserts here too, skipped entirely for `ALLOWS_RECONFIGURE` filters. The
/// audio branch (1089-1105) is not ported.
pub fn filter_frame(g: &mut FilterGraph, link: LinkId, frame: Frame) -> Result<(), Error> {
    let dst = g.links[link.0].dst;
    if !g.nodes[dst.0].def.flags.contains(FilterFlags::ALLOWS_RECONFIGURE) {
        let l = &g.links[link.0];
        if let Some(f) = l.format {
            debug_assert_eq!(frame.format, f, "frame format does not match link format");
        }
        debug_assert_eq!(frame.width, l.w, "frame width does not match link");
        debug_assert_eq!(frame.height, l.h, "frame height does not match link");
        // alpha_mode check (1086-1087): not ported — no alpha-mode axis.
    }

    {
        let l = &mut g.links[link.0];
        // A frame arriving clears both request state and blocked state
        // (avfilter.c:1107).
        l.frame_blocked_in = false;
        l.frame_wanted_out = false;
        l.frame_count_in += 1;
    }
    filter_unblock(g, dst);
    // ff_framequeue_add — ref transfer. The capacity/ENOMEM backpressure
    // (1111-1115) is not ported: C's default max is SIZE_MAX (unbounded).
    g.links[link.0].fifo.push_back(frame);
    set_ready(g, dst, 300);
    Ok(())
}

/// `inlink_consume_frame` + `consume_update` (avfilter.c:1509-1538): pop one
/// frame from a link fifo, tracking `current_pts` and `frame_count_out`.
pub fn inlink_consume_frame(g: &mut FilterGraph, link: LinkId) -> Option<Frame> {
    let frame = g.links[link.0].fifo.pop_front()?;
    update_link_current_pts(g, link, frame.pts);
    g.links[link.0].frame_count_out += 1;
    Some(frame)
}

/// `filter_frame_to_filter` (avfilter.c:1195-1226) — activate-step "drain
/// one frame from the link fifo into the destination filter". `imp` is the
/// destination node's impl, threaded by `default_activate`.
///
/// Generic over the impl type (not `&mut dyn`) so the trait's default
/// `activate` can forward `&mut Self` without a `Sized` bound.
pub fn filter_frame_to_filter<I: FilterImpl + ?Sized>(
    g: &mut FilterGraph,
    imp: &mut I,
    link: LinkId,
) -> Result<(), Error> {
    debug_assert!(
        !g.links[link.0].fifo.is_empty(),
        "filter_frame_to_filter on an empty fifo"
    );
    let frame = match inlink_consume_frame(g, link) {
        Some(f) => f,
        None => unreachable!("asserted non-empty"),
    };
    let (dst, dstpad) = {
        let l = &g.links[link.0];
        (l.dst, l.dstpad)
    };
    // "The filter will soon have received a new frame, that may allow it to
    // produce one or more: unblock its outputs." (avfilter.c:1211-1213)
    filter_unblock(g, dst);
    // CRITICAL (1214-1216): the callback must see the PRE-frame
    // frame_count_out — consume_update above incremented it,
    // filter_frame_framed re-increments AFTER the callback (1060). vf_scale's
    // VAR_N reads this value (vf_scale.c:799).
    g.links[link.0].frame_count_out -= 1;
    let ret = imp.filter_frame(g, dst, dstpad, frame);
    g.links[link.0].frame_count_out += 1;
    match &ret {
        Err(e) => {
            // (1218-1219) a filter_frame error closes the link from the
            // destination side — unless it merely echoes the current
            // out-status. frame_wanted_out is cleared defensively first so
            // link_set_out_status's assert holds (C relies on it being
            // already false here).
            let already = g.links[link.0]
                .status_out
                .as_ref()
                .is_some_and(|s| status_eq(s, e));
            if !already {
                g.links[link.0].frame_wanted_out = false;
                link_set_out_status(g, link, clone_status(e), NOPTS);
            }
        }
        Ok(()) => {
            // (1221-1223) run again — more frames may have queued, or the
            // input status also changed.
            set_ready(g, dst, 300);
        }
    }
    ret
}

// ---------------------------------------------------------------------------
// Pull path (avfilter.c:483-551)
// ---------------------------------------------------------------------------

/// `ff_request_frame` (avfilter.c:483-508) — the legacy pull primitive: dst
/// asks src for a frame over an input link. THIS is the path EOF propagates:
/// upstream status_in is acknowledged into status_out and returned.
pub fn ff_request_frame(g: &mut FilterGraph, inlink: LinkId) -> Result<(), Error> {
    // C asserts the destination is not activate-style (avfilter.c:489) —
    // not checkable without trait introspection; document the contract.
    let (status_out, status_in, fifo_nonempty) = {
        let l = &g.links[inlink.0];
        (
            l.status_out.as_ref().map(clone_status),
            l.status_in.as_ref().map(clone_status),
            !l.fifo.is_empty(),
        )
    };
    if let Some(so) = status_out {
        return Err(so); // link already closed from this side
    }
    if let Some(si) = status_in {
        if fifo_nonempty {
            // Frames are drained first by stage 1 of default_activate; the
            // destination is ready >= 300 (avfilter.c:493-496).
            debug_assert!(!g.links[inlink.0].frame_wanted_out);
            return Ok(());
        }
        // Acknowledge: the status becomes the out-status, and is returned —
        // request_frame_to_filter maps it onto the OUTPUT link's in-status.
        let pts = g.links[inlink.0].status_in_pts;
        link_set_out_status(g, inlink, clone_status(&si), pts);
        return Err(si);
    }
    g.links[inlink.0].frame_wanted_out = true;
    let src = g.links[inlink.0].src;
    set_ready(g, src, 100);
    Ok(())
}

/// `guess_status_pts` (avfilter.c:510-530): estimate the pts of a status
/// being injected into a link, from the node's input links.
fn guess_status_pts(
    g: &FilterGraph,
    node: NodeId,
    status: &Error,
    link_time_base: Rational,
) -> i64 {
    let mut r = i64::MAX;
    for i in &g.nodes[node.0].inputs {
        let Some(l) = i else { continue };
        let li = &g.links[l.0];
        if li.status_out.as_ref().is_some_and(|s| status_eq(s, status)) {
            r = r.min(mathematics::rescale_q(
                li.current_pts,
                li.time_base,
                link_time_base,
            ));
        }
    }
    if r < i64::MAX {
        return r;
    }
    log_warning!(Some("graph"), "EOF timestamp not reliable");
    for i in &g.nodes[node.0].inputs {
        let Some(l) = i else { continue };
        let li = &g.links[l.0];
        r = r.min(mathematics::rescale_q(
            li.status_in_pts,
            li.time_base,
            link_time_base,
        ));
    }
    if r < i64::MAX {
        r
    } else {
        NOPTS
    }
}

/// `request_frame_to_filter` (avfilter.c:532-551) — ask the source of
/// `link` to produce toward it. Not ported: the `srcpad->request_frame` pad
/// callback (540-541) — no ported filter defines one; single-input filters
/// without it forward the request upstream (the fallback below), and sources
/// use activate-style instead.
pub(crate) fn request_frame_to_filter(g: &mut FilterGraph, link: LinkId) -> Result<(), Error> {
    // Assume the filter is blocked; the method clears it by calling
    // filter_frame (avfilter.c:539).
    g.links[link.0].frame_blocked_in = true;
    let src = g.links[link.0].src;
    let src_in0 = g.nodes[src.0].inputs.first().copied().flatten();
    let ret = match src_in0 {
        Some(inlink) => ff_request_frame(g, inlink),
        None => Err(Error::Unsupported(format!(
            "filter '{}' has no request_frame path (C: avfilter.c:542-543 returns -1)",
            g.nodes[src.0].name
        ))),
    };
    if let Err(e) = &ret {
        // Error triage (544-546): anything but EAGAIN and the current
        // in-status becomes the link's in-status (EOF included).
        let is_again = matches!(e, Error::Again);
        let is_status_in = g.links[link.0]
            .status_in
            .as_ref()
            .is_some_and(|s| status_eq(s, e));
        if !is_again && !is_status_in {
            let pts = guess_status_pts(g, src, e, g.links[link.0].time_base);
            set_in_status(g, link, clone_status(e), pts);
        }
    }
    // (547-548) EOF is a normal outcome for the caller.
    match ret {
        Err(Error::Eof) => Ok(()),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Status helpers for activate-style filters (avfilter.c:1467-1660)
// ---------------------------------------------------------------------------

/// `ff_inlink_acknowledge_status` (avfilter.c:1467-1481): `Some` only on the
/// transition — fifo empty AND `status_in` set AND `status_out` unset; the
/// ack is performed into `status_out` and the (status, pts) pair returned.
pub fn inlink_acknowledge_status(g: &mut FilterGraph, link: LinkId) -> Option<(Error, i64)> {
    if !g.links[link.0].fifo.is_empty() {
        return None; // frames still queued — status is held behind them
    }
    if g.links[link.0].status_out.is_some() {
        return None; // already acknowledged
    }
    let status = clone_status(g.links[link.0].status_in.as_ref()?);
    let status_pts = g.links[link.0].status_in_pts;
    g.links[link.0].status_out = Some(clone_status(&status));
    if status_pts != NOPTS {
        g.links[link.0].current_pts = status_pts;
    }
    Some((status, g.links[link.0].current_pts))
}

/// `ff_inlink_set_status` (avfilter.c:1632-1646) — HARD close from the
/// destination side: ack via `link_set_out_status` AND discard all queued
/// frames AND set `status_in` too if unset (both directions dead, so
/// upstream `ff_request_frame` short-circuits).
pub fn inlink_set_status(g: &mut FilterGraph, link: LinkId, status: Error) {
    if g.links[link.0].status_out.is_some() {
        return;
    }
    g.links[link.0].frame_wanted_out = false;
    g.links[link.0].frame_blocked_in = false;
    link_set_out_status(g, link, status.clone(), NOPTS);
    // Discard all queued frames (avfilter.c:1640-1643) — intentional: the
    // destination declared it will never consume them.
    g.links[link.0].fifo.clear();
    if g.links[link.0].status_in.is_none() {
        g.links[link.0].status_in = Some(status);
    }
}

/// `ff_inlink_request_frame` (avfilter.c:1623-1630): set
/// `frame_wanted_out` + ready(100) the source — the activate-style
/// counterpart of `ff_request_frame` (never mix the two on one filter).
pub fn inlink_request_frame(g: &mut FilterGraph, link: LinkId) {
    debug_assert!(g.links[link.0].status_in.is_none());
    debug_assert!(g.links[link.0].status_out.is_none());
    g.links[link.0].frame_wanted_out = true;
    let src = g.links[link.0].src;
    set_ready(g, src, 100);
}

// ---------------------------------------------------------------------------
// Default activation (avfilter.c:1228-1306)
// ---------------------------------------------------------------------------

/// `forward_status_change` (avfilter.c:1228-1261) — stage 2 of the default
/// activation: an input has `status_in`; drive the filter to acknowledge it
/// by requesting from its outputs until the input's `status_out` appears.
/// Round-robin + progress bookkeeping must stay exact or multi-output
/// filters livelock (C comment at 1250-1253).
pub(crate) fn forward_status_change(
    g: &mut FilterGraph,
    node: NodeId,
    inlink: LinkId,
) -> Result<(), Error> {
    debug_assert!(g.links[inlink.0].status_out.is_none());
    let outputs: Vec<Option<LinkId>> = g.nodes[node.0].outputs.clone();
    if outputs.is_empty() {
        // "not necessary with the current API and sinks" (1235-1238)
        return Ok(());
    }
    let mut out = 0usize;
    let mut progress = 0usize;
    while g.links[inlink.0].status_out.is_none() {
        let li_out = outputs[out].expect("output pad unlinked during activation");
        if g.links[li_out.0].status_in.is_none() {
            progress += 1;
            request_frame_to_filter(g, li_out)?;
        }
        out += 1;
        if out == outputs.len() {
            if progress == 0 {
                // Every output already closed: the input is no longer
                // interesting (example: overlay in shortest mode).
                let l = &g.links[inlink.0];
                let status = clone_status(l.status_in.as_ref().expect("stage-2 precondition"));
                let pts = l.status_in_pts;
                link_set_out_status(g, inlink, status, pts);
                return Ok(());
            }
            progress = 0;
            out = 0;
        }
    }
    set_ready(g, node, 200);
    Ok(())
}

/// `filter_activate_default` (avfilter.c:1263-1306) — EXACT port, in C's
/// order. What a filter without its own `activate()` does in one step:
///
/// 0. ALL output links closed (their `status_in` is EOF — the view direction
///    of `ff_outlink_get_status`) → hard-close every input, done.
/// 1. An input has a frame queued → drain ONE frame into the filter
///    (`filter_frame_to_filter`), return.
/// 2. An input has `status_in` && !`status_out` (fifo MUST be drained) →
///    `forward_status_change`, return.
/// 3. An output wants a frame and is not blocked → `request_frame_to_filter`.
/// 4. An output wants a frame (even blocked) → `request_frame_to_filter`.
/// 5. No outputs (sink) → `inlink_request_frame(inputs[0])`.
/// 6. Nothing to do → `FFERROR_NOT_READY`, mapped to Ok by
///    `ff_filter_activate` (avfilter.c:1462-1463).
pub fn default_activate<I: FilterImpl + ?Sized>(
    imp: &mut I,
    g: &mut FilterGraph,
    node: NodeId,
) -> Result<(), Error> {
    let inputs: Vec<Option<LinkId>> = g.nodes[node.0].inputs.clone();
    let outputs: Vec<Option<LinkId>> = g.nodes[node.0].outputs.clone();

    // Stage A (1268-1274): EOF back-propagation.
    let nb_outputs = outputs.len();
    if nb_outputs > 0
        && outputs.iter().all(|o| {
            o.is_some_and(|l| matches!(g.links[l.0].status_in, Some(Error::Eof)))
        })
    {
        for i in &inputs {
            if let Some(l) = i {
                inlink_set_status(g, *l, Error::Eof);
            }
        }
        return Ok(());
    }

    // Stage B (1276-1281): process queued frames — one input per activation.
    // Video: min_samples == 0, so "samples ready" is simply "fifo non-empty"
    // (samples_ready, avfilter.c:1127-1132).
    for i in &inputs {
        if let Some(l) = i {
            if !g.links[l.0].fifo.is_empty() {
                return filter_frame_to_filter(g, imp, *l);
            }
        }
    }

    // Stage C (1282-1288): forward input status.
    for i in &inputs {
        if let Some(l) = i {
            let (has_in, has_out, queued) = {
                let li = &g.links[l.0];
                (li.status_in.is_some(), li.status_out.is_some(), !li.fifo.is_empty())
            };
            if has_in && !has_out {
                debug_assert!(!queued, "status_in forwarding with frames still queued");
                return forward_status_change(g, node, *l);
            }
        }
    }

    // Stages D/D2 (1289-1300): serve frame requests — unblocked first.
    for require_unblocked in [true, false] {
        for o in &outputs {
            if let Some(l) = o {
                let li = &g.links[l.0];
                if li.frame_wanted_out && (!require_unblocked || !li.frame_blocked_in) {
                    return request_frame_to_filter(g, *l);
                }
            }
        }
    }

    // Stage E (1301-1304): sinks.
    if nb_outputs == 0 {
        if let Some(Some(l)) = inputs.first() {
            inlink_request_frame(g, *l);
        }
        return Ok(());
    }

    // FFERROR_NOT_READY (1305) → Ok(()) (ff_filter_activate, 1462-1463).
    Ok(())
}

impl FilterGraph {
    /// `ff_filter_graph_run_once` (avfiltergraph.c:1616-1633) +
    /// `ff_filter_activate` (avfilter.c:1451-1465): activate the single
    /// most-ready filter (largest `ready`, first on ties — C's `>`
    /// comparison keeps the first maximum). `Err(Error::Again)` when nothing
    /// is ready.
    ///
    /// `ready` is cleared to 0 BEFORE `activate` runs (avfilter.c:1460); the
    /// node's impl is taken out for the duration (a second take is a bug).
    ///
    /// `FFERROR_BUFFERSRC_EMPTY` (filters.h:35 — a source with data queued
    /// upstream of an empty fifo) is NOT yet distinguished here: wave 2's
    /// buffersrc represents it as a plain `Ok(())`/`Err(Again)` outcome and
    /// the buffersink `get_frame` loop latches it; revisit when buffersink
    /// lands.
    pub fn run_once(&mut self) -> Result<(), Error> {
        let mut best: Option<usize> = None;
        let mut best_ready = 0u32;
        for (i, n) in self.nodes.iter().enumerate() {
            if n.ready > best_ready {
                best_ready = n.ready;
                best = Some(i);
            }
        }
        let node = NodeId(match best {
            Some(n) => n,
            None => return Err(Error::Again),
        });
        self.nodes[node.0].ready = 0;
        let mut imp = self.nodes[node.0]
            .imp
            .take()
            .expect("run_once: node imp already taken");
        let ret = imp.activate(self, node);
        debug_assert!(
            self.nodes[node.0].imp.is_none(),
            "run_once: imp was restored while taken"
        );
        self.nodes[node.0].imp = Some(imp);
        ret
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::link::Link;

    /// Minimal one-in/one-out passthrough impl for engine tests.
    struct Passthrough;
    impl FilterImpl for Passthrough {
        fn filter_frame(
            &mut self,
            g: &mut FilterGraph,
            node: NodeId,
            _pad: usize,
            frame: Frame,
        ) -> Result<(), Error> {
            let out = g.outlink(node, 0);
            filter_frame(g, out, frame)
        }
    }

    fn test_def() -> &'static FilterDef {
        static PAD: PadDef = PadDef { name: "default", needs_writable: false };
        static DEF: FilterDef = FilterDef {
            name: "passthrough",
            inputs: &[PAD],
            outputs: &[PAD],
            flags: FilterFlags(0),
            shorthand: &[],
            make: || Box::new(Passthrough),
        };
        &DEF
    }

    /// src(0) -> mid(1): two nodes, one link; frame pushed lands in mid's fifo.
    fn two_node_graph() -> (FilterGraph, NodeId, NodeId, LinkId) {
        let mut g = FilterGraph::new();
        let def = test_def();
        for _ in 0..2 {
            g.nodes.push(FilterNode {
                def,
                name: def.name.to_string(),
                inputs: vec![None],
                outputs: vec![None],
                imp: Some((def.make)()),
                ready: 0,
                initialized: true,
                opts: Options::default(),
            });
        }
        let l = g.link(NodeId(0), 0, NodeId(1), 0).unwrap();
        (g, NodeId(0), NodeId(1), l)
    }

    fn frame(pts: i64) -> Frame {
        let mut f = Frame::alloc(crate::util::pixfmt::PixelFormat::Gray8, 4, 4).unwrap();
        f.pts = pts;
        f
    }

    #[test]
    fn set_ready_keeps_max() {
        let (mut g, a, _b, _l) = two_node_graph();
        set_ready(&mut g, a, 200);
        set_ready(&mut g, a, 100);
        assert_eq!(g.nodes[a.0].ready, 200, "ready must keep the max");
        set_ready(&mut g, a, 300);
        assert_eq!(g.nodes[a.0].ready, 300);
    }

    #[test]
    fn run_once_again_when_idle_and_clears_ready() {
        let (mut g, _a, _b, _l) = two_node_graph();
        assert!(matches!(g.run_once(), Err(Error::Again)));
        set_ready(&mut g, NodeId(0), 200);
        g.run_once().unwrap(); // no work → Ok, ready cleared
        assert_eq!(g.nodes[0].ready, 0);
        assert!(matches!(g.run_once(), Err(Error::Again)));
    }

    #[test]
    fn push_path_queues_and_counts() {
        let (mut g, src, dst, l) = two_node_graph();
        filter_frame(&mut g, l, frame(7)).unwrap();
        // Frame queued in dst's input fifo; dst woken at 300.
        assert_eq!(g.links[l.0].fifo.len(), 1);
        assert_eq!(g.links[l.0].frame_count_in, 1);
        assert_eq!(g.links[l.0].frame_count_out, 0);
        assert_eq!(g.nodes[dst.0].ready, 300);
        assert_eq!(g.nodes[src.0].ready, 0);
        // current_pts not yet updated (that happens on consume).
        assert_eq!(g.links[l.0].current_pts, NOPTS);
    }

    #[test]
    fn consume_updates_pts_and_count() {
        let (mut g, _src, _dst, l) = two_node_graph();
        filter_frame(&mut g, l, frame(42)).unwrap();
        let f = inlink_consume_frame(&mut g, l).unwrap();
        assert_eq!(f.pts, 42);
        assert_eq!(g.links[l.0].current_pts, 42);
        assert_eq!(g.links[l.0].frame_count_out, 1);
        assert!(inlink_consume_frame(&mut g, l).is_none());
    }

    #[test]
    fn status_protocol_one_shot_transitions() {
        let (mut g, _src, dst, l) = two_node_graph();
        set_in_status(&mut g, l, Error::Eof, 100);
        assert!(g.links[l.0].status_in.is_some());
        assert_eq!(g.links[l.0].status_in_pts, 100);
        assert_eq!(g.nodes[dst.0].ready, 200);
        assert!(!g.links[l.0].frame_wanted_out);
        // One-way: the same status again is a no-op.
        set_in_status(&mut g, l, Error::Eof, 200);
        assert_eq!(g.links[l.0].status_in_pts, 100);

        // Acknowledge: only when fifo drained.
        filter_frame(&mut g, l, frame(1)).unwrap();
        assert!(inlink_acknowledge_status(&mut g, l).is_none());
        inlink_consume_frame(&mut g, l).unwrap();
        let (status, pts) = inlink_acknowledge_status(&mut g, l).expect("transition");
        assert!(matches!(status, Error::Eof));
        assert_eq!(pts, 1); // current_pts of the last consumed frame
        // Second acknowledge: already acked → None.
        assert!(inlink_acknowledge_status(&mut g, l).is_none());
    }

    #[test]
    fn inlink_set_status_discards_queued_frames() {
        let (mut g, _src, _dst, l) = two_node_graph();
        filter_frame(&mut g, l, frame(1)).unwrap();
        filter_frame(&mut g, l, frame(2)).unwrap();
        inlink_set_status(&mut g, l, Error::Eof);
        // Hard close: both directions dead, fifo discarded.
        assert!(g.links[l.0].fifo.is_empty());
        assert!(g.links[l.0].status_out.is_some());
        assert!(g.links[l.0].status_in.is_some());
        assert!(!g.links[l.0].frame_wanted_out);
        assert!(!g.links[l.0].frame_blocked_in);
    }

    #[test]
    fn ff_request_frame_pull_and_eof() {
        let (mut g, _src, _dst, l) = two_node_graph();
        // Pull: frame_wanted_out + ready(100) on src.
        ff_request_frame(&mut g, l).unwrap();
        assert!(g.links[l.0].frame_wanted_out);
        assert_eq!(g.nodes[0].ready, 100);
        // A pushed frame clears the want.
        filter_frame(&mut g, l, frame(1)).unwrap();
        assert!(!g.links[l.0].frame_wanted_out);
        // EOF: status_in held while frames queued → Ok.
        set_in_status(&mut g, l, Error::Eof, 50);
        ff_request_frame(&mut g, l).unwrap();
        // After drain: acknowledges and returns the status.
        inlink_consume_frame(&mut g, l).unwrap();
        assert!(matches!(ff_request_frame(&mut g, l), Err(Error::Eof)));
        assert!(g.links[l.0].status_out.is_some());
        // And thereafter short-circuits on status_out.
        assert!(matches!(ff_request_frame(&mut g, l), Err(Error::Eof)));
    }
}
