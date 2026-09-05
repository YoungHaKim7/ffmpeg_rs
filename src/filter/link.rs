//! Filter links — port of the link parts of `libavfilter/avfilter.h` +
//! `filters.h` (`FilterLink`) + `avfilter_internal.h` (`FilterLinkInternal`)
//! + `framequeue.{h,c}`, flattened into one struct.
//!
//! C has three layers for one edge (public `AVFilterLink`, the `FilterLink`
//! extension, the internal scheduling state); the port flattens them into
//! [`Link`] owned by `FilterGraph::links`, referenced by [`LinkId`] from both
//! endpoint nodes' pad arrays.
//!
//! Not ported (audio/hw axes out of scope, C: `avfilter.h:367-427`):
//! `sample_rate`, `ch_layout`, `min/max_samples`, `sample_count_in/out`
//! (audio); `hw_frames_ctx`, `frame_pool`, `side_data`, `alpha_mode`
//! (hwaccel/side-data/alpha negotiation); `age_index` + the sink heap
//! (`avfiltergraph.c:1516-1565`, only consumed by
//! `avfilter_graph_request_oldest`, not ported — `current_pts` is still
//! tracked so the field survives for that API).

use std::collections::VecDeque;

use crate::util::color::{ColorRange, ColorSpace};
use crate::util::error::Error;
use crate::util::frame::Frame;
use crate::util::pixfmt::PixelFormat;
use crate::util::rational::Rational;
use crate::NOPTS;

/// Index of a filter instance in `FilterGraph::nodes` (C: the `AVFilterContext*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Index of a link in `FilterGraph::links` (C: the `AVFilterLink*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinkId(pub usize);

/// Index of a negotiation list in one of the graph's format-list arenas
/// (C: the refcounted `AVFilterFormats*` — identity is pointer equality
/// there, index equality here).
pub type ListIdx = u32;

/// `AVLinkStatus` (`avfilter_internal.h`) — link configuration state for
/// `ff_filter_config_links`'s cycle detection (`avfilter.c:352-360`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LinkInitState {
    /// `AVLINK_UNINIT` — not configured yet.
    #[default]
    Uninit,
    /// `AVLINK_STARTINIT` — configuration in progress (a link in this state
    /// means a circular chain; C logs and returns success).
    StartInit,
    /// `AVLINK_INIT` — fully configured.
    Init,
}

/// `AVFilterFormatsConfig` — ONE HALF of a link's negotiation state
/// (`avfilter.h:120-148`).
///
/// DIRECTION (settled against `formats.c:839-860` — do not get this wrong):
///   * `incfg`  = the half SUPPLIED BY THE LINK'S SRC filter (what src offers
///     INTO the link); a filter's query writes this half on its OUTPUT links.
///   * `outcfg` = the half SUPPLIED BY THE LINK'S DST filter (what dst accepts
///     OUT of the link); a filter's query writes this half on its INPUT links.
///
/// `ff_set_common_formats` refs into `inputs[i]->outcfg` and
/// `outputs[i]->incfg`; `merge(a, b)` takes `a = incfg` (survivor keeps the
/// SRC's order); `pick_format` reads `incfg`; `formats_declared` checks a
/// filter's INPUTS' `outcfg` and OUTPUTS' `incfg` (`avfiltergraph.c:413-444`).
///
/// `None` = not yet declared, or already picked (C unrefs to NULL after pick
/// — the None IS the already-picked marker; `pick_format`'s
/// `incfg.formats.is_none()` early-return depends on it).
///
/// Not ported: `samplerates`/`channel_layouts` (audio axes, C:
/// `avfilter.h:124-131`), `alpha_modes` (alpha-mode negotiation axis out of
/// scope — `formats.c:416-423`'s fourth merger).
#[derive(Clone, Debug, Default)]
pub struct FormatsConfig {
    pub formats: Option<ListIdx>,
    pub color_spaces: Option<ListIdx>,
    pub color_ranges: Option<ListIdx>,
}

/// `AVFilterLink` + `FilterLink` + `FilterLinkInternal` flattened. Owned by
/// `FilterGraph::links`; both endpoint nodes reference it by [`LinkId`].
#[derive(Debug)]
pub struct Link {
    /// Source filter (`link->src`).
    pub src: NodeId,
    /// Index of the source pad within `src.outputs` (`link->srcpad` is a pad
    /// pointer in C; we keep the index — `FF_OUTLINK_IDX`, `filters.h:215`).
    pub srcpad: usize,
    /// Destination filter (`link->dst`).
    pub dst: NodeId,
    /// Index of the destination pad within `dst.inputs` (`link->dstpad`).
    pub dstpad: usize,
    /// `link->format` — `AV_PIX_FMT_NONE`'s `-1` sentinel becomes `None`
    /// ("unconfigured", distinguishable from a picked format; `avfilter.c:191`).
    /// Set by `pick_format`, NOT by `config_props`.
    pub format: Option<PixelFormat>,
    /// `link->w` / `link->h` — set by `config_props`/defaults, NOT by
    /// `pick_format` (`avfiltergraph.c:801-898` never touches geometry).
    pub w: u32,
    pub h: u32,
    /// `link->sample_aspect_ratio`.
    pub sample_aspect_ratio: Rational,
    /// `link->colorspace` — set by `pick_format`.
    pub colorspace: ColorSpace,
    /// `link->color_range` — set by `pick_format`.
    pub color_range: ColorRange,
    /// `link->time_base`.
    pub time_base: Rational,
    /// `FilterLink::frame_rate` (`filters.h`).
    pub frame_rate: Rational,
    /// `FilterLink::current_pts` — last seen pts in link time base
    /// (`current_pts_us` and the age heap are not ported).
    pub current_pts: i64,
    /// src-supplied negotiation half.
    pub incfg: FormatsConfig,
    /// dst-supplied negotiation half.
    pub outcfg: FormatsConfig,
    // ---- runtime: the status protocol (the spec comment is avfilter.c:1308-1449) ----
    /// `FFFrameQueue` — `push_back` == `ff_framequeue_add` (ref transfer),
    /// `pop_front` == `ff_framequeue_take`.
    pub fifo: VecDeque<Frame>,
    /// `status_in` — HELD behind queued frames until acknowledged. Set by the
    /// source (`ff_avfilter_link_set_in_status`); `None` = no status.
    pub status_in: Option<Error>,
    /// `status_in_pts` — timestamp of the status change, link time base.
    pub status_in_pts: i64,
    /// `status_out` — one-shot acknowledgment (`link_set_out_status`; a
    /// second set is a bug, debug-asserted).
    pub status_out: Option<Error>,
    /// `frame_wanted_out` — dst needs a frame on this link.
    pub frame_wanted_out: bool,
    /// `frame_blocked_in` — src cannot produce as-is; avoid re-requesting.
    pub frame_blocked_in: bool,
    /// `FilterLink::frame_count_in` — frames pushed into the fifo.
    pub frame_count_in: u64,
    /// `FilterLink::frame_count_out` — frames taken out of the fifo.
    pub frame_count_out: u64,
    /// `FilterLinkInternal::init_state`.
    pub init_state: LinkInitState,
}

impl Default for Link {
    /// The zero-initialized C link as `avfilter_link` creates it
    /// (`avfilter.c:177-193`): format sentinel, zeroed rationals (C zeroes
    /// the struct; `Rational`'s `0/0` matches), `UNSPECIFIED` color fields.
    fn default() -> Self {
        Link {
            src: NodeId(0),
            srcpad: 0,
            dst: NodeId(0),
            dstpad: 0,
            format: None,
            w: 0,
            h: 0,
            sample_aspect_ratio: Rational::UNKNOWN,
            colorspace: ColorSpace::Unspecified,
            color_range: ColorRange::Unspecified,
            time_base: Rational::UNKNOWN,
            frame_rate: Rational::UNKNOWN,
            current_pts: NOPTS,
            incfg: FormatsConfig::default(),
            outcfg: FormatsConfig::default(),
            fifo: VecDeque::new(),
            status_in: None,
            status_in_pts: 0,
            status_out: None,
            frame_wanted_out: false,
            frame_blocked_in: false,
            frame_count_in: 0,
            frame_count_out: 0,
            init_state: LinkInitState::Uninit,
        }
    }
}

/// `li->status_in == status` — the early-return test of
/// `ff_avfilter_link_set_in_status` (`avfilter.c:254-255`) and the
/// `ret != li->status_in` triage of `request_frame_to_filter`
/// (`avfilter.c:545`). `Error` carries no `PartialEq` (the `Io` variant), so
/// this compares variant-by-variant; the string variants compare by message,
/// `Eof`/`Again`/unit variants by discriminant.
pub(crate) fn status_eq(a: &Error, b: &Error) -> bool {
    use Error::*;
    match (a, b) {
        (Eof, Eof) | (Again, Again) | (StreamNotFound, StreamNotFound)
        | (BufferTooSmall, BufferTooSmall) | (OutOfRange, OutOfRange) => true,
        (InvalidData(x), InvalidData(y))
        | (Unsupported(x), Unsupported(y))
        | (NotFound(x), NotFound(y))
        | (InvalidArgument(x), InvalidArgument(y)) => x == y,
        (Io(x), Io(y)) => x.kind() == y.kind() && x.to_string() == y.to_string(),
        _ => false,
    }
}

/// Clone a status `Error` (status values must be storable in both
/// `status_in` and `status_out`; `Error: !Clone` because of `io::Error`).
pub(crate) fn clone_status(e: &Error) -> Error {
    use Error::*;
    match e {
        Eof => Eof,
        Again => Again,
        InvalidData(s) => InvalidData(s.clone()),
        Unsupported(s) => Unsupported(s.clone()),
        Io(io) => Io(std::io::Error::new(io.kind(), io.to_string())),
        NotFound(s) => NotFound(s.clone()),
        StreamNotFound => StreamNotFound,
        BufferTooSmall => BufferTooSmall,
        OutOfRange => OutOfRange,
        InvalidArgument(s) => InvalidArgument(s.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_link_state_matches_c_zero_init() {
        let l = Link::default();
        // avfilter.c:190-192: format sentinel + UNSPECIFIED colorspace.
        assert_eq!(l.format, None);
        assert_eq!(l.colorspace, ColorSpace::Unspecified);
        assert_eq!(l.color_range, ColorRange::Unspecified);
        // Zeroed rationals (C memset; 0/0 == Rational::UNKNOWN).
        assert_eq!(l.time_base, Rational::UNKNOWN);
        assert_eq!(l.sample_aspect_ratio, Rational::UNKNOWN);
        assert_eq!(l.frame_rate, Rational::UNKNOWN);
        assert_eq!(l.w, 0);
        assert_eq!(l.h, 0);
        assert_eq!(l.current_pts, NOPTS);
        // Scheduling state idle.
        assert!(l.fifo.is_empty());
        assert!(l.status_in.is_none());
        assert!(l.status_out.is_none());
        assert!(!l.frame_wanted_out);
        assert!(!l.frame_blocked_in);
        assert_eq!(l.frame_count_in, 0);
        assert_eq!(l.frame_count_out, 0);
        assert_eq!(l.init_state, LinkInitState::Uninit);
    }

    #[test]
    fn formats_config_none_semantics() {
        // FormatsConfig starts fully undeclared...
        let cfg = FormatsConfig::default();
        assert!(cfg.formats.is_none());
        assert!(cfg.color_spaces.is_none());
        assert!(cfg.color_ranges.is_none());
        // ...and None means BOTH "not yet declared" (query_formats has not
        // run) AND "already picked" (pick_format unrefs to NULL — the None IS
        // the picked marker its early-return depends on). The distinction is
        // carried by link.format: Some(_) + None list = picked.
        let mut picked = cfg.clone();
        picked.formats = None;
        assert!(picked.formats.is_none());
        // A declared-not-picked half holds a ListIdx.
        let mut declared = cfg.clone();
        declared.formats = Some(0);
        assert_eq!(declared.formats, Some(0));
    }

    #[test]
    fn status_eq_distinguishes_variants() {
        assert!(status_eq(&Error::Eof, &Error::Eof));
        assert!(!status_eq(&Error::Eof, &Error::Again));
        assert!(status_eq(
            &Error::Unsupported("a".into()),
            &Error::Unsupported("a".into())
        ));
        assert!(!status_eq(
            &Error::Unsupported("a".into()),
            &Error::Unsupported("b".into())
        ));
        assert_eq!(clone_status(&Error::Eof).to_string(), "End of file");
    }
}
