//! libavfilter — the filtergraph (Phase 3).
//!
//! Port of the `libavfilter` core: filter/link/graph objects
//! (`avfilter.{h,c}`, `avfilter_internal.h`, `filters.h`), the activation
//! engine (`avfilter.c:216-560, 1018-1700`), format negotiation primitives
//! (`formats.{h,c}` + `libavutil/pixdesc.c`'s best-of-2 scoring), the graph
//! driver (`avfiltergraph.c`) and the filter set `buffer`/`buffersink`/
//! `scale`/`format`/`noformat`/`null`.
//!
//! ## Module map
//!
//! | file | ports | what it holds |
//! |---|---|---|
//! | [`link`] | `avfilter.h` link parts + `filters.h` `FilterLink` + `avfilter_internal.h` `FilterLinkInternal` + `framequeue.{h,c}` | flattened [`Link`], [`link::FormatsConfig`], status-protocol state |
//! | [`filter`] | `avfilter.c` | [`filter::FilterDef`]/[`filter::PadDef`]/[`filter::FilterFlags`], [`filter::FilterImpl`], [`filter::FilterNode`], the engine: `filter_frame`, `filter_frame_to_filter`, `request_frame_to_filter`, `ff_request_frame`, `set_ready`, [`filter::default_activate`], [`FilterGraph::run_once`] |
//! | [`formats`] | `formats.{h,c}` + `pixdesc.c:av_find_best_pix_fmt_of_2` | merge/can-merge + chroma/alpha guard, best-of-2 scoring, `regular_yuv`/`forced_full_range`, all-list builders |
//! | [`graph`] | `avfiltergraph.c` + `avfilter.c:149-326` | [`FilterGraph`]: arenas, create/alloc/init/link/insert, list plumbing, `config()`, runtime API stubs |
//! | [`vf_null`] | `vf_null.c` | the `null` filter |
//! | *(wave 2)* `parser` | `graphparser.c` + `avstring.c:av_get_token` | `get_token`, segment parse/apply, `graph_parse_ptr` |
//! | *(wave 2)* `buffersrc` / `buffersink` | `buffersrc.c` / `buffersink.c` | the source/sink filters + runtime API |
//! | *(wave 2/3)* `vf_scale`, `vf_format` | `vf_scale.c` + `scale_eval.c`; `vf_format.c` | `scale`, `format`, `noformat` |
//!
//! ## Wave 1 status
//!
//! This module lands in waves. Wave 1 (this) delivers the core contract:
//! link/filter types, the full status/activation engine, the negotiation
//! primitives, graph lifecycle + list plumbing, and the `null` filter. The
//! registry below intentionally returns `None` for the not-yet-ported names
//! (`buffer`, `buffersink`, `scale`, `format`, `noformat`) — simpler than
//! registering placeholder defs that error at init; the wave-2 subagents add
//! their `FilterDef`s here. `config()` runs `graph_check_validity` only;
//! `add_frame`/`close_source`/`get_frame`/`parse_ptr` are signature stubs.
//!
//! ## Conventions (crate-wide)
//!
//! * C names stay (`pts`, `time_base`, `incfg`, `frame_wanted_out`).
//! * `Result<_, Error>` with `Error::Eof`/`Error::Again` as VALUES — a link
//!   status is `Option<Error>`, never a panic.
//! * The pointer web (filter → pads → links → neighbors) becomes arena
//!   indices ([`NodeId`]/[`LinkId`]) into `FilterGraph`.
//! * `AVFilterFormats` refcounting becomes arena slots + [`link::ListIdx`]
//!   identity; merges sweep losers graph-wide (C's `MERGE_REF`).

pub mod buffersink;
pub mod buffersrc;
pub mod filter;
pub mod formats;
pub mod graph;
pub mod link;
pub mod parser;
pub mod vf_format;
pub mod vf_null;

// Crate-root re-exports mirror how C code includes libavfilter headers:
// `use crate::filter::{FilterGraph, NodeId, LinkId};`
pub use filter::{
    FilterDef, FilterFlags, FilterImpl, FilterNode, PadDef, PadRef, default_activate,
};
pub use graph::{Axis, FilterGraph, InOut};
pub use link::{FormatsConfig, Link, LinkId, LinkInitState, ListIdx, NodeId};

/// Parsed filter options — the `AVDictionary` of `avfilter_init_dict`
/// (avfilter.c:929-933): ordered `(key, value)` pairs, duplicate keys
/// allowed (C's `AV_DICT_MULTIKEY`), applied in insertion order.
///
/// Wave-1 stand-in: built by `graph::create_filter`'s simple `:`/`=`
/// splitter. The parser module (wave 2) takes over construction with the
/// real `av_get_token`-based `ff_filter_opt_parse` port (quoting/escaping,
/// shorthand rules) — this type then moves to `parser.rs` (re-exported from
/// here so call sites keep the `filter::Options` path).
#[derive(Clone, Debug, Default)]
pub struct Options {
    pub entries: Vec<(String, String)>,
}

/// `avfilter_get_by_name` (allfilters.c:658-671: a linear scan — so is this).
///
/// Registry: `null`, `format`, `noformat`, `buffer` (buffersrc),
/// `buffersink` are ported. `scale` arrives with wave 2C (vf_scale.c);
/// until then it resolves to `None` and `alloc_filter` fails with
/// `No such filter: 'scale'` — the same text C produces for an unknown
/// filter.
pub fn filter_def(name: &str) -> Option<&'static FilterDef> {
    match name {
        "buffer" => Some(&buffersrc::BUFFER_SRC_DEF),
        "buffersink" => Some(&buffersink::BUFFERSINK_DEF),
        "format" => Some(&vf_format::FORMAT_DEF),
        "noformat" => Some(&vf_format::NOFORMAT_DEF),
        "null" => Some(&vf_null::NULL_DEF),
        // Wave 2C: "scale"
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_wave2() {
        let def = filter_def("null").expect("null is registered");
        assert_eq!(def.name, "null");
        assert_eq!(def.inputs.len(), 1);
        assert_eq!(def.outputs.len(), 1);
        assert_eq!(def.inputs[0].name, "default");
        assert!(def.shorthand.is_empty());
        for (name, shorthand) in [
            ("buffer", "width"),
            ("buffersink", "pixel_formats"),
            ("format", "pix_fmts"),
            ("noformat", "pix_fmts"),
        ] {
            let def = filter_def(name).unwrap_or_else(|| panic!("{name} is registered"));
            assert_eq!(def.name, name);
            assert_eq!(def.shorthand.first(), Some(&shorthand), "{name}");
        }
        // Wave 2C: scale is not ported yet.
        assert!(filter_def("scale").is_none());
        assert!(filter_def("idet").is_none());
    }
}
