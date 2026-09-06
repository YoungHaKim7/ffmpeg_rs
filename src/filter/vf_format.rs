//! `format` and `noformat` video filters — port of
//! `libavfilter/vf_format.c` (all 228 lines, BOTH filters).
//!
//! One impl type ([`FormatContext`]) serves both [`FORMAT_DEF`] and
//! [`NOFORMAT_DEF`], branching only on the def name inside `init` — exactly
//! C's `strcmp(ctx->filter->name, "noformat")` (vf_format.c:136); the two C
//! descriptors share the class, the option table and the whole filter body.
//!
//! The filter is pure negotiation metadata: `init` parses the
//! `pix_fmts`/`color_spaces`/`color_ranges` option strings ('|'-separated),
//! `noformat` subtracts them from the port's all-lists
//! (`filter::formats`), `query_formats` hands the resulting lists to the
//! graph's `set_common_*` fill-if-unset plumbing on the crossed halves
//! (inputs' `outcfg` / outputs' `incfg`), and `filter_frame` is the framework
//! default forward (avfilter.c:1007-1010). The filter never CONVERTS pixels —
//! it only constrains negotiation; the actual conversion is the
//! auto-inserted `scale` filter's job. That graph-level machinery
//! (avfiltergraph.c:552-737 + pick_formats 1311-1357) is NOT ported in this
//! module: `format=rgb24` relies on a scale filter the caller/CLI inserts or
//! on a later wave's negotiation round. Do not expect this filter alone to
//! convert anything.
//!
//! ## Dropped from C (each also noted at its site)
//!
//! * the `alpha_modes` option and the whole alpha-mode negotiation axis —
//!   no `AVAlphaMode` enum and no alpha merger in the port (link.rs /
//!   formats.rs document the axis as out of scope); a graph string carrying
//!   `alpha_modes=...` leaves the entry unconsumed and init fails with
//!   `No such option: alpha_modes` — C's own unknown-option error shape,
//!   honest about the missing axis.
//! * the numeric `strtol` fallback of `DEFINE_PARSE` (vf_format.c:96-102) —
//!   names only. Consequences: `color_spaces=1` errors instead of Bt709, and
//!   EMPTY tokens (trailing '|', '||', `pix_fmts=`) are hard parse errors
//!   (`Invalid pixel format ''`) where C's tokenizer silently appends enum
//!   value 0 (yuv420p / Rgb / Unspecified). Strictly stricter, deliberate.
//! * C init's `ff_formats_ref` self-hold (vf_format.c:144-149) — the arena
//!   owns the lists; storing a `ListIdx` in the context would in fact be
//!   WRONG because the Rust merge sweep (`formats.rs` `sweep_list_idx`)
//!   retargets only LINK halves, unlike C's `MERGE_REF` which retargets every
//!   ref slot including `s->formats`. Hence the Vecs-in-context +
//!   fresh-alloc-per-query shape.
//! * `uninit` (vf_format.c:53-60) — `Vec::drop` is it; and the input pad's
//!   `ff_null_get_video_buffer` callback (no frame pools in the port).
//! * `AVFILTER_FLAG_METADATA_ONLY` (vf_format.c:196, 216) — ownership
//!   transfer IS the metadata-only semantics (vf_null.rs precedent).
//!
//! ## Legacy pixel-format names
//!
//! `av_get_pix_fmt`'s legacy `rgb32`/`bgr32` rewrite and `{name}` +
//! host-endian-suffix retry (pixdesc.c:3392-3410) are ported LOCALLY in
//! [`get_pix_fmt`] on top of `PixelFormat::from_name`, keeping this module
//! self-contained; other `from_name` call sites (CLI `-pix_fmt` etc.) do not
//! gain those aliases. Promote into `util::pixfmt.rs` if another wave needs
//! them there.

use crate::log_error;
use crate::util::color::{ColorRange, ColorSpace};
use crate::util::error::{Error, Result};
use crate::util::frame::Frame;
use crate::util::pixfmt::PixelFormat;

use super::filter::{filter_frame, FilterDef, FilterFlags, FilterImpl, PadDef};
use super::formats;
use super::graph::FilterGraph;
use super::link::NodeId;

// ---------------------------------------------------------------------------
// Private context (vf_format.c:40-51)
// ---------------------------------------------------------------------------

/// `FormatContext` (vf_format.c:40-51), reduced to the ported axes: the
/// parsed (and, for `noformat`, inverted) lists. `None` = option absent = no
/// restriction on that axis (C's NULL `AVFilterFormats`).
///
/// C's `char *pix_fmts/csps/ranges/alphamodes` option strings (42-45) are
/// consumed by `init` and not kept; the alpha-mode field/axis (50) is out of
/// scope. C's separate ref-held copies (47-50) collapse into these Vecs — the
/// arena slot is allocated fresh inside each `query_formats` call.
#[derive(Default)]
pub struct FormatContext {
    /// Parsed from `pix_fmts`.
    pub formats: Option<Vec<PixelFormat>>,
    /// Parsed from `color_spaces`.
    pub color_spaces: Option<Vec<ColorSpace>>,
    /// Parsed from `color_ranges`.
    pub color_ranges: Option<Vec<ColorRange>>,
}

// ---------------------------------------------------------------------------
// Name parsing (DEFINE_PARSE, vf_format.c:91-109)
// ---------------------------------------------------------------------------

/// `av_get_pix_fmt` (pixdesc.c:3392-3410), LE host only (the port has no
/// big-endian formats, so `X_NE` always picks its `le` argument):
///
/// 1. rewrite the legacy aliases — `"rgb32"` → `bgra`, `"bgr32"` → `rgba`
///    (pixdesc.c:3396-3399, `X_NE("argb","bgra")` / `X_NE("abgr","rgba")`);
/// 2. `PixelFormat::from_name` (the `get_pix_fmt_internal` analog — canonical
///    names plus the alias table entries pixfmt.rs already carries);
/// 3. retry with the host-endian suffix appended (`{name}le`,
///    pixdesc.c:3402-3407) — what makes `gray16`, `rgb565`, `yuv420p10`
///    resolve;
/// 4. else `AV_PIX_FMT_NONE`.
///
/// Like this C version's `av_get_pix_fmt`, never logs by itself.
fn get_pix_fmt(name: &str) -> Option<PixelFormat> {
    let name = match name {
        "rgb32" => "bgra", // X_NE("argb", "bgra") on LE
        "bgr32" => "rgba", // X_NE("abgr", "rgba") on LE
        n => n,
    };
    if let Some(f) = PixelFormat::from_name(name) {
        return Some(f);
    }
    PixelFormat::from_name(&format!("{name}le"))
}

/// The failure tail every `DEFINE_PARSE` instance shares (vf_format.c:98-100):
/// one ERROR log line (C's `av_log(log_ctx, AV_LOG_ERROR, ...)`; this C
/// `av_get_pix_fmt` never logs, so exactly one line per bad name comes from
/// the filter) plus `AVERROR(EINVAL)` → `Error::InvalidArgument` carrying the
/// same message text (the wave-1 pattern, cf. graph.rs `check_validity`).
/// `log_ctx` is the def name ("format"/"noformat"), mirroring C's item name.
fn parse_failed<T>(msg: String, log_ctx: &'static str) -> Result<T> {
    log_error!(Some(log_ctx), "{msg}\n");
    Err(Error::InvalidArgument(msg))
}

/// `parse_pixel_format` (DEFINE_PARSE instance, vf_format.c:107): name lookup
/// only — the `strtol(arg,&tail,0)` numeric fallback (96-102) is NOT ported
/// (C enum values ≠ our enum order, so any numeric mapping is guesswork), and
/// its empty-token quirk (strtol("")==0 with `*tail=='\0'` silently appending
/// enum value 0 = yuv420p) is deliberately replaced by a hard error.
fn parse_pixel_format(arg: &str, log_ctx: &'static str) -> Result<PixelFormat> {
    match get_pix_fmt(arg) {
        Some(f) => Ok(f),
        None => parse_failed(format!("Invalid pixel format '{arg}'"), log_ctx),
    }
}

/// `parse_color_space` (DEFINE_PARSE instance, vf_format.c:108):
/// `av_color_space_from_name` (pixdesc.c:3866-3877) is an exact `strcmp`
/// against `color_space_names` (pixdesc.c:3330-3346) — no alias matching for
/// the color enums (contrast `av_match_name` for pix_fmt aliases). Restricted
/// to the ported `ColorSpace` subset; every other C-table name (`ycgco`,
/// `bt2020c`, `smpte2085`, `chroma-derived-nc`, `chroma-derived-c`, `ictcp`,
/// `ipt-c2`, `ycgco-re`, `ycgco-ro`) is outside the enum subset → error.
/// Note the tables do NOT contain the enum spellings — `unspecified` or
/// `bt470bg-yuv`-style names are INVALID in C too.
fn parse_color_space(arg: &str, log_ctx: &'static str) -> Result<ColorSpace> {
    match arg {
        "gbr" => Ok(ColorSpace::Rgb),
        "bt709" => Ok(ColorSpace::Bt709),
        "unknown" => Ok(ColorSpace::Unspecified),
        "reserved" => Ok(ColorSpace::Reserved),
        "fcc" => Ok(ColorSpace::Fcc),
        "bt470bg" => Ok(ColorSpace::Bt470bg),
        "smpte170m" => Ok(ColorSpace::Smpte170m),
        "smpte240m" => Ok(ColorSpace::Smpte240m),
        "bt2020nc" => Ok(ColorSpace::Bt2020Ncl),
        _ => parse_failed(format!("Invalid color space '{arg}'"), log_ctx),
    }
}

/// `parse_color_range` (DEFINE_PARSE instance, vf_format.c:109):
/// `av_color_range_from_name` (pixdesc.c:3782-3792) over
/// `color_range_names` (pixdesc.c:3276-3280). The table spells `tv`/`pc`/
/// `unknown` — `mpeg`/`jpeg`/`unspecified` are invalid in C too.
fn parse_color_range(arg: &str, log_ctx: &'static str) -> Result<ColorRange> {
    match arg {
        "unknown" => Ok(ColorRange::Unspecified),
        "tv" => Ok(ColorRange::Mpeg),
        "pc" => Ok(ColorRange::Jpeg),
        _ => parse_failed(format!("Invalid color range '{arg}'"), log_ctx),
    }
}

// ---------------------------------------------------------------------------
// invert_formats (vf_format.c:62-89)
// ---------------------------------------------------------------------------

/// `invert_formats` — the `noformat` transform, generic over the axis.
///
/// * `if (!allfmts) return AVERROR(ENOMEM)` (65-66) — allocation failure,
///   not portable, dropped (no `Result`: C's only failure mode is ENOMEM).
/// * `if (!*fmts)` (67-71) — an ABSENT restriction list means NO restriction
///   regardless of filter type: the `None` stays `None`, the all-list is
///   discarded. This is what makes `noformat` with no options a pure
///   passthrough.
/// * Otherwise remove from `allfmts` every element present in `*fmts`,
///   PRESERVING `allfmts` order (C memmoves and decrements `nb_formats`,
///   `i--` rechecks the same index — `retain` is identical). Duplicates in
///   the input list are harmless (C's inner j-loop breaks on the first match;
///   `retain` is idempotent).
/// * Replace `*fmts` with the inverted list (86-87). Inverting against the
///   FULL universe can yield an EMPTY `Some(vec![])` — that is legal and
///   flows onward (an empty negotiable list; the graph then fails to
///   negotiate, which is downstream's business).
fn invert_formats<T: Copy + PartialEq>(fmts: &mut Option<Vec<T>>, allfmts: Vec<T>) {
    let Some(forbidden) = fmts.as_ref() else {
        return; // !*fmts: no restriction (vf_format.c:67-71)
    };
    let mut allfmts = allfmts;
    allfmts.retain(|f| !forbidden.contains(f));
    *fmts = Some(allfmts);
}

// ---------------------------------------------------------------------------
// FilterImpl — init / query_formats / filter_frame
// ---------------------------------------------------------------------------

impl FilterImpl for FormatContext {
    /// `init` (vf_format.c:112-152), minus the ref-hold tail (144-149, see
    /// module doc) and the alpha axis (134, 140).
    ///
    /// Step 0 — OPTION INTAKE: C's AVOption string fields (`s->pix_fmts`/
    /// `csps`/`ranges`) filled by the `AV_DICT_MULTIKEY` apply loop of
    /// `avfilter_init_dict` (avfilter.c:919-986), where the LAST occurrence
    /// of a repeated key wins (each `av_opt_set` overwrites the string
    /// field). Unrecognized entries are left in `opts` — `init_filter`'s
    /// leftover check then yields `No such option: <key>` (the wave-1 port
    /// of avfilter.c:976-980).
    ///
    /// Step 1 — PARSE_LIST (121-129): split on '|', parse and append each
    /// token IN ORDER; the FIRST bad token aborts init (earlier tokens were
    /// appended to the discarded list — irrelevant, C tears the filter
    /// down). Appending is the `ff_add_format` analog: plain push, NO dedup
    /// (formats.c:548-576 — `yuv420p|yuv420p` stays a 2-element list). Parse
    /// order: pix_fmts → color_spaces → color_ranges (131-134).
    ///
    /// Step 2 — the `noformat` branch (136-142): invert all three axes
    /// against their all-lists. Only the filter named exactly "noformat"
    /// inverts, read from the DEF (shared impl), per instance.
    ///
    /// Runs BEFORE links exist (FilterImpl contract) — never touches
    /// `g.links`. On Err, entries already consumed stay consumed — fine, C
    /// discards the failed filter too.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let log_ctx: &'static str = g.nodes[node.0].def.name;

        // Step 0 — drain the three known keys, last occurrence wins.
        let mut pix_fmts_arg: Option<String> = None;
        let mut csps_arg: Option<String> = None;
        let mut ranges_arg: Option<String> = None;
        let entries = std::mem::take(&mut g.nodes[node.0].opts.entries);
        let mut leftovers = Vec::new();
        for (key, value) in entries {
            match key.as_str() {
                "pix_fmts" => pix_fmts_arg = Some(value),
                "color_spaces" => csps_arg = Some(value),
                "color_ranges" => ranges_arg = Some(value),
                _ => leftovers.push((key, value)),
            }
        }
        g.nodes[node.0].opts.entries = leftovers;

        // Step 1 — parse each axis's '|'-separated list in order.
        // Divergence: an empty token (trailing/leading/doubled '|', or an
        // empty `pix_fmts=` value) is a hard error here; C's strtol
        // fallback would silently append enum value 0.
        if let Some(arg) = &pix_fmts_arg {
            let mut list = Vec::new();
            for tok in arg.split('|') {
                list.push(parse_pixel_format(tok, log_ctx)?);
            }
            self.formats = Some(list);
        }
        if let Some(arg) = &csps_arg {
            let mut list = Vec::new();
            for tok in arg.split('|') {
                list.push(parse_color_space(tok, log_ctx)?);
            }
            self.color_spaces = Some(list);
        }
        if let Some(arg) = &ranges_arg {
            let mut list = Vec::new();
            for tok in arg.split('|') {
                list.push(parse_color_range(tok, log_ctx)?);
            }
            self.color_ranges = Some(list);
        }

        // Step 2 — noformat inversion (vf_format.c:136-142) against the
        // port's all-lists (formats.rs:41-68 — the ff_all_formats(VIDEO)/
        // ff_all_color_spaces/ff_all_color_ranges equivalents; narrower than
        // C's universes, see module risks in formats.rs:34-40). Order
        // matters on the csp axis: all_color_spaces keeps Unspecified first
        // because pick_format takes element [0] on ties, so
        // `noformat=bt709` must still leave Unspecified at the head.
        if log_ctx == "noformat" {
            invert_formats(&mut self.formats, formats::all_pix_fmts());
            invert_formats(&mut self.color_spaces, formats::all_color_spaces());
            invert_formats(&mut self.color_ranges, formats::all_color_ranges());
        }

        // Step 3 — C's ref-hold (144-149) has NO Rust counterpart: arena
        // slots are graph-owned and never freed. uninit (53-60) is likewise
        // a no-op — Vec::drop is it.

        Ok(())
    }

    /// `query_formats` (vf_format.c:154-168, FILTER_QUERY_FUNC2 at 206/226):
    /// for each axis whose list is `Some`, declare it on the crossed halves
    /// of the filter's pads — inputs' `outcfg` + outputs' `incfg` — via
    /// `g.set_common_*` (C's `SET_COMMON_FORMATS2`, formats.c:978-1015, with
    /// the cfg arrays built crossed at avfiltergraph.c:373-390). Fill-if-unset:
    /// an already-declared half is never overwritten (formats.c:983). The
    /// SAME `ListIdx` goes on every pad — identity sharing is the point: one
    /// merge collapses all the filter's pads at once.
    ///
    /// Axes are independent (C's `||` short-circuit): an unset axis is
    /// simply not declared, and the ENGINE's `g.default_query_formats(node)`
    /// afterwards (avfiltergraph.c:411) fills the remaining axes with
    /// all-lists. DO NOT call `default_query_formats` from here. A `None`
    /// axis on format/noformat therefore means "no restriction on this
    /// axis"; an empty `Some(vec![])` DOES get declared (C: non-NULL empty
    /// list) — negotiation then fails downstream, which is correct.
    ///
    /// The arena slot is allocated FRESH per call (the parsed Vecs are
    /// stored in the context, never a `ListIdx`): C's `MERGE_REF` retargets
    /// every ref holder including `s->formats`, but the Rust sweep
    /// (`formats.rs` `sweep_list_idx`) only retargets LINK halves — a
    /// context-held `ListIdx` whose list lost a merge would go stale.
    /// Fresh-alloc also survives any future renegotiation re-query (arena
    /// slot leak at worst). The engine runs `query_formats` once per filter
    /// during config (avfiltergraph.c:346-411).
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        if let Some(fmts) = &self.formats {
            let list = g.alloc_pix_list(fmts.clone());
            g.set_common_formats(node, list)?;
        }
        if let Some(csps) = &self.color_spaces {
            let list = g.alloc_csp_list(csps.clone());
            g.set_common_color_spaces(node, list)?;
        }
        if let Some(ranges) = &self.color_ranges {
            let list = g.alloc_rng_list(ranges.clone());
            g.set_common_color_ranges(node, list)?;
        }
        Ok(())
    }

    /// `default_filter_frame` (avfilter.c:1007-1010 — vf_format defines NO
    /// filter_frame of its own): forward the frame to outputs[0] verbatim,
    /// identical to vf_null. The frame passes untouched — ownership transfer
    /// IS the METADATA_ONLY semantics.
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

// ---------------------------------------------------------------------------
// Filter descriptors (vf_format.c:171-228)
// ---------------------------------------------------------------------------

/// The shared single video pad (vf_format.c:182-188 input; the output is
/// `ff_video_default_filterpad`). The input's `ff_null_get_video_buffer`
/// callback is not ported — pad buffer callbacks do not exist (the engine
/// has no frame pools).
static DEFAULT_PAD: PadDef = PadDef { name: "default", needs_writable: false };

/// `ff_vf_format` (vf_format.c:190-208):
/// "Convert the input video to one of the specified pixel formats."
///
/// * `AVFILTER_FLAG_METADATA_ONLY` (196) maps to nothing — see module doc.
/// * flags carry [`FilterFlags::ALLOWS_RECONFIGURE`]: "format" IS in C's
///   `ff_filter_frame` validation skip list (avfilter.c:1075-1082: buffersink,
///   format, idet, null, scale, libplacebo, hqdn3d) — the filter genuinely
///   can see frames differing from a not-yet-renegotiated link format during
///   auto-convert insertion, so the debug asserts must be skipped for it.
/// * shorthand is `["pix_fmts"]` per the crate convention (filter.rs:79-84).
///   Divergence: C's `ff_filter_opt_parse` derives shorthand from the AVOption
///   table in declaration order (avfilter.c:855-902), so C would positionally
///   fill `color_spaces` (2nd) and `color_ranges` (3rd) too —
///   `format=yuv420p:bt709` is valid C. The port restricts positional to
///   `pix_fmts`; users must write explicit `key=value` for csps/ranges.
pub static FORMAT_DEF: FilterDef = FilterDef {
    name: "format",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags::ALLOWS_RECONFIGURE,
    shorthand: &["pix_fmts"],
    make: || Box::new(FormatContext::default()),
};

/// `ff_vf_noformat` (vf_format.c:210-228):
/// "Force libavfilter not to use any of the specified pixel formats for the
/// input to the next filter."
///
/// Identical to [`FORMAT_DEF`] except the name and the flags: `noformat` is
/// ABSENT from C's avfilter.c:1075-1082 validation-skip list, so it carries
/// `FilterFlags(0)` — its links keep the frame-vs-link debug asserts.
pub static NOFORMAT_DEF: FilterDef = FilterDef {
    name: "noformat",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags(0),
    shorthand: &["pix_fmts"],
    make: || Box::new(FormatContext::default()),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::graph::engine_test_helpers::run_to_quiescence;
    use crate::filter::filter_def;
    use crate::filter::link::LinkId;

    /// src -> filter(args) -> sink (the engine-test endpoints stand in for
    /// buffersrc/buffersink until wave 2).
    fn chain(filter: &str, args: &str) -> (FilterGraph, NodeId, LinkId, LinkId) {
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let f = g.create_filter(filter, args).expect("filter created");
        let sink = g.alloc_test_sink();
        let lin = g.link(src, 0, f, 0).unwrap();
        let lout = g.link(f, 0, sink, 0).unwrap();
        (g, f, lin, lout)
    }

    /// The engine step at avfiltergraph.c:392: call the node's
    /// query_formats (imp taken out of the node for the duration).
    fn run_query_formats(g: &mut FilterGraph, node: NodeId) {
        let mut imp = g.nodes[node.0].imp.take().expect("imp present");
        let ret = imp.query_formats(g, node);
        g.nodes[node.0].imp = Some(imp);
        ret.unwrap();
    }

    // ---- parse: crossed halves + engine default query -----------------------

    #[test]
    fn format_parse_and_query_sets_both_crossed_halves() {
        let (mut g, f, lin, lout) = chain("format", "yuv420p|rgb24");
        run_query_formats(&mut g, f);
        // SET_COMMON_FORMATS2 crossing (formats.c:978-1015 +
        // avfiltergraph.c:373-390): the INPUT link's outcfg and the OUTPUT
        // link's incfg hold the SAME list (identity sharing)...
        let in_half = g.links[lin.0].outcfg.formats.expect("input outcfg declared");
        let out_half = g.links[lout.0].incfg.formats.expect("output incfg declared");
        assert_eq!(in_half, out_half);
        // ...whose arena vec is the option order.
        assert_eq!(
            g.fmt_lists[in_half as usize],
            vec![PixelFormat::Yuv420p, PixelFormat::Rgb24]
        );
        // Then the engine step (avfiltergraph.c:411): default_query_formats
        // fills the REMAINING axes with the all-lists; the declared pix axis
        // survives (fill-if-unset, formats.c:983).
        g.default_query_formats(f).unwrap();
        assert_eq!(g.links[lin.0].outcfg.formats, Some(in_half));
        assert_eq!(g.links[lout.0].incfg.formats, Some(out_half));
        let csp = g.links[lin.0].outcfg.color_spaces.expect("csp filled");
        assert_eq!(g.csp_lists[csp as usize], formats::all_color_spaces());
        assert!(g.links[lout.0].incfg.color_spaces.is_some());
        assert!(g.links[lin.0].outcfg.color_ranges.is_some());
    }

    // ---- parse: errors ------------------------------------------------------

    #[test]
    fn format_unknown_name_error() {
        let mut g = FilterGraph::new();
        match g.create_filter("format", "yuv420p|bogus").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid pixel format 'bogus'"),
            other => panic!("unexpected error: {other}"),
        }
        // First bad token aborts init — later tokens are never reached.
        match g.create_filter("format", "bogus|yuv420p").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid pixel format 'bogus'"),
            other => panic!("unexpected error: {other}"),
        }
        match g.create_filter("format", "color_spaces=nope").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid color space 'nope'"),
            other => panic!("unexpected error: {other}"),
        }
        // The C tables spell tv/pc/unknown (pixdesc.c:3276-3280) — the enum
        // spelling 'mpeg' is invalid in C too.
        match g.create_filter("format", "color_ranges=mpeg").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid color range 'mpeg'"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn format_empty_token_rejected() {
        // Trailing '|' yields one empty token — a hard error in the port
        // (C's strtol fallback would silently append enum value 0).
        let mut g = FilterGraph::new();
        match g.create_filter("format", "yuv420p|").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid pixel format ''"),
            other => panic!("unexpected error: {other}"),
        }
        // Same for an empty explicit value and a doubled separator.
        match g.create_filter("format", "pix_fmts=").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid pixel format ''"),
            other => panic!("unexpected error: {other}"),
        }
        match g.create_filter("format", "gray8||gray16").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid pixel format ''"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn numeric_and_positional_rejections() {
        // strtol numeric fallback not ported (vf_format.c:96-102).
        let mut g = FilterGraph::new();
        match g.create_filter("format", "color_spaces=1").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid color space '1'"),
            other => panic!("unexpected error: {other}"),
        }
        // Explicit key=value disables the rest of the shorthand chain
        // (avfilter.c:889-892) — and the port's shorthand is pix_fmts only,
        // so 'bt709' has no positional slot at all.
        match g.create_filter("format", "pix_fmts=yuv420p:bt709").unwrap_err() {
            Error::InvalidArgument(msg) => assert_eq!(msg, "No option name near 'bt709'"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn last_option_occurrence_wins() {
        // AV_DICT_MULTIKEY apply order (avfilter.c:919-986): each av_opt_set
        // overwrites the string field, so the LAST occurrence wins.
        let (mut g, f, _lin, lout) = chain("format", "pix_fmts=yuv420p:pix_fmts=rgb24");
        run_query_formats(&mut g, f);
        let idx = g.links[lout.0].incfg.formats.expect("declared");
        assert_eq!(g.fmt_lists[idx as usize], vec![PixelFormat::Rgb24]);
    }

    #[test]
    fn alpha_modes_rejected_as_unknown_option() {
        // The alpha axis is out of scope: the option is not in the ported
        // table, so the entry is a leftover — init_filter's check is C's own
        // unknown-option error shape (avfilter.c:976-980).
        let mut g = FilterGraph::new();
        match g.create_filter("format", "alpha_modes=straight").unwrap_err() {
            Error::NotFound(msg) => assert_eq!(msg, "No such option: alpha_modes"),
            other => panic!("unexpected error: {other}"),
        }
    }

    // ---- parse: aliases -----------------------------------------------------

    #[test]
    fn format_alias_resolution() {
        for (arg, want) in [
            ("gray", PixelFormat::Gray8),          // from_name alias
            ("gray8", PixelFormat::Gray8),         // from_name alias
            ("gray16", PixelFormat::Gray16le),     // le-retry (pixdesc.c:3402-3407)
            ("rgb32", PixelFormat::Bgra),          // LE rewrite (pixdesc.c:3396-3399)
            ("bgr32", PixelFormat::Rgba),          // LE rewrite
            ("yuv420p10", PixelFormat::Yuv420p10le),
        ] {
            let (mut g, f, _lin, lout) = chain("format", arg);
            run_query_formats(&mut g, f);
            let idx = g.links[lout.0].incfg.formats.expect("declared");
            assert_eq!(g.fmt_lists[idx as usize], vec![want], "arg '{arg}'");
        }
    }

    #[test]
    fn format_duplicates_kept() {
        // ff_add_format does NOT dedup (formats.c:548-576).
        let (mut g, f, _lin, lout) = chain("format", "yuv420p|yuv420p");
        run_query_formats(&mut g, f);
        let idx = g.links[lout.0].incfg.formats.expect("declared");
        assert_eq!(
            g.fmt_lists[idx as usize],
            vec![PixelFormat::Yuv420p, PixelFormat::Yuv420p]
        );
    }

    // ---- noformat inversion --------------------------------------------------

    #[test]
    fn noformat_inverts_against_all_preserving_order() {
        // vf_format.c:73-84 memmove semantics == retain.
        let (mut g, f, lin, lout) = chain("noformat", "yuv420p");
        run_query_formats(&mut g, f);
        let in_half = g.links[lin.0].outcfg.formats.expect("declared");
        let out_half = g.links[lout.0].incfg.formats.expect("declared");
        assert_eq!(in_half, out_half, "both pads share one list");
        let expected: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&p| p != PixelFormat::Yuv420p)
            .collect();
        assert_eq!(g.fmt_lists[in_half as usize], expected);
    }

    #[test]
    fn noformat_covering_all_yields_empty_list() {
        // Forbidding the whole universe is legal: an empty Some(vec![]) is
        // still declared on the pads (C: non-NULL empty list).
        let args = PixelFormat::ALL
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join("|");
        let (mut g, f, lin, lout) = chain("noformat", &args);
        run_query_formats(&mut g, f);
        let in_half = g.links[lin.0].outcfg.formats.expect("declared");
        let out_half = g.links[lout.0].incfg.formats.expect("declared");
        assert_eq!(in_half, out_half);
        assert!(g.fmt_lists[in_half as usize].is_empty());
    }

    #[test]
    fn noformat_without_options_is_unrestricted() {
        // invert_formats' !*fmts early-out (vf_format.c:67-71): no options
        // means NO restriction regardless of filter type.
        let (mut g, f, lin, lout) = chain("noformat", "");
        run_query_formats(&mut g, f);
        assert!(g.links[lin.0].outcfg.formats.is_none());
        assert!(g.links[lout.0].incfg.formats.is_none());
        assert!(g.links[lin.0].outcfg.color_spaces.is_none());
        assert!(g.links[lout.0].incfg.color_ranges.is_none());
        // The engine's default query then fills the all-lists.
        g.default_query_formats(f).unwrap();
        assert!(g.links[lin.0].outcfg.formats.is_some());
        assert!(g.links[lout.0].incfg.formats.is_some());
    }

    #[test]
    fn invert_formats_none_stays_none_and_tolerates_duplicates() {
        // Direct unit test of the private helper (vf_format.c:62-89).
        let mut fmts: Option<Vec<PixelFormat>> = None;
        invert_formats(&mut fmts, formats::all_pix_fmts());
        assert!(fmts.is_none(), "absent list stays absent");
        // Duplicate forbidden entries are harmless; order is preserved.
        let mut some = Some(vec![PixelFormat::Yuv420p, PixelFormat::Yuv420p, PixelFormat::Rgb24]);
        invert_formats(&mut some, formats::all_pix_fmts());
        let expected: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&p| p != PixelFormat::Yuv420p && p != PixelFormat::Rgb24)
            .collect();
        assert_eq!(some, Some(expected));
    }

    // ---- color axes -----------------------------------------------------------

    #[test]
    fn color_axis_names_and_noformat_inversion() {
        // format declares only the named axis; formats stays untouched (None).
        let (mut g, f, lin, lout) = chain("format", "color_spaces=gbr|bt709");
        run_query_formats(&mut g, f);
        assert!(g.links[lin.0].outcfg.formats.is_none());
        assert!(g.links[lout.0].incfg.formats.is_none());
        let idx = g.links[lout.0].incfg.color_spaces.expect("csp declared");
        assert_eq!(
            g.csp_lists[idx as usize],
            vec![ColorSpace::Rgb, ColorSpace::Bt709]
        );

        // noformat inverts against ff_all_color_spaces (formats.c:698-710)
        // with Unspecified STILL FIRST — pick_format takes element [0] on
        // ties, so the head must survive the subtraction.
        let (mut g, f, _lin, lout) = chain("noformat", "color_spaces=bt709|smpte170m");
        run_query_formats(&mut g, f);
        let idx = g.links[lout.0].incfg.color_spaces.expect("csp declared");
        assert_eq!(
            g.csp_lists[idx as usize],
            vec![
                ColorSpace::Unspecified,
                ColorSpace::Rgb,
                ColorSpace::Fcc,
                ColorSpace::Bt470bg,
                ColorSpace::Smpte240m,
                ColorSpace::Bt2020Ncl,
            ]
        );

        let (mut g, f, _lin, lout) = chain("format", "color_ranges=tv|pc");
        run_query_formats(&mut g, f);
        let idx = g.links[lout.0].incfg.color_ranges.expect("rng declared");
        assert_eq!(g.rng_lists[idx as usize], vec![ColorRange::Mpeg, ColorRange::Jpeg]);

        let (mut g, f, _lin, lout) = chain("noformat", "color_ranges=tv");
        run_query_formats(&mut g, f);
        let idx = g.links[lout.0].incfg.color_ranges.expect("rng declared");
        assert_eq!(
            g.rng_lists[idx as usize],
            vec![ColorRange::Unspecified, ColorRange::Jpeg]
        );
    }

    #[test]
    fn parse_functions_accept_and_reject_names() {
        // Direct calls; log_ctx mirrors the def name like C's av_log item.
        assert_eq!(
            parse_pixel_format("yuv420p", "format").unwrap(),
            PixelFormat::Yuv420p
        );
        assert_eq!(parse_color_space("bt2020nc", "format").unwrap(), ColorSpace::Bt2020Ncl);
        assert_eq!(parse_color_space("unknown", "noformat").unwrap(), ColorSpace::Unspecified);
        assert_eq!(parse_color_range("tv", "format").unwrap(), ColorRange::Mpeg);
        assert_eq!(parse_color_range("pc", "format").unwrap(), ColorRange::Jpeg);
        // C color_space_names outside the ported enum subset -> error.
        for bad in [
            "ycgco",
            "bt2020c",
            "smpte2085",
            "chroma-derived-nc",
            "chroma-derived-c",
            "ictcp",
            "ipt-c2",
        ] {
            assert!(parse_color_space(bad, "format").is_err(), "{bad}");
        }
        // Enum spellings the C tables do not carry.
        for bad in ["unspecified", "mpeg", "jpeg"] {
            assert!(parse_color_range(bad, "format").is_err(), "{bad}");
            assert!(parse_color_space(bad, "format").is_err(), "{bad}");
        }
    }

    // ---- data path -----------------------------------------------------------

    #[test]
    fn filter_frame_forwards_verbatim() {
        let (mut g, _f, lin, lout) = chain("format", "yuv420p");
        // Configure the links' geometry as a real config() would
        // (pick_format + config_props — wave-2 territory, done by hand).
        for l in [lin, lout] {
            g.links[l.0].format = Some(PixelFormat::Gray8);
            g.links[l.0].w = 8;
            g.links[l.0].h = 8;
        }
        let mut frame = Frame::alloc(PixelFormat::Gray8, 8, 8).unwrap();
        frame.pts = 5;
        let pixels_before: Vec<u8> = frame.plane(0).to_vec();
        filter_frame(&mut g, lin, frame).unwrap();
        // One activation drains the format node (default_filter_frame,
        // avfilter.c:1007-1010): the frame queues on the sink side VERBATIM.
        g.run_once().unwrap();
        let out = g.links[lout.0].fifo.front().expect("frame forwarded");
        assert_eq!(out.pts, 5);
        assert_eq!(out.format, PixelFormat::Gray8);
        assert_eq!(out.width, 8);
        assert_eq!(out.height, 8);
        assert_eq!(out.plane(0), &pixels_before[..]);
        // Run to quiescence: the sink consumes it.
        run_to_quiescence(&mut g);
        assert_eq!(g.links[lout.0].fifo.len(), 0, "sink consumed the frame");
        for (name, l) in [("lin", lin), ("lout", lout)] {
            assert_eq!(g.links[l.0].frame_count_in, 1, "{name} in");
            assert_eq!(g.links[l.0].frame_count_out, 1, "{name} out");
        }
    }

    // ---- defs / registry -------------------------------------------------------

    #[test]
    fn flags_split_between_the_two_filters() {
        // "format" is in avfilter.c:1075-1082's validation-skip list;
        // "noformat" is not.
        assert!(FORMAT_DEF.flags.contains(FilterFlags::ALLOWS_RECONFIGURE));
        assert_eq!(NOFORMAT_DEF.flags, FilterFlags(0));
    }

    #[test]
    fn registry_and_def_shape() {
        let f = filter_def("format").expect("format registered");
        let n = filter_def("noformat").expect("noformat registered");
        assert!(std::ptr::eq(f, &FORMAT_DEF));
        assert!(std::ptr::eq(n, &NOFORMAT_DEF));
        for def in [f, n] {
            assert_eq!(def.inputs.len(), 1);
            assert_eq!(def.outputs.len(), 1);
            assert_eq!(def.inputs[0].name, "default");
            assert_eq!(def.outputs[0].name, "default");
            assert!(!def.inputs[0].needs_writable);
            assert_eq!(def.shorthand, &["pix_fmts"][..]);
        }
        // With buffer/buffersink landed alongside, only wave-2C scale
        // remains unregistered.
        for not_yet in ["scale"] {
            assert!(filter_def(not_yet).is_none(), "{not_yet}");
        }
        assert!(filter_def("null").is_some());
        assert!(filter_def("buffer").is_some());
        assert!(filter_def("buffersink").is_some());
    }
}
