//! `buffersink` — port of `libavfilter/buffersink.c` (the video sink) plus
//! the option-parsing bits of `libavutil/opt.c` its AVOption table rides on
//! (`opt_set_array` opt.c:805-895, `set_string_fmt`/`set_string_pixel_fmt`
//! opt.c:613-670, `set_string_number` opt.c:426-518, `write_number`
//! opt.c:275-286).
//!
//! THE structural fact (this FFmpeg's buffersink has no private fifo): frames
//! stay queued in the input link's framequeue ([`Link::fifo`], wave 1) and are
//! pulled directly by the runtime API via
//! [`filter::inlink_consume_frame`]. The sink's `activate`
//! (buffersink.c:196-212) is deliberately a no-op — "The frame is queued, the
//! rest is up to get_frame_internal" (buffersink.c:210) — whose only behavior
//! is the queued-frames warning. The only frame the context itself holds is
//! `peeked_frame` (buffersink.c:71), implementing
//! `AV_BUFFERSINK_FLAG_PEEK`. [`Frame::clone`] is `av_frame_ref` (shallow
//! Arc bumps), so PEEK stores the consumed frame and hands back a clone; the
//! plain get moves the frame out (`av_frame_move_ref`, buffersink.c:87).
//!
//! Not ported (audio/hw/alpha axes, stated per site below): `abuffersink`
//! (buffersink.c:395-414), `asink_query_formats` (326-350), `get_frame_internal`'s
//! samples branch / `av_buffersink_get_samples` (141-145), the `min_samples`
//! plumbing of `av_buffersink_get_frame_flags` (135-139 — video links have
//! `min_samples == 0`, so the ternary at buffersink.c:107-108 collapses to
//! the frame path), `config_input_audio` (214-222),
//! `av_buffersink_set_frame_size` (224-233 — audio-only; the Rust [`Link`]
//! has no min/max_samples), the `alphamodes` option (361-362) and
//! `av_buffersink_get_alpha_mode` (250 — alpha-mode negotiation out of
//! scope), `av_buffersink_get_hw_frames_ctx` (261-266) and
//! `av_buffersink_get_side_data` (287-293), `av_buffersink_get_type` (241),
//! the channels/ch_layout/sample_rate accessors (252, 268-285), the
//! `TERMINATE_ARRAY` sentinels (155-162 — `Vec` carries its length; C needs
//! them only to terminate iteration), and the filter description string
//! (`FilterDef` has no description field).
//!
//! Divergences from C semantics (all forced by the wave-1 engine or the Rust
//! type system, each also marked at its site):
//!
//! * D1 — **starved latch**: C arms `buffersrc_empty` on
//!   `FFERROR_BUFFERSRC_EMPTY` (filters.h:35; produced by buffersrc.c:608,
//!   propagated by `ff_filter_graph_run_once`, avfiltergraph.c:1616-1633).
//!   Wave-1's `run_once` cannot produce that sentinel; the loop arms its
//!   latch on the re-request path instead — one extra graph round at most,
//!   `Err(Error::Again)` surfaces in every case C's does.
//! * D2 — **EOF idempotence**: C's `ff_inlink_acknowledge_status`
//!   (avfilter.c:1472-1474) returns an already-set `status_out` as a truthy
//!   negative, so repeated post-EOF gets keep returning EOF; wave-1's
//!   `Option` port reports `None` there. A local `status_out` re-check inside
//!   the get loop compensates.
//! * D3 — **option parsing subset**: pixel formats parse by NAME only (C's
//!   numeric-strtol fallback and `"none"`, opt.c:620-626, are dropped);
//!   `colorspaces`/`colorranges` elements parse as plain decimal ints (C's
//!   `av_expr` accepts hex, arithmetic and the constants
//!   `none`/`all`/`max`/`min`, opt.c:484-498); an in-range int with no Rust
//!   enum variant (e.g. colorspace 8, C `AVCOL_SPC_RESERVED`) errors with the
//!   out-of-range text where C would carry the raw discriminant into
//!   negotiation.

use crate::{
    log_error, log_warning,
    util::{
        color::{ColorRange, ColorSpace},
        error::{Error, Result},
        frame::Frame,
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::{
    filter::{self, FilterDef, FilterFlags, FilterImpl, PadDef},
    graph::FilterGraph,
    link::{NodeId, clone_status},
};

// ---------------------------------------------------------------------------
// Context (buffersink.c:43-72)
// ---------------------------------------------------------------------------

/// The `buffersink` private context — `BufferSinkContext`
/// (buffersink.c:43-72), video fields only.
///
/// Not ported from the C struct: `frame_size` (46, audio only), `alphamodes`
/// (58-59, alpha axis out of scope), `sample_formats`/`samplerates`/
/// `channel_layouts` (62-69, audio).
pub struct BufferSinkContext {
    /// Queued-frames warning threshold (C:45): starts at 100
    /// (`common_init`, buffersink.c:151), escalates ×10 each time it fires,
    /// 0 disables permanently.
    warning_limit: u32,
    /// `pixel_formats`/`nb_pixel_formats` (C:49-50) — `Vec::len()` is C's
    /// count; duplicates kept, order preserved (ff_add_format appends
    /// unconditionally, formats.c:572-577).
    pixel_formats: Vec<PixelFormat>,
    /// `colorspaces`/`nb_colorspaces` (C:53-54), C discriminants.
    colorspaces: Vec<ColorSpace>,
    /// `colorranges`/`nb_colorranges` (C:55-56), C discriminants.
    colorranges: Vec<ColorRange>,
    /// `peeked_frame` (C:71) — the one frame the context owns: the head frame
    /// parked by `AV_BUFFERSINK_FLAG_PEEK`, shadowing newer queue arrivals
    /// until a plain get takes it.
    peeked_frame: Option<Frame>,
}

impl Default for BufferSinkContext {
    /// The post-`common_init` zero state: `warning_limit` 100 (C:45 zeroed by
    /// calloc, set to 100 at init; the port bakes the 100 in — `init` sets it
    /// again, idempotently), empty vecs, no peeked frame.
    fn default() -> Self {
        BufferSinkContext {
            warning_limit: 100,
            pixel_formats: Vec::new(),
            colorspaces: Vec::new(),
            colorranges: Vec::new(),
            peeked_frame: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Flags (buffersink.h:85-92)
// ---------------------------------------------------------------------------

/// `AV_BUFFERSINK_FLAG_*` (buffersink.h:85-92).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuffersinkFlags(pub u32);

impl BuffersinkFlags {
    /// `AV_BUFFERSINK_FLAG_PEEK` (1): return a clone of the head frame
    /// without consuming the peek state (buffersink.h:85).
    pub const PEEK: BuffersinkFlags = BuffersinkFlags(1 << 0);
    /// `AV_BUFFERSINK_FLAG_NO_REQUEST` (1<<1): never request frames from the
    /// graph — `Err(Again)` immediately when starved (buffersink.h:92).
    pub const NO_REQUEST: BuffersinkFlags = BuffersinkFlags(1 << 1);

    pub const fn contains(self, other: BuffersinkFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for BuffersinkFlags {
    type Output = BuffersinkFlags;
    fn bitor(self, rhs: BuffersinkFlags) -> BuffersinkFlags {
        BuffersinkFlags(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Filter definition (buffersink.c:382-393)
// ---------------------------------------------------------------------------

const DEFAULT_PAD: PadDef = PadDef {
    name: "default",
    needs_writable: false,
};

/// `ff_vsink_buffer` (buffersink.c:382-393).
///
/// * inputs: `ff_video_default_filterpad` (video.c:37-41) — one pad
///   `"default"`, no callbacks (C's pad has no `filter_frame`; see
///   [`BufferSinkContext`]'s `filter_frame` for what that means).
/// * outputs: `.p.outputs = NULL` (buffersink.c:386) — the graph endpoint.
/// * `ALLOWS_RECONFIGURE`: buffersink is in `ff_filter_frame`'s
///   skip-validation name list (avfilter.c:1076-1082; ported subset noted in
///   filter.rs:49-52).
/// * shorthand: the priv_class options in declaration order
///   (avfilter.c:863-866): a positional value maps to `pixel_formats` first,
///   then `colorspaces`, then `colorranges` (`alphamodes` dropped).
/// * C's description string is dropped (`FilterDef` has no such field).
pub static BUFFERSINK_DEF: FilterDef = FilterDef {
    name: "buffersink",
    inputs: &[DEFAULT_PAD],
    outputs: &[],
    flags: FilterFlags::ALLOWS_RECONFIGURE,
    shorthand: &["pixel_formats", "colorspaces", "colorranges"],
    make: || Box::new(BufferSinkContext::default()),
};

// ---------------------------------------------------------------------------
// FilterImpl (init / query_formats / activate / filter_frame)
// ---------------------------------------------------------------------------

impl FilterImpl for BufferSinkContext {
    /// `init_video` + `common_init` (buffersink.c:147-187) fed by the option
    /// dict `avfilter_init_dict` applies via `av_opt_set` BEFORE init
    /// (avfilter.c:929-933): consume the recognized entries
    /// (`pixel_formats`/`colorspaces`/`colorranges`) out of
    /// `g.nodes[node].opts`; unknown keys stay for `init_filter`'s leftover
    /// "No such option" error (graph.rs:168-170).
    ///
    /// A duplicate KEY replaces the whole array — C's `opt_set_array` frees
    /// the previous array before assigning (opt.c:865-866, 876-878) — so each
    /// recognized entry simply overwrites the field. `TERMINATE_ARRAY`
    /// (buffersink.c:155-162) is not ported: its sentinels exist only to
    /// terminate C's raw-array iteration; `Vec` carries its length.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let ctx_name = g.nodes[node.0].name.clone();
        let entries = std::mem::take(&mut g.nodes[node.0].opts.entries);
        let mut leftovers: Vec<(String, String)> = Vec::new();
        for (key, value) in entries {
            let assigned = match key.as_str() {
                "pixel_formats" => parse_pixel_formats(&value).map(|v| self.pixel_formats = v),
                "colorspaces" => parse_enum_array(&value, "colorspaces", colorspace_from_i64)
                    .map(|v| self.colorspaces = v),
                "colorranges" => parse_enum_array(&value, "colorranges", colorrange_from_i64)
                    .map(|v| self.colorranges = v),
                // "alphamodes" (buffersink.c:361-362) is not ported (alpha
                // axis out of scope) — it stays a leftover like any unknown
                // key and fires the same "No such option" error.
                _ => {
                    leftovers.push((key, value));
                    continue;
                }
            };
            if let Err(e) = assigned {
                // Verbatim C text, logged with the filter instance as context
                // (C: av_log(ctx, ...), opt.c:500-501 / 628-629 / 283-284).
                if let Error::InvalidArgument(msg) = &e {
                    log_error!(Some(ctx_name.as_str()), "{msg}\n");
                }
                g.nodes[node.0].opts.entries = leftovers; // best effort; init fails anyway
                return Err(e);
            }
        }
        g.nodes[node.0].opts.entries = leftovers;
        // common_init (buffersink.c:147-153).
        self.warning_limit = 100;
        Ok(())
    }

    /// `vsink_query_formats` (buffersink.c:295-324): for each NON-EMPTY
    /// option vec, in this exact order, declare it as a fresh arena list via
    /// the fill-if-unset common setters — which for a sink (no outputs) means
    /// exactly the single input link's `outcfg` half (the dst-accepts
    /// declarations; graph.rs:364-412 crossing). An empty vec leaves the axis
    /// untouched (C's `nb_ == 0` skip) so the ENGINE's
    /// `g.default_query_formats` — which runs AFTER this returns
    /// (avfiltergraph.c:410) — fills it with the all-list. The `alphamodes`
    /// branch (317-321) is not ported; `asink_query_formats` (326-350) is
    /// audio and not ported.
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        if !self.pixel_formats.is_empty() {
            // ff_set_pixel_formats_from_list2 (formats.c:1160-1168) — the
            // Rust Vec needs no terminator.
            let list = g.alloc_pix_list(self.pixel_formats.clone());
            g.set_common_formats(node, list)?;
        }
        if !self.colorspaces.is_empty() {
            let list = g.alloc_csp_list(self.colorspaces.clone());
            g.set_common_color_spaces(node, list)?;
        }
        if !self.colorranges.is_empty() {
            let list = g.alloc_rng_list(self.colorranges.clone());
            g.set_common_color_ranges(node, list)?;
        }
        Ok(())
    }

    /// `activate` (buffersink.c:196-212). Deliberately NOT the default
    /// activation: stage B of `default_activate` would drain a queued frame
    /// into `filter_frame` (whose C pad default would forward to a NULL
    /// output) and stage E would request frames on the app's behalf; C's
    /// custom activate does neither. All it does is warn when too many frames
    /// pile up (the app is not calling `get_frame`), escalating the threshold
    /// ×10 each time (wrapping like C's `unsigned`, buffersink.c:207);
    /// `warning_limit == 0` disables permanently. Also a no-op when only a
    /// status arrived (`set_in_status` wakes the sink at 200) — EOF is
    /// surfaced solely by the get loop's acknowledge step.
    fn activate(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let inlink = g.inlink(node, 0);
        let queued = g.links[inlink.0].fifo.len();
        if self.warning_limit != 0 && queued >= self.warning_limit as usize {
            // ff_framequeue_queued_frames — frames, not bytes/samples. The
            // FIRST %d is the threshold (100/1000/…), NOT the queued count;
            // %s is the instance name (av_x_if_null(ctx->name,
            // ctx->filter->name) — FilterNode::name is always set).
            let name = g.nodes[node.0].name.clone();
            log_warning!(
                Some(name.as_str()),
                "{} buffers queued in {}, something may be wrong.\n",
                self.warning_limit,
                name
            );
            self.warning_limit = self.warning_limit.wrapping_mul(10);
        }
        // "The frame is queued, the rest is up to get_frame_internal"
        // (buffersink.c:210).
        Ok(())
    }

    /// NEVER reachable: `activate` never drains, so `filter_frame` is never
    /// dispatched. C's pad (`ff_video_default_filterpad`, video.c:37-41) has
    /// no `filter_frame` callback; the framework default
    /// (`default_filter_frame`, avfilter.c:1007-1010) would forward to
    /// `ctx->outputs[0] == NULL` — in C this path is a latent crash,
    /// prevented by the custom activate. The port mirrors that
    /// unreachable-ness with a loud error instead of dropping the frame: a
    /// future engine change that starts draining sinks must fail visibly.
    fn filter_frame(
        &mut self,
        _g: &mut FilterGraph,
        _node: NodeId,
        _pad: usize,
        _frame: Frame,
    ) -> Result<()> {
        Err(Error::Unsupported(
            "buffersink has no filter_frame (C: the default pad callback would \
             dereference outputs[0] == NULL; unreachable because activate never \
             drains — buffersink.c:210-211)"
                .into(),
        ))
    }

    /// Endpoint downcast hook — C's typed `ctx->priv` access from the public
    /// API functions (buffersink.c:237 asserts the filter, then reads
    /// `ctx->priv`). The buffersink runtime functions must reach
    /// [`BufferSinkContext::peeked_frame`] (buffersink.c:71, 104) outside
    /// activation; generic impls return `None`.
    fn as_any(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Option parsing (opt.c:805-895, 613-670, 426-518, 275-286)
// ---------------------------------------------------------------------------

/// `opt_set_array`'s splitter (opt.c:824-846): split on ',' with backslash
/// escaping — a `'\\'` before any character is consumed and the next char
/// taken literally (a trailing backslash is literal). C emits one token per
/// `while (*val)` iteration, so an EMPTY value yields ZERO elements
/// (opt.c:818-825 — `if (val && *val)`) while a TRAILING separator emits no
/// final empty token (`"a,"` → `["a"]`); interior empty tokens do occur
/// (`"a,,b"` → `["a", "", "b"]`).
fn split_array_value(value: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    if value.is_empty() {
        return tokens; // opt.c:821 — zero elements = unconstrained
    }
    let chars: Vec<char> = value.chars().collect();
    let mut i = 0usize;
    // while (*val) { for (; *val; val++, p++) { … } emit token }
    while i < chars.len() {
        let mut cur = String::new();
        while i < chars.len() {
            let c = chars[i];
            if c == '\\' && i + 1 < chars.len() {
                i += 1; // consume the backslash, take the next char literally
                cur.push(chars[i]);
            } else if c == ',' {
                i += 1; // consume the separator
                break;
            } else {
                cur.push(c);
            }
            i += 1;
        }
        tokens.push(cur);
    }
    tokens
}

/// Parse a `pixel_formats` value (C: `opt_set_array` opt.c:805-887 →
/// `opt_set_elem` → `set_string_pixel_fmt` opt.c:665-670 → `set_string_fmt`
/// opt.c:613-658). Names only — [`PixelFormat::from_name`] (aliases included,
/// like `av_get_pix_fmt`); C's numeric-strtol fallback and `"none"`
/// (opt.c:620-626) are NOT ported (module divergence D3). Unknown name:
/// the verbatim C error text (opt.c:628-629, desc = "pixel format") as
/// `Err(Error::InvalidArgument)`.
fn parse_pixel_formats(value: &str) -> Result<Vec<PixelFormat>> {
    let mut out = Vec::new();
    for tok in split_array_value(value) {
        match PixelFormat::from_name(&tok) {
            Some(fmt) => out.push(fmt), // duplicates KEPT, order preserved
            None => {
                return Err(Error::InvalidArgument(format!(
                    "Unable to parse \"pixel_formats\" option value \"{tok}\" as pixel format"
                )));
            }
        }
    }
    Ok(out)
}

/// Parse a `colorspaces`/`colorranges` value — the plain-int subset of C's
/// `set_string_number` (opt.c:426-518) + `write_number` range gate
/// (opt.c:280-286; the option defs carry min=0, max=INT_MAX,
/// buffersink.c:357-360). Each element: decimal i64, optional sign; then
/// mapped to the enum by C discriminant. Non-numeric → the "Unable to parse"
/// text (opt.c:500-501, val is the failing TOKEN); an int with no Rust
/// variant (or outside [0, INT_MAX]) → write_number's out-of-range text
/// (opt.c:283-284; C's `%g` exponent form of INT_MAX is printed plainly).
fn parse_enum_array<T>(
    value: &str,
    opt_name: &str,
    from_i64: fn(i64) -> Option<T>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for tok in split_array_value(value) {
        let n: i64 = match tok.parse() {
            Ok(n) => n,
            Err(_) => {
                return Err(Error::InvalidArgument(format!(
                    "Unable to parse \"{opt_name}\" option value \"{tok}\""
                )));
            }
        };
        match from_i64(n) {
            Some(v) => out.push(v), // duplicates KEPT, order preserved
            None => {
                return Err(Error::InvalidArgument(format!(
                    "Value {:.6} for parameter '{opt_name}' out of range [0 - 2147483647]",
                    n as f64
                )));
            }
        }
    }
    Ok(out)
}

/// C discriminant mapping for `colorspaces` — `enum AVColorSpace`
/// (pixfmt.h); the Rust [`ColorSpace`] (color.rs) matches. 8
/// (`AVCOL_SPC_RESERVED` in modern headers) has no variant here — see
/// divergence D3.
fn colorspace_from_i64(v: i64) -> Option<ColorSpace> {
    Some(match v {
        0 => ColorSpace::Rgb,
        1 => ColorSpace::Bt709,
        2 => ColorSpace::Unspecified,
        3 => ColorSpace::Reserved,
        4 => ColorSpace::Fcc,
        5 => ColorSpace::Bt470bg,
        6 => ColorSpace::Smpte170m,
        7 => ColorSpace::Smpte240m,
        9 => ColorSpace::Bt2020Ncl,
        _ => return None,
    })
}

/// C discriminant mapping for `colorranges` — `enum AVColorRange`.
fn colorrange_from_i64(v: i64) -> Option<ColorRange> {
    Some(match v {
        0 => ColorRange::Unspecified,
        1 => ColorRange::Mpeg,
        2 => ColorRange::Jpeg,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Runtime API (buffersink.c:74-145, 235-259)
// ---------------------------------------------------------------------------

/// `av_buffersink_get_frame` (buffersink.c:74-77): flags 0 — consume and
/// move the frame out. `Err(Again)` = starved (feed the graph more);
/// `Err(Eof)` = drained (and on every later call); any other `Err` is a real
/// failure. Return-code contract: buffersink.h:140-146. This is the function
/// `FilterGraph::get_frame` delegates to (graph.rs:453-458).
pub fn buffersink_get_frame(g: &mut FilterGraph, sink: NodeId) -> Result<Frame> {
    buffersink_get_frame_flags(g, sink, BuffersinkFlags(0))
}

/// `AV_BUFFERSINK_FLAG_PEEK` get (buffersink.h:85 via
/// `av_buffersink_get_frame_flags`, buffersink.c:81-83): return a clone
/// (`av_frame_ref` semantics — shared planes) of the head frame WITHOUT
/// removing it from the sink's surface: the frame moves from the link fifo
/// into `peeked_frame` and the clone is returned. The next
/// `buffersink_peek` returns the SAME frame again (buffersink.c:103-104);
/// the next [`buffersink_get_frame`] returns it and clears the peek state.
/// On a starved sink returns `Err(Again)` — PEEK does NOT imply NO_REQUEST
/// (the ffmpeg CLI composes `PEEK|NO_REQUEST` itself, fftools/ffmpeg_filter.c:2896,
/// because it drives run_once on its own).
pub fn buffersink_peek(g: &mut FilterGraph, sink: NodeId) -> Result<Frame> {
    buffersink_get_frame_flags(g, sink, BuffersinkFlags::PEEK)
}

/// `return_or_keep_frame` (buffersink.c:79-91): hand one frame to the caller
/// under the flags. PEEK: park `frame` in the peek slot and return it
/// (C parks `in` and `av_frame_ref`s a NEW reference out; storing the CLONE
/// and returning the original is the same aliasing — planes shared either
/// way). Non-PEEK: clear the slot and return `frame` (C's
/// `av_frame_move_ref` + `av_frame_free` — the move-out is zero-copy).
///
/// Both call shapes: (a) re-serving an already-peeked frame — PEEK re-returns
/// it and KEEPS it peeked, non-PEEK takes it and clears the slot; (b) a
/// just-consumed frame — PEEK parks it and returns it, non-PEEK passes it
/// through (the slot was already None).
fn return_or_keep_frame(peeked_slot: &mut Option<Frame>, frame: Frame, peek: bool) -> Frame {
    if peek {
        *peeked_slot = Some(frame.clone());
        frame
    } else {
        *peeked_slot = None;
        frame
    }
}

/// `get_frame_internal` (buffersink.c:93-133) via
/// `av_buffersink_get_frame_flags` (135-139 — the `min_samples` argument is
/// video-dead: 0 for video links, so the samples branch of buffersink.c:107
/// collapses to `ff_inlink_consume_frame`).
///
/// STEP ORDER IS LOAD-BEARING:
/// 1. peeked short-circuit (103-104) — BEFORE any consume/acknowledge: the
///    peeked frame shadows everything queued since;
/// 2. consume (107-113) — frames drain ahead of status;
/// 3. acknowledge status (114-115) — the one-shot `status_in`→`status_out`
///    transition, returned as `Err(status)`;
/// 4. status_out re-check (port-only, divergence D2): an already-acked link
///    keeps returning `Err(status)` like C's truthy negative
///    (avfilter.c:1472-1474) — makes post-EOF gets idempotent;
/// 5. NO_REQUEST (116-117) — `Err(Again)` without requesting anything;
/// 6. frame_wanted_out (118-131) — drive `run_once`; on its `Err(Again)` arm
///    the starved latch (divergence D1: C arms on
///    FFERROR_BUFFERSRC_EMPTY, which wave-1's run_once cannot produce — the
///    re-request arms it instead, costing at most one extra graph round);
/// 7. not wanted (129-130) — `inlink_request_frame` (activate-style; do NOT
///    mix with the legacy `ff_request_frame` pull, filter.rs:536-542).
///
/// Loop termination is trusted to the engine reaching quiescence, as in C
/// (no timeout there either).
pub fn buffersink_get_frame_flags(
    g: &mut FilterGraph,
    sink: NodeId,
    flags: BuffersinkFlags,
) -> Result<Frame> {
    check_sink(g, sink)?;
    let peek = flags.contains(BuffersinkFlags::PEEK);
    let inlink = g.inlink(sink, 0);

    // 1. PEEKED SHORT-CIRCUIT (C:103-104). Scoped take/restore of the impl —
    //    NEVER held across run_once (run_once takes the activated node's imp
    //    and may pick THIS sink; the debug_assert at filter.rs:706-714).
    let early = with_sink_ctx(g, sink, |ctx| match ctx.peeked_frame.take() {
        Some(frame) => Some(return_or_keep_frame(&mut ctx.peeked_frame, frame, peek)),
        None => None,
    });
    if let Some(frame) = early {
        return Ok(frame);
    }

    // C:101 — `buffersrc_empty` latch (armed on the re-request path here,
    // divergence D1).
    let mut starved = false;
    loop {
        // 2. CONSUME (C:107-113). The `ret < 0` branch (109-110) is
        //    unreachable: consume never fails, it returns Option.
        if let Some(frame) = filter::inlink_consume_frame(g, inlink) {
            return Ok(with_sink_ctx(g, sink, |ctx| {
                return_or_keep_frame(&mut ctx.peeked_frame, frame, peek)
            }));
        }
        // 3. ACKNOWLEDGE (C:114-115): fires only when the fifo is empty AND
        //    status_in set AND status_out unset — the ack side effects
        //    (status_out = status_in, current_pts = status_in_pts) already
        //    happened inside.
        if let Some((status, _pts)) = filter::inlink_acknowledge_status(g, inlink) {
            return Err(status);
        }
        // 4. STATUS_OUT RE-CHECK (port-only, divergence D2): the consume at
        //    (2) just returned None, so the fifo is provably empty — C's
        //    fifo-first ordering preserved. Standing in for avfilter.c:
        //    1472-1474's truthy negative.
        if let Some(so) = g.links[inlink.0].status_out.as_ref() {
            return Err(clone_status(so));
        }
        // 5. NO_REQUEST (C:116-117): before the wanted check, requesting
        //    nothing — frame_wanted_out untouched, nobody readied.
        if flags.contains(BuffersinkFlags::NO_REQUEST) {
            return Err(Error::Again);
        }
        // 6./7. WANTED / NOT WANTED (C:118-131).
        if g.links[inlink.0].frame_wanted_out {
            match g.run_once() {
                Err(Error::Again) => {
                    if starved {
                        return Err(Error::Again); // C:123-124
                    }
                    starved = true; // divergence D1's latch arm
                    filter::inlink_request_frame(g, inlink); // C:125
                }
                Err(e) => return Err(e), // C:126-127
                Ok(()) => {}             // C:128 — loop continues
            }
        } else {
            filter::inlink_request_frame(g, inlink); // C:129-130
        }
    }
}

// ---------------------------------------------------------------------------
// Link accessors (buffersink.c:235-259)
// ---------------------------------------------------------------------------

/// The is-a-buffersink guard shared by every runtime entry point — the port
/// of the accessor macro's `av_assert0(fffilter(ctx->filter)->activate ==
/// activate)` (buffersink.c:237). A foreign filter gets
/// `Err(InvalidArgument)` where C would abort.
fn check_sink(g: &FilterGraph, sink: NodeId) -> Result<()> {
    if g.nodes[sink.0].def.name != "buffersink" {
        return Err(Error::InvalidArgument(format!(
            "filter '{}' is not a buffersink",
            g.nodes[sink.0].name
        )));
    }
    debug_assert!(
        std::ptr::eq(g.nodes[sink.0].def, &BUFFERSINK_DEF),
        "a def named buffersink that is not BUFFERSINK_DEF"
    );
    Ok(())
}

/// Scoped `ctx->priv` access (C reads the typed private context from its
/// public API functions after asserting the filter, buffersink.c:237): take
/// the node's impl, downcast, run `f`, restore. The take/restore MUST stay
/// scoped — never held across `g.run_once()`, which takes the activated
/// node's impl and may pick this very sink (its `ready` may still be 300 from
/// the push that queued the current frame; the debug_assert at
/// filter.rs:706-714 would fire).
fn with_sink_ctx<R>(
    g: &mut FilterGraph,
    sink: NodeId,
    f: impl FnOnce(&mut BufferSinkContext) -> R,
) -> R {
    let mut imp = g.nodes[sink.0]
        .imp
        .take()
        .expect("buffersink: node imp already taken");
    let ctx = imp
        .as_any()
        .and_then(|a| a.downcast_mut::<BufferSinkContext>())
        .expect("buffersink: node impl is not a BufferSinkContext");
    let r = f(ctx);
    g.nodes[sink.0].imp = Some(imp);
    r
}

/// `av_buffersink_get_time_base` (buffersink.c:242). Meaningful only after
/// graph config (wave 2); the zero rational before.
pub fn buffersink_get_time_base(g: &FilterGraph, sink: NodeId) -> Result<Rational> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].time_base)
}

/// `av_buffersink_get_format` (buffersink.c:243). `None` IS C's
/// `AV_PIX_FMT_NONE` sentinel encoding (link.rs:95-97); `Some` only after
/// the negotiation picks (wave 2).
pub fn buffersink_get_format(g: &FilterGraph, sink: NodeId) -> Result<Option<PixelFormat>> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].format)
}

/// `av_buffersink_get_w` (buffersink.c:245).
pub fn buffersink_get_w(g: &FilterGraph, sink: NodeId) -> Result<u32> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].w)
}

/// `av_buffersink_get_h` (buffersink.c:246).
pub fn buffersink_get_h(g: &FilterGraph, sink: NodeId) -> Result<u32> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].h)
}

/// `av_buffersink_get_sample_aspect_ratio` (buffersink.c:247).
pub fn buffersink_get_sample_aspect_ratio(g: &FilterGraph, sink: NodeId) -> Result<Rational> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].sample_aspect_ratio)
}

/// `av_buffersink_get_colorspace` (buffersink.c:248) — set by `pick_format`;
/// `Unspecified` pre-negotiation.
pub fn buffersink_get_colorspace(g: &FilterGraph, sink: NodeId) -> Result<ColorSpace> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].colorspace)
}

/// `av_buffersink_get_color_range` (buffersink.c:249).
pub fn buffersink_get_color_range(g: &FilterGraph, sink: NodeId) -> Result<ColorRange> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].color_range)
}

/// `av_buffersink_get_frame_rate` (buffersink.c:254-259) — note this one
/// reads the `FilterLink`, not the public `AVFilterLink`, in C; the Rust
/// [`Link`] is flat, so it is the same expression.
pub fn buffersink_get_frame_rate(g: &FilterGraph, sink: NodeId) -> Result<Rational> {
    check_sink(g, sink)?;
    Ok(g.links[g.inlink(sink, 0).0].frame_rate)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Options;
    use crate::filter::filter::FilterNode;
    use crate::filter::link::LinkId;
    use std::cell::Cell;
    use std::sync::Arc;

    // -- helpers ------------------------------------------------------------

    /// TestSrc → buffersink, linked (the engine-test helper source produces
    /// nothing itself; tests push frames into its output link directly, the
    /// way buffersrc's add_frame will).
    fn sink_graph(opts: &str) -> (FilterGraph, NodeId, NodeId, LinkId) {
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let sink = g.create_filter("buffersink", opts).unwrap();
        let l = g.link(src, 0, sink, 0).unwrap();
        (g, src, sink, l)
    }

    /// Link geometry as a config pass would derive it from the frames.
    fn configure_video(g: &mut FilterGraph, l: LinkId, fmt: PixelFormat, w: u32, h: u32) {
        g.links[l.0].format = Some(fmt);
        g.links[l.0].w = w;
        g.links[l.0].h = h;
    }

    fn gray_frame(w: u32, h: u32, pts: i64) -> Frame {
        let mut f = Frame::alloc(PixelFormat::Gray8, w, h).unwrap();
        f.pts = pts;
        f
    }

    /// A source that counts its activations (wave-2 buffersrc's
    /// emptiness-as-Ok activate shape), observable from outside via the
    /// shared counter.
    struct CountingSrc {
        activations: Arc<Cell<u32>>,
    }
    impl FilterImpl for CountingSrc {
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
            self.activations.set(self.activations.get() + 1);
            Ok(()) // nothing queued upstream: would-be BUFFERSRC_EMPTY
        }
    }

    static COUNTING_SRC_PAD: PadDef = PadDef {
        name: "default",
        needs_writable: false,
    };
    static COUNTING_SRC_DEF: FilterDef = FilterDef {
        name: "countingsrc",
        inputs: &[],
        outputs: &[COUNTING_SRC_PAD],
        flags: FilterFlags(0),
        shorthand: &[],
        make: || {
            Box::new(CountingSrc {
                activations: Arc::new(Cell::new(0)),
            })
        },
    };

    fn push_counting_src(g: &mut FilterGraph) -> (NodeId, Arc<Cell<u32>>) {
        let counter = Arc::new(Cell::new(0u32));
        g.nodes.push(FilterNode {
            def: &COUNTING_SRC_DEF,
            name: COUNTING_SRC_DEF.name.to_string(),
            inputs: vec![],
            outputs: vec![None],
            imp: Some(Box::new(CountingSrc {
                activations: counter.clone(),
            })),
            ready: 0,
            initialized: true,
            opts: Options::default(),
        });
        (NodeId(g.nodes.len() - 1), counter)
    }

    // -- def + registry ------------------------------------------------------

    #[test]
    fn def_shape_and_registry() {
        // mod.rs wiring: the registry must resolve the name to OUR def
        // (avfilter_get_by_name, allfilters.c:658-671).
        let def = crate::filter::filter_def("buffersink").expect("buffersink registered");
        assert!(std::ptr::eq(def, &BUFFERSINK_DEF));
        // ff_video_default_filterpad (video.c:37-41): one input "default".
        assert_eq!(BUFFERSINK_DEF.name, "buffersink");
        assert_eq!(BUFFERSINK_DEF.inputs.len(), 1);
        assert_eq!(BUFFERSINK_DEF.inputs[0].name, "default");
        assert!(!BUFFERSINK_DEF.inputs[0].needs_writable);
        // .p.outputs = NULL (buffersink.c:386) — the graph endpoint.
        assert!(BUFFERSINK_DEF.outputs.is_empty());
        // ff_filter_frame's skip-validation name list (avfilter.c:1076-1082).
        assert!(
            BUFFERSINK_DEF
                .flags
                .contains(FilterFlags::ALLOWS_RECONFIGURE)
        );
        // ff_filter_opt_parse walks the priv_class options in declaration
        // order (avfilter.c:863-866); alphamodes dropped.
        assert_eq!(
            BUFFERSINK_DEF.shorthand,
            ["pixel_formats", "colorspaces", "colorranges"]
        );
    }

    // -- init / option parsing ----------------------------------------------

    #[test]
    fn init_parses_named_and_positional_options() {
        let mut g = FilterGraph::new();
        // Named, comma list: order preserved (opt.c:805-895 split).
        let sink = g
            .create_filter("buffersink", "pixel_formats=yuv420p,rgb24")
            .unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()),
            vec![PixelFormat::Yuv420p, PixelFormat::Rgb24]
        );
        assert_eq!(with_sink_ctx(&mut g, sink, |c| c.warning_limit), 100);

        // Positional → pixel_formats via shorthand slot 0 (avfilter.c:863-866).
        let sink = g.create_filter("buffersink", "yuv420p").unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()),
            vec![PixelFormat::Yuv420p]
        );

        // Aliases resolve (av_get_pix_fmt).
        let sink = g
            .create_filter("buffersink", "pixel_formats=gray8")
            .unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()),
            vec![PixelFormat::Gray8]
        );

        // EMPTY value → ZERO elements = unconstrained (opt.c:821).
        let sink = g.create_filter("buffersink", "pixel_formats=").unwrap();
        assert!(with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()).is_empty());

        // Duplicate KEY replaces the vec (opt.c:860-878 clear-then-fill), not
        // appends.
        let sink = g
            .create_filter("buffersink", "pixel_formats=yuv420p:pixel_formats=rgb24")
            .unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()),
            vec![PixelFormat::Rgb24]
        );

        // Duplicates WITHIN one value are KEPT — ff_add_format appends
        // unconditionally (formats.c:572-577).
        let sink = g
            .create_filter("buffersink", "pixel_formats=gray,gray")
            .unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()),
            vec![PixelFormat::Gray8, PixelFormat::Gray8]
        );

        // colorspaces by C discriminant (AVCOL_SPC_BT709=1, BT470BG=5).
        let sink = g.create_filter("buffersink", "colorspaces=1,5").unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.colorspaces.clone()),
            vec![ColorSpace::Bt709, ColorSpace::Bt470bg]
        );
        // colorranges (AVCOL_RANGE_MPEG=1, JPEG=2).
        let sink = g.create_filter("buffersink", "colorranges=1,2").unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.colorranges.clone()),
            vec![ColorRange::Mpeg, ColorRange::Jpeg]
        );
    }

    #[test]
    fn split_array_value_c_semantics() {
        // opt.c:824-846: one token per while-iteration.
        assert!(split_array_value("").is_empty()); // opt.c:821
        assert_eq!(split_array_value("a,b"), ["a", "b"]);
        assert_eq!(split_array_value("a,"), ["a"]); // trailing sep: no final token
        assert_eq!(split_array_value(",a"), ["", "a"]);
        assert_eq!(split_array_value("a,,b"), ["a", "", "b"]); // interior kept
        assert_eq!(split_array_value("a\\,b"), ["a,b"]); // escaped sep
        assert_eq!(split_array_value("a\\\\b"), ["a\\b"]); // escaped backslash
        assert_eq!(split_array_value("a\\"), ["a\\"]); // trailing backslash literal
    }

    #[test]
    fn option_error_texts() {
        let mut g = FilterGraph::new();
        // Unknown pixel format name — set_string_fmt's text (opt.c:628-629,
        // desc "pixel format"), val is the failing TOKEN.
        let err = g
            .create_filter("buffersink", "pixel_formats=bogus")
            .unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Unable to parse \"pixel_formats\" option value \"bogus\" as pixel format"
            ),
            other => panic!("expected InvalidArgument, got {other}"),
        }
        // The failing token, not the whole value.
        let err = g
            .create_filter("buffersink", "pixel_formats=yuv420p,bogus")
            .unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => assert!(msg.contains("\"bogus\" as pixel format")),
            other => panic!("expected InvalidArgument, got {other}"),
        }

        // Non-numeric int element — set_string_number's text (opt.c:500-501).
        let err = g
            .create_filter("buffersink", "colorspaces=xyz")
            .unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Unable to parse \"colorspaces\" option value \"xyz\"")
            }
            other => panic!("expected InvalidArgument, got {other}"),
        }
        let err = g
            .create_filter("buffersink", "colorranges=xyz")
            .unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Unable to parse \"colorranges\" option value \"xyz\"")
            }
            other => panic!("expected InvalidArgument, got {other}"),
        }

        // Negative int — write_number's range text (opt.c:283-284; option
        // defs carry min=0 max=INT_MAX, buffersink.c:357-360); C's %f prints
        // 6 decimals, %g bounds printed plainly (documented divergence).
        let err = g.create_filter("buffersink", "colorspaces=-1").unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Value -1.000000 for parameter 'colorspaces' out of range [0 - 2147483647]"
            ),
            other => panic!("expected InvalidArgument, got {other}"),
        }
        // In-range int with no Rust variant (8 = AVCOL_SPC_RESERVED in C)
        // errors the same way — documented degradation D3 (C would carry the
        // raw discriminant into negotiation).
        let err = g.create_filter("buffersink", "colorspaces=8").unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Value 8.000000 for parameter 'colorspaces' out of range [0 - 2147483647]"
            ),
            other => panic!("expected InvalidArgument, got {other}"),
        }
        let err = g.create_filter("buffersink", "colorranges=-1").unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Value -1.000000 for parameter 'colorranges' out of range [0 - 2147483647]"
            ),
            other => panic!("expected InvalidArgument, got {other}"),
        }

        // Unknown key → the pre-existing leftover error (graph.rs:168-170).
        let err = g.create_filter("buffersink", "nosuch=1").unwrap_err();
        match &err {
            Error::NotFound(msg) => assert_eq!(msg, "No such option: nosuch"),
            other => panic!("expected NotFound, got {other}"),
        }
    }

    #[test]
    fn init_accepts_all_three_axes_and_positional_chain() {
        // Positional slots 2 and 3 (shorthand chain, avfilter.c:863-866).
        let mut g = FilterGraph::new();
        let sink = g.create_filter("buffersink", "yuv420p:1:2").unwrap();
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.pixel_formats.clone()),
            vec![PixelFormat::Yuv420p]
        );
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.colorspaces.clone()),
            vec![ColorSpace::Bt709]
        );
        assert_eq!(
            with_sink_ctx(&mut g, sink, |c| c.colorranges.clone()),
            vec![ColorRange::Jpeg]
        );
    }

    // -- query_formats --------------------------------------------------------

    #[test]
    fn query_formats_sets_own_halves_only() {
        let (mut g, _src, sink, l) = sink_graph("pixel_formats=yuv420p");
        // Run the sink's query as the engine would (avfiltergraph.c:407-410:
        // impl query, then the default query).
        {
            let mut imp = g.nodes[sink.0].imp.take().unwrap();
            imp.query_formats(&mut g, sink).unwrap();
            g.nodes[sink.0].imp = Some(imp);
        }
        // vsink_query_formats → ff_set_pixel_formats_from_list2 (formats.c:
        // 1160-1168): the option values, on the input link's outcfg half
        // (the dst-accepts declarations — the sink has no other pad).
        let list = g.links[l.0].outcfg.formats.expect("sink declared formats");
        assert_eq!(g.fmt_lists[list as usize], vec![PixelFormat::Yuv420p]);
        // The untouched axes stay unset (C's nb_==0 skip)...
        assert!(g.links[l.0].outcfg.color_spaces.is_none());
        assert!(g.links[l.0].outcfg.color_ranges.is_none());
        // ...until the ENGINE's default query fills them with the all-lists
        // (avfiltergraph.c:410 — the trait contract; the impl never calls it
        // itself). Mirrors graph.rs:602-620's null test.
        g.default_query_formats(sink).unwrap();
        let csp = g.links[l.0]
            .outcfg
            .color_spaces
            .expect("csp filled by default");
        assert_eq!(
            g.csp_lists[csp as usize],
            crate::filter::formats::all_color_spaces()
        );
        assert!(g.links[l.0].outcfg.color_ranges.is_some());
        // fill-if-unset: the explicitly declared format list survives.
        assert_eq!(g.links[l.0].outcfg.formats, Some(list));
    }

    #[test]
    fn query_formats_declares_all_three_axes_in_order() {
        let (mut g, _src, sink, l) = sink_graph("colorspaces=1,5:colorranges=2");
        {
            let mut imp = g.nodes[sink.0].imp.take().unwrap();
            imp.query_formats(&mut g, sink).unwrap();
            g.nodes[sink.0].imp = Some(imp);
        }
        let csp = g.links[l.0].outcfg.color_spaces.expect("declared");
        assert_eq!(
            g.csp_lists[csp as usize],
            vec![ColorSpace::Bt709, ColorSpace::Bt470bg]
        );
        let rng = g.links[l.0].outcfg.color_ranges.expect("declared");
        assert_eq!(g.rng_lists[rng as usize], vec![ColorRange::Jpeg]);
        // No pixel_formats option → axis untouched.
        assert!(g.links[l.0].outcfg.formats.is_none());
    }

    // -- activate ---------------------------------------------------------------

    #[test]
    fn activate_does_not_drain_and_warns() {
        let (mut g, _src, sink, l) = sink_graph("");
        configure_video(&mut g, l, PixelFormat::Gray8, 1, 1);
        for pts in 0..100i64 {
            filter::filter_frame(&mut g, l, gray_frame(1, 1, pts)).unwrap();
        }
        assert_eq!(g.links[l.0].fifo.len(), 100);
        assert_eq!(g.nodes[sink.0].ready, 300, "pushes wake the sink");
        // Activate: warns (threshold 100) and escalates ×10...
        g.run_once().unwrap();
        // ...but NEVER drains — "the rest is up to get_frame_internal"
        // (buffersink.c:210).
        assert_eq!(g.links[l.0].fifo.len(), 100, "activate must not drain");
        assert_eq!(with_sink_ctx(&mut g, sink, |c| c.warning_limit), 1000);
        assert_eq!(g.links[l.0].frame_count_out, 0);
        // Below the new threshold: no second escalation.
        filter::filter_frame(&mut g, l, gray_frame(1, 1, 100)).unwrap();
        g.run_once().unwrap();
        assert_eq!(g.links[l.0].fifo.len(), 101);
        assert_eq!(with_sink_ctx(&mut g, sink, |c| c.warning_limit), 1000);
    }

    // -- get / peek ---------------------------------------------------------------

    #[test]
    fn get_frame_returns_pushed_frame() {
        let (mut g, _src, sink, l) = sink_graph("");
        configure_video(&mut g, l, PixelFormat::Gray8, 8, 8);
        filter::filter_frame(&mut g, l, gray_frame(8, 8, 7)).unwrap();
        let out = buffersink_get_frame(&mut g, sink).unwrap();
        assert_eq!(out.pts, 7);
        assert_eq!(out.format, PixelFormat::Gray8);
        assert_eq!((out.width, out.height), (8, 8));
        // inlink_consume_frame side effects (filter.rs:308-313).
        assert!(g.links[l.0].fifo.is_empty());
        assert_eq!(g.links[l.0].frame_count_out, 1);
        assert_eq!(g.links[l.0].current_pts, 7);
        // Starved sink: drives the graph, then Err(Again) (C:101, 120-125).
        assert!(matches!(
            buffersink_get_frame(&mut g, sink),
            Err(Error::Again)
        ));
    }

    #[test]
    fn peek_protocol() {
        let (mut g, _src, sink, l) = sink_graph("");
        configure_video(&mut g, l, PixelFormat::Gray8, 4, 4);
        filter::filter_frame(&mut g, l, gray_frame(4, 4, 9)).unwrap();

        // PEEK returns a clone AND advances the fifo: the frame moved into
        // peeked_frame (C:81-83 + the consume at 107-108).
        let p1 = buffersink_peek(&mut g, sink).unwrap();
        assert_eq!(p1.pts, 9);
        assert!(g.links[l.0].fifo.is_empty(), "PEEK advanced the fifo");
        assert_eq!(g.links[l.0].frame_count_out, 1);
        // av_frame_ref semantics: the returned clone shares planes with the
        // stored frame.
        assert!(
            Arc::strong_count(&p1.planes[0].buf) >= 2,
            "peeked clone shares the plane buffers"
        );

        // A second peek re-serves the SAME frame (early return, C:103-104),
        // even after newer frames queue behind it.
        filter::filter_frame(&mut g, l, gray_frame(4, 4, 10)).unwrap();
        let p2 = buffersink_peek(&mut g, sink).unwrap();
        assert_eq!(p2.pts, 9, "peeked frame shadows the newer queue entry");
        assert_eq!(g.links[l.0].fifo.len(), 1);
        assert_eq!(g.links[l.0].frame_count_out, 1, "no second consume");

        // The peeked frame also shadows for the next plain get; taking it
        // clears the peek state.
        let got = buffersink_get_frame(&mut g, sink).unwrap();
        assert_eq!(got.pts, 9);
        assert!(with_sink_ctx(&mut g, sink, |c| c.peeked_frame.is_none()));
        let got2 = buffersink_get_frame(&mut g, sink).unwrap();
        assert_eq!(got2.pts, 10);
        assert!(matches!(
            buffersink_get_frame(&mut g, sink),
            Err(Error::Again)
        ));
    }

    #[test]
    fn frames_drain_before_eof_and_eof_is_idempotent() {
        let (mut g, _src, sink, l) = sink_graph("");
        configure_video(&mut g, l, PixelFormat::Gray8, 4, 4);
        filter::filter_frame(&mut g, l, gray_frame(4, 4, 3)).unwrap();
        // EOF announced while a frame is still queued (what buffersrc_close
        // does); the status is HELD behind the frame.
        filter::set_in_status(&mut g, l, Error::Eof, 100);
        // Consume precedes acknowledge (buffersink.c:106-114).
        let got = buffersink_get_frame(&mut g, sink).unwrap();
        assert_eq!(got.pts, 3);
        // Drained: the acknowledge transition fires → Err(Eof), status_out
        // now set (avfilter.c:1477-1479; the ack's pts is 100, not 3 —
        // current_pts is set from status_in_pts inside the ack).
        assert!(matches!(
            buffersink_get_frame(&mut g, sink),
            Err(Error::Eof)
        ));
        assert!(matches!(g.links[l.0].status_out, Some(Error::Eof)));
        assert_eq!(g.links[l.0].current_pts, 100);
        // Idempotent: the status_out re-check stands in for C's truthy
        // negative (avfilter.c:1472-1474) — and must not touch the request
        // path (frame_wanted_out stays false).
        assert!(matches!(
            buffersink_get_frame(&mut g, sink),
            Err(Error::Eof)
        ));
        assert!(!g.links[l.0].frame_wanted_out);
    }

    #[test]
    fn starved_returns_again_after_driving_the_graph() {
        let mut g = FilterGraph::new();
        let (src, counter) = push_counting_src(&mut g);
        let sink = g.create_filter("buffersink", "").unwrap();
        let l = g.link(src, 0, sink, 0).unwrap();
        // Nothing queued, no status: the loop must TERMINATE (latch) and
        // surface EAGAIN like C (C:101, 120-125; divergence D1 arms the latch
        // on the re-request instead of FFERROR_BUFFERSRC_EMPTY).
        assert!(matches!(
            buffersink_get_frame(&mut g, sink),
            Err(Error::Again)
        ));
        // The request was registered and the source was driven: run_once
        // activated it at least once at priority 100 (inlink_request_frame,
        // C:125/130).
        assert!(g.links[l.0].frame_wanted_out);
        assert!(
            counter.get() >= 1,
            "source activated {} times",
            counter.get()
        );
    }

    #[test]
    fn no_request_flag_never_requests() {
        let (mut g, src, sink, l) = sink_graph("");
        // C:116-117 short-circuits before any request.
        assert!(matches!(
            buffersink_get_frame_flags(&mut g, sink, BuffersinkFlags::NO_REQUEST),
            Err(Error::Again)
        ));
        assert!(!g.links[l.0].frame_wanted_out, "no request registered");
        assert_eq!(g.nodes[src.0].ready, 0);
        assert_eq!(g.nodes[sink.0].ready, 0);
        // Combined PEEK|NO_REQUEST on a QUEUED frame still returns it — the
        // fftools shape (fftools/ffmpeg_filter.c:2896): consume runs before
        // the NO_REQUEST check.
        filter::filter_frame(&mut g, l, gray_frame(2, 2, 5)).unwrap();
        let f = buffersink_get_frame_flags(
            &mut g,
            sink,
            BuffersinkFlags::PEEK | BuffersinkFlags::NO_REQUEST,
        )
        .unwrap();
        assert_eq!(f.pts, 5);
        assert!(with_sink_ctx(&mut g, sink, |c| c.peeked_frame.is_some()));
    }

    #[test]
    fn not_a_buffersink_guard() {
        let mut g = FilterGraph::new();
        let null_node = g.create_filter("null", "").unwrap();
        let err = buffersink_get_frame(&mut g, null_node).unwrap_err();
        match &err {
            Error::InvalidArgument(msg) => {
                assert!(msg.contains("not a buffersink"), "{msg}");
            }
            other => panic!("expected InvalidArgument, got {other}"),
        }
        // The accessors share the guard (C: av_assert0, buffersink.c:237).
        assert!(matches!(
            buffersink_get_time_base(&g, null_node),
            Err(Error::InvalidArgument(_))
        ));
    }

    // -- accessors ---------------------------------------------------------------

    #[test]
    fn accessors_read_the_input_link() {
        let (mut g, _src, sink, l) = sink_graph("");
        // Post-negotiation shape, set by hand (pick_format/config_props are
        // wave 2); pre-config these read the Link::default zero values.
        g.links[l.0].time_base = Rational::new(1, 25);
        g.links[l.0].format = Some(PixelFormat::Yuv420p);
        g.links[l.0].w = 64;
        g.links[l.0].h = 48;
        g.links[l.0].sample_aspect_ratio = Rational::new(16, 9);
        g.links[l.0].colorspace = ColorSpace::Bt709;
        g.links[l.0].color_range = ColorRange::Mpeg;
        g.links[l.0].frame_rate = Rational::new(25, 1);

        assert_eq!(
            buffersink_get_time_base(&g, sink).unwrap(),
            Rational::new(1, 25)
        );
        assert_eq!(
            buffersink_get_format(&g, sink).unwrap(),
            Some(PixelFormat::Yuv420p)
        );
        assert_eq!(buffersink_get_w(&g, sink).unwrap(), 64);
        assert_eq!(buffersink_get_h(&g, sink).unwrap(), 48);
        assert_eq!(
            buffersink_get_sample_aspect_ratio(&g, sink).unwrap(),
            Rational::new(16, 9)
        );
        assert_eq!(
            buffersink_get_colorspace(&g, sink).unwrap(),
            ColorSpace::Bt709
        );
        assert_eq!(
            buffersink_get_color_range(&g, sink).unwrap(),
            ColorRange::Mpeg
        );
        // frame_rate reads the FilterLink field (C:254-259) — same flat
        // expression in the port.
        assert_eq!(
            buffersink_get_frame_rate(&g, sink).unwrap(),
            Rational::new(25, 1)
        );

        // Pre-config zero values on a fresh link: the Option format is the
        // AV_PIX_FMT_NONE sentinel encoding.
        let (g2, _s, sink2, _l) = sink_graph("");
        assert_eq!(buffersink_get_format(&g2, sink2).unwrap(), None);
        assert_eq!(
            buffersink_get_colorspace(&g2, sink2).unwrap(),
            ColorSpace::Unspecified
        );
    }

    // -- filter_frame unreachable -------------------------------------------------

    #[test]
    fn filter_frame_is_a_loud_unreachable() {
        // C has no pad callback here; the framework default would deref
        // outputs[0] == NULL (avfilter.c:1007-1010). The port fails loudly
        // instead of dropping the frame — a future engine change that starts
        // draining sinks must surface, not swallow.
        let (mut g, _src, sink, _l) = sink_graph("");
        let mut imp = g.nodes[sink.0].imp.take().unwrap();
        let err = imp
            .filter_frame(&mut g, sink, 0, gray_frame(1, 1, 1))
            .unwrap_err();
        g.nodes[sink.0].imp = Some(imp);
        assert!(err.to_string().contains("buffersink has no filter_frame"));
    }
}
