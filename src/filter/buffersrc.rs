//! `buffer` — the memory buffer source, port of `libavfilter/buffersrc.c`
//! (`ff_vsrc_buffer`, buffersrc.c:619-630).
//!
//! C description: *"Buffer video frames, and make them accessible to the
//! filterchain."* (`FilterDef` has no description field, so it lives here.)
//!
//! ## Shape
//!
//! * [`BufferSource`] IS the filter state (C's `BufferSourceContext`,
//!   buffersrc.c:44-72, video-only subset) — per the crate convention that
//!   per-filter state is the `FilterImpl` (filter.rs). The free runtime
//!   functions ([`buffersrc_add_frame`], [`buffersrc_close`],
//!   [`buffersrc_get_nb_failed_requests`], [`buffersrc_get_status`]) reach it
//!   alongside `&mut FilterGraph` through the take/restore dance
//!   (precedents: `init_filter` graph.rs, `run_once` filter.rs) plus the
//!   `FilterImpl::as_any` downcast hook.
//! * [`BUFFER_SRC_DEF`] registers the filter; the graph's runtime stubs
//!   `add_frame`/`close_source` delegate to the free functions here.
//!
//! ## WAVE-2 CONTRACT NOTE for the buffersink implementer
//!
//! `activate`'s stage 3 returns `Ok(())` where C returns
//! `FFERROR_BUFFERSRC_EMPTY` (filters.h:35) — the mapping wave 1 pinned in
//! `FilterGraph::run_once`'s doc (C itself maps it to 0 at
//! avfiltergraph.c:1604, and `push_frame` treats it as a plain continue,
//! buffersrc.c:204). Consequence: a `run_once()` that returned `Ok(())` does
//! NOT prove the source made progress — it may mean "source starved". A
//! buffersink `get_frame` loop therefore MUST NOT blindly
//! `inlink_request_frame` on every `Err(Again)` round (request → activate
//! `Ok(())` → `Again` → request … is an infinite loop): poll
//! [`buffersrc_get_nb_failed_requests`] — the technique the ffmpeg CLI uses —
//! or bound consecutive request-Again rounds, the way C's internal latch does
//! (buffersink.c:117-124).
//!
//! ## Not ported (each skipped C site)
//!
//! * **audio** — `ff_asrc_abuffer` (buffersrc.c:632-651), `abuffer_options`
//!   (407-414), `init_audio` (418-466; error texts "Sample format was not
//!   set or was invalid", "Sample rate not set" are audio-only), the audio
//!   branches of `query_formats` (530-541) and `config_props` (566-572), and
//!   `CHECK_AUDIO_PARAM_CHANGE` (98-106, "Changing audio frame properties on
//!   the fly is not supported." + EINVAL) — this port is video-only.
//! * **`hw_frames_ctx`** — the HW-format branch of `init_video`
//!   (buffersrc.c:326-332; text: "Setting BufferSourceContext.pix_fmt to a
//!   HW format requires hw_frames_ctx to be non-NULL!"), the `sw_format`
//!   lookup in `query_formats` (492-493) and the hw ref in `config_props`
//!   (560-564). Unreachable: `PixelFormat::ALL` carries no HWACCEL formats
//!   (pixfmt.rs). Frame pools / `AVHWFramesContext` are wave-3+ concerns.
//! * **`side_data`** — the `config_props` clone loop (577-586) and the
//!   parameters-set copies (171-180): a source cannot inject link side-data;
//!   nothing in wave 2 consumes it.
//! * **`av_buffersrc_parameters_alloc`/`_set`** (buffersrc.c:108-183) — the
//!   options string is the only parameter path (a programmatic
//!   `set_parameters` can be added later). Consequence: the `prev_*` twins
//!   are NEVER synced at init time (the options path does not write them in
//!   C either — only `av_buffersrc_parameters_set` does, buffersrc.c:132-153),
//!   which drives `CHECK_VIDEO_PARAM_CHANGE`'s warning hysteresis.
//! * **`uninit`** (buffersrc.c:468-474) — it only frees `hw_frames_ctx`,
//!   `ch_layout` and `side_data`, all unported; no `Drop` is needed.
//! * **`alpha_mode` axis** (option 397-401, comparison in CHECK 74-96, link
//!   fill 265-266, query 517-527) — the port has no alpha negotiation axis
//!   anywhere (link.rs, formats.rs). `alpha_mode=…` stays an unrecognized
//!   leftover and init errors `No such option: alpha_mode` (documented
//!   degradation); the two log lines print the fixed literal "unspecified"
//!   (pixdesc.c's name for `AVALPHA_MODE_UNSPECIFIED`) in its slots.
//! * **flags** — `AV_BUFFERSRC_FLAG_NO_CHECK_FORMAT` (the checks always run;
//!   the video checks never fail, they only log) and `AV_BUFFERSRC_FLAG_PUSH`
//!   (no synchronous `push_frame` drain — the caller drives the graph via
//!   `get_frame`/`run_once`). `KEEP_REF` vs `NOKEEP_REF` collapses: our
//!   frames are always Arc-backed, so `&Frame` + `clone()` is exactly C's
//!   refcounted `KEEP_REF` (zero-copy), and a move would be the same Arc
//!   bump — only the KEEP_REF shape exists.
//! * NOTE: there is no `process_frame` function in this buffersrc.c (the
//!   task brief's line hint is stale) — the param-change checks and the push
//!   live inline in `av_buffersrc_add_frame_flags` (buffersrc.c:227-249,
//!   268), on the `CHECK_VIDEO_PARAM_CHANGE` macro (74-96).
//!
//! ## Companion fix recommended in graph.rs (integrator-owned)
//!
//! C's link-defaults pass replaces an unset link time_base/SAR when
//! `!num && !den` (avfilter.c:397-402), which for a buffer created WITHOUT a
//! `time_base` (0/0 — passes init thanks to the NaN quirk below) heals the
//! link to `AV_TIME_BASE_Q`. The wave-2 `config_links_for` conditions must
//! test `num == 0 && den == 0`, not `== Rational::ZERO` (0/1), or a 0/0
//! buffersrc time_base stays 0/0 on the link. See `divergences` in the port
//! notes; coordinate with the graph.rs owner.

use std::any::Any;

use crate::util::color::{ColorRange, ColorSpace};
use crate::util::error::{Error, Result};
use crate::util::frame::Frame;
use crate::util::pixfmt::PixelFormat;
use crate::util::rational::Rational;
use crate::{log_debug, log_error, log_verbose, log_warning, NOPTS};

use super::filter::{filter_frame, outlink_set_status, set_in_status, FilterDef, FilterFlags, FilterImpl, PadDef, PadRef};
use super::formats;
use super::graph::FilterGraph;
use super::link::NodeId;

// ---------------------------------------------------------------------------
// State (buffersrc.c:44-72 BufferSourceContext, video-only subset)
// ---------------------------------------------------------------------------

/// `BufferSourceContext` — the `buffer` filter's private state. Field names
/// are C's. Not carried: `class`, `hw_frames_ctx`, the audio fields
/// (`sample_rate`/`sample_fmt`/`channels`/`ch_layout`/`side_data`/
/// `nb_side_data`) and `alpha_mode`/`prev_alpha_mode` (no alpha axis).
///
/// `pix_fmt` is `Option<PixelFormat>`: `None` == `AV_PIX_FMT_NONE` (-1).
/// `w`/`h` stay `i32` so C's `<= 0` checks and the "Invalid size %dx%d" text
/// are faithful. `last_pts` starts at 0 (C zero-init), NOT `NOPTS`.
/// `time_base`/`frame_rate`/`pixel_aspect` start 0/0 (C zero-init; `NaN`
/// quirk in `init`).
#[derive(Debug, Default)]
pub struct BufferSource {
    /// `time_base` — to set in the output link.
    pub time_base: Rational,
    /// `frame_rate` — to set in the output link.
    pub frame_rate: Rational,
    /// `nb_failed_requests` — starvation counter (see
    /// [`buffersrc_get_nb_failed_requests`]).
    pub nb_failed_requests: u32,
    /// `warning_limit` — fifo watchdog threshold; 0 until `init` sets 100.
    pub warning_limit: u32,
    pub w: i32,
    pub h: i32,
    pub prev_w: i32,
    pub prev_h: i32,
    pub pix_fmt: Option<PixelFormat>,
    pub prev_pix_fmt: Option<PixelFormat>,
    pub color_space: ColorSpace,
    pub prev_color_space: ColorSpace,
    pub color_range: ColorRange,
    pub prev_color_range: ColorRange,
    /// C name `pixel_aspect`; the option keys are `sar`/`pixel_aspect`.
    pub pixel_aspect: Rational,
    pub eof: bool,
    /// End time of the last pushed frame (its `pts + duration`); the pts the
    /// close path announces. C zero-init: 0, not `NOPTS`.
    pub last_pts: i64,
    pub link_delta: bool,
    pub prev_delta: bool,
}

// ---------------------------------------------------------------------------
// FilterImpl
// ---------------------------------------------------------------------------

impl FilterImpl for BufferSource {
    /// `init_video` (buffersrc.c:318-350) + the option application of
    /// `avfilter_init_dict`/`av_opt_set` over the `buffer_options` table
    /// (buffersrc.c:361-403; parse behavior from opt.c as cited at each
    /// helper). Two phases in C order.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        // ---- Phase 0: unknown-option pre-scan -------------------------------
        // C parses and sets options one by one BEFORE init runs
        // (avfilter.c:855-902 process_options), so `width=1:foo=2` errors on
        // 'foo' at parse time — the unknown option beats init's own
        // validation. The port's dict-consume architecture defers leftover
        // detection to after init, so the C order is restored by scanning
        // keys up front with the same text the leftover check produces.
        const KNOWN_KEYS: &[&str] = &[
            "width", "height", "video_size", "pix_fmt", "sar", "pixel_aspect",
            "time_base", "frame_rate", "colorspace", "range",
        ];
        if let Some((key, _)) = g.nodes[node.0]
            .opts
            .entries
            .iter()
            .find(|(k, _)| !KNOWN_KEYS.contains(&k.as_str()))
        {
            return Err(Error::NotFound(format!("No such option: {key}")));
        }

        // ---- Phase 1: option application ----------------------------------
        // Ordered entries (last duplicate wins, C's AV_DICT_MULTIKEY);
        // recognized keys are consumed, unrecognized ones stay for
        // init_filter's "No such option" check.
        let name = g.nodes[node.0].name.clone();
        let mut entries = std::mem::take(&mut g.nodes[node.0].opts.entries);
        let applied = self.apply_options(&mut entries, &name);
        g.nodes[node.0].opts.entries = entries;
        applied?;

        // ---- Phase 2: validation (C's exact order) -------------------------
        // (a) buffersrc.c:322-325
        if self.pix_fmt.is_none() {
            log_error!(Some(&name), "Unspecified pixel format\n");
            return Err(Error::InvalidArgument("Unspecified pixel format".into()));
        }
        // (b) buffersrc.c:326-332 — the HW-format branch is unreachable here:
        // PixelFormat::ALL carries no AV_PIX_FMT_FLAG_HWACCEL formats
        // (pixfmt.rs). C's text, for citation only:
        //   "Setting BufferSourceContext.pix_fmt to a HW format requires
        //    hw_frames_ctx to be non-NULL!\n"
        // (c) buffersrc.c:333-336
        if self.w <= 0 || self.h <= 0 {
            log_error!(Some(&name), "Invalid size {}x{}\n", self.w, self.h);
            return Err(Error::InvalidArgument(format!(
                "Invalid size {}x{}",
                self.w, self.h
            )));
        }
        // (d) buffersrc.c:337-340 — NaN QUIRK kept bit-for-bit:
        // av_q2d(0/0) is NaN and NaN <= 0 is FALSE, so an UNSET time_base
        // (0/0) PASSES; 0/1 and negatives fail; 1/0 (inf) passes.
        if self.time_base.to_f64() <= 0.0 {
            log_error!(
                Some(&name),
                "Invalid time base {}/{}\n",
                self.time_base.num,
                self.time_base.den
            );
            return Err(Error::InvalidArgument(format!(
                "Invalid time base {}/{}",
                self.time_base.num, self.time_base.den
            )));
        }

        // (e) buffersrc.c:342-347 — the VERBOSE parameter dump. `alpha` is
        // the fixed literal "unspecified" (av_alpha_mode_name of the
        // zero-init alpha_mode, pixdesc.c:3371-3375).
        log_verbose!(
            Some(&name),
            "w:{} h:{} pixfmt:{} tb:{}/{} fr:{}/{} sar:{}/{} csp:{} range:{} alpha:unspecified\n",
            self.w,
            self.h,
            self.pix_fmt.expect("validated above").name(),
            self.time_base.num,
            self.time_base.den,
            self.frame_rate.num,
            self.frame_rate.den,
            self.pixel_aspect.num,
            self.pixel_aspect.den,
            color_space_name(self.color_space),
            color_range_name(self.color_range),
        );

        // (f) common_init (buffersrc.c:310-316), called at the END of
        // init_video in C (349) — after all checks.
        self.warning_limit = 100;
        Ok(())
    }

    /// `query_formats`, video branch (buffersrc.c:476-547, video at
    /// 490-529). Declares halves on OWN pads only; the ENGINE runs
    /// `g.default_query_formats` after — do not call it here. One-shot: never
    /// returns `Err(Again)`.
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let pix_fmt = self.pix_fmt.expect("init validated pix_fmt");
        // (1) singleton pixel-format list (buffersrc.c:494-496). For a source
        // with no inputs this lands in the output link's incfg half — the
        // declared format becomes the negotiated one.
        let pix = g.alloc_pix_list(vec![pix_fmt]);
        g.set_common_formats(node, pix)?;
        // (2) buffersrc.c:491-493 — swfmt == pix_fmt; the hw sw_format lookup
        // is unreachable (no HWACCEL formats in the enum).
        // (3) "force specific colorspace/range downstream only for ordinary
        // YUV" (buffersrc.c:497-516).
        if formats::regular_yuv(pix_fmt) {
            let csp = g.alloc_csp_list(vec![self.color_space]);
            g.set_common_color_spaces(node, csp)?;
            // buffersrc.c:502-504 — the forced-full-range arm (YUVJ formats)
            // is dead code here: formats::forced_full_range always returns
            // false (no YUVJ variants in the enum); the else arm is C's.
            let mut ranges = vec![self.color_range];
            if self.color_range == ColorRange::Unspecified {
                // "allow implicitly promoting unspecified to mpeg"
                // (buffersrc.c:508-512). ORDER matters: pick_format takes
                // element 0, so [Unspecified, Mpeg] yields an Unspecified
                // link range unless the downstream forces mpeg.
                ranges.push(ColorRange::Mpeg);
            }
            let rng = g.alloc_rng_list(ranges);
            g.set_common_color_ranges(node, rng)?;
        }
        // (4) buffersrc.c:517-527 — the alpha-mode block (ALPHA formats like
        // rgba/gbrap) is NOT ported: no alpha negotiation axis. For
        // NON-regular-yuv formats (gray8, rgb24, gbrp) NO csp/rng halves are
        // declared — the engine's default fill covers them and pick_format's
        // non-regular branch assigns Rgb/Jpeg; grayscale is full-range by
        // convention, exactly C.
        Ok(())
    }

    /// `config_props`, video branch (buffersrc.c:549-591: video 555-565, tail
    /// 588-590). C wires it ONLY on the output pad
    /// (`avfilter_vsrc_buffer_outputs`, buffersrc.c:611-617).
    ///
    /// Does NOT set link.format / colorspace / color_range — the format comes
    /// from pick_format over the query-declared lists, the color fields from
    /// pick_format's list selection.
    fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
        let PadRef::Out(0) = pad else {
            return Ok(()); // not our output pad
        };
        let l = g.outlink(node, 0);
        // buffersrc.c:556-558 …
        g.links[l.0].w = self.w as u32;
        g.links[l.0].h = self.h as u32;
        g.links[l.0].sample_aspect_ratio = self.pixel_aspect;
        // buffersrc.c:560-564 — hw_frames_ctx ref: skipped (unreachable).
        // buffersrc.c:577-586 — side_data clone: skipped (unported).
        // buffersrc.c:588-589 — the tail assignments.
        g.links[l.0].time_base = self.time_base;
        g.links[l.0].frame_rate = self.frame_rate;
        Ok(())
    }

    /// Unreachable — the buffer source has no input pads (mirrors graph.rs's
    /// TestSrc shape).
    fn filter_frame(
        &mut self,
        _g: &mut FilterGraph,
        _node: NodeId,
        _pad: usize,
        _frame: Frame,
    ) -> Result<()> {
        Err(Error::Unsupported(
            "buffer source has no input pads".into(),
        ))
    }

    /// `activate` (buffersrc.c:593-609) — exact three-stage port.
    fn activate(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let outlink = g.outlink(node, 0);
        // (1) buffersrc.c:598-601 — the downstream closed the link
        // (ff_outlink_get_status is status_in viewed from the src side; a
        // hard downstream close via inlink_set_status sets status_in).
        // Latches eof and returns IMMEDIATELY: the out-status push happens on
        // the NEXT activation.
        if !self.eof && g.links[outlink.0].status_in.is_some() {
            self.eof = true;
            return Ok(());
        }
        // (2) buffersrc.c:603-606 — ff_outlink_set_status == set_in_status;
        // a second firing with the same Eof is the same-status no-op.
        if self.eof {
            outlink_set_status(g, outlink, Error::Eof, self.last_pts);
            return Ok(());
        }
        // (3) buffersrc.c:607-608 — FFERROR_BUFFERSRC_EMPTY is represented as
        // Ok(()) per the wave-1 pinned mapping (see the module doc's
        // buffersink warning). frame_wanted_out is deliberately NOT cleared
        // on this path (C leaves it set; a later add_frame clears it through
        // filter_frame). The counter wraps like C's unsigned.
        self.nb_failed_requests = self.nb_failed_requests.wrapping_add(1);
        Ok(())
    }

    /// Downcast hook for the runtime API: the imp is taken out of the node
    /// (take/restore dance) and downcast to the concrete filter so its state
    /// can be driven alongside `&mut FilterGraph`.
    fn as_any(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

impl BufferSource {
    /// Phase 1 of `init`: apply recognized entries from the option list, in
    /// order, removing them; a parse failure aborts immediately (C's
    /// av_opt_set return), leaving the remaining entries unapplied.
    fn apply_options(
        &mut self,
        entries: &mut Vec<(String, String)>,
        name: &str,
    ) -> Result<()> {
        let mut i = 0;
        while i < entries.len() {
            let (key, val) = entries[i].clone();
            // buffer_options (buffersrc.c:361-403): recognized keys. Values
            // parsed per the AVOption type each entry declares (opt.c cites
            // at the helpers).
            let recognized = match key.as_str() {
                "width" => {
                    self.w = parse_int(name, "width", &val)?;
                    true
                }
                "height" => {
                    self.h = parse_int(name, "height", &val)?;
                    true
                }
                // AV_OPT_TYPE_IMAGE_SIZE at OFFSET(w): writes int[2] — BOTH
                // w and h (buffersrc.c:363, opt.c:522-539).
                "video_size" => {
                    let (w, h) = parse_image_size(name, &val)?;
                    self.w = w;
                    self.h = h;
                    true
                }
                "pix_fmt" => {
                    self.pix_fmt = parse_pix_fmt(name, &val)?;
                    true
                }
                // Both keys share OFFSET(pixel_aspect) (buffersrc.c:366-367).
                "sar" | "pixel_aspect" => {
                    self.pixel_aspect = parse_rational(name, &key, &val)?;
                    true
                }
                "time_base" => {
                    self.time_base = parse_rational(name, "time_base", &val)?;
                    true
                }
                "frame_rate" => {
                    self.frame_rate = parse_rational(name, "frame_rate", &val)?;
                    true
                }
                "colorspace" => {
                    self.color_space = parse_csp_name(name, &val)?;
                    true
                }
                "range" => {
                    self.color_range = parse_range_name(name, &val)?;
                    true
                }
                // "alpha_mode" (buffersrc.c:397-401) is deliberately absent —
                // no alpha axis; it stays a leftover ("No such option:
                // alpha_mode"), the documented degradation.
                _ => false,
            };
            if recognized {
                entries.remove(i);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// `CHECK_VIDEO_PARAM_CHANGE` (buffersrc.c:74-96, invoked at 231-233) —
    /// the on-the-fly property-change detector. NEVER errors for video
    /// (contrast the audio macro's EINVAL, buffersrc.c:105). Runs on the
    /// ORIGINAL frame values, before the link-fill of the copy.
    ///
    /// Hysteresis walk (options-only init leaves prev_* zeroed):
    /// frame #1 (matching declared): link_delta=false, prev_delta=true →
    /// silent VERBOSE sync; frame #2 (different): link_delta=true,
    /// prev_delta=false → DEBUG; frame #3 (different again): both true →
    /// WARNING.
    fn check_video_param_change(&mut self, g: &FilterGraph, node: NodeId, frame: &Frame) {
        // (74-78) — alpha_mode comparisons dropped (no axis).
        self.link_delta = self.w != frame.width as i32
            || self.h != frame.height as i32
            || self.pix_fmt != Some(frame.format)
            || self.color_space != frame.color_space
            || self.color_range != frame.color_range;
        self.prev_delta = self.prev_w != frame.width as i32
            || self.prev_h != frame.height as i32
            || self.prev_pix_fmt != Some(frame.format)
            || self.prev_color_space != frame.color_space
            || self.prev_color_range != frame.color_range;

        // pts_time against the OUT link's time base (buffersrc.c:85, 89).
        let outlink = g.outlink(node, 0);
        let pts_time = ts2timestr(frame.pts, g.links[outlink.0].time_base);
        let name = g.nodes[node.0].name.clone();

        if self.link_delta {
            // int loglevel = c->prev_delta ? AV_LOG_WARNING : AV_LOG_DEBUG
            // (buffersrc.c:80).
            let (cw, ch, cf, cc, cr) =
                (self.w, self.h, self.pix_fmt, self.color_space, self.color_range);
            // (81) then (82-85), same level. The `fmt:` fields print the
            // numeric pixel format — the PixelFormat::ALL index here, which
            // coincides with C's AVPixelFormat value only for yuv420p (0);
            // cosmetic divergence, documented. The two `alpha:` slots print
            // the fixed literal "unspecified".
            let detail = format!(
                "filter context - w: {cw} h: {ch} fmt: {} csp: {} range: {} alpha: unspecified, \
                 incoming frame - w: {} h: {} fmt: {} csp: {} range: {} alpha: unspecified pts_time: {pts_time}\n",
                pix_fmt_number(cf),
                color_space_name(cc),
                color_range_name(cr),
                frame.width,
                frame.height,
                pix_fmt_number(Some(frame.format)),
                color_space_name(frame.color_space),
                color_range_name(frame.color_range),
            );
            if self.prev_delta {
                log_warning!(
                    Some(&name),
                    "Changing video frame properties on the fly is not supported by all filters.\n"
                );
                log_warning!(Some(&name), "{detail}");
            } else {
                log_debug!(
                    Some(&name),
                    "Changing video frame properties on the fly is not supported by all filters.\n"
                );
                log_debug!(Some(&name), "{detail}");
            }
        }
        if self.prev_delta {
            // (87-89)
            if !self.link_delta {
                log_verbose!(
                    Some(&name),
                    "video frame properties congruent with link at pts_time: {pts_time}\n"
                );
            }
            // (90-95) — unconditional inside prev_delta; prev_alpha_mode
            // dropped.
            self.prev_w = frame.width as i32;
            self.prev_h = frame.height as i32;
            self.prev_pix_fmt = Some(frame.format);
            self.prev_color_space = frame.color_space;
            self.prev_color_range = frame.color_range;
        }
    }
}

// ---------------------------------------------------------------------------
// Option value parsers (opt.c / parseutils.c subsets)
// ---------------------------------------------------------------------------

/// The generic parse-failure error: opt.c:500's
/// `Unable to parse "%s" option value "%s"\n` (set_string_number's expression
/// failure) — used for bad width/height ints, bad rationals and unknown
/// colorspace/range CONST names.
fn parse_failed(name: &str, key: &str, val: &str) -> Error {
    log_error!(Some(name), "Unable to parse \"{key}\" option value \"{val}\"\n");
    Error::InvalidArgument(format!("Unable to parse \"{key}\" option value \"{val}\""))
}

/// C `strtol(s, &p, 10)` core: optional leading whitespace and sign, then
/// decimal digits. Returns `(value, bytes_consumed)`; consumed == 0 when no
/// digits were found (C's "no conversion" case). Overflow saturates (C clamps
/// to LONG_MAX; values that large fail the size checks anyway).
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
    let n: i64 = text.parse().unwrap_or(if b[start] == b'-' {
        i64::MIN
    } else {
        i64::MAX
    });
    (n.clamp(i32::MIN as i64, i32::MAX as i64) as i32, i)
}

/// `AV_OPT_TYPE_INT` (width/height, buffersrc.c:362/364) via
/// set_string_number. Decimal only — C's expression evaluator additionally
/// accepts hex/float spellings (av_expr), an accepted cosmetic narrowing.
/// Negative values parse fine here and are rejected by init's own
/// `w <= 0 || h <= 0` check (C would fail the option's min=0 range check
/// slightly earlier; same `EINVAL` outcome, different text).
fn parse_int(name: &str, key: &str, val: &str) -> Result<i32> {
    let t = val.trim();
    if !t.is_empty() {
        if let Ok(n) = t.parse::<i32>() {
            return Ok(n);
        }
    }
    Err(parse_failed(name, key, val))
}

/// `AV_OPT_TYPE_RATIONAL` (sar/pixel_aspect/time_base/frame_rate) via
/// opt.c:434-441: `sscanf(val, "%d%*1[:/]%d%c")` — "n/d" (one separator
/// character; "n:" cannot reach us because create_filter splits options on
/// ':') or a bare integer "n" → n/1 (the sscanf returns 1 and the expression
/// fallback stores den=1). "1/0" is ACCEPTED (C's sscanf takes it; init's
/// av_q2d check passes since inf > 0).
fn parse_rational(name: &str, key: &str, val: &str) -> Result<Rational> {
    let t = val.trim();
    let (num, consumed) = strtol_i32(t);
    if consumed > 0 {
        let rest = &t[consumed..];
        if rest.is_empty() {
            return Ok(Rational::new(num, 1));
        }
        if rest.as_bytes()[0] == b'/' {
            let (den, consumed2) = strtol_i32(&rest[1..]);
            if consumed2 > 0 && rest[1 + consumed2..].is_empty() {
                return Ok(Rational::new(num, den));
            }
        }
    }
    Err(parse_failed(name, key, val))
}

/// `AV_OPT_TYPE_IMAGE_SIZE` (`video_size`, buffersrc.c:363) via
/// set_string_image_size (opt.c:521-539) + `av_parse_video_size`
/// (parseutils.c:150-179), minus the named-size abbreviation table
/// (ntsc/pal/vga/…). "none" zeroes both (C), which init then rejects as
/// "Invalid size 0x0".
fn parse_image_size(name: &str, val: &str) -> Result<(i32, i32)> {
    if val == "none" {
        return Ok((0, 0));
    }
    let (w, mut p) = strtol_i32(val);
    if p < val.len() {
        p += 1; // `if (*p) p++` — skip the ONE separator character (usually 'x')
    }
    let tail = val.get(p..).unwrap_or("");
    let (h, consumed2) = strtol_i32(tail);
    // "trailing extraneous data detected, like in 123x345foobar"
    if consumed2 != tail.len() || w <= 0 || h <= 0 {
        log_error!(
            Some(name),
            "Unable to parse \"video_size\" option value \"{val}\" as image size\n"
        );
        return Err(Error::InvalidArgument(format!(
            "Unable to parse \"video_size\" option value \"{val}\" as image size"
        )));
    }
    Ok((w, h))
}

/// `AV_OPT_TYPE_PIXEL_FMT` (`pix_fmt`, buffersrc.c:365) via set_string_fmt /
/// set_string_pixel_fmt (opt.c:613-669): "none" → `AV_PIX_FMT_NONE` (= our
/// `None`); otherwise the name lookup. The numeric-string fallback
/// (opt.c:621-627 strtol) is NOT ported — only names.
fn parse_pix_fmt(name: &str, val: &str) -> Result<Option<PixelFormat>> {
    if val == "none" {
        return Ok(None);
    }
    match PixelFormat::from_name(val) {
        Some(f) => Ok(Some(f)),
        None => {
            log_error!(
                Some(name),
                "Unable to parse \"pix_fmt\" option value \"{val}\" as pixel format\n"
            );
            Err(Error::InvalidArgument(format!(
                "Unable to parse \"pix_fmt\" option value \"{val}\" as pixel format"
            )))
        }
    }
}

/// The `colorspace` CONST names our `ColorSpace` enum carries
/// (buffersrc.c:371-381). C additionally accepts ycgco/ycgco-re/ycgco-ro/
/// bt2020c/smpte2085/chroma-derived-nc/chroma-derived-c/ictcp/ipt-c2 — all
/// absent from util/color.rs, so those names error here (documented
/// degradation: generic "Unable to parse" text, exactly what C prints for a
/// name with no matching CONST).
fn parse_csp_name(name: &str, val: &str) -> Result<ColorSpace> {
    let csp = match val {
        "gbr" => ColorSpace::Rgb,
        "bt709" => ColorSpace::Bt709,
        "unknown" => ColorSpace::Unspecified,
        "fcc" => ColorSpace::Fcc,
        "bt470bg" => ColorSpace::Bt470bg,
        "smpte170m" => ColorSpace::Smpte170m,
        "smpte240m" => ColorSpace::Smpte240m,
        "bt2020nc" => ColorSpace::Bt2020Ncl,
        _ => return Err(parse_failed(name, "colorspace", val)),
    };
    Ok(csp)
}

/// The `range` CONST names (buffersrc.c:389-396).
fn parse_range_name(name: &str, val: &str) -> Result<ColorRange> {
    let range = match val {
        "unspecified" | "unknown" => ColorRange::Unspecified,
        "limited" | "tv" | "mpeg" => ColorRange::Mpeg,
        "full" | "pc" | "jpeg" => ColorRange::Jpeg,
        _ => return Err(parse_failed(name, "range", val)),
    };
    Ok(range)
}

// ---------------------------------------------------------------------------
// Name tables (pixdesc.c:3276-3280, 3330-3349) + ts2timestr (timestamp.c)
// ---------------------------------------------------------------------------

/// `av_color_space_name` subset for the enum values the port carries
/// (pixdesc.c:3330-3349).
fn color_space_name(csp: ColorSpace) -> &'static str {
    match csp {
        ColorSpace::Rgb => "gbr",
        ColorSpace::Bt709 => "bt709",
        ColorSpace::Unspecified => "unknown",
        ColorSpace::Reserved => "reserved",
        ColorSpace::Fcc => "fcc",
        ColorSpace::Bt470bg => "bt470bg",
        ColorSpace::Smpte170m => "smpte170m",
        ColorSpace::Smpte240m => "smpte240m",
        ColorSpace::Bt2020Ncl => "bt2020nc",
    }
}

/// `av_color_range_name` (pixdesc.c:3276-3280).
fn color_range_name(range: ColorRange) -> &'static str {
    match range {
        ColorRange::Unspecified => "unknown",
        ColorRange::Mpeg => "tv",
        ColorRange::Jpeg => "pc",
    }
}

/// The numeric form for the `fmt:` log slots: C prints the `AVPixelFormat`
/// discriminant; the port's `PixelFormat` has none, so this is the index in
/// `PixelFormat::ALL`. Coincides with C only for yuv420p (0); `None`
/// (`AV_PIX_FMT_NONE`) prints -1 like C.
fn pix_fmt_number(fmt: Option<PixelFormat>) -> i32 {
    match fmt {
        None => -1,
        Some(f) => PixelFormat::ALL
            .iter()
            .position(|&p| p == f)
            .map(|i| i as i32)
            .unwrap_or(-1),
    }
}

/// `av_ts_make_time_string2` (libavutil/timestamp.c:21-36, via
/// `av_ts2timestr`, timestamp.h) — the pts_time formatter of the two
/// CHECK_VIDEO_PARAM_CHANGE logs. Byte-faithful port of both trim loops:
/// the first strips trailing '0's (stopping at index 0), the second strips
/// one separator-ish char ('.'/'-' are < '0' in ASCII) subject to the same
/// index-0 guard — net effect "4.000000" → "4", "0.500000" → "0.5",
/// "10.000000" → "10", exactly as the compiled C behaves (verified against
/// the C source; the port spec's "trailing '.' survives" reading of loop 2
/// is incorrect — see divergences). `NOPTS` prints "NOPTS".
fn ts2timestr(ts: i64, tb: Rational) -> String {
    if ts == NOPTS {
        return "NOPTS".to_string();
    }
    let val = tb.to_f64() * ts as f64; // av_q2d(tb) * ts
    let log = if val == 0.0 {
        f64::NEG_INFINITY // fpclassify(val) == FP_ZERO
    } else {
        val.abs().log10().floor()
    };
    let precision: usize = if log.is_finite() && log < 0.0 {
        (-log + 5.0) as usize
    } else {
        6
    };
    let mut b = format!("{val:.precision$}").into_bytes();
    let mut last = b.len() - 1;
    // timestamp.c:31 — for (; last && buf[last] == '0'; last--);
    while last != 0 && b[last] == b'0' {
        last -= 1;
    }
    // timestamp.c:32 — for (; last && buf[last] != 'f' && (buf[last] < '0' ||
    // buf[0] > '9'); last--);
    while last != 0 && b[last] != b'f' && (b[last] < b'0' || b[0] > b'9') {
        last -= 1;
    }
    b.truncate(last + 1);
    String::from_utf8(b).expect("digits, '.', '-' and maybe 'NaN' only")
}

// ---------------------------------------------------------------------------
// Filter definition (buffersrc.c:611-630)
// ---------------------------------------------------------------------------

const DEFAULT_PAD: PadDef = PadDef {
    name: "default",
    needs_writable: false,
};

/// `ff_vsrc_buffer` (buffersrc.c:619-630) + `avfilter_vsrc_buffer_outputs`
/// (611-617). The pad's `.type`/`.config_props` fields are expressed by the
/// `FilterImpl` trait instead (config_props matches `PadRef::Out(0)` itself).
///
/// Shorthand = `ff_filter_opt_parse`'s walk of buffer_options skipping
/// `AV_OPT_TYPE_CONST` and duplicate OFFSETs (avfilter.c:866-871):
/// width/height/pix_fmt/sar/time_base/frame_rate/colorspace/range —
/// `video_size` (dup offset of width) and `pixel_aspect` (dup of sar) are
/// EXPLICIT-KEY ONLY; C's trailing `alpha_mode` slot is dropped (unported).
pub static BUFFER_SRC_DEF: FilterDef = FilterDef {
    name: "buffer",
    inputs: &[],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags(0),
    shorthand: &[
        "width",
        "height",
        "pix_fmt",
        "sar",
        "time_base",
        "frame_rate",
        "colorspace",
        "range",
    ],
    make: || Box::new(BufferSource::default()),
};

// ---------------------------------------------------------------------------
// Runtime API (buffersrc.c:185-308) — the free functions the graph stubs
// (add_frame/close_source) and the CLI delegate to.
// ---------------------------------------------------------------------------

/// The take/restore dance + downcast shared by every runtime entry point
/// (same shape as `init_filter` graph.rs / `run_once` filter.rs). The imp
/// leaves the node for the duration so the concrete state and `&mut
/// FilterGraph` are usable side by side; the slot must stay empty meanwhile.
fn with_buffer_source<T>(
    g: &mut FilterGraph,
    node: NodeId,
    f: impl FnOnce(&mut FilterGraph, &mut BufferSource) -> Result<T>,
) -> Result<T> {
    let mut imp = g.nodes[node.0]
        .imp
        .take()
        .expect("with_buffer_source: node imp already taken");
    let ret = match imp.as_any().and_then(|a| a.downcast_mut::<BufferSource>()) {
        Some(s) => f(g, s),
        None => Err(Error::InvalidArgument(format!(
            "filter '{}' is not a buffer source",
            g.nodes[node.0].name
        ))),
    };
    debug_assert!(
        g.nodes[node.0].imp.is_none(),
        "buffersrc: imp was restored while taken"
    );
    g.nodes[node.0].imp = Some(imp);
    ret
}

/// `av_buffersrc_write_frame` (buffersrc.c:185-189) ==
/// `av_buffersrc_add_frame_flags(KEEP_REF)`, and with `frame == None` also
/// covers `av_buffersrc_add_frame(ctx, NULL)` — the EOF-at-last-pts close.
///
/// The graph must be wired (and, for the frame checks to be meaningful,
/// configured) before the call: `g.outlink` panics on an unlinked pad where
/// C would dereference NULL.
///
/// KEEP_REF for our Arc-backed `Frame` is a plain `clone()` — plane refcount
/// bumps, never a pixel copy; there is no non-refcounted copy path to port.
pub fn buffersrc_add_frame(
    g: &mut FilterGraph,
    node: NodeId,
    frame: Option<&Frame>,
) -> Result<()> {
    with_buffer_source(g, node, |g, s| {
        buffersrc_add_frame_inner(g, s, node, frame)
    })
}

/// The body of `av_buffersrc_add_frame_flags` (buffersrc.c:210-289),
/// KEEP_REF, flags = 0 (no PUSH drain). C's step numbers in the comments.
fn buffersrc_add_frame_inner(
    g: &mut FilterGraph,
    s: &mut BufferSource,
    node: NodeId,
    frame: Option<&Frame>,
) -> Result<()> {
    // (1) buffersrc.c:216 — BEFORE the NULL/eof checks: even a NULL add or
    // an eof-rejected add resets the starvation counter.
    s.nb_failed_requests = 0;

    // (2) buffersrc.c:218-219 — NULL frame: close at the last pushed pts.
    let Some(frame) = frame else {
        let pts = s.last_pts;
        return buffersrc_close_inner(g, s, node, pts);
    };

    // (3) buffersrc.c:220-221 — BEFORE last_pts is touched.
    if s.eof {
        return Err(Error::Eof);
    }

    // (4) buffersrc.c:223 — wrapping: NOPTS (i64::MIN) + duration > 0 must
    // wrap like C's unchecked add, not panic under debug arithmetic.
    s.last_pts = frame.pts.wrapping_add(frame.duration);

    // (5) buffersrc.c:227-249 — NO_CHECK_FORMAT not ported (checks always
    // run; the video checks never fail, they only log). The audio branch
    // (235-244, with its EINVAL) and the default case (245-246) are not
    // ported — video-only.
    s.check_video_param_change(g, node, frame);

    // (6) buffersrc.c:251-259 — the KEEP_REF clone (av_frame_clone of a
    // refcounted frame: cheap Arc share).
    let mut copy = frame.clone();

    // (7) buffersrc.c:261-266 — fill unspecified color fields from the
    // output link (the alpha_mode fill at 265-266 is skipped: no axis).
    // Runs OUTSIDE the NOCHECK block in C, so always.
    let outlink = g.outlink(node, 0);
    if copy.color_space == ColorSpace::Unspecified {
        copy.color_space = g.links[outlink.0].colorspace;
    }
    if copy.color_range == ColorRange::Unspecified {
        copy.color_range = g.links[outlink.0].color_range;
    }

    // (8) buffersrc.c:268-270 — the engine push: validates (debug asserts),
    // queues into the destination fifo, clears frame_wanted_out /
    // frame_blocked_in, readies the destination at 300. Errors propagate
    // verbatim.
    filter_frame(g, outlink, copy)?;

    // (9) buffersrc.c:272-276 — the PUSH synchronous drain is NOT ported:
    // the caller drives the graph (get_frame / run_once).

    // (10) buffersrc.c:278-286 — the queued-buffers watchdog. Counts ONLY
    // the frames queued on the source's OWN output link; the number printed
    // is the LIMIT, not the queue length; the name is the instance name
    // (av_x_if_null(ctx->name, filter->name)).
    let queued = g.links[outlink.0].fifo.len() as u32;
    if s.warning_limit != 0 && queued >= s.warning_limit {
        let name = g.nodes[node.0].name.clone();
        log_warning!(
            Some(&name),
            "{} buffers queued in {}, something may be wrong.\n",
            s.warning_limit,
            name
        );
        s.warning_limit = s.warning_limit.wrapping_mul(10); // C unsigned wrap
    }

    Ok(())
}

/// `av_buffersrc_close` (buffersrc.c:291-298), flags = 0 (the PUSH branch of
/// 297 is not ported). `pts` is in the OUTPUT link's time base — callers
/// pass the end time of the last frame (`last_pts`) or a stream-derived
/// value.
///
/// IDEMPOTENT for the same Eof status: `set_in_status`'s same-status
/// early-return keeps the FIRST `status_in_pts` (a second close with a
/// different pts does not move it); a second close after a downstream
/// NON-Eof in-status trips `set_in_status`'s debug assert, same as C's
/// assertion there. Does not require the fifo to be drained — the status is
/// HELD behind queued frames until the destination acknowledges it.
pub fn buffersrc_close(g: &mut FilterGraph, node: NodeId, pts: i64) -> Result<()> {
    with_buffer_source(g, node, |g, s| buffersrc_close_inner(g, s, node, pts))
}

/// `av_buffersrc_close`'s body — shared with add_frame's NULL path so it can
/// run without re-taking the imp.
fn buffersrc_close_inner(
    g: &mut FilterGraph,
    s: &mut BufferSource,
    node: NodeId,
    pts: i64,
) -> Result<()> {
    // buffersrc.c:295
    s.eof = true;
    // buffersrc.c:296 — ff_avfilter_link_set_in_status (filter.rs
    // set_in_status): clears frame_wanted_out/frame_blocked_in, unblocks
    // and readies the destination at 200.
    let outlink = g.outlink(node, 0);
    set_in_status(g, outlink, Error::Eof, pts);
    Ok(())
}

/// `av_buffersrc_get_nb_failed_requests` (buffersrc.c:352-355).
///
/// This is ALSO the wave-2 replacement for C's FFERROR_BUFFERSRC_EMPTY
/// latch: since `activate` maps BUFFERSRC_EMPTY to `Ok(())`, a drain loop
/// detects "source starved, go feed it" by polling this counter (the ffmpeg
/// CLI's technique) instead of inspecting run_once's return. Semantics:
/// incremented once per activation that reaches stage 3; reset to 0 by EVERY
/// `buffersrc_add_frame` call — including the NULL/eof-rejected ones.
///
/// Needs `&mut FilterGraph` because the state sits inside the boxed impl
/// (take/restore) — acceptable for a polling API in this single-threaded
/// port. C returns `unsigned` and cannot fail; this panics if the node is
/// not a buffer source (C would read the wrong priv pointer).
pub fn buffersrc_get_nb_failed_requests(g: &mut FilterGraph, node: NodeId) -> u32 {
    with_buffer_source(g, node, |_g, s| Ok(s.nb_failed_requests))
        .expect("buffersrc_get_nb_failed_requests on a buffer source")
}

/// `av_buffersrc_get_status` (buffersrc.c:300-308): `Ok(())` = still open,
/// `Err(Error::Eof)` = closed (by close, by add(NULL), or by the downstream
/// no longer accepting). NOT a pure predicate: it latches `eof` from a
/// downstream close, exactly like C.
pub fn buffersrc_get_status(g: &mut FilterGraph, node: NodeId) -> Result<()> {
    with_buffer_source(g, node, |g, s| {
        // buffersrc.c:304-305
        if !s.eof {
            let outlink = g.outlink(node, 0);
            if g.links[outlink.0].status_in.is_some() {
                s.eof = true;
            }
        }
        // buffersrc.c:307 — s->eof ? AVERROR(EOF) : 0
        if s.eof {
            Err(Error::Eof)
        } else {
            Ok(())
        }
    })
}

/// Read the source's `last_pts` (C exposes it only through the close path;
/// `FilterGraph::close_source`'s NULL-frame delegation already carries it,
/// so this accessor is a convenience for graph.rs / tests). Panics like
/// [`buffersrc_get_nb_failed_requests`] on a non-buffer node.
pub fn buffersrc_last_pts(g: &mut FilterGraph, node: NodeId) -> i64 {
    with_buffer_source(g, node, |_g, s| Ok(s.last_pts))
        .expect("buffersrc_last_pts on a buffer source")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::filter;
    use crate::filter::graph::engine_test_helpers::run_to_quiescence;
    use crate::filter::link::LinkId;
    use crate::util::pixfmt::PixelFormat::*;

    /// buffer → testsink, created + linked; the graph is NOT configured —
    /// tests call [`configure`] where needed.
    fn buffer_graph(args: &str) -> (FilterGraph, NodeId, NodeId, LinkId) {
        let mut g = FilterGraph::new();
        let src = g.create_filter("buffer", args).unwrap();
        let sink = g.alloc_test_sink();
        let l = g.link(src, 0, sink, 0).unwrap();
        (g, src, sink, l)
    }

    /// Run the node's query_formats + the engine's default fill + the output
    /// config_props — the per-node slice of what the wave-2 engine does
    /// (avfiltergraph.c:410 query round, then graph_config_links).
    fn configure(g: &mut FilterGraph, src: NodeId) {
        with_imp(g, src, |g, imp| {
            imp.query_formats(g, src).unwrap();
            g.default_query_formats(src).unwrap();
            imp.config_props(g, src, PadRef::Out(0)).unwrap();
        });
    }

    /// Take/restore around `&mut dyn FilterImpl` — for driving trait methods
    /// directly in tests (the engine's own dance).
    fn with_imp<T>(
        g: &mut FilterGraph,
        node: NodeId,
        f: impl FnOnce(&mut FilterGraph, &mut dyn FilterImpl) -> T,
    ) -> T {
        let mut imp = g.nodes[node.0].imp.take().expect("imp taken");
        let out = f(g, imp.as_mut());
        g.nodes[node.0].imp = Some(imp);
        out
    }

    /// Take/restore around `&mut BufferSource` — for asserting on state.
    fn with_state<T>(
        g: &mut FilterGraph,
        node: NodeId,
        f: impl FnOnce(&mut BufferSource, &FilterGraph) -> T,
    ) -> T {
        let mut imp = g.nodes[node.0].imp.take().expect("imp taken");
        let s = imp
            .as_any()
            .and_then(|a| a.downcast_mut::<BufferSource>())
            .expect("buffer source");
        let out = f(s, g);
        g.nodes[node.0].imp = Some(imp);
        out
    }

    fn video_frame(fmt: PixelFormat, w: u32, h: u32, pts: i64) -> Frame {
        let mut f = Frame::alloc(fmt, w, h).unwrap();
        f.pts = pts;
        f.duration = 40;
        f
    }

    // ---- init: validation errors (buffersrc.c:318-350) --------------------

    /// buffersrc.c:322-325.
    #[test]
    fn init_missing_pix_fmt() {
        let mut g = FilterGraph::new();
        let err = g
            .create_filter("buffer", "width=320:height=240:time_base=1/25")
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        assert!(err.to_string().contains("Unspecified pixel format"));
        // The C "none" sentinel maps to NONE and hits the same check.
        let err = g
            .create_filter("buffer", "pix_fmt=none:video_size=1x1:time_base=1/1")
            .unwrap_err();
        assert!(err.to_string().contains("Unspecified pixel format"));
    }

    /// buffersrc.c:333-336.
    #[test]
    fn init_invalid_size() {
        let mut g = FilterGraph::new();
        let err = g
            .create_filter("buffer", "pix_fmt=yuv420p:time_base=1/25")
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        assert!(err.to_string().contains("Invalid size 0x0"));
    }

    /// buffersrc.c:337-340 + the NaN quirk: an UNSET (0/0) time_base passes
    /// because av_q2d(0/0) = NaN and NaN <= 0 is false; 0/1 fails; 1/0
    /// (inf) passes.
    #[test]
    fn init_invalid_time_base() {
        let mut g = FilterGraph::new();
        let err = g
            .create_filter(
                "buffer",
                "pix_fmt=yuv420p:video_size=320x240:time_base=0/1",
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        assert!(err.to_string().contains("Invalid time base 0/1"));

        // NaN quirk: no time_base at all (C zero-init 0/0) — init succeeds.
        let src = g
            .create_filter("buffer", "pix_fmt=yuv420p:video_size=320x240")
            .unwrap();
        with_state(&mut g, src, |s, _| {
            assert_eq!(s.time_base, Rational::new(0, 0));
            assert_eq!(s.warning_limit, 100);
        });
        // inf passes: av_q2d(1/0) = inf > 0.
        g.create_filter("buffer", "pix_fmt=yuv420p:video_size=320x240:time_base=1/0")
            .unwrap();
    }

    /// Option application + config_props (buffersrc.c:361-403, 555-590) and
    /// the query singletons (490-516).
    #[test]
    fn init_option_parsing_and_config_props() {
        let (mut g, src, _sink, _l) = buffer_graph(
            "video_size=320x240:pix_fmt=yuv420p:time_base=1/25:pixel_aspect=1/1:frame_rate=25/1:colorspace=bt709:range=tv",
        );
        configure(&mut g, src);
        let l = g.outlink(src, 0);
        // config_props (buffersrc.c:556-558, 588-589)
        assert_eq!(g.links[l.0].w, 320);
        assert_eq!(g.links[l.0].h, 240);
        assert_eq!(g.links[l.0].sample_aspect_ratio, Rational::new(1, 1));
        assert_eq!(g.links[l.0].time_base, Rational::new(1, 25));
        assert_eq!(g.links[l.0].frame_rate, Rational::new(25, 1));
        // query_formats singletons (494-514) — declared into incfg.
        assert_eq!(
            g.fmt_lists[g.links[l.0].incfg.formats.unwrap() as usize],
            vec![Yuv420p]
        );
        assert_eq!(
            g.csp_lists[g.links[l.0].incfg.color_spaces.unwrap() as usize],
            vec![ColorSpace::Bt709]
        );
        assert_eq!(
            g.rng_lists[g.links[l.0].incfg.color_ranges.unwrap() as usize],
            vec![ColorRange::Mpeg]
        );
        // prev_* are NEVER written by the options path (only
        // av_buffersrc_parameters_set writes them, and it is not ported).
        with_state(&mut g, src, |s, _| {
            assert_eq!(s.w, 320);
            assert_eq!(s.h, 240);
            assert_eq!(s.pix_fmt, Some(Yuv420p));
            assert_eq!(s.prev_w, 0);
            assert_eq!(s.prev_h, 0);
            assert_eq!(s.prev_pix_fmt, None);
            assert_eq!(s.warning_limit, 100);
        });

        // Positional shorthand: width/height/pix_fmt/sar/time_base/frame_rate.
        let (mut g2, src2, _sink2, _l2) = buffer_graph("64:48:gray:1/1:1/25:25/1");
        with_state(&mut g2, src2, |s, _| {
            assert_eq!(s.w, 64);
            assert_eq!(s.h, 48);
            assert_eq!(s.pix_fmt, Some(Gray8));
            assert_eq!(s.pixel_aspect, Rational::new(1, 1));
            assert_eq!(s.time_base, Rational::new(1, 25));
            assert_eq!(s.frame_rate, Rational::new(25, 1));
        });
    }

    /// Unknown options are leftovers: init_filter reports the FIRST one
    /// (avfilter.c:976-980). `alpha_mode` and `sws_param` are the documented
    /// degradations (no alpha axis; this buffersrc.c has no sws_param).
    #[test]
    fn init_unknown_option() {
        let mut g = FilterGraph::new();
        let err = g.create_filter("buffer", "width=1:foo=2").unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
        assert!(err.to_string().contains("No such option: foo"));
        // alpha_mode: recognized by C, unported here.
        let err = g
            .create_filter(
                "buffer",
                "video_size=1x1:pix_fmt=gray:time_base=1/1:alpha_mode=straight",
            )
            .unwrap_err();
        assert!(err.to_string().contains("No such option: alpha_mode"));
        // sws_param: does not exist in this buffersrc.c at all.
        let err = g
            .create_filter(
                "buffer",
                "video_size=1x1:pix_fmt=gray:time_base=1/1:sws_param=1",
            )
            .unwrap_err();
        assert!(err.to_string().contains("No such option: sws_param"));
    }

    /// Bad option VALUES abort init with the generic opt.c:500 text.
    #[test]
    fn init_bad_option_values() {
        let mut g = FilterGraph::new();
        for (args, needle) in [
            ("width=abc:height=1:pix_fmt=gray:time_base=1/1", "width"),
            ("pix_fmt=nosuchfmt:video_size=1x1:time_base=1/1", "pixel format"),
            ("video_size=1x1:pix_fmt=gray:time_base=25/", "time_base"),
            ("video_size=320:pix_fmt=gray:time_base=1/1", "image size"),
            ("video_size=1x1:pix_fmt=gray:time_base=1/1:colorspace=ycgco", "colorspace"),
            ("video_size=1x1:pix_fmt=gray:time_base=1/1:range=widescreen", "range"),
        ] {
            let err = g.create_filter("buffer", args).unwrap_err();
            assert!(matches!(err, Error::InvalidArgument(_)), "{args}");
            assert!(
                err.to_string().contains("Unable to parse"),
                "{args} -> {err}"
            );
            assert!(err.to_string().contains(needle), "{args} -> {err}");
        }
    }

    /// The implicit promote-unspecified-to-mpeg list (buffersrc.c:508-512):
    /// ORDER matters for pick ([Unspecified, Mpeg]); a declared range is a
    /// singleton. NON-regular-yuv formats declare NO csp/rng halves at all.
    #[test]
    fn query_unspecified_range_promotion() {
        let (mut g, src, _sink, l) = buffer_graph("video_size=8x8:pix_fmt=yuv420p:time_base=1/25");
        with_imp(&mut g, src, |g, imp| imp.query_formats(g, src).unwrap());
        g.default_query_formats(src).unwrap(); // engine's follow-up fill
        let rng = g.links[l.0].incfg.color_ranges.unwrap();
        assert_eq!(
            g.rng_lists[rng as usize],
            vec![ColorRange::Unspecified, ColorRange::Mpeg]
        );
        // declared range → singleton [tv]
        let (mut g2, src2, _s2, l2) =
            buffer_graph("video_size=8x8:pix_fmt=yuv420p:time_base=1/25:range=tv");
        with_imp(&mut g2, src2, |g, imp| imp.query_formats(g, src2).unwrap());
        assert_eq!(
            g2.rng_lists[g2.links[l2.0].incfg.color_ranges.unwrap() as usize],
            vec![ColorRange::Mpeg]
        );
        // rgb24 / gray8: NOT regular yuv — the filter declares neither the
        // csp nor the rng half (the default fill owns them).
        for args in [
            "video_size=8x8:pix_fmt=rgb24:time_base=1/25",
            "video_size=8x8:pix_fmt=gray:time_base=1/25",
        ] {
            let (mut g3, src3, _s3, l3) = buffer_graph(args);
            with_imp(&mut g3, src3, |g, imp| imp.query_formats(g, src3).unwrap());
            assert!(g3.links[l3.0].incfg.color_spaces.is_none(), "{args}");
            assert!(g3.links[l3.0].incfg.color_ranges.is_none(), "{args}");
            // the format singleton is still declared
            assert_eq!(
                g3.fmt_lists[g3.links[l3.0].incfg.formats.unwrap() as usize].len(),
                1
            );
        }
    }

    /// CHECK_VIDEO_PARAM_CHANGE's hysteresis (buffersrc.c:74-96), asserted on
    /// the fields (not captured stderr). prev_* start zeroed (options path).
    #[test]
    fn param_change_hysteresis() {
        let (mut g, src, _sink, _l) =
            buffer_graph("video_size=320x240:pix_fmt=yuv420p:time_base=1/25");
        configure(&mut g, src);

        // frame #1 matches the declared params: link_delta false (congruent),
        // prev_delta true (prev_* are 0/0/None) → silent VERBOSE sync.
        let f1 = video_frame(Yuv420p, 320, 240, 0);
        with_state(&mut g, src, |s, g| s.check_video_param_change(g, src, &f1));
        with_state(&mut g, src, |s, _| {
            assert!(!s.link_delta);
            assert!(s.prev_delta);
            assert_eq!(s.prev_w, 320);
            assert_eq!(s.prev_h, 240);
            assert_eq!(s.prev_pix_fmt, Some(Yuv420p));
            assert_eq!(s.prev_color_space, ColorSpace::Unspecified);
        });

        // frame #2 differs from BOTH the context and the synced prev_*:
        // link_delta true AND prev_delta true — the WARNING case; prev_*
        // re-sync to frame #2 (CHECK_VIDEO_PARAM_CHANGE, buffersrc.c:74-96).
        let f2 = video_frame(Yuv420p, 640, 480, 40);
        let mut f2 = f2;
        f2.color_space = ColorSpace::Bt709;
        with_state(&mut g, src, |s, g| s.check_video_param_change(g, src, &f2));
        with_state(&mut g, src, |s, _| {
            assert!(s.link_delta);
            assert!(s.prev_delta);
            assert_eq!(s.prev_w, 640); // prev_* follow frame #2
        });

        // frame #3 = frame #2 again: still off the CONTEXT params
        // (link_delta true) but matches prev_* (prev_delta false) — the
        // DEBUG level, no re-sync.
        let f3 = video_frame(Yuv420p, 640, 480, 80);
        let mut f3 = f3;
        f3.color_space = ColorSpace::Bt709;
        with_state(&mut g, src, |s, g| s.check_video_param_change(g, src, &f3));
        with_state(&mut g, src, |s, _| {
            assert!(s.link_delta);
            assert!(!s.prev_delta);
            assert_eq!(s.prev_w, 640);
        });

        // frame #4 back to the declared geometry: link_delta false (the
        // VERBOSE "congruent" case) and prev_delta true (re-sync).
        let f4 = video_frame(Yuv420p, 320, 240, 120);
        with_state(&mut g, src, |s, g| s.check_video_param_change(g, src, &f4));
        with_state(&mut g, src, |s, _| {
            assert!(!s.link_delta);
            assert!(s.prev_delta);
            assert_eq!(s.prev_w, 320);
            assert_eq!(s.prev_color_space, ColorSpace::Unspecified);
            // the declared (link) params never move
            assert_eq!(s.w, 320);
            assert_eq!(s.h, 240);
        });
    }

    /// The link color fill (buffersrc.c:261-264): Unspecified fields inherit
    /// the negotiated link values; explicit values survive.
    #[test]
    fn color_fill_from_link() {
        let (mut g, src, _sink, l) = buffer_graph(
            "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:colorspace=bt709:range=tv",
        );
        configure(&mut g, src);
        // pick_format's link color selection (wave-2 graph.rs) simulated.
        g.links[l.0].colorspace = ColorSpace::Bt709;
        g.links[l.0].color_range = ColorRange::Mpeg;

        let f1 = video_frame(Yuv420p, 64, 48, 0);
        buffersrc_add_frame(&mut g, src, Some(&f1)).unwrap();
        let front = g.links[l.0].fifo.front().unwrap();
        assert_eq!(front.color_space, ColorSpace::Bt709);
        assert_eq!(front.color_range, ColorRange::Mpeg);

        let mut f2 = video_frame(Yuv420p, 64, 48, 40);
        f2.color_space = ColorSpace::Smpte170m;
        f2.color_range = ColorRange::Jpeg;
        buffersrc_add_frame(&mut g, src, Some(&f2)).unwrap();
        let back = g.links[l.0].fifo.back().unwrap();
        assert_eq!(back.color_space, ColorSpace::Smpte170m);
        assert_eq!(back.color_range, ColorRange::Jpeg);
    }

    /// KEEP_REF is an Arc share, never a copy (buffersrc.c:251-259).
    #[test]
    fn keep_ref_zero_copy() {
        let (mut g, src, _sink, l) = buffer_graph("video_size=16x16:pix_fmt=gray:time_base=1/25");
        configure(&mut g, src);
        let f = video_frame(Gray8, 16, 16, 0);
        f.plane(0); // readable
        buffersrc_add_frame(&mut g, src, Some(&f)).unwrap();
        let front = g.links[l.0].fifo.front().unwrap();
        assert!(std::sync::Arc::ptr_eq(&f.planes[0].buf, &front.planes[0].buf));
        assert_eq!(front.plane(0), f.plane(0));
    }

    /// last_pts tracking + the close path (buffersrc.c:223, 218-219,
    /// 291-298): EOF at last_pts, idempotent same-status re-close, and
    /// add-after-eof rejected with the counter reset FIRST (216 before
    /// 220-221).
    #[test]
    fn last_pts_and_close() {
        let (mut g, src, _sink, l) = buffer_graph("video_size=16x16:pix_fmt=gray:time_base=1/25");
        configure(&mut g, src);

        // Starve once so the counter is nonzero, then check the reset.
        filter::ff_request_frame(&mut g, l).unwrap();
        g.run_once().unwrap(); // activate stage 3
        assert_eq!(buffersrc_get_nb_failed_requests(&mut g, src), 1);

        let f = video_frame(Gray8, 16, 16, 100); // duration 40
        buffersrc_add_frame(&mut g, src, Some(&f)).unwrap();
        assert_eq!(buffersrc_get_nb_failed_requests(&mut g, src), 0); // reset
        assert_eq!(buffersrc_last_pts(&mut g, src), 140);

        // NULL frame == close at last_pts.
        buffersrc_add_frame(&mut g, src, None).unwrap();
        assert!(matches!(g.links[l.0].status_in, Some(Error::Eof)));
        assert_eq!(g.links[l.0].status_in_pts, 140);

        // Idempotent: a second close keeps the FIRST status_in_pts.
        buffersrc_close(&mut g, src, 9999).unwrap();
        assert_eq!(g.links[l.0].status_in_pts, 140);

        // add after eof: Err(EOF), and the counter was reset BEFORE the
        // rejection (buffersrc.c:216 runs before 220-221).
        let err = buffersrc_add_frame(&mut g, src, Some(&f)).unwrap_err();
        assert!(matches!(err, Error::Eof));
        assert_eq!(buffersrc_get_nb_failed_requests(&mut g, src), 0);
        assert_eq!(buffersrc_last_pts(&mut g, src), 140, "rejected add must not touch last_pts");
    }

    /// The BUFFERSRC_EMPTY shape (buffersrc.c:607-608): request → activate
    /// returns Ok (NOT Again), counter bumps, frame_wanted_out stays set
    /// (C leaves it; a later add_frame clears it via filter_frame).
    #[test]
    fn nb_failed_requests_and_empty() {
        let (mut g, src, _sink, l) = buffer_graph("video_size=16x16:pix_fmt=gray:time_base=1/25");
        configure(&mut g, src);
        filter::ff_request_frame(&mut g, l).unwrap();
        assert!(g.links[l.0].frame_wanted_out);
        assert_eq!(g.nodes[src.0].ready, 100);

        g.run_once().unwrap(); // activate: BUFFERSRC_EMPTY → Ok(())
        assert_eq!(buffersrc_get_nb_failed_requests(&mut g, src), 1);
        assert!(g.links[l.0].frame_wanted_out, "C leaves the request standing");
        assert!(matches!(g.run_once(), Err(Error::Again)));

        // Feeding the source resets the counter and clears the want.
        let f = video_frame(Gray8, 16, 16, 0);
        buffersrc_add_frame(&mut g, src, Some(&f)).unwrap();
        assert_eq!(buffersrc_get_nb_failed_requests(&mut g, src), 0);
        assert!(!g.links[l.0].frame_wanted_out);
    }

    /// The downstream-close latch (buffersrc.c:598-606): a hard downstream
    /// close makes the NEXT activation latch eof and return IMMEDIATELY (no
    /// out-status push); the activation after THAT pushes the out status
    /// (same-status no-op — no panic); get_status reflects and latches Eof.
    #[test]
    fn activate_downstream_close_latch() {
        let (mut g, src, _sink, l) = buffer_graph("video_size=16x16:pix_fmt=gray:time_base=1/25");
        configure(&mut g, src);
        filter::inlink_set_status(&mut g, l, Error::Eof); // hard downstream close
        assert!(g.links[l.0].status_in.is_some());

        // Stage 1: latch + immediate return; the counter is NOT bumped.
        g.run_once().unwrap();
        assert_eq!(buffersrc_get_nb_failed_requests(&mut g, src), 0);

        // Stage 2 on a later activation: outlink_set_status with the same
        // Eof → same-status no-op inside set_in_status, no panic.
        filter::set_ready(&mut g, src, 100);
        g.run_once().unwrap();
        assert!(matches!(g.links[l.0].status_in, Some(Error::Eof)));

        // get_status latches and reports Eof (buffersrc.c:300-308).
        assert!(matches!(buffersrc_get_status(&mut g, src), Err(Error::Eof)));
        // And a post-eof add is rejected (220-221).
        let f = video_frame(Gray8, 16, 16, 0);
        assert!(matches!(
            buffersrc_add_frame(&mut g, src, Some(&f)),
            Err(Error::Eof)
        ));
    }

    /// The queued-buffers watchdog (buffersrc.c:278-286): fires when the
    /// source's OWN output fifo reaches the LIMIT; the limit then ×10.
    #[test]
    fn watchdog() {
        let (mut g, src, _sink, l) = buffer_graph("video_size=8x8:pix_fmt=gray:time_base=1/25");
        configure(&mut g, src);
        for i in 0..99i64 {
            buffersrc_add_frame(&mut g, src, Some(&video_frame(Gray8, 8, 8, i))).unwrap();
        }
        with_state(&mut g, src, |s, _| assert_eq!(s.warning_limit, 100));
        // The 100th queued frame hits the limit exactly.
        buffersrc_add_frame(&mut g, src, Some(&video_frame(Gray8, 8, 8, 99))).unwrap();
        assert_eq!(g.links[l.0].fifo.len(), 100);
        with_state(&mut g, src, |s, _| assert_eq!(s.warning_limit, 1000));
    }

    /// End to end through the real runtime API: frames traverse, EOF at
    /// last_pts propagates, the graph quiesces (the wave-1 engine test
    /// rewritten against buffersrc).
    #[test]
    fn eof_end_to_end_runtime() {
        let mut g = FilterGraph::new();
        let src = g
            .create_filter("buffer", "video_size=32x24:pix_fmt=gray:time_base=1/25")
            .unwrap();
        let n = g.create_filter("null", "").unwrap();
        let sink = g.alloc_test_sink();
        let la = g.link(src, 0, n, 0).unwrap();
        let lb = g.link(n, 0, sink, 0).unwrap();
        g.config().unwrap();
        configure(&mut g, src);
        // graph_config_links' defaults pass (wave 2) would inherit w/h down
        // the chain; simulated for the middle link.
        g.links[lb.0].w = 32;
        g.links[lb.0].h = 24;

        buffersrc_add_frame(&mut g, src, Some(&video_frame(Gray8, 32, 24, 0))).unwrap();
        buffersrc_add_frame(&mut g, src, Some(&video_frame(Gray8, 32, 24, 40))).unwrap();
        run_to_quiescence(&mut g);
        for l in [la, lb] {
            assert_eq!(g.links[l.0].frame_count_in, 2);
            assert_eq!(g.links[l.0].frame_count_out, 2);
        }

        // EOF at last_pts = 40 + 40 = 80.
        buffersrc_add_frame(&mut g, src, None).unwrap();
        assert_eq!(g.links[la.0].status_in_pts, 80);
        run_to_quiescence(&mut g);
        assert!(
            matches!(g.links[lb.0].status_in, Some(Error::Eof)),
            "EOF must reach the sink-side link"
        );
        assert!(g.links[la.0].fifo.is_empty());
        assert!(g.links[lb.0].fifo.is_empty());
        assert!(matches!(g.run_once(), Err(Error::Again)));
    }

    // ---- ts2timestr (timestamp.c:21-36) ------------------------------------

    /// Expected values verified by compiling the exact C function: both trim
    /// loops strip the trailing '.' whenever the digits before it keep `last`
    /// above 0, so whole seconds print bare ("4", "10", "0").
    #[test]
    fn ts2timestr_matches_c() {
        assert_eq!(ts2timestr(NOPTS, Rational::new(1, 25)), "NOPTS");
        assert_eq!(ts2timestr(100, Rational::new(1, 25)), "4"); // 4.000000 → "4"
        assert_eq!(ts2timestr(0, Rational::new(1, 25)), "0"); // 0.000000 → "0"
        assert_eq!(ts2timestr(500, Rational::new(1, 1000)), "0.5");
        assert_eq!(ts2timestr(250, Rational::new(1, 25)), "10");
        assert_eq!(ts2timestr(-3200, Rational::new(1, 25)), "-128");
        // precision = -log10(val) + 5 for |val| < 1 (timestamp.c:28)
        assert_eq!(ts2timestr(1, Rational::new(1, 90000)), "0.0000111111");
    }

    // ---- parse helpers ------------------------------------------------------

    #[test]
    fn parse_image_size_forms() {
        assert_eq!(parse_image_size("n", "320x240").unwrap(), (320, 240));
        assert_eq!(parse_image_size("n", "320X240").unwrap(), (320, 240)); // any one separator
        assert_eq!(parse_image_size("n", "none").unwrap(), (0, 0));
        // "320": no height → EINVAL (parseutils.c width/height > 0 check)
        assert!(parse_image_size("n", "320").is_err());
        assert!(parse_image_size("n", "320x").is_err());
        assert!(parse_image_size("n", "0x240").is_err());
        // "123x345foobar": trailing extraneous data
        assert!(parse_image_size("n", "320x240junk").is_err());
        assert!(parse_image_size("n", "x240").is_err());
    }

    #[test]
    fn parse_rational_forms() {
        assert_eq!(
            parse_rational("n", "k", "25").unwrap(),
            Rational::new(25, 1)
        );
        assert_eq!(
            parse_rational("n", "k", "-1/25").unwrap(),
            Rational::new(-1, 25)
        );
        // "1/0" is ACCEPTED (C sscanf takes it; init's av_q2d check passes)
        assert_eq!(parse_rational("n", "k", "1/0").unwrap(), Rational::new(1, 0));
        assert!(parse_rational("n", "k", "25/").is_err());
        assert!(parse_rational("n", "k", "abc").is_err());
        assert!(parse_rational("n", "k", "").is_err());
    }

    #[test]
    fn csp_and_range_names() {
        assert_eq!(parse_csp_name("n", "gbr").unwrap(), ColorSpace::Rgb);
        assert_eq!(parse_csp_name("n", "bt709").unwrap(), ColorSpace::Bt709);
        assert_eq!(
            parse_csp_name("n", "unknown").unwrap(),
            ColorSpace::Unspecified
        );
        assert_eq!(
            parse_csp_name("n", "bt2020nc").unwrap(),
            ColorSpace::Bt2020Ncl
        );
        // In the enum but not an option name... fcc IS one:
        assert_eq!(parse_csp_name("n", "fcc").unwrap(), ColorSpace::Fcc);
        // C-only names (ycgco family etc.) are absent from our enum:
        assert!(parse_csp_name("n", "ycgco").is_err());
        assert!(parse_csp_name("n", "ictcp").is_err());

        assert_eq!(
            parse_range_name("n", "unknown").unwrap(),
            ColorRange::Unspecified
        );
        assert_eq!(parse_range_name("n", "tv").unwrap(), ColorRange::Mpeg);
        assert_eq!(parse_range_name("n", "mpeg").unwrap(), ColorRange::Mpeg);
        assert_eq!(parse_range_name("n", "full").unwrap(), ColorRange::Jpeg);
        assert_eq!(parse_range_name("n", "jpeg").unwrap(), ColorRange::Jpeg);
        assert!(parse_range_name("n", "widescreen").is_err());

        // The log-side name tables are the inverses (pixdesc.c tables).
        assert_eq!(color_space_name(ColorSpace::Rgb), "gbr");
        assert_eq!(color_space_name(ColorSpace::Unspecified), "unknown");
        assert_eq!(color_space_name(ColorSpace::Bt2020Ncl), "bt2020nc");
        assert_eq!(color_range_name(ColorRange::Unspecified), "unknown");
        assert_eq!(color_range_name(ColorRange::Mpeg), "tv");
        assert_eq!(color_range_name(ColorRange::Jpeg), "pc");
    }

    /// The runtime API rejects non-buffer nodes.
    #[test]
    fn add_frame_rejects_non_buffer_node() {
        let mut g = FilterGraph::new();
        let n = g.create_filter("null", "").unwrap();
        let f = Frame::default();
        let err = buffersrc_add_frame(&mut g, n, Some(&f)).unwrap_err();
        assert!(err.to_string().contains("is not a buffer source"));
    }
}
