//! `scale` video filter — port of `libavfilter/vf_scale.c` (1270 lines) plus
//! `ff_scale_adjust_dimensions` from `libavfilter/scale_eval.c` (123-201).
//!
//! "Scale the input video size and/or convert the image format."
//! (vf_scale.c:1191)
//!
//! All pixel work is DELEGATED to [`crate::swscale::ScaleContext`] (imported
//! here as `SwsScaler` — both C structs are named `ScaleContext`): this
//! module is the option/geometry/negotiation plumbing around it. The C
//! filter's shape, kept in C order:
//!
//! * `init` (vf_scale.c:328-439, preinit 310-324 folded in) — option intake
//!   (w/h/size/flags/interl/color overrides), the w-only swap quirk
//!   (342-343), the `av_parse_video_size` port (parseutils.c:150-179), the
//!   expression parse + `check_exprs` (184-255), the color-matrix
//!   validation (394-404) and the flags→algorithm selection (409-413, where
//!   C sets `sws_flags` on the SwsContext).
//! * `query_formats` (451-530) — ONE-SIDED lists: the input pad offers the
//!   swscale-supported input formats (`sws_test_format(fmt, 0)`, 463-471),
//!   the output pad the supported outputs (`sws_test_format(fmt, 1)`, PAL8
//!   dropped, 473-483), the color lists all-lists except singletons where
//!   the user overrode the output side (502-521).
//! * `config_props` on the OUTPUT pad (621-714) — evaluate the w/h
//!   expressions against the input link (`scale_eval_dimensions`, the
//!   filter's OWN 532-619 version, not the shared `ff_scale_eval_dimensions`
//!   of scale_eval.c:58-121), then [`scale_adjust_dimensions`]
//!   (scale_eval.c:123-201: the `-1`/`-N` factor semantics,
//!   `force_original_aspect_ratio`, `force_divisible_by`, `reset_sar`'s
//!   `w_adj`), then the output SAR (658-664) and the verbose dump
//!   (666-674). A Rust ADDITION at the end probes the conversion pair with
//!   the Scaler so an unsupported pair fails graph CONFIG, not the first
//!   frame.
//! * `filter_frame` (961-973) → `scale_frame` (744-896) — the frame-changed
//!   reconfigure dance (758-812), the output-frame assembly with input tag
//!   overrides + `copy_props` (818-861), the output SAR `av_reduce`
//!   (863-870), the `sws_is_noop` pass-through (872-877, format.c:693-705 +
//!   `sanitize_fmt` 305-335), then `sws_scale_frame` (884) as the Scaler
//!   call, lazily created per conversion key (C creates the SwsContext
//!   inside `sws_scale_frame` on property change).
//!
//! ## Not ported (documented degradations, each also noted at its site)
//!
//! * **scale2ref and the dynamic `ref` pad** — the whole second filter
//!   (vf_scale.c:1205-1270), `config_props_ref` (716-731),
//!   `filter_frame_ref` (975-1008), the ref-variable scans of `check_exprs`
//!   (212-242) and the dynamic in-pad append (428-436). No `FFFrameSync`
//!   exists in the port. Consequences: `rw`/`rh`/`ref_*`/`main_*` are
//!   unknown identifiers — rejected at parse with C's `Cannot parse
//!   expression for ...` text; `scale2ref` itself is not a registered
//!   filter. Use `scale=w=...:h=...`.
//! * **framesync / `activate`** (1035-1039, 687-708, 898-959) — the single
//!   input degenerates to the engine's [`super::filter::default_activate`]
//!   (push frames through `filter_frame`; EOF forwarded by
//!   `forward_status_change`), which is behaviorally identical for a
//!   1-input/1-output EXT_STOP filter. `do_scale`'s pts rescale
//!   (`av_rescale_q_rnd(fs->pts, ...)`, 953) is thereby dropped: pts passes
//!   through unchanged. Divergence: in C the pts is rescaled between link
//!   time bases; in the port link time bases inherit unchanged down the
//!   chain (graph.rs video defaults), making the rescale an identity in
//!   every single-input graph the port can build.
//! * **`process_command`** (1010-1033) — runtime w/h updates (and
//!   `scale_parse_expr`'s save/revert machinery, 257-308) have no command
//!   queue in the port. The one retained re-parse is the init-mode freeze
//!   inside `scale_frame` (780-793).
//! * **threads/slices** — preinit's `sws->threads` plumbing (310-324,
//!   424-426) and `slice_y`; single-threaded port.
//! * **primaries/transfer options** (`in_/out_primaries`, `in_/out_transfer`,
//!   validation 370-392, side-data removal 682-685/858-861) and
//!   **`param0`/`param1`**, **`in_/out_{h,v}_chr_pos`** (415-422) — the
//!   ported option table does not carry them; the entries stay unconsumed
//!   and init fails with `No such option: in_primaries` etc. (C's own
//!   unknown-option error shape). `Frame`'s primaries/trc ride `copy_props`.
//! * **interlaced field scaling** — C's `interl` drives per-field processing
//!   (`ff_fmt_from_frame` halves the height per field, format.c:384-388);
//!   the port's Scaler resamples whole frames. `interl` still toggles the
//!   input frame's INTERLACED flag before scaling (835-838) and restores
//!   the original flags on the output (886) — net metadata no-op.
//! * **PAL8** output special case (477, 879-882) — no pal8 in
//!   [`crate::util::pixfmt::PixelFormat`].
//! * **`av_parse_video_size`'s abbreviation table** (ntsc/pal/vga/...,
//!   parseutils.c:125-138) — those strings now fail as invalid sizes.
//! * **`ff_scale_eval_dimensions`** (scale_eval.c:58-121, the SHARED twin
//!   used by other filters) is not separately ported; vf_scale's own
//!   3-eval dance is (see [`ScaleContext::scale_eval_dimensions`] for the
//!   differences — first pass without NaN check, no INT32 range checks).
//!
//! ## The expression evaluator
//!
//! The crate has no port of `libavutil/eval.c`; [`mod@expr`] is a NEW subset
//! written against the actual grammar (eval.c:560-690): decimal numbers,
//! the 13 retained variables, `+ - * /` with C's precedence, `^` binding
//! TIGHTER than unary sign and LEFT-folded (so `-2^2` = -4, `2^-3^2` =
//! `(2^-3)^2`, exactly eval.c:587-611's sign-multiplier trick), parentheses,
//! and the functions `min max mod floor ceil trunc round abs clip`. There
//! is NO ternary/comparison/`&&`/`||`/`%` OPERATOR syntax in av_expr (those
//! are the `if`/`eq`/`lt`/`mod`... FUNCTIONS, eval.c:483-517 — only the
//! nine listed are retained); division by zero follows eval.c:348
//! (`d2 ? d/d2 : d*INFINITY`: 5/0 = +inf, -5/0 = -inf, 0/0 = NaN) and
//! `mod` is eval.c:337's floored modulo `d - floor(d/d2)*d2`. Expressions
//! using `sin`/`pow`/`sqrt`/`not`/`st`/`ld`/`random`/`while`/`';'`/hex/
//! `dB` or the scale2ref variables fail at parse with C's `Cannot parse
//! expression for width/height: '...'` text — an honest degradation.

pub(crate) mod expr;

use crate::{
    NOPTS, log_error, log_verbose, log_warning,
    swscale::{self, ScaleAlgorithm, ScaleContext as SwsScaler, ScaleEngine, ScaleOptions},
    util::{
        color::{ChromaLocation, ColorRange, ColorSpace},
        error::{Error, Result},
        frame::{Frame, FrameFlags},
        mathematics,
        pixdesc::{self, PixFmtFlags},
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::{
    filter::{self, FilterDef, FilterFlags, FilterImpl, PadDef, PadRef},
    formats,
    graph::FilterGraph,
    link::NodeId,
};

// ---------------------------------------------------------------------------
// Option value parsers (opt.c / parseutils.c subsets)
// ---------------------------------------------------------------------------

/// C `strtol(s, &p, 10)` core — copied locally from buffersrc.rs (private
/// there): optional leading whitespace and sign, then decimal digits.
/// Returns `(value, bytes_consumed)`; consumed == 0 when no digits were
/// found. Overflow saturates (C clamps to LONG_MAX; values that large fail
/// the range checks anyway).
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
    (n.clamp(i32::MIN as i64, i32::MAX as i64) as i32, i)
}

/// The buffersrc-style generic parse failure (opt.c:500's
/// `Unable to parse "%s" option value "%s"`): one log line plus
/// `Error::InvalidArgument` carrying the same text.
fn option_parse_failed(name: &str, key: &str, val: &str) -> Error {
    log_error!(
        Some(name),
        "Unable to parse \"{key}\" option value \"{val}\"\n"
    );
    Error::InvalidArgument(format!("Unable to parse \"{key}\" option value \"{val}\""))
}

/// `set_string_bool` (opt.c:226-254): `auto` → -1, the true/false name
/// families, else a fully-consumed decimal strtol; then the min/max range
/// check with C's error text shape.
fn parse_bool_val(name: &str, key: &str, val: &str, min: i32, max: i32) -> Result<i32> {
    let n = match val {
        "auto" => -1,
        "true" | "y" | "yes" | "enable" | "enabled" | "on" => 1,
        "false" | "n" | "no" | "disable" | "disabled" | "off" => 0,
        v => {
            let (n, consumed) = strtol_i32(v);
            if consumed != v.len() {
                return Err(bool_parse_failed(name, key, val));
            }
            n
        }
    };
    if n < min || n > max {
        return Err(bool_parse_failed(name, key, val));
    }
    Ok(n)
}

/// set_string_bool's failure text (opt.c:252-253).
fn bool_parse_failed(name: &str, key: &str, val: &str) -> Error {
    log_error!(
        Some(name),
        "Unable to parse \"{key}\" option value \"{val}\" as boolean\n"
    );
    Error::InvalidArgument(format!(
        "Unable to parse \"{key}\" option value \"{val}\" as boolean"
    ))
}

/// A fully-consumed decimal i64 (`set_string_number`'s strtol arm), or None.
fn strtol_i64_full(val: &str) -> Option<i64> {
    let (n, consumed) = strtol_i32(val);
    if consumed == val.len() && !val.is_empty() {
        Some(n as i64)
    } else {
        None
    }
}

/// The `in_color_matrix`/`out_color_matrix` CONST names (vf_scale.c:1080-1089
/// — the shared "color" unit): `auto` maps to None ("no override", a
/// deliberate smoothing of C's -1-vs-2 sentinel split: in_ defaults to -1
/// with `auto` = -1, out_ defaults to UNSPECIFIED with min 0 — where C's
/// `auto` (-1) actually trips the option range check; the port treats both
/// sides uniformly).
fn parse_color_matrix(name: &str, key: &str, val: &str) -> Result<Option<ColorSpace>> {
    Ok(match val {
        "auto" => return Ok(None),
        // BT.601: four aliases all map to AVCOL_SPC_BT470BG in C.
        "bt601" | "bt470" | "smpte170m" | "bt470bg" => Some(ColorSpace::Bt470bg),
        "bt709" => Some(ColorSpace::Bt709),
        "fcc" => Some(ColorSpace::Fcc),
        "smpte240m" => Some(ColorSpace::Smpte240m),
        "bt2020" | "bt2020nc" => Some(ColorSpace::Bt2020Ncl),
        _ => return Err(option_parse_failed(name, key, val)),
    })
}

/// The `in_range`/`out_range` CONST names (vf_scale.c:1092-1099): `auto` /
/// `unknown` mean "no override" (C's UNSPECIFIED sentinel).
fn parse_range_opt(name: &str, key: &str, val: &str) -> Result<Option<ColorRange>> {
    Ok(match val {
        "auto" | "unknown" => None,
        "full" | "jpeg" | "pc" => Some(ColorRange::Jpeg),
        "limited" | "mpeg" | "tv" => Some(ColorRange::Mpeg),
        _ => return Err(option_parse_failed(name, key, val)),
    })
}

/// The `in_chroma_loc`/`out_chroma_loc` CONST names (vf_scale.c:1102-1109).
/// Unlike the matrix/range options there is no "absent" state: the default
/// Unspecified IS "auto".
fn parse_chroma_loc(name: &str, key: &str, val: &str) -> Result<ChromaLocation> {
    Ok(match val {
        "auto" | "unknown" => ChromaLocation::Unspecified,
        "left" => ChromaLocation::Left,
        "center" => ChromaLocation::Center,
        "topleft" => ChromaLocation::TopLeft,
        "top" => ChromaLocation::Top,
        "bottomleft" => ChromaLocation::BottomLeft,
        "bottom" => ChromaLocation::Bottom,
        _ => return Err(option_parse_failed(name, key, val)),
    })
}

/// `force_original_aspect_ratio` (vf_scale.c:1150-1153): CONST names, then
/// the numeric fallback of `set_string_number` (0..=2).
fn parse_foar(name: &str, key: &str, val: &str) -> Result<ForceOriginalAspectRatio> {
    let n = match val {
        "disable" => 0,
        "decrease" => 1,
        "increase" => 2,
        _ => {
            let Some(n) = strtol_i64_full(val) else {
                return Err(option_parse_failed(name, key, val));
            };
            if !(0..=2).contains(&n) {
                return Err(Error::InvalidArgument(format!(
                    "Value {n} for parameter {key} out of range"
                )));
            }
            n
        }
    };
    Ok(match n {
        0 => ForceOriginalAspectRatio::Disable,
        1 => ForceOriginalAspectRatio::Decrease,
        _ => ForceOriginalAspectRatio::Increase,
    })
}

/// `force_divisible_by` (vf_scale.c:1154): plain INT, validated 1..=256;
/// out-of-range reuses the write_number range text shape (opt.c:283).
fn parse_divisible_by(name: &str, key: &str, val: &str) -> Result<i64> {
    let Some(n) = strtol_i64_full(val) else {
        return Err(option_parse_failed(name, key, val));
    };
    if !(1..=256).contains(&n) {
        return Err(Error::InvalidArgument(format!(
            "Value {n} for parameter {key} out of range"
        )));
    }
    Ok(n)
}

/// `eval` (vf_scale.c:1158-1160): `init`/`frame`, or the numeric fallback.
fn parse_eval_mode(name: &str, key: &str, val: &str) -> Result<EvalMode> {
    match val {
        "init" => Ok(EvalMode::Init),
        "frame" => Ok(EvalMode::Frame),
        _ => match strtol_i64_full(val) {
            Some(0) => Ok(EvalMode::Init),
            Some(1) => Ok(EvalMode::Frame),
            _ => Err(option_parse_failed(name, key, val)),
        },
    }
}

/// `av_parse_video_size` (parseutils.c:150-179) minus the named-size
/// abbreviation table (ntsc/pal/vga/... — those strings now fail): strtol
/// width, skip exactly ONE separator byte, strtol height; trailing data or
/// non-positive values → EINVAL. The caller (init) logs the ONE error line,
/// matching C where init logs and this function is silent.
fn parse_video_size(s: &str) -> Result<(i32, i32)> {
    let (w, mut p) = strtol_i32(s);
    if p < s.len() {
        p += 1; // `if (*p) p++` — the single separator (usually 'x')
    }
    let tail = s.get(p..).unwrap_or("");
    let (h, consumed2) = strtol_i32(tail);
    // "trailing extraneous data detected, like in 123x345foobar"
    if consumed2 != tail.len() || w <= 0 || h <= 0 {
        return Err(Error::InvalidArgument(format!("Invalid size '{s}'")));
    }
    Ok((w, h))
}

// ---------------------------------------------------------------------------
// check_exprs / scale_parse_expr (vf_scale.c:184-308)
// ---------------------------------------------------------------------------

/// `check_exprs` (vf_scale.c:184-255), ref/scale2ref arms dropped (those
/// identifiers fail at parse — error 3 — instead of being counted here):
///
/// 1. w uses `ow`/`out_w` → error (197-200);
/// 2. h uses `oh`/`out_h` → error (202-205);
/// 3. w uses `oh` AND h uses `ow` → WARNING only (207-210);
/// 4. eval_mode == Init && either uses `n`/`t` → error (244-252; C also
///    counted the scale2ref N/T/POS which collapse to n/t here).
fn check_exprs(
    w: &expr::Expr,
    h: &expr::Expr,
    w_expr: &str,
    h_expr: &str,
    eval_mode: EvalMode,
    name: &str,
) -> Result<()> {
    if expr::uses(w, "out_w") {
        let msg = format!("Width expression cannot be self-referencing: '{w_expr}'.");
        log_error!(Some(name), "{msg}\n");
        return Err(Error::InvalidArgument(msg));
    }
    if expr::uses(h, "out_h") {
        let msg = format!("Height expression cannot be self-referencing: '{h_expr}'.");
        log_error!(Some(name), "{msg}\n");
        return Err(Error::InvalidArgument(msg));
    }
    if expr::uses(w, "out_h") && expr::uses(h, "out_w") {
        log_warning!(
            Some(name),
            "Circular references detected for width '{w_expr}' and height '{h_expr}' - possibly invalid.\n"
        );
    }
    if eval_mode == EvalMode::Init
        && (expr::uses(w, "n") || expr::uses(w, "t") || expr::uses(h, "n") || expr::uses(h, "t"))
    {
        let msg =
            "Expressions with frame variables 'n', 't', 'pos' are not valid in init eval_mode.";
        log_error!(Some(name), "{msg}\n");
        return Err(Error::InvalidArgument(msg.to_string()));
    }
    Ok(())
}

/// `scale_parse_expr`'s retained body (vf_scale.c:257-308): parse-or-error
/// with C's text. The old-str/old-pexpr save-revert dance existed only for
/// the `process_command` runtime path (dropped) and the frame_changed
/// re-parse (whose failure propagates as an error anyway) — a plain
/// parse-or-error is behaviorally identical for both retained call sites.
/// (`check_exprs` runs separately at the call sites, mirroring C's call
/// inside `scale_parse_expr`.)
fn parse_scale_expr(expr_str: &str, which: &str, name: &str) -> Result<expr::Expr> {
    match expr::parse(expr_str) {
        Ok(e) => Ok(e),
        Err(_) => {
            let msg = format!("Cannot parse expression for {which}: '{expr_str}'");
            log_error!(Some(name), "{msg}\n");
            Err(Error::InvalidArgument(msg))
        }
    }
}

// ---------------------------------------------------------------------------
// ff_scale_adjust_dimensions (scale_eval.c:123-201)
// ---------------------------------------------------------------------------

/// `SCALE_FORCE_OAR_*` (scale_eval.h:29-33).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForceOriginalAspectRatio {
    #[default]
    Disable,
    Decrease,
    Increase,
}

/// `ff_scale_adjust_dimensions` (scale_eval.c:123-201) — exact port in i64,
/// `av_rescale` = [`mathematics::rescale`] (NEAR_INF). The caller passes the
/// INPUT link geometry; `*w`/`*h` carry the evaluated dimensions where
/// `-1` means "keep aspect" and `-N` "keep aspect, divisible by N"
/// (128-143); `w_adj` is the reset_sar width adjustment. Non-positive
/// results error with C's log line (190-195); a value unrepresentable as
/// i32 errors with a port-chosen message (C returns bare `AVERROR(EINVAL)`
/// at 187-188 — the port reuses the neighboring config_props log text).
pub(crate) fn scale_adjust_dimensions(
    in_w: i32,
    in_h: i32,
    w: &mut i64,
    h: &mut i64,
    foar: ForceOriginalAspectRatio,
    force_divisible_by: i64,
    w_adj: f64,
) -> Result<()> {
    // C's double→int64 argument conversions truncate toward zero at the
    // av_rescale call sites (NaN is UB there; mapped to 0 — unreachable
    // without a NaN SAR).
    fn d64(x: f64) -> i64 {
        if x.is_nan() { 0 } else { x as i64 }
    }

    let factor_w = if *w < -1 { -*w } else { 1 };
    let factor_h = if *h < -1 { -*h } else { 1 };

    if *w < 0 && *h < 0 {
        *w = d64(in_w as f64 * w_adj);
        *h = in_h as i64;
    }

    if *w < 0 {
        *w = mathematics::rescale(*h, d64(in_w as f64 * w_adj), in_h as i64 * factor_w) * factor_w;
    }
    if *h < 0 {
        // The third argument is the DOUBLE product (in_w * w_adj) * factor_h,
        // truncated once to i64 at the call (scale_eval.c:156).
        *h = mathematics::rescale(
            *w,
            in_h as i64,
            d64((in_w as f64 * w_adj) * factor_h as f64),
        ) * factor_h;
    }

    if foar != ForceOriginalAspectRatio::Disable {
        let tmp_w = mathematics::rescale(
            *h,
            d64(in_w as f64 * w_adj),
            in_h as i64 * force_divisible_by,
        ) * force_divisible_by;
        let tmp_h = mathematics::rescale(
            *w,
            in_h as i64,
            d64((in_w as f64 * w_adj) * force_divisible_by as f64),
        ) * force_divisible_by;

        if foar == ForceOriginalAspectRatio::Decrease {
            *w = tmp_w.min(*w);
            *h = tmp_h.min(*h);
            if force_divisible_by > 1 {
                // Round DOWN (positive values: i64 division is floor).
                *w = *w / force_divisible_by * force_divisible_by;
                *h = *h / force_divisible_by * force_divisible_by;
            }
        } else {
            *w = tmp_w.max(*w);
            *h = tmp_h.max(*h);
            if force_divisible_by > 1 {
                // Round UP.
                *w = (*w + force_divisible_by - 1) / force_divisible_by * force_divisible_by;
                *h = (*h + force_divisible_by - 1) / force_divisible_by * force_divisible_by;
            }
        }
    }

    if *w < i32::MIN as i64 || *w > i32::MAX as i64 || *h < i32::MIN as i64 || *h > i32::MAX as i64
    {
        return Err(Error::InvalidArgument(
            "Rescaled value for width or height is too big".into(),
        ));
    }

    if *w <= 0 || *h <= 0 {
        log_error!(
            None,
            "Rescaled dimensions {}x{} are invalid, output dimensions must be positive.\n",
            *w,
            *h
        );
        return Err(Error::InvalidArgument(format!(
            "Rescaled dimensions {}x{} are invalid, output dimensions must be positive",
            *w, *h
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Name tables (pixdesc.c:3276-3280, 3330-3349 — local copies; buffersrc.rs
// keeps private twins, do not share)
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

/// `sws_test_colorspace` (format.c:627-641): every value in the switch
/// passes; over the port's enum the only failing variant is `Reserved`.
fn sws_test_colorspace(csp: ColorSpace) -> bool {
    !matches!(csp, ColorSpace::Reserved)
}

/// The conversion-gate matrix classes: which swscale conversion table a
/// color space selects. BT470BG and SMPTE170M share the BT.601 coefficients,
/// so an Unspecified input resolves to the same class as an explicit
/// bt470bg tag (no gate error).
fn matrix_class(csp: ColorSpace) -> &'static str {
    match csp {
        ColorSpace::Unspecified | ColorSpace::Bt470bg | ColorSpace::Smpte170m => "bt601",
        ColorSpace::Bt709 => "bt709",
        ColorSpace::Fcc => "fcc",
        ColorSpace::Smpte240m => "smpte240m",
        ColorSpace::Bt2020Ncl => "bt2020nc",
        ColorSpace::Rgb => "gbr",
        ColorSpace::Reserved => "reserved",
    }
}

// ---------------------------------------------------------------------------
// Private context (vf_scale.c:127-177)
// ---------------------------------------------------------------------------

/// `enum EvalMode` (vf_scale.c:121-125; `EVAL_MODE_NB` dropped).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvalMode {
    /// `EVAL_MODE_INIT` — expressions evaluated once, frozen afterwards.
    #[default]
    Init,
    /// `EVAL_MODE_FRAME` — re-evaluated per frame (variables `n`/`t` live).
    Frame,
}

/// `ScaleContext` (vf_scale.c:127-177), reduced to the ported subset. C
/// names stay. Fields dropped: `sws`/`fs` handles (the Scaler is built
/// lazily; framesync does not exist), `param[2]`, `hsub`/`vsub`/`slice_y`
/// (unused in this C version), `uses_ref` (no ref pad), the `var_values`
/// array (replaced by [`expr::Vars`] plus the persistent `var_n`/`var_t`/
/// `var_sar` slots), `in_/out_primaries`, `in_/out_transfer`,
/// `in_/out_{h,v}_chr_pos` (options not ported).
pub struct ScaleContext {
    /// `w_expr`/`h_expr` — the expression STRINGS ("" = unset pre-init; the
    /// defaults "iw"/"ih" are filled by init, 357-360).
    w_expr: String,
    h_expr: String,
    /// `w_pexpr`/`h_pexpr` — the parsed ASTs.
    w_pexpr: expr::Expr,
    h_pexpr: expr::Expr,
    /// `size_str` (None = C's NULL).
    size_str: Option<String>,
    /// `flags_str` — the scaler-algorithm name (None = "" = use the graph
    /// default).
    flags_str: Option<String>,
    /// `interlaced` (vf_scale.c:1075): BOOL with default 0, range -1..=1
    /// (0 = force progressive for the conversion, -1 = leave, 1 = force
    /// interlaced; metadata-only in this port — see module doc).
    interlaced: i32,
    /// `in_color_matrix`/`out_color_matrix` (None = C's -1 / UNSPECIFIED).
    in_color_matrix: Option<ColorSpace>,
    out_color_matrix: Option<ColorSpace>,
    /// `in_range`/`out_range` (None = UNSPECIFIED = no override).
    in_range: Option<ColorRange>,
    out_range: Option<ColorRange>,
    /// `in_chroma_loc`/`out_chroma_loc` — NOTE the C asymmetry: the input
    /// tag is stamped UNCONDITIONALLY (vf_scale.c:832), the output tag only
    /// when != UNSPECIFIED (846-847).
    in_chroma_loc: ChromaLocation,
    out_chroma_loc: ChromaLocation,
    force_original_aspect_ratio: ForceOriginalAspectRatio,
    force_divisible_by: i64,
    reset_sar: bool,
    eval_mode: EvalMode,
    /// `w`/`h` — the last evaluated dimensions (may be negative: -1/-N).
    w: i32,
    h: i32,
    /// Persistent `var_values` slots only the frame path writes:
    /// `var_values[VAR_SAR]` (cached by scale_eval_dimensions for
    /// reset_sar's w_adj) and `VAR_N`/`VAR_T` (799-801).
    var_sar: f64,
    var_n: f64,
    var_t: f64,
    /// The `av_expr_count_vars` results for the frame-mode fast path
    /// (772-778): does each expression use `n`/`t`?
    w_expr_uses_n: bool,
    w_expr_uses_t: bool,
    h_expr_uses_n: bool,
    h_expr_uses_t: bool,
    /// The resolved kernel (C's `sws_flags` on the SwsContext); when the
    /// user gave no `flags` option init copies `g.scale_algorithm` — how the
    /// CLI's engine/algorithm selection flows in (C: preinit's threads
    /// plumbing, 424-426, is the analogous graph-level default).
    algorithm: ScaleAlgorithm,
    /// The lazily created Scaler + the conversion key it was built for
    /// (C re-creates the SwsContext inside `sws_scale_frame` on property
    /// change; the key comparison is the port's shape of that).
    scaler: Option<SwsScaler>,
    last_scaler_key: Option<(PixelFormat, u32, u32, PixelFormat, u32, u32)>,
}

impl Default for ScaleContext {
    /// C's zero-initialized priv + the AVOption-table defaults applied by
    /// `av_opt_set_defaults` before the user dict (avfilter.c:929-933).
    /// The nonzero table defaults (vf_scale.c:1069-1162): interl=0,
    /// force_divisible_by=1, reset_sar=0, eval=init, flags=""; everything
    /// else is the zero/NULL value.
    fn default() -> Self {
        ScaleContext {
            w_expr: String::new(),
            h_expr: String::new(),
            w_pexpr: expr::Expr::default(),
            h_pexpr: expr::Expr::default(),
            size_str: None,
            flags_str: None,
            interlaced: 0,
            in_color_matrix: None,
            out_color_matrix: None,
            in_range: None,
            out_range: None,
            in_chroma_loc: ChromaLocation::Unspecified,
            out_chroma_loc: ChromaLocation::Unspecified,
            force_original_aspect_ratio: ForceOriginalAspectRatio::Disable,
            force_divisible_by: 1,
            reset_sar: false,
            eval_mode: EvalMode::Init,
            w: 0,
            h: 0,
            var_sar: 0.0,
            var_n: 0.0,
            var_t: 0.0,
            w_expr_uses_n: false,
            w_expr_uses_t: false,
            h_expr_uses_n: false,
            h_expr_uses_t: false,
            algorithm: ScaleAlgorithm::default(),
            scaler: None,
            last_scaler_key: None,
        }
    }
}

/// `sws_is_noop` (format.c:693-705) + `sanitize_fmt` (format.c:305-335) +
/// `ff_fmt_equal` (format.h:125-135), reduced to the modeled field subset:
/// format, geometry, the sanitized (range, csp) pair and the RAW chroma
/// location. Primaries/trc/luma are equal by construction here
/// (`copy_props` precedes the check and there is no primaries/trc override
/// option), so they are not compared; interlaced/field likewise (the
/// `interl` override is applied to BOTH sides before the comparison —
/// copy_props runs after it — and is restored on the output afterwards).
///
/// The per-format sanitize (305-335): RGB-flagged formats are
/// full-range/gbr; formats with <3 components (grayscale) are full-range
/// with an unspecified matrix. The YUVJ range override and the
/// chroma-location reset for non-subsampled formats have no enum variants
/// to trigger them (no YUVJ formats; the only chroma-carrying supported
/// format, yuv420p, is subsampled).
fn sanitized(fmt: PixelFormat, range: ColorRange, csp: ColorSpace) -> (ColorRange, ColorSpace) {
    let desc = pixdesc::descriptor(fmt);
    if desc.flags.contains(PixFmtFlags::RGB) {
        (ColorRange::Jpeg, ColorSpace::Rgb)
    } else if desc.nb_components < 3 {
        (ColorRange::Jpeg, ColorSpace::Unspecified)
    } else {
        (range, csp)
    }
}

/// The noop predicate: a frame pair sws would pass through unchanged.
fn is_noop(out: &Frame, in_: &Frame) -> bool {
    if in_.format != out.format || in_.width != out.width || in_.height != out.height {
        return false;
    }
    if sanitized(in_.format, in_.color_range, in_.color_space)
        != sanitized(out.format, out.color_range, out.color_space)
    {
        return false;
    }
    in_.chroma_location == out.chroma_location
}

// ---------------------------------------------------------------------------
// FilterImpl — init / query_formats / config_props / filter_frame
// ---------------------------------------------------------------------------

impl FilterImpl for ScaleContext {
    /// `init` (vf_scale.c:328-439; `preinit` 310-324 folded in — there is
    /// no SwsContext to pre-allocate). The C option table (1069-1162) is
    /// the authority for accepted names; the ported subset is exactly:
    /// `w`/`width`, `h`/`height`, `flags`, `interl`, `size`/`s`,
    /// `in_/out_color_matrix`, `in_/out_range`, `in_/out_chroma_loc`,
    /// `force_original_aspect_ratio`, `force_divisible_by`, `reset_sar`,
    /// `eval`. Duplicate keys: LAST occurrence wins (the
    /// `AV_DICT_MULTIKEY` apply, same as buffersrc/vf_format). Unknown
    /// options are pre-scanned FIRST so they beat init's own validation —
    /// C applies AVOptions before init runs.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        const KNOWN_KEYS: &[&str] = &[
            "w",
            "width",
            "h",
            "height",
            "flags",
            "interl",
            "size",
            "s",
            "in_color_matrix",
            "out_color_matrix",
            "in_range",
            "out_range",
            "in_chroma_loc",
            "out_chroma_loc",
            "force_original_aspect_ratio",
            "force_divisible_by",
            "reset_sar",
            "eval",
        ];
        if let Some((key, _)) = g.nodes[node.0]
            .opts
            .entries
            .iter()
            .find(|(k, _)| !KNOWN_KEYS.contains(&k.as_str()))
        {
            return Err(Error::NotFound(format!("No such option: {key}")));
        }

        let name = g.nodes[node.0].name.clone();

        // ---- Phase 1: option application (last occurrence wins) ----------
        let mut w_opt: Option<String> = None;
        let mut h_opt: Option<String> = None;
        {
            let entries = std::mem::take(&mut g.nodes[node.0].opts.entries);
            let mut leftovers = Vec::new();
            for (key, value) in entries {
                // Fine-grained error texts need the fail-fast ?; unknown
                // keys were already rejected by the pre-scan, so the `_`
                // arm only receives them on a concurrent mutation (never).
                match key.as_str() {
                    "w" | "width" => w_opt = Some(value),
                    "h" | "height" => h_opt = Some(value),
                    "flags" => self.flags_str = Some(value),
                    "interl" => self.interlaced = parse_bool_val(&name, &key, &value, -1, 1)?,
                    "size" | "s" => self.size_str = Some(value),
                    "in_color_matrix" => {
                        self.in_color_matrix = parse_color_matrix(&name, &key, &value)?
                    }
                    "out_color_matrix" => {
                        self.out_color_matrix = parse_color_matrix(&name, &key, &value)?
                    }
                    "in_range" => self.in_range = parse_range_opt(&name, &key, &value)?,
                    "out_range" => self.out_range = parse_range_opt(&name, &key, &value)?,
                    "in_chroma_loc" => self.in_chroma_loc = parse_chroma_loc(&name, &key, &value)?,
                    "out_chroma_loc" => {
                        self.out_chroma_loc = parse_chroma_loc(&name, &key, &value)?
                    }
                    "force_original_aspect_ratio" => {
                        self.force_original_aspect_ratio = parse_foar(&name, &key, &value)?
                    }
                    "force_divisible_by" => {
                        self.force_divisible_by = parse_divisible_by(&name, &key, &value)?
                    }
                    "reset_sar" => self.reset_sar = parse_bool_val(&name, &key, &value, 0, 1)? != 0,
                    "eval" => self.eval_mode = parse_eval_mode(&name, &key, &value)?,
                    _ => leftovers.push((key, value)),
                }
            }
            g.nodes[node.0].opts.entries = leftovers;
        }

        // ---- (1) size and w/h are mutually exclusive (336-340) ------------
        if self.size_str.is_some() && (w_opt.is_some() || h_opt.is_some()) {
            let msg = "Size and width/height expressions cannot be set at the same time.";
            log_error!(Some(&name), "{msg}\n");
            return Err(Error::InvalidArgument(msg.to_string()));
        }

        // ---- (2) the w-only quirk (342-343): `scale=640` (w set, h not)
        // SWAPS w_expr into size_str — so it must then FAIL
        // `av_parse_video_size` ("Invalid size '640'", the function needs
        // WxH). Do not "fix" this.------------------------------------------
        let mut w_field = w_opt;
        if w_field.is_some() && h_opt.is_none() {
            std::mem::swap(&mut w_field, &mut self.size_str);
        }
        if let Some(w) = w_field {
            self.w_expr = w;
        }
        if let Some(h) = h_opt {
            self.h_expr = h;
        }

        // ---- (3) size_str → literal dimensions (345-356) -------------------
        if let Some(size) = self.size_str.take() {
            match parse_video_size(&size) {
                Ok((w, h)) => {
                    self.w_expr = format!("{w}");
                    self.h_expr = format!("{h}");
                }
                Err(_) => {
                    log_error!(Some(&name), "Invalid size '{size}'\n");
                    return Err(Error::InvalidArgument(format!("Invalid size '{size}'")));
                }
            }
        }

        // ---- (4) defaults (357-360) -----------------------------------------
        if self.w_expr.is_empty() {
            self.w_expr = "iw".to_string();
        }
        if self.h_expr.is_empty() {
            self.h_expr = "ih".to_string();
        }

        // ---- (5) parse + check (362-368, via scale_parse_expr) --------------
        self.w_pexpr = parse_scale_expr(&self.w_expr, "width", &name)?;
        self.h_pexpr = parse_scale_expr(&self.h_expr, "height", &name)?;
        // C runs check_exprs after EACH parse; the intermediate w-only check
        // is a subset of this one (every individual test is monotone in the
        // set of parsed expressions), so one final check is equivalent.
        check_exprs(
            &self.w_pexpr,
            &self.h_pexpr,
            &self.w_expr,
            &self.h_expr,
            self.eval_mode,
            &name,
        )?;
        self.w_expr_uses_n = expr::uses(&self.w_pexpr, "n");
        self.w_expr_uses_t = expr::uses(&self.w_pexpr, "t");
        self.h_expr_uses_n = expr::uses(&self.h_pexpr, "n");
        self.h_expr_uses_t = expr::uses(&self.h_pexpr, "t");

        // ---- (6) color matrix validation (394-404) --------------------------
        // Dead code via the NAME parser (every accepted name passes
        // sws_test_colorspace) but ported for shape: C reaches it with
        // NUMERIC option values, which the port does not accept.
        if let Some(m) = self.in_color_matrix {
            if !sws_test_colorspace(m) {
                let msg = format!("Unsupported input color matrix '{}'", color_space_name(m));
                log_error!(Some(&name), "{msg}\n");
                return Err(Error::InvalidArgument(msg));
            }
        }
        if let Some(m) = self.out_color_matrix {
            if !sws_test_colorspace(m) {
                let msg = format!("Unsupported output color matrix '{}'", color_space_name(m));
                log_error!(Some(&name), "{msg}\n");
                return Err(Error::InvalidArgument(msg));
            }
        }

        // ---- (7) the verbose parameter dump (406-407) ------------------------
        log_verbose!(
            Some(&name),
            "w:{} h:{} flags:'{}' interl:{}\n",
            self.w_expr,
            self.h_expr,
            self.flags_str.as_deref().unwrap_or(""),
            self.interlaced
        );

        // ---- (8) flags → algorithm (409-413) ----------------------------------
        // Divergence: C's `av_opt_set(sws, "sws_flags", ...)` also accepts
        // `fast_bilinear`/`x`/`bicublin` — kernels the Scaler does not
        // implement; those names error here instead.
        if let Some(flags) = self.flags_str.clone().filter(|f| !f.is_empty()) {
            match ScaleAlgorithm::from_name(&flags) {
                Some(a) => self.algorithm = a,
                None => {
                    let msg = format!(
                        "Unable to parse \"flags\" option value \"{flags}\" as scaler flags \
                         (supported: nearest, point, bilinear, bicubic, area, gauss, sinc, \
                         lanczos, spline)"
                    );
                    log_error!(Some(&name), "{msg}\n");
                    return Err(Error::InvalidArgument(msg));
                }
            }
        } else {
            self.algorithm = g.scale_algorithm;
        }

        // (9) param/chr-pos copy (415-422) and the dynamic ref pad (428-436):
        // dropped (options not ported; no ref pad). Nothing else to store —
        // the Scaler is created lazily at config/frame time.
        Ok(())
    }

    /// `query_formats` (vf_scale.c:451-530) — ONE-SIDED lists on OWN pads
    /// (this filter's two sides carry DIFFERENT lists, so no
    /// `set_common_*`): the input link's `outcfg` (what scale accepts) and
    /// the output link's `incfg` (what it produces). Every slot is written
    /// fill-if-unset (matching `set_common_*` semantics — a re-query never
    /// overwrites; the engine runs `g.default_query_formats(node)` after
    /// this to fill the far halves).
    ///
    /// List ORDER is load-bearing: `pick_format` takes element [0], so an
    /// unconstrained scale graph negotiates yuv420p on both sides — the
    /// guaranteed-supported path. The `out_color_matrix`/`out_range`
    /// singletons (502-521) are what make an auto-inserted scale actually
    /// settle a colorspace/range mismatch.
    ///
    /// Not ported: PAL8 in the output list (477 — no enum variant) and the
    /// `alpha_blend` singleton block (523-527 — no alpha axis).
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let inl = g.inlink(node, 0);
        let outl = g.outlink(node, 0);

        // (a) input formats: sws_test_format(fmt, 0) over the descriptor
        // walk (463-471) == supported_input over PixelFormat::ALL order.
        if g.links[inl.0].outcfg.formats.is_none() {
            let list: Vec<PixelFormat> = PixelFormat::ALL
                .iter()
                .copied()
                .filter(|&f| swscale::supported_input(f))
                .collect();
            g.links[inl.0].outcfg.formats = Some(g.alloc_pix_list(list));
        }
        // (b) output formats: sws_test_format(fmt, 1) (473-483).
        if g.links[outl.0].incfg.formats.is_none() {
            let list: Vec<PixelFormat> = PixelFormat::ALL
                .iter()
                .copied()
                .filter(|&f| swscale::supported_output(f))
                .collect();
            g.links[outl.0].incfg.formats = Some(g.alloc_pix_list(list));
        }
        // (c) input color spaces: ff_all_color_spaces filtered by
        // sws_test_colorspace (486-493) — a no-op over the port's enum
        // (every value except Reserved passes, and all_color_spaces()
        // already excludes Reserved).
        if g.links[inl.0].outcfg.color_spaces.is_none() {
            let list = g.alloc_csp_list(formats::all_color_spaces());
            g.links[inl.0].outcfg.color_spaces = Some(list);
        }
        // (d) input color ranges (497-499).
        if g.links[inl.0].outcfg.color_ranges.is_none() {
            let list = g.alloc_rng_list(formats::all_color_ranges());
            g.links[inl.0].outcfg.color_ranges = Some(list);
        }
        // (e) output color spaces: the out_color_matrix singleton (502-513).
        if g.links[outl.0].incfg.color_spaces.is_none() {
            let list = match self.out_color_matrix {
                Some(m) => g.alloc_csp_list(vec![m]),
                None => g.alloc_csp_list(formats::all_color_spaces()),
            };
            g.links[outl.0].incfg.color_spaces = Some(list);
        }
        // (f) output color ranges (517-521).
        if g.links[outl.0].incfg.color_ranges.is_none() {
            let list = match self.out_range {
                Some(r) => g.alloc_rng_list(vec![r]),
                None => g.alloc_rng_list(formats::all_color_ranges()),
            };
            g.links[outl.0].incfg.color_ranges = Some(list);
        }
        Ok(())
    }

    /// `config_props` — wired on the OUTPUT pad only (vf_scale.c:1181-1187).
    fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
        match pad {
            PadRef::Out(0) => self.config_props_out(g, node),
            _ => Ok(()),
        }
    }

    /// `filter_frame` (vf_scale.c:961-973): scale, then push. (C's
    /// `if (out)` check around the push is about the `sws_scale_frame`
    /// failure path freeing the frame — the port's `?` already returned.)
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        _pad: usize,
        frame: Frame,
    ) -> Result<()> {
        let inlink = g.inlink(node, 0);
        let out_frame = self.scale_frame(g, node, inlink, frame)?;
        let outlink = g.outlink(node, 0);
        filter::filter_frame(g, outlink, out_frame)
    }
}

impl ScaleContext {
    /// `scale_eval_dimensions` — vf_scale.c's OWN version (532-619), NOT the
    /// shared `ff_scale_eval_dimensions` (scale_eval.c:58-121). Differences
    /// from the shared twin, kept deliberately: the FIRST w-pass has no NaN
    /// check (592 — a NaN width casts to i32::MIN and is consumed by the
    /// height pass), there are no INT32 range checks, and the fallback for
    /// a zero cast is the input dimension (not `trunc(res)`).
    ///
    /// The `(int)res` casts are x86 `cvttsd2si` (NaN/overflow → i32::MIN);
    /// Rust's bare `as` saturates (NaN → 0), so [`c_int`] reproduces C.
    fn scale_eval_dimensions(&mut self, g: &FilterGraph, node: NodeId) -> Result<()> {
        let name = g.nodes[node.0].name.clone();
        let inlink = g.inlink(node, 0);
        let outlink = g.outlink(node, 0);
        let l = &g.links[inlink.0];
        // Formats are picked before link config (avfiltergraph.c order).
        let desc = pixdesc::descriptor(
            l.format
                .expect("formats picked before config_props (query_formats round)"),
        );
        let out_desc = pixdesc::descriptor(
            g.links[outlink.0]
                .format
                .expect("formats picked before config_props (query_formats round)"),
        );

        // Fill the variable slots (552-563).
        let mut vars = expr::Vars {
            in_w: l.w as f64,
            in_h: l.h as f64,
            out_w: f64::NAN,
            out_h: f64::NAN,
            a: l.w as f64 / l.h as f64,
            sar: if l.sample_aspect_ratio.num != 0 {
                l.sample_aspect_ratio.num as f64 / l.sample_aspect_ratio.den as f64
            } else {
                1.0
            },
            dar: 0.0,
            hsub: (1u32 << desc.log2_chroma_w) as f64,
            vsub: (1u32 << desc.log2_chroma_h) as f64,
            ohsub: (1u32 << out_desc.log2_chroma_w) as f64,
            ovsub: (1u32 << out_desc.log2_chroma_h) as f64,
            n: self.var_n,
            t: self.var_t,
        };
        vars.dar = vars.a * vars.sar;
        self.var_sar = vars.sar; // cached for reset_sar's w_adj (639-641)

        // Pass 1 (591-592): w, with its result published to ow BEFORE the h
        // evaluation (so `h = oh/2` sees it) and NO NaN check.
        let res = expr::eval(&self.w_pexpr, &vars);
        let w0 = if c_int(res) == 0 {
            l.w as i32
        } else {
            c_int(res)
        };
        vars.out_w = w0 as f64;

        // Pass 2 (594-600): h.
        let res = expr::eval(&self.h_pexpr, &vars);
        if res.is_nan() {
            return Err(eval_failed(&self.h_expr, &name));
        }
        let eval_h = if c_int(res) == 0 {
            l.h as i32
        } else {
            c_int(res)
        };
        vars.out_h = eval_h as f64;

        // Pass 3 (602-608): w again, now that oh is available.
        let res = expr::eval(&self.w_pexpr, &vars);
        if res.is_nan() {
            return Err(eval_failed(&self.w_expr, &name));
        }
        let eval_w = if c_int(res) == 0 {
            l.w as i32
        } else {
            c_int(res)
        };

        self.w = eval_w;
        self.h = eval_h;
        Ok(())
    }

    /// `config_props` (vf_scale.c:621-714), factored out so the
    /// frame-changed reconfigure path can re-run it (810).
    fn config_props_out(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let name = g.nodes[node.0].name.clone();
        let inlink = g.inlink(node, 0);
        let outlink = g.outlink(node, 0);

        // (1) evaluate the expressions (633-634).
        self.scale_eval_dimensions(g, node)?;

        // (2) provisional geometry (636-637) — possibly still negative
        // (-1/-N semantics); only assigned to the link AFTER adjust below
        // (C stores into outlink->w/h and hands those pointers to
        // ff_scale_adjust_dimensions; the port keeps i64 locals instead so
        // the negative sentinels never round-trip through the u32 link
        // fields).
        let mut w = self.w as i64;
        let mut h = self.h as i64;

        // (3) w_adj (639-641).
        let w_adj = if self.reset_sar { self.var_sar } else { 1.0 };

        // (4) ff_scale_adjust_dimensions (643-648).
        let (in_w, in_h) = (g.links[inlink.0].w as i32, g.links[inlink.0].h as i32);
        scale_adjust_dimensions(
            in_w,
            in_h,
            &mut w,
            &mut h,
            self.force_original_aspect_ratio,
            self.force_divisible_by,
            w_adj,
        )?;
        {
            let ol = &mut g.links[outlink.0];
            ol.w = w as u32;
            ol.h = h as u32;
        }

        // (5) the INT_MAX check (650-654) — log-only in C, kept verbatim
        // (practically dead after adjust's own i32 check; the two
        // cross-product lines are why it survives).
        {
            let ol = &g.links[outlink.0];
            let il = &g.links[inlink.0];
            if ol.w > i32::MAX as u32
                || ol.h > i32::MAX as u32
                || (ol.h as u64) * (il.w as u64) > i32::MAX as u64
                || (ol.w as u64) * (il.h as u64) > i32::MAX as u64
            {
                log_error!(
                    Some(&name),
                    "Rescaled value for width or height is too big.\n"
                );
            }
        }

        // (6) the output SAR (658-664).
        {
            let (in_sar, in_w, in_h) = {
                let il = &g.links[inlink.0];
                (il.sample_aspect_ratio, il.w, il.h)
            };
            let ol = &mut g.links[outlink.0];
            if self.reset_sar {
                ol.sample_aspect_ratio = Rational::ONE;
            } else if in_sar.num != 0 {
                // av_div_q(av_make_q(in_w, in_h), av_make_q(out_w, out_h)) *
                // in_sar — Rational's Div/Mul are av_div_q/av_mul_q.
                let q = Rational::new(in_w as i32, in_h as i32)
                    / Rational::new(ol.w as i32, ol.h as i32);
                ol.sample_aspect_ratio = q * in_sar;
            } else {
                ol.sample_aspect_ratio = in_sar;
            }
        }

        // (7) the verbose dump (666-674) — C prints the numeric sws_flags;
        // the port prints the algorithm name in that slot.
        {
            let il = &g.links[inlink.0];
            let ol = &g.links[outlink.0];
            log_verbose!(
                Some(&name),
                "w:{} h:{} fmt:{} csp:{} range:{} sar:{}/{} -> w:{} h:{} fmt:{} csp:{} range:{} sar:{}/{} flags:{}\n",
                il.w,
                il.h,
                il.format.map(|f| f.name()).unwrap_or("none"),
                color_space_name(il.colorspace),
                color_range_name(il.color_range),
                il.sample_aspect_ratio.num,
                il.sample_aspect_ratio.den,
                ol.w,
                ol.h,
                ol.format.map(|f| f.name()).unwrap_or("none"),
                color_space_name(ol.colorspace),
                color_range_name(ol.color_range),
                ol.sample_aspect_ratio.num,
                ol.sample_aspect_ratio.den,
                self.algorithm.name(),
            );
        }

        // (8) size-dependent/color-dependent side-data removal (677-685)
        // and the framesync (re)init (687-708): not ported (no side data,
        // no framesync).

        // (9) RUST ADDITION — conversion-pair probe: build (and drop) a
        // Scaler for the negotiated pair so an unsupported conversion fails
        // graph CONFIG with the Scaler's clear text instead of the first
        // frame. Probed with the CPU engine so no GPU device is touched
        // during configuration; the REAL scaler (with g.scale_engine) is
        // built at filter_frame like C's dynamic sws_scale_frame init.
        let (ifmt, iw, ih) = {
            let il = &g.links[inlink.0];
            (
                il.format.expect("formats picked before config_props"),
                il.w,
                il.h,
            )
        };
        let (ofmt, ow, oh) = {
            let ol = &g.links[outlink.0];
            (
                ol.format.expect("formats picked before config_props"),
                ol.w,
                ol.h,
            )
        };
        SwsScaler::new(
            (ifmt, iw, ih),
            (ofmt, ow, oh),
            ScaleOptions {
                algorithm: self.algorithm,
                engine: ScaleEngine::Cpu,
            },
        )
        .map(|_| ())?;

        Ok(())
    }

    /// `scale_frame` (vf_scale.c:744-896) — takes ownership of the input
    /// frame, returns the output frame. The framesync wrapper (`do_scale`,
    /// 898-959) is collapsed: single input, no ref, pts passed through
    /// unchanged.
    fn scale_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        inlink: crate::filter::LinkId,
        mut in_frame: Frame,
    ) -> Result<Frame> {
        let name = g.nodes[node.0].name.clone();
        let outlink = g.outlink(node, 0);

        // (1) frame_changed (758-764) — num and den compared separately.
        let frame_changed = {
            let l = &g.links[inlink.0];
            in_frame.width != l.w
                || in_frame.height != l.h
                || l.format != Some(in_frame.format)
                || in_frame.sample_aspect_ratio.den != l.sample_aspect_ratio.den
                || in_frame.sample_aspect_ratio.num != l.sample_aspect_ratio.num
                || in_frame.color_space != l.colorspace
                || in_frame.color_range != l.color_range
        };

        // (2) the reconfigure dance (766-812).
        if self.eval_mode == EvalMode::Frame || frame_changed {
            // Fast path (772-778): frame mode, nothing changed, the
            // expressions need no per-frame variables, and the last
            // evaluated dimensions are non-zero → skip straight to scaling.
            let fast_path = self.eval_mode == EvalMode::Frame
                && !frame_changed
                && !self.w_expr_uses_n
                && !self.w_expr_uses_t
                && !self.h_expr_uses_n
                && !self.h_expr_uses_t
                && self.w != 0
                && self.h != 0;
            if !fast_path {
                // Init mode: freeze the expressions to the literals of the
                // last evaluated dimensions (780-793) — the re-parse this
                // port retains of scale_parse_expr's revert machinery.
                if self.eval_mode == EvalMode::Init {
                    self.w_expr = format!("{}", self.w);
                    self.h_expr = format!("{}", self.h);
                    self.w_pexpr = parse_scale_expr(&self.w_expr, "width", &name)?;
                    self.h_pexpr = parse_scale_expr(&self.h_expr, "height", &name)?;
                    // (C re-runs check_exprs and config_props inside
                    // scale_parse_expr here — the inner config_props call
                    // with the OLD link state is overwritten by the one at
                    // (810) below; literals trivially pass check_exprs.)
                    self.w_expr_uses_n = expr::uses(&self.w_pexpr, "n");
                    self.w_expr_uses_t = expr::uses(&self.w_pexpr, "t");
                    self.h_expr_uses_n = expr::uses(&self.h_pexpr, "n");
                    self.h_expr_uses_t = expr::uses(&self.h_pexpr, "t");
                }

                // The frame variables (795-801): frame_count_out still
                // carries C's pre-callback value (the engine's
                // filter_frame_to_filter decremented it back, filter.rs).
                // TS2T (filters.h:483): NOPTS → NaN.
                self.var_n = g.links[inlink.0].frame_count_out as f64;
                self.var_t = if in_frame.pts == NOPTS {
                    f64::NAN
                } else {
                    in_frame.pts as f64 * g.links[inlink.0].time_base.to_f64()
                };

                // Update the LINK from the frame (803-808) BEFORE the
                // re-config.
                {
                    let l = &mut g.links[inlink.0];
                    l.format = Some(in_frame.format);
                    l.w = in_frame.width;
                    l.h = in_frame.height;
                    l.colorspace = in_frame.color_space;
                    l.color_range = in_frame.color_range;
                    l.sample_aspect_ratio = in_frame.sample_aspect_ratio;
                }
                self.config_props_out(g, node)?;
            }
        }

        // (3) SCALE — assemble the output frame.
        let (out_fmt, out_w, out_h) = {
            let l = &g.links[outlink.0];
            (
                l.format
                    .expect("formats picked before filter_frame (query_formats round)"),
                l.w,
                l.h,
            )
        };
        let mut out = Frame::alloc(out_fmt, out_w, out_h)?;

        // Input tag overrides on the OWNED input frame (824-832). The
        // in_chroma_loc stamp is UNCONDITIONAL — the C quirk: the default
        // Unspecified OVERWRITES a decoded Left/Center tag.
        if let Some(m) = self.in_color_matrix {
            in_frame.color_space = m;
        }
        if let Some(r) = self.in_range {
            in_frame.color_range = r;
        }
        in_frame.chroma_location = self.in_chroma_loc;

        let flags_orig = in_frame.flags;
        if self.interlaced > 0 {
            in_frame.flags = in_frame.flags.union(FrameFlags::INTERLACED);
        } else if self.interlaced == 0 {
            in_frame.flags = FrameFlags(in_frame.flags.0 & !FrameFlags::INTERLACED.0);
        }

        // copy_props, then stamp the negotiated output color metadata
        // (840-847); out.width/height already hold the link geometry
        // (Frame::alloc).
        out.copy_props(&in_frame);
        out.color_range = g.links[outlink.0].color_range;
        out.color_space = g.links[outlink.0].colorspace;
        if self.out_chroma_loc != ChromaLocation::Unspecified {
            out.chroma_location = self.out_chroma_loc;
        }

        // The output SAR (863-870): reset_sar takes the link value;
        // otherwise the av_reduce of the dimension ratios — note the INPUT
        // LINK dims (possibly updated above), not the frame's.
        if self.reset_sar {
            out.sample_aspect_ratio = g.links[outlink.0].sample_aspect_ratio;
        } else {
            let l = &g.links[inlink.0];
            out.sample_aspect_ratio = Rational::reduce(
                in_frame.sample_aspect_ratio.num as i64 * out_h as i64 * l.w as i64,
                in_frame.sample_aspect_ratio.den as i64 * out_w as i64 * l.h as i64,
                i32::MAX as i64,
            )
            .0;
        }

        // NOOP pass-through (872-877): drop the assembled output, restore
        // the ORIGINAL flags; the returned frame keeps the overridden color
        // tags but the original flags.
        if is_noop(&out, &in_frame) {
            drop(out);
            in_frame.flags = flags_orig;
            return Ok(in_frame);
        }

        // CONVERSION GATES (Rust addition — C converts, the Scaler cannot):
        // only the same-matrix yuv420p→yuv420p resize is tag-sensitive.
        // gray8→gray8 skips both (the kernel ignores range/matrix); RGB
        // outputs skip both (full-range by construction; the input tag is
        // consumed by the converter).
        if in_frame.format == PixelFormat::Yuv420p && out.format == PixelFormat::Yuv420p {
            let in_csp = in_frame.color_space;
            let out_csp = g.links[outlink.0].colorspace;
            if matrix_class(in_csp) != matrix_class(out_csp) {
                let msg = format!(
                    "color matrix conversion {} -> {} is not supported by the swscale port",
                    color_space_name(in_csp),
                    color_space_name(out_csp)
                );
                log_error!(Some(&name), "{msg}\n");
                return Err(Error::Unsupported(msg));
            }
            let resolve = |r: ColorRange| {
                if r == ColorRange::Unspecified {
                    ColorRange::Mpeg
                } else {
                    r
                }
            };
            let (in_rng, out_rng) = (resolve(in_frame.color_range), resolve(out.color_range));
            if in_rng != out_rng {
                let msg = format!(
                    "color range conversion {} -> {} is not supported by the swscale port",
                    color_range_name(in_rng),
                    color_range_name(out_rng)
                );
                log_error!(Some(&name), "{msg}\n");
                return Err(Error::Unsupported(msg));
            }
        }

        // The Scaler, lazily (re)created when the conversion key changes —
        // the port's shape of C's sws_scale_frame reinitializing the
        // SwsContext on property change. Creation errors (e.g. a
        // table-driven kernel on the Vulkan engine) surface at frame time
        // exactly like C's.
        let key = (
            in_frame.format,
            in_frame.width,
            in_frame.height,
            out.format,
            out.width,
            out.height,
        );
        if self.scaler.is_none() || self.last_scaler_key != Some(key) {
            self.scaler = Some(SwsScaler::new(
                (in_frame.format, in_frame.width, in_frame.height),
                (out.format, out.width, out.height),
                ScaleOptions {
                    algorithm: self.algorithm,
                    engine: g.scale_engine,
                },
            )?);
            self.last_scaler_key = Some(key);
        }
        self.scaler
            .as_mut()
            .expect("scaler created above")
            .scale(&in_frame, &mut out)?;

        // (886-887): restore the original flags; out.format already equals
        // the link format (no PAL8 rewrite to undo).
        out.flags = flags_orig;
        Ok(out)
    }
}

/// x86 `cvttsd2si` — the semantics of C's `(int)double` cast: NaN and
/// out-of-range become i32::MIN (Rust's bare `as` saturates, NaN → 0).
fn c_int(r: f64) -> i32 {
    if r.is_nan() || r >= 2147483648.0 || r < -2147483648.0 {
        i32::MIN
    } else {
        r as i32
    }
}

/// The `fail:` tail of scale_eval_dimensions (615-618).
fn eval_failed(expr_str: &str, name: &str) -> Error {
    let msg = format!("Error when evaluating the expression '{expr_str}'.");
    log_error!(Some(name), "{msg}\n");
    Error::InvalidArgument(msg)
}

// ---------------------------------------------------------------------------
// Filter descriptor (vf_scale.c:1164-1203)
// ---------------------------------------------------------------------------

/// The shared single video pad (vf_scale.c:1174-1187). The output pad's
/// `config_props` callback lives on [`FilterImpl`]; the input pad's default
/// buffer callback is not ported (no frame pools).
static DEFAULT_PAD: PadDef = PadDef {
    name: "default",
    needs_writable: false,
};

/// `ff_vf_scale` (vf_scale.c:1189-1203).
///
/// * flags carry [`FilterFlags::ALLOWS_RECONFIGURE`]: "scale" IS in C's
///   `ff_filter_frame` validation skip list (avfilter.c:1075-1082) — the
///   frame-changed reconfigure path genuinely pushes frames that differ
///   from the not-yet-renegotiated link.
/// * `AVFILTER_FLAG_DYNAMIC_INPUTS` (1193) exists only for the ref pad —
///   not applicable.
/// * shorthand `["w","h","flags","interl","size"]`: C's
///   `ff_filter_opt_parse` derives it from the AVOption table in
///   declaration order skipping duplicate OFFSETs (width/height/s are
///   aliases — avfilter.c:855-902).
pub static SCALE_DEF: FilterDef = FilterDef {
    name: "scale",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags::ALLOWS_RECONFIGURE,
    shorthand: &["w", "h", "flags", "interl", "size"],
    make: || Box::new(ScaleContext::default()),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{LinkId, filter_def};
    use crate::swscale::ScaleEngine;

    /// buffer(yuv420p WxH, tb 1/25, sar 1/1) → scale(args) → buffersink,
    /// configured. CPU engine so no GPU device is touched.
    fn scale_graph(args: &str, size: (u32, u32)) -> (FilterGraph, NodeId, NodeId, LinkId, LinkId) {
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                &format!(
                    "video_size={}x{}:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
                    size.0, size.1
                ),
            )
            .expect("buffer init");
        let scale = g.create_filter("scale", args).expect("scale init");
        let sink = g.create_filter("buffersink", "").unwrap();
        let lin = g.link(src, 0, scale, 0).unwrap();
        let lout = g.link(scale, 0, sink, 0).unwrap();
        g.config().expect("graph config");
        (g, src, sink, lin, lout)
    }

    /// A constant-color yuv420p frame (the upscale of a constant is the
    /// same constant under every kernel — that is the point).
    fn yuv_frame(w: u32, h: u32, pts: i64) -> Frame {
        let mut f = Frame::alloc(PixelFormat::Yuv420p, w, h).unwrap();
        f.pts = pts;
        f.duration = 1;
        f.time_base = Rational::new(1, 25);
        for (p, v) in [(0usize, 100u8), (1, 120), (2, 130)] {
            for b in f.plane_mut(p) {
                *b = v;
            }
        }
        f
    }

    fn drain(g: &mut FilterGraph, sink: NodeId) -> Vec<Frame> {
        let mut out = Vec::new();
        while let Ok(f) = g.get_frame(sink) {
            out.push(f);
        }
        out
    }

    fn scale_err(args: &str) -> Error {
        let mut g = FilterGraph::new();
        g.create_filter("scale", args).expect_err("init must fail")
    }

    /// The engine step at avfiltergraph.c:392: call the node's
    /// query_formats (imp taken out of the node for the duration).
    fn run_query_formats(g: &mut FilterGraph, node: NodeId) {
        let mut imp = g.nodes[node.0].imp.take().expect("imp present");
        let ret = imp.query_formats(g, node);
        g.nodes[node.0].imp = Some(imp);
        ret.unwrap();
    }

    // ---- option intake / dimension evaluation -------------------------------

    #[test]
    fn size_option_parses_and_sets_geometry() {
        // av_parse_video_size via the size/s option (vf_scale.c:345-356).
        for args in ["size=640x480", "s=640x480"] {
            let (g, _src, _sink, _lin, lout) = scale_graph(args, (32, 24));
            assert_eq!(g.links[lout.0].w, 640, "{args}");
            assert_eq!(g.links[lout.0].h, 480, "{args}");
        }
        // Positional w:h fills both expressions (no size string involved).
        let (g, _s, _k, _i, lout) = scale_graph("100:50", (32, 24));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (100, 50));
    }

    #[test]
    fn w_only_quirk_and_size_conflict() {
        // `scale=640` (w set, h not) swaps w into size_str (342-343) and
        // then FAILS av_parse_video_size, which needs WxH — do not "fix".
        match scale_err("640") {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid size '640'"),
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("w=640") {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid size '640'"),
            other => panic!("unexpected error: {other}"),
        }
        // size and w/h are mutually exclusive (336-340).
        match scale_err("size=320x240:w=640") {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Size and width/height expressions cannot be set at the same time."
            ),
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("s=320x240:h=480") {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Size and width/height expressions cannot be set at the same time."
            ),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn parse_video_size_forms() {
        assert_eq!(parse_video_size("320x240").unwrap(), (320, 240));
        assert_eq!(parse_video_size("320X240").unwrap(), (320, 240)); // any ONE separator
        // Leading whitespace is skipped (strtol); a TRAILING space is
        // "extraneous data" — C rejects it (parseutils.c:171-175).
        assert!(parse_video_size(" 320x240 ").is_err());
        assert!(parse_video_size("320").is_err()); // height 0
        assert!(parse_video_size("320x240junk").is_err()); // trailing data
        assert!(parse_video_size("320x").is_err());
        assert!(parse_video_size("-1x5").is_err()); // non-positive
        assert!(parse_video_size("").is_err());
        // The named-size abbreviation table (ntsc/pal/vga) is not ported.
        assert!(parse_video_size("vga").is_err());
    }

    #[test]
    fn defaults_and_expression_evaluation() {
        // No options: w_expr='iw', h_expr='ih' (357-360) — same geometry.
        let (g, _s, _k, _i, lout) = scale_graph("", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (64, 48));
        let (g, _s, _k, _i, lout) = scale_graph("w=iw/2:h=ih/2", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (32, 24));
        // The (int)res == 0 quirk (592/608): a w expression EVALUATING to
        // (int)0 falls back to the INPUT width. (Note `w=0` the OPTION
        // would take the swap quirk below and fail av_parse_video_size —
        // the fallback needs an expression, hence h is set here too.)
        let (g, _s, _k, _i, lout) = scale_graph("w=iw*0:h=ih", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (64, 48));
        let (g, _s, _k, _i, lout) = scale_graph("w=0.4:h=ih", (100, 20));
        assert_eq!(g.links[lout.0].w, 100, "(int)0.4 == 0 -> in_w");
        // w/h expressions may NEVER reference ow/oh — check_exprs rejects
        // at INIT (vf_scale.c:197-204; verified against system ffmpeg),
        // WITH the trailing period (unlike "Cannot parse expression", 280).
        match scale_err("w=100:h=oh/2") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Height expression cannot be self-referencing: 'oh/2'.")
            }
            other => panic!("unexpected error: {other}"),
        }
        // hsub/vsub/ohsub/ovsub (560-563): yuv420p -> 2 in and out.
        let (g, _s, _k, _i, lout) = scale_graph("w=iw/hsub*4:h=ih/vsub*4", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (128, 96));
        // Last occurrence wins (AV_DICT_MULTIKEY apply order).
        let (g, _s, _k, _i, lout) = scale_graph("w=100:w=iw/2:h=ih", (64, 48));
        assert_eq!(g.links[lout.0].w, 32);
        // width is an alias of w in the option table (1070-1071).
        let (g, _s, _k, _i, lout) = scale_graph("width=16:height=9", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (16, 9));
    }

    #[test]
    fn w_only_expression_also_takes_the_swap_quirk() {
        // Any w-only option (even a pure expression) reaches the swap and
        // then fails av_parse_video_size — C behavior, not just for "640".
        match scale_err("w=iw/2") {
            Error::InvalidArgument(msg) => assert_eq!(msg, "Invalid size 'iw/2'"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn keep_aspect_negative_dimensions() {
        // scale_eval.c:145-156 with NEAR_INF rounding:
        // w=-1: av_rescale(100, 64, 48) = round(133.33) = 133.
        let (g, _s, _k, _i, lout) = scale_graph("w=-1:h=100", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (133, 100));
        // w=-2: av_rescale(100, 64, 96) = 67, then *2 = 134.
        let (g, _s, _k, _i, lout) = scale_graph("w=-2:h=100", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (134, 100));
        // h=-1 from w.
        let (g, _s, _k, _i, lout) = scale_graph("w=100:h=-1", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (100, 75));
        // Both negative: w = in_w * w_adj, h = in_h (145-148).
        let (g, _s, _k, _i, lout) = scale_graph("w=-1:h=-1", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (64, 48));
    }

    #[test]
    fn force_original_aspect_ratio() {
        // 64x48 is 4:3. Hand-computed per scale_eval.c:161-184:
        // decrease, fdb=1: tmp_w=133,tmp_h=75 -> min(133,100),min(75,100).
        let (g, _s, _k, _i, lout) =
            scale_graph("w=100:h=100:force_original_aspect_ratio=decrease", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (100, 75));
        // increase: max(133,100), max(75,100).
        let (g, _s, _k, _i, lout) =
            scale_graph("w=100:h=100:force_original_aspect_ratio=increase", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (133, 100));
        // decrease with fdb=2: tmp_w=67*2=134, tmp_h=38*2=76 ->
        // w=min(134,100)=100 (100/2*2), h=min(76,100)=76.
        let (g, _s, _k, _i, lout) = scale_graph(
            "w=100:h=100:force_original_aspect_ratio=decrease:force_divisible_by=2",
            (64, 48),
        );
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (100, 76));
        // increase with fdb=2: w=max(134,100)=134, h=max(76,100)=100
        // (100 rounds UP to the multiple of 2 it already is).
        let (g, _s, _k, _i, lout) = scale_graph(
            "w=100:h=100:force_original_aspect_ratio=increase:force_divisible_by=2",
            (64, 48),
        );
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (134, 100));
        // Numeric fallbacks (set_string_number over the unit CONSTs).
        let (g, _s, _k, _i, lout) =
            scale_graph("w=100:h=100:force_original_aspect_ratio=1", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (100, 75));
    }

    #[test]
    fn adjust_dimensions_unit_errors() {
        // w rounds to 0 -> non-positive (scale_eval.c:190-195).
        let mut w = 0i64;
        let mut h = 100i64;
        match scale_adjust_dimensions(
            1000,
            1000,
            &mut w,
            &mut h,
            ForceOriginalAspectRatio::Disable,
            1,
            1.0,
        )
        .unwrap_err()
        {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Rescaled dimensions 0x100 are invalid, output dimensions must be positive"
            ),
            other => panic!("unexpected error: {other}"),
        }
        // Not representable as i32 (187-188; port-chosen message text).
        let mut w = 3_000_000_000i64;
        let mut h = 100i64;
        match scale_adjust_dimensions(
            64,
            48,
            &mut w,
            &mut h,
            ForceOriginalAspectRatio::Disable,
            1,
            1.0,
        )
        .unwrap_err()
        {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Rescaled value for width or height is too big")
            }
            other => panic!("unexpected error: {other}"),
        }
        // A negative w_adj (negative SAR) hits av_rescale's b < 0 contract:
        // the i64::MIN sentinel propagates like C into the i32 check.
        let mut w = -1i64;
        let mut h = 100i64;
        assert!(
            scale_adjust_dimensions(
                64,
                48,
                &mut w,
                &mut h,
                ForceOriginalAspectRatio::Disable,
                1,
                -0.5
            )
            .is_err()
        );
    }

    #[test]
    fn sar_propagation() {
        // reset_sar=1: outlink SAR is 1/1 (658-659) and the frame takes it.
        let (mut g, src, sink, _lin, lout) = scale_graph("w=128:h=ih:reset_sar=1", (64, 48));
        assert_eq!(g.links[lout.0].sample_aspect_ratio, Rational::ONE);
        let mut f = yuv_frame(64, 48, 0);
        f.sample_aspect_ratio = Rational::new(1, 1); // decoded frames carry SAR
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!(out.sample_aspect_ratio, Rational::ONE);

        // Without reset (660-663): q = (64/48)/(128/48) = 1/2, times in_sar
        // 1/1 -> outlink 1/2. Frame SAR via av_reduce (866-869):
        // (1*48*64)/(1*128*48) = 1/2. The av_reduce reads the FRAME's SAR
        // (what the decoder stamped), never the link's.
        let (mut g, src, sink, _lin, lout) = scale_graph("w=128:h=ih", (64, 48));
        assert_eq!(g.links[lout.0].sample_aspect_ratio, Rational::new(1, 2));
        let mut f = yuv_frame(64, 48, 0);
        f.sample_aspect_ratio = Rational::new(1, 1);
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!(out.sample_aspect_ratio, Rational::new(1, 2));
    }

    #[test]
    fn unknown_option_rejected_with_c_shape() {
        // Dropped AVOptions stay unconsumed -> init_filter's leftover check.
        for key in [
            "param0",
            "in_primaries",
            "out_transfer",
            "in_h_chr_pos",
            "sws_param",
            "bogus",
        ] {
            match scale_err(&format!("{key}=1")) {
                Error::NotFound(msg) => assert_eq!(msg, format!("No such option: {key}")),
                other => panic!("unexpected error for {key}: {other}"),
            }
        }
    }

    #[test]
    fn interl_option_parsing() {
        // set_string_bool (opt.c:226-254): auto=-1, name families, decimal.
        for (val, want) in [
            ("auto", -1),
            ("-1", -1),
            ("0", 0),
            ("1", 1),
            ("true", 1),
            ("false", 0),
            ("yes", 1),
            ("off", 0),
        ] {
            let mut g = FilterGraph::new();
            let n = g.alloc_filter("scale").unwrap();
            g.nodes[n.0].opts.entries = vec![("interl".to_string(), val.to_string())];
            let mut imp = ScaleContext::default();
            FilterImpl::init(&mut imp, &mut g, n).unwrap_or_else(|e| panic!("interl={val}: {e}"));
            assert_eq!(imp.interlaced, want, "interl={val}");
        }
        // Out of the -1..=1 range: the boolean error text.
        match scale_err("interl=2") {
            Error::InvalidArgument(msg) => {
                assert_eq!(
                    msg,
                    "Unable to parse \"interl\" option value \"2\" as boolean"
                )
            }
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("interl=nope") {
            Error::InvalidArgument(msg) => {
                assert_eq!(
                    msg,
                    "Unable to parse \"interl\" option value \"nope\" as boolean"
                )
            }
            other => panic!("unexpected error: {other}"),
        }
        // reset_sar is BOOL 0..1: "auto" (-1) trips ITS range check.
        match scale_err("reset_sar=auto") {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Unable to parse \"reset_sar\" option value \"auto\" as boolean"
            ),
            other => panic!("unexpected error: {other}"),
        }
        // force_divisible_by is INT 1..=256 (write_number's range text).
        match scale_err("force_divisible_by=512") {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Value 512 for parameter force_divisible_by out of range"
            ),
            other => panic!("unexpected error: {other}"),
        }
    }

    // ---- check_exprs / parse errors ------------------------------------------

    #[test]
    fn expression_self_reference_checks() {
        // NOTE: a w-only option would hit the swap quirk first (see
        // w_only_expression_also_takes_the_swap_quirk), so these set h too.
        match scale_err("w=ow:h=ih") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Width expression cannot be self-referencing: 'ow'.")
            }
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("w=ih:h=oh") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Height expression cannot be self-referencing: 'oh'.")
            }
            other => panic!("unexpected error: {other}"),
        }
        // out_w is ow's alias (197: vars_w[VAR_OUT_W] || vars_w[VAR_OW]).
        match scale_err("w=out_w+1:h=ih") {
            Error::InvalidArgument(msg) => {
                assert_eq!(
                    msg,
                    "Width expression cannot be self-referencing: 'out_w+1'."
                )
            }
            other => panic!("unexpected error: {other}"),
        }
        // Circular w<->h only WARNS (207-210): init succeeds.
        let mut g = FilterGraph::new();
        g.create_filter("scale", "w=oh:h=ow")
            .expect("circular only warns");
        // n/t in init mode (244-252) — 'pos' stays in the text verbatim.
        match scale_err("w=iw+n:h=ih") {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Expressions with frame variables 'n', 't', 'pos' are not valid in init eval_mode."
            ),
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("h=ih+t") {
            Error::InvalidArgument(msg) => assert_eq!(
                msg,
                "Expressions with frame variables 'n', 't', 'pos' are not valid in init eval_mode."
            ),
            other => panic!("unexpected error: {other}"),
        }
        // eval=frame makes them legal.
        let mut g = FilterGraph::new();
        g.create_filter("scale", "eval=frame:w=iw+n:h=ih")
            .expect("frame mode");
        // Unknown identifiers (incl. the dropped scale2ref variables) are
        // parse errors with C's text (280).
        match scale_err("w=main_w:h=ih") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Cannot parse expression for width: 'main_w'")
            }
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("h=rw") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Cannot parse expression for height: 'rw'")
            }
            other => panic!("unexpected error: {other}"),
        }
        match scale_err("w=sin(iw):h=ih") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Cannot parse expression for width: 'sin(iw)'")
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn nan_evaluation_error() {
        // w='t' with a NOPTS frame in frame mode: TS2T(NOPTS) = NaN ->
        // "Error when evaluating the expression 't'." (615-618). The first
        // w pass has NO NaN check (592); the third (602-607) fires.
        let (mut g, src, sink, _lin, _lout) = scale_graph("eval=frame:w=t:h=ih", (64, 48));
        let mut f = yuv_frame(64, 48, NOPTS);
        f.pts = NOPTS;
        g.add_frame(src, &f).unwrap();
        match g.get_frame(sink).expect_err("NaN width must fail") {
            Error::InvalidArgument(msg) => {
                assert_eq!(msg, "Error when evaluating the expression 't'.")
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    // ---- the expression evaluator ---------------------------------------------

    fn ev(src: &str, v: &expr::Vars) -> f64 {
        let e = expr::parse(src).unwrap_or_else(|e| panic!("parse '{src}': {e}"));
        expr::eval(&e, v)
    }

    #[test]
    fn evaluator_precedence_and_pow_sign() {
        let v = expr::Vars::default();
        assert_eq!(ev("2+3*4", &v), 14.0);
        assert_eq!(ev("(2+3)*4", &v), 20.0);
        assert_eq!(ev("2*3+4*5", &v), 26.0);
        assert_eq!(ev("10-4-3", &v), 3.0); // left assoc
        // parse_factor's sign-multiplier trick (eval.c:587-611).
        assert_eq!(ev("-2^2", &v), -4.0);
        assert_eq!(ev("2^-3^2", &v), (2f64.powf(-3.0)).powf(2.0)); // (2^-3)^2
        assert_eq!(ev("(-2)^2", &v), 4.0);
        assert_eq!(ev("-(2^2)", &v), -4.0);
        // Whitespace is insignificant (stripped up front, eval.c:748-750).
        assert_eq!(ev(" 2 + 3 * 4 ", &v), 14.0);
        // Exponent spellings.
        assert_eq!(ev("1e2", &v), 100.0);
        assert_eq!(ev("1.5e-1", &v), 0.15);
        assert_eq!(ev(".5", &v), 0.5);
    }

    #[test]
    fn evaluator_division_and_modulo_semantics() {
        let v = expr::Vars::default();
        // eval.c:348: d2 ? d/d2 : d*INFINITY.
        assert_eq!(ev("1/0", &v), f64::INFINITY);
        assert_eq!(ev("-1/0", &v), f64::NEG_INFINITY);
        assert!(ev("0/0", &v).is_nan());
        // mod is a FUNCTION in av_expr (eval.c:485) and is eval.c:337's
        // floored modulo, NOT fmod: mod(-5.5, 2) == 0.5.
        assert_eq!(ev("mod(5.5,2)", &v), 1.5);
        assert_eq!(ev("mod(-5.5,2)", &v), 0.5);
        assert!(ev("mod(5,0)", &v).is_nan()); // floor(5*inf)=inf; 5-inf*0=NaN
        // '%' is NOT an operator in av_expr — a parse error.
        assert!(expr::parse("5.5%2").is_err());
    }

    #[test]
    fn evaluator_functions() {
        let v = expr::Vars::default();
        assert_eq!(ev("min(3,5)", &v), 3.0);
        assert_eq!(ev("max(3,5)", &v), 5.0);
        // C's raw ternaries: NaN on the left falls through to the right.
        assert_eq!(ev("min(0/0,5)", &v), 5.0);
        assert_eq!(ev("max(0/0,5)", &v), 5.0);
        assert_eq!(ev("floor(2.7)", &v), 2.0);
        assert_eq!(ev("ceil(2.1)", &v), 3.0);
        assert_eq!(ev("trunc(-2.7)", &v), -2.0);
        assert_eq!(ev("round(2.5)", &v), 3.0);
        assert_eq!(ev("round(-2.5)", &v), -3.0); // half away from zero
        assert_eq!(ev("abs(-3)", &v), 3.0);
        assert_eq!(ev("clip(10,0,255)", &v), 10.0);
        assert_eq!(ev("clip(300,0,255)", &v), 255.0);
        assert_eq!(ev("clip(-1,0,255)", &v), 0.0);
        assert!(ev("clip(0/0,0,255)", &v).is_nan()); // eval.c:223-230
        // Nested calls.
        assert_eq!(ev("max(min(1,2),3)", &v), 3.0);
        // Arity is verified (verify_expr, eval.c:701-737).
        assert!(expr::parse("min(1)").is_err());
        assert!(expr::parse("clip(1,2)").is_err());
        assert!(expr::parse("floor(1,2)").is_err());
        assert!(expr::parse("mod(1)").is_err());
        // Unknown functions / trailing junk / ';' chains are parse errors.
        assert!(expr::parse("foo(1)").is_err());
        assert!(expr::parse("iw(2)").is_err(), "var binds before the '('");
        // Whitespace is STRIPPED up front (eval.c:748-750): "1 2" is "12",
        // exactly like C.
        assert_eq!(ev("1 2", &v), 12.0);
        assert!(expr::parse("1;2").is_err());
        assert!(expr::parse("(1+2").is_err());
        assert!(expr::parse("2..5").is_err());
        assert!(expr::parse("0x10").is_err(), "hex is not in the subset");
    }

    #[test]
    fn evaluator_variables() {
        // The slots as scale_eval_dimensions fills them for a 64x48
        // yuv420p -> yuv420p link with SAR 2/1 (552-563).
        let v = expr::Vars {
            in_w: 64.0,
            in_h: 48.0,
            out_w: f64::NAN,
            out_h: f64::NAN,
            a: 64.0 / 48.0,
            sar: 2.0,
            dar: (64.0 / 48.0) * 2.0,
            hsub: 2.0,
            vsub: 2.0,
            ohsub: 2.0,
            ovsub: 2.0,
            n: 3.0,
            t: 0.04,
        };
        assert_eq!(ev("iw", &v), 64.0);
        assert_eq!(ev("in_w", &v), 64.0);
        assert_eq!(ev("ih", &v), 48.0);
        assert_eq!(ev("a", &v), 64.0 / 48.0);
        assert_eq!(ev("sar", &v), 2.0);
        assert_eq!(ev("dar", &v), 8.0 / 3.0);
        assert_eq!(ev("hsub", &v), 2.0);
        assert_eq!(ev("ovsub", &v), 2.0);
        assert_eq!(ev("n", &v), 3.0);
        assert_eq!(ev("t", &v), 0.04);
        // uses() covers the alias pairs via canonicalization.
        let e = expr::parse("ow + oh/2 + n + t + 1").unwrap();
        assert!(expr::uses(&e, "out_w") && expr::uses(&e, "out_h"));
        assert!(expr::uses(&e, "n") && expr::uses(&e, "t"));
        assert!(!expr::uses(&e, "sar"));
        let e = expr::parse("iw").unwrap();
        assert!(!expr::uses(&e, "n"));
    }

    // ---- query_formats ---------------------------------------------------------

    #[test]
    fn query_formats_declares_one_sided_lists() {
        // (463-521): the input side offers sws-supported INPUTS, the output
        // side sws-supported OUTPUTS (PixelFormat::ALL order); the color
        // axes are all-lists except output singletons when overridden.
        // Built WITHOUT config() so the declared halves are inspectable
        // (config's pick_format drops them).
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let scale = g
            .create_filter("scale", "out_color_matrix=bt709:out_range=tv")
            .unwrap();
        let sink = g.create_filter("buffersink", "").unwrap();
        let lin = g.link(src, 0, scale, 0).unwrap();
        let lout = g.link(scale, 0, sink, 0).unwrap();
        run_query_formats(&mut g, scale);
        let in_list = g.links[lin.0].outcfg.formats.expect("input declared");
        assert_eq!(
            g.fmt_lists[in_list as usize],
            vec![PixelFormat::Yuv420p, PixelFormat::Gray8]
        );
        let out_list = g.links[lout.0].incfg.formats.expect("output declared");
        assert_eq!(
            g.fmt_lists[out_list as usize],
            vec![
                PixelFormat::Yuv420p,
                PixelFormat::Gray8,
                PixelFormat::Rgb24,
                PixelFormat::Bgr24,
                PixelFormat::Rgba,
                PixelFormat::Bgra,
                PixelFormat::Argb,
                PixelFormat::Abgr,
            ]
        );
        let in_csp = g.links[lin.0].outcfg.color_spaces.expect("in csp");
        assert_eq!(g.csp_lists[in_csp as usize], formats::all_color_spaces());
        let in_rng = g.links[lin.0].outcfg.color_ranges.expect("in rng");
        assert_eq!(g.rng_lists[in_rng as usize], formats::all_color_ranges());
        // The out_color_matrix/out_range singletons (502-521).
        let out_csp = g.links[lout.0].incfg.color_spaces.expect("out csp");
        assert_eq!(g.csp_lists[out_csp as usize], vec![ColorSpace::Bt709]);
        let out_rng = g.links[lout.0].incfg.color_ranges.expect("out rng");
        assert_eq!(g.rng_lists[out_rng as usize], vec![ColorRange::Mpeg]);

        // An UNCONSTRAINED scale: all-lists on the color axes, and the
        // format lists' first element is yuv420p — the load-bearing ORDER
        // that makes an unconstrained scale graph negotiate the
        // guaranteed-supported same-format resize.
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let scale = g.create_filter("scale", "").unwrap();
        let sink = g.create_filter("buffersink", "").unwrap();
        let lin = g.link(src, 0, scale, 0).unwrap();
        let lout = g.link(scale, 0, sink, 0).unwrap();
        run_query_formats(&mut g, scale);
        let in_list = g.links[lin.0].outcfg.formats.expect("input declared");
        assert_eq!(
            g.fmt_lists[in_list as usize],
            vec![PixelFormat::Yuv420p, PixelFormat::Gray8]
        );
        let out_list = g.links[lout.0].incfg.formats.expect("output declared");
        assert_eq!(
            g.fmt_lists[out_list as usize].first(),
            Some(&PixelFormat::Yuv420p)
        );
        assert_eq!(
            g.csp_lists[g.links[lout.0].incfg.color_spaces.unwrap() as usize],
            formats::all_color_spaces()
        );
        assert_eq!(
            g.rng_lists[g.links[lout.0].incfg.color_ranges.unwrap() as usize],
            formats::all_color_ranges()
        );
    }

    #[test]
    fn config_props_probe_rejects_unsupported_pair() {
        // yuv420p -> gray8 is outside the Scaler's conversion matrix: the
        // config-time probe fails the graph with the Scaler's own text.
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let scale = g.create_filter("scale", "").unwrap();
        let sink = g
            .create_filter("buffersink", "pixel_formats=gray8")
            .unwrap();
        g.link(src, 0, scale, 0).unwrap();
        g.link(scale, 0, sink, 0).unwrap();
        let err = g.config().expect_err("unsupported pair must fail config");
        // The Scaler's own text (Gray8's canonical name is "gray").
        assert!(
            err.to_string().contains("cannot convert yuv420p to gray"),
            "got: {err}"
        );
    }

    // ---- data path ---------------------------------------------------------------

    #[test]
    fn same_format_resize_end_to_end() {
        // THE must-work path: 32x24 -> 64x48 yuv420p, constant color
        // preserved under any kernel, props copied via copy_props.
        let (mut g, src, sink, _lin, lout) = scale_graph("w=64:h=48", (32, 24));
        g.add_frame(src, &yuv_frame(32, 24, 7)).unwrap();
        let frames = drain(&mut g, sink);
        assert_eq!(frames.len(), 1);
        let out = &frames[0];
        assert_eq!(out.format, PixelFormat::Yuv420p);
        assert_eq!((out.width, out.height), (64, 48));
        assert_eq!(out.pts, 7);
        assert_eq!(out.duration, 1);
        for (p, v) in [(0usize, 100u8), (1, 120), (2, 130)] {
            assert!(
                out.plane(p).iter().all(|&b| b == v),
                "plane {p} constant {v}"
            );
        }
        assert_eq!(g.links[lout.0].frame_count_in, 1);
        assert_eq!(g.links[lout.0].frame_count_out, 1);
    }

    #[test]
    fn noop_passthrough_returns_the_input_frame() {
        // sws_is_noop (872-877): identity geometry + matching tags -> the
        // emitted frame IS the input frame (shared plane storage).
        let (mut g, src, sink, _lin, _lout) = scale_graph("", (64, 48));
        let sentinel = yuv_frame(64, 48, 5);
        g.add_frame(src, &sentinel).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &out.planes[0].buf,
            &sentinel.planes[0].buf
        ));
        assert_eq!(out.plane(0), sentinel.plane(0));
        assert_eq!(out.pts, 5);
        // A tag-only mismatch on the RANGE defeats the noop AND trips the
        // conversion gate (both compare in vs out range for yuv420p): the
        // limited<->full conversion the port does not implement errors —
        // see color_range_and_matrix_gates. A chroma-location "mismatch"
        // does NOT defeat the noop: the unconditional in_chroma_loc stamp
        // (vf_scale.c:829) rewrites the input tag to Unspecified BEFORE
        // the noop comparison, so C (and the port) pass the SAME frame
        // through with the tag corrected.
        let (mut g, src, sink, _lin, _lout) = scale_graph("", (64, 48));
        let mut f = yuv_frame(64, 48, 5);
        f.chroma_location = ChromaLocation::Left; // in_chroma_loc stamps Unspecified
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&out.planes[0].buf, &f.planes[0].buf),
            "noop pass-through shares the buffer (C returns the same AVFrame)"
        );
        assert_eq!(out.plane(0), f.plane(0), "bit-equal pixels");
        assert_eq!(out.chroma_location, ChromaLocation::Unspecified);
    }

    #[test]
    fn in_chroma_loc_quirk() {
        // C 832 stamps in_chroma_loc UNCONDITIONALLY: the default
        // Unspecified OVERWRITES a decoded Left tag.
        let (mut g, src, sink, _lin, _lout) = scale_graph("", (64, 48));
        let mut f = yuv_frame(64, 48, 0);
        f.chroma_location = ChromaLocation::Left;
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!(out.chroma_location, ChromaLocation::Unspecified);

        // With the option the tag survives (and matches the noop compare).
        let (mut g, src, sink, _lin, _lout) = scale_graph("in_chroma_loc=left", (64, 48));
        let mut f = yuv_frame(64, 48, 0);
        f.chroma_location = ChromaLocation::Center; // overridden either way
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!(out.chroma_location, ChromaLocation::Left);

        // out_chroma_loc is applied only when set (846-847); a resize (not
        // noop) carries it onto the output frame.
        let (mut g, src, sink, _lin, _lout) =
            scale_graph("w=32:h=ih:out_chroma_loc=center", (64, 48));
        g.add_frame(src, &yuv_frame(64, 48, 0)).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!(out.chroma_location, ChromaLocation::Center);
    }

    #[test]
    fn interl_is_metadata_only_with_flag_restore() {
        // interl=1 on a progressive frame: the INTERLACED flag is set for
        // the conversion and restored afterwards (835-838 + 886) — the
        // OUTPUT is not interlaced.
        let (mut g, src, sink, _lin, _lout) = scale_graph("interl=1", (64, 48));
        let f = yuv_frame(64, 48, 0);
        assert!(!f.flags.contains(FrameFlags::INTERLACED));
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert!(!out.flags.contains(FrameFlags::INTERLACED));

        // interl=0 on an interlaced frame: cleared for the conversion,
        // restored on the output.
        let (mut g, src, sink, _lin, _lout) = scale_graph("interl=0", (64, 48));
        let mut f = yuv_frame(64, 48, 0);
        f.flags = f.flags.union(FrameFlags::INTERLACED);
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert!(out.flags.contains(FrameFlags::INTERLACED));

        // interl=-1 leaves the flag alone.
        let (mut g, src, sink, _lin, _lout) = scale_graph("interl=auto", (64, 48));
        let mut f = yuv_frame(64, 48, 0);
        f.flags = f.flags.union(FrameFlags::INTERLACED);
        g.add_frame(src, &f).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert!(out.flags.contains(FrameFlags::INTERLACED));
    }

    #[test]
    fn frame_mode_expression_growth() {
        // eval=frame with w='iw+n' (766-801): the width grows by one each
        // frame (frame_count_out is C's pre-callback value).
        let (mut g, src, sink, _lin, _lout) = scale_graph("eval=frame:w=iw+n:h=ih", (64, 48));
        let mut widths = Vec::new();
        for pts in [0i64, 1, 2] {
            g.add_frame(src, &yuv_frame(64, 48, pts)).unwrap();
            for f in drain(&mut g, sink) {
                widths.push(f.width);
            }
        }
        assert_eq!(widths, vec![64, 65, 66]);
    }

    #[test]
    fn init_mode_freezes_dimensions_on_change() {
        // eval=init (default): a dimension change updates the link from the
        // frame (803-808) and re-runs config, but w_expr/h_expr were
        // frozen to the literals of the PREVIOUS size (780-793) — the
        // output keeps the old geometry.
        let (mut g, src, sink, _lin, lout) = scale_graph("w=iw:h=ih", (64, 48));
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (64, 48));
        // Same-size frame first (no reconfig).
        g.add_frame(src, &yuv_frame(64, 48, 0)).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!((out.width, out.height), (64, 48));
        // A 32x24 frame: link follows, output stays 64x48.
        g.add_frame(src, &yuv_frame(32, 24, 1)).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!((out.width, out.height), (64, 48));
        assert!(out.plane(0).iter().all(|&b| b == 100), "constant survives");
    }

    #[test]
    fn color_range_and_matrix_gates() {
        // yuv420p->yuv420p with a range mismatch: the Scaler has no
        // limited<->full kernel (Rust addition; C would convert). The sink
        // is constrained to tv (colorranges=1 = AVCOL_RANGE_MPEG).
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let scale = g.create_filter("scale", "").unwrap();
        let sink = g
            .create_filter("buffersink", "pixel_formats=yuv420p:colorranges=1")
            .unwrap();
        g.link(src, 0, scale, 0).unwrap();
        g.link(scale, 0, sink, 0).unwrap();
        g.config().unwrap();
        let mut f = yuv_frame(64, 48, 0);
        f.color_range = ColorRange::Jpeg; // pc in, tv out
        g.add_frame(src, &f).unwrap();
        match g.get_frame(sink).expect_err("range gate") {
            Error::Unsupported(msg) => assert_eq!(
                msg,
                "color range conversion pc -> tv is not supported by the swscale port"
            ),
            other => panic!("unexpected error: {other}"),
        }

        // Matrix gate: bt709 frame vs a bt470bg-constrained sink.
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let scale = g.create_filter("scale", "").unwrap();
        let sink = g
            .create_filter("buffersink", "pixel_formats=yuv420p:colorspaces=5")
            .unwrap(); // 5 = AVCOL_SPC_BT470BG
        g.link(src, 0, scale, 0).unwrap();
        g.link(scale, 0, sink, 0).unwrap();
        g.config().unwrap();
        let mut f = yuv_frame(64, 48, 0);
        f.color_space = ColorSpace::Bt709;
        g.add_frame(src, &f).unwrap();
        match g.get_frame(sink).expect_err("matrix gate") {
            Error::Unsupported(msg) => assert_eq!(
                msg,
                "color matrix conversion bt709 -> bt470bg is not supported by the swscale port"
            ),
            other => panic!("unexpected error: {other}"),
        }

        // Unspecified resolves to bt601 — the SAME class as bt470bg — so an
        // unconstrained frame passes into a bt470bg sink (no error).
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let scale = g.create_filter("scale", "").unwrap();
        let sink = g
            .create_filter("buffersink", "pixel_formats=yuv420p:colorspaces=5")
            .unwrap();
        g.link(src, 0, scale, 0).unwrap();
        g.link(scale, 0, sink, 0).unwrap();
        g.config().unwrap();
        g.add_frame(src, &yuv_frame(64, 48, 0)).unwrap();
        let out = g.get_frame(sink).expect("bt601 class == bt470bg class");
        assert_eq!((out.width, out.height), (64, 48));
    }

    #[test]
    fn out_range_option_forces_negotiated_range() {
        // out_range=tv (517-521) constrains the OUTPUT link to Mpeg: the
        // gate then fires for a full-range input frame. (The frame error
        // closes the link — the passing case needs a fresh graph.)
        let (mut g, src, sink, _lin, _lout) = scale_graph("out_range=tv", (64, 48));
        let mut f = yuv_frame(64, 48, 0);
        f.color_range = ColorRange::Jpeg;
        g.add_frame(src, &f).unwrap();
        match g.get_frame(sink).expect_err("range gate via out_range") {
            Error::Unsupported(msg) => assert!(msg.contains("color range conversion pc -> tv")),
            other => panic!("unexpected error: {other}"),
        }
        // An unspecified (resolves-to-tv) frame passes and the output
        // carries the constrained range.
        let (mut g, src, sink, _lin, _lout) = scale_graph("out_range=tv", (64, 48));
        g.add_frame(src, &yuv_frame(64, 48, 1)).unwrap();
        let out = g.get_frame(sink).expect("matching range passes");
        assert_eq!(out.color_range, ColorRange::Mpeg);
    }

    // ---- flags / algorithm ------------------------------------------------------

    #[test]
    fn flags_option_and_graph_default_algorithm() {
        // No flags option: init copies the GRAPH's algorithm (how the CLI's
        // engine/algorithm selection flows in).
        let mut g = FilterGraph::new();
        g.scale_algorithm = ScaleAlgorithm::Area;
        let n = g.alloc_filter("scale").unwrap();
        let mut imp = ScaleContext::default();
        FilterImpl::init(&mut imp, &mut g, n).unwrap();
        assert_eq!(imp.algorithm, ScaleAlgorithm::Area);

        // flags=<name> selects the kernel (409-413).
        for (flag, want) in [
            ("nearest", ScaleAlgorithm::Nearest),
            ("point", ScaleAlgorithm::Nearest), // alias
            ("lanczos", ScaleAlgorithm::Lanczos),
            ("spline", ScaleAlgorithm::Spline),
        ] {
            let mut g = FilterGraph::new();
            let n = g.alloc_filter("scale").unwrap();
            g.nodes[n.0].opts.entries = vec![("flags".to_string(), flag.to_string())];
            let mut imp = ScaleContext::default();
            FilterImpl::init(&mut imp, &mut g, n).unwrap();
            assert_eq!(imp.algorithm, want, "flags={flag}");
        }

        // An empty flags string is C's option default ("" = no override).
        let mut g = FilterGraph::new();
        g.scale_algorithm = ScaleAlgorithm::Gauss;
        let n = g.alloc_filter("scale").unwrap();
        g.nodes[n.0].opts.entries = vec![("flags".to_string(), String::new())];
        let mut imp = ScaleContext::default();
        FilterImpl::init(&mut imp, &mut g, n).unwrap();
        assert_eq!(imp.algorithm, ScaleAlgorithm::Gauss);

        // Kernels the Scaler does not implement error with the port's text.
        match scale_err("flags=fast_bilinear") {
            Error::InvalidArgument(msg) => {
                assert_eq!(
                    msg,
                    "Unable to parse \"flags\" option value \"fast_bilinear\" as scaler flags \
                     (supported: nearest, point, bilinear, bicubic, area, gauss, sinc, lanczos, \
                     spline)"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn flags_option_changes_the_pixels() {
        // A horizontal gradient downscaled 4x: lanczos and bicubic must not
        // agree everywhere (proof the flag reached the kernel).
        let mut widths: Vec<Vec<u8>> = Vec::new();
        for flag in ["bicubic", "lanczos"] {
            let (mut g, src, sink, _lin, _lout) =
                scale_graph(&format!("w=16:h=12:flags={flag}"), (64, 48));
            let mut f = Frame::alloc(PixelFormat::Yuv420p, 64, 48).unwrap();
            f.pts = 0;
            f.time_base = Rational::new(1, 25);
            for (y, row) in f.plane_mut(0).chunks_exact_mut(64).enumerate() {
                for (x, b) in row.iter_mut().enumerate() {
                    *b = (x * 4 % 251) as u8;
                    let _ = y;
                }
            }
            for b in f.plane_mut(1) {
                *b = 128;
            }
            for b in f.plane_mut(2) {
                *b = 128;
            }
            g.add_frame(src, &f).unwrap();
            let out = g.get_frame(sink).unwrap();
            widths.push(out.plane(0).to_vec());
        }
        let (bicubic, lanczos) = (&widths[0], &widths[1]);
        assert_ne!(bicubic, lanczos, "kernels must differ on a gradient");
    }

    // ---- defs / registry / auto-insertion ----------------------------------------

    #[test]
    fn registry_and_def_shape() {
        let def = filter_def("scale").expect("scale registered");
        assert!(std::ptr::eq(def, &SCALE_DEF));
        assert_eq!(def.name, "scale");
        assert_eq!(def.inputs.len(), 1);
        assert_eq!(def.outputs.len(), 1);
        assert_eq!(def.inputs[0].name, "default");
        assert_eq!(def.outputs[0].name, "default");
        assert!(!def.inputs[0].needs_writable);
        // C's ff_filter_opt_parse derives the shorthand from the AVOption
        // order skipping duplicate OFFSETs (width/height/s are aliases).
        assert_eq!(def.shorthand, &["w", "h", "flags", "interl", "size"][..]);
        // "scale" is in C's ff_filter_frame validation skip list
        // (avfilter.c:1075-1082) — the frame-changed path needs it.
        assert!(def.flags.contains(FilterFlags::ALLOWS_RECONFIGURE));
        let _ = (SCALE_DEF.make)();
    }

    #[test]
    fn auto_inserted_converter_end_to_end() {
        // The wave-3 proof: a yuv420p -> rgb24 mismatch auto-inserts
        // auto_scale_0 (avfiltergraph.c:611-641) and the frame converts
        // through the real engine.
        let mut g = FilterGraph::new();
        g.scale_engine = ScaleEngine::Cpu;
        let src = g
            .create_filter(
                "buffer",
                "video_size=16x16:pix_fmt=yuv420p:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let fmt = g.create_filter("format", "rgb24").unwrap();
        let sink = g.create_filter("buffersink", "").unwrap();
        g.link(src, 0, fmt, 0).unwrap();
        g.link(fmt, 0, sink, 0).unwrap();
        g.config().expect("auto converter settles negotiation");
        assert!(
            g.nodes.iter().any(|n| n.name == "auto_scale_0"),
            "converter inserted"
        );
        g.add_frame(src, &yuv_frame(16, 16, 3)).unwrap();
        let out = g.get_frame(sink).unwrap();
        assert_eq!(out.format, PixelFormat::Rgb24);
        assert_eq!((out.width, out.height), (16, 16));
        assert_eq!(out.pts, 3);
        // The identity-geometry yuv420p→RGB table converter turned the
        // constant-color frame into a constant-color rgb24 frame: PACKED,
        // so plane 0 is the repeating 3-byte pattern [R, G, B].
        let px = &out.plane(0);
        assert!(px.len() % 3 == 0, "packed rgb24: 3 bytes per pixel");
        let [r, g, b] = [px[0], px[1], px[2]];
        assert!(
            px.chunks_exact(3).all(|c| c == [r, g, b]),
            "constant in, constant out: every pixel equals [{r}, {g}, {b}]"
        );
        assert!(r != 0 || g != 0 || b != 0, "non-trivial conversion");
    }
}
