//! `crop` video filter — port of `libavfilter/vf_crop.c` (all 398 lines).
//!
//! Crops a sub-rectangle out of every frame. `config_input` (the INPUT pad's
//! `config_props`, vf_crop.c:127-231) evaluates the `w`/`h` expressions once
//! against the link geometry, aligns them to the chroma grid (unless
//! `exact=1`, 185-188), parses (but does not yet evaluate) the `x`/`y`
//! expressions, derives the output SAR (`keep_aspect`, 199-205) and centers
//! the default rectangle (219-225). `config_output` (233-248) stamps the
//! output link's geometry + SAR. `filter_frame` (250-315) re-evaluates `x`/`y`
//! PER FRAME (variables `n`/`t` live there; `x` is even evaluated twice so it
//! may be expressed from `y`, 261-264), clamps the rectangle back into the
//! frame (269-280) and then performs pure POINTER SURGERY on the frame —
//! `data[i] += row*linesize + x*step` per plane with chroma shifts, 295-311 —
//! forwarding the SAME frame, zero-copy.
//!
//! ## The Rust translation of the pointer surgery
//!
//! The port has no raw pointers; a [`Plane`] is `{Arc buffer, offset,
//! linesize, rows}` and the sub-rectangle view is exactly C's pointer math
//! applied to `offset` (+ per-plane chroma shifts `(y>>vsub)` rows and
//! `(x*max_step)>>hsub` bytes, vf_crop.c:301-302), with `rows` set to the
//! cropped plane height. One divergence is forced by the port's storage
//! model: C's allocation always covers the moved pointers, but the port's
//! planes are EXACT-sized `Arc`s (`Plane::data()` bounds are
//! `offset+linesize*rows`), and a crop that touches the BOTTOM edge with a
//! nonzero `x` would push that bound past the buffer end by `x*step` bytes.
//! That one case re-buffers the plane compactly (copying only the visible
//! rows — the same pixels C would expose); every other crop is zero-copy and
//! SHARES the input buffer, like C.
//!
//! ## Dropped from C (each also noted at its site)
//!
//! * the HWACCEL branches — `hsub/vsub` forced to 1 (152-158), the
//!   crop-field output path (286-290): the port's pixel-format universe has
//!   no `AV_PIX_FMT_FLAG_HWACCEL` format.
//! * the `AV_PIX_FMT_FLAG_PAL` guard around the chroma loop (298): no PAL
//!   formats exist in the port (the palette "plane" is unrepresentable).
//! * `process_command` (317-351): the command queue is not ported
//!   (filter.rs module doc) — runtime `w`/`h`/`x`/`y` changes go through a
//!   new filter instance.
//! * `uninit` (102-110): the parsed `x`/`y` ASTs are plain values; `drop` is
//!   it.
//!
//! ## Expressions
//!
//! vf_crop keeps its OWN `var_names` table (vf_crop.c:40-55 — `in_w/iw`,
//! `in_h/ih`, `out_w/ow`, `out_h/oh`, `a`, `sar`, `dar`, `hsub`, `vsub`,
//! `x`, `y`, `n`, `t`) separate from vf_scale's, and so does the port: the
//! shared evaluator of C (`av_expr`) is parameterized by the names table,
//! but `vf_scale/expr.rs` bakes vf_scale's 13 slots into its `Vars` struct
//! (no `x`/`y`), and adding fields there would break vf_scale's exhaustive
//! struct literals — so this module carries its own instance of the same
//! grammar (see [`expr`]). Consolidating the two parsers into one
//! name-parameterized module is listed under wiring notes.
//!
//! Divergences kept consistent with the existing port: `VAR_A` is computed
//! in f64 (vf_scale.rs does the same for its `a`; C casts to float,
//! vf_crop.c:138) and `lrint` is reproduced exactly (round-to-nearest,
//! ties-to-even — the default FP rounding mode), NOT Rust's
//! half-away-from-zero `f64::round`.

use crate::{
    NOPTS, log_error, log_verbose,
    util::{
        error::{Error, Result},
        frame::Frame,
        pixdesc::{self, PixFmtDescriptor},
        rational::Rational,
    },
};

use super::{
    filter::{FilterDef, FilterFlags, FilterImpl, PadDef, PadRef, filter_frame},
    formats,
    graph::FilterGraph,
    link::NodeId,
};

// ---------------------------------------------------------------------------
// The expression evaluator (libavutil/eval.c subset — vf_crop's instance)
// ---------------------------------------------------------------------------

/// vf_crop's `av_expr` instance. The grammar is eval.c:560-690, identical to
/// `vf_scale/expr.rs` (same precedence ladder, same `-2^2 == -4` leading-sign
/// rule, same division-by-zero `d*INFINITY`, same floored `mod`); only the
/// VARIABLE TABLE differs — this is vf_crop.c:40-72's `var_names`/`enum
/// var_name`, which C passes to the shared parser. See module doc for why the
/// parser is duplicated rather than shared.
mod expr {
    /// One AST node (C's `AVExpr`, retained subset). Variable and function
    /// names are canonicalized `&'static str`s (`in_w`/`iw` etc. share one
    /// slot, mirroring the alias pairs of vf_crop.c:41-44).
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum Expr {
        Num(f64),
        Var(&'static str),
        /// `-x`: C models the leading sign as a node value multiplier.
        Neg(Box<Expr>),
        /// `+ - * / ^` (the op char; evaluation at [`eval`]).
        Bin(char, Box<Expr>, Box<Expr>),
        /// `min/max/mod/floor/ceil/trunc/round/abs/clip`.
        Call(&'static str, Vec<Expr>),
    }

    /// The `var_values` slots vf_crop uses (vf_crop.c:74-90's array, as a
    /// struct). `x`/`y`/`n`/`t` are only written by the frame path; `ow`/
    /// `oh` flow between the three config-time evaluation passes.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Vars {
        pub in_w: f64,
        pub in_h: f64,
        pub out_w: f64,
        pub out_h: f64,
        pub a: f64,
        pub sar: f64,
        pub dar: f64,
        pub hsub: f64,
        pub vsub: f64,
        pub x: f64,
        pub y: f64,
        pub n: f64,
        pub t: f64,
    }

    impl Default for Vars {
        fn default() -> Self {
            Vars {
                in_w: 0.0,
                in_h: 0.0,
                out_w: 0.0,
                out_h: 0.0,
                a: 0.0,
                sar: 0.0,
                dar: 0.0,
                hsub: 0.0,
                vsub: 0.0,
                x: f64::NAN,
                y: f64::NAN,
                n: 0.0,
                t: f64::NAN,
            }
        }
    }

    /// Canonical variable slot of a name, if it is one of vf_crop.c:40-55's
    /// names (`var_names`). A known variable binds even when followed by `(`
    /// (eval.c:388-397 checks const_names first) — `iw(2)` is a parse error.
    fn var_slot(name: &str) -> Option<&'static str> {
        Some(match name {
            "in_w" | "iw" => "in_w",
            "in_h" | "ih" => "in_h",
            "out_w" | "ow" => "out_w",
            "out_h" | "oh" => "out_h",
            "a" => "a",
            "sar" => "sar",
            "dar" => "dar",
            "hsub" => "hsub",
            "vsub" => "vsub",
            "x" => "x",
            "y" => "y",
            "n" => "n",
            "t" => "t",
            _ => return None,
        })
    }

    /// eval.c's recursion guard (`p.stack_index = 100`, 762-764).
    const MAX_DEPTH: usize = 100;

    struct Parser<'a> {
        s: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn peek(&self) -> u8 {
            if self.pos < self.s.len() {
                self.s[self.pos]
            } else {
                0
            }
        }

        fn eat(&mut self, c: u8) -> bool {
            if self.peek() == c {
                self.pos += 1;
                true
            } else {
                false
            }
        }

        fn parse_expr(&mut self, depth: usize) -> std::result::Result<Expr, String> {
            if depth > MAX_DEPTH {
                return Err("expression nesting too deep".into());
            }
            // The top-level ';' chain (e_last, eval.c:666-680) is not
            // ported: a ';' is an invalid character here.
            self.parse_subexpr(depth)
        }

        fn parse_subexpr(&mut self, depth: usize) -> std::result::Result<Expr, String> {
            let mut e = self.parse_term(depth)?;
            loop {
                let c = self.peek();
                if c == b'+' || c == b'-' {
                    self.pos += 1;
                    let rhs = self.parse_term(depth)?;
                    e = Expr::Bin(c as char, Box::new(e), Box::new(rhs));
                } else {
                    break;
                }
            }
            Ok(e)
        }

        fn parse_term(&mut self, depth: usize) -> std::result::Result<Expr, String> {
            let mut e = self.parse_factor(depth)?;
            loop {
                let c = self.peek();
                if c == b'*' || c == b'/' {
                    self.pos += 1;
                    let rhs = self.parse_factor(depth)?;
                    e = Expr::Bin(c as char, Box::new(e), Box::new(rhs));
                } else {
                    break;
                }
            }
            Ok(e)
        }

        /// `parse_factor` (eval.c:587-611): optional sign, primary, then a
        /// LEFT-folded `^` chain. The FIRST sign multiplies the whole chain;
        /// each subsequent operand's sign multiplies only that operand.
        fn parse_factor(&mut self, depth: usize) -> std::result::Result<Expr, String> {
            let neg = self.parse_sign();
            let mut e = self.parse_primary(depth)?;
            while self.eat(b'^') {
                let neg2 = self.parse_sign();
                let mut rhs = self.parse_primary(depth)?;
                if neg2 {
                    rhs = Expr::Neg(Box::new(rhs));
                }
                e = Expr::Bin('^', Box::new(e), Box::new(rhs));
            }
            if neg {
                e = Expr::Neg(Box::new(e));
            }
            Ok(e)
        }

        /// The sign scan of `parse_pow`/`parse_dB` (eval.c:565-585) without
        /// the `dB` special case.
        fn parse_sign(&mut self) -> bool {
            if self.peek() == b'-' {
                self.pos += 1;
                true
            } else if self.peek() == b'+' {
                self.pos += 1;
                false
            } else {
                false
            }
        }

        fn parse_primary(&mut self, depth: usize) -> std::result::Result<Expr, String> {
            let c = self.peek();
            if c == b'(' {
                self.pos += 1;
                let e = self.parse_expr(depth + 1)?;
                if !self.eat(b')') {
                    return Err("Missing ')' in expression".into());
                }
                return Ok(e);
            }
            if c.is_ascii_digit() || c == b'.' {
                return Ok(Expr::Num(self.parse_number()?));
            }
            if c.is_ascii_alphabetic() || c == b'_' {
                let start = self.pos;
                while self.peek().is_ascii_alphanumeric() || self.peek() == b'_' {
                    self.pos += 1;
                }
                let ident = std::str::from_utf8(&self.s[start..self.pos])
                    .expect("identifier scan produced ascii slice");
                // const_names take precedence over functions (eval.c:388-397).
                if let Some(v) = var_slot(ident) {
                    return Ok(Expr::Var(v));
                }
                if self.eat(b'(') {
                    let mut args = vec![self.parse_expr(depth + 1)?];
                    let mut commas = 0usize;
                    while self.eat(b',') {
                        commas += 1;
                        args.push(self.parse_expr(depth + 1)?);
                    }
                    if !self.eat(b')') {
                        return Err("Missing ')' or too many args in expression".into());
                    }
                    let name: &'static str = match ident {
                        "min" => "min",
                        "max" => "max",
                        "mod" => "mod",
                        "floor" => "floor",
                        "ceil" => "ceil",
                        "trunc" => "trunc",
                        "round" => "round",
                        "abs" => "abs",
                        "clip" => "clip",
                        _ => return Err(format!("Unknown function '{ident}' in expression")),
                    };
                    // verify_expr's arity rules (eval.c:701-737).
                    let arity = match name {
                        "min" | "max" | "mod" => 2,
                        "clip" => 3,
                        _ => 1,
                    };
                    if args.len() != arity || commas + 1 != arity {
                        return Err(format!(
                            "Incorrect number of arguments to '{ident}' (need {arity})"
                        ));
                    }
                    return Ok(Expr::Call(name, args));
                }
                // eval.c:427-430: not a constant and no '(' anywhere after.
                return Err(format!("Undefined constant or missing '(' in '{ident}'"));
            }
            Err(format!(
                "Invalid character '{}' in expression",
                char::from(c)
            ))
        }

        /// Decimal number with optional fraction and exponent (the non-hex,
        /// non-dB subset of `av_strtod` — as in vf_scale/expr.rs).
        fn parse_number(&mut self) -> std::result::Result<f64, String> {
            let start = self.pos;
            while self.peek().is_ascii_digit() {
                self.pos += 1;
            }
            if self.peek() == b'.' {
                self.pos += 1;
                while self.peek().is_ascii_digit() {
                    self.pos += 1;
                }
            }
            if self.pos == start {
                return Err("Invalid number in expression".into());
            }
            let mut end = self.pos;
            if self.peek() == b'e' || self.peek() == b'E' {
                let save = self.pos;
                self.pos += 1;
                if self.peek() == b'+' || self.peek() == b'-' {
                    self.pos += 1;
                }
                let digits_start = self.pos;
                while self.peek().is_ascii_digit() {
                    self.pos += 1;
                }
                if self.pos == digits_start {
                    self.pos = save; // a trailing 'e' belongs to no number
                } else {
                    end = self.pos;
                }
            }
            let text =
                std::str::from_utf8(&self.s[start..end]).expect("number scan produced ascii slice");
            text.parse::<f64>()
                .map_err(|_| format!("Invalid number '{text}' in expression"))
        }
    }

    /// `av_expr_parse` (subset): strip whitespace, parse one expression,
    /// reject trailing characters (eval.c:768-772).
    pub(crate) fn parse(src: &str) -> std::result::Result<Expr, String> {
        let stripped: String = src.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        if stripped.is_empty() {
            return Err("Empty expression".into());
        }
        let mut p = Parser {
            s: stripped.as_bytes(),
            pos: 0,
        };
        let e = p.parse_expr(0)?;
        if p.pos != p.s.len() {
            return Err(format!(
                "Invalid chars '{}' at the end of expression",
                &stripped[p.pos..]
            ));
        }
        Ok(e)
    }

    /// `av_expr_eval` — same semantics as vf_scale/expr.rs (division by
    /// zero is `d*INFINITY`, `mod` is floored, `min`/`max` propagate NaN
    /// from the left operand).
    pub(crate) fn eval(e: &Expr, v: &Vars) -> f64 {
        match e {
            Expr::Num(x) => *x,
            Expr::Var(name) => match *name {
                "in_w" => v.in_w,
                "in_h" => v.in_h,
                "out_w" => v.out_w,
                "out_h" => v.out_h,
                "a" => v.a,
                "sar" => v.sar,
                "dar" => v.dar,
                "hsub" => v.hsub,
                "vsub" => v.vsub,
                "x" => v.x,
                "y" => v.y,
                "n" => v.n,
                "t" => v.t,
                _ => f64::NAN,
            },
            Expr::Neg(x) => -eval(x, v),
            Expr::Bin(op, a, b) => {
                let d = eval(a, v);
                let d2 = eval(b, v);
                match op {
                    '+' => d + d2,
                    '-' => d - d2,
                    '*' => d * d2,
                    '/' => {
                        if d2 != 0.0 {
                            d / d2
                        } else {
                            d * f64::INFINITY
                        }
                    }
                    '^' => d.powf(d2),
                    _ => f64::NAN,
                }
            }
            Expr::Call(name, args) => {
                let x = || eval(&args[0], v);
                match *name {
                    "floor" => x().floor(),
                    "ceil" => x().ceil(),
                    "trunc" => x().trunc(),
                    "round" => x().round(),
                    "abs" => x().abs(),
                    "min" => {
                        let (a, b) = (eval(&args[0], v), eval(&args[1], v));
                        if a < b { a } else { b }
                    }
                    "mod" => {
                        let d = eval(&args[0], v);
                        let d2 = eval(&args[1], v);
                        let quot = if d2 != 0.0 { d / d2 } else { d * f64::INFINITY };
                        d - quot.floor() * d2
                    }
                    "max" => {
                        let (a, b) = (eval(&args[0], v), eval(&args[1], v));
                        if a > b { a } else { b }
                    }
                    "clip" => {
                        let x = eval(&args[0], v);
                        let min = eval(&args[1], v);
                        let max = eval(&args[2], v);
                        if x.is_nan() || min.is_nan() || max.is_nan() || min > max {
                            f64::NAN
                        } else if x < min {
                            min
                        } else if x > max {
                            max
                        } else {
                            x
                        }
                    }
                    _ => f64::NAN,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (imgutils analogs the port keeps private)
// ---------------------------------------------------------------------------

/// `av_image_fill_max_pixsteps(max_step, NULL, desc)` (imgutils.c:363-376):
/// per plane, the largest component `step` in bytes. The port's
/// `imgutils::fill_max_pixsteps` is private, and shared-file edits are off
/// limits for this zone, so the 8-line walk lives here (`pub(crate)` —
/// vf_transpose/vf_flip import it; promote into util::imgutils when that
/// file is next open).
pub(crate) fn fill_max_pixsteps(desc: &PixFmtDescriptor) -> [usize; 4] {
    let mut max_step = [0usize; 4];
    for i in 0..4 {
        let comp = &desc.comp[i];
        if comp.step as usize > max_step[comp.plane as usize] {
            max_step[comp.plane as usize] = comp.step as usize;
        }
    }
    max_step
}

/// `AV_CEIL_RSHIFT(a, b)` — ceil(a / 2^b) (the imgutils macro).
pub(crate) fn ceil_rshift(a: u32, b: u32) -> u32 {
    (a + (1 << b) - 1) >> b
}

/// C `lrint(d)` under the default rounding mode: round to NEAREST, ties to
/// EVEN — not Rust's `f64::round` (half away from zero): `lrint(2.5) == 2`,
/// `lrint(3.5) == 4`, `lrint(-0.5) == 0`.
pub(crate) fn lrint(d: f64) -> i32 {
    let f = d.floor();
    let c = f + 1.0;
    if c - d < d - f || (c - d == d - f && (c as i64) % 2 == 0) {
        c as i32
    } else {
        f as i32
    }
}

// ---------------------------------------------------------------------------
// Private context (vf_crop.c:74-90)
// ---------------------------------------------------------------------------

/// `CropContext` (vf_crop.c:74-90). C's `x_pexpr`/`y_pexpr` pointers become
/// `Option<Expr>` (None until `config_input` parses them, 193-196); the
/// `var_values[VAR_VARS_NB]` array becomes [`expr::Vars`].
pub struct CropContext {
    /// `x` — x offset of the non-cropped area (evaluated per frame).
    pub x: i32,
    /// `y` — y offset of the non-cropped area.
    pub y: i32,
    /// `w`/`h` — the cropped size.
    pub w: i32,
    pub h: i32,
    /// `out_sar` — the output SAR (`keep_aspect` adjusted).
    pub out_sar: Rational,
    /// `keep_aspect` — keep display aspect ratio when cropping.
    pub keep_aspect: bool,
    /// `exact` — exact cropping for subsampled formats.
    pub exact: bool,
    /// `max_step[4]` — max pixel step per plane in bytes.
    pub max_step: [usize; 4],
    /// `hsub`/`vsub` — chroma subsampling LOG2s (note: the EXPRESSION vars
    /// `hsub`/`vsub` carry `1<<log2`, vf_crop.c:141-142 vs 156-157).
    pub hsub: u32,
    pub vsub: u32,
    /// The four option strings (defaults per the AVOption table,
    /// vf_crop.c:357-366).
    pub x_expr: String,
    pub y_expr: String,
    pub w_expr: String,
    pub h_expr: String,
    /// Parsed `x`/`y` ASTs (`x_pexpr`/`y_pexpr`).
    x_pexpr: Option<expr::Expr>,
    y_pexpr: Option<expr::Expr>,
    /// `var_values` — persistent across frames (n/t/x/y updated per frame).
    var_values: expr::Vars,
}

impl Default for CropContext {
    /// C's zero-initialized priv + the AVOption defaults of vf_crop.c:357-366
    /// (`av_opt_set_defaults` runs before the user dict): w="iw", h="ih",
    /// x="(in_w-out_w)/2", y="(in_h-out_h)/2", keep_aspect=0, exact=0.
    fn default() -> Self {
        CropContext {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
            out_sar: Rational::UNKNOWN, // C zero-init 0/0
            keep_aspect: false,
            exact: false,
            max_step: [0; 4],
            hsub: 0,
            vsub: 0,
            x_expr: "(in_w-out_w)/2".to_string(),
            y_expr: "(in_h-out_h)/2".to_string(),
            w_expr: "iw".to_string(),
            h_expr: "ih".to_string(),
            x_pexpr: None,
            y_pexpr: None,
            var_values: expr::Vars::default(),
        }
    }
}

/// `normalize_double` (vf_crop.c:112-125): NaN → EINVAL with `*n` UNTOUCHED
/// (this is what makes a NaN `x` keep the previous/default rectangle);
/// out-of-i32 → clamped + EINVAL; else `*n = lrint(d)`. Returns Ok(()) for
/// the success case (C returns 0).
fn normalize_double(n: &mut i32, d: f64) -> Result<()> {
    if d.is_nan() {
        return Err(Error::InvalidArgument("nan".into()));
    }
    if d > i32::MAX as f64 || d < i32::MIN as f64 {
        *n = if d > i32::MAX as f64 {
            i32::MAX
        } else {
            i32::MIN
        };
        return Err(Error::InvalidArgument("out of range".into()));
    }
    *n = lrint(d);
    Ok(())
}

/// C `strtol(s, &p, 10)` core (the set_string_bool numeric arm) — same
/// shape as vf_scale.rs's private twin: optional whitespace/sign, decimal
/// digits, `(value, consumed)`.
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
    (n as i32, i)
}

/// `set_string_bool` (opt.c:226-254) for crop's BOOL options (`keep_aspect`,
/// `exact`, both range 0..1): name families, else a fully-consumed strtol.
fn parse_bool_val(name: &str, key: &str, val: &str) -> Result<bool> {
    let n = match val {
        "true" | "y" | "yes" | "enable" | "enabled" | "on" => 1,
        "false" | "n" | "no" | "disable" | "disabled" | "off" => 0,
        v => {
            let (n, consumed) = strtol_i32(v);
            if consumed != v.len() || !(0..=1).contains(&n) {
                log_error!(
                    Some(name),
                    "Unable to parse \"{key}\" option value \"{val}\" as boolean\n"
                );
                return Err(Error::InvalidArgument(format!(
                    "Unable to parse \"{key}\" option value \"{val}\" as boolean"
                )));
            }
            n
        }
    };
    Ok(n != 0)
}

// ---------------------------------------------------------------------------
// FilterImpl — init / query_formats / config_props / filter_frame
// ---------------------------------------------------------------------------

impl FilterImpl for CropContext {
    /// Option intake only (the AVOption table vf_crop.c:357-366): the four
    /// expression STRINGS (`out_w`/`w`, `out_h`/`h`, `x`, `y` — alias pairs
    /// share one field, last occurrence wins) and the two BOOLs
    /// (`keep_aspect`, `exact`). Everything else — the actual evaluation —
    /// happens in `config_input` when the link geometry exists. C applies
    /// AVOptions before init; unknown keys are pre-scanned FIRST so they
    /// beat every other failure, same as vf_scale.
    fn init(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        const KNOWN_KEYS: &[&str] = &["out_w", "w", "out_h", "h", "x", "y", "keep_aspect", "exact"];
        if let Some((key, _)) = g.nodes[node.0]
            .opts
            .entries
            .iter()
            .find(|(k, _)| !KNOWN_KEYS.contains(&k.as_str()))
        {
            return Err(Error::NotFound(format!("No such option: {key}")));
        }

        let name = g.nodes[node.0].name.clone();
        let entries = std::mem::take(&mut g.nodes[node.0].opts.entries);
        let mut leftovers = Vec::new();
        for (key, value) in entries {
            match key.as_str() {
                "out_w" | "w" => self.w_expr = value,
                "out_h" | "h" => self.h_expr = value,
                "x" => self.x_expr = value,
                "y" => self.y_expr = value,
                "keep_aspect" => self.keep_aspect = parse_bool_val(&name, &key, &value)?,
                "exact" => self.exact = parse_bool_val(&name, &key, &value)?,
                _ => leftovers.push((key, value)),
            }
        }
        g.nodes[node.0].opts.entries = leftovers;
        Ok(())
    }

    /// `query_formats` (vf_crop.c:92-100): reject bitstream (+ the internal
    /// flat-sub) formats — the port's universe contains neither flag, so the
    /// list is the full [`formats::all_pix_fmts`], set on BOTH pads with one
    /// handle (`ff_set_common_formats2` = `set_common_formats`).
    fn query_formats(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let list = g.alloc_pix_list(formats::all_pix_fmts());
        g.set_common_formats(node, list)?;
        Ok(())
    }

    /// `config_props`: crop wires BOTH pads — `config_input` on the input
    /// pad (vf_crop.c:127-231, run LAST by the link-config order) and
    /// `config_output` on the output pad (233-248, run as the source step
    /// of the downstream link's configuration).
    fn config_props(&mut self, g: &mut FilterGraph, node: NodeId, pad: PadRef) -> Result<()> {
        match pad {
            PadRef::In(0) => self.config_input(g, node),
            PadRef::Out(0) => self.config_output(g, node),
            _ => Ok(()),
        }
    }

    /// `filter_frame` (vf_crop.c:250-315) — see the method body for the
    /// per-stage anchors.
    fn filter_frame(
        &mut self,
        g: &mut FilterGraph,
        node: NodeId,
        _pad: usize,
        mut frame: Frame,
    ) -> Result<()> {
        let inlink = g.inlink(node, 0);

        // (258-264) per-frame variables: n = the PRE-frame frame_count_out
        // (the engine adjusts it around the callback, filter.rs), t = pts in
        // seconds; then y, then x, then x AGAIN — "It is necessary if x is
        // expressed from y".
        {
            let l = &g.links[inlink.0];
            self.var_values.n = l.frame_count_out as f64;
            self.var_values.t = if frame.pts == NOPTS {
                f64::NAN
            } else {
                frame.pts as f64 * l.time_base.to_f64()
            };
        }
        let (x_ast, y_ast) = (
            self.x_pexpr.clone().expect("config_input parsed x"),
            self.y_pexpr.clone().expect("config_input parsed y"),
        );
        self.var_values.y = expr::eval(&y_ast, &self.var_values);
        self.var_values.x = expr::eval(&x_ast, &self.var_values);
        self.var_values.x = expr::eval(&x_ast, &self.var_values);

        // (266-267) normalize — errors IGNORED here (C drops the return
        // value): a NaN keeps the previous x/y, an out-of-range value clamps
        // the target and is then re-clamped below.
        let _ = normalize_double(&mut self.x, self.var_values.x);
        let _ = normalize_double(&mut self.y, self.var_values.y);

        // (269-280) keep the rectangle inside the frame; the C unsigned-add
        // wrap is reproduced with wrapping_add.
        let (lw, lh) = (g.links[inlink.0].w as i32, g.links[inlink.0].h as i32);
        if self.x < 0 {
            self.x = 0;
        }
        if self.y < 0 {
            self.y = 0;
        }
        if (self.x as u32).wrapping_add(self.w as u32) > lw as u32 {
            self.x = lw - self.w;
        }
        if (self.y as u32).wrapping_add(self.h as u32) > lh as u32 {
            self.y = lh - self.h;
        }
        if !self.exact {
            self.x &= !((1 << self.hsub) - 1);
            self.y &= !((1 << self.vsub) - 1);
        }

        log_verbose!(
            Some("crop"),
            "n:{} t:{} x:{} y:{} x+w:{} y+h:{}\n",
            self.var_values.n,
            self.var_values.t,
            self.x,
            self.y,
            self.x + self.w,
            self.y + self.h
        );

        // (292-311) the pointer surgery — the SAME frame is forwarded,
        // planes re-windowed per C's data[i] math. The PAL guard (298) and
        // the HWACCEL branch (286-290) have no formats to fire on.
        frame.width = self.w as u32;
        frame.height = self.h as u32;
        let (x, y, w, h) = (self.x as usize, self.y as usize, self.w, self.h);
        let (hsub, vsub) = (self.hsub, self.vsub);
        for (i, p) in frame.planes.iter_mut().enumerate() {
            let (dy, dx, rows) = if i == 1 || i == 2 {
                // (301-302) chroma: rows shift y>>vsub, bytes (x*step)>>hsub.
                (
                    (y as u32 >> vsub) as usize,
                    (x * self.max_step[i]) >> hsub,
                    ceil_rshift(h as u32, vsub) as usize,
                )
            } else {
                // plane 0 (295-296) and the alpha plane 3 (309-310) take the
                // UNSHIFTED x/y.
                (y, x * self.max_step[i], h as usize)
            };
            p.offset += dy * p.linesize + dx;
            p.rows = rows;
            // Rust-only bounds guard (see module doc): C's allocation always
            // covers the moved pointers; the port's exact-sized plane Arcs
            // do not when the crop touches the bottom edge with x != 0. That
            // case re-buffers the plane compactly — same visible pixels.
            if p.offset + p.linesize * p.rows > p.buf.len() {
                let visible = ceil_rshift(w as u32, if i == 1 || i == 2 { hsub } else { 0 })
                    as usize
                    * self.max_step[i];
                let old_ls = p.linesize;
                let src = p.buf.clone();
                let mut fresh = vec![0u8; visible * rows];
                for r in 0..rows {
                    let so = p.offset + r * old_ls;
                    fresh[r * visible..(r + 1) * visible].copy_from_slice(&src[so..so + visible]);
                }
                p.buf = std::sync::Arc::from(fresh);
                p.offset = 0;
                p.linesize = visible;
            }
        }

        let outlink = g.outlink(node, 0);
        filter_frame(g, outlink, frame)
    }
}

impl CropContext {
    /// `config_input` (vf_crop.c:127-231) — the INPUT pad's config_props.
    fn config_input(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let name = g.nodes[node.0].name.clone();
        let inlink = g.inlink(node, 0);
        let l = &g.links[inlink.0];
        let fmt = l
            .format
            .expect("formats picked before config_props (query_formats round)");
        let desc = pixdesc::descriptor(fmt);

        // (136-148) fill the variable slots. VAR_HSUB/VAR_VSUB carry
        // 1<<log2 (141-142) — NOT the log2s stored on the context (156-157).
        // VAR_A in f64 (see module doc for the float-cast divergence).
        self.var_values.in_w = l.w as f64;
        self.var_values.in_h = l.h as f64;
        self.var_values.a = l.w as f64 / l.h as f64;
        self.var_values.sar = if l.sample_aspect_ratio.num != 0 {
            l.sample_aspect_ratio.to_f64()
        } else {
            1.0
        };
        self.var_values.dar = self.var_values.a * self.var_values.sar;
        self.var_values.hsub = (1u32 << desc.log2_chroma_w) as f64;
        self.var_values.vsub = (1u32 << desc.log2_chroma_h) as f64;
        self.var_values.x = f64::NAN;
        self.var_values.y = f64::NAN;
        self.var_values.out_w = f64::NAN;
        self.var_values.out_h = f64::NAN;
        self.var_values.n = 0.0;
        self.var_values.t = f64::NAN;

        // (150) max_step; (152-158) hsub/vsub (HWACCEL branch dropped).
        self.max_step = fill_max_pixsteps(desc);
        self.hsub = desc.log2_chroma_w as u32;
        self.vsub = desc.log2_chroma_h as u32;

        // (160-175) w, h, w AGAIN ("evaluate again ow as it may depend on
        // oh"). av_expr_parse_and_eval = parse + eval; a parse failure is
        // the fail_expr tail (228-230).
        let eval_expr = |src: &str, vars: &expr::Vars| -> Result<f64> {
            match expr::parse(src) {
                Ok(ast) => Ok(expr::eval(&ast, vars)),
                Err(_) => {
                    let msg = format!("Error when evaluating the expression '{src}'");
                    log_error!(Some(&name), "{msg}\n");
                    Err(Error::InvalidArgument(msg))
                }
            }
        };
        let res = eval_expr(&self.w_expr.clone(), &self.var_values)?;
        self.var_values.out_w = res;
        let res = eval_expr(&self.h_expr.clone(), &self.var_values)?;
        self.var_values.out_h = res;
        let res = eval_expr(&self.w_expr.clone(), &self.var_values)?;
        self.var_values.out_w = res;

        // (176-183) normalize both dimensions.
        if normalize_double(&mut self.w, self.var_values.out_w).is_err()
            || normalize_double(&mut self.h, self.var_values.out_h).is_err()
        {
            let msg = format!(
                "Too big value or invalid expression for out_w/ow or out_h/oh. Maybe the \
                 expression for out_w:'{}' or for out_h:'{}' is self-referencing.",
                self.w_expr, self.h_expr
            );
            log_error!(Some(&name), "{msg}\n");
            return Err(Error::InvalidArgument(msg));
        }

        // (185-188) align to the chroma grid unless exact.
        if !self.exact {
            self.w &= !((1 << self.hsub) - 1);
            self.h &= !((1 << self.vsub) - 1);
        }

        // (190-197) parse x/y. C returns a bare AVERROR(EINVAL) with NO log
        // here; the port attaches the parse detail (documented divergence).
        self.x_pexpr = Some(expr::parse(&self.x_expr.clone()).map_err(|_| {
            Error::InvalidArgument(format!(
                "Error when parsing the expression '{}'",
                self.x_expr
            ))
        })?);
        self.y_pexpr = Some(expr::parse(&self.y_expr.clone()).map_err(|_| {
            Error::InvalidArgument(format!(
                "Error when parsing the expression '{}'",
                self.y_expr
            ))
        })?);

        // (199-205) keep_aspect: out_sar = reduce(dar.num * h, dar.den * w)
        // with dar = sar * (w, h); else the input SAR verbatim.
        if self.keep_aspect {
            let dar = l.sample_aspect_ratio * Rational::new(l.w as i32, l.h as i32);
            self.out_sar = Rational::reduce(
                dar.num as i64 * self.h as i64,
                dar.den as i64 * self.w as i64,
                i32::MAX as i64,
            )
            .0;
        } else {
            self.out_sar = l.sample_aspect_ratio;
        }

        log_verbose!(
            Some(&name),
            "w:{} h:{} sar:{}/{} -> w:{} h:{} sar:{}/{}\n",
            l.w,
            l.h,
            l.sample_aspect_ratio.num,
            l.sample_aspect_ratio.den,
            self.w,
            self.h,
            self.out_sar.num,
            self.out_sar.den
        );

        // (211-217) the size sanity check.
        if self.w <= 0 || self.h <= 0 || self.w > l.w as i32 || self.h > l.h as i32 {
            let msg = format!(
                "Invalid too big or non positive size for width '{}' or height '{}'",
                self.w, self.h
            );
            log_error!(Some(&name), "{msg}\n");
            return Err(Error::InvalidArgument(msg));
        }

        // (219-225) the centered default rectangle ("required in the case
        // the first computed value for x/y is NAN").
        self.x = (l.w as i32 - self.w) / 2;
        self.y = (l.h as i32 - self.h) / 2;
        if !self.exact {
            self.x &= !((1 << self.hsub) - 1);
            self.y &= !((1 << self.vsub) - 1);
        }
        Ok(())
    }

    /// `config_output` (vf_crop.c:233-248) — the OUTPUT pad's config_props.
    /// The HWACCEL branch (238-241) is dropped: no hw formats.
    fn config_output(&mut self, g: &mut FilterGraph, node: NodeId) -> Result<()> {
        let outlink = g.outlink(node, 0);
        g.links[outlink.0].w = self.w as u32;
        g.links[outlink.0].h = self.h as u32;
        g.links[outlink.0].sample_aspect_ratio = self.out_sar;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Filter descriptor (vf_crop.c:371-398)
// ---------------------------------------------------------------------------

/// The shared single video pad (vf_crop.c:371-386).
static DEFAULT_PAD: PadDef = PadDef {
    name: "default",
    needs_writable: false,
};

/// `ff_vf_crop` (vf_crop.c:388-398): "Crop the input video."
///
/// * flags: crop is NOT in C's `ff_filter_frame` validation skip list
///   (avfilter.c:1075-1082), so `FilterFlags(0)` — the frame-vs-link asserts
///   stay on (and hold: filter_frame restamps width/height to the link's).
/// * shorthand: the `ff_filter_opt_parse` walk of the option table
///   (avfilter.c:855-902) in declaration order, skipping the duplicate
///   OFFSET aliases (`w` of `out_w`, `h` of `out_h`):
///   `out_w, out_h, x, y, keep_aspect, exact` — so `crop=640:480:10:20`
///   binds exactly like C.
/// * `AVFILTER_FLAG_SLICE_THREADS` is not modeled (single-threaded port).
pub static CROP_DEF: FilterDef = FilterDef {
    name: "crop",
    inputs: &[DEFAULT_PAD],
    outputs: &[DEFAULT_PAD],
    flags: FilterFlags(0),
    shorthand: &["out_w", "out_h", "x", "y", "keep_aspect", "exact"],
    make: || Box::new(CropContext::default()),
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::filter_def;
    use crate::filter::link::LinkId;
    use crate::util::pixfmt::PixelFormat;

    /// buffer(pix_fmt WxH, tb 1/25, sar) → filter(args) → buffersink,
    /// configured. Returns the graph, source, sink and the filter's two
    /// links.
    fn chain(
        filter: &str,
        args: &str,
        fmt: PixelFormat,
        size: (u32, u32),
        sar: &str,
    ) -> (FilterGraph, NodeId, NodeId, LinkId, LinkId) {
        let mut g = FilterGraph::new();
        let src = g
            .create_filter(
                "buffer",
                &format!(
                    "video_size={}x{}:pix_fmt={}:time_base=1/25:sar={}",
                    size.0,
                    size.1,
                    fmt.name(),
                    sar
                ),
            )
            .expect("buffer init");
        let f = g.create_filter(filter, args).expect("filter init");
        let sink = g.create_filter("buffersink", "").unwrap();
        let lin = g.link(src, 0, f, 0).unwrap();
        let lout = g.link(f, 0, sink, 0).unwrap();
        g.config().expect("graph config");
        (g, src, sink, lin, lout)
    }

    /// Deterministic per-plane pixel generator: byte `b` of the pixel at
    /// (row, col) of plane `p`. Geometry-independent, so source and expected
    /// crop/flip windows recompute from the same function.
    fn px(plane: usize, row: usize, col: usize, byte: usize) -> u8 {
        (row.wrapping_mul(31) + col.wrapping_mul(7) + plane * 13 + byte * 101 + 1) as u8
    }

    /// Fill a frame with [`px`] values (per-plane widths via the descriptor).
    fn fill(g: &mut Frame) {
        let desc = pixdesc::descriptor(g.format);
        let steps = fill_max_pixsteps(desc);
        for p in 0..g.planes.len() {
            let shift = if p == 1 || p == 2 {
                desc.log2_chroma_w as u32
            } else {
                0
            };
            let cols = ceil_rshift(g.width, shift) as usize;
            let ls = g.planes[p].linesize;
            for r in 0..g.planes[p].rows {
                for c in 0..cols {
                    for b in 0..steps[p] {
                        g.planes[p].data_mut()[r * ls + c * steps[p] + b] = px(p, r, c, b);
                    }
                }
            }
        }
    }

    fn frame_at(fmt: PixelFormat, w: u32, h: u32, pts: i64) -> Frame {
        let mut f = Frame::alloc(fmt, w, h).unwrap();
        f.pts = pts;
        f.duration = 1;
        f.time_base = Rational::new(1, 25);
        fill(&mut f);
        f
    }

    /// Push frames through and collect the sink output.
    fn run(g: &mut FilterGraph, src: NodeId, sink: NodeId, frames: Vec<Frame>) -> Vec<Frame> {
        for f in frames {
            g.add_frame(src, &f).unwrap();
        }
        g.close_source(src).unwrap();
        let mut out = Vec::new();
        while let Ok(f) = g.get_frame(sink) {
            out.push(f);
        }
        out
    }

    /// Expected byte of the CROPPED frame's plane p at (row, col, byte),
    /// given the source frame and the crop origin (ox, oy) — the chroma
    /// shifts are the C pointer math (vf_crop.c:301-302).
    fn crop_expect(
        src: &Frame,
        p: usize,
        row: usize,
        col: usize,
        byte: usize,
        ox: usize,
        oy: usize,
        step: usize,
    ) -> u8 {
        let desc = pixdesc::descriptor(src.format);
        let (hs, vs) = if p == 1 || p == 2 {
            (desc.log2_chroma_w as usize, desc.log2_chroma_h as usize)
        } else {
            (0, 0)
        };
        let srow = row + (oy >> vs);
        let scol_bytes = (ox * step) >> hs;
        let ls = src.planes[p].linesize;
        src.planes[p].data()[srow * ls + scol_bytes + col * step + byte]
    }

    /// Compare a cropped output frame against the source window.
    fn assert_crop(out: &Frame, src: &Frame, ox: usize, oy: usize) {
        let desc = pixdesc::descriptor(src.format);
        let steps = fill_max_pixsteps(desc);
        assert_eq!(out.format, src.format);
        assert!(out.width as usize + ox <= src.width as usize);
        assert!(out.height as usize + oy <= src.height as usize);
        for p in 0..out.planes.len() {
            let shift = if p == 1 || p == 2 {
                desc.log2_chroma_w as u32
            } else {
                0
            };
            let cols = ceil_rshift(out.width, shift) as usize;
            let ols = out.planes[p].linesize;
            for r in 0..out.planes[p].rows {
                for c in 0..cols {
                    for b in 0..steps[p] {
                        assert_eq!(
                            out.planes[p].data()[r * ols + c * steps[p] + b],
                            crop_expect(src, p, r, c, b, ox, oy, steps[p]),
                            "plane {p} ({r},{c},{b})"
                        );
                    }
                }
            }
        }
    }

    // ---- geometry / expressions --------------------------------------------

    #[test]
    fn crop_defaults_full_size() {
        // Default w=iw h=ih, x/y centered to 0: a no-op crop.
        let (mut g, src, sink, _lin, lout) = chain("crop", "", PixelFormat::Yuv420p, (8, 6), "1/1");
        assert_eq!(g.links[lout.0].w, 8);
        assert_eq!(g.links[lout.0].h, 6);
        let out = run(
            &mut g,
            src,
            sink,
            vec![frame_at(PixelFormat::Yuv420p, 8, 6, 0)],
        );
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].width, out[0].height), (8, 6));
        assert_crop(&out[0], &frame_at(PixelFormat::Yuv420p, 8, 6, 0), 0, 0);
    }

    #[test]
    fn crop_rect_math_and_chroma_offsets() {
        // 8x8 yuv420p, crop 4x4 centered: x=y=2, chroma offsets (2>>1)=1 row,
        // (2*1)>>1 = 1 byte.
        let src = frame_at(PixelFormat::Yuv420p, 8, 8, 0);
        let (mut g, s, k, _i, lout) = chain("crop", "w=4:h=4", PixelFormat::Yuv420p, (8, 8), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (4, 4));
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_crop(&out[0], &src, 2, 2);
        // Plane 0 offset surgery: 2*8 + 2*1 = 18 bytes into the source plane.
        assert_eq!(out[0].planes[0].offset, 18);
        // Chroma: (2>>1)*4 + (2*1)>>1 = 5.
        assert_eq!(out[0].planes[1].offset, 5);
        assert_eq!(out[0].planes[2].offset, 5);
        // Zero-copy: the plane buffers are SHARED with the source (same Arc).
        assert!(std::sync::Arc::ptr_eq(
            &out[0].planes[0].buf,
            &src.planes[0].buf
        ));
    }

    #[test]
    fn crop_sixteen_bit_and_packed() {
        // yuv420p10le (step 2): chroma x byte offset (x*2)>>1 == x.
        let src = frame_at(PixelFormat::Yuv420p10le, 8, 8, 3);
        let (mut g, s, k, _i, _l) =
            chain("crop", "w=4:h=4", PixelFormat::Yuv420p10le, (8, 8), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_crop(&out[0], &src, 2, 2);

        // rgb24 (packed, step 3): plane-0-only surgery, x*3 bytes.
        let src = frame_at(PixelFormat::Rgb24, 7, 5, 3);
        let (mut g, s, k, _i, _l) =
            chain("crop", "w=3:h=3:x=1:y=1", PixelFormat::Rgb24, (7, 5), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(out[0].planes.len(), 1);
        assert_eq!(out[0].planes[0].offset, 1 * 7 * 3 + 1 * 3);
        assert_crop(&out[0], &src, 1, 1);
    }

    #[test]
    fn crop_bottom_right_corner_rebuffers() {
        // x>0 with y+h == H: the windowed Plane would overrun the exact-sized
        // Arc — the guard re-buffers compactly (module doc). Same pixels.
        let src = frame_at(PixelFormat::Gray8, 8, 8, 0);
        let (mut g, s, k, _i, _l) =
            chain("crop", "w=7:h=8:x=1:y=0", PixelFormat::Gray8, (8, 8), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!((out[0].width, out[0].height), (7, 8));
        assert_eq!(out[0].planes[0].linesize, 7, "compact re-buffer");
        assert!(!std::sync::Arc::ptr_eq(
            &out[0].planes[0].buf,
            &src.planes[0].buf
        ));
        assert_crop(&out[0], &src, 1, 0);
    }

    #[test]
    fn crop_exact_mode() {
        // 8x8 yuv420p, w=3:h=3:exact=1 keeps odd sizes; x=(8-3)/2=2 (no mask).
        let src = frame_at(PixelFormat::Yuv420p, 8, 8, 0);
        let (mut g, s, k, _i, _l) = chain(
            "crop",
            "w=3:h=3:exact=1",
            PixelFormat::Yuv420p,
            (8, 8),
            "1/1",
        );
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!((out[0].width, out[0].height), (3, 3));
        assert_crop(&out[0], &src, 2, 2);

        // !exact (default): w=3→2, h=3→2; x=(8-2)/2=3 → masked to 2, y=2.
        let (mut g, s, k, _i, _l) = chain("crop", "w=3:h=3", PixelFormat::Yuv420p, (8, 8), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!((out[0].width, out[0].height), (2, 2));
        assert_crop(&out[0], &src, 2, 2);
    }

    #[test]
    fn crop_alpha_plane_unshifted() {
        // gbrap: plane 3 takes the UNSHIFTED x/y (vf_crop.c:308-311).
        let src = frame_at(PixelFormat::Gbrap, 8, 8, 0);
        let (mut g, s, k, _i, _l) = chain("crop", "w=4:h=4", PixelFormat::Gbrap, (8, 8), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_eq!(out[0].planes.len(), 4);
        assert_eq!(out[0].planes[3].offset, 2 * 8 + 2, "alpha plane unshifted");
        assert_crop(&out[0], &src, 2, 2);
    }

    // ---- per-frame x/y ------------------------------------------------------

    #[test]
    fn crop_per_frame_reval_keeps_rect_in_range() {
        // x = mod(n*3, iw-ow): frames n=0,1,2 → x=0,3,2 (mod(6,4)=2).
        let srcs: Vec<Frame> = (0..3)
            .map(|n| frame_at(PixelFormat::Gray8, 8, 8, n * 25))
            .collect();
        let (mut g, s, k, _i, _l) = chain(
            "crop",
            "w=4:h=8:x=mod(n*3\\,iw-ow):y=0",
            PixelFormat::Gray8,
            (8, 8),
            "1/1",
        );
        let out = run(&mut g, s, k, srcs.clone());
        assert_eq!(out.len(), 3);
        assert_crop(&out[0], &srcs[0], 0, 0); // n=0 → 0
        assert_crop(&out[1], &srcs[1], 3, 0); // n=1 → 3
        assert_crop(&out[2], &srcs[2], 2, 0); // n=2 → mod(6,4)=2
    }

    #[test]
    fn crop_out_of_range_clamped_per_frame() {
        // x=10 (constant, out of range) → clamped to link_w - w = 4.
        let (mut g, s, k, _i, _l) = chain(
            "crop",
            "w=4:h=4:x=10:y=-3",
            PixelFormat::Gray8,
            (8, 8),
            "1/1",
        );
        let src = frame_at(PixelFormat::Gray8, 8, 8, 0);
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_crop(&out[0], &src, 4, 0);

        // x = n*10: n=0 → 0; n=1 → 10 → clamped to 4.
        let (mut g, s, k, _i, _l) = chain(
            "crop",
            "w=4:h=4:x=n*10:y=0",
            PixelFormat::Gray8,
            (8, 8),
            "1/1",
        );
        let srcs = vec![
            frame_at(PixelFormat::Gray8, 8, 8, 0),
            frame_at(PixelFormat::Gray8, 8, 8, 25),
        ];
        let out = run(&mut g, s, k, srcs.clone());
        assert_crop(&out[0], &srcs[0], 0, 0);
        assert_crop(&out[1], &srcs[1], 4, 0);
    }

    #[test]
    fn crop_nan_keeps_previous_value() {
        // x = 0/0 = NaN every frame (eval.c division rule) → normalize_double
        // leaves the CENTERED default (8-4)/2 = 2 in place (vf_crop.c:219-221
        // + the ignored return at 266).
        let (mut g, s, k, _i, _l) = chain(
            "crop",
            "w=4:h=4:x=0/0:y=0",
            PixelFormat::Gray8,
            (8, 8),
            "1/1",
        );
        let srcs = vec![
            frame_at(PixelFormat::Gray8, 8, 8, 0),
            frame_at(PixelFormat::Gray8, 8, 8, 25),
        ];
        let out = run(&mut g, s, k, srcs.clone());
        assert_crop(&out[0], &srcs[0], 2, 0);
        assert_crop(&out[1], &srcs[1], 2, 0);
    }

    #[test]
    fn crop_x_expressed_from_y() {
        // y=1, x=y+2 → 3 (the double x evaluation, vf_crop.c:263-264).
        let (mut g, s, k, _i, _l) = chain(
            "crop",
            "w=4:h=4:y=1:x=y+2",
            PixelFormat::Gray8,
            (8, 8),
            "1/1",
        );
        let src = frame_at(PixelFormat::Gray8, 8, 8, 0);
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_crop(&out[0], &src, 3, 1);
    }

    #[test]
    fn crop_t_variable() {
        // t = pts * q2d(1/25): pts 25 → t=1 → x=1.
        let (mut g, s, k, _i, _l) =
            chain("crop", "w=4:h=8:x=t:y=0", PixelFormat::Gray8, (8, 8), "1/1");
        let src = frame_at(PixelFormat::Gray8, 8, 8, 25);
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_crop(&out[0], &src, 1, 0);
    }

    // ---- expressions: ow/oh cross-talk, sar/dar/hsub -------------------------

    #[test]
    fn crop_w_h_cross_evaluation() {
        // w = oh*2 (the third pass sees oh), h = ih/2 → 8x8 → 8x4.
        let (g, _s, _k, _i, lout) =
            chain("crop", "w=oh*2:h=ih/2", PixelFormat::Gray8, (8, 8), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (8, 4));

        // w = iw/3 with a/sar/dar in play.
        let (g, _s, _k, _i, lout) = chain(
            "crop",
            "w=iw/3:h=ih/4:x=(iw-ow)/2:y=(ih-oh)/2",
            PixelFormat::Gray8,
            (9, 8),
            "2/1",
        );
        // w: 9/3 = 3, h: 8/4 = 2; x = (9-3)/2 = 3, y = (8-2)/2 = 3.
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (3, 2));
    }

    #[test]
    fn crop_keep_aspect_sar() {
        // dar = sar*(w,h) = (2/1)*(8/8) = 2/1; out_sar = reduce(2*h', 1*w')
        // with h'=2, w'=4 → reduce(4,4) = 1/1 (vf_crop.c:199-203).
        let (g, _s, _k, _i, lout) = chain(
            "crop",
            "w=4:h=2:keep_aspect=1",
            PixelFormat::Gray8,
            (8, 8),
            "2/1",
        );
        assert_eq!(g.links[lout.0].sample_aspect_ratio, Rational::new(1, 1));

        // Without keep_aspect the input SAR is passed through verbatim.
        let (g, _s, _k, _i, lout) = chain("crop", "w=4:h=2", PixelFormat::Gray8, (8, 8), "2/1");
        assert_eq!(g.links[lout.0].sample_aspect_ratio, Rational::new(2, 1));

        // The FRAME's own sar is untouched by crop (only the link's changes).
        let (mut g, s, k, _i, _l) = chain("crop", "w=4:h=2", PixelFormat::Gray8, (8, 8), "2/1");
        let mut f = frame_at(PixelFormat::Gray8, 8, 8, 0);
        f.sample_aspect_ratio = Rational::new(9, 4);
        let out = run(&mut g, s, k, vec![f]);
        assert_eq!(out[0].sample_aspect_ratio, Rational::new(9, 4));
    }

    #[test]
    fn crop_expression_variables_exposed() {
        // The EXPRESSION vars hsub/vsub carry 1<<log2 (yuv420p → 2,
        // vf_crop.c:141-142): w = 8/2 = 4, h = 6/2 = 3 → h aligned to 2.
        // a = w/h and dar = a*sar are exercised through x/y (×0).
        let (g, _s, _k, _i, lout) = chain(
            "crop",
            "w=iw/hsub:h=ih/vsub:x=dar*0:y=a*0",
            PixelFormat::Yuv420p,
            (8, 6),
            "1/2",
        );
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (4, 2));
    }

    // ---- errors (verbatim texts) ---------------------------------------------

    #[test]
    fn crop_error_texts() {
        // (211-216) the size sanity check, exact texts: the message carries
        // the EVALUATED w/h.
        for (args, msg) in [
            (
                "w=iw+1",
                "Invalid too big or non positive size for width '9' or height '8'",
            ),
            (
                "h=ih+1",
                "Invalid too big or non positive size for width '8' or height '9'",
            ),
            (
                "w=0",
                "Invalid too big or non positive size for width '0' or height '8'",
            ),
            (
                "w=-3",
                "Invalid too big or non positive size for width '-3' or height '8'",
            ),
        ] {
            let mut g = FilterGraph::new();
            let src = g
                .create_filter(
                    "buffer",
                    "video_size=8x8:pix_fmt=gray:time_base=1/25:sar=1/1",
                )
                .unwrap();
            let f = g.create_filter("crop", args).expect("init ok");
            let sink = g.create_filter("buffersink", "").unwrap();
            g.link(src, 0, f, 0).unwrap();
            g.link(f, 0, sink, 0).unwrap();
            match g.config().unwrap_err() {
                Error::InvalidArgument(m) => assert_eq!(m, msg, "args '{args}'"),
                other => panic!("unexpected error for '{args}': {other}"),
            }
        }

        // (176-183) the too-big/NaN dimension text with the actual
        // expressions embedded — both the overflow and the NaN shape (w=x
        // reads the still-NaN x slot) land here.
        for args in ["w=iw*1e9:h=ih", "w=x:h=ih"] {
            let mut g = FilterGraph::new();
            let src = g
                .create_filter(
                    "buffer",
                    "video_size=8x8:pix_fmt=gray:time_base=1/25:sar=1/1",
                )
                .unwrap();
            let (we, he) = (
                args.split(':').next().unwrap().rsplit('=').next().unwrap(),
                "ih",
            );
            let f = g.create_filter("crop", args).expect("init ok");
            let sink = g.create_filter("buffersink", "").unwrap();
            g.link(src, 0, f, 0).unwrap();
            g.link(f, 0, sink, 0).unwrap();
            match g.config().unwrap_err() {
                Error::InvalidArgument(m) => assert_eq!(
                    m,
                    format!(
                        "Too big value or invalid expression for out_w/ow or out_h/oh. Maybe \
                         the expression for out_w:'{we}' or for out_h:'{he}' is \
                         self-referencing."
                    ),
                    "args '{args}'"
                ),
                other => panic!("unexpected error for '{args}': {other}"),
            }
        }
    }

    #[test]
    fn crop_expression_parse_errors() {
        // (228-230) w/h: "Error when evaluating the expression '%s'" — the
        // evaluation happens in config_input, so the failure lands at
        // graph-config time (C's avfilter_graph_config likewise).
        for (args, bad_expr) in [
            ("w=bogus", "bogus"),
            ("w=", ""),
            ("h=nosuchfunc(1)", "nosuchfunc(1)"),
        ] {
            let mut g = FilterGraph::new();
            let src = g
                .create_filter(
                    "buffer",
                    "video_size=8x8:pix_fmt=gray:time_base=1/25:sar=1/1",
                )
                .unwrap();
            let f = g
                .create_filter("crop", args)
                .expect("init stores the string");
            let sink = g.create_filter("buffersink", "").unwrap();
            g.link(src, 0, f, 0).unwrap();
            g.link(f, 0, sink, 0).unwrap();
            match g.config().unwrap_err() {
                Error::InvalidArgument(m) => assert_eq!(
                    m,
                    format!("Error when evaluating the expression '{bad_expr}'"),
                    "args '{args}'"
                ),
                other => panic!("unexpected error for '{args}': {other}"),
            }
        }
        // (193-197) x/y parse failure surfaces at CONFIG time (C parses the
        // expressions in config_input, not init): bare EINVAL without logging
        // in C; the port attaches a descriptive message (documented
        // divergence).
        let mut g = FilterGraph::new();
        let src = g
            .create_filter(
                "buffer",
                "video_size=8x8:pix_fmt=gray:time_base=1/25:sar=1/1",
            )
            .unwrap();
        let f = g.create_filter("crop", "x=nosuchfunc(2)").expect("init ok");
        let sink = g.create_filter("buffersink", "").unwrap();
        g.link(src, 0, f, 0).unwrap();
        g.link(f, 0, sink, 0).unwrap();
        match g.config().unwrap_err() {
            Error::InvalidArgument(m) => {
                assert_eq!(m, "Error when parsing the expression 'nosuchfunc(2)'")
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn crop_unknown_and_bool_options() {
        let mut g = FilterGraph::new();
        match g.create_filter("crop", "zzz=1").unwrap_err() {
            Error::NotFound(m) => assert_eq!(m, "No such option: zzz"),
            other => panic!("unexpected error: {other}"),
        }
        // BOOL parse failure (set_string_bool text).
        let mut g = FilterGraph::new();
        match g.create_filter("crop", "exact=perhaps").unwrap_err() {
            Error::InvalidArgument(m) => {
                assert_eq!(
                    m,
                    "Unable to parse \"exact\" option value \"perhaps\" as boolean"
                )
            }
            other => panic!("unexpected error: {other}"),
        }
        // Alias + last-occurrence-wins + positional shorthand.
        let (g, _s, _k, _i, lout) =
            chain("crop", "out_w=6:w=4:h=4", PixelFormat::Gray8, (8, 8), "1/1");
        assert_eq!(g.links[lout.0].w, 4, "w (alias) overrides out_w, last wins");
        let (g, _s, _k, _i, lout) = chain("crop", "6:4:1:2", PixelFormat::Gray8, (8, 8), "1/1");
        assert_eq!((g.links[lout.0].w, g.links[lout.0].h), (6, 4));
        let src = frame_at(PixelFormat::Gray8, 8, 8, 0);
        let (mut g, s, k, _i, _l) = chain("crop", "4:4:1:2", PixelFormat::Gray8, (8, 8), "1/1");
        let out = run(&mut g, s, k, vec![src.clone()]);
        assert_crop(&out[0], &src, 1, 2);
    }

    // ---- lrint / normalize_double units ------------------------------------

    #[test]
    fn lrint_ties_to_even() {
        assert_eq!(lrint(2.5), 2);
        assert_eq!(lrint(3.5), 4);
        assert_eq!(lrint(-0.5), 0);
        assert_eq!(lrint(-1.5), -2);
        assert_eq!(lrint(2.7), 3);
        assert_eq!(lrint(2.3), 2);
        assert_eq!(lrint(-2.7), -3);
    }

    #[test]
    fn normalize_double_nan_leaves_target() {
        let mut n = 7;
        assert!(normalize_double(&mut n, f64::NAN).is_err());
        assert_eq!(n, 7, "NaN leaves the target untouched");
        assert!(normalize_double(&mut n, 1.6).is_ok());
        assert_eq!(n, 2);
        assert!(normalize_double(&mut n, 3e9).is_err());
        assert_eq!(n, i32::MAX, "out of range clamps");
    }

    // ---- expression evaluator units ----------------------------------------

    #[test]
    fn crop_expr_evaluator() {
        let v = expr::Vars {
            in_w: 8.0,
            in_h: 6.0,
            out_w: 4.0,
            out_h: 4.0,
            a: 8.0 / 6.0,
            sar: 2.0,
            dar: 8.0 / 3.0,
            hsub: 1.0,
            vsub: 1.0,
            x: 1.0,
            y: 2.0,
            n: 3.0,
            t: 12.5,
        };
        let ev = |src: &str| {
            let ast = expr::parse(src).unwrap_or_else(|e| panic!("parse '{src}': {e}"));
            expr::eval(&ast, &v)
        };
        assert_eq!(ev("(in_w-out_w)/2"), 2.0);
        assert_eq!(ev("iw-ow"), 4.0);
        assert_eq!(ev("y+2"), 4.0);
        assert_eq!(ev("mod(n*3, iw-ow)"), 1.0);
        assert_eq!(ev("min(t, x)"), 1.0);
        assert_eq!(ev("clip(x, 0, iw-ow)"), 1.0);
        assert!(ev("0/0").is_nan(), "0/0 = NaN (d*INFINITY rule)");
        assert_eq!(ev("5/0"), f64::INFINITY);
        // -2^2 == -4: the leading sign applies to the whole ^ chain.
        assert_eq!(ev("-2^2"), -4.0);
        assert_eq!(ev("2^3^2"), 64.0, "^ is LEFT-folded (as in vf_scale)");
        assert!(expr::parse("x(").is_err());
        assert!(expr::parse("in_w 2").is_err());
        assert!(expr::parse("").is_err());
        assert!(expr::parse("foo(1)").is_err());
        assert!(expr::parse("min(1)").is_err());
    }

    // ---- def / registry ------------------------------------------------------

    #[test]
    fn crop_def_shape() {
        let def = filter_def("crop").expect("crop registered");
        assert!(std::ptr::eq(def, &CROP_DEF));
        assert_eq!(def.name, "crop");
        assert_eq!(def.inputs.len(), 1);
        assert_eq!(def.outputs.len(), 1);
        assert_eq!(def.inputs[0].name, "default");
        assert_eq!(def.flags, FilterFlags(0));
        assert_eq!(
            def.shorthand,
            &["out_w", "out_h", "x", "y", "keep_aspect", "exact"][..]
        );
    }

    #[test]
    fn crop_query_formats_all() {
        // vf_crop.c:92-100 over the port's universe = the full all-list.
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let f = g.create_filter("crop", "").unwrap();
        let sink = g.alloc_test_sink();
        let lin = g.link(src, 0, f, 0).unwrap();
        let lout = g.link(f, 0, sink, 0).unwrap();
        let mut imp = g.nodes[f.0].imp.take().expect("imp present");
        imp.query_formats(&mut g, f).unwrap();
        g.nodes[f.0].imp = Some(imp);
        let a = g.links[lin.0].outcfg.formats.expect("declared");
        let b = g.links[lout.0].incfg.formats.expect("declared");
        assert_eq!(a, b, "one list on both pads");
        assert_eq!(g.fmt_lists[a as usize], formats::all_pix_fmts());
    }
}
