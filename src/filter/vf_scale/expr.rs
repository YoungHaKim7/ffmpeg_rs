//! Mini recursive-descent evaluator for the `av_expr` subset vf_scale
//! needs. Grammar (eval.c:560-690), LOWEST binder first:
//!
//! ```text
//! expr    := subexpr                      (';' sequences not ported)
//! subexpr := term (('+'|'-') term)*
//! term    := factor (('*'|'/') factor)*
//! factor  := ['-'|'+'] primary ('^' ['-'|'+'] primary)*   (LEFT fold)
//! primary := number | var | '(' expr ')' | func '(' expr (',' expr)* ')'
//! ```
//!
//! The C trick for `-2^2 == -4` (eval.c:587-611): a leading sign applies
//! to the WHOLE `^` chain (as a `-1` node multiplier), while each
//! exponent's own sign applies to that operand only. Whitespace is
//! stripped up front exactly like `av_expr_parse` (eval.c:748-750).
//!
/// One AST node (C's `AVExpr`, reduced to the retained types). Variable
/// and function names are canonicalized `&'static str`s (the alias pairs
/// `in_w`/`iw`, `out_w`/`ow`, ... share one slot), so [`uses`] matches a
/// canonical name and covers both aliases.

// ---------------------------------------------------------------------------
// The expression evaluator (libavutil/eval.c subset — see module doc)
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    Num(f64),
    Var(&'static str),
    /// `-x`: C models the leading sign as a node `value` multiplier; the
    /// negation node is the same thing.
    Neg(Box<Expr>),
    /// `+ - * / % ^` (the op char; evaluation at [`eval`]).
    Bin(char, Box<Expr>, Box<Expr>),
    /// `min/max/floor/ceil/trunc/round/abs/clip`.
    Call(&'static str, Vec<Expr>),
}

impl Default for Expr {
    fn default() -> Self {
        Expr::Num(0.0)
    }
}

/// The `var_values` slots of vf_scale.c:151 the retained subset uses
/// (vf_scale.c:552-563 fill). `n`/`t` live in the filter context (C keeps
/// them in the persistent array; only the frame path writes them).
#[derive(Clone, Copy, Debug, Default)]
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
    pub ohsub: f64,
    pub ovsub: f64,
    pub n: f64,
    pub t: f64,
}

/// Canonical variable slot of a name, if it is one of the retained 13
/// (`var_names`, vf_scale.c:46-59 minus the ref/scale2ref entries).
/// A known variable binds even when followed by `(` — eval.c checks
/// `const_names` before functions (eval.c:388-397) and then fails on the
/// leftover `(`, so `iw(2)` is a parse error, not a call.
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
        "ohsub" => "ohsub",
        "ovsub" => "ovsub",
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
    /// the `dB` special case: `-` negates, `+` is consumed and dropped.
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
                // verify_expr's arity rules (eval.c:701-737): min/max/mod
                // take exactly 2, the 1-arg functions exactly 1, clip 3.
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

    /// Decimal number with optional fraction and exponent (the
    /// non-hex, non-dB subset of `av_strtod`). C's `0x...` and `5dB`
    /// spellings are deliberately not accepted.
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

/// `av_expr_parse` (minus everything outside the subset): strip
/// whitespace, parse one expression, then reject trailing characters
/// (eval.c:768-772's "Invalid chars ... at the end of expression").
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

/// `av_expr_eval` over the retained node types. Division by zero is
/// eval.c:348's `d2 ? d/d2 : d*INFINITY`; `%` is eval.c:334's
/// `d - floor(d2 ? d/d2 : d*INFINITY) * d2` (a floored modulo — NOT
/// `fmod`: `-5.5 % 2` is `0.5`); `min`/`max` are C's raw `<`/`>`
/// ternaries, so NaN on the LEFT propagates to the right operand.
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
            "ohsub" => v.ohsub,
            "ovsub" => v.ovsub,
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
                    // eval.c:348: d2 ? d/d2 : d*INFINITY (5/0 = +inf,
                    // -5/0 = -inf, 0/0 = NaN) — never panics.
                    if d2 != 0.0 { d / d2 } else { d * f64::INFINITY }
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
                // e_mod (eval.c:337): a floored modulo — NOT fmod
                // (`mod(-5.5, 2)` is `0.5`).
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
                    // eval.c:223-230: NaN anywhere or min > max → NaN,
                    // else av_clipd.
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

/// `av_expr_count_vars` analog: does the tree reference the CANONICAL
/// variable `name`? (C counts per alias index; callers of old code
/// checked `vars[VAR_OUT_W] || vars[VAR_OW]` — the canonicalization
/// folds each alias pair into one name.)
pub(crate) fn uses(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Var(v) => *v == name,
        Expr::Num(_) => false,
        Expr::Neg(x) => uses(x, name),
        Expr::Bin(_, a, b) => uses(a, name) || uses(b, name),
        Expr::Call(_, args) => args.iter().any(|a| uses(a, name)),
    }
}
