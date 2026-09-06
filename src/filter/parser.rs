//! The filtergraph-description parser — port of `libavfilter/graphparser.c`
//! (the FFmpeg 7.1+ **segment-based rewrite**, all 1042 lines) plus its three
//! support sites: `av_get_token` (`libavutil/avstring.c:143-177`),
//! `is_key_char`/`get_key`/`av_opt_get_key_value` (`libavutil/opt.c:1915-1973`)
//! and `ff_filter_opt_parse` (`libavfilter/avfilter.c:853-904`).
//!
//! The ported pipeline (each stage = one function below):
//!
//! ```text
//! graph_parse_ptr (graphparser.c:920-1042)
//!   |= segment_parse          (460-514)  sws_flags prefix + chains
//!   |    |= chain_parse       (403-458)  filters split on ',' / ended by ';'
//!   |    |    |= filter_parse (338-401)  [labels] name@inst=opts [labels]
//!   |    |    |    |= linklabels_parse (292-336) '[' label ']' runs
//!   |    |    |    |    |= parse_link_name (43-69)
//!   |    |    |    |= filter_opt_parse  (avfilter.c:853-904)
//!   |    |= parse_sws_flags   (115-136)
//!   |= segment_create_filters (516-575)  alloc + "Parsed_name_idx" naming
//!   |= segment_apply_opts_and_init (586-641, fused) options + init
//!   |= [in]/[out] injection   (952-971, parse_ptr only)
//!   |= segment_apply -> segment_link (814-854)
//!   |    |= link_inputs       (702-745)  find_linklabel(output=true)
//!   |    |= link_outputs      (747-812)  find_linklabel(output=false) +
//!   |    |                    unlabeled next-filter chaining (785-802)
//!   |    |= find_linklabel    (643-676)  FORWARD search, inclusive start
//!   |= user open-input/output matching (978-1018)  extract_inout (86-101)
//! ```
//!
//! ## Entries
//!
//! * [`parse_ptr`] — the `graph.rs` stub signature: fresh empty user lists,
//!   so the returned lists are exactly the parsed graph's unmatched open
//!   pads (labeled ones carry their label, the injected "in"/"out" where
//!   applicable).
//! * [`graph_parse_ptr`] — the FULL C shape: caller-supplied
//!   `open_inputs`/`open_outputs` lists are matched against (named entries
//!   consumed and linked; C's `AVFilterInOut **` in/out parameters become
//!   `&mut Vec<InOut>`).
//! * [`graph_parse2`] — no injection, no user matching: the open pads of the
//!   parsed graph come back unnamed.
//!
//! ## Not ported (documented per C site)
//!
//! * `avfilter_graph_parse` (graphparser.c:164-225) — the LEGACY entry that
//!   renames the first unnamed input to "in" / last unnamed output to "out"
//!   and hard-errors on any other unnamed pad ("Not enough inputs specified
//!   for the \"%s\" filter." / "Invalid filterchain containing an unlabelled
//!   output pad: \"%s\""). Both modern entries cover its use; if ever needed
//!   it is a thin wrapper over the same segment machinery.
//! * ENOMEM branches everywhere (allocation is infallible in Rust), the
//!   `flags != 0 -> AVERROR(ENOSYS)` guards of the `avfilter_graph_segment_*`
//!   API (no flags parameters), and `seg->graph` back-pointers (the graph is
//!   a function argument here).
//! * `ff_filter_opt_parse`'s "Unable to parse '%s': %s" (avfilter.c:883) —
//!   the non-EINVAL (ENOMEM-only) branch of `av_opt_get_key_value`.
//! * Audio: `graphparser.c` is media-agnostic — no audio branches exist in
//!   it; the port's video-only scope changes nothing here.
//!
//! ## Divergences from C semantics
//!
//! * Instance-name truncation: C `snprintf(name, 64, ...)` (graphparser.c:549)
//!   cuts `Parsed_<filter>_<idx>` names at 63 bytes; Rust `String`s are
//!   unbounded. No ported filter name gets near the limit.
//! * Options/init staging: C's `avfilter_graph_segment_apply_opts` applies
//!   options to ALL filters before `avfilter_graph_segment_init` initializes
//!   ANY (an option error on a later filter outranks an init error on an
//!   earlier one); the Rust port FUSES the two stages per filter (its
//!   `init_filter` IS the option applier), so the first broken filter in
//!   order wins. Only affects which message surfaces when several filters
//!   are broken. See [`segment_apply_opts_and_init`].
//! * `AVOption`/`AVDictionary` reflection has no Rust counterpart: the
//!   leftover ("unknown option") case is detected as `init_filter` returning
//!   `Error::NotFound` while the node's `opts` still holds entries.
//! * The scale `scale_sws_opts` pre-apply (graphparser.c:557-565) uses
//!   [`filter_opt_parse`] instead of C's `av_set_options_string` (whose
//!   shorthand is empty — explicit `key=value` only): a value without '='
//!   would fill `w` positionally here where C errors. Dead code until the
//!   `scale` filter is registered (wave 2C).
//! * Error `Display` texts are the crate `Error` enum's, not `av_err2str`;
//!   the LOG lines carry C's texts verbatim. Code matching on text should
//!   match the logged forms.
//! * `get_token`'s output is always valid UTF-8 for valid UTF-8 input (every
//!   delimiter, quote, backslash and whitespace byte is ASCII, so none can
//!   split a multibyte char); a hypothetical invalid result is passed through
//!   `from_utf8_lossy` rather than panicking.

use crate::{
    log_debug, log_error,
    util::error::{Error, Result},
};

use super::link::NodeId;
use super::{FilterDef, FilterGraph, InOut, Options, filter_def};

// ---------------------------------------------------------------------------
// Segment data model (avfilter.h:820-941) — the parsed intermediate form
// ---------------------------------------------------------------------------

/// `AVFilterPadParams` (avfilter.h:820-829): one parsed pad label.
/// `label: None` covers the C "caller may free the label" escape hatch
/// (treating the pad as unlabeled); labels produced by parsing are always
/// `Some(non-empty)` — [`parse_link_name`] rejects empty ones.
#[derive(Clone, Debug)]
pub struct PadParams {
    pub label: Option<String>,
}

/// `AVFilterParams` (avfilter.h:837-896): one filter specification.
/// `filter` is `None` until `segment_create_filters` allocates the node;
/// `filter_name`/`instance_name` are reset to `None` once consumed (C frees
/// the strings — the creation-pending protocol of avfilter.h:839-877).
#[derive(Clone, Debug, Default)]
pub struct FilterParams {
    pub filter: Option<NodeId>,
    pub filter_name: Option<String>,
    pub instance_name: Option<String>,
    pub opts: Options,
    pub inputs: Vec<PadParams>,
    pub outputs: Vec<PadParams>,
}

/// `AVFilterChain` (avfilter.h:904-907).
#[derive(Clone, Debug, Default)]
pub struct FilterChain {
    pub filters: Vec<FilterParams>,
}

/// `AVFilterGraphSegment` (avfilter.h:918-941) minus the `graph`
/// back-pointer (a function argument in Rust).
#[derive(Clone, Debug, Default)]
pub struct GraphSegment {
    pub chains: Vec<FilterChain>,
    pub scale_sws_opts: Option<String>,
}

// ---------------------------------------------------------------------------
// av_get_token (libavutil/avstring.c:143-177)
// ---------------------------------------------------------------------------

/// C's `WHITESPACES` (avstring.c:141; also opt.c:1915, graphparser.c:35).
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

/// `strspn(s, WHITESPACES)` advance — skip leading whitespace in place.
/// Whitespace bytes are ASCII, so the slice lands on a char boundary.
fn skip_ws(buf: &mut &str) {
    let n = buf.bytes().take_while(|&b| is_ws(b)).count();
    *buf = &buf[n..];
}

/// `av_get_token` (avstring.c:143-177): byte-oriented scanner over a shared
/// cursor.
///
/// 1. Skip leading `" \n\t\r"` (150).
/// 2. Loop while the current byte is non-NUL and NOT in `term` (152):
///    on `\\` with a following byte, append that byte verbatim and mark the
///    protection point (154-156); on `'`, copy until the next `'` verbatim —
///    NO escape processing inside quotes; a closing `'` is consumed and marks
///    the protection point, an unterminated one does not (157-163); anything
///    else is appended as-is (164-166).
/// 3. Trim trailing whitespace that sits BEYOND the protection point `end`
///    (169-171) — this is what keeps escaped/quoted trailing whitespace
///    (`"foo\\ "` -> `"foo "`) while plain trailing spaces are trimmed.
/// 4. Leave the cursor ON the terminator (or at end); callers advance.
///
/// A lone backslash at end of input is copied literally (the escape branch
/// requires a following byte). Term chars inside quotes or after a backslash
/// do not terminate. C's realloc dance is dropped.
pub fn get_token(buf: &mut &str, term: &str) -> String {
    let bytes = buf.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && is_ws(bytes[i]) {
        i += 1;
    }
    let mut out: Vec<u8> = Vec::new();
    let mut end = 0usize; // `end` protection marker, as an output length
    while i < bytes.len() && !term.as_bytes().contains(&bytes[i]) {
        let c = bytes[i];
        i += 1;
        if c == b'\\' && i < bytes.len() {
            out.push(bytes[i]);
            i += 1;
            end = out.len();
        } else if c == b'\'' {
            while i < bytes.len() && bytes[i] != b'\'' {
                out.push(bytes[i]);
                i += 1;
            }
            if i < bytes.len() {
                i += 1; // consume the closing quote
                end = out.len();
            }
        } else {
            out.push(c);
        }
    }
    // Trailing-whitespace trim, stopping at `end` (the do/while of 169-171).
    while out.len() > end && is_ws(out[out.len() - 1]) {
        out.pop();
    }
    *buf = &buf[i..];
    // All delimiters/quotes/escapes are ASCII, so `out` slices whole chars;
    // the lossy fallback is purely defensive (see module doc).
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

// ---------------------------------------------------------------------------
// parse_link_name (graphparser.c:43-69)
// ---------------------------------------------------------------------------

/// `parse_link_name` (graphparser.c:43-69): parse `[label]`. The cursor is
/// expected ON the `'['` (callers check); it is consumed unconditionally,
/// then the label is `av_get_token(buf, "]")`. Empty labels and a missing
/// closing `']'` are `AVERROR(EINVAL)` -> `Error::InvalidArgument`; both
/// messages print the input FROM the `'['` inclusive. Quoting and escaping
/// are active inside the label (`[a\]b]` -> `a]b`).
fn parse_link_name(buf: &mut &str) -> Result<String> {
    let start = *buf; // both error texts print from the '[' inclusive
    *buf = &buf[1..]; // consume '[' (43-47)
    let name = get_token(buf, "]");
    if name.is_empty() {
        let msg = format!("Bad (empty?) label found in the following: \"{start}\".");
        log_error!(Some("graph"), "{msg}\n");
        return Err(Error::InvalidArgument(msg));
    }
    if !buf.starts_with(']') {
        // covers end-of-input too (**buf == '\0' at 59)
        let msg = format!("Mismatched '[' found in the following: \"{start}\".");
        log_error!(Some("graph"), "{msg}\n");
        return Err(Error::InvalidArgument(msg));
    }
    *buf = &buf[1..]; // consume ']'
    Ok(name)
}

// ---------------------------------------------------------------------------
// parse_sws_flags (graphparser.c:115-136)
// ---------------------------------------------------------------------------

/// `parse_sws_flags` (graphparser.c:115-136): if the cursor sits on the
/// case-sensitive prefix `"sws_flags="`, take everything up to the FIRST
/// `';'` as the value — KEEPING the `"flags="` substring (C: `*buf += 4; //
/// keep the 'flags=' part`, 127) — and advance past the `';'`.
/// `Ok(None)` leaves the cursor untouched for any other input. A missing
/// `';'` is `AVERROR(EINVAL)`.
fn parse_sws_flags(buf: &mut &str) -> Result<Option<String>> {
    if !buf.starts_with("sws_flags=") {
        return Ok(None);
    }
    let Some(semi) = buf.find(';') else {
        log_error!(Some("graph"), "sws_flags not terminated with ';'.\n");
        return Err(Error::InvalidArgument(
            "sws_flags not terminated with ';'.".into(),
        ));
    };
    let value = buf[4..semi].to_string(); // [4..] keeps "flags=..."
    *buf = &buf[semi + 1..];
    Ok(Some(value))
}

// ---------------------------------------------------------------------------
// linklabels_parse (graphparser.c:292-336)
// ---------------------------------------------------------------------------

/// `linklabels_parse` (graphparser.c:292-336): consume `[label]` runs while
/// the cursor is on `'['`, skipping whitespace AFTER EACH label (324) so
/// `[a] [b]null` parses two labels. Whitespace before the first `'['` is the
/// caller's business. This is also literally what the `[in]`/`[out]`
/// injection runs (952-971 call it on the strings "[in]" / "[out]").
fn linklabels_parse(buf: &mut &str) -> Result<Vec<PadParams>> {
    let mut pads = Vec::new();
    while buf.starts_with('[') {
        let label = parse_link_name(buf)?;
        pads.push(PadParams { label: Some(label) });
        skip_ws(buf);
    }
    Ok(pads)
}

// ---------------------------------------------------------------------------
// ff_filter_opt_parse (avfilter.c:853-904) + get_key (opt.c:1917-1951)
// ---------------------------------------------------------------------------

/// `is_key_char` (opt.c:1917-1922): ASCII letters (via the `(c|32)-'a' < 26`
/// trick), digits, `-`, `_`, `/`, `.`.
fn is_key_char(c: u8) -> bool {
    (c | 32).wrapping_sub(b'a') < 26
        || c.wrapping_sub(b'0') < 10
        || c == b'-'
        || c == b'_'
        || c == b'/'
        || c == b'.'
}

/// `get_key` (opt.c:1932-1951) with `key_val_sep = "="`: skip whitespace,
/// take the maximal `is_key_char` run, skip whitespace again, then require a
/// `'='` and consume it. The key may be EMPTY (`"=5"` -> `""`). On failure
/// the cursor is NOT advanced (C only writes `*ropts` on success).
fn get_key(opts: &mut &str) -> Option<String> {
    let bytes = opts.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && is_ws(bytes[i]) {
        i += 1;
    }
    let key_start = i;
    while i < bytes.len() && is_key_char(bytes[i]) {
        i += 1;
    }
    let key_end = i;
    while i < bytes.len() && is_ws(bytes[i]) {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'=' {
        let key = opts[key_start..key_end].to_string(); // ASCII-only run
        *opts = &opts[i + 1..];
        Some(key)
    } else {
        None // !*opts || !strchr(delim, *opts) -> EINVAL (1942-1943)
    }
}

/// `ff_filter_opt_parse` (avfilter.c:853-904): split an option string into
/// ordered `(key, value)` pairs (`AV_DICT_MULTIKEY` — duplicates appended in
/// order). This REPLACES the wave-1 `parse_args_simple` for parsed graphs:
/// it adds quoting/escaping (`w=100\:2`, `w='100:2'`), whitespace tolerance
/// (`w = 8`), key-charset validation (only `[A-Za-z0-9-_/.]` before `=`),
/// one separator byte per pair, and legal empty values (`w=`).
///
/// `def.shorthand` is the precomputed `av_opt_next` walk of C's AVOption
/// table (skipping CONST and duplicate-offset options, avfilter.c:869-873):
/// positional values fill its slots in order until the first explicit
/// `key=value` (which sets C's `priv_class = NULL`, killing all remaining
/// slots, avfilter.c:892).
pub fn filter_opt_parse(args: &str, def: &FilterDef) -> Result<Options> {
    opt_parse(args, def.shorthand)
}

/// The shared body: `shorthand` is `def.shorthand`, or empty when the filter
/// is unknown (C passes `priv_class == NULL`, avfilter.c:381's
/// `f ? f->priv_class : NULL`) — every entry then needs an explicit key.
fn opt_parse(args: &str, shorthand: &[&str]) -> Result<Options> {
    let mut opts = Options::default();
    let mut cursor = args;
    let mut shorthand_pos = 0usize; // the av_opt_next cursor, flattened
    let mut shorthand_enabled = true; // C's priv_class != NULL
    while !cursor.is_empty() {
        let iter_start = cursor; // what "No option name near '%s'" prints
        let implicit_ok = shorthand_enabled && shorthand_pos < shorthand.len();
        let key = match get_key(&mut cursor) {
            Some(k) => {
                shorthand_enabled = false; // priv_class = NULL (avfilter.c:892)
                Some(k)
            }
            None if implicit_ok => None,
            None => {
                log_error!(Some("graph"), "No option name near '{iter_start}'\n");
                return Err(Error::InvalidArgument(format!(
                    "No option name near '{iter_start}'"
                )));
            }
        };
        // av_opt_get_key_value's value half: get_token(&args, ":")
        // (opt.c:1965); get_key already consumed the '=' on success.
        let value = get_token(&mut cursor, ":");
        if !cursor.is_empty() {
            cursor = &cursor[1..]; // exactly ONE pair-separator byte (887-888)
        }
        match key {
            Some(k) => {
                log_debug!(Some("graph"), "Setting '{k}' to value '{value}'\n"); // 897
                opts.entries.push((k, value));
            }
            None => {
                let name = shorthand[shorthand_pos];
                log_debug!(Some("graph"), "Setting '{name}' to value '{value}'\n");
                opts.entries.push((name.to_string(), value));
                shorthand_pos += 1; // the av_opt_next walk advances
            }
        }
    }
    Ok(opts)
}

// ---------------------------------------------------------------------------
// filter_parse (graphparser.c:338-401)
// ---------------------------------------------------------------------------

/// `filter_parse` (graphparser.c:338-401): parse ONE filter spec —
/// `[in-labels] filter_name[@instance_name][=options] [out-labels]` — into
/// [`FilterParams`].
///
/// * The name token stops at the first raw `=`, `,`, `;` or `[` (353); it may
///   be EMPTY (`,null` -> `Some("")`), failing only at creation with
///   `No such filter: ''`.
/// * The `'@'` split takes the FIRST occurrence (359-367) and — a C quirk
///   kept — escaping does NOT protect it: `av_get_token` has already
///   resolved any `\@` or quoted `@` into a literal `@` before `strchr`.
/// * The options branch fires only on a RAW `'='` at the cursor (369);
///   `null\=` escapes the `=` INTO the name and no options attach. Option
///   syntax errors for an UNKNOWN filter fire here, BEFORE "No such filter"
///   at creation (`bogus=x:y` fails "No option name near 'x:y'" — the whole
///   remaining opts string, since the failed `get_key` did not advance it) —
///   C passes a NULL priv_class, so no positional slots exist (381).
/// * On any failure the wrapper logs `Error parsing a filter description
///   around: %s` with the CURRENT cursor (397-398) and the cursor stays
///   where the failure left it (C advanced it through `*filter`).
fn filter_parse(buf: &mut &str) -> Result<FilterParams> {
    let mut p = FilterParams::default();
    let ret = (|| -> Result<()> {
        p.inputs = linklabels_parse(buf)?;
        let mut filter_name = get_token(buf, "=,;[");
        // '@' split, FIRST occurrence (strchr, 359).
        if let Some(at) = filter_name.find('@') {
            p.instance_name = Some(filter_name[at + 1..].to_string());
            filter_name.truncate(at);
        }
        p.filter_name = Some(filter_name);
        if buf.starts_with('=') {
            *buf = &buf[1..];
            let opts_str = get_token(buf, "[],;");
            // Post-split name; None for an unknown filter -> empty shorthand.
            let def = p.filter_name.as_deref().and_then(filter_def);
            p.opts = opt_parse(&opts_str, def.map(|d| d.shorthand).unwrap_or(&[]))?;
        }
        p.outputs = linklabels_parse(buf)?;
        skip_ws(buf); // 392
        Ok(())
    })();
    match ret {
        Ok(()) => Ok(p),
        Err(e) => {
            // 396-400: the failure wrapper; the cursor stays where the
            // failure left it (the closure advanced `buf` in place).
            log_error!(
                Some("graph"),
                "Error parsing a filter description around: {}\n",
                buf
            );
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// chain_parse (graphparser.c:403-458)
// ---------------------------------------------------------------------------

/// `chain_parse` (graphparser.c:403-458): parse filter specs until
/// end-of-string, a `','` (next filter, same chain) or a `';'` (chain ends;
/// the cursor sits just past it). Anything else after a filter is
/// `Trailing garbage after a filter: %s` (434-435) — reachable only after a
/// closing `']'` of a label run, since the name token swallows everything up
/// to `=,;[` (a plain space is part of the NAME: `"null x"` names the filter
/// `"null x"`, exactly like C).
///
/// A trailing `','` at end of input is accepted (`"null,"` -> one filter).
/// A `';'` where a filter is expected (`";null"`, the middle of `"a;;b"`)
/// yields a filter with an EMPTY name — accepted at parse, failing at
/// creation with `No such filter: ''`.
///
/// The chain-start position for the failure wrapper (454-455) is captured
/// BEFORE consuming anything; the second `%s` is the current cursor.
fn chain_parse(buf: &mut &str) -> Result<FilterChain> {
    let chain_start = *buf;
    let mut cursor = chain_start;
    let mut ch = FilterChain {
        filters: Vec::new(),
    };
    let result: Result<()> = loop {
        if cursor.is_empty() {
            break Ok(());
        }
        let p = match filter_parse(&mut cursor) {
            Ok(p) => p,
            Err(e) => break Err(e),
        };
        ch.filters.push(p);
        // a filter ends with one of: , ; end-of-string (431-438)
        let chr = cursor.as_bytes().first().copied().unwrap_or(0);
        if chr != 0 && chr != b',' && chr != b';' {
            log_error!(Some("graph"), "Trailing garbage after a filter: {cursor}\n");
            break Err(Error::InvalidArgument(format!(
                "Trailing garbage after a filter: {cursor}"
            )));
        }
        if chr != 0 {
            cursor = &cursor[1..]; // consume ',' or ';'
            skip_ws(&mut cursor);
            if chr == b';' {
                break Ok(()); // 444-445
            }
        }
    };
    match result {
        Ok(()) => {
            *buf = cursor; // *pchain = chain (449)
            Ok(ch)
        }
        Err(e) => {
            log_error!(
                Some("graph"),
                "Error parsing filterchain '{}' around: {}\n",
                chain_start,
                cursor
            );
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// avfilter_graph_segment_parse (graphparser.c:460-514)
// ---------------------------------------------------------------------------

/// `avfilter_graph_segment_parse` (graphparser.c:460-514): skip whitespace,
/// take the optional `sws_flags=...;` PREFIX (case-sensitive; a later
/// `sws_flags=` is ordinary filter syntax), then chains until end of string.
/// Empty descriptions (`""`, `"   "`, `"sws_flags=bicubic;"`) are
/// `No filters specified in the graph description` (502-505). C's `flags`
/// param (ENOSYS when nonzero) and the graph back-pointer are dropped.
pub fn segment_parse(desc: &str) -> Result<GraphSegment> {
    let mut seg = GraphSegment::default();
    let mut cursor = desc;
    skip_ws(&mut cursor); // 477
    seg.scale_sws_opts = parse_sws_flags(&mut cursor)?; // 479
    skip_ws(&mut cursor); // 483
    while !cursor.is_empty() {
        let ch = chain_parse(&mut cursor)?;
        seg.chains.push(ch);
        skip_ws(&mut cursor); // 499
    }
    if seg.chains.is_empty() {
        log_error!(
            Some("graph"),
            "No filters specified in the graph description\n"
        );
        return Err(Error::InvalidArgument(
            "No filters specified in the graph description".into(),
        ));
    }
    Ok(seg)
}

// ---------------------------------------------------------------------------
// avfilter_graph_segment_create_filters (graphparser.c:516-575)
// ---------------------------------------------------------------------------

/// `avfilter_graph_segment_create_filters` (graphparser.c:516-575):
///
/// 1. A `scale_sws_opts` in the segment REPLACES `g.scale_sws_opts` (523-527).
/// 2. Walk chains then filters with a segment-global `idx` counting CREATED
///    filters only. Skip already-processed params (`p.filter` set or
///    `filter_name` gone — the programmatic states, unreachable from
///    [`segment_parse`]). Unknown names log C's line and are
///    `AVERROR_FILTER_NOT_FOUND` (542-545).
/// 3. Instance name: `Parsed_{def.name}_{idx}` or `{def.name}@{instance}`
///    (548-551). `alloc_filter` seeds `name = def.name`; the parser owns the
///    overwrite. Divergence: C `snprintf` truncates at 63 bytes.
/// 4. The scale pre-apply (557-565): C runs
///    `av_set_options_string(scale_sws_opts, "=", ":")` AT CREATION, before
///    the `'='`-options of the description — prepending the parsed entries
///    reproduces later-wins under sequential application. Dead until `scale`
///    registers (wave 2C); see the module doc for the parser-shape nuance.
fn segment_create_filters(g: &mut FilterGraph, seg: &mut GraphSegment) -> Result<()> {
    if let Some(sws) = &seg.scale_sws_opts {
        g.scale_sws_opts = sws.clone(); // 523-527
    }
    let mut idx = 0usize;
    for chain in &mut seg.chains {
        for p in &mut chain.filters {
            if p.filter.is_some() || p.filter_name.is_none() {
                continue; // 538-540
            }
            let fname = p.filter_name.clone().expect("checked is_some");
            let Some(def) = filter_def(&fname) else {
                log_error!(Some("graph"), "No such filter: '{fname}'\n");
                return Err(Error::NotFound(format!("No such filter: '{fname}'")));
            };
            let instance_name = match &p.instance_name {
                None => format!("Parsed_{}_{}", def.name, idx), // 549
                Some(inst) => format!("{}@{}", def.name, inst), // 551
            };
            let node = g.alloc_filter(&fname)?;
            g.nodes[node.0].name = instance_name;
            // The scale hook (557-565): PREPEND the sws options so the
            // description's own entries override same-named keys later.
            if def.name == "scale" && !g.scale_sws_opts.is_empty() {
                let mut entries = filter_opt_parse(&g.scale_sws_opts, def)?.entries;
                entries.append(&mut p.opts.entries);
                p.opts.entries = entries;
            }
            p.filter = Some(node);
            p.filter_name = None; // av_freep (567-568)
            p.instance_name = None;
            idx += 1; // 570 — counts CREATED filters only
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// segment_apply_opts + segment_init (graphparser.c:586-641), FUSED
// ---------------------------------------------------------------------------

/// `avfilter_graph_segment_apply_opts` (586-614) + `segment_init`
/// (616-641), fused per filter: the Rust port's `init_filter` IS the option
/// applier (its impls drain recognized entries from `nodes[node].opts`;
/// leftovers are the caller's `No such option: {key}`), so C's two passes
/// collapse into one loop in filter order.
///
/// ORDERING DIVERGENCE (doc'd in the module doc): C applies options to ALL
/// filters before initializing ANY, so an option error on a later filter
/// outranks an init error on an earlier one; this loop fails at the FIRST
/// filter in order. Only which message surfaces first differs.
///
/// The C `log_unknown_opt` line (857-880): when `init_filter` reports
/// `NotFound` and the node still holds option entries, that leftover pair is
/// `AVERROR_OPTION_NOT_FOUND` — log `Could not set non-existent option '%s'
/// to value '%s'` with the FILTER INSTANCE as context, then propagate. Every
/// other error is init-stage (C would log "Error initializing filters").
fn segment_apply_opts_and_init(g: &mut FilterGraph, seg: &GraphSegment) -> Result<()> {
    for chain in &seg.chains {
        for p in &chain.filters {
            // Creation-pending check (599-600, 628-629) is unreachable here:
            // create always ran. Filter disabled (p->filter == NULL): skip.
            let Some(node) = p.filter else { continue };
            if g.nodes[node.0].initialized {
                continue; // 630-632
            }
            match g.init_filter(node, p.opts.clone()) {
                Ok(()) => {}
                Err(e) => {
                    if matches!(e, Error::NotFound(_))
                        && let Some((key, value)) = g.nodes[node.0].opts.entries.first()
                    {
                        log_error!(
                            Some(&g.nodes[node.0].name),
                            "Could not set non-existent option '{key}' to value '{value}'\n"
                        );
                    }
                    return Err(e);
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// find_linklabel (graphparser.c:643-676)
// ---------------------------------------------------------------------------

/// `find_linklabel` (graphparser.c:643-676): search FORWARD from
/// `(idx_chain, idx_filter)` — INCLUSIVE of the starting filter — chain-major,
/// filter-minor (`idx_filter` resets to 0 for later chains, 671) — for the
/// first not-yet-linked pad at index `i` whose PARSED label at that same
/// index equals `label`. `output` selects which side is scanned: `true`
/// searches producers (their `outputs`), `false` consumers (their `inputs`).
///
/// The inclusive start is what makes the label pairing order-independent:
/// an earlier same-labeled producer already linked this pad during its own
/// `link_outputs`, so the occupied pad is skipped and only not-yet-processed
/// producers are found. A self-labeled filter (`[a]null[a]`) links its own
/// output to its own input — C has no same-filter guard.
fn find_linklabel(
    g: &FilterGraph,
    seg: &GraphSegment,
    label: &str,
    output: bool,
    idx_chain: usize,
    idx_filter: usize,
) -> Option<(NodeId, usize)> {
    let mut ic = idx_chain;
    let mut ifl = idx_filter;
    while ic < seg.chains.len() {
        let ch = &seg.chains[ic];
        while ifl < ch.filters.len() {
            let p = &ch.filters[ifl];
            ifl += 1;
            let Some(node) = p.filter else { continue }; // 658-659
            let labels = if output { &p.outputs } else { &p.inputs };
            let pads = if output {
                &g.nodes[node.0].outputs
            } else {
                &g.nodes[node.0].inputs
            };
            for i in 0..labels.len().min(pads.len()) {
                if pads[i].is_none() && labels[i].label.as_deref() == Some(label) {
                    return Some((node, i)); // 665-668
                }
            }
        }
        ifl = 0; // 671
        ic += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// inout_add / extract_inout (graphparser.c:678-700, 86-101)
// ---------------------------------------------------------------------------

/// `inout_add` + `append_inout` (graphparser.c:678-700, 103-113):
/// tail-append one open pad. C's alloc/list-splice plumbing is Vec ownership.
fn inout_add(list: &mut Vec<InOut>, node: NodeId, pad: usize, label: Option<&str>) {
    list.push(InOut {
        name: label.map(str::to_string),
        node,
        pad,
    });
}

/// `extract_inout` (graphparser.c:86-101): remove and return the FIRST entry
/// whose `name` equals `label` (None names never match — C's
/// `!*links->name || strcmp` skips them). The list is untouched on no-match.
fn extract_inout(label: &str, list: &mut Vec<InOut>) -> Option<InOut> {
    let idx = list
        .iter()
        .position(|io| io.name.as_deref() == Some(label))?;
    Some(list.remove(idx))
}

// ---------------------------------------------------------------------------
// link_inputs (graphparser.c:702-745)
// ---------------------------------------------------------------------------

/// `link_inputs` (graphparser.c:702-745): wire the filter's input pads.
/// More parsed labels than actual pads is EINVAL (711-716; the message names
/// the FILTER TYPE, `f->filter->name`). Each unlinked input with a parsed
/// label tries [`find_linklabel`] for a same-labeled OUTPUT (producer)
/// forward from this filter (728); on match `g.link(producer, idx, f, in)`.
/// Otherwise the pad goes to the open list — an unmatched LABELED input
/// carries its label, an unlabeled one gets `name: None` (739).
fn link_inputs(
    g: &mut FilterGraph,
    seg: &GraphSegment,
    idx_chain: usize,
    idx_filter: usize,
    inputs: &mut Vec<InOut>,
) -> Result<()> {
    let p = &seg.chains[idx_chain].filters[idx_filter];
    let node = p.filter.expect("segment_link guarantees created filters");
    let def_name = g.nodes[node.0].def.name;
    let actual = g.nodes[node.0].def.inputs.len();
    if actual < p.inputs.len() {
        let msg = format!(
            "More input link labels specified for filter '{def_name}' than it has inputs: {} > {actual}",
            p.inputs.len()
        );
        log_error!(Some("graph"), "{msg}\n");
        return Err(Error::InvalidArgument(msg));
    }
    for in_pad in 0..actual {
        if g.nodes[node.0].inputs[in_pad].is_some() {
            continue; // 722-724: already linked by an earlier producer
        }
        let label = p.inputs.get(in_pad).and_then(|x| x.label.as_deref());
        if let Some(label) = label
            && let Some((src, spad)) = find_linklabel(g, seg, label, true, idx_chain, idx_filter)
        {
            g.link(src, spad, node, in_pad)?;
            continue;
        }
        inout_add(inputs, node, in_pad, label);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// link_outputs (graphparser.c:747-812)
// ---------------------------------------------------------------------------

/// `link_outputs` (graphparser.c:747-812): wire the filter's output pads.
/// Pad-count overload mirrors [`link_inputs`] (756-762). Per unlinked output:
///
/// * A parsed label tries [`find_linklabel`] for a same-labeled INPUT
///   (consumer) forward from this filter (772); on match
///   `g.link(f, out, consumer, idx)`. A labeled-but-unmatched output goes to
///   the open list WITH its label — the next-filter fallback never runs for
///   it (the `&& !label` of 785).
/// * An UNLABELED output tries the next-filter fallback (785-802): scan the
///   following filters of the SAME chain, skipping disabled ones, and link
///   to the first unlinked input pad that is beyond the parsed labels or
///   unlabeled. ONLY the first non-disabled following filter is examined —
///   the `break` at 801 gives up after it (`"null;null"` stays unlinked,
///   `"null,null"` links; cross-chain unlabeled connection never happens).
/// * No link made -> the open list (label or None).
fn link_outputs(
    g: &mut FilterGraph,
    seg: &GraphSegment,
    idx_chain: usize,
    idx_filter: usize,
    outputs: &mut Vec<InOut>,
) -> Result<()> {
    let ch = &seg.chains[idx_chain];
    let p = &ch.filters[idx_filter];
    let node = p.filter.expect("segment_link guarantees created filters");
    let def_name = g.nodes[node.0].def.name;
    let actual = g.nodes[node.0].def.outputs.len();
    if actual < p.outputs.len() {
        let msg = format!(
            "More output link labels specified for filter '{def_name}' than it has outputs: {} > {actual}",
            p.outputs.len()
        );
        log_error!(Some("graph"), "{msg}\n");
        return Err(Error::InvalidArgument(msg));
    }
    for out in 0..actual {
        if g.nodes[node.0].outputs[out].is_some() {
            continue; // 766-768
        }
        let label = p.outputs.get(out).and_then(|x| x.label.as_deref());
        if let Some(label) = label
            && let Some((dst, dpad)) = find_linklabel(g, seg, label, false, idx_chain, idx_filter)
        {
            g.link(node, out, dst, dpad)?;
            continue;
        }
        if label.is_none() {
            // The next-filter fallback (785-802).
            let mut linked = false;
            for p_next in &ch.filters[idx_filter + 1..] {
                let Some(next_node) = p_next.filter else {
                    continue; // disabled filter: skip without stopping
                };
                let nb_inputs = g.nodes[next_node.0].def.inputs.len();
                let mut matched = false;
                for in_pad in 0..nb_inputs {
                    let unlinked = g.nodes[next_node.0].inputs[in_pad].is_none();
                    let unlabeled =
                        in_pad >= p_next.inputs.len() || p_next.inputs[in_pad].label.is_none();
                    if unlinked && unlabeled {
                        g.link(node, out, next_node, in_pad)?;
                        matched = true; // the goto cont (798)
                        break;
                    }
                }
                if matched {
                    linked = true;
                }
                break; // 801: examine ONLY the first non-disabled follower
            }
            if linked {
                continue; // cont:
            }
        }
        inout_add(outputs, node, out, label);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// avfilter_graph_segment_link (graphparser.c:814-854)
// ---------------------------------------------------------------------------

/// `avfilter_graph_segment_link` (graphparser.c:814-854): walk chains then
/// filters IN ORDER, running `link_inputs` then `link_outputs` per filter —
/// that iteration order IS the label-pairing algorithm (see
/// [`find_linklabel`]). C's creation-pending error (832-835) is unreachable
/// (create always ran) and omitted. Links are created only between
/// INITIALIZED nodes — the stage order guarantees it, and `g.link` enforces
/// the init gate (graph.rs:243-251).
fn segment_link(
    g: &mut FilterGraph,
    seg: &GraphSegment,
    inputs: &mut Vec<InOut>,
    outputs: &mut Vec<InOut>,
) -> Result<()> {
    for ic in 0..seg.chains.len() {
        for ifl in 0..seg.chains[ic].filters.len() {
            if seg.chains[ic].filters[ifl].filter.is_none() {
                continue; // 837-838
            }
            link_inputs(g, seg, ic, ifl, inputs)?;
            link_outputs(g, seg, ic, ifl, outputs)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// avfilter_graph_segment_apply (graphparser.c:882-918)
// ---------------------------------------------------------------------------

/// `avfilter_graph_segment_apply` (graphparser.c:882-918): create -> apply
/// opts/init -> link, each failure wrapped with C's one-line reason. The
/// fused Rust opts/init stage maps its leftover (`NotFound`) case to C's
/// "Error applying filter options" wrapper and everything else to "Error
/// initializing filters" — a heuristic, since the port cannot distinguish a
/// bad option VALUE (init error) from a leftover key except by that shape.
fn segment_apply(g: &mut FilterGraph, seg: &mut GraphSegment) -> Result<(Vec<InOut>, Vec<InOut>)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    if let Err(e) = segment_create_filters(g, seg) {
        log_error!(Some("graph"), "Error creating filters\n");
        return Err(e);
    }
    if let Err(e) = segment_apply_opts_and_init(g, seg) {
        if matches!(e, Error::NotFound(_)) {
            log_error!(Some("graph"), "Error applying filter options\n");
        } else {
            log_error!(Some("graph"), "Error initializing filters\n");
        }
        return Err(e);
    }
    if let Err(e) = segment_link(g, seg, &mut inputs, &mut outputs) {
        log_error!(Some("graph"), "Error linking filters\n");
        return Err(e);
    }
    Ok((inputs, outputs))
}

// ---------------------------------------------------------------------------
// avfilter_graph_parse_ptr (graphparser.c:920-1042)
// ---------------------------------------------------------------------------

/// The graph-clearing half of `avfilter_graph_parse_ptr`'s `end:` block
/// (1020-1041, shared with `avfilter_graph_parse2`'s 156-161): on ANY error
/// free EVERY filter in the graph — including pre-existing ones. The port
/// clears the node/link arenas plus the three negotiation list arenas
/// (nothing references a list once its links are gone); C keeps
/// `scale_sws_opts`, so the port does too. Only `parse_ptr` logs the
/// "Error processing filtergraph" line (1024); `parse2` clears silently.
fn teardown(g: &mut FilterGraph) {
    g.nodes.clear();
    g.links.clear();
    g.fmt_lists.clear();
    g.csp_lists.clear();
    g.rng_lists.clear();
}

/// `avfilter_graph_parse_ptr` (graphparser.c:920-1042) — the full C entry.
///
/// 1. `segment_parse` (933).
/// 2. `segment_create_filters` + `segment_apply_opts_and_init` DIRECTLY
///    (937-950) — no wrapper logs for these stages here (C runs them inline;
///    only [`segment_apply`]'s own wrappers exist, and its create/opts-init
///    stages re-run as provable no-ops at step 5).
/// 3. INJECTION, parse_ptr-only (952-971): the FIRST filter of the FIRST
///    chain gets input label "in" when it has exactly 1 input pad and ZERO
///    parsed input labels; the LAST filter of the LAST chain gets output
///    label "out" under the same shape. An explicitly written label
///    suppresses injection (C's `!p->inputs` — pointer NULL = zero labels);
///    multi-pad filters never inject. Injected labels are ORDINARY labels
///    (they can match a same-named pad inside the graph).
/// 4. `segment_apply` (973): link + collect the segment's open pads.
/// 5. USER MATCHING (978-1018): drain the parsed `inputs` head-first — an
///    entry whose name matches a `user_outputs` entry consumes it and links
///    producer->consumer; otherwise it is appended to `user_inputs`. The
///    `outputs` list is symmetric (matches against `user_inputs`, links
///    cur->match). DIRECTION (avfilter.h:745-755): the user INPUTS list
///    holds INPUTS OF THE EXISTING GRAPH (its sinks), the user OUTPUTS list
///    holds its OUTPUTS (its sources) — from the parsed graph's viewpoint
///    they are the other way around.
/// 6. On any error: the `end:` teardown clears the graph; the user lists
///    still hold whatever was not yet matched (C writes the partially
///    consumed lists back, 1033-1036).
pub fn graph_parse_ptr(
    g: &mut FilterGraph,
    desc: &str,
    open_inputs: &mut Vec<InOut>,
    open_outputs: &mut Vec<InOut>,
) -> Result<()> {
    // The caller's lists are consumed and rewritten (C's ** parameters).
    let mut user_inputs = std::mem::take(open_inputs);
    let mut user_outputs = std::mem::take(open_outputs);
    let ret = graph_parse_ptr_inner(g, desc, &mut user_inputs, &mut user_outputs);
    let out = match ret {
        Ok(()) => Ok(()),
        Err(e) => {
            // 1023-1030: log, then free every filter in the graph.
            log_error!(Some("graph"), "Error processing filtergraph: {e}\n");
            teardown(g);
            Err(e)
        }
    };
    *open_inputs = user_inputs;
    *open_outputs = user_outputs;
    out
}

fn graph_parse_ptr_inner(
    g: &mut FilterGraph,
    desc: &str,
    user_inputs: &mut Vec<InOut>,
    user_outputs: &mut Vec<InOut>,
) -> Result<()> {
    let mut seg = segment_parse(desc)?; // 933
    segment_create_filters(g, &mut seg)?; // 937
    segment_apply_opts_and_init(g, &seg)?; // 941-948 (log_unknown_opt inside)

    // (952-971) [in] / [out] injection.
    {
        let first = &mut seg.chains[0].filters[0];
        let first_node = first.filter.expect("created above");
        if g.nodes[first_node.0].def.inputs.len() == 1 && first.inputs.is_empty() {
            first.inputs.push(PadParams {
                label: Some("in".to_string()),
            });
        }
    }
    {
        let last_chain = seg.chains.last_mut().expect("segment has chains");
        let last = last_chain.filters.last_mut().expect("chain has filters");
        let last_node = last.filter.expect("created above");
        if g.nodes[last_node.0].def.outputs.len() == 1 && last.outputs.is_empty() {
            last.outputs.push(PadParams {
                label: Some("out".to_string()),
            });
        }
    }

    // (973) segment_apply: steps 1-2 re-run as no-ops; only linking works.
    let (mut inputs, mut outputs) = segment_apply(g, &mut seg)?;

    // (978-998) parsed inputs against user outputs (the existing sources).
    let parsed_inputs = std::mem::take(&mut inputs);
    for cur in parsed_inputs {
        let matched = cur
            .name
            .as_deref()
            .and_then(|n| extract_inout(n, user_outputs));
        if let Some(m) = matched {
            g.link(m.node, m.pad, cur.node, cur.pad)?;
        } else {
            user_inputs.push(cur);
        }
    }
    // (999-1018) parsed outputs against user inputs (the existing sinks).
    let parsed_outputs = std::mem::take(&mut outputs);
    for cur in parsed_outputs {
        let matched = cur
            .name
            .as_deref()
            .and_then(|n| extract_inout(n, user_inputs));
        if let Some(m) = matched {
            g.link(cur.node, cur.pad, m.node, m.pad)?;
        } else {
            user_outputs.push(cur);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// avfilter_graph_parse2 (graphparser.c:138-162)
// ---------------------------------------------------------------------------

/// `avfilter_graph_parse2` (graphparser.c:138-162): parse + apply (with the
/// four [`segment_apply`] wrapper logs) and return the open pads. NO
/// `[in]`/`[out]` injection and NO user-list matching — so
/// `graph_parse2(g, "null")` yields an UNNAMED open input, unlike
/// [`parse_ptr`] which names it "in". A `segment_apply` failure tears the
/// graph down (156-161); a parse failure leaves it untouched (nothing was
/// created yet).
pub fn graph_parse2(g: &mut FilterGraph, desc: &str) -> Result<(Vec<InOut>, Vec<InOut>)> {
    let mut seg = segment_parse(desc)?;
    match segment_apply(g, &mut seg) {
        Ok(io) => Ok(io),
        Err(e) => {
            teardown(g); // 156-161: silent clear, no wrapper log here
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// The graph.rs stub signature (delegation entry)
// ---------------------------------------------------------------------------

/// `avfilter_graph_parse_ptr` with NO user lists — the shape of the
/// `FilterGraph::parse_ptr` stub in `graph.rs`. The returned lists are just
/// the unmatched open pads of the parsed graph (labeled entries carry their
/// label, the injected "in"/"out" where applicable). Callers wanting the
/// full C behavior (matching against pre-existing pads) use
/// [`graph_parse_ptr`] with their own lists.
pub fn parse_ptr(g: &mut FilterGraph, desc: &str) -> Result<(Vec<InOut>, Vec<InOut>)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    graph_parse_ptr(g, desc, &mut inputs, &mut outputs)?;
    Ok((inputs, outputs))
}

// ---------------------------------------------------------------------------
// Tests — pin the C behaviors (texts, state transitions, quoting rules)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::graph::engine_test_helpers::run_to_quiescence;
    use crate::filter::{FilterFlags, filter};
    use crate::util::frame::Frame;

    /// A minimal def with one shorthand slot, for `filter_opt_parse` rules.
    struct Noop;
    impl crate::filter::FilterImpl for Noop {
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
    static W_DEF: FilterDef = FilterDef {
        name: "wtest",
        inputs: &[],
        outputs: &[],
        flags: FilterFlags(0),
        shorthand: &["w"],
        make: || Box::new(Noop),
    };

    fn invalid_arg<T: std::fmt::Debug>(res: Result<T>) -> String {
        match res {
            Err(Error::InvalidArgument(msg)) => msg,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    fn not_found<T: std::fmt::Debug>(res: Result<T>) -> String {
        match res {
            Err(Error::NotFound(msg)) => msg,
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    // ---- av_get_token (avstring.c:150-171) -----------------------------------

    #[test]
    fn get_token_basics() {
        // Leading skip + trailing trim (150, 169-171).
        let mut b = "  foo  ";
        assert_eq!(get_token(&mut b, ""), "foo");
        assert!(b.is_empty());
        // Escaped trailing space survives — the `end` protection marker.
        let mut b = "foo\\ ";
        assert_eq!(get_token(&mut b, ""), "foo ");
        // A lone backslash at end of input is copied literally (the escape
        // branch requires a following byte, 154).
        let mut b = "foo\\";
        assert_eq!(get_token(&mut b, ""), "foo\\");
        // Terminators do not fire inside quotes...
        let mut b = "'a]b']";
        assert_eq!(get_token(&mut b, "]"), "a]b");
        assert_eq!(b, "]");
        // ...or after a backslash (the cursor lands past BOTH bytes).
        let mut b = "a\\]b";
        assert_eq!(get_token(&mut b, "]"), "a]b");
        assert!(b.is_empty());
        // Unterminated quote: interior kept, trailing ws still trimmed
        // (the marker was never advanced, 157-163).
        let mut b = "a 'b ";
        assert_eq!(get_token(&mut b, ""), "a b");
        // Quoted interior whitespace is protected from the trim.
        let mut b = "  ' x '  ";
        assert_eq!(get_token(&mut b, ""), " x ");
        // Empty token: cursor stops ON the terminator (not consumed).
        let mut b = "  ,foo";
        assert_eq!(get_token(&mut b, ",;["), "");
        assert_eq!(b, ",foo");
    }

    // ---- parse_link_name (graphparser.c:53-64) --------------------------------

    #[test]
    fn parse_link_name_errors() {
        let mut b = "[]";
        assert_eq!(
            invalid_arg(parse_link_name(&mut b)),
            "Bad (empty?) label found in the following: \"[]\"."
        );
        let mut b = "[abc";
        assert_eq!(
            invalid_arg(parse_link_name(&mut b)),
            "Mismatched '[' found in the following: \"[abc\"."
        );
        // Quoting/escaping inside the label; the ']' is consumed, 'x' left.
        let mut b = "[a\\]b]x";
        assert_eq!(parse_link_name(&mut b).unwrap(), "a]b");
        assert_eq!(b, "x");
    }

    // ---- ff_filter_opt_parse (avfilter.c:863-901 + opt.c:1937-1948) -----------

    #[test]
    fn filter_opt_parse_rules() {
        let null_def = filter_def("null").unwrap(); // empty shorthand
        // Whitespace tolerance around '=' and after the value.
        let o = filter_opt_parse("w = 8", &W_DEF).unwrap();
        assert_eq!(o.entries, vec![("w".into(), "8".into())]);
        // Positional value fills the shorthand slot.
        let o = filter_opt_parse("800", &W_DEF).unwrap();
        assert_eq!(o.entries, vec![("w".into(), "800".into())]);
        // Separator quoting: escape or single quotes both keep the ':'.
        for args in ["w=100\\:2", "w='100:2'"] {
            let o = filter_opt_parse(args, &W_DEF).unwrap();
            assert_eq!(o.entries, vec![("w".into(), "100:2".into())], "{args}");
        }
        // An explicit key kills the remaining positional slots.
        assert_eq!(
            invalid_arg(filter_opt_parse("w=1:2", &W_DEF)),
            "No option name near '2'"
        );
        // Positional with NO shorthand slot available (null).
        assert_eq!(
            invalid_arg(filter_opt_parse("800", null_def)),
            "No option name near '800'"
        );
        // Empty explicit value is legal.
        let o = filter_opt_parse("w=", &W_DEF).unwrap();
        assert_eq!(o.entries, vec![("w".into(), "".into())]);
        // Leading ':' with a slot left: an EMPTY positional value, slot 0
        // (the ':' pair separator is then skipped and the loop ends).
        let o = opt_parse(":", &["w"]).unwrap();
        assert_eq!(o.entries, vec![("w".into(), "".into())]);
        // AV_DICT_MULTIKEY: duplicate keys appended in order.
        let fmt_def = filter_def("format").unwrap();
        let o = filter_opt_parse("pix_fmts=a:pix_fmts=b", fmt_def).unwrap();
        assert_eq!(o.entries.len(), 2);
        assert_eq!(o.entries[1], ("pix_fmts".into(), "b".into()));
        // Key charset: letters, digits, '-', '_', '/', '.'; may be empty.
        let o = filter_opt_parse("a-b.c/d_e=1", &W_DEF).unwrap();
        assert_eq!(o.entries, vec![("a-b.c/d_e".into(), "1".into())]);
        let o = filter_opt_parse("=5", &W_DEF).unwrap();
        assert_eq!(o.entries, vec![("".into(), "5".into())]);
        // "a b=3" with a slot left: get_key reads "a", hits ws, then
        // 'b' != '=' -> failure -> the WHOLE remainder is the positional
        // value (opt.c:1937-1948 quirk).
        let o = filter_opt_parse("a b=3", &W_DEF).unwrap();
        assert_eq!(o.entries, vec![("w".into(), "a b=3".into())]);
        // Unknown filter => empty shorthand: explicit keys still parse.
        let o = opt_parse("x=1", &[]).unwrap();
        assert_eq!(o.entries, vec![("x".into(), "1".into())]);
        assert_eq!(
            invalid_arg(opt_parse("val", &[])),
            "No option name near 'val'"
        );
    }

    // ---- chain_parse / segment_parse (graphparser.c:416-506) ------------------

    #[test]
    fn chain_and_segment_parse() {
        let seg = segment_parse("null,null;null@x").unwrap();
        assert_eq!(seg.chains.len(), 2);
        assert_eq!(seg.chains[0].filters.len(), 2);
        assert_eq!(seg.chains[1].filters.len(), 1);
        assert_eq!(seg.chains[1].filters[0].instance_name.as_deref(), Some("x"));
        assert_eq!(
            seg.chains[1].filters[0].filter_name.as_deref(),
            Some("null")
        );
        // A trailing ',' at end of input is accepted.
        let seg = segment_parse("null,").unwrap();
        assert_eq!(seg.chains.len(), 1);
        assert_eq!(seg.chains[0].filters.len(), 1);
        // Trailing garbage after a label run.
        assert_eq!(
            invalid_arg(segment_parse("[a]null[b]x")),
            "Trailing garbage after a filter: x"
        );
        // A space is NOT a name terminator (term is "=,;["): "null x" is a
        // single filter NAME, failing only at creation.
        let seg = segment_parse("null x").unwrap();
        assert_eq!(
            seg.chains[0].filters[0].filter_name.as_deref(),
            Some("null x")
        );
        // Empty descriptions.
        for desc in ["", "   ", "sws_flags=bicubic;"] {
            assert_eq!(
                invalid_arg(segment_parse(desc)),
                "No filters specified in the graph description",
                "{desc:?}"
            );
        }
        // Empty-name filters parse; they fail at CREATE time.
        let seg = segment_parse(",null").unwrap();
        assert_eq!(seg.chains[0].filters[0].filter_name.as_deref(), Some(""));
        let seg = segment_parse("null;;null").unwrap();
        assert_eq!(seg.chains.len(), 3);
        assert_eq!(seg.chains[1].filters[0].filter_name.as_deref(), Some(""));
        // '@' splits on the FIRST occurrence: instance keeps later '@'s.
        let seg = segment_parse("a@b@c").unwrap();
        let p = &seg.chains[0].filters[0];
        assert_eq!(p.filter_name.as_deref(), Some("a"));
        assert_eq!(p.instance_name.as_deref(), Some("b@c"));
    }

    // ---- parse_sws_flags (graphparser.c:117-135) -------------------------------

    #[test]
    fn sws_flags() {
        let seg = segment_parse("sws_flags=bicubic;null").unwrap();
        assert_eq!(seg.scale_sws_opts.as_deref(), Some("flags=bicubic")); // prefix kept
        assert_eq!(seg.chains.len(), 1);
        // Missing ';' terminator.
        assert_eq!(
            invalid_arg(segment_parse("sws_flags=bicubic")),
            "sws_flags not terminated with ';'."
        );
        // The value may contain anything except ';' (127-132), prefix kept.
        let seg = segment_parse("sws_flags=a[b:c=1;null").unwrap();
        assert_eq!(seg.scale_sws_opts.as_deref(), Some("flags=a[b:c=1"));
        // Full pipeline: the value lands on the graph field.
        let mut g = FilterGraph::new();
        graph_parse2(&mut g, "sws_flags=bicubic;null").unwrap();
        assert_eq!(g.scale_sws_opts, "flags=bicubic");
        // A non-prefix sws_flags is ordinary filter syntax; the graph field
        // stays untouched.
        let mut g = FilterGraph::new();
        graph_parse2(&mut g, "null;null").unwrap();
        assert_eq!(g.scale_sws_opts, "");
    }

    // ---- segment_create_filters naming + lookup (graphparser.c:542-553) --------

    #[test]
    fn create_naming_and_lookup() {
        let mut g = FilterGraph::new();
        graph_parse2(&mut g, "null,null;null").unwrap();
        let names: Vec<&str> = g.nodes.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["Parsed_null_0", "Parsed_null_1", "Parsed_null_2"]
        );
        let mut g = FilterGraph::new();
        graph_parse2(&mut g, "null@foo").unwrap();
        assert_eq!(g.nodes[0].name, "null@foo");
        // Unknown filter: C's text + AVERROR_FILTER_NOT_FOUND.
        let mut g = FilterGraph::new();
        assert_eq!(
            not_found(graph_parse2(&mut g, "bogus")),
            "No such filter: 'bogus'"
        );
        // Option-syntax errors for an UNKNOWN filter fire at PARSE, before
        // "No such filter" (priv_class NULL, avfilter.c:381+869-884).
        assert_eq!(
            invalid_arg(segment_parse("bogus=x:y")),
            "No option name near 'x:y'"
        );
    }

    // ---- leftover options (graphparser.c:857-880 + init leftover path) ---------

    #[test]
    fn options_leftover() {
        // An explicit key null does not know: leftover after init.
        let mut g = FilterGraph::new();
        assert_eq!(
            not_found(graph_parse2(&mut g, "null=x=5")),
            "No such option: x"
        );
        // format consumes pix_fmts/color_spaces/color_ranges only.
        let mut g = FilterGraph::new();
        assert_eq!(
            not_found(graph_parse2(&mut g, "format=badkey=yuv420p")),
            "No such option: badkey"
        );
        // Positional (shorthand) options ARE consumed: format's slot 0.
        let mut g = FilterGraph::new();
        let (_, outputs) = graph_parse2(&mut g, "format=yuv420p").unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(g.nodes[0].opts.entries.is_empty(), "pix_fmts consumed");
    }

    // ---- [in]/[out] injection (graphparser.c:952-971) --------------------------

    #[test]
    fn injection_in_out() {
        let mut g = FilterGraph::new();
        let (ins, outs) = parse_ptr(&mut g, "null").unwrap();
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].name.as_deref(), Some("in"));
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].name.as_deref(), Some("out"));
        assert_eq!(ins[0].node, outs[0].node, "the same null instance");

        // Only the FIRST filter of the FIRST chain / LAST of the LAST chain
        // is injected — in a SINGLE chain the middle is chained by ','.
        let mut g = FilterGraph::new();
        let (ins, outs) = parse_ptr(&mut g, "null,null").unwrap();
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].name.as_deref(), Some("in"));
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].name.as_deref(), Some("out"));

        // Across chains (';'), the injected labels pair nothing: the second
        // chain's input is a plain UNNAMED open pad and the first chain's
        // tail output stays open unlabeled.
        let mut g = FilterGraph::new();
        let (ins, outs) = parse_ptr(&mut g, "null,null;null").unwrap();
        assert_eq!(ins.len(), 2);
        assert_eq!(ins[0].name.as_deref(), Some("in"));
        assert_eq!(ins[1].name, None);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].name, None);
        assert_eq!(outs[1].name.as_deref(), Some("out"));

        // An explicitly written label suppresses injection (C's !p->inputs).
        let mut g = FilterGraph::new();
        let (ins, outs) = parse_ptr(&mut g, "[foo]null").unwrap();
        assert_eq!(ins[0].name.as_deref(), Some("foo"));
        assert_eq!(outs[0].name.as_deref(), Some("out"));
        let mut g = FilterGraph::new();
        let (ins, outs) = parse_ptr(&mut g, "null[foo]").unwrap();
        assert_eq!(ins[0].name.as_deref(), Some("in"));
        assert_eq!(outs[0].name.as_deref(), Some("foo"));

        // graph_parse2 does NOT inject: the open pads come back unnamed —
        // the two entries' difference, pinned.
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "null").unwrap();
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].name, None);
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].name, None);
    }

    // ---- label pairing (graphparser.c:643-676, 702-812) ------------------------

    #[test]
    fn label_pairing_forward_search() {
        // A 2-cycle: filter2.out -> filter1.in via label a (found during
        // f1's link_inputs by the INCLUSIVE forward search), then
        // f1.out -> f2.in via label b. Zero open pads.
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "[a]null[b];[b]null[a]").unwrap();
        assert!(ins.is_empty());
        assert!(outs.is_empty());
        assert_eq!(g.links.len(), 2);
        for n in &g.nodes {
            assert!(n.inputs[0].is_some() && n.outputs[0].is_some());
        }
        assert_ne!(g.links[0].src, g.links[0].dst);

        // A labeled INPUT with no same-labeled producer stays open CARRYING
        // its label; the ','-chained tail's output is the unnamed open pad.
        // (The labeled path never falls through to the next-filter search,
        // which is for unlabeled OUTPUTS.)
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "[a]null,null").unwrap();
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].name.as_deref(), Some("a"));
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].name, None);

        // The inclusive start pairs a filter with ITSELF: [a]null[a]
        // self-loops (C has no same-filter guard).
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "[a]null[a]").unwrap();
        assert!(ins.is_empty() && outs.is_empty());
        assert_eq!(g.links.len(), 1);
        assert_eq!(g.links[0].src, g.links[0].dst);

        // The forward search IS chain-crossing: a chain-0 input pairs with a
        // chain-1 output (f1.in "a" <- f2.out "a"), leaving only the two
        // unlabeled tail pads open.
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "[a]null;null[a]").unwrap();
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].name, None);
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].name, None);
        assert_eq!(g.links.len(), 1);
        assert_eq!(g.links[0].src, NodeId(1), "chain-1 filter is the producer");

        // Same-side labels never pair: two labeled INPUTS both stay open.
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "[a]null;[a]null").unwrap();
        assert_eq!(ins.len(), 2);
        assert_eq!(ins[0].name.as_deref(), Some("a"));
        assert_eq!(ins[1].name.as_deref(), Some("a"));
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].name, None);
    }

    // ---- unlabeled next-filter chaining (graphparser.c:785-802) ----------------

    #[test]
    fn unlabeled_chaining_scope() {
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "null,null").unwrap();
        assert!(g.nodes[0].outputs[0].is_some(), "',' links in-chain");
        assert_eq!(ins.len(), 1);
        assert_eq!(outs.len(), 1);

        // ';' ends the chain: the fallback never crosses it.
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "null;null").unwrap();
        assert!(g.nodes[0].outputs[0].is_none(), "';' breaks the chain");
        assert_eq!(ins.len(), 2);
        assert_eq!(outs.len(), 2);

        // A LABELED output skips the fallback entirely (the `&& !label` of
        // 785): the next filter's input stays open instead.
        let mut g = FilterGraph::new();
        let (ins, outs) = graph_parse2(&mut g, "null[x],null").unwrap();
        assert!(g.nodes[0].outputs[0].is_none());
        assert_eq!(outs[0].name.as_deref(), Some("x"));
        assert_eq!(ins[0].name, None);
    }

    // ---- pad-count overloads (graphparser.c:711-717, 756-762) ------------------

    #[test]
    fn pad_overload_errors() {
        let mut g = FilterGraph::new();
        assert_eq!(
            invalid_arg(parse_ptr(&mut g, "[a][b]null")),
            "More input link labels specified for filter 'null' than it has inputs: 2 > 1"
        );
        let mut g = FilterGraph::new();
        assert_eq!(
            invalid_arg(parse_ptr(&mut g, "null[a][b]")),
            "More output link labels specified for filter 'null' than it has outputs: 2 > 1"
        );
    }

    // ---- failure teardown (graphparser.c:1020-1030) -----------------------------

    #[test]
    fn failure_teardown() {
        // A mid-parse create failure clears EVERYTHING (C frees every
        // filter in the graph).
        let mut g = FilterGraph::new();
        assert!(parse_ptr(&mut g, "null;bogus").is_err());
        assert!(g.nodes.is_empty());
        assert!(g.links.is_empty());
        // ...including pre-existing nodes.
        let mut g = FilterGraph::new();
        let pre = g.create_filter("null", "").unwrap();
        let _ = pre;
        assert!(parse_ptr(&mut g, "bogus").is_err());
        assert!(g.nodes.is_empty());
        // A parse failure too (the end: block runs for ANY error).
        let mut g = FilterGraph::new();
        let err = invalid_arg(parse_ptr(&mut g, "[unclosed"));
        assert!(err.starts_with("Mismatched '['"), "{err}");
        assert!(g.nodes.is_empty());
    }

    // ---- user-list matching, the full form (graphparser.c:978-1018) ------------

    #[test]
    fn user_list_matching_full_form() {
        let mut g = FilterGraph::new();
        let src = g.alloc_test_src();
        let sink = g.alloc_test_sink();
        // CLI shape: the "out" entry (a SINK input) in open_inputs, the
        // "in" entry (a SOURCE output) in open_outputs.
        let mut ins = vec![InOut {
            name: Some("out".into()),
            node: sink,
            pad: 0,
        }];
        let mut outs = vec![
            InOut {
                name: Some("in".into()),
                node: src,
                pad: 0,
            },
            // Unnamed entries never match (extract_inout skips NULL names).
            InOut {
                name: None,
                node: sink,
                pad: 0,
            },
        ];
        graph_parse_ptr(&mut g, "null", &mut ins, &mut outs).unwrap();
        // Both named entries consumed and LINKED src -> null -> sink.
        assert!(ins.is_empty(), "the out entry was consumed");
        assert_eq!(outs.len(), 1, "only the unnamed entry remains");
        assert_eq!(outs[0].name, None);
        assert_eq!(g.links.len(), 2);
        let lsrc = g.nodes[src.0].outputs[0].unwrap();
        let lsink = g.nodes[sink.0].inputs[0].unwrap();
        let mid = g.links[lsrc.0].dst;
        assert_eq!(g.nodes[mid.0].def.name, "null");
        assert_eq!(g.links[lsink.0].src, mid);

        // The second call: pass the returned unmatched pads the other way.
        let mut g2 = FilterGraph::new();
        let src2 = g2.alloc_test_src();
        let (parsed_ins, parsed_outs) = parse_ptr(&mut g2, "null,null").unwrap();
        // parsed_ins: the "in" open pad; link a source to it manually.
        let mid2 = parsed_ins[0].node;
        g2.link(src2, 0, mid2, parsed_ins[0].pad).unwrap();
        let _ = parsed_outs;
        // extract_inout never matches None names: direct check.
        let mut list = vec![InOut {
            name: None,
            node: mid2,
            pad: 0,
        }];
        assert!(extract_inout("in", &mut list).is_none());
        assert_eq!(list.len(), 1);
    }

    // ---- end-to-end smoke: a config-able, runnable parsed graph ----------------

    #[test]
    fn end_to_end_smoke() {
        let mut g = FilterGraph::new();
        let (ins, outs) = parse_ptr(&mut g, "null").unwrap();
        let null = ins[0].node;
        assert_eq!(ins[0].pad, 0);
        assert_eq!(outs[0].node, null);
        let src = g.alloc_test_src();
        let sink = g.alloc_test_sink();
        let la = g.link(src, 0, null, ins[0].pad).unwrap();
        let lc = g.link(outs[0].node, outs[0].pad, sink, 0).unwrap();

        // Full config: validity -> negotiation -> link config (the parsed
        // graph is a first-class citizen of the engine).
        g.config().unwrap();
        assert_eq!(g.links[la.0].w, 8, "TestSrc geometry via config_props");
        assert!(g.links[la.0].format.is_some());

        // One frame through, then EOF propagation end to end.
        let negotiated = g.links[la.0].format.unwrap();
        let mut frame = Frame::alloc(negotiated, 8, 8).unwrap();
        frame.pts = 5;
        filter::filter_frame(&mut g, la, frame).unwrap();
        run_to_quiescence(&mut g);
        assert_eq!(g.links[lc.0].fifo.len(), 0, "sink consumed the frame");
        assert_eq!(g.links[lc.0].frame_count_in, 1);
        filter::set_in_status(&mut g, la, Error::Eof, 100);
        run_to_quiescence(&mut g);
        assert!(
            matches!(g.links[lc.0].status_in, Some(Error::Eof)),
            "EOF must reach the sink link"
        );
    }
}
