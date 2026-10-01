#!/usr/bin/env python3
"""Extract the H.264 CABAC tables from the FFmpeg C sources into
cabac_tables.rs (same pipeline as the CAVLC tables in tables.rs — the C
arrays are the spec, do not hand-edit the generated file).

Sources:
  libavcodec/cabac.c     ff_h264_cabac_tables (norm_shift | lps_range |
                         mlps_state | last_coeff_flag_offset_8x8)
  libavcodec/h264_cabac.c cabac_context_init_I[1024][2],
                         cabac_context_init_PB[3][1024][2]
"""

import re
import sys

FF = sys.argv[1] if len(sys.argv) > 1 else "FFmpeg/libavcodec"
OUT = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "src/codec/video/h264/cabac_tables.rs"
)

def parse_int(tok):
    return int(tok) & 0xFF  # C uint8_t initializers with negatives wrap


def extract_flat(text, name):
    """Extract a flat uint8_t initializer `{ ... };`."""
    m = re.search(re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{", text)
    if not m:
        sys.exit(f"{name} not found")
    body = text[m.end():]
    end = body.index("};")
    body = re.sub(r"//[^\n]*", "", body[:end])
    toks = [t for t in re.split(r"[,\s]+", body) if t]
    return [parse_int(t) for t in toks]


def extract_pairs(text, name):
    """Extract an array of { a, b } pairs (int8_t)."""
    m = re.search(re.escape(name) + r"\s*\[[^\]]*\](\[\d+\])?\s*=\s*\{", text)
    if not m:
        sys.exit(f"{name} not found")
    body = text[m.end():]
    end = body.index("};")
    body = body[:end]
    pairs = re.findall(r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}", body)
    return [(int(a), int(b)) for a, b in pairs]


def rs_array_u8(name, vals, doc, per_line=16):
    lines = [f"/// {doc}", f"pub static {name}: [u8; {len(vals)}] = ["]
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(str(v) for v in vals[i:i + per_line]) + ",")
    lines.append("];")
    return "\n".join(lines)


def rs_pairs(name, pairs, doc, per_line=8):
    lines = [f"/// {doc}", f"pub static {name}: [[i8; 2]; {len(pairs)}] = ["]
    for i in range(0, len(pairs), per_line):
        row = ", ".join(f"[{a}, {b}]" for a, b in pairs[i:i + per_line])
        lines.append("    " + row + ",")
    lines.append("];")
    return "\n".join(lines)


cabac_c = open(f"{FF}/cabac.c").read()
h264_cabac_c = open(f"{FF}/h264_cabac.c").read()

tables = extract_flat(
    cabac_c, "DECLARE_ASM_ALIGNED(1, const uint8_t, ff_h264_cabac_tables)"
)
assert len(tables) == 512 + 4 * 2 * 64 + 4 * 64 + 63, len(tables)

ctx_i = extract_pairs(h264_cabac_c, "cabac_context_init_I")
assert len(ctx_i) == 1024, len(ctx_i)

# cabac_context_init_PB[3][1024][2]: three consecutive inner initializers
# of 1024 pairs (comments stripped; the pair stream is the array contents).
m = re.search(r"cabac_context_init_PB\[3\]\[1024\]\[2\]\s*=\s*\{", h264_cabac_c)
if not m:
    sys.exit("cabac_context_init_PB not found")
body = h264_cabac_c[m.end():]
body = re.sub(r"//[^\n]*", "", body[: body.index("};")])
flat = re.findall(r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}", body)
assert len(flat) == 3 * 1024, len(flat)
pb = [
    [(int(a), int(b)) for a, b in flat[j * 1024:(j + 1) * 1024]] for j in range(3)
]

with open(OUT, "w") as f:
    f.write(
        """//! H.264 CABAC tables — extracted verbatim from `libavcodec/cabac.c`
//! (`ff_h264_cabac_tables`) and `libavcodec/h264_cabac.c`
//! (`cabac_context_init_I` / `cabac_context_init_PB`). Regenerate with
//! `tools/extract_cabac_tables.py`; do not hand-edit — the C arrays are
//! the spec here.
#![allow(clippy::all)]

/// `ff_h264_cabac_tables` (cabac.c:32), layout per the C offsets
/// (cabac.h): [0..512) norm_shift, [512..1024) lps_range,
/// [1024..1280) mlps_state, [1280..1343) last_coeff_flag_offset_8x8.
/// The uint8_t initializers with negative values (lps_range) wrap exactly
/// as C sees them.
pub static FF_H264_CABAC_TABLES: [u8; 1343] = [
"""
    )
    for i in range(0, 1343, 16):
        f.write("    " + ", ".join(str(v) for v in tables[i:i + 16]) + ",\n")
    f.write("];\n\n")

    f.write(
        rs_pairs(
            "CTX_INIT_I",
            ctx_i,
            "`cabac_context_init_I[1024][2]` (h264_cabac.c:50).",
        )
    )
    f.write("\n\n")
    for j in range(3):
        f.write(
            rs_pairs(
                f"CTX_INIT_PB_{j}",
                pb[j],
                f"`cabac_context_init_PB[{j}][1024][2]` (h264_cabac.c:362).",
            )
        )
        f.write("\n\n")

print(f"wrote {OUT}")
