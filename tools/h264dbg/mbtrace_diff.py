#!/usr/bin/env python3
"""Diff the per-MB CABAC decode traces of the instrumented FFmpeg tree
(tools/h264dbg/h264_harness.c) against the Rust port's H264_DUMP output.
Whitelists the comparable lines: CMB/SKIP, IROW, chroma_pred, cbp,
qscale, RES; C's bare "CMB x y" + SKIP pair maps to the Rust
"CMB x y SKIP" marker."""
import re
import sys

KEEP = re.compile(r"^(CMB |  (IROW|chroma_pred|cbp|qscale|RES))")


def norm_r(path):
    out = []
    for ln in open(path):
        ln = re.sub(r"type=Intra\((\d+)\)", r"IROW \1", ln)
        ln = re.sub(r"\s*type=\S+", "", ln)
        ln = re.sub(r"cbp-tail", "", ln)
        ln = re.sub(r"cbp=0x([0-9a-f]+)", lambda m: "cbp=" + str(int(m.group(1), 16)), ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        first = ln.split(" ")[0]
        if ln.startswith("CMB"):
            parts = ln.split(" ")
            if len(parts) >= 4 and parts[3] == "IROW":
                out.append(" ".join(parts[:3]))
                out.append(" ".join(parts[3:]))
            else:
                out.append(ln)
        elif first in ("IROW", "chroma_pred", "cbp", "qscale", "RES"):
            out.append(ln)
    return out


def norm_c(path):
    lines = [l.rstrip("\n") for l in open(path)]
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("CMB") and i + 1 < len(lines) and "SKIP" in lines[i + 1]:
            out.append(ln + " SKIP")
            i += 2
            continue
        if ln.startswith("CMB") or ln.split(" ")[0] in ("IROW", "chroma_pred", "cbp", "qscale", "RES"):
            ln = re.sub(r"\s+", " ", ln)
            out.append(ln)
        i += 1
    return out


def main():
    r = norm_r(sys.argv[1])
    c = norm_c(sys.argv[2])
    for i, (a, b) in enumerate(zip(r, c)):
        if a != b:
            print(f"FIRST DIFF at line {i}:")
            for j in range(max(0, i - 6), min(len(r), i + 4)):
                print(f"  R: [{r[j]}]")
                print(f"  C: [{c[j]}]")
            return
    print(f"identical for {min(len(r), len(c))} lines; lens R={len(r)} C={len(c)}")
    print("tail R:", r[-2:])
    print("tail C:", c[-2:])


if __name__ == "__main__":
    main()
