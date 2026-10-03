# H.264 bit-exactness debugging kit

Recipes for the CABAC/CAVLC bring-up workflow: generate fixtures, decode
them with the port and with the LOCAL FFmpeg tree (instrumented), and
diff the per-MB traces until every `max pixel diff = 0`.

## 1. Fixtures (`gen_fixtures.sh`)

`/tmp` is wiped periodically on this machine — regenerate before any
test run:

```sh
bash tools/h264dbg/gen_fixtures.sh
for t in cabac_testsrc2 cabac_gray cabac_black cabac_ms cabac_q18 \
         testsrc2 gray black ms; do
  H264_TEST=/tmp/h264_$t.h264 \
    cargo test -q --lib -- codec::video::h264::tests::decodes_all_intra_fixture --nocapture
done
```

Each `h264_<stem>.h264` needs its `h264_ref_<stem>.yuv` (a DEFAULT
ffmpeg decode). The CABAC fixtures use `-profile:v main -bf 0 -weightp 0`
(B slices + weighted prediction are Phase C). `H264_DUMP=1` adds the
port's per-MB trace.

## 2. C reference harness (`h264_harness.c`)

The local `FFmpeg/` tree can be rebuilt with just the h264 decoder and
instrumented for ground-truth traces (revert with its own git!):

```sh
cd FFmpeg
./configure --disable-everything --enable-decoder=h264 --enable-demuxer=h264 \
  --enable-parser=h264 --enable-protocol=file --enable-muxer=rawvideo \
  --disable-doc --disable-autodetect --disable-network --disable-iconv \
  --disable-asm --disable-swresample --disable-swscale
find libavcodec libavutil libavformat -name '*.o' -delete   # stale ELF objects!
make -j8 libavcodec/libavcodec.a libavutil/libavutil.a libavformat/libavformat.a
cd ..
cc -O1 -IFFmpeg -IFFmpeg/libavcodec -IFFmpeg/libavformat tools/h264dbg/h264_harness.c \
  FFmpeg/libavformat/libavformat.a FFmpeg/libavcodec/libavcodec.a \
  FFmpeg/libavutil/libavutil.a -framework CoreFoundation -framework VideoToolbox \
  -o /tmp/h264_harness
/tmp/h264_harness /tmp/h264_cabac_testsrc2.h264
```

Useful fprintf insertion points (all previously used, add again as
needed): `ff_h264_decode_mb_cabac` (mb start/type/cbp/qscale),
`decode_cabac_residual_internal` (per-block cat/n/coeff_count),
`decode_cabac_mb_skip`, `get_cabac_cbf_ctx`, `ff_h264_init_cabac_states`,
`decode_slice` CABAC init (byte offset + slice qp), `ff_h264_decode_ref_pic_marking`.

## 3. Trace diff (`mbtrace_diff.py`)

```sh
/tmp/h264_harness /tmp/h264_cabac_testsrc2.h264 2>&1 >/dev/null \
  | awk '/CABACSTART/{sn++} sn==2' > /tmp/c2.txt      # slice 2's C trace
H264_DUMP=1 H264_TEST=/tmp/h264_cabac_testsrc2.h264 \
  cargo test -q --lib -- codec::video::h264::tests::decodes_all_intra_fixture --nocapture \
  2>&1 | awk '/CABACSTART/{sn++} sn==2' > /tmp/r2.txt  # slice 2's Rust trace
python3 tools/h264dbg/mbtrace_diff.py /tmp/r2.txt /tmp/c2.txt
```

## 4. Engine-only probe (when the syntax layer is suspected)

`H264_DUMP`-gated `CABACSTART` prints the slice's qp/byte offset; dump
`&nal.rbsp[start..end]` and replay the first bins through a tiny C
program calling FFmpeg's own `ff_init_cabac_decoder` + `get_cabac` —
`cabac.rs` has a twin unit test (`engine_matches_c_probe`) that skips
unless `/tmp/cabac_slice.bin` exists.

## Bugs this kit caught (keep in mind when extending)

- CABAC chroma-pred ctx: BOTH neighbors contribute +1 (ctx 0..2) — not
  +1/+2 like cbp.
- `decode_cabac_mb_mvd` first-bin ctx: C's sign-shift expression
  `((amvd-3)>>31)` SUBTRACTS while below the threshold (amvd<3 → +0,
  3..32 → +1, ≥33 → +2 from ctxbase+2), the `(amvd>2)+(amvd>32)` form
  in the comment is dead code.
- nnz cache borders: for CABAC INTER MBs an unavailable neighbour is 0,
  not 64 (h264_mvpred.h:698/725).
- C's `mb_stride` is mb_width+1: raw `mb_xy-1`/`mb_xy-stride` ctx
  arithmetic (decode_cabac_mb_skip) hits the padding column at row
  starts — the port's flat mb_xy must guard mb_x/mb_y explicitly.
- New-picture detection: ONLY first_mb==0 starts a picture; later IDR
  slices (nal type 5) continue it. And `mb_skip_run` resets per SLICE.
