//! MP3 (MPEG audio layer 3) float decoder — port of the `libavcodec`
//! mpegaudio decode family, **float** variant
//! (`mpegaudiodec_float.c` = `mpegaudiodec_template.c` with `USE_FLOATS 1`).
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `ff_mpa_check_header` (`mpegaudiodecheader.h:62-79`) | [`ff_mpa_check_header`] |
//! | `avpriv_mpegaudio_decode_header` (`mpegaudiodecheader.c:34-118`) | [`avpriv_mpegaudio_decode_header`] |
//! | `ff_mpa_decode_header` (`mpegaudiodecheader.c:120-152`) | [`ff_mpa_decode_header`] |
//! | `ff_mpa_bitrate_tab`/`ff_mpa_freq_tab` (`mpegaudiotabs.h:27-37`) | [`FF_MPA_BITRATE_TAB`]/[`FF_MPA_FREQ_TAB`] |
//! | `ff_slen_table`, `ff_lsf_nsf_table`, `ff_mpa_huff_data` (`mpegaudiodec_common.c:52-348`) | [`FF_SLEN_TABLE`] etc. |
//! | `ff_band_size_long`/`ff_band_size_short`/`ff_mpa_pretab` (`mpegaudiodec_common.c:362-400`) | [`FF_BAND_SIZE_LONG`] etc. |
//! | `ff_band_index_long` build (`mpegaudiodec_common.c:450-457`) | [`Tables::band_index_long`] |
//! | `mpa_hufflens`/`mpa_huffsymbols` + `ff_vlc_init_from_lengths` (`mpegaudiodec_common.c:73-435`, `vlc.c:306-351`) | [`BigVlc`] — canonical codes assigned in array order, exactly `vlc.c:319-345` |
//! | `mpa_quad_bits`/`mpa_quad_codes` + `vlc_init` (`mpegaudiodec_common.c:352-360,438-447`) | [`QuadVlc`] |
//! | `exp_table_float`/`expval_table_float` (`mpegaudio_tablegen.h:48-84`) | [`Tables::exp_table`]/[`Tables::expval_table`] |
//! | `ff_table_4_3_exp`/`ff_table_4_3_value` (`mpegaudiodec_common_tablegen.h:44-69`) | [`Tables::table_4_3_exp`]/[`Tables::table_4_3_value`] |
//! | `ff_mdct_win_float` build (`mpegaudiodsp.c:30-79`) | [`Tables::mdct_win`] |
//! | `ff_mpa_synth_window_float` build (`mpegaudiodsp_template.c:197-224`) from `ff_mpa_enwindow` (`mpegaudiodsp_data.c:22-56`) | [`Tables::synth_window`] |
//! | `is_table` (`mpegaudiodec_float.c:43-50`), `csa_table` (`:55-72`) | [`IS_TABLE`]/[`CSA_TABLE`] |
//! | `is_table_lsf` build (`mpegaudiodec_template.c:264-278`) | [`Tables::is_table_lsf`] |
//! | `GetBitContext`/`bits_*` (`get_bits.h`/`bitstream_template.h`, safe reader) | [`GetBits`] |
//! | `GranuleDef` (`mpegaudiodec_template.c:58-75`) | [`GranuleDef`] |
//! | `region_offset2size`…`compute_band_indexes` (`:126-188`) | [`GranuleDef`] methods |
//! | `lsf_sf_expand`/`SPLIT` (`:659-686`) | [`lsf_sf_expand`] |
//! | `exponents_from_scale_factors` (`:688-723`) | [`MpaDecodeCore::exponents_from_scale_factors`] |
//! | `huffman_decode` (`:756-903`) | [`MpaDecodeCore::huffman_decode`] |
//! | `reorder_block` (`:908-939`) | [`MpaDecodeCore::reorder_block`] |
//! | `compute_stereo` (`:943-1071`) | [`MpaDecodeCore::compute_stereo`] |
//! | `compute_antialias` (`:1101-1129`) | [`MpaDecodeCore::compute_antialias`] |
//! | `compute_imdct` (`:1132-1209`) + `imdct12` (`:329-368`) | [`MpaDecodeCore::compute_imdct`]/[`imdct12`] |
//! | `mp_decode_layer3` (`:1212-1469`) | [`MpaDecodeCore::mp_decode_layer3`] |
//! | `mp_decode_frame` (`:1471-1556`) | [`MpaDecodeCore::mp_decode_frame`] |
//! | `decode_frame` (`:1558-1628`) | [`Mp3Decoder::decode_frame`] |
//! | `mp_flush` (`:1630-1636`) | [`Mp3Decoder::flush`] |
//! | `ff_dct32_float` (`dct32_template.c:126-288`) | [`dct32`] |
//! | `ff_imdct36_blocks_float`/`imdct36` (`mpegaudiodsp_template.c:274-371`) | [`imdct36_blocks`]/[`imdct36`] |
//! | `ff_mpadsp_apply_window_float` (`:123-174`) | [`apply_window`] |
//! | `ff_mpa_synth_filter_float` (`:178-195`) | [`mpa_synth_filter`] |
//!
//! ## What the decode pipeline is
//!
//! A frame is `[4-byte header][side info][main data]`. Layer 3 decodes
//! 1 (LSF) or 2 (MPEG-1) *granules* per channel; each granule is 576
//! spectral lines carried in the shared *bit reservoir* (C's `last_buf`):
//! `main_data_begin` says how many bytes of previous-frame reservoir the
//! granule's data starts in, and `switch_buffer`
//! (`mpegaudiodec_template.c:725-738`) hops the bit reader between the
//! reservoir view (`gb`) and the current frame's remainder (`in_gb`) as
//! the granule's `part2_3_length` window is consumed. Per granule:
//! scale factors → [exponents](MpaDecodeCore::exponents_from_scale_factors)
//! → Huffman requantize into `sb_hybrid[576]` → MS/intensity stereo →
//! reorder short blocks → antialias butterflies → IMDCT (36-point for
//! long bands, three 12-point for short) into `sb_samples[ch][36][32]`
//! → per-32-sample polyphase synthesis ([`dct32`] + [`apply_window`])
//! into the output frame. Output is `FLTP` (`OUT_FMT_P`,
//! `mpegaudiodec_float.c:39`).
//!
//! ## Skipped C paths (documented, with the C guard that keeps them out)
//!
//! | C path | Guard | Ported? |
//! |---|---|---|
//! | Layer 1 decode (`mp_decode_layer1`, `:397-465`), layer 2 (`mp_decode_layer2`, `:467-657`) + `ff_mpa_sblimit_table`/`ff_mpa_quant_steps`/`ff_mpa_quant_bits`/`ff_mpa_alloc_tables`/`ff_division_tabs`/`ff_scale_factor_modshift`/`scale_factor_mult`/`scale_factor_mult2` | `switch (s->layer)` (`:1481-1494`) | no — layer 3 only (this is the MP3 zone); a layer-1/2 header inside a packet returns `Error::Unsupported` from decode, and `Mp1`/`Mp2` codec ids are rejected at `init` |
//! | CRC verification (`handle_crc`, `:370-394`) | `s->err_recognition & AV_EF_CRCCHECK` (off by default) | no — the port has no `err_recognition` surface; the 16-bit CRC field is still consumed when `error_protection` is set (`:1478-1479`) so bit positions match |
//! | free-format frames (`bitrate_index == 0`, `:1586-1590`) | `avpriv_mpegaudio_decode_header` returns 1 | no — C's `decode_frame` also returns `AVERROR_INVALIDDATA` ("free format: prepare to compute frame size"); the frame-size search lives in the mp3 parser (not ported) |
//! | MP3ADU (`decode_frame_adu`, `:1643-1694`), MP3on4 (`:1697-1899`) | `CONFIG_MP3ADU*_DECODER` etc. | no — separate codec ids, not in the `CodecId` family |
//! | AHX (`avctx->codec_id != AV_CODEC_ID_AHX`, `:1612`) | codec id | no |
//! | `err_recognition` bitstream checks (`:862-894`) | `AV_EF_*` flags | no — same off-by-default guards |
//! | x86/ARM/MIPS SIMD dispatch (`ff_mpadsp_init_x86` …, `mpegaudiodsp.c:94-108`) | arch macros | no — scalar template only (identical math) |
//! | Xing/Info/VBRI tag skip | lives in `libavformat`'s mp3 demuxer in this tree (there is no `libavcodec/mp3dec.c`); the decoder itself only skips leading zero bytes and ID3v1 `TAG` blocks (`:1567-1581`) | ported exactly |
//!
//! ## Port notes (where Rust had to choose)
//!
//! * The reader is the *safe* C semantics (`bitstream_template.h`):
//! reads past the end return 0 bits and the position saturates at
//! `ceil(size_in_bits/8) * 8`. `skip_bits_long` with a negative count
//! follows the *legacy signed* `get_bits` reader (index += n, may go
//! negative) — the surrounding C code (`:1466-1467` clamps negative
//! positions back to 0) was written for that reader; FFmpeg's current
//! `bits_skip` takes an unsigned count and produces accidental wrapped
//! positions on the same corrupt inputs.
//! * C's `gb`/`in_gb` point into the packet / `last_buf`; here
//! [`GetBits`] owns a copy of its window (≤ ~2 KB per frame) — same
//! bytes, no lifetimes. Backstep reads that C performs past the logical
//! end of `gb`'s region (only possible on malformed streams; C reads
//! stale reservoir bytes) are clamped here.
//! * Negative exponents (`v0 < 0` is representable from
//! `gain - (scalefac << shift) + 400` with extreme `global_gain`) index
//! before `expval_table` in C (undefined behavior); here they clamp to
//! row 0. Values ≥ 512 clamp to row 511.
//! * One frame per packet: C's `decode_frame` returns the consumed byte
//! count so `decode.c` re-loops for multiple frames in one packet; the
//! port decodes the first frame and drops trailing bytes (the demuxer
//! zone feeds one frame per packet).
//! * Output frames are planar f32 (`FLTP`), the C default
//! (`avctx->sample_fmt = OUT_FMT_P` when no `request_sample_fmt`).

use crate::{
    codec::{
        packet::Packet,
        params::{CodecId, CodecParameters, MediaType},
        traits::AudioDecoder,
    },
    util::{
        audio_frame::AudioFrame,
        channel_layout::ChannelLayout,
        error::{Error, Result},
        samplefmt::SampleFormat,
    },
};

static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();

/// `exp2_lut` shared by both tablegen headers
/// (`mpegaudio_tablegen.h:51-56`, `mpegaudiodec_common_tablegen.h:46-51`).
const EXP2_LUT: [f64; 4] = [
    1.00000000000000000000,   // 2 ^ (0 * 0.25)
    1.18920711500272106672,   // 2 ^ (1 * 0.25)
    std::f64::consts::SQRT_2, // 2 ^ (2 * 0.25)
    1.68179283050742908606,   // 2 ^ (3 * 0.25)
];

// `#define C3 FIXHR(0.86602540378443864676/2)` etc.
// (mpegaudiodec_template.c:322-325 — imdct12's constants)
const I12_C3: f32 = fx(0.86602540378443864676 / 2.0);
const I12_C4: f32 = fx(0.70710678118654752439 / 2.0); // 0.5 / cos(pi*(9)/36)
const I12_C5: f32 = fx(0.51763809020504152469 / 2.0); // 0.5 / cos(pi*(5)/36)
const I12_C6: f32 = fx(1.93185165257813657349 / 4.0); // 0.5 / cos(pi*(15)/36)

// `cos(pi*i/18)` set — `#define C1..C8` (mpegaudiodsp_template.c:238-245)
const C1: f32 = fx(0.98480775301220805936 / 2.0);
const C2: f32 = fx(0.93969262078590838405 / 2.0);
const C3: f32 = fx(0.86602540378443864676 / 2.0);
const C4: f32 = fx(0.76604444311897803520 / 2.0);
const C5: f32 = fx(0.64278760968653932632 / 2.0);
#[allow(dead_code)] // defined by C's cos(pi*i/18) set; unused there too
const C6: f32 = fx(0.5 / 2.0);
const C7: f32 = fx(0.34202014332566873304 / 2.0);
const C8: f32 = fx(0.17364817766693034885 / 2.0);

/// `icos36[9]` — `0.5 / cos(pi*(2*i+1)/36)` (mpegaudiodsp_template.c:248-258).
const ICOS36: [f32; 9] = [
    fx(0.50190991877167369479),
    fx(0.51763809020504152469), //0
    fx(0.55168895948124587824),
    fx(0.61038729438072803416),
    fx(0.70710678118654752439), //1
    fx(0.87172339781054900991),
    fx(1.18310079157624925896),
    fx(1.93185165257813657349), //2
    fx(5.73685662283492756461),
];

/// `icos36h[9]` — same values halved (the /4 entries 6-7 stay /4)
/// (mpegaudiodsp_template.c:261-271).
const ICOS36H: [f32; 8] = [
    fx(0.50190991877167369479 / 2.0),
    fx(0.51763809020504152469 / 2.0), //0
    fx(0.55168895948124587824 / 2.0),
    fx(0.61038729438072803416 / 2.0),
    fx(0.70710678118654752439 / 2.0), //1
    fx(0.87172339781054900991 / 2.0),
    fx(1.18310079157624925896 / 4.0),
    fx(1.93185165257813657349 / 4.0), //2
];

// ---------------------------------------------------------------------
// Constants (mpegaudio.h)
// ---------------------------------------------------------------------

/// `MPA_MAX_CHANNELS` (`mpegaudio.h:42`).
pub const MPA_MAX_CHANNELS: usize = 2;
/// `SBLIMIT` — number of subbands (`mpegaudio.h:44`).
pub const SBLIMIT: usize = 32;
/// `MPA_FRAME_SIZE` — max frame size in samples (`mpegaudio.h:37`).
pub const MPA_FRAME_SIZE: usize = 1152;
/// `BACKSTEP_SIZE` (`mpegaudiodec_template.c:53`).
pub const BACKSTEP_SIZE: usize = 512;
/// `EXTRABYTES` (`mpegaudiodec_template.c:54`).
const EXTRABYTES: usize = 24;
/// `LAST_BUF_SIZE = 2 * BACKSTEP_SIZE + EXTRABYTES` (`:55`).
const LAST_BUF_SIZE: usize = 2 * BACKSTEP_SIZE + EXTRABYTES;

/// `MPA_STEREO`..`MPA_MONO` (`mpegaudio.h:46-49`).
const MPA_JSTEREO: i32 = 1;
const MPA_MONO: i32 = 3;
/// `MODE_EXT_I_STEREO` / `MODE_EXT_MS_STEREO` (`mpegaudiodata.h:37-38`).
const MODE_EXT_I_STEREO: i32 = 1;
const MODE_EXT_MS_STEREO: i32 = 2;

/// `IMDCT_SCALAR` (`mpegaudio.h:56`, `mpegaudio_tablegen.h:46`).
const IMDCT_SCALAR: f64 = 1.759;
/// `HEADER_SIZE` (`mpegaudiodec_template.c:101`).
const HEADER_SIZE: usize = 4;
/// `TABLE_4_3_SIZE` (`mpegaudiodata.h:48`).
const TABLE_4_3_SIZE: usize = (8191 + 16) * 4;
/// `MDCT_BUF_SIZE = FFALIGN(36, 2*4)` (`mpegaudiodsp.h:89`).
const MDCT_BUF_SIZE: usize = 40;

// ---------------------------------------------------------------------
// Static tables — mpegaudiotabs.h / mpegaudiodec_common.c / mpegaudiodec_float.c
// ---------------------------------------------------------------------

/// `ff_mpa_bitrate_tab[2][3][15]` (`mpegaudiotabs.h:27-35`), kbit/s.
pub const FF_MPA_BITRATE_TAB: [[[u16; 15]; 3]; 2] = [
    [
        [
            0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
        ],
        [
            0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
        ],
        [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ],
    ],
    [
        [
            0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
        ],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    ],
];

/// `ff_mpa_freq_tab[3]` (`mpegaudiotabs.h:37`).
pub const FF_MPA_FREQ_TAB: [u32; 3] = [44100, 48000, 32000];

/// `ff_slen_table[2][16]` (`mpegaudiodec_common.c:52-55`).
pub const FF_SLEN_TABLE: [[u8; 16]; 2] = [
    [0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3],
];

/// `ff_lsf_nsf_table[6][3][4]` (`mpegaudiodec_common.c:57-64`).
pub const FF_LSF_NSF_TABLE: [[[u8; 4]; 3]; 6] = [
    [[6, 5, 5, 5], [9, 9, 9, 9], [6, 9, 9, 9]],
    [[6, 5, 7, 3], [9, 9, 12, 6], [6, 9, 12, 6]],
    [[11, 10, 0, 0], [18, 18, 0, 0], [15, 18, 0, 0]],
    [[7, 7, 7, 0], [12, 12, 12, 0], [6, 15, 12, 0]],
    [[6, 6, 6, 3], [12, 9, 9, 6], [6, 12, 9, 6]],
    [[8, 8, 5, 0], [15, 12, 9, 0], [6, 18, 9, 0]],
];

/// `ff_mpa_huff_data[32][2]` (`mpegaudiodec_common.c:315-348`):
/// `[vlc table index, linbits]` per `table_select`.
pub const FF_MPA_HUFF_DATA: [[u8; 2]; 32] = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
    [0, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [10, 0],
    [11, 0],
    [12, 0],
    [0, 0],
    [13, 0],
    [14, 1],
    [14, 2],
    [14, 3],
    [14, 4],
    [14, 6],
    [14, 8],
    [14, 10],
    [14, 13],
    [15, 4],
    [15, 5],
    [15, 6],
    [15, 7],
    [15, 8],
    [15, 9],
    [15, 11],
    [15, 13],
];

/// `ff_band_size_long[9][22]` (`mpegaudiodec_common.c:362-381`).
pub const FF_BAND_SIZE_LONG: [[u8; 22]; 9] = [
    [
        4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76, 158,
    ], // 44100
    [
        4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 10, 12, 16, 18, 22, 28, 34, 40, 46, 54, 54, 192,
    ], // 48000
    [
        4, 4, 4, 4, 4, 4, 6, 6, 8, 10, 12, 16, 20, 24, 30, 38, 46, 56, 68, 84, 102, 26,
    ], // 32000
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54,
    ], // 22050
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 18, 22, 26, 32, 38, 46, 54, 62, 70, 76, 36,
    ], // 24000
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54,
    ], // 16000
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54,
    ], // 11025
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54,
    ], // 12000
    [
        12, 12, 12, 12, 12, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 76, 90, 2, 2, 2, 2, 2,
    ], // 8000
];

/// `ff_band_size_short[9][13]` (`mpegaudiodec_common.c:383-393`).
pub const FF_BAND_SIZE_SHORT: [[u8; 13]; 9] = [
    [4, 4, 4, 4, 6, 8, 10, 12, 14, 18, 22, 30, 56], // 44100
    [4, 4, 4, 4, 6, 6, 10, 12, 14, 16, 20, 26, 66], // 48000
    [4, 4, 4, 4, 6, 8, 12, 16, 20, 26, 34, 42, 12], // 32000
    [4, 4, 4, 6, 6, 8, 10, 14, 18, 26, 32, 42, 18], // 22050
    [4, 4, 4, 6, 8, 10, 12, 14, 18, 24, 32, 44, 12], // 24000
    [4, 4, 4, 6, 8, 10, 12, 14, 18, 24, 30, 40, 18], // 16000
    [4, 4, 4, 6, 8, 10, 12, 14, 18, 24, 30, 40, 18], // 11025
    [4, 4, 4, 6, 8, 10, 12, 14, 18, 24, 30, 40, 18], // 12000
    [8, 8, 8, 12, 16, 20, 24, 28, 36, 2, 2, 2, 26], // 8000
];

/// `ff_mpa_pretab[2][22]` (`mpegaudiodec_common.c:397-400`).
pub const FF_MPA_PRETAB: [[u8; 22]; 2] = [
    [0; 22],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0,
    ],
];

/// `is_table[2][16]` (`mpegaudiodec_float.c:43-50`) — intensity stereo
/// coefficients (MPEG-1; `sf_max = 7`). C initializes the first 7
/// entries of each 16-wide row; the rest stay zero.
pub const IS_TABLE: [[f32; 16]; 2] = [
    [
        0.000000000000000000e+00,
        2.113248705863952637e-01,
        3.660253882408142090e-01,
        5.000000000000000000e-01,
        6.339746117591857910e-01,
        7.886751294136047363e-01,
        1.000000000000000000e+00,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.000000000000000000e+00,
        7.886751294136047363e-01,
        6.339746117591857910e-01,
        5.000000000000000000e-01,
        3.660253882408142090e-01,
        2.113248705863952637e-01,
        0.000000000000000000e+00,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
];

/// `csa_table[8][4]` (`mpegaudiodec_float.c:55-72`) — antialias
/// butterfly coefficients.
pub const CSA_TABLE: [[f32; 4]; 8] = [
    [
        8.574929237365722656e-01,
        -5.144957900047302246e-01,
        3.429971337318420410e-01,
        -1.371988654136657715e+00,
    ],
    [
        8.817420005798339844e-01,
        -4.717319905757904053e-01,
        4.100100100040435791e-01,
        -1.353474020957946777e+00,
    ],
    [
        9.496286511421203613e-01,
        -3.133774697780609131e-01,
        6.362511515617370605e-01,
        -1.263006091117858887e+00,
    ],
    [
        9.833145737648010254e-01,
        -1.819131970405578613e-01,
        8.014013767242431641e-01,
        -1.165227770805358887e+00,
    ],
    [
        9.955177903175354004e-01,
        -9.457419067621231079e-02,
        9.009436368942260742e-01,
        -1.090092062950134277e+00,
    ],
    [
        9.991605877876281738e-01,
        -4.096558317542076111e-02,
        9.581949710845947266e-01,
        -1.040126085281372070e+00,
    ],
    [
        9.998992085456848145e-01,
        -1.419856864959001541e-02,
        9.857006072998046875e-01,
        -1.014097809791564941e+00,
    ],
    [
        9.999931454658508301e-01,
        -3.699974622577428818e-03,
        9.962931871414184570e-01,
        -1.003693103790283203e+00,
    ],
];

/// `ff_mpa_enwindow[257]` (`mpegaudiodsp_data.c:22-56`) — the half
/// MPEG synthesis window in Q16 integer form.
pub const FF_MPA_ENWINDOW: [i32; 257] = [
    0, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -4, -4, -5, -5, -6, -7, -7, -8, -9, -10,
    -11, -13, -14, -16, -17, -19, -21, -24, -26, -29, -31, -35, -38, -41, -45, -49, -53, -58, -63,
    -68, -73, -79, -85, -91, -97, -104, -111, -117, -125, -132, -139, -147, -154, -161, -169, -176,
    -183, -190, -196, -202, -208, 213, 218, 222, 225, 227, 228, 228, 227, 224, 221, 215, 208, 200,
    189, 177, 163, 146, 127, 106, 83, 57, 29, -2, -36, -72, -111, -153, -197, -244, -294, -347,
    -401, -459, -519, -581, -645, -711, -779, -848, -919, -991, -1064, -1137, -1210, -1283, -1356,
    -1428, -1498, -1567, -1634, -1698, -1759, -1817, -1870, -1919, -1962, -2001, -2032, -2057,
    -2075, -2085, -2087, -2080, -2063, 2037, 2000, 1952, 1893, 1822, 1739, 1644, 1535, 1414, 1280,
    1131, 970, 794, 605, 402, 185, -45, -288, -545, -814, -1095, -1388, -1692, -2006, -2330, -2663,
    -3004, -3351, -3705, -4063, -4425, -4788, -5153, -5517, -5879, -6237, -6589, -6935, -7271,
    -7597, -7910, -8209, -8491, -8755, -8998, -9219, -9416, -9585, -9727, -9838, -9916, -9959,
    -9966, -9935, -9863, -9750, -9592, -9389, -9139, -8840, -8492, -8092, -7640, -7134, 6574, 5959,
    5288, 4561, 3776, 2935, 2037, 1082, 70, -998, -2122, -3300, -4533, -5818, -7154, -8540, -9975,
    -11455, -12980, -14548, -16155, -17799, -19478, -21189, -22929, -24694, -26482, -28289, -30112,
    -31947, -33791, -35640, -37489, -39336, -41176, -43006, -44821, -46617, -48390, -50137, -51853,
    -53534, -55178, -56778, -58333, -59838, -61289, -62684, -64019, -65290, -66494, -67629, -68692,
    -69679, -70590, -71420, -72169, -72835, -73415, -73908, -74313, -74630, -74856, -74992, 75038,
];

// ---------------------------------------------------------------------
// Layer 3 Huffman tables — mpegaudiodec_common.c:73-360
// ---------------------------------------------------------------------

/// `mpa_hufflens[]` (`mpegaudiodec_common.c:73-168`), 15 tables
/// concatenated (tables 1,2,3,5,6,7,8,9,10,11,12,13,15,16,24).
const MPA_HUFFLENS: &[u8] = &[
    // Huffman table 1 - 4 entries
    3, 3, 2, 1, // Huffman table 2 - 9 entries
    6, 6, 5, 5, 5, 3, 3, 3, 1, // Huffman table 3 - 9 entries
    6, 6, 5, 5, 5, 3, 2, 2, 2, // Huffman table 5 - 16 entries
    8, 8, 7, 6, 7, 7, 7, 7, 6, 6, 6, 6, 3, 3, 3, 1, // Huffman table 6 - 16 entries
    7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 3, 2, 3, 3, // Huffman table 7 - 36 entries
    10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 9, 9, 8, 9, 9, 8, 8, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 6, 5,
    6, 6, 4, 3, 3, 1, // Huffman table 8 - 36 entries
    11, 11, 10, 9, 10, 10, 9, 9, 9, 8, 8, 9, 9, 9, 9, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6,
    4, 4, 2, 3, 3, 2, // Huffman table 9 - 36 entries
    9, 9, 8, 8, 9, 9, 8, 8, 8, 8, 7, 7, 7, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 6, 6, 5, 5, 4, 4, 4,
    3, 3, 3, 3, // Huffman table 10 - 64 entries
    11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 11, 11, 10, 9, 9, 10, 10, 9, 9, 10, 10, 9,
    10, 10, 8, 8, 9, 9, 10, 10, 9, 9, 10, 10, 8, 8, 8, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 7, 7, 7,
    7, 6, 6, 6, 6, 4, 3, 3, 1, // Huffman table 11 - 64 entries
    10, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 9, 9, 9, 10, 10, 10, 10, 8, 8, 9, 9, 7, 8, 8, 8, 8,
    8, 9, 9, 9, 9, 8, 7, 8, 8, 7, 7, 8, 8, 8, 9, 9, 8, 8, 8, 8, 8, 8, 7, 7, 6, 6, 7, 7, 6, 5, 4, 5,
    5, 3, 3, 3, 2, // Huffman table 12 - 64 entries
    10, 10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 8, 8, 8, 8, 8, 9, 9, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 7, 7, 7, 6, 6, 6, 6, 7, 7, 6, 5, 5, 5, 4, 4, 5, 5, 4, 3, 3,
    3, // Huffman table 13 - 256 entries
    19, 19, 18, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 15, 15, 16, 16, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 16, 16, 15, 16, 16, 14, 14, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 15, 15, 14, 13, 14, 14, 13, 13, 14, 14, 13, 14, 14, 13, 14, 14, 13, 14, 14, 13,
    13, 14, 14, 12, 12, 12, 13, 13, 13, 13, 13, 13, 12, 13, 13, 12, 12, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 12, 12, 13, 13, 12, 12, 12, 12, 13, 13, 13, 13, 12, 13, 13, 12, 11, 12, 12,
    12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 12, 12, 11, 11, 12, 12, 11, 12, 12, 12, 12, 11, 11, 12,
    12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10,
    10, 11, 11, 10, 11, 11, 10, 11, 11, 11, 11, 10, 10, 11, 11, 10, 10, 11, 11, 11, 11, 11, 11, 9,
    9, 10, 10, 10, 10, 10, 11, 11, 9, 9, 9, 10, 10, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    8, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9, 8, 8, 9, 9, 9, 9, 9, 9, 8, 7, 8, 8, 8, 8, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 4, 4, 3, 1, // Huffman table 15 - 256 entries
    13, 13, 13, 13, 12, 13, 13, 13, 13, 13, 13, 12, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 11, 11, 12, 12, 12, 12, 11, 11, 11,
    11, 11, 11, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 11, 11, 11, 11, 11, 11,
    10, 11, 11, 11, 11, 11, 11, 10, 10, 11, 11, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 10,
    11, 11, 10, 10, 10, 10, 10, 11, 11, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10,
    10, 10, 10, 9, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9,
    9, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 9, 9, 9, 9, 9,
    9, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 8, 8, 8, 8, 8, 8, 8, 9, 9, 8, 7, 8, 8, 7, 7, 7, 7, 8, 8, 7, 7,
    7, 7, 7, 6, 7, 7, 6, 6, 7, 7, 6, 6, 6, 5, 5, 5, 5, 5, 3, 4, 4, 3,
    // Huffman table 16 - 256 entries
    11, 11, 11, 11, 11, 11, 11, 11, 10, 11, 11, 11, 11, 10, 10, 10, 10, 10, 8, 10, 10, 9, 9, 9, 9,
    10, 16, 17, 17, 15, 15, 16, 16, 14, 15, 15, 14, 14, 15, 15, 14, 14, 15, 15, 15, 15, 14, 15, 15,
    14, 13, 8, 9, 9, 8, 8, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 14, 13,
    14, 14, 13, 13, 13, 14, 14, 14, 14, 13, 13, 14, 14, 13, 14, 14, 12, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 12, 13, 13, 13, 13, 13, 13, 12, 13, 13, 12, 12, 13, 13, 11, 12, 12,
    12, 12, 12, 12, 12, 13, 13, 11, 12, 12, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12, 12,
    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12,
    11, 10, 10, 11, 11, 11, 11, 11, 11, 10, 10, 11, 11, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 10,
    11, 11, 10, 10, 10, 11, 11, 10, 10, 11, 11, 10, 10, 11, 11, 10, 9, 9, 10, 10, 10, 10, 10, 10,
    9, 9, 9, 10, 10, 9, 10, 10, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 9, 9, 8, 8, 7, 7, 8, 8, 7,
    6, 6, 6, 6, 4, 4, 3, 1, // Huffman table 24 - 256 entries
    8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 7, 7, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 4, 11, 11, 11, 11, 12, 12, 11, 10, 11, 11, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10,
    11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 10, 11, 11, 10, 9, 10, 10, 10, 10, 11, 11, 10, 9, 9,
    10, 10, 9, 10, 10, 10, 10, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9, 10, 10, 8, 9, 9, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 7, 8, 8, 7, 7, 7, 7, 7, 8, 8, 7,
    7, 6, 6, 7, 7, 6, 5, 5, 6, 6, 4, 4, 4, 4,
];

/// `mpa_huffsymbols[]` (`mpegaudiodec_common.c:170-308`), same
/// concatenation; each byte packs `(x << 4) | y`.
const MPA_HUFFSYMBOLS: &[u8] = &[
    // Huffman table 1 - 4 entries
    0x11, 0x01, 0x10, 0x00, // Huffman table 2 - 9 entries
    0x22, 0x02, 0x12, 0x21, 0x20, 0x11, 0x01, 0x10, 0x00, // Huffman table 3 - 9 entries
    0x22, 0x02, 0x12, 0x21, 0x20, 0x10, 0x11, 0x01, 0x00, // Huffman table 5 - 16 entries
    0x33, 0x23, 0x32, 0x31, 0x13, 0x03, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
    // Huffman table 6 - 16 entries
    0x33, 0x03, 0x23, 0x32, 0x30, 0x13, 0x31, 0x22, 0x02, 0x12, 0x21, 0x20, 0x01, 0x11, 0x10, 0x00,
    // Huffman table 7 - 36 entries
    0x55, 0x45, 0x54, 0x53, 0x35, 0x44, 0x25, 0x52, 0x15, 0x51, 0x05, 0x34, 0x50, 0x43, 0x33, 0x24,
    0x42, 0x14, 0x41, 0x40, 0x04, 0x23, 0x32, 0x03, 0x13, 0x31, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20,
    0x11, 0x01, 0x10, 0x00, // Huffman table 8 - 36 entries
    0x55, 0x54, 0x45, 0x53, 0x35, 0x44, 0x25, 0x52, 0x05, 0x15, 0x51, 0x34, 0x43, 0x50, 0x33, 0x24,
    0x42, 0x14, 0x41, 0x04, 0x40, 0x23, 0x32, 0x13, 0x31, 0x03, 0x30, 0x22, 0x02, 0x20, 0x12, 0x21,
    0x11, 0x01, 0x10, 0x00, // Huffman table 9 - 36 entries
    0x55, 0x45, 0x35, 0x53, 0x54, 0x05, 0x44, 0x25, 0x52, 0x15, 0x51, 0x34, 0x43, 0x50, 0x04, 0x24,
    0x42, 0x33, 0x40, 0x14, 0x41, 0x23, 0x32, 0x13, 0x31, 0x03, 0x30, 0x22, 0x02, 0x12, 0x21, 0x20,
    0x11, 0x01, 0x10, 0x00, // Huffman table 10 - 64 entries
    0x77, 0x67, 0x76, 0x57, 0x75, 0x66, 0x47, 0x74, 0x56, 0x65, 0x37, 0x73, 0x46, 0x55, 0x54, 0x63,
    0x27, 0x72, 0x64, 0x07, 0x70, 0x62, 0x45, 0x35, 0x06, 0x53, 0x44, 0x17, 0x71, 0x36, 0x26, 0x25,
    0x52, 0x15, 0x51, 0x34, 0x43, 0x16, 0x61, 0x60, 0x05, 0x50, 0x24, 0x42, 0x33, 0x04, 0x14, 0x41,
    0x40, 0x23, 0x32, 0x03, 0x13, 0x31, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
    // Huffman table 11 - 64 entries
    0x77, 0x67, 0x76, 0x75, 0x66, 0x47, 0x74, 0x57, 0x55, 0x56, 0x65, 0x37, 0x73, 0x46, 0x45, 0x54,
    0x35, 0x53, 0x27, 0x72, 0x64, 0x07, 0x71, 0x17, 0x70, 0x36, 0x63, 0x60, 0x44, 0x25, 0x52, 0x05,
    0x15, 0x62, 0x26, 0x06, 0x16, 0x61, 0x51, 0x34, 0x50, 0x43, 0x33, 0x24, 0x42, 0x14, 0x41, 0x04,
    0x40, 0x23, 0x32, 0x13, 0x31, 0x03, 0x30, 0x22, 0x21, 0x12, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
    // Huffman table 12 - 64 entries
    0x77, 0x67, 0x76, 0x57, 0x75, 0x66, 0x47, 0x74, 0x65, 0x56, 0x37, 0x73, 0x55, 0x27, 0x72, 0x46,
    0x64, 0x17, 0x71, 0x07, 0x70, 0x36, 0x63, 0x45, 0x54, 0x44, 0x06, 0x05, 0x26, 0x62, 0x61, 0x16,
    0x60, 0x35, 0x53, 0x25, 0x52, 0x15, 0x51, 0x34, 0x43, 0x50, 0x04, 0x24, 0x42, 0x14, 0x33, 0x41,
    0x23, 0x32, 0x40, 0x03, 0x30, 0x13, 0x31, 0x22, 0x12, 0x21, 0x02, 0x20, 0x00, 0x11, 0x01, 0x10,
    // Huffman table 13 - 256 entries
    0xFE, 0xFC, 0xFD, 0xED, 0xFF, 0xEF, 0xDF, 0xEE, 0xCF, 0xDE, 0xBF, 0xFB, 0xCE, 0xDC, 0xAF, 0xE9,
    0xEC, 0xDD, 0xFA, 0xCD, 0xBE, 0xEB, 0x9F, 0xF9, 0xEA, 0xBD, 0xDB, 0x8F, 0xF8, 0xCC, 0xAE, 0x9E,
    0x8E, 0x7F, 0x7E, 0xF7, 0xDA, 0xAD, 0xBC, 0xCB, 0xF6, 0x6F, 0xE8, 0x5F, 0x9D, 0xD9, 0xF5, 0xE7,
    0xAC, 0xBB, 0x4F, 0xF4, 0xCA, 0xE6, 0xF3, 0x3F, 0x8D, 0xD8, 0x2F, 0xF2, 0x6E, 0x9C, 0x0F, 0xC9,
    0x5E, 0xAB, 0x7D, 0xD7, 0x4E, 0xC8, 0xD6, 0x3E, 0xB9, 0x9B, 0xAA, 0x1F, 0xF1, 0xF0, 0xBA, 0xE5,
    0xE4, 0x8C, 0x6D, 0xE3, 0xE2, 0x2E, 0x0E, 0x1E, 0xE1, 0xE0, 0x5D, 0xD5, 0x7C, 0xC7, 0x4D, 0x8B,
    0xB8, 0xD4, 0x9A, 0xA9, 0x6C, 0xC6, 0x3D, 0xD3, 0x7B, 0x2D, 0xD2, 0x1D, 0xB7, 0x5C, 0xC5, 0x99,
    0x7A, 0xC3, 0xA7, 0x97, 0x4B, 0xD1, 0x0D, 0xD0, 0x8A, 0xA8, 0x4C, 0xC4, 0x6B, 0xB6, 0x3C, 0x2C,
    0xC2, 0x5B, 0xB5, 0x89, 0x1C, 0xC1, 0x98, 0x0C, 0xC0, 0xB4, 0x6A, 0xA6, 0x79, 0x3B, 0xB3, 0x88,
    0x5A, 0x2B, 0xA5, 0x69, 0xA4, 0x78, 0x87, 0x94, 0x77, 0x76, 0xB2, 0x1B, 0xB1, 0x0B, 0xB0, 0x96,
    0x4A, 0x3A, 0xA3, 0x59, 0x95, 0x2A, 0xA2, 0x1A, 0xA1, 0x0A, 0x68, 0xA0, 0x86, 0x49, 0x93, 0x39,
    0x58, 0x85, 0x67, 0x29, 0x92, 0x57, 0x75, 0x38, 0x83, 0x66, 0x47, 0x74, 0x56, 0x65, 0x73, 0x19,
    0x91, 0x09, 0x90, 0x48, 0x84, 0x72, 0x46, 0x64, 0x28, 0x82, 0x18, 0x37, 0x27, 0x17, 0x71, 0x55,
    0x07, 0x70, 0x36, 0x63, 0x45, 0x54, 0x26, 0x62, 0x35, 0x81, 0x08, 0x80, 0x16, 0x61, 0x06, 0x60,
    0x53, 0x44, 0x25, 0x52, 0x05, 0x15, 0x51, 0x34, 0x43, 0x50, 0x24, 0x42, 0x33, 0x14, 0x41, 0x04,
    0x40, 0x23, 0x32, 0x13, 0x31, 0x03, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
    // Huffman table 15 - 256 entries
    0xFF, 0xEF, 0xFE, 0xDF, 0xEE, 0xFD, 0xCF, 0xFC, 0xDE, 0xED, 0xBF, 0xFB, 0xCE, 0xEC, 0xDD, 0xAF,
    0xFA, 0xBE, 0xEB, 0xCD, 0xDC, 0x9F, 0xF9, 0xEA, 0xBD, 0xDB, 0x8F, 0xF8, 0xCC, 0x9E, 0xE9, 0x7F,
    0xF7, 0xAD, 0xDA, 0xBC, 0x6F, 0xAE, 0x0F, 0xCB, 0xF6, 0x8E, 0xE8, 0x5F, 0x9D, 0xF5, 0x7E, 0xE7,
    0xAC, 0xCA, 0xBB, 0xD9, 0x8D, 0x4F, 0xF4, 0x3F, 0xF3, 0xD8, 0xE6, 0x2F, 0xF2, 0x6E, 0xF0, 0x1F,
    0xF1, 0x9C, 0xC9, 0x5E, 0xAB, 0xBA, 0xE5, 0x7D, 0xD7, 0x4E, 0xE4, 0x8C, 0xC8, 0x3E, 0x6D, 0xD6,
    0xE3, 0x9B, 0xB9, 0x2E, 0xAA, 0xE2, 0x1E, 0xE1, 0x0E, 0xE0, 0x5D, 0xD5, 0x7C, 0xC7, 0x4D, 0x8B,
    0xD4, 0xB8, 0x9A, 0xA9, 0x6C, 0xC6, 0x3D, 0xD3, 0xD2, 0x2D, 0x0D, 0x1D, 0x7B, 0xB7, 0xD1, 0x5C,
    0xD0, 0xC5, 0x8A, 0xA8, 0x4C, 0xC4, 0x6B, 0xB6, 0x99, 0x0C, 0x3C, 0xC3, 0x7A, 0xA7, 0xA6, 0xC0,
    0x0B, 0xC2, 0x2C, 0x5B, 0xB5, 0x1C, 0x89, 0x98, 0xC1, 0x4B, 0xB4, 0x6A, 0x3B, 0x79, 0xB3, 0x97,
    0x88, 0x2B, 0x5A, 0xB2, 0xA5, 0x1B, 0xB1, 0xB0, 0x69, 0x96, 0x4A, 0xA4, 0x78, 0x87, 0x3A, 0xA3,
    0x59, 0x95, 0x2A, 0xA2, 0x1A, 0xA1, 0x0A, 0xA0, 0x68, 0x86, 0x49, 0x94, 0x39, 0x93, 0x77, 0x09,
    0x58, 0x85, 0x29, 0x67, 0x76, 0x92, 0x91, 0x19, 0x90, 0x48, 0x84, 0x57, 0x75, 0x38, 0x83, 0x66,
    0x47, 0x28, 0x82, 0x18, 0x81, 0x74, 0x08, 0x80, 0x56, 0x65, 0x37, 0x73, 0x46, 0x27, 0x72, 0x64,
    0x17, 0x55, 0x71, 0x07, 0x70, 0x36, 0x63, 0x45, 0x54, 0x26, 0x62, 0x16, 0x06, 0x60, 0x35, 0x61,
    0x53, 0x44, 0x25, 0x52, 0x15, 0x51, 0x05, 0x50, 0x34, 0x43, 0x24, 0x42, 0x33, 0x41, 0x14, 0x04,
    0x23, 0x32, 0x40, 0x03, 0x13, 0x31, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
    // Huffman table 16 - 256 entries
    0xEF, 0xFE, 0xDF, 0xFD, 0xCF, 0xFC, 0xBF, 0xFB, 0xAF, 0xFA, 0x9F, 0xF9, 0xF8, 0x8F, 0x7F, 0xF7,
    0x6F, 0xF6, 0xFF, 0x5F, 0xF5, 0x4F, 0xF4, 0xF3, 0xF0, 0x3F, 0xCE, 0xEC, 0xDD, 0xDE, 0xE9, 0xEA,
    0xD9, 0xEE, 0xED, 0xEB, 0xBE, 0xCD, 0xDC, 0xDB, 0xAE, 0xCC, 0xAD, 0xDA, 0x7E, 0xAC, 0xCA, 0xC9,
    0x7D, 0x5E, 0xBD, 0xF2, 0x2F, 0x0F, 0x1F, 0xF1, 0x9E, 0xBC, 0xCB, 0x8E, 0xE8, 0x9D, 0xE7, 0xBB,
    0x8D, 0xD8, 0x6E, 0xE6, 0x9C, 0xAB, 0xBA, 0xE5, 0xD7, 0x4E, 0xE4, 0x8C, 0xC8, 0x3E, 0x6D, 0xD6,
    0x9B, 0xB9, 0xAA, 0xE1, 0xD4, 0xB8, 0xA9, 0x7B, 0xB7, 0xD0, 0xE3, 0x0E, 0xE0, 0x5D, 0xD5, 0x7C,
    0xC7, 0x4D, 0x8B, 0x9A, 0x6C, 0xC6, 0x3D, 0x5C, 0xC5, 0x0D, 0x8A, 0xA8, 0x99, 0x4C, 0xB6, 0x7A,
    0x3C, 0x5B, 0x89, 0x1C, 0xC0, 0x98, 0x79, 0xE2, 0x2E, 0x1E, 0xD3, 0x2D, 0xD2, 0xD1, 0x3B, 0x97,
    0x88, 0x1D, 0xC4, 0x6B, 0xC3, 0xA7, 0x2C, 0xC2, 0xB5, 0xC1, 0x0C, 0x4B, 0xB4, 0x6A, 0xA6, 0xB3,
    0x5A, 0xA5, 0x2B, 0xB2, 0x1B, 0xB1, 0x0B, 0xB0, 0x69, 0x96, 0x4A, 0xA4, 0x78, 0x87, 0xA3, 0x3A,
    0x59, 0x2A, 0x95, 0x68, 0xA1, 0x86, 0x77, 0x94, 0x49, 0x57, 0x67, 0xA2, 0x1A, 0x0A, 0xA0, 0x39,
    0x93, 0x58, 0x85, 0x29, 0x92, 0x76, 0x09, 0x19, 0x91, 0x90, 0x48, 0x84, 0x75, 0x38, 0x83, 0x66,
    0x28, 0x82, 0x47, 0x74, 0x18, 0x81, 0x80, 0x08, 0x56, 0x37, 0x73, 0x65, 0x46, 0x27, 0x72, 0x64,
    0x55, 0x07, 0x17, 0x71, 0x70, 0x36, 0x63, 0x45, 0x54, 0x26, 0x62, 0x16, 0x61, 0x06, 0x60, 0x53,
    0x35, 0x44, 0x25, 0x52, 0x51, 0x15, 0x05, 0x34, 0x43, 0x50, 0x24, 0x42, 0x33, 0x14, 0x41, 0x04,
    0x40, 0x23, 0x32, 0x13, 0x31, 0x03, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
    // Huffman table 24 - 256 entries
    0xEF, 0xFE, 0xDF, 0xFD, 0xCF, 0xFC, 0xBF, 0xFB, 0xFA, 0xAF, 0x9F, 0xF9, 0xF8, 0x8F, 0x7F, 0xF7,
    0x6F, 0xF6, 0x5F, 0xF5, 0x4F, 0xF4, 0x3F, 0xF3, 0x2F, 0xF2, 0xF1, 0x1F, 0xF0, 0x0F, 0xEE, 0xDE,
    0xED, 0xCE, 0xEC, 0xDD, 0xBE, 0xEB, 0xCD, 0xDC, 0xAE, 0xEA, 0xBD, 0xDB, 0xCC, 0x9E, 0xE9, 0xAD,
    0xDA, 0xBC, 0xCB, 0x8E, 0xE8, 0x9D, 0xD9, 0x7E, 0xE7, 0xAC, 0xFF, 0xCA, 0xBB, 0x8D, 0xD8, 0x0E,
    0xE0, 0x0D, 0xE6, 0x6E, 0x9C, 0xC9, 0x5E, 0xBA, 0xE5, 0xAB, 0x7D, 0xD7, 0xE4, 0x8C, 0xC8, 0x4E,
    0x2E, 0x3E, 0x6D, 0xD6, 0xE3, 0x9B, 0xB9, 0xAA, 0xE2, 0x1E, 0xE1, 0x5D, 0xD5, 0x7C, 0xC7, 0x4D,
    0x8B, 0xB8, 0xD4, 0x9A, 0xA9, 0x6C, 0xC6, 0x3D, 0xD3, 0x2D, 0xD2, 0x1D, 0x7B, 0xB7, 0xD1, 0x5C,
    0xC5, 0x8A, 0xA8, 0x99, 0x4C, 0xC4, 0x6B, 0xB6, 0xD0, 0x0C, 0x3C, 0xC3, 0x7A, 0xA7, 0x2C, 0xC2,
    0x5B, 0xB5, 0x1C, 0x89, 0x98, 0xC1, 0x4B, 0xC0, 0x0B, 0x3B, 0xB0, 0x0A, 0x1A, 0xB4, 0x6A, 0xA6,
    0x79, 0x97, 0xA0, 0x09, 0x90, 0xB3, 0x88, 0x2B, 0x5A, 0xB2, 0xA5, 0x1B, 0xB1, 0x69, 0x96, 0xA4,
    0x4A, 0x78, 0x87, 0x3A, 0xA3, 0x59, 0x95, 0x2A, 0xA2, 0xA1, 0x68, 0x86, 0x77, 0x49, 0x94, 0x39,
    0x93, 0x58, 0x85, 0x29, 0x67, 0x76, 0x92, 0x19, 0x91, 0x48, 0x84, 0x57, 0x75, 0x38, 0x83, 0x66,
    0x28, 0x82, 0x18, 0x47, 0x74, 0x81, 0x08, 0x80, 0x56, 0x65, 0x17, 0x07, 0x70, 0x73, 0x37, 0x27,
    0x72, 0x46, 0x64, 0x55, 0x71, 0x36, 0x63, 0x45, 0x54, 0x26, 0x62, 0x16, 0x61, 0x06, 0x60, 0x35,
    0x53, 0x44, 0x25, 0x52, 0x15, 0x05, 0x50, 0x51, 0x34, 0x43, 0x24, 0x42, 0x33, 0x14, 0x41, 0x04,
    0x40, 0x23, 0x32, 0x13, 0x31, 0x03, 0x30, 0x22, 0x12, 0x21, 0x02, 0x20, 0x11, 0x01, 0x10, 0x00,
];

/// `mpa_huff_sizes_minus_one[]` (`mpegaudiodec_common.c:310-313`).
const MPA_HUFF_SIZES_MINUS_ONE: &[usize] =
    &[3, 8, 8, 15, 15, 35, 35, 35, 63, 63, 63, 255, 255, 255, 255];

/// `mpa_quad_codes[2][16]` (`mpegaudiodec_common.c:352-355`).
const MPA_QUAD_CODES: [[u8; 16]; 2] = [
    [1, 5, 4, 5, 6, 5, 4, 4, 7, 3, 6, 0, 7, 2, 3, 1],
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

/// `mpa_quad_bits[2][16]` (`mpegaudiodec_common.c:357-360`).
const MPA_QUAD_BITS: [[u8; 16]; 2] = [[1, 4, 4, 5, 4, 6, 5, 6, 4, 5, 5, 6, 5, 6, 6, 6], [4; 16]];

// dct32 coefficients — `#define COS0_0` etc. (dct32_template.c:53-87),
// each still carrying its C divisor so `mulh3`'s `1 << s` cancels it.
const COS0_0: f32 = fx(0.50060299823519630134 / 2.0);
const COS0_1: f32 = fx(0.50547095989754365998 / 2.0);
const COS0_2: f32 = fx(0.51544730992262454697 / 2.0);
const COS0_3: f32 = fx(0.53104259108978417447 / 2.0);
const COS0_4: f32 = fx(0.55310389603444452782 / 2.0);
const COS0_5: f32 = fx(0.58293496820613387367 / 2.0);
const COS0_6: f32 = fx(0.62250412303566481615 / 2.0);
const COS0_7: f32 = fx(0.67480834145500574602 / 2.0);
const COS0_8: f32 = fx(0.74453627100229844977 / 2.0);
const COS0_9: f32 = fx(0.83934964541552703873 / 2.0);
const COS0_10: f32 = fx(0.97256823786196069369 / 2.0);
const COS0_11: f32 = fx(1.16943993343288495515 / 4.0);
const COS0_12: f32 = fx(1.48416461631416627724 / 4.0);
const COS0_13: f32 = fx(2.05778100995341155085 / 8.0);
const COS0_14: f32 = fx(3.40760841846871878570 / 8.0);
const COS0_15: f32 = fx(10.19000812354805681150 / 32.0);

const COS1_0: f32 = fx(0.50241928618815570551 / 2.0);
const COS1_1: f32 = fx(0.52249861493968888062 / 2.0);
const COS1_2: f32 = fx(0.56694403481635770368 / 2.0);
const COS1_3: f32 = fx(0.64682178335999012954 / 2.0);
const COS1_4: f32 = fx(0.78815462345125022473 / 2.0);
const COS1_5: f32 = fx(1.06067768599034747134 / 4.0);
const COS1_6: f32 = fx(1.72244709823833392782 / 4.0);
const COS1_7: f32 = fx(5.10114861868916385802 / 16.0);

const COS2_0: f32 = fx(0.50979557910415916894 / 2.0);
const COS2_1: f32 = fx(0.60134488693504528054 / 2.0);
const COS2_2: f32 = fx(0.89997622313641570463 / 2.0);
const COS2_3: f32 = fx(2.56291544774150617881 / 8.0);

const COS3_0: f32 = fx(0.54119610014619698439 / 2.0);
const COS3_1: f32 = fx(1.30656296487637652785 / 4.0);

const COS4_0: f32 = fx(std::f64::consts::FRAC_1_SQRT_2 / 2.0);

/// `ISQRT2` — `FIXR(0.70710678118654752440)` (template.c:941).
const ISQRT2: f32 = fx(0.70710678118654752440);

// ---------------------------------------------------------------------
// Bit reservoir state — the gb / in_gb pair + switch_buffer
// (mpegaudiodec_template.c:725-738)
// ---------------------------------------------------------------------

/// The reader pair C keeps in `MPADecodeContext`: `gb` is the current
/// window (bit reservoir `last_buf`, later the frame remainder after
/// `switch_buffer`), `in_gb` stashes the frame's post-side-info reader
/// while `gb` covers the reservoir, `extrasize` counts the bytes of
/// the *current* frame that were appended to the reservoir view.
#[derive(Debug)]
struct Bitstream {
    gb: GetBits,
    in_gb: Option<GetBits>,
    extrasize: usize,
}

// ---------------------------------------------------------------------
// GranuleDef — mpegaudiodec_template.c:58-75
// ---------------------------------------------------------------------

/// Layer 3 "granule".
#[derive(Clone, Debug)]
struct GranuleDef {
    scfsi: u8,
    part2_3_length: i32,
    big_values: i32,
    global_gain: i32,
    scalefac_compress: i32,
    block_type: u8,
    switch_point: u8,
    table_select: [i32; 3],
    subblock_gain: [i32; 3],
    scalefac_scale: u8,
    count1table_select: u8,
    /// number of huffman codes in each region.
    region_size: [i32; 3],
    preflag: i32,
    /// long/short band indexes.
    short_start: i32,
    long_end: i32,
    scale_factors: [u8; 40],
    /// 576 samples.
    sb_hybrid: Box<[f32; SBLIMIT * 18]>,
}

impl Default for GranuleDef {
    fn default() -> Self {
        GranuleDef {
            scfsi: 0,
            part2_3_length: 0,
            big_values: 0,
            global_gain: 0,
            scalefac_compress: 0,
            block_type: 0,
            switch_point: 0,
            table_select: [0; 3],
            subblock_gain: [0; 3],
            scalefac_scale: 0,
            count1table_select: 0,
            region_size: [0; 3],
            preflag: 0,
            short_start: 0,
            long_end: 0,
            scale_factors: [0; 40],
            sb_hybrid: Box::new([0.0; SBLIMIT * 18]),
        }
    }
}

impl GranuleDef {
    /// `region_offset2size` (`mpegaudiodec_template.c:126-135`) —
    /// convert region offsets to region sizes and truncate size to
    /// big_values.
    fn region_offset2size(&mut self) {
        self.region_size[2] = (576 / 2) as i32;
        let mut j = 0;
        for i in 0..3 {
            let k = self.region_size[i].min(self.big_values);
            self.region_size[i] = k - j;
            j = k;
        }
    }

    /// `init_short_region` (`mpegaudiodec_template.c:137-153`).
    fn init_short_region(&mut self, sample_rate_index: i32) {
        if self.block_type == 2 {
            if sample_rate_index != 8 {
                self.region_size[0] = 36 / 2;
            } else {
                self.region_size[0] = 72 / 2;
            }
        } else if sample_rate_index <= 2 {
            self.region_size[0] = 36 / 2;
        } else if sample_rate_index != 8 {
            self.region_size[0] = 54 / 2;
        } else {
            self.region_size[0] = 108 / 2;
        }
        self.region_size[1] = 576 / 2;
    }

    /// `init_long_region` (`mpegaudiodec_template.c:155-163`) is
    /// inlined at its only call site (side-info parse) because the
    /// region sizes are computed there from freshly read bits.

    /// `compute_band_indexes` (`mpegaudiodec_template.c:165-188`).
    fn compute_band_indexes(&mut self, sample_rate_index: i32) {
        if self.block_type == 2 {
            if self.switch_point != 0 {
                // if switched mode, we handle the 36 first samples as
                // long blocks.  For 8000Hz, we handle the 72 first
                // exponents as long blocks
                if sample_rate_index <= 2 {
                    self.long_end = 8;
                } else {
                    self.long_end = 6;
                }
                self.short_start = 3;
            } else {
                self.long_end = 0;
                self.short_start = 0;
            }
        } else {
            self.short_start = 13;
            self.long_end = 22;
        }
    }
}

// ---------------------------------------------------------------------
// The decode core — MPADecodeContext minus the AVCodecContext soup
// (mpegaudiodec_template.c:77-99)
// ---------------------------------------------------------------------

#[derive(Debug)]
struct MpaDecodeCore {
    header: MpaDecodeHeader,
    /// `last_buf[LAST_BUF_SIZE]` — the bit reservoir.
    last_buf: Vec<u8>,
    last_buf_size: usize,
    bs: Bitstream,
    synth_buf: [Vec<f32>; MPA_MAX_CHANNELS],
    #[cfg(test)]
    dump_tag: u32,
    synth_buf_offset: [usize; MPA_MAX_CHANNELS],
    sb_samples: [[f32; 36 * SBLIMIT]; MPA_MAX_CHANNELS],
    mdct_buf: [[f32; SBLIMIT * 18]; MPA_MAX_CHANNELS],
    /// `granules[2][2]` — indexed `[ch][gr]`.
    granules: [[GranuleDef; 2]; 2],
    dither_state: i32,
    crc: u32,
}

impl MpaDecodeCore {
    fn new() -> MpaDecodeCore {
        MpaDecodeCore {
            header: MpaDecodeHeader::default(),
            last_buf: vec![0; LAST_BUF_SIZE],
            last_buf_size: 0,
            bs: Bitstream {
                gb: GetBits::init(Vec::new(), 0),
                in_gb: None,
                extrasize: 0,
            },
            synth_buf: [Vec::new(), Vec::new()],
            #[cfg(test)]
            dump_tag: 0,
            synth_buf_offset: [0; 2],
            sb_samples: [[0.0; 36 * SBLIMIT]; MPA_MAX_CHANNELS],
            mdct_buf: [[0.0; SBLIMIT * 18]; MPA_MAX_CHANNELS],
            granules: Default::default(),
            dither_state: 0,
            crc: 0,
        }
    }

    /// `mp_flush` (`mpegaudiodec_template.c:1630-1636`).
    fn flush(&mut self) {
        for ch in 0..MPA_MAX_CHANNELS {
            self.synth_buf[ch].fill(0.0);
        }
        self.mdct_buf = [[0.0; SBLIMIT * 18]; MPA_MAX_CHANNELS];
        self.last_buf_size = 0;
        self.dither_state = 0;
    }

    /// `mp_decode_layer3` (`mpegaudiodec_template.c:1212-1469`) —
    /// returns the number of 32-sample synth blocks
    /// (`nb_granules * 18`).
    fn mp_decode_layer3(&mut self) -> Result<usize> {
        let nb_granules: usize;
        let main_data_begin: usize;

        // read side info
        if self.header.lsf != 0 {
            main_data_begin = self.bs.gb.get_bits(8) as usize;
            self.bs.gb.skip_bits(self.header.nb_channels as u32);
            nb_granules = 1;
        } else {
            main_data_begin = self.bs.gb.get_bits(9) as usize;
            if self.header.nb_channels == 2 {
                self.bs.gb.skip_bits(3);
            } else {
                self.bs.gb.skip_bits(5);
            }
            nb_granules = 2;
            for ch in 0..self.header.nb_channels as usize {
                self.granules[ch][0].scfsi = 0; // all scale factors are transmitted
                self.granules[ch][1].scfsi = self.bs.gb.get_bits(4) as u8;
            }
        }

        let sri = self.header.sample_rate_index;
        let lsf = self.header.lsf != 0;
        let nch = self.header.nb_channels as usize;

        for gr in 0..nb_granules {
            for ch in 0..nch {
                // side info bits are read into locals first so the
                // granule borrow does not overlap the reader borrow.
                let part2_3_length = self.bs.gb.get_bits(12) as i32;
                let big_values = self.bs.gb.get_bits(9) as i32;
                if big_values > 288 {
                    return Err(Error::InvalidData("big_values too big".into()));
                }
                let mut global_gain = self.bs.gb.get_bits(8) as i32;
                // if MS stereo only is selected, we precompute the
                // 1/sqrt(2) renormalization factor
                if (self.header.mode_ext & (MODE_EXT_MS_STEREO | MODE_EXT_I_STEREO))
                    == MODE_EXT_MS_STEREO
                {
                    global_gain -= 2;
                }
                let scalefac_compress = if lsf {
                    self.bs.gb.get_bits(9) as i32
                } else {
                    self.bs.gb.get_bits(4) as i32
                };
                let blocksplit_flag = self.bs.gb.get_bits1();
                // (block_type, switch_point, table_select, region_size /
                // subblock_gain scratch)
                let (block_type, switch_point, table_select, gains_or_regions) = if blocksplit_flag
                    != 0
                {
                    let block_type = self.bs.gb.get_bits(2) as u8;
                    if block_type == 0 {
                        return Err(Error::InvalidData("invalid block type".into()));
                    }
                    let switch_point = self.bs.gb.get_bits1() as u8;
                    let mut table_select = [0i32; 3];
                    for i in 0..2 {
                        table_select[i] = self.bs.gb.get_bits(5) as i32;
                    }
                    let mut subblock_gain = [0i32; 3];
                    for i in 0..3 {
                        subblock_gain[i] = self.bs.gb.get_bits(3) as i32;
                    }
                    (block_type, switch_point, table_select, subblock_gain)
                } else {
                    let mut table_select = [0i32; 3];
                    for i in 0..3 {
                        table_select[i] = self.bs.gb.get_bits(5) as i32;
                    }
                    // compute huffman coded region sizes
                    // (init_long_region, template.c:155-163)
                    let region_address1 = self.bs.gb.get_bits(4) as i32;
                    let region_address2 = self.bs.gb.get_bits(3) as i32;
                    let rs0 = tables().band_index_long[sri as usize][(region_address1 + 1) as usize]
                        as i32;
                    let l = (region_address1 + region_address2 + 2).min(22);
                    let rs1 = tables().band_index_long[sri as usize][l as usize] as i32;
                    (0u8, 0u8, table_select, [rs0, rs1, 0])
                };
                let preflag = if !lsf {
                    self.bs.gb.get_bits1() as i32
                } else {
                    0
                };
                let scalefac_scale = self.bs.gb.get_bits1() as u8;
                let count1table_select = self.bs.gb.get_bits1() as u8;

                let g = &mut self.granules[ch][gr];
                g.part2_3_length = part2_3_length;
                g.big_values = big_values;
                g.global_gain = global_gain;
                g.scalefac_compress = scalefac_compress;
                g.block_type = block_type;
                g.switch_point = switch_point;
                g.table_select = table_select;
                g.preflag = preflag;
                g.scalefac_scale = scalefac_scale;
                g.count1table_select = count1table_select;
                if blocksplit_flag != 0 {
                    g.subblock_gain = gains_or_regions;
                    g.init_short_region(sri);
                } else {
                    g.subblock_gain = [0; 3];
                    g.region_size = gains_or_regions;
                }
                g.region_offset2size();
                g.compute_band_indexes(sri);
            }
        }

        // ---- bit reservoir plumbing (template.c:1302-1336) ----
        // !adu_mode is the only mode here.
        let byte_pos = (self.bs.gb.get_bits_count() >> 3) as usize;
        let avail = (self.bs.gb.get_bits_left() >> 3) - self.bs.extrasize as i64;
        self.bs.extrasize = avail.clamp(0, (LAST_BUF_SIZE - self.last_buf_size) as i64) as usize;
        // memcpy(s->last_buf + s->last_buf_size, ptr, s->extrasize)
        let src = self
            .bs
            .gb
            .buf
            .get(byte_pos..)
            .map(|s| s[..self.bs.extrasize.min(s.len())].to_vec())
            .unwrap_or_default();
        copy_clamped(
            &mut self.last_buf[self.last_buf_size..self.last_buf_size + self.bs.extrasize],
            &src,
        );
        self.bs.in_gb = Some(self.bs.gb.clone());
        let reservoir_bits = ((self.last_buf_size + self.bs.extrasize) * 8) as i64;
        self.bs.gb = GetBits::init(
            self.last_buf[..self.last_buf_size + self.bs.extrasize].to_vec(),
            reservoir_bits,
        );
        // s->last_buf_size <<= 3 (now in bits, local only — C resets it
        // to bytes in mp_decode_frame)
        let mut last_buf_bits = (self.last_buf_size * 8) as i64;

        // now we get bits from the main_data_begin offset
        let mut gr = 0usize;
        while gr < nb_granules && (last_buf_bits >> 3) < main_data_begin as i64 {
            for ch in 0..nch {
                let g = &mut self.granules[ch][gr];
                last_buf_bits += g.part2_3_length as i64;
                g.sb_hybrid.fill(0.0);
                compute_imdct(
                    ch,
                    gr,
                    &mut self.granules,
                    &mut self.sb_samples,
                    &mut self.mdct_buf,
                );
            }
            gr += 1;
        }
        let skip = last_buf_bits - 8 * main_data_begin as i64;
        if skip >= self.bs.gb.size_in_bits - self.bs.extrasize as i64 * 8 && self.bs.in_gb.is_some()
        {
            let mut ig = self.bs.in_gb.take().unwrap();
            ig.skip_bits_long(skip - self.bs.gb.size_in_bits + self.bs.extrasize as i64 * 8);
            self.bs.gb = ig;
            self.bs.extrasize = 0;
        } else {
            self.bs.gb.skip_bits_long(skip);
        }

        // ---- granule decode loop (template.c:1338-1465) ----
        while gr < nb_granules {
            for ch in 0..nch {
                let bits_pos = self.bs.gb.get_bits_count();
                let mut exponents = [0i16; 576];
                let sc_prev = self.granules[ch][0].scale_factors;
                {
                    let g = &mut self.granules[ch][gr];
                    if !lsf {
                        // MPEG-1 scale factors (template.c:1343-1393)
                        let slen1 = FF_SLEN_TABLE[0][g.scalefac_compress as usize] as u32;
                        let slen2 = FF_SLEN_TABLE[1][g.scalefac_compress as usize] as u32;
                        if g.block_type == 2 {
                            let n = if g.switch_point != 0 { 17 } else { 18 };
                            let mut j = 0usize;
                            if slen1 != 0 {
                                for _ in 0..n {
                                    g.scale_factors[j] = self.bs.gb.get_bits(slen1) as u8;
                                    j += 1;
                                }
                            } else {
                                for _ in 0..n {
                                    g.scale_factors[j] = 0;
                                    j += 1;
                                }
                            }
                            if slen2 != 0 {
                                for _ in 0..18 {
                                    g.scale_factors[j] = self.bs.gb.get_bits(slen2) as u8;
                                    j += 1;
                                }
                                for _ in 0..3 {
                                    g.scale_factors[j] = 0;
                                    j += 1;
                                }
                            } else {
                                for _ in 0..21 {
                                    g.scale_factors[j] = 0;
                                    j += 1;
                                }
                            }
                        } else {
                            // scfsi groups may copy granule 0's factors.
                            // The copy is hoisted before the granule
                            // borrow; for gr == 0 scfsi is 0 so the copy
                            // branch never runs (same as C, which reads
                            // the same object it is filling).
                            let mut j = 0usize;
                            for k in 0..4u32 {
                                let n = if k == 0 { 6 } else { 5 };
                                if g.scfsi & (0x8 >> k) == 0 {
                                    let slen = if k < 2 { slen1 } else { slen2 };
                                    if slen != 0 {
                                        for _ in 0..n {
                                            g.scale_factors[j] = self.bs.gb.get_bits(slen) as u8;
                                            j += 1;
                                        }
                                    } else {
                                        for _ in 0..n {
                                            g.scale_factors[j] = 0;
                                            j += 1;
                                        }
                                    }
                                } else {
                                    // simply copy from last granule
                                    for _ in 0..n {
                                        g.scale_factors[j] = sc_prev[j];
                                        j += 1;
                                    }
                                }
                            }
                            g.scale_factors[j] = 0;
                        }
                    } else {
                        // LSF scale factors (template.c:1394-1447)
                        let tindex: usize = if g.block_type == 2 {
                            if g.switch_point != 0 { 2 } else { 1 }
                        } else {
                            0
                        };
                        let mut sf = g.scalefac_compress;
                        let mut slen = [0i32; 4];
                        let tindex2: usize;
                        if (self.header.mode_ext & MODE_EXT_I_STEREO) != 0 && ch == 1 {
                            // intensity stereo case
                            sf >>= 1;
                            if sf < 180 {
                                lsf_sf_expand(&mut slen, sf, 6, 6, 0);
                                tindex2 = 3;
                            } else if sf < 244 {
                                lsf_sf_expand(&mut slen, sf - 180, 4, 4, 0);
                                tindex2 = 4;
                            } else {
                                lsf_sf_expand(&mut slen, sf - 244, 3, 0, 0);
                                tindex2 = 5;
                            }
                        } else {
                            // normal case
                            if sf < 400 {
                                lsf_sf_expand(&mut slen, sf, 5, 4, 4);
                                tindex2 = 0;
                            } else if sf < 500 {
                                lsf_sf_expand(&mut slen, sf - 400, 5, 4, 0);
                                tindex2 = 1;
                            } else {
                                lsf_sf_expand(&mut slen, sf - 500, 3, 0, 0);
                                tindex2 = 2;
                                g.preflag = 1;
                            }
                        }
                        let mut j = 0usize;
                        for k in 0..4 {
                            let n = FF_LSF_NSF_TABLE[tindex2][tindex][k] as usize;
                            let sl = slen[k];
                            if sl != 0 {
                                for _ in 0..n {
                                    g.scale_factors[j] = self.bs.gb.get_bits(sl as u32) as u8;
                                    j += 1;
                                }
                            } else {
                                for _ in 0..n {
                                    g.scale_factors[j] = 0;
                                    j += 1;
                                }
                            }
                        }
                        // XXX: should compute exact size
                        for sf in g.scale_factors.iter_mut().skip(j) {
                            *sf = 0;
                        }
                    }
                    exponents_from_scale_factors(sri, g, &mut exponents);
                }

                // read Huffman coded residue
                let end = bits_pos + self.granules[ch][gr].part2_3_length as i64;
                huffman_decode(&mut self.bs, &mut self.granules[ch][gr], &exponents, end);
            } /* ch */

            if self.header.mode == MPA_JSTEREO {
                compute_stereo(self.header.mode_ext, lsf, sri, &mut self.granules, gr);
            }

            if std::env::var("MP3_DUMP").is_ok() && self.dump_tag < 9 {
                let g = &self.granules[0][gr];
                let mut prof = String::new();
                for band in 0..32usize {
                    let e: f32 = g.sb_hybrid[band * 18..band * 18 + 18]
                        .iter()
                        .map(|v| v * v)
                        .sum();
                    prof.push_str(&format!("{:1.0} ", e * 1e6));
                }
                let nz = g.sb_hybrid.iter().filter(|v| v.abs() > 1e-9).count();
                eprintln!(
                    "SBHYB ch0 gr{gr} bt{} sp{} le{} nz={nz}: {prof}",
                    g.block_type, g.switch_point, g.long_end
                );
                self.dump_tag += 1;
            }
            for ch in 0..nch {
                reorder_block(sri, &mut self.granules[ch][gr]);
                compute_antialias(&mut self.granules[ch][gr]);
                compute_imdct(
                    ch,
                    gr,
                    &mut self.granules,
                    &mut self.sb_samples,
                    &mut self.mdct_buf,
                );
            }
            gr += 1;
        } /* gr */
        if self.bs.gb.get_bits_count() < 0 {
            let n = -self.bs.gb.get_bits_count();
            self.bs.gb.skip_bits_long(n);
        }
        Ok(nb_granules * 18)
    }

    /// `mp_decode_frame` (`mpegaudiodec_template.c:1471-1556`) — layer
    /// dispatch plus the layer-3 backstep bookkeeping; returns
    /// `nb_frames` (32-sample synth blocks).
    fn mp_decode_frame(&mut self, buf: &[u8]) -> Result<usize> {
        self.bs.gb = GetBits::init(
            buf[HEADER_SIZE..].to_vec(),
            ((buf.len() - HEADER_SIZE) * 8) as i64,
        );
        if self.header.error_protection != 0 {
            self.crc = self.bs.gb.get_bits(16);
        }

        match self.header.layer {
            1 | 2 => {
                return Err(Error::Unsupported(format!(
                    "MPEG audio layer {} decode is not ported (this decoder is layer 3 / mp3)",
                    self.header.layer
                )));
            }
            _ => {
                let nb_frames = self.mp_decode_layer3()?;

                self.last_buf_size = 0;
                if let Some(ig) = self.bs.in_gb.take() {
                    let pos = self.bs.gb.align_get_bits();
                    let i = (self.bs.gb.get_bits_left() >> 3) - self.bs.extrasize as i64;
                    if (0..=BACKSTEP_SIZE as i64).contains(&i) {
                        let src = self
                            .bs
                            .gb
                            .buf
                            .get(pos..)
                            .map(|s| s[..(i as usize).min(s.len())].to_vec())
                            .unwrap_or_default();
                        copy_clamped(&mut self.last_buf[..i as usize], &src);
                        self.last_buf_size = i as usize;
                    }
                    // else: "invalid old backstep" — C logs and keeps 0
                    self.bs.gb = ig;
                    self.bs.extrasize = 0;
                }

                let _ = self.bs.gb.align_get_bits();
                let mut i = (self.bs.gb.get_bits_left() >> 3) - self.bs.extrasize as i64;
                if !(0..=BACKSTEP_SIZE as i64).contains(&i) {
                    // "invalid new backstep" recovery
                    i = (BACKSTEP_SIZE as i64).min((buf.len() - HEADER_SIZE) as i64);
                }
                // memcpy(s->last_buf + s->last_buf_size,
                //        s->gb.buffer + buf_size - HEADER_SIZE - i, i)
                let main_len = buf.len() - HEADER_SIZE;
                let start = main_len.saturating_sub(i as usize);
                let src = self
                    .bs
                    .gb
                    .buf
                    .get(start..)
                    .map(|s| s[..(i as usize).min(s.len())].to_vec())
                    .unwrap_or_default();
                copy_clamped(
                    &mut self.last_buf[self.last_buf_size..self.last_buf_size + i as usize],
                    &src,
                );
                self.last_buf_size += i as usize;
                Ok(nb_frames)
            }
        }
    }
}

// ---------------------------------------------------------------------
// Mp3Decoder — the AVCodec wrapper (ff_mp3float_decoder)
// ---------------------------------------------------------------------

/// `ff_mp3float_decoder` (`mpegaudiodec_float.c:122-136`) — the float
/// MPEG audio layer 3 decoder, `FLTP` output, one frame per packet.
#[derive(Debug)]
pub struct Mp3Decoder {
    core: MpaDecodeCore,
    params: CodecParameters,
    /// One-frame output queue.
    pending: Option<AudioFrame>,
    /// Drain requested (`avcodec_send_packet(avctx, NULL)`).
    eof: bool,
}

impl Default for Mp3Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp3Decoder {
    pub fn new() -> Self {
        let mut core = MpaDecodeCore::new();
        core.synth_buf = [vec![0.0; 512 * 2], vec![0.0; 512 * 2]];
        Mp3Decoder {
            core,
            params: CodecParameters::default(),
            pending: None,
            eof: false,
        }
    }

    /// The `flush` codec callback → `mp_flush` (`mpegaudiodec_template.c:1630-1636`).
    pub fn flush(&mut self) {
        self.core.flush();
        self.pending = None;
    }

    /// `decode_frame` (`mpegaudiodec_template.c:1558-1628`). Returns
    /// `Ok(None)` for a consumed-but-frameless packet (the ID3v1
    /// `TAG` skip).
    fn decode_frame(&mut self, pkt: &Packet) -> Result<Option<AudioFrame>> {
        let mut buf = pkt.as_slice();
        let mut _skipped = 0usize;
        while !buf.is_empty() && buf[0] == 0 {
            buf = &buf[1..];
            _skipped += 1;
        }

        if buf.len() < HEADER_SIZE {
            return Err(Error::InvalidData(
                "packet too small for an MPEG audio frame".into(),
            ));
        }

        let header = u32::from_be_bytes(buf[..HEADER_SIZE].try_into().unwrap());
        if header >> 8 == 0x544_147 {
            // AV_RB32("TAG") >> 8 — "discarding ID3 tag"
            return Ok(None);
        }
        if avpriv_mpegaudio_decode_header(&mut self.core.header, header)? {
            // free format: prepare to compute frame size
            return Err(Error::InvalidData(
                "free-format frame: frame size must be computed externally \
                 (the mp3 parser is not ported)"
                    .into(),
            ));
        }
        // update codec info
        self.params.ch_layout = if self.core.header.nb_channels == 1 {
            ChannelLayout::MONO
        } else {
            ChannelLayout::STEREO
        };
        if self.params.bit_rate == 0 {
            self.params.bit_rate = self.core.header.bit_rate as i64;
        }
        self.params.sample_rate = self.core.header.sample_rate;
        self.params.frame_size = if self.core.header.lsf != 0 { 576 } else { 1152 };

        if self.core.header.frame_size <= 0 {
            return Err(Error::InvalidData("incomplete frame".into()));
        }
        let frame_bytes = (self.core.header.frame_size as usize).min(buf.len());

        let nb_frames = self.core.mp_decode_frame(&buf[..frame_bytes])?;

        // get output buffer + apply the synthesis filter
        // (template.c:1526-1553)
        let frame_size = self.params.frame_size as usize;
        let mut frame = AudioFrame::alloc(SampleFormat::Fltp, self.params.ch_layout, frame_size)?;
        frame.pts = pkt.pts;
        frame.duration = pkt.duration;
        frame.time_base = pkt.time_base;
        frame.sample_rate = self.core.header.sample_rate;

        let t = tables();
        let mut samples = vec![0f32; frame_size];
        for ch in 0..self.core.header.nb_channels as usize {
            samples.fill(0.0);
            for i in 0..nb_frames {
                let row = &self.core.sb_samples[ch][i * SBLIMIT..(i + 1) * SBLIMIT];
                mpa_synth_filter(
                    &mut self.core.synth_buf[ch],
                    &mut self.core.synth_buf_offset[ch],
                    &t.synth_window,
                    &mut self.core.dither_state,
                    &mut samples[i * 32..(i + 1) * 32],
                    1,
                    row,
                );
            }
            let plane = frame.plane_mut(ch);
            for (dst, src) in plane.chunks_exact_mut(4).zip(samples.iter()) {
                dst.copy_from_slice(&src.to_ne_bytes());
            }
        }
        Ok(Some(frame))
    }
}

impl AudioDecoder for Mp3Decoder {
    /// `decode_ctx_init` (`mpegaudiodec_template.c:283-315`) — the mp3
    /// float codec gate: output `OUT_FMT_P` = FLTP.
    fn init(&mut self, params: &CodecParameters) -> Result<()> {
        match params.codec_id {
            CodecId::Mp3 => {}
            CodecId::Mp1 | CodecId::Mp2 => {
                return Err(Error::Unsupported(
                    "mp1/mp2 (MPEG audio layer 1/2) decode is not ported — \
                     this decoder is layer 3 only"
                        .into(),
                ));
            }
            id => {
                return Err(Error::Unsupported(format!(
                    "codec '{}' is not a ported MP3 decoder",
                    id.name()
                )));
            }
        }
        self.params = params.clone();
        self.params.codec_type = MediaType::Audio;
        self.params.sample_fmt = SampleFormat::Fltp;
        Ok(())
    }

    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<()> {
        let Some(pkt) = pkt else {
            self.eof = true;
            return Ok(());
        };
        if self.eof {
            // C rejects input after drain in avcodec_send_packet.
            return Err(Error::Eof);
        }
        self.pending = None;
        self.pending = self.decode_frame(pkt)?;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<AudioFrame> {
        match self.pending.take() {
            Some(frame) => Ok(frame),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::Again),
        }
    }
}

// ---------------------------------------------------------------------
// Bit reader — get_bits.h / bitstream_template.h (safe BE reader)
// ---------------------------------------------------------------------

/// `GetBitContext` — big-endian MSB-first reader with the *safe*
/// semantics: reads at or past the end return 0 bits, and the position
/// saturates at `ceil(size_in_bits / 8) * 8` (C's `buffer_end * 8`).
/// Unlike C the buffer is owned (a copy of the window), so the decoder
/// can stash/restore readers freely (`gb` / `in_gb`).
#[derive(Clone, Debug)]
struct GetBits {
    buf: Vec<u8>,
    index: i64,
    size_in_bits: i64,
}

impl GetBits {
    /// `init_get_bits(gb, buf, size_in_bits)`.
    fn init(buf: Vec<u8>, size_in_bits: i64) -> GetBits {
        debug_assert_eq!(buf.len(), ((size_in_bits + 7) >> 3) as usize);
        GetBits {
            buf,
            index: 0,
            size_in_bits,
        }
    }

    /// C's saturation point: `buffer_end * 8`.
    fn cap(&self) -> i64 {
        (self.size_in_bits + 7) & !7
    }

    /// One bit at absolute position `i` (0 past the buffer, like the
    /// safe reader's zero-refill).
    fn bit(&self, i: i64) -> u32 {
        if i >= 0 && (i >> 3) < self.buf.len() as i64 {
            ((self.buf[(i >> 3) as usize] >> (7 - (i & 7))) & 1) as u32
        } else {
            0
        }
    }

    /// `get_bits_count`.
    fn get_bits_count(&self) -> i64 {
        self.index
    }

    /// `get_bits_left`.
    fn get_bits_left(&self) -> i64 {
        self.size_in_bits - self.index
    }

    /// `get_bits1` / one bit of `get_bits`.
    fn get_bits1(&mut self) -> u32 {
        let v = self.bit(self.index);
        self.index = (self.index + 1).min(self.cap());
        v
    }

    /// `get_bits` (`bits_read_nz`) for n in 1..=32; `get_bitsz` for
    /// n == 0 (returns 0, consumes nothing).
    fn get_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let mut v = 0u32;
        for k in 0..n {
            v = (v << 1) | self.bit(self.index + k as i64);
        }
        self.index = (self.index + n as i64).min(self.cap());
        v
    }

    /// `show_bits` (peek without consuming) for n in 0..=32.
    fn peek(&self, n: u32) -> u32 {
        let mut v = 0u32;
        for k in 0..n {
            v = (v << 1) | self.bit(self.index + k as i64);
        }
        v
    }

    /// `skip_bits` (unsigned in C's new reader; clamped at the cap).
    fn skip_bits(&mut self, n: u32) {
        self.index = (self.index + n as i64).min(self.cap());
    }

    /// `skip_bits_long` — the legacy signed reader semantics the
    /// decoder was written for (`mpegaudiodec_template.c:1466`): a
    /// negative count moves the index back (possibly below 0, where
    /// reads return 0).
    fn skip_bits_long(&mut self, n: i64) {
        self.index = (self.index + n).clamp(0, self.cap());
    }

    /// `align_get_bits` — skip to the next byte boundary; returns the
    /// byte offset of the new position (C returns the pointer).
    fn align_get_bits(&mut self) -> usize {
        let n = (-self.index) & 7;
        if n != 0 {
            self.skip_bits(n as u32);
        }
        (self.index >> 3) as usize
    }
}

/// One of the 15 big-value Huffman tables, built from code lengths the
/// way `ff_vlc_init_from_lengths` (`vlc.c:306-351`) does: a running
/// 32-bit top-aligned counter advances by `1 << (32 - len)` per symbol
/// **in array order** (`vlc.c:319-345`) — not sorted by length, so the
/// codes must be matched explicitly rather than by canonical ranges.
/// Decoding walks bit by bit and matches the accumulated prefix.
struct BigVlc {
    /// Per code length: sorted `(code, symbol)` pairs.
    by_len: Vec<Vec<(u32, i32)>>,
    max_len: u32,
}

impl BigVlc {
    /// `ff_vlc_init_from_lengths` + the mpa symbol packing
    /// (`mpegaudiodec_common.c:423-427`):
    /// `tmp_symbols[j] = high << 1 | ((high && low) << 4) | low`.
    fn build(lens: &[u8], syms: &[u8]) -> BigVlc {
        if lens.is_empty() {
            // index 0 is the unused dummy slot (ff_huff_vlc[0]).
            return BigVlc {
                by_len: vec![Vec::new(); 1],
                max_len: 0,
            };
        }
        let mut code: u64 = 0;
        let max_len = *lens.iter().max().unwrap() as u32;
        let mut by_len: Vec<Vec<(u32, i32)>> = vec![Vec::new(); (max_len + 1) as usize];
        for j in 0..lens.len() {
            let len = lens[j] as u32;
            let c = (code >> (32 - len)) as u32;
            let high = (syms[j] & 0xf0) as i32;
            let low = (syms[j] & 0x0f) as i32;
            let sym = (high << 1) | (((high != 0 && low != 0) as i32) << 4) | low;
            by_len[len as usize].push((c, sym));
            code += 1u64 << (32 - len);
        }
        debug_assert_eq!(code, 1 << 32, "over/under-determined VLC tree");
        for v in by_len.iter_mut() {
            v.sort_unstable();
        }
        BigVlc { by_len, max_len }
    }

    /// `get_vlc2(gb, table, 7, 3)` for a lengths-built table. Complete
    /// trees always terminate; returns -1 only if the tree is somehow
    /// incomplete (C returns -1 for invalid codes).
    fn decode(&self, gb: &mut GetBits) -> i32 {
        let mut code: u32 = 0;
        for len in 1..=self.max_len as usize {
            let b = gb.get_bits1();
            code = (code << 1) | b;
            #[cfg(test)]
            eprintln!(
                "DBG decode len={len} bit={b} code={code} looking in {:?}",
                self.by_len[len]
            );
            if let Ok(idx) = self.by_len[len].binary_search_by_key(&code, |&(c, _)| c) {
                return self.by_len[len][idx].1;
            }
        }
        -1
    }
}

/// One of the 2 quad (count1) tables — built from explicit
/// (code, length) pairs like `vlc_init` (`mpegaudiodec_common.c:438-447`),
/// with a flat `1 << bits` LUT exactly like C's one-level table.
struct QuadVlc {
    lut: Vec<(i32, u8)>, // (sym, len); len 0 = invalid (C: sym -1)
}

impl QuadVlc {
    fn build(bits: u32, lengths: &[u8], codes: &[u8]) -> QuadVlc {
        let size = 1usize << bits;
        let mut lut = vec![(-1i32, 0u8); size];
        for i in 0..16 {
            let len = lengths[i] as u32;
            let code = codes[i] as u32;
            let lo = (code << (bits - len)) as usize;
            for slot in lut.iter_mut().take(lo + (1usize << (bits - len))).skip(lo) {
                *slot = (i as i32, len as u8);
            }
        }
        QuadVlc { lut }
    }

    /// `get_vlc2(gb, vlc->table, vlc->bits, 1)` — invalid code consumes
    /// no bits and returns -1 (bitstream_template.h:499-529).
    fn decode(&self, gb: &mut GetBits, bits: u32) -> i32 {
        let idx = gb.peek(bits) as usize;
        let (sym, len) = self.lut[idx];
        gb.skip_bits(len as u32);
        sym
    }
}

/// Everything C builds in the `decode_init_static` /
/// `ff_mpegaudiodec_common_init_static` / `ff_mpadsp_init` once-blocks.
struct Tables {
    /// `ff_band_index_long[9][23]` (`mpegaudiodec_common.c:450-457`).
    band_index_long: [[u16; 23]; 9],
    /// `exp_table_float[512]` (`mpegaudio_tablegen.h`).
    exp_table: Vec<f32>,
    /// `expval_table_float[512][16]`.
    expval_table: Vec<[f32; 16]>,
    /// `ff_table_4_3_exp[TABLE_4_3_SIZE]`.
    table_4_3_exp: Vec<i8>,
    /// `ff_table_4_3_value[TABLE_4_3_SIZE]`.
    table_4_3_value: Vec<u32>,
    /// `ff_mdct_win_float[8][MDCT_BUF_SIZE]` (`mpegaudiodsp.c:30-79`).
    mdct_win: [[f32; MDCT_BUF_SIZE]; 8],
    /// `ff_mpa_synth_window_float[512+256]` (`mpegaudiodsp_template.c:197-224`).
    synth_window: Vec<f32>,
    /// `is_table_lsf[2][2][16]` (`mpegaudiodec_template.c:107,264-278`).
    is_table_lsf: [[[f32; 16]; 2]; 2],
    /// `ff_huff_vlc[1..=15]` (index 0 unused, dummy present).
    huff_vlc: Vec<BigVlc>,
    /// `ff_huff_quad_vlc[2]`.
    huff_quad: [QuadVlc; 2],
}

// ---------------------------------------------------------------------
// Header decode — mpegaudiodecheader.c + mpegaudiodecheader.h
// ---------------------------------------------------------------------

/// `MPA_DECODE_HEADER` (`mpegaudiodecheader.h:35-49`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MpaDecodeHeader {
    pub frame_size: i32,
    pub error_protection: i32,
    pub layer: i32,
    pub sample_rate: i32,
    /// between 0 and 8.
    pub sample_rate_index: i32,
    pub bit_rate: i32,
    pub nb_channels: i32,
    pub mode: i32,
    pub mode_ext: i32,
    pub lsf: i32,
}

/// `ff_mpa_check_header` (`mpegaudiodecheader.h:62-79`) — the fast
/// resync check. C returns a bare -1; here each rejection is named.
pub fn ff_mpa_check_header(header: u32) -> Result<()> {
    // header
    if (header & 0xffe0_0000) != 0xffe0_0000 {
        return Err(Error::InvalidData("invalid frame sync".into()));
    }
    // version check
    if (header & (3 << 19)) == 1 << 19 {
        return Err(Error::InvalidData("reserved MPEG audio version".into()));
    }
    // layer check
    if (header & (3 << 17)) == 0 {
        return Err(Error::InvalidData("invalid MPEG audio layer".into()));
    }
    // bit rate
    if (header & (0xf << 12)) == 0xf << 12 {
        return Err(Error::InvalidData("invalid bitrate index".into()));
    }
    // frequency
    if (header & (3 << 10)) == 3 << 10 {
        return Err(Error::InvalidData("invalid sample rate index".into()));
    }
    Ok(())
}

/// `avpriv_mpegaudio_decode_header` (`mpegaudiodecheader.c:34-118`).
///
/// Returns `Ok(true)` for a free-format frame (bitrate index 0; C's
/// return value 1 — "frame size must be computed externally", rejected
/// by `decode_frame`), `Ok(false)` when the header filled in fully.
pub fn avpriv_mpegaudio_decode_header(s: &mut MpaDecodeHeader, header: u32) -> Result<bool> {
    ff_mpa_check_header(header)?;

    let (lsf, mpeg25) = if header & (1 << 20) != 0 {
        (((header & (1 << 19)) == 0) as i32, 0)
    } else {
        (1, 1)
    };
    s.lsf = lsf;

    s.layer = 4 - ((header >> 17) & 3) as i32;
    // extract frequency
    let mut sample_rate_index = ((header >> 10) & 3) as i32;
    if sample_rate_index as usize >= FF_MPA_FREQ_TAB.len() {
        sample_rate_index = 0;
    }
    let sample_rate = (FF_MPA_FREQ_TAB[sample_rate_index as usize] >> (lsf + mpeg25)) as i32;
    sample_rate_index += 3 * (lsf + mpeg25);
    s.sample_rate_index = sample_rate_index;
    s.error_protection = (((header >> 16) & 1) ^ 1) as i32;
    s.sample_rate = sample_rate;

    let bitrate_index = ((header >> 12) & 0xf) as usize;
    let padding = ((header >> 9) & 1) as i32;
    s.mode = ((header >> 6) & 3) as i32;
    s.mode_ext = ((header >> 4) & 3) as i32;

    s.nb_channels = if s.mode == MPA_MONO { 1 } else { 2 };

    if bitrate_index != 0 {
        let mut frame_size =
            FF_MPA_BITRATE_TAB[s.lsf as usize][(s.layer - 1) as usize][bitrate_index] as i32;
        s.bit_rate = frame_size * 1000;
        match s.layer {
            1 => {
                frame_size = (frame_size * 12000) / sample_rate;
                frame_size = (frame_size + padding) * 4;
            }
            2 => {
                frame_size = (frame_size * 144000) / sample_rate;
                frame_size += padding;
            }
            _ => {
                frame_size = (frame_size * 144000) / (sample_rate << s.lsf);
                frame_size += padding;
            }
        }
        s.frame_size = frame_size;
    } else {
        // if no frame size computed, signal it
        return Ok(true);
    }
    Ok(false)
}

/// `ff_mpa_decode_header` (`mpegaudiodecheader.c:120-152`) — the
/// stream-probe helper: full header parse plus the layer → codec /
/// samples-per-frame mapping. Returns the coded frame size in bytes.
pub fn ff_mpa_decode_header(
    head: u32,
    sample_rate: &mut i32,
    channels: &mut i32,
    frame_size: &mut i32,
    bit_rate: &mut i32,
    codec_id: &mut CodecId,
) -> Result<i32> {
    let mut s = MpaDecodeHeader::default();
    if avpriv_mpegaudio_decode_header(&mut s, head)? {
        return Err(Error::InvalidData(
            "free-format frame: frame size must be computed externally".into(),
        ));
    }

    *frame_size = match s.layer {
        1 => {
            *codec_id = CodecId::Mp1;
            384
        }
        2 => {
            *codec_id = CodecId::Mp2;
            1152
        }
        _ => {
            // C keeps AV_CODEC_ID_MP3ADU when the caller passed it; the
            // port has no ADU codec, so the mapping is unconditional.
            *codec_id = CodecId::Mp3;
            if s.lsf != 0 { 576 } else { 1152 }
        }
    };

    *sample_rate = s.sample_rate;
    *channels = s.nb_channels;
    *bit_rate = s.bit_rate;
    Ok(s.frame_size)
}

// ---------------------------------------------------------------------
// Runtime-generated tables (C's tablegen headers, computed at startup)
// ---------------------------------------------------------------------

/// `frexp(3)` — C uses it in `mpegaudiodec_common_tablegen.h:59`; std
/// has no frexp. Returns `(m, e)` with `f == m * 2^e`, `m ∈ [0.5, 1)`.
fn frexp(f: f64) -> (f64, i32) {
    if f == 0.0 || !f.is_finite() {
        return (f, 0);
    }
    let bits = f.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    if raw_exp == 0 {
        // subnormal: scale up then recurse (never hit by these tables,
        // kept for exactness of the helper).
        let (m, e) = frexp(f * 2f64.powi(64));
        return (m, e - 64);
    }
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (0x3feu64 << 52));
    (m, raw_exp - 1022)
}

/// `llrint` with the C default rounding mode (round-to-nearest-even);
/// `f64::round` is half-away-from-zero and differs on exact `.5`.
fn llrint_even(v: f64) -> i64 {
    let mut r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r -= r.signum();
    }
    r as i64
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> Tables {
    // ---- ff_band_index_long (mpegaudiodec_common.c:450-457) ----
    let mut band_index_long = [[0u16; 23]; 9];
    for i in 0..9 {
        let mut k = 0u16;
        for j in 0..22 {
            band_index_long[i][j] = k;
            k += (FF_BAND_SIZE_LONG[i][j] >> 1) as u16;
        }
        band_index_long[i][22] = k;
    }

    // ---- exp/expval float tables (mpegaudio_tablegen.h:48-84) ----
    let mut pow43_lut = [0f64; 16];
    for (i, v) in pow43_lut.iter_mut().enumerate() {
        *v = i as f64 * (i as f64).cbrt();
    }
    // 2^(-72), written as the exact C decimal literal.
    let exp2_base_0 = 2.11758236813575084767080625169910490512847900390625e-22f64;
    let mut exp_table = vec![0f32; 512];
    let mut expval_table = vec![[0f32; 16]; 512];
    let mut exp2_base = exp2_base_0;
    for exponent in 0..512usize {
        if exponent > 0 && exponent & 3 == 0 {
            exp2_base *= 2.0;
        }
        let exp2_val = exp2_base * EXP2_LUT[exponent & 3] / IMDCT_SCALAR;
        for value in 0..16 {
            expval_table[exponent][value] = (pow43_lut[value] * exp2_val) as f32;
        }
        exp_table[exponent] = expval_table[exponent][1];
    }

    // ---- ff_table_4_3 (mpegaudiodec_common_tablegen.h:44-69) ----
    // FRAC_BITS = 23 there.
    let mut table_4_3_exp = vec![0i8; TABLE_4_3_SIZE];
    let mut table_4_3_value = vec![0u32; TABLE_4_3_SIZE];
    let mut pow43_val = 0f64;
    for i in 1..TABLE_4_3_SIZE {
        let value = i as f64 / 4.0;
        if (i & 3) == 0 {
            pow43_val = value / IMDCT_SCALAR * value.cbrt();
        }
        let f = pow43_val * EXP2_LUT[i & 3];
        let (fm, e) = frexp(f);
        let m = llrint_even(fm * (1u64 << 31) as f64);
        let e = e + 23 - 31 + 5 - 100; // FRAC_BITS - 31 + 5 - 100
        table_4_3_value[i] = m as u32;
        table_4_3_exp[i] = (-e) as i8;
    }

    // ---- ff_mdct_win (mpegaudiodsp.c:30-79) ----
    let mut mdct_win = [[0f32; MDCT_BUF_SIZE]; 8];
    for i in 0..36usize {
        for j in 0..4usize {
            if j == 2 && i % 3 != 1 {
                continue;
            }
            let mut d = (std::f64::consts::PI * (i as f64 + 0.5) / 36.0).sin();
            if j == 1 {
                if i >= 30 {
                    d = 0.0;
                } else if i >= 24 {
                    d = (std::f64::consts::PI * (i as f64 - 18.0 + 0.5) / 12.0).sin();
                } else if i >= 18 {
                    d = 1.0;
                }
            } else if j == 3 {
                if i < 6 {
                    d = 0.0;
                } else if i < 12 {
                    d = (std::f64::consts::PI * (i as f64 - 6.0 + 0.5) / 12.0).sin();
                } else if i < 18 {
                    d = 1.0;
                }
            }
            // merge last stage of imdct into the window coefficients
            d *= 0.5 * IMDCT_SCALAR / (std::f64::consts::PI * (2.0 * i as f64 + 19.0) / 72.0).cos();
            if j == 2 {
                mdct_win[j][i / 3] = (d / 32.0) as f32;
            } else {
                let idx = if i < 18 {
                    i
                } else {
                    i + (MDCT_BUF_SIZE / 2 - 18)
                };
                mdct_win[j][idx] = (d / 32.0) as f32;
            }
        }
    }
    // NOTE: we do frequency inversion after the MDCT by changing
    // the sign of the right window coefs (mpegaudiodsp.c:65-74).
    for j in 0..4 {
        let mut i = 0;
        while i < MDCT_BUF_SIZE {
            mdct_win[j + 4][i] = mdct_win[j][i];
            mdct_win[j + 4][i + 1] = -mdct_win[j][i + 1];
            i += 2;
        }
    }

    // ---- ff_mpa_synth_window (mpegaudiodsp_template.c:197-224) ----
    // max = 18760, max sum over all 16 coefs : 44736
    let mut synth_window = vec![0f32; 512 + 256];
    for i in 0..257usize {
        let mut v = (FF_MPA_ENWINDOW[i] as f64 * (1.0 / (1u64 << (16 + 23)) as f64)) as f32;
        synth_window[i] = v;
        if (i & 63) != 0 {
            v = -v;
        }
        if i != 0 {
            synth_window[512 - i] = v;
        }
    }
    // Needed for avoiding shuffles in ASM implementations
    for i in 0..8usize {
        for j in 0..16usize {
            synth_window[512 + 16 * i + j] = synth_window[64 * i + 32 - j];
        }
    }
    for i in 0..8usize {
        for j in 0..16usize {
            synth_window[512 + 128 + 16 * i + j] = synth_window[64 * i + 48 - j];
        }
    }

    // ---- is_table_lsf (mpegaudiodec_template.c:264-278) ----
    let mut is_table_lsf = [[[0f32; 16]; 2]; 2];
    for i in 0..16usize {
        for j in 0..2usize {
            let e = -((j + 1) as f64) * (((i + 1) >> 1) as f64);
            let f = (e / 4.0).exp2();
            let k = i & 1;
            is_table_lsf[j][k ^ 1][i] = f as f32;
            is_table_lsf[j][k][i] = 1.0f32;
        }
    }

    // ---- ff_huff_vlc[1..=15] (mpegaudiodec_common.c:404-436) ----
    let mut huff_vlc = Vec::with_capacity(16);
    huff_vlc.push(BigVlc::build(&[], &[])); // index 0 unused
    let mut lens_off = 0usize;
    let mut sym_off = 0usize;
    for &nb_codes_minus_one in MPA_HUFF_SIZES_MINUS_ONE {
        let n = nb_codes_minus_one + 1;
        huff_vlc.push(BigVlc::build(
            &MPA_HUFFLENS[lens_off..lens_off + n],
            &MPA_HUFFSYMBOLS[sym_off..sym_off + n],
        ));
        lens_off += n;
        sym_off += n;
    }
    debug_assert_eq!(lens_off, MPA_HUFFLENS.len());
    debug_assert_eq!(sym_off, MPA_HUFFSYMBOLS.len());

    // ---- ff_huff_quad_vlc[2] (mpegaudiodec_common.c:438-447) ----
    let huff_quad = [
        QuadVlc::build(6, &MPA_QUAD_BITS[0], &MPA_QUAD_CODES[0]),
        QuadVlc::build(4, &MPA_QUAD_BITS[1], &MPA_QUAD_CODES[1]),
    ];

    Tables {
        band_index_long,
        exp_table,
        expval_table,
        table_4_3_exp,
        table_4_3_value,
        mdct_win,
        synth_window,
        is_table_lsf,
        huff_vlc,
        huff_quad,
    }
}
// ---------------------------------------------------------------------
// Float DSP helpers — the USE_FLOATS macro set
// (mpegaudiodsp_template.c:35-49, dct32_template.c:34-46)
// ---------------------------------------------------------------------

/// `MULH3(x, y, s)` with `s = 1 << shift` (float build: `(s)*(y)*(x)`).
#[inline(always)]
fn mulh3(x: f32, y: f32, s: f32) -> f32 {
    s * y * x
}

/// `MULLx(x, y, FRAC_BITS)` (float build: `(y)*(x)`).
#[inline(always)]
fn mullx(x: f32, y: f32) -> f32 {
    y * x
}

/// `SHR(a, b)` (float build: `a * (1.0 / (1 << b))`).
#[inline(always)]
fn shr(a: f32, b: u32) -> f32 {
    a * (1.0f32 / (1u32 << b) as f32)
}

/// `FIXR`/`FIXHR` for the float build: `((x) as f64) as f32` mirrors
/// C's double constant folding followed by the float cast.
#[inline(always)]
const fn fx(x: f64) -> f32 {
    x as f32
}

/// 12 points IMDCT — `imdct12` (`mpegaudiodec_template.c:329-368`),
/// computed "by hand" by factorizing obvious cases. `in` is the
/// stride-3 triple starting at `base` (the three short-window
/// subblocks of one subband).
fn imdct12(out: &mut [f32; 12], sb: &[f32], base: usize) {
    let at = |k: usize| sb[base + k];
    let in0 = at(0 * 3);
    let in1 = at(1 * 3) + at(0 * 3);
    let mut in2 = at(2 * 3) + at(1 * 3);
    let mut in3 = at(3 * 3) + at(2 * 3);
    let in4 = at(4 * 3) + at(3 * 3);
    let mut in5 = at(5 * 3) + at(4 * 3);
    in5 += in3;
    in3 += in1;

    in2 = mulh3(in2, I12_C3, 2.0);
    in3 = mulh3(in3, I12_C3, 4.0);

    let t1 = in0 - in4;
    let t2 = mulh3(in1 - in5, I12_C4, 2.0);

    out[7] = t1 + t2;
    out[10] = t1 + t2;
    out[1] = t1 - t2;
    out[4] = t1 - t2;

    let mut in0 = in0 + shr(in4, 1);
    let in4b = in0 + in2;
    in5 += 2.0 * in1;
    let in1b = mulh3(in5 + in3, I12_C5, 1.0);
    out[8] = in4b + in1b;
    out[9] = in4b + in1b;
    out[2] = in4b - in1b;
    out[3] = in4b - in1b;

    in0 -= in2;
    let in5b = mulh3(in5 - in3, I12_C6, 2.0);
    out[0] = in0 - in5b;
    out[5] = in0 - in5b;
    out[6] = in0 + in5b;
    out[11] = in0 + in5b;
}

/// `imdct36` — `mpegaudiodsp_template.c:274-352`, Lee-like
/// decomposition followed by a hand coded 9-point DCT. `out`/`buf` are
/// flat arrays with the block's base offsets applied; `in` is this
/// block's 18 spectral lines (modified in place: prefix sums).
fn imdct36(
    out: &mut [f32],
    out_off: usize,
    buf: &mut [f32],
    buf_off: usize,
    input: &mut [f32],
    win: &[f32],
) {
    let mut tmp = [0f32; 18];

    let mut i = 17i32;
    while i >= 1 {
        input[i as usize] += input[(i - 1) as usize];
        i -= 1;
    }
    let mut i = 17i32;
    while i >= 3 {
        input[i as usize] += input[(i - 2) as usize];
        i -= 2;
    }

    for j in 0..2usize {
        let in1 = &input[j..];
        let t2 = in1[2 * 4] + in1[2 * 8] - in1[2 * 2];

        let t3 = in1[2 * 0] + shr(in1[2 * 6], 1);
        let t1 = in1[2 * 0] - in1[2 * 6];
        tmp[j + 6] = t1 - shr(t2, 1);
        tmp[j + 16] = t1 + t2;

        let t0 = mulh3(in1[2 * 2] + in1[2 * 4], C2, 2.0);
        let t1 = mulh3(in1[2 * 4] - in1[2 * 8], -2.0 * C8, 1.0);
        let t2 = mulh3(in1[2 * 2] + in1[2 * 8], -C4, 2.0);

        tmp[j + 10] = t3 - t0 - t2;
        tmp[j + 2] = t3 + t0 + t1;
        tmp[j + 14] = t3 + t2 - t1;

        tmp[j + 4] = mulh3(in1[2 * 5] + in1[2 * 7] - in1[2 * 1], -C3, 2.0);
        let t2 = mulh3(in1[2 * 1] + in1[2 * 5], C1, 2.0);
        let t3 = mulh3(in1[2 * 5] - in1[2 * 7], -2.0 * C7, 1.0);
        let t0 = mulh3(in1[2 * 3], C3, 2.0);

        let t1 = mulh3(in1[2 * 1] + in1[2 * 7], -C5, 2.0);

        tmp[j] = t2 + t3 + t0;
        tmp[j + 12] = t2 + t1 - t0;
        tmp[j + 8] = t3 - t1 - t0;
    }

    let mut i = 0usize;
    for j in 0..4usize {
        let t0 = tmp[i];
        let t1 = tmp[i + 2];
        let s0 = t1 + t0;
        let s2 = t1 - t0;

        let t2 = tmp[i + 1];
        let t3 = tmp[i + 3];
        let s1 = mulh3(t3 + t2, ICOS36H[j], 2.0);
        let s3 = mullx(t3 - t2, ICOS36[8 - j]);

        let t0 = s0 + s1;
        let t1 = s0 - s1;
        out[out_off + (9 + j) * SBLIMIT] = mulh3(t1, win[9 + j], 1.0) + buf[buf_off + 4 * (9 + j)];
        out[out_off + (8 - j) * SBLIMIT] = mulh3(t1, win[8 - j], 1.0) + buf[buf_off + 4 * (8 - j)];
        buf[buf_off + 4 * (9 + j)] = mulh3(t0, win[MDCT_BUF_SIZE / 2 + 9 + j], 1.0);
        buf[buf_off + 4 * (8 - j)] = mulh3(t0, win[MDCT_BUF_SIZE / 2 + 8 - j], 1.0);

        let t0 = s2 + s3;
        let t1 = s2 - s3;
        out[out_off + (9 + 8 - j) * SBLIMIT] =
            mulh3(t1, win[9 + 8 - j], 1.0) + buf[buf_off + 4 * (9 + 8 - j)];
        out[out_off + j * SBLIMIT] = mulh3(t1, win[j], 1.0) + buf[buf_off + 4 * j];
        buf[buf_off + 4 * (9 + 8 - j)] = mulh3(t0, win[MDCT_BUF_SIZE / 2 + 9 + 8 - j], 1.0);
        buf[buf_off + 4 * j] = mulh3(t0, win[MDCT_BUF_SIZE / 2 + j], 1.0);
        i += 4;
    }

    let s0 = tmp[16];
    let s1 = mulh3(tmp[17], ICOS36H[4], 2.0);
    let t0 = s0 + s1;
    let t1 = s0 - s1;
    out[out_off + (9 + 4) * SBLIMIT] = mulh3(t1, win[9 + 4], 1.0) + buf[buf_off + 4 * (9 + 4)];
    out[out_off + (8 - 4) * SBLIMIT] = mulh3(t1, win[8 - 4], 1.0) + buf[buf_off + 4 * (8 - 4)];
    buf[buf_off + 4 * (9 + 4)] = mulh3(t0, win[MDCT_BUF_SIZE / 2 + 9 + 4], 1.0);
    buf[buf_off + 4 * (8 - 4)] = mulh3(t0, win[MDCT_BUF_SIZE / 2 + 8 - 4], 1.0);
}

/// `ff_imdct36_blocks_float` (`mpegaudiodsp_template.c:354-371`).
/// `out` is the granule's `sb_samples` base, `buf` the channel's
/// `mdct_buf`, `input` the granule's `sb_hybrid`.
fn imdct36_blocks(
    out: &mut [f32],
    out_base: usize,
    buf: &mut [f32],
    buf_base: usize,
    input: &mut [f32],
    count: usize,
    switch_point: bool,
    block_type: u8,
) {
    let t = tables();
    let mut in_off = 0usize;
    let mut buf_off = buf_base;
    let mut out_off = out_base;
    for j in 0..count {
        // select window
        let win_idx = if switch_point && j < 2 {
            0
        } else {
            block_type as usize
        };
        let win = &t.mdct_win[win_idx + (4 & (0usize.wrapping_sub(j & 1)))];
        imdct36(
            out,
            out_off,
            buf,
            buf_off,
            &mut input[in_off..in_off + 18],
            win,
        );
        in_off += 18;
        buf_off += if (j & 3) != 3 { 1 } else { 72 - 3 };
        out_off += 1;
    }
}

/// `ff_dct32_float` (`dct32_template.c:126-288`) — DCT32 without
/// 1/sqrt(2) coefficient scaling, butterfly form. `tab` holds the 32
/// subband samples; `out` receives 32 values.
pub fn dct32(out: &mut [f32], tab: &[f32]) {
    let mut val = [0f32; 32];

    // BF(a, b, c, s): butterfly on the register array
    macro_rules! bf {
        ($a:literal, $b:literal, $c:expr, $s:literal) => {{
            let tmp0 = val[$a] + val[$b];
            let tmp1 = val[$a] - val[$b];
            val[$a] = tmp0;
            val[$b] = mulh3(tmp1, $c, (1u32 << $s) as f32);
        }};
    }
    // BF0: butterfly straight from the input table
    macro_rules! bf0 {
        ($a:literal, $b:literal, $c:expr, $s:literal) => {{
            let tmp0 = tab[$a] + tab[$b];
            let tmp1 = tab[$a] - tab[$b];
            val[$a] = tmp0;
            val[$b] = mulh3(tmp1, $c, (1u32 << $s) as f32);
        }};
    }
    macro_rules! bf1 {
        ($a:literal, $b:literal, $c:literal, $d:literal) => {{
            bf!($a, $b, COS4_0, 1);
            bf!($c, $d, -COS4_0, 1);
            val[$c] += val[$d];
        }};
    }
    macro_rules! bf2 {
        ($a:literal, $b:literal, $c:literal, $d:literal) => {{
            bf!($a, $b, COS4_0, 1);
            bf!($c, $d, -COS4_0, 1);
            val[$c] += val[$d];
            val[$a] += val[$c];
            val[$c] += val[$b];
            val[$b] += val[$d];
        }};
    }
    macro_rules! add {
        ($a:literal, $b:literal) => {
            val[$a] += val[$b]
        };
    }

    /* pass 1 */
    bf0!(0, 31, COS0_0, 1);
    bf0!(15, 16, COS0_15, 5);
    /* pass 2 */
    bf!(0, 15, COS1_0, 1);
    bf!(16, 31, -COS1_0, 1);
    /* pass 1 */
    bf0!(7, 24, COS0_7, 1);
    bf0!(8, 23, COS0_8, 1);
    /* pass 2 */
    bf!(7, 8, COS1_7, 4);
    bf!(23, 24, -COS1_7, 4);
    /* pass 3 */
    bf!(0, 7, COS2_0, 1);
    bf!(8, 15, -COS2_0, 1);
    bf!(16, 23, COS2_0, 1);
    bf!(24, 31, -COS2_0, 1);
    /* pass 1 */
    bf0!(3, 28, COS0_3, 1);
    bf0!(12, 19, COS0_12, 2);
    /* pass 2 */
    bf!(3, 12, COS1_3, 1);
    bf!(19, 28, -COS1_3, 1);
    /* pass 1 */
    bf0!(4, 27, COS0_4, 1);
    bf0!(11, 20, COS0_11, 2);
    /* pass 2 */
    bf!(4, 11, COS1_4, 1);
    bf!(20, 27, -COS1_4, 1);
    /* pass 3 */
    bf!(3, 4, COS2_3, 3);
    bf!(11, 12, -COS2_3, 3);
    bf!(19, 20, COS2_3, 3);
    bf!(27, 28, -COS2_3, 3);
    /* pass 4 */
    bf!(0, 3, COS3_0, 1);
    bf!(4, 7, -COS3_0, 1);
    bf!(8, 11, COS3_0, 1);
    bf!(12, 15, -COS3_0, 1);
    bf!(16, 19, COS3_0, 1);
    bf!(20, 23, -COS3_0, 1);
    bf!(24, 27, COS3_0, 1);
    bf!(28, 31, -COS3_0, 1);

    /* pass 1 */
    bf0!(1, 30, COS0_1, 1);
    bf0!(14, 17, COS0_14, 3);
    /* pass 2 */
    bf!(1, 14, COS1_1, 1);
    bf!(17, 30, -COS1_1, 1);
    /* pass 1 */
    bf0!(6, 25, COS0_6, 1);
    bf0!(9, 22, COS0_9, 1);
    /* pass 2 */
    bf!(6, 9, COS1_6, 2);
    bf!(22, 25, -COS1_6, 2);
    /* pass 3 */
    bf!(1, 6, COS2_1, 1);
    bf!(9, 14, -COS2_1, 1);
    bf!(17, 22, COS2_1, 1);
    bf!(25, 30, -COS2_1, 1);

    /* pass 1 */
    bf0!(2, 29, COS0_2, 1);
    bf0!(13, 18, COS0_13, 3);
    /* pass 2 */
    bf!(2, 13, COS1_2, 1);
    bf!(18, 29, -COS1_2, 1);
    /* pass 1 */
    bf0!(5, 26, COS0_5, 1);
    bf0!(10, 21, COS0_10, 1);
    /* pass 2 */
    bf!(5, 10, COS1_5, 2);
    bf!(21, 26, -COS1_5, 2);
    /* pass 3 */
    bf!(2, 5, COS2_2, 1);
    bf!(10, 13, -COS2_2, 1);
    bf!(18, 21, COS2_2, 1);
    bf!(26, 29, -COS2_2, 1);
    /* pass 4 */
    bf!(1, 2, COS3_1, 2);
    bf!(5, 6, -COS3_1, 2);
    bf!(9, 10, COS3_1, 2);
    bf!(13, 14, -COS3_1, 2);
    bf!(17, 18, COS3_1, 2);
    bf!(21, 22, -COS3_1, 2);
    bf!(25, 26, COS3_1, 2);
    bf!(29, 30, -COS3_1, 2);

    /* pass 5 */
    bf1!(0, 1, 2, 3);
    bf2!(4, 5, 6, 7);
    bf1!(8, 9, 10, 11);
    bf2!(12, 13, 14, 15);
    bf1!(16, 17, 18, 19);
    bf2!(20, 21, 22, 23);
    bf1!(24, 25, 26, 27);
    bf2!(28, 29, 30, 31);

    /* pass 6 */

    add!(8, 12);
    add!(12, 10);
    add!(10, 14);
    add!(14, 9);
    add!(9, 13);
    add!(13, 11);
    add!(11, 15);

    out[0] = val[0];
    out[16] = val[1];
    out[8] = val[2];
    out[24] = val[3];
    out[4] = val[4];
    out[20] = val[5];
    out[12] = val[6];
    out[28] = val[7];
    out[2] = val[8];
    out[18] = val[9];
    out[10] = val[10];
    out[26] = val[11];
    out[6] = val[12];
    out[22] = val[13];
    out[14] = val[14];
    out[30] = val[15];

    add!(24, 28);
    add!(28, 26);
    add!(26, 30);
    add!(30, 25);
    add!(25, 29);
    add!(29, 27);
    add!(27, 31);

    out[1] = val[16] + val[24];
    out[17] = val[17] + val[25];
    out[9] = val[18] + val[26];
    out[25] = val[19] + val[27];
    out[5] = val[20] + val[28];
    out[21] = val[21] + val[29];
    out[13] = val[22] + val[30];
    out[29] = val[23] + val[31];
    out[3] = val[24] + val[20];
    out[19] = val[25] + val[21];
    out[11] = val[26] + val[22];
    out[27] = val[27] + val[23];
    out[7] = val[28] + val[18];
    out[23] = val[29] + val[19];
    out[15] = val[30] + val[17];
    out[31] = val[31];
}

/// `round_sample` float build (`mpegaudiodsp_template.c:35-40`):
/// return the accumulator and zero it.
#[inline(always)]
fn round_sample(sum: &mut f32) -> f32 {
    let sum1 = *sum;
    *sum = 0.0;
    sum1
}

/// `ff_mpadsp_apply_window_float` (`mpegaudiodsp_template.c:123-174`).
/// `synth_buf` is the 1024-float channel window at the current offset
/// base; `samples`/`incr` form the output pointer/stride.
fn apply_window(
    synth_buf: &mut [f32],
    window: &[f32],
    dither_state: &mut i32,
    samples: &mut [f32],
    incr: usize,
) {
    // copy to avoid wrap
    synth_buf.copy_within(0..32, 512);

    let mut s_idx = 0usize;
    let mut s2_idx = 31 * incr;
    let mut w = 0usize;
    let mut w2 = 31usize;

    let mut sum = *dither_state as f32;
    // SUM8(MACS, sum, w, p) with p = synth_buf + 16
    for k in 0..8 {
        sum += window[w + k * 64] * synth_buf[16 + k * 64];
    }
    // SUM8(MLSS, sum, w + 32, p) with p = synth_buf + 48
    for k in 0..8 {
        sum -= window[w + 32 + k * 64] * synth_buf[48 + k * 64];
    }
    samples[s_idx] = round_sample(&mut sum);
    s_idx += incr;
    w += 1;

    // we calculate two samples at the same time to avoid one memory
    // access per two sample
    for j in 1..16usize {
        let mut sum2 = 0f32;
        // SUM8P2(sum, MACS, sum2, MLSS, w, w2, p), p = synth_buf + 16 + j
        for k in 0..8 {
            let t = synth_buf[16 + j + k * 64];
            sum += window[w + k * 64] * t;
            sum2 -= window[w2 + k * 64] * t;
        }
        // SUM8P2(sum, MLSS, sum2, MLSS, w + 32, w2 + 32, p), p = synth_buf + 48 - j
        for k in 0..8 {
            let t = synth_buf[48 - j + k * 64];
            sum -= window[w + 32 + k * 64] * t;
            sum2 -= window[w2 + 32 + k * 64] * t;
        }

        samples[s_idx] = round_sample(&mut sum);
        s_idx += incr;
        sum += sum2;
        samples[s2_idx] = round_sample(&mut sum);
        s2_idx -= incr;
        w += 1;
        w2 -= 1;
    }

    // SUM8(MLSS, sum, w + 32, p), p = synth_buf + 32
    for k in 0..8 {
        sum -= window[w + 32 + k * 64] * synth_buf[32 + k * 64];
    }
    samples[s_idx] = round_sample(&mut sum);
    *dither_state = sum as i32;
}

/// `ff_mpa_synth_filter_float` (`mpegaudiodsp_template.c:178-195`) —
/// 32 sub band synthesis: input 32 subband samples, output 32 samples.
fn mpa_synth_filter(
    synth_buf: &mut [f32],
    synth_buf_offset: &mut usize,
    window: &[f32],
    dither_state: &mut i32,
    samples: &mut [f32],
    incr: usize,
    sb_samples: &[f32],
) {
    let offset = *synth_buf_offset;
    dct32(&mut synth_buf[offset..offset + 32], sb_samples);
    apply_window(
        &mut synth_buf[offset..],
        window,
        dither_state,
        samples,
        incr,
    );
    *synth_buf_offset = (offset + 512 - 32) & 511;
}
/// `l3_unscale` (`mpegaudiodec_template.c:222-239`) — compute
/// `value^(4/3) * 2^(exponent/4)` normalized to FRAC_BITS, via the
/// shared fixed-point 4/3 table (used by the float decoder's linbits
/// escape path exactly as in C).
fn l3_unscale(value: i64, exponent: i32) -> i32 {
    let t = tables();
    let idx = (4 * value + (exponent & 3) as i64) as usize;
    let idx = idx.min(TABLE_4_3_SIZE - 1);
    let mut e = t.table_4_3_exp[idx] as i32;
    let m = t.table_4_3_value[idx];
    e -= exponent >> 2;
    if !(0..=31).contains(&e) {
        // C: `if (e > (SUINT)31) return 0;` — negative e reads as
        // unsigned, so both out-of-range directions return 0.
        return 0;
    }
    let m = (m + ((1u32 << e) >> 1)) >> e;
    m as i32
}

// ---------------------------------------------------------------------
// Layer 3 scale-factor helpers — mpegaudiodec_template.c:659-723
// ---------------------------------------------------------------------

/// The `SPLIT` macro (`mpegaudiodec_template.c:659-677`).
fn split(dst: &mut i32, sf: &mut i32, n: i32) {
    match n {
        3 => {
            let m = (*sf * 171) >> 9;
            *dst = *sf - 3 * m;
            *sf = m;
        }
        4 => {
            *dst = *sf & 3;
            *sf >>= 2;
        }
        5 => {
            let m = (*sf * 205) >> 10;
            *dst = *sf - 5 * m;
            *sf = m;
        }
        6 => {
            let m = (*sf * 171) >> 10;
            *dst = *sf - 6 * m;
            *sf = m;
        }
        _ => *dst = 0,
    }
}

/// `lsf_sf_expand` (`mpegaudiodec_template.c:679-686`).
pub fn lsf_sf_expand(slen: &mut [i32; 4], mut sf: i32, n1: i32, n2: i32, n3: i32) {
    split(&mut slen[3], &mut sf, n3);
    split(&mut slen[2], &mut sf, n2);
    split(&mut slen[1], &mut sf, n1);
    slen[0] = sf;
}

/// `exponents_from_scale_factors` (`mpegaudiodec_template.c:688-723`).
fn exponents_from_scale_factors(sri: i32, g: &GranuleDef, exponents: &mut [i16; 576]) {
    let mut ptr = 0usize;
    let gain = g.global_gain - 210;
    let shift = g.scalefac_scale as i32 + 1;

    let sri = sri as usize;
    let bstab = &FF_BAND_SIZE_LONG[sri];
    let pretab = &FF_MPA_PRETAB[(g.preflag != 0) as usize];
    for i in 0..g.long_end as usize {
        let v0 = gain - ((g.scale_factors[i] as i32 + pretab[i] as i32) << shift) + 400;
        for _ in 0..bstab[i] {
            exponents[ptr] = v0 as i16;
            ptr += 1;
        }
    }

    if g.short_start < 13 {
        let bstab = &FF_BAND_SIZE_SHORT[sri];
        let gains = [
            gain - (g.subblock_gain[0] << 3),
            gain - (g.subblock_gain[1] << 3),
            gain - (g.subblock_gain[2] << 3),
        ];
        let mut k = g.long_end as usize;
        for i in g.short_start as usize..13 {
            let len = bstab[i];
            for l in 0..3usize {
                let v0 = gains[l] - ((g.scale_factors[k] as i32) << shift) + 400;
                k += 1;
                for _ in 0..len {
                    exponents[ptr] = v0 as i16;
                    ptr += 1;
                }
            }
        }
    }
}

/// `switch_buffer` (`mpegaudiodec_template.c:725-738`) — hop from the
/// reservoir view back to the frame reader when the position runs past
/// the reservoir-only bytes, fixing up the caller's position and
/// end-position bookkeeping.
fn switch_buffer(bs: &mut Bitstream, pos: &mut i64, end_pos: &mut i64, end_pos2: &mut i64) {
    if bs.in_gb.is_some() && *pos >= bs.gb.size_in_bits - bs.extrasize as i64 * 8 {
        let mut gb = bs.in_gb.take().unwrap();
        bs.extrasize = 0;
        gb.skip_bits_long(*pos - *end_pos);
        *end_pos2 = *end_pos2 + gb.get_bits_count() - *pos;
        *end_pos = *end_pos2;
        *pos = gb.get_bits_count();
        bs.gb = gb;
    }
}

/// `huffman_decode` (`mpegaudiodec_template.c:756-903`) — read the
/// Huffman-coded residue into `g->sb_hybrid`.
fn huffman_decode(
    bs: &mut Bitstream,
    g: &mut GranuleDef,
    exponents: &[i16; 576],
    end_pos2_in: i64,
) {
    let mut end_pos2 = end_pos2_in;
    let mut end_pos = end_pos2.min(bs.gb.size_in_bits - bs.extrasize as i64 * 8);
    let t = tables();
    let mut s_index = 0usize;

    /* low frequencies (called big values) */
    for i in 0..3usize {
        let mut j = g.region_size[i];
        if j == 0 {
            continue;
        }
        // select vlc table
        let k = g.table_select[i] as usize;
        let l = FF_MPA_HUFF_DATA[k][0] as usize;
        let linbits = FF_MPA_HUFF_DATA[k][1] as u32;

        if l == 0 {
            let n = 2 * j as usize;
            g.sb_hybrid[s_index..s_index + n].fill(0.0);
            s_index += n;
            continue;
        }
        let vlc = &t.huff_vlc[l];

        // read huffcode and compute each couple
        while j > 0 {
            let mut pos = bs.gb.get_bits_count();
            if pos >= end_pos {
                switch_buffer(bs, &mut pos, &mut end_pos, &mut end_pos2);
                if pos >= end_pos {
                    break;
                }
            }
            let y = vlc.decode(&mut bs.gb);

            if y == 0 {
                g.sb_hybrid[s_index] = 0.0;
                g.sb_hybrid[s_index + 1] = 0.0;
                s_index += 2;
                continue;
            }

            // C indexes the float tables with the raw exponent; values
            // outside [0, 512) are undefined behavior there and clamp
            // to the table here (see module notes).
            let exponent = exponents[s_index].clamp(0, 511) as usize;
            let raw_exponent = exponents[s_index] as i32;
            if y & 16 != 0 {
                let x = y >> 5;
                let yy = y & 0x0f;
                g.sb_hybrid[s_index] = if x < 15 {
                    let mut v = t.expval_table[exponent][x as usize];
                    if bs.gb.get_bits1() != 0 {
                        v = -v;
                    }
                    v
                } else {
                    let xv = x as i64 + bs.gb.get_bits(linbits) as i64;
                    let mut v = l3_unscale(xv, raw_exponent) as f32;
                    if bs.gb.get_bits1() != 0 {
                        v = -v;
                    }
                    v
                };
                g.sb_hybrid[s_index + 1] = if yy < 15 {
                    let mut v = t.expval_table[exponent][yy as usize];
                    if bs.gb.get_bits1() != 0 {
                        v = -v;
                    }
                    v
                } else {
                    let yv = yy as i64 + bs.gb.get_bits(linbits) as i64;
                    let mut v = l3_unscale(yv, raw_exponent) as f32;
                    if bs.gb.get_bits1() != 0 {
                        v = -v;
                    }
                    v
                };
            } else {
                let x0 = y >> 5;
                let yy = y & 0x0f;
                let x = x0 + yy;
                let dst = s_index + (yy != 0) as usize;
                g.sb_hybrid[dst] = if x < 15 {
                    let mut v = t.expval_table[exponent][x as usize];
                    if bs.gb.get_bits1() != 0 {
                        v = -v;
                    }
                    v
                } else {
                    let xv = x as i64 + bs.gb.get_bits(linbits) as i64;
                    let mut v = l3_unscale(xv, raw_exponent) as f32;
                    if bs.gb.get_bits1() != 0 {
                        v = -v;
                    }
                    v
                };
                g.sb_hybrid[s_index + (yy == 0) as usize] = 0.0;
            }
            s_index += 2;
            j -= 1;
        }
    }

    /* high frequencies */
    let quad_bits = [6u32, 4u32];
    let vlc = &t.huff_quad[g.count1table_select as usize];
    let mut last_pos = 0i64;
    while s_index <= 572 {
        let mut pos = bs.gb.get_bits_count();
        if pos >= end_pos {
            if pos > end_pos2 && last_pos != 0 {
                // some encoders generate an incorrect size for this
                // part. We must go back into the data
                s_index -= 4;
                bs.gb.skip_bits_long(last_pos - pos);
                break;
            }
            switch_buffer(bs, &mut pos, &mut end_pos, &mut end_pos2);
            if pos >= end_pos {
                break;
            }
        }
        last_pos = pos;

        let mut code = vlc.decode(&mut bs.gb, quad_bits[g.count1table_select as usize]);
        if code < 0 {
            code = 0; // unreachable for these complete tables (C: -1)
        }
        g.sb_hybrid[s_index..s_index + 4].fill(0.0);
        while code != 0 {
            const IDXTAB: [usize; 16] = [3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];
            let idx = IDXTAB[code as usize];
            let pos_i = s_index + idx;
            let exp = exponents[pos_i].clamp(0, 511) as usize;
            let mut v = t.exp_table[exp];
            if bs.gb.get_bits1() != 0 {
                v = -v;
            }
            g.sb_hybrid[pos_i] = v;
            code ^= (8 >> idx) as i32;
        }
        s_index += 4;
    }
    // skip extension bits (the err_recognition checks around this are
    // off by default in C and not ported)
    let bits_left = end_pos2 - bs.gb.get_bits_count();
    g.sb_hybrid[s_index.min(576)..].fill(0.0);
    bs.gb.skip_bits_long(bits_left);

    let mut i = bs.gb.get_bits_count();
    switch_buffer(bs, &mut i, &mut end_pos, &mut end_pos2);
}

/// `reorder_block` (`mpegaudiodec_template.c:908-939`) — reorder short
/// blocks from bitstream order to interleaved order.
fn reorder_block(sri: i32, g: &mut GranuleDef) {
    if g.block_type != 2 {
        return;
    }
    let sri = sri as usize;
    let mut tmp = [0f32; 576];

    let mut ptr = if g.switch_point != 0 {
        if sri != 8 { 36 } else { 72 }
    } else {
        0
    };

    for i in g.short_start as usize..13 {
        let len = FF_BAND_SIZE_SHORT[sri][i] as usize;
        let ptr1 = ptr;
        let mut dst = 0usize;
        for _ in 0..len {
            tmp[dst] = g.sb_hybrid[ptr];
            dst += 1;
            tmp[dst] = g.sb_hybrid[ptr + len];
            dst += 1;
            tmp[dst] = g.sb_hybrid[ptr + 2 * len];
            dst += 1;
            ptr += 1;
        }
        ptr += 2 * len;
        for k in 0..len * 3 {
            g.sb_hybrid[ptr1 + k] = tmp[k];
        }
    }
}

/// `compute_stereo` (`mpegaudiodec_template.c:943-1071`) — intensity
/// stereo + mid/side stereo over the granule pair.
fn compute_stereo(
    mode_ext: i32,
    lsf: bool,
    sri: i32,
    granules: &mut [[GranuleDef; 2]; 2],
    gr: usize,
) {
    let (left, right) = granules.split_at_mut(1);
    let g0 = &mut left[0][gr];
    let g1 = &mut right[0][gr];
    let sb0 = &mut *g0.sb_hybrid;
    let sb1 = &mut *g1.sb_hybrid;
    let sri = sri as usize;

    if mode_ext & MODE_EXT_I_STEREO != 0 {
        let (is_tab, sf_max): (&[[f32; 16]; 2], i32) = if !lsf {
            (&IS_TABLE, 7)
        } else {
            (
                &tables().is_table_lsf[(g1.scalefac_compress & 1) as usize],
                16,
            )
        };

        let mut t0o = 576usize;
        let mut t1o = 576usize;
        let mut non_zero_found_short = [false; 3];
        let mut k = (13 - g1.short_start) * 3 + g1.long_end - 3;
        for i in (g1.short_start..=12).rev() {
            // for last band, use previous scale factor
            if i != 11 {
                k -= 3;
            }
            let len = FF_BAND_SIZE_SHORT[sri][i as usize] as usize;
            for l in (0..3usize).rev() {
                t0o -= len;
                t1o -= len;
                let mut did_istereo = false;
                if !non_zero_found_short[l] {
                    // test if non zero band. if so, stop doing i-stereo
                    let mut found = false;
                    for j in 0..len {
                        if sb1[t1o + j] != 0.0 {
                            non_zero_found_short[l] = true;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let sf = g1.scale_factors[(k + l as i32) as usize] as i32;
                        if sf < sf_max {
                            let v1 = is_tab[0][sf as usize];
                            let v2 = is_tab[1][sf as usize];
                            for j in 0..len {
                                let tmp0 = sb0[t0o + j];
                                sb0[t0o + j] = mullx(tmp0, v1);
                                sb1[t1o + j] = mullx(tmp0, v2);
                            }
                            did_istereo = true;
                        }
                    }
                }
                if !did_istereo {
                    // found1: lower part of the spectrum : do ms stereo
                    // if enabled
                    if mode_ext & MODE_EXT_MS_STEREO != 0 {
                        for j in 0..len {
                            let tmp0 = sb0[t0o + j];
                            let tmp1 = sb1[t1o + j];
                            sb0[t0o + j] = mullx(tmp0 + tmp1, ISQRT2);
                            sb1[t1o + j] = mullx(tmp0 - tmp1, ISQRT2);
                        }
                    }
                }
            }
        }

        let mut non_zero_found =
            non_zero_found_short[0] | non_zero_found_short[1] | non_zero_found_short[2];

        for i in (0..g1.long_end).rev() {
            let len = FF_BAND_SIZE_LONG[sri][i as usize] as usize;
            t0o -= len;
            t1o -= len;
            let mut did_istereo = false;
            // test if non zero band. if so, stop doing i-stereo
            if !non_zero_found {
                let mut found = false;
                for j in 0..len {
                    if sb1[t1o + j] != 0.0 {
                        non_zero_found = true;
                        found = true;
                        break;
                    }
                }
                if !found {
                    // for last band, use previous scale factor
                    let k = if i == 21 { 20 } else { i };
                    let sf = g1.scale_factors[k as usize] as i32;
                    if sf < sf_max {
                        let v1 = is_tab[0][sf as usize];
                        let v2 = is_tab[1][sf as usize];
                        for j in 0..len {
                            let tmp0 = sb0[t0o + j];
                            sb0[t0o + j] = mullx(tmp0, v1);
                            sb1[t1o + j] = mullx(tmp0, v2);
                        }
                        did_istereo = true;
                    }
                }
            }
            if !did_istereo {
                // found2
                if mode_ext & MODE_EXT_MS_STEREO != 0 {
                    for j in 0..len {
                        let tmp0 = sb0[t0o + j];
                        let tmp1 = sb1[t1o + j];
                        sb0[t0o + j] = mullx(tmp0 + tmp1, ISQRT2);
                        sb1[t1o + j] = mullx(tmp0 - tmp1, ISQRT2);
                    }
                }
            }
        }
    } else if mode_ext & MODE_EXT_MS_STEREO != 0 {
        // ms stereo ONLY. NOTE: the 1/sqrt(2) normalization factor is
        // included in the global gain (butterflies_float_c semantics).
        for i in 0..576 {
            let tmp0 = sb0[i];
            let tmp1 = sb1[i];
            sb0[i] = tmp0 + tmp1;
            sb1[i] = tmp0 - tmp1;
        }
    }
}

/// `compute_antialias` (`mpegaudiodec_template.c:1101-1129`) — the
/// float `AA` butterflies against `csa_table`.
fn compute_antialias(g: &mut GranuleDef) {
    // we antialias only "long" bands
    let n = if g.block_type == 2 {
        if g.switch_point == 0 {
            return;
        }
        // XXX: check this for 8000Hz case
        1
    } else {
        SBLIMIT - 1
    };

    let sb = &mut g.sb_hybrid;
    let mut ptr = 18usize;
    for _ in 0..n {
        for j in 0..8usize {
            let tmp0 = sb[ptr - 1 - j];
            let tmp1 = sb[ptr + j];
            sb[ptr - 1 - j] = tmp0 * CSA_TABLE[j][0] - tmp1 * CSA_TABLE[j][1];
            sb[ptr + j] = tmp0 * CSA_TABLE[j][1] + tmp1 * CSA_TABLE[j][0];
        }
        ptr += 18;
    }
}

/// `compute_imdct` (`mpegaudiodec_template.c:1132-1209`) — find the
/// last non-zero block, run the long IMDCTs, then the 12-point ones
/// for short bands and pure overlap for zero bands.
fn compute_imdct(
    ch: usize,
    gr: usize,
    granules: &mut [[GranuleDef; 2]; 2],
    sb_samples: &mut [[f32; 36 * SBLIMIT]; MPA_MAX_CHANNELS],
    mdct_buf: &mut [[f32; SBLIMIT * 18]; MPA_MAX_CHANNELS],
) {
    let g = &mut granules[ch][gr];

    // find last non zero block (C ORs the float bits as int32)
    let mut ptr: i64 = 576;
    let ptr1: i64 = 2 * 18;
    while ptr >= ptr1 {
        ptr -= 6;
        if (0..6).any(|i| g.sb_hybrid[(ptr + i) as usize].to_bits() != 0) {
            break;
        }
    }
    let sblimit = ((ptr / 18) + 1) as usize;

    let mdct_long_end = if g.block_type == 2 {
        // XXX: check for 8000 Hz
        if g.switch_point != 0 { 2 } else { 0 }
    } else {
        sblimit
    };

    imdct36_blocks(
        &mut sb_samples[ch],
        32 * 18 * gr,
        &mut mdct_buf[ch],
        0,
        &mut g.sb_hybrid[..],
        mdct_long_end,
        g.switch_point != 0,
        g.block_type,
    );

    let mut buf = 4 * 18 * (mdct_long_end >> 2) + (mdct_long_end & 3);
    let mut ptr = 18 * mdct_long_end;
    let t = tables();

    for j in mdct_long_end..sblimit {
        // select frequency inversion
        let win = &t.mdct_win[2 + (4 & (0usize.wrapping_sub(j & 1)))];
        let sb_h = &g.sb_hybrid[..];
        let sb_out = &mut sb_samples[ch];
        let mdct = &mut mdct_buf[ch];
        let mut out_ptr = j;
        let mut out2 = [0f32; 12];

        for i in 0..6usize {
            sb_out[out_ptr] = mdct[buf + 4 * i];
            out_ptr += SBLIMIT;
        }
        imdct12(&mut out2, sb_h, ptr);
        for i in 0..6usize {
            sb_out[out_ptr] = mulh3(out2[i], win[i], 1.0) + mdct[buf + 4 * (i + 6)];
            mdct[buf + 4 * (i + 6 * 2)] = mulh3(out2[i + 6], win[i + 6], 1.0);
            out_ptr += SBLIMIT;
        }
        imdct12(&mut out2, sb_h, ptr + 1);
        for i in 0..6usize {
            sb_out[out_ptr] = mulh3(out2[i], win[i], 1.0) + mdct[buf + 4 * (i + 6 * 2)];
            mdct[buf + 4 * (i + 6 * 0)] = mulh3(out2[i + 6], win[i + 6], 1.0);
            out_ptr += SBLIMIT;
        }
        imdct12(&mut out2, sb_h, ptr + 2);
        for i in 0..6usize {
            mdct[buf + 4 * (i + 6 * 0)] = mulh3(out2[i], win[i], 1.0) + mdct[buf + 4 * (i + 6 * 0)];
            mdct[buf + 4 * (i + 6 * 1)] = mulh3(out2[i + 6], win[i + 6], 1.0);
            mdct[buf + 4 * (i + 6 * 2)] = 0.0;
        }
        ptr += 18;
        buf += if (j & 3) != 3 { 1 } else { 4 * 18 - 3 };
    }
    // zero bands
    for j in sblimit..SBLIMIT {
        // overlap
        let sb_out = &mut sb_samples[ch];
        let mdct = &mut mdct_buf[ch];
        let mut out_ptr = j;
        for i in 0..18usize {
            sb_out[out_ptr] = mdct[buf + 4 * i];
            mdct[buf + 4 * i] = 0.0;
            out_ptr += SBLIMIT;
        }
        buf += if (j & 3) != 3 { 1 } else { 4 * 18 - 3 };
    }
}

/// Copy `src` into `dst`, zero-filling where C would read stale
/// bytes past the reader's region (malformed streams only).
fn copy_clamped(dst: &mut [u8], src: &[u8]) {
    let n = src.len().min(dst.len());
    dst[..n].copy_from_slice(&src[..n]);
    for b in dst[n..].iter_mut() {
        *b = 0;
    }
}
// ---------------------------------------------------------------------
// Tests — C-line-verified vectors and hand-built bitstreams; no system
// ffmpeg anywhere near these.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{codec::packet::PacketFlags, util::rational::Rational};

    // ================= header decode (mpegaudiodecheader.c) =========

    #[test]
    fn check_header_names_each_rejection() {
        // ff_mpa_check_header (mpegaudiodecheader.h:62-79)
        assert_eq!(
            ff_mpa_check_header(0xFF300000).unwrap_err(),
            Error::InvalidData("invalid frame sync".into())
        );
        assert_eq!(
            ff_mpa_check_header(0xFFE80000).unwrap_err(),
            Error::InvalidData("reserved MPEG audio version".into())
        );
        assert_eq!(
            ff_mpa_check_header(0xFFE00000).unwrap_err(),
            Error::InvalidData("invalid MPEG audio layer".into())
        );
        assert_eq!(
            ff_mpa_check_header(0xFFFBF000).unwrap_err(),
            Error::InvalidData("invalid bitrate index".into())
        );
        assert_eq!(
            ff_mpa_check_header(0xFFFB9C00).unwrap_err(),
            Error::InvalidData("invalid sample rate index".into())
        );
        // MPEG-1 layer 3 44.1 kHz stereo: the canonical 0xFFFB9000.
        assert!(ff_mpa_check_header(0xFFFB9000).is_ok());
    }

    #[test]
    fn decode_header_pins_bitfields_and_frame_sizes() {
        // MPEG-1 L3 44100 stereo 128k, no padding.
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFFB9000).unwrap());
        assert_eq!(
            (
                h.layer,
                h.sample_rate,
                h.sample_rate_index,
                h.bit_rate,
                h.frame_size
            ),
            (3, 44100, 0, 128000, 417)
        );
        assert_eq!(
            (h.nb_channels, h.mode, h.mode_ext, h.lsf, h.error_protection),
            (2, 0, 0, 0, 0)
        );

        // MPEG-1 L3 48000 mono 192k + padding: frame = 576 + 1.
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFFBB6C0).unwrap());
        assert_eq!(
            (
                h.sample_rate,
                h.sample_rate_index,
                h.nb_channels,
                h.frame_size
            ),
            (48000, 1, 1, 577)
        );
        assert_eq!(h.lsf, 0);

        // MPEG-2 LSF L3 22050 stereo 48k: rate halves, index +3, frame
        // divides by (rate << lsf). Bitrate index 6 on the LSF L3 row
        // (mpegaudiotabs.h [1][2] = 8,16,24,32,40,48,…) is 48 kbps:
        // 48·144000 / (22050·2) = 156.
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFF36000).unwrap());
        assert_eq!(
            (h.sample_rate, h.sample_rate_index, h.lsf, h.frame_size),
            (22050, 3, 1, 156)
        );

        // MPEG-2.5 L3 11025 8k: rate >> 2, index +6.
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFE31000).unwrap());
        assert_eq!(
            (h.sample_rate, h.sample_rate_index, h.lsf, h.frame_size),
            (11025, 6, 1, 52)
        );

        // Layer 1, MPEG-1 44100 64k: (br*12000)/sr then (fs+pad)*4.
        // (0xFFF72000 is MPEG-2 LSF — bit 19 = 0 — giving 22050 Hz/104;
        // MPEG-1 needs bits 20-19 = 11: 0xFFFE_.)
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFFE2000).unwrap());
        assert_eq!((h.layer, h.frame_size), (1, 68));

        // Layer 2, MPEG-1 44100 128k: (br*144000)/sr. (0xFFF59000 is
        // MPEG-2 LSF — 522; MPEG-1 needs bits 20-19 = 11: 0xFFFC_.)
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFFC8000).unwrap());
        assert_eq!((h.layer, h.frame_size), (2, 417));

        // error_protection: protection bit 0 → CRC present.
        let mut h = MpaDecodeHeader::default();
        assert!(!avpriv_mpegaudio_decode_header(&mut h, 0xFFFA9000).unwrap());
        assert_eq!(h.error_protection, 1);
    }

    #[test]
    fn free_format_frame_is_flagged() {
        // bitrate_index 0 passes check_header but signals free format
        // (mpegaudiodecheader.c:96-99 — C returns 1). Bits 15-12 = 0000:
        // 0xFFFB0000 (0xFFFB1000 has index 1 = 32 kbps, not free).
        let mut h = MpaDecodeHeader::default();
        assert!(avpriv_mpegaudio_decode_header(&mut h, 0xFFFB0000).unwrap());
        assert_eq!(h.frame_size, 0);
    }

    #[test]
    fn mpa_decode_header_maps_codec_and_samples() {
        let (mut sr, mut ch, mut fs, mut br, mut id) = (0, 0, 0, 0, CodecId::None);
        let coded =
            ff_mpa_decode_header(0xFFFB9000, &mut sr, &mut ch, &mut fs, &mut br, &mut id).unwrap();
        assert_eq!((sr, ch, br), (44100, 2, 128000));
        assert_eq!((id, fs, coded), (CodecId::Mp3, 1152, 417));

        let coded =
            ff_mpa_decode_header(0xFFF36000, &mut sr, &mut ch, &mut fs, &mut br, &mut id).unwrap();
        assert_eq!((id, fs, coded), (CodecId::Mp3, 576, 156)); // lsf → 576 samples

        let coded =
            ff_mpa_decode_header(0xFFFC8000, &mut sr, &mut ch, &mut fs, &mut br, &mut id).unwrap();
        assert_eq!((id, fs, coded), (CodecId::Mp2, 1152, 417));

        let coded =
            ff_mpa_decode_header(0xFFFE2000, &mut sr, &mut ch, &mut fs, &mut br, &mut id).unwrap();
        assert_eq!((id, fs, coded), (CodecId::Mp1, 384, 68));

        // free format is an error through this helper too (C: avpriv
        // returns 1 → ff_mpa returns -1).
        assert!(
            ff_mpa_decode_header(0xFFFB0000, &mut sr, &mut ch, &mut fs, &mut br, &mut id).is_err()
        );
    }

    // ================= static tables ================================

    #[test]
    fn bitrate_and_freq_tables_pin() {
        // ff_mpa_bitrate_tab (mpegaudiotabs.h:27-35): spot rows.
        assert_eq!(
            FF_MPA_BITRATE_TAB[0][2],
            [
                0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320
            ]
        );
        assert_eq!(FF_MPA_BITRATE_TAB[1][0][14], 256);
        assert_eq!(FF_MPA_BITRATE_TAB[1][2][1], 8);
        assert_eq!(FF_MPA_FREQ_TAB, [44100, 48000, 32000]);
    }

    #[test]
    fn slen_and_lsf_nsf_tables_pin() {
        // mpegaudiodec_common.c:52-64
        assert_eq!(
            FF_SLEN_TABLE[0],
            [0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
        );
        assert_eq!(FF_SLEN_TABLE[1][15], 3);
        assert_eq!(FF_LSF_NSF_TABLE[0][0], [6, 5, 5, 5]);
        assert_eq!(FF_LSF_NSF_TABLE[2][1], [18, 18, 0, 0]);
        assert_eq!(FF_LSF_NSF_TABLE[5][2], [6, 18, 9, 0]);
        // ff_mpa_huff_data (common.c:315-348): spot rows.
        assert_eq!(FF_MPA_HUFF_DATA[1], [1, 0]);
        assert_eq!(FF_MPA_HUFF_DATA[16], [14, 1]);
        assert_eq!(FF_MPA_HUFF_DATA[23], [14, 13]);
        assert_eq!(FF_MPA_HUFF_DATA[31], [15, 13]);
    }

    #[test]
    fn band_tables_and_index_long_pin() {
        // ff_band_size_long (common.c:362-381), rows 0 and 8.
        assert_eq!(FF_BAND_SIZE_LONG[0][0], 4);
        assert_eq!(FF_BAND_SIZE_LONG[0][21], 158);
        assert_eq!(
            FF_BAND_SIZE_LONG[8],
            [
                12, 12, 12, 12, 12, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 76, 90, 2, 2, 2, 2, 2
            ]
        );
        // ff_band_size_short row 0.
        assert_eq!(
            FF_BAND_SIZE_SHORT[0],
            [4, 4, 4, 4, 6, 8, 10, 12, 14, 18, 22, 30, 56]
        );
        // ff_band_index_long (common.c:450-457): cumulative half-sums.
        let t = tables();
        assert_eq!(
            t.band_index_long[0],
            [
                0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 31, 37, 45, 55, 67, 81, 98, 119, 144, 171,
                209, 288
            ]
        );
        // row 8 (8 kHz): 12/2 = 6 per first band.
        assert_eq!(t.band_index_long[8][1], 6);
        // bands 0-5 are 12 (half 6, sum 36); bands 6..=16 are
        // 16,20,24,28,32,40,48,56,64,76,90 → halves sum 247; total 283.
        assert_eq!(
            t.band_index_long[8][17],
            36 + 8 + 10 + 12 + 14 + 16 + 20 + 24 + 28 + 32 + 38 + 45
        );
        // ff_mpa_pretab (common.c:397-400).
        assert_eq!(FF_MPA_PRETAB[0], [0; 22]);
        assert_eq!(FF_MPA_PRETAB[1][11], 1);
        assert_eq!(FF_MPA_PRETAB[1][17], 3);
    }

    // ================= Huffman tables ===============================

    /// Slice the concatenated lens per table.
    fn table_lens(t: usize) -> Vec<u8> {
        let mut off = 0;
        for (i, &m) in MPA_HUFF_SIZES_MINUS_ONE.iter().enumerate() {
            if i == t {
                return MPA_HUFFLENS[off..off + m + 1].to_vec();
            }
            off += m + 1;
        }
        unreachable!()
    }

    fn table_syms(t: usize) -> Vec<u8> {
        let mut off = 0;
        for (i, &m) in MPA_HUFF_SIZES_MINUS_ONE.iter().enumerate() {
            if i == t {
                return MPA_HUFFSYMBOLS[off..off + m + 1].to_vec();
            }
            off += m + 1;
        }
        unreachable!()
    }

    #[test]
    fn huffman_lens_are_kraft_complete() {
        // Exact integer Kraft: sum of 2^(max-len) must equal 2^max for
        // every table — any transcription slip in the 1378 lengths
        // breaks this.
        for t in 0..15 {
            let lens = table_lens(t);
            let max = *lens.iter().max().unwrap() as u32;
            let sum: u64 = lens.iter().map(|&l| 1u64 << (max - l as u32)).sum();
            assert_eq!(sum, 1 << max, "table {t} lens not complete");
        }
    }

    #[test]
    fn huffman_symbols_are_exact_xy_pairs() {
        // Table t has m+1 entries covering (x, y), 0 <= x, y <= r-1
        // with r = sqrt(m+1) (mpa_huff_sizes_minus_one is the entry
        // count minus one) — the multiset check catches any slip.
        for (t, &m) in MPA_HUFF_SIZES_MINUS_ONE.iter().enumerate() {
            let n = m + 1;
            let r = (n as f64).sqrt().round() as usize;
            assert_eq!(r * r, n, "table {t} is not square");
            let syms = table_syms(t);
            assert_eq!(syms.len(), n, "table {t}");
            let mut sorted = syms.clone();
            sorted.sort_unstable();
            let expect: Vec<u8> = (0..r)
                .flat_map(|x| (0..r).map(move |y| ((x << 4) | y) as u8))
                .collect();
            let mut expect = expect;
            expect.sort_unstable();
            assert_eq!(sorted, expect, "table {t} symbol multiset");
        }
    }

    #[test]
    fn huff_table1_codes_follow_ffmpeg_assignment() {
        // vlc.c:319-345 assigns canonical codes in ARRAY order:
        // 0x11→'000', 0x01→'001', 0x10→'01', 0x00→'1'.
        // Decoded symbols are the packed form
        // (high << 1 | both << 4 | low) (common.c:423-427).
        let vlc = BigVlc::build(&table_lens(0), &table_syms(0));
        let decode = |bits: &[u32]| -> i32 {
            // Bits are MSB-first in the byte stream (the reader's order):
            // pack each chunk left-aligned.
            let bytes: Vec<u8> = bits
                .chunks(8)
                .map(|c| {
                    let mut byte = 0u8;
                    for (k, &b) in c.iter().enumerate() {
                        byte |= (b as u8) << (7 - k);
                    }
                    byte
                })
                .collect();
            let mut gb = GetBits::init(bytes, bits.len() as i64);
            vlc.decode(&mut gb)
        };
        assert_eq!(decode(&[0, 0, 0]), 49); // 0x11: (0x10<<1)|16|1
        assert_eq!(decode(&[0, 0, 1]), 1); //  0x01
        assert_eq!(decode(&[0, 1]), 32); //   0x10: 0x10<<1
        assert_eq!(decode(&[1]), 0); //     0x00
    }

    #[test]
    fn quad_vlc_decodes_both_tables() {
        // Table A (count1table_select 0): mpa_quad_codes[0]/bits[0]
        // (common.c:352-360) — symbol 0 has code '1' (1 bit).
        let q = &tables().huff_quad[0];
        let mut gb = GetBits::init(vec![0b1000_0000], 8);
        assert_eq!(q.decode(&mut gb, 6), 0);
        assert_eq!(gb.get_bits_count(), 1);
        // symbol 1: code 5, len 4.
        let mut gb = GetBits::init(vec![0b0101_0000], 8);
        assert_eq!(q.decode(&mut gb, 6), 1);
        assert_eq!(gb.get_bits_count(), 4);
        // Table B: 4-bit code i → symbol 15 - i.
        let q = &tables().huff_quad[1];
        let mut gb = GetBits::init(vec![0b0000_1111], 8);
        assert_eq!(q.decode(&mut gb, 4), 15);
        assert_eq!(q.decode(&mut gb, 4), 0);
    }

    // ================= generated tables =============================

    #[test]
    fn dct32_matches_spec_cosine_matrix() {
        // dct32 computes out[i] = Σ_k tab[k]·cos(π(2k+1)i/64) — the
        // ISO 11172-3 2.4.3.2 synthesis matrix V(i), i < 32 (verified
        // against the butterfly network for impulses and a ramp).
        let mut tab = [0f32; 32];
        for (i, v) in tab.iter_mut().enumerate() {
            *v = ((i * 7 + 3) % 11) as f32 - 5.0;
        }
        tab[5] = 100.5;
        let mut out = [0f32; 32];
        dct32(&mut out, &tab);
        for i in 0..32usize {
            let expect: f32 = (0..32)
                .map(|k| {
                    tab[k] as f64
                        * ((k * 2 + 1) as f64 * i as f64 * std::f64::consts::PI / 64.0).cos()
                })
                .sum::<f64>() as f32;
            assert!(
                (out[i] - expect).abs() < 1e-4 * expect.abs().max(1.0),
                "out[{i}] = {} vs {expect}",
                out[i]
            );
        }
    }

    #[test]
    fn synth_window_spot_values() {
        // mpa_synth_init (mpegaudiodsp_template.c:197-224) from
        // ff_mpa_enwindow: v = enwindow[i] / 2^39, mirrored (negated
        // when i & 63 != 0), plus the two SIMD-shuffle tails.
        let t = tables();
        let scale = (FF_MPA_ENWINDOW[64] as f64) * (1.0 / (1u64 << 39) as f64);
        assert_eq!(t.synth_window[0], 0.0);
        assert!((t.synth_window[64] as f64 - scale).abs() < 1e-12);
        assert!((t.synth_window[448] as f64 - scale).abs() < 1e-12); // 64 & 63 == 0: no flip
        let s63 = (FF_MPA_ENWINDOW[63] as f64) * (1.0 / (1u64 << 39) as f64);
        assert!((t.synth_window[63] as f64 - s63).abs() < 1e-12);
        assert!((t.synth_window[449] as f64 + s63).abs() < 1e-12); // flipped mirror
        assert!((t.synth_window[256] as f64 - 75038.0 / (1u64 << 39) as f64).abs() < 1e-12);
        // tails: window[512+16i+j] = window[64i+32-j]
        assert_eq!(t.synth_window[512], t.synth_window[32]);
        assert_eq!(
            t.synth_window[512 + 16 * 7 + 15],
            t.synth_window[64 * 7 + 32 - 15]
        );
        assert_eq!(t.synth_window[512 + 128], t.synth_window[48]);
        assert_eq!(
            t.synth_window[512 + 128 + 16 * 7 + 15],
            t.synth_window[64 * 7 + 48 - 15]
        );
        assert_eq!(t.synth_window.len(), 512 + 256);
    }

    #[test]
    fn mdct_win_spot_values() {
        // mpadsp_init_tabs (mpegaudiodsp.c:34-63): the C formula,
        // recomputed here from sin/cos.
        let t = tables();
        let build = |i: usize, j: usize| -> f64 {
            let mut d = (std::f64::consts::PI * (i as f64 + 0.5) / 36.0).sin();
            if j == 1 {
                if i >= 30 {
                    d = 0.0;
                } else if i >= 24 {
                    d = (std::f64::consts::PI * (i as f64 - 18.0 + 0.5) / 12.0).sin();
                } else if i >= 18 {
                    d = 1.0;
                }
            } else if j == 3 {
                if i < 6 {
                    d = 0.0;
                } else if i < 12 {
                    d = (std::f64::consts::PI * (i as f64 - 6.0 + 0.5) / 12.0).sin();
                } else if i < 18 {
                    d = 1.0;
                }
            }
            d * 0.5 * IMDCT_SCALAR
                / (std::f64::consts::PI * (2.0 * i as f64 + 19.0) / 72.0).cos()
                / 32.0
        };
        let near = |a: f32, b: f64| assert!((a as f64 - b).abs() < 1e-6, "{a} vs {b}");
        near(t.mdct_win[0][0], build(0, 0));
        near(t.mdct_win[0][17], build(17, 0));
        near(t.mdct_win[0][26], build(24, 0)); // i >= 18 maps to i + 20 - 18
        near(t.mdct_win[1][26], build(24, 1));
        near(t.mdct_win[1][37], build(35, 1));
        near(t.mdct_win[3][6], build(6, 3));
        near(t.mdct_win[3][13], build(13, 3));
        near(t.mdct_win[2][0], build(1, 2)); // j == 2 stored at i/3
        near(t.mdct_win[2][4], build(13, 2));
        // frequency inversion rows 4..8 (mpegaudiodsp.c:67-74).
        for i in 0..MDCT_BUF_SIZE {
            if i % 2 == 0 {
                assert_eq!(t.mdct_win[4][i], t.mdct_win[0][i]);
                assert_eq!(t.mdct_win[7][i], t.mdct_win[3][i]);
            } else {
                assert_eq!(t.mdct_win[4][i], -t.mdct_win[0][i]);
                assert_eq!(t.mdct_win[6][i], -t.mdct_win[2][i]);
            }
        }
    }

    #[test]
    fn exp_tables_and_table_4_3_spot_values() {
        let t = tables();
        // mpegaudio_tablegen.h:48-84 — exponent 0 row.
        let base = 2.11758236813575084767080625169910490512847900390625e-22f64;
        for v in 0..16usize {
            let f = (v as f64 * (v as f64).cbrt()) * base / IMDCT_SCALAR;
            assert!((t.expval_table[0][v] as f64 - f).abs() < 1e-6 * f.max(1e-30) + 1e-30);
        }
        assert_eq!(t.exp_table[0], t.expval_table[0][1]);
        // exponent 511: 127 doublings then the 3/4 LUT entry.
        let f511 = base * 2f64.powi(127) * 1.68179283050742908606 / IMDCT_SCALAR;
        assert!((t.exp_table[511] as f64 - f511).abs() < 1e-6 * f511);
        // ff_table_4_3 (mpegaudiodec_common_tablegen.h:44-69) at
        // i = 4 (value = 1): f = 1/1.759.
        let f = 1.0f64 / IMDCT_SCALAR;
        let (m, e) = frexp(f);
        assert_eq!(t.table_4_3_exp[4], (-(e + 23 - 31 + 5 - 100)) as i8);
        assert_eq!(
            t.table_4_3_value[4],
            llrint_even(m * (1u64 << 31) as f64) as u32
        );
        // l3_unscale round trip at that entry: value^(4/3)·2^0/1.759
        // ≈ 0.5685; frexp stores m ∈ [0.5, 1) and e = 103 - stored.
        let mrec = t.table_4_3_value[4] as f64 / (1u64 << 31) as f64;
        let erec = 103 - t.table_4_3_exp[4] as i32;
        assert!((mrec * 2f64.powi(erec) - f).abs() < 1e-9);
    }

    #[test]
    fn is_table_and_lsf_spots() {
        // is_table (mpegaudiodec_float.c:43-50).
        assert_eq!(IS_TABLE[0][0], 0.0);
        assert_eq!(IS_TABLE[0][6], 1.0);
        assert_eq!(IS_TABLE[1][0], 1.0);
        assert_eq!(IS_TABLE[1][6], 0.0);
        assert!((IS_TABLE[0][1] - 0.2113248705863952637).abs() < 1e-9);
        // is_table_lsf (template.c:264-278).
        let t = tables();
        assert_eq!(t.is_table_lsf[0][0][0], 1.0);
        assert_eq!(t.is_table_lsf[0][1][0], 1.0);
        assert!((t.is_table_lsf[0][0][1] - 2f64.powf(-0.25) as f32).abs() < 1e-7);
        assert_eq!(t.is_table_lsf[0][1][1], 1.0);
        assert!((t.is_table_lsf[1][1][2] - 2f64.powf(-0.5) as f32).abs() < 1e-7);
        assert_eq!(t.is_table_lsf[1][0][2], 1.0);
    }

    #[test]
    fn lsf_sf_expand_splits() {
        let mut slen = [0i32; 4];
        lsf_sf_expand(&mut slen, 399, 5, 4, 4);
        // SPLIT (template.c:659-677): 399 = 12·31 + 27? check piecewise:
        // slen[3] = 399 % 5-ish chain — verify against hand split.
        // sf=399, n3=4: d = 399 & 3 = 3, sf = 99; n2=4: d = 99 & 3 = 3,
        // sf = 24; n1=5: m = (24·205)>>10 = 4, d = 24-20 = 4, sf = 4.
        assert_eq!(slen, [4, 4, 3, 3]);
        let mut slen = [0i32; 4];
        lsf_sf_expand(&mut slen, 7, 6, 6, 0);
        assert_eq!(slen, [0, 1, 1, 0]);
    }

    // ================= bit reader ==================================

    #[test]
    fn getbits_safe_reader_semantics() {
        let mut gb = GetBits::init(vec![0b1010_1100, 0xff], 16);
        assert_eq!(gb.get_bits(4), 0b1010);
        assert_eq!(gb.get_bits1(), 1);
        assert_eq!(gb.get_bits(7), 0b1001_111); // '1','00' + four 1s
        assert_eq!(gb.get_bits(5), 0b11110); // 4 real bits + a past-end 0
        assert_eq!(gb.get_bits_count(), 16); // saturated at buffer end
        // reads past the end return 0 and the position saturates.
        let mut gb = GetBits::init(vec![0xff], 8);
        assert_eq!(gb.get_bits(4), 0xf);
        // C's reader shows the 4 real bits + zero padding to n (the
        // readable window is padded, not zeroed wholesale): 0b1111_0000.
        assert_eq!(gb.get_bits(8), 0xf0);
        assert_eq!(gb.get_bits_count(), 8); // saturated at buffer end
        // partial last byte: bits beyond size_in_bits but inside the
        // byte are still real (C's buffer_end = buffer + ceil).
        let mut gb = GetBits::init(vec![0b1000_1000], 4);
        assert_eq!(gb.get_bits(4), 0b1000);
        assert_eq!(gb.get_bits(1), 1); // real bit from the partial byte
        // skip_bits_long is the legacy signed reader.
        let mut gb = GetBits::init(vec![0xff, 0x00], 16);
        gb.skip_bits_long(11);
        assert_eq!(gb.get_bits_count(), 11);
        gb.skip_bits_long(-4);
        assert_eq!(gb.get_bits_count(), 7);
        // align_get_bits.
        let mut gb = GetBits::init(vec![0xff, 0x00], 16);
        gb.skip_bits(3);
        assert_eq!(gb.align_get_bits(), 1);
        assert_eq!(gb.get_bits_count(), 8);
    }

    // ================= end-to-end: hand-built frames ===============

    /// MSB-first bit writer for building test frames.
    struct BitWriter {
        bytes: Vec<u8>,
        nbits: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            BitWriter {
                bytes: Vec::new(),
                nbits: 0,
            }
        }
        fn put(&mut self, val: u32, n: u32) {
            for i in (0..n).rev() {
                if self.nbits % 8 == 0 {
                    self.bytes.push(0);
                }
                if (val >> i) & 1 != 0 {
                    *self.bytes.last_mut().unwrap() |= 1 << (7 - (self.nbits % 8));
                }
                self.nbits += 1;
            }
        }
        fn pad_to(&mut self, total: usize) {
            while self.bytes.len() < total {
                self.bytes.push(0);
            }
        }
    }

    /// One granule's side-info shape (MPEG-1 long blocks).
    struct GranuleSpec {
        part2_3_length: u32,
        big_values: u32,
        global_gain: u32,
        table_select0: u32,
    }

    impl Default for GranuleSpec {
        fn default() -> Self {
            GranuleSpec {
                part2_3_length: 0,
                big_values: 0,
                global_gain: 0,
                table_select0: 0,
            }
        }
    }

    /// MPEG-1 layer-3 44100 mono 32 kbps frame (104 bytes):
    /// header + 17-byte mono side info + main data.
    fn mono_frame(mdb: u32, g0: &GranuleSpec, g1: &GranuleSpec, crc: bool, w: &mut BitWriter) {
        // header 0xFFFB 10C0 (protection bit reflects crc).
        w.put(0x7FF, 11); // the 11-bit syncword (mask 0xFFE00000 >> 21)
        w.put(3, 2); // MPEG-1
        w.put(1, 2); // layer 3
        w.put(!crc as u32, 1);
        w.put(1, 4); // 32 kbps
        w.put(0, 2); // 44100
        w.put(0, 1); // no padding
        w.put(0, 1); // private
        w.put(3, 2); // mono
        w.put(0, 2); // mode_ext
        w.put(0, 1);
        w.put(0, 1);
        w.put(0, 2); // emphasis
        if crc {
            w.put(0xAAAA, 16); // unverified CRC field
        }
        // side info: mono MPEG-1 = 17 bytes.
        w.put(mdb, 9);
        w.put(0, 5); // private bits
        w.put(0, 4); // scfsi
        for g in [g0, g1] {
            w.put(g.part2_3_length, 12);
            w.put(g.big_values, 9);
            w.put(g.global_gain, 8);
            w.put(0, 4); // scalefac_compress → slen1 = slen2 = 0
            w.put(0, 1); // no window switching
            w.put(g.table_select0, 5);
            w.put(0, 5); // table_select[1]
            w.put(0, 5); // table_select[2]
            w.put(0, 4); // region_address1
            w.put(0, 3); // region_address2
            w.put(0, 1); // preflag
            w.put(0, 1); // scalefac_scale
            w.put(0, 1); // count1table_select
        }
    }

    fn silent_frame() -> Vec<u8> {
        let mut w = BitWriter::new();
        mono_frame(
            0,
            &GranuleSpec::default(),
            &GranuleSpec::default(),
            false,
            &mut w,
        );
        w.pad_to(104);
        w.bytes
    }

    fn pkt(bytes: &[u8]) -> Packet {
        let mut p = Packet::from_vec(bytes.to_vec());
        p.pts = 1234;
        p.duration = 1152;
        p.time_base = Rational::new(1, 44100);
        p.flags = PacketFlags::KEY;
        p
    }

    fn params() -> CodecParameters {
        let mut p = CodecParameters::default();
        p.codec_type = MediaType::Audio;
        p.codec_id = CodecId::Mp3;
        p.sample_rate = 44100;
        p.ch_layout = ChannelLayout::MONO;
        p.sample_fmt = SampleFormat::Fltp;
        p
    }

    #[test]
    fn debug_frame_header() {
        let mut w = BitWriter::new();
        mono_frame(
            0,
            &GranuleSpec::default(),
            &GranuleSpec::default(),
            false,
            &mut w,
        );
        println!("first bytes: {:02X?}", &w.bytes[..6]);
    }

    #[test]
    fn end_to_end_silent_frame() {
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&silent_frame()))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.format, SampleFormat::Fltp);
        assert_eq!(f.nb_samples, 1152);
        assert_eq!(f.channels(), 1);
        assert_eq!(f.planes.len(), 1);
        assert_eq!(f.plane(0).len(), 1152 * 4);
        // pure silence: every sample exactly zero.
        for b in f.plane(0) {
            assert_eq!(*b, 0);
        }
        assert_eq!(f.pts, 1234);
        assert_eq!(f.duration, 1152);
        assert_eq!(f.time_base, Rational::new(1, 44100));
        assert_eq!(f.sample_rate, 44100);
    }

    #[test]
    fn end_to_end_frame_with_huffman_line() {
        // Granule 0 carries one big-value pair from huff table 1:
        // code '10' → symbol 0x10 → x = 1, y = 0; sign bit 0 →
        // sb_hybrid[0] = +expval[300][1] = 2^3/1.759 ≈ 4.548
        // (global_gain 110 → exponent = 110 - 210 + 400 = 300).
        let g = GranuleSpec {
            part2_3_length: 3,
            big_values: 1,
            global_gain: 110,
            table_select0: 1,
        };
        let mut w = BitWriter::new();
        mono_frame(0, &g, &GranuleSpec::default(), false, &mut w);
        w.put(0b10, 2); // huff table 1 code for 0x10
        w.put(0, 1); // sign of x
        w.pad_to(104);

        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&w.bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
        let floats: Vec<f32> = f
            .plane(0)
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let nonzero = floats.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > 100,
            "expected a real signal, got {nonzero} nonzero samples"
        );
        // the very first synth block sees a nonzero spectrum
        assert!(floats[..64].iter().any(|v| *v != 0.0));
        // sanity: no absurd amplitude (values are O(1) here)
        assert!(floats.iter().all(|v| v.abs() < 100.0));

        // a second (silent) frame decodes too — state continuity,
        // reservoir carry-over included.
        dec.send_packet(Some(&pkt(&silent_frame()))).unwrap();
        let f2 = dec.receive_frame().unwrap();
        assert_eq!(f2.nb_samples, 1152);
        assert_eq!(f2.pts, 1234);
    }

    #[test]
    fn stereo_ms_frame_decodes() {
        // Stereo, mode_ext = 2 (MS only): both granules get one huff
        // pair per channel; MS butterflies mix the channels
        // (template.c:1054-1070) and global_gain is pre-reduced by 2
        // (template.c:1256-1258).
        let g = GranuleSpec {
            part2_3_length: 3,
            big_values: 1,
            global_gain: 112,
            table_select0: 1,
        };
        let mut w = BitWriter::new();
        // header: 0xFFFB 1080 — stereo, 32 kbps, 44100.
        w.put(0x7FF, 11); // the 11-bit syncword (mask 0xFFE00000 >> 21)
        w.put(3, 2);
        w.put(1, 2);
        w.put(1, 1);
        w.put(1, 4);
        w.put(0, 2);
        w.put(0, 1);
        w.put(0, 1);
        w.put(0, 2); // stereo
        w.put(2, 2); // mode_ext = MS
        w.put(0, 1);
        w.put(0, 1);
        w.put(0, 2);
        // side info: stereo MPEG-1 = 32 bytes.
        w.put(0, 9); // main_data_begin
        w.put(0, 3); // private bits
        w.put(0, 4); // scfsi ch0
        w.put(0, 4); // scfsi ch1
        for _ in 0..2 {
            for _ in 0..2 {
                w.put(g.part2_3_length, 12);
                w.put(g.big_values, 9);
                w.put(g.global_gain, 8);
                w.put(0, 4);
                w.put(0, 1);
                w.put(g.table_select0, 5);
                w.put(0, 5);
                w.put(0, 5);
                w.put(0, 4);
                w.put(0, 3);
                w.put(0, 1);
                w.put(0, 1);
                w.put(0, 1);
            }
        }
        // main data: per granule, per channel: huff '10' + sign 0.
        for _ in 0..2 {
            for _ in 0..2 {
                w.put(0b10, 2);
                w.put(0, 1);
            }
        }
        w.pad_to(104); // 32 kbps stereo frame is also 104 bytes

        let mut p = params();
        p.ch_layout = ChannelLayout::STEREO;
        let mut dec = Mp3Decoder::new();
        dec.init(&p).unwrap();
        dec.send_packet(Some(&pkt(&w.bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
        assert_eq!(f.channels(), 2);
        assert_eq!(f.planes.len(), 2);
        for ch in 0..2 {
            let floats: Vec<f32> = f
                .plane(ch)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            assert!(floats.iter().any(|v| *v != 0.0), "channel {ch} all zero");
        }
    }

    #[test]
    fn bit_reservoir_shortage_mutes_granules() {
        // main_data_begin = 100 with an empty reservoir: the granule
        // skip loop (template.c:1316-1323) mutes both granules →
        // a decodable, silent frame (error concealment shape).
        let mut w = BitWriter::new();
        mono_frame(
            100,
            &GranuleSpec::default(),
            &GranuleSpec::default(),
            false,
            &mut w,
        );
        w.pad_to(104);
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&w.bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
        assert!(f.plane(0).iter().all(|b| *b == 0));
    }

    #[test]
    fn error_protection_consumes_crc() {
        // protection bit 0 → 16 CRC bits after the header before the
        // side info (template.c:1478-1479); the CRC is not verified
        // (err_recognition off) but the offset must be consumed.
        let mut w = BitWriter::new();
        mono_frame(
            0,
            &GranuleSpec::default(),
            &GranuleSpec::default(),
            true,
            &mut w,
        );
        w.pad_to(104);
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&w.bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
        assert!(f.plane(0).iter().all(|b| *b == 0));
    }

    #[test]
    fn frame_larger_than_packet_decodes() {
        // C truncates buf_size to frame_size (template.c:1601-1604);
        // the reverse — a packet larger than the frame — also decodes
        // the first frame (trailing bytes are dropped here, see the
        // module notes).
        let mut bytes = silent_frame();
        bytes.extend_from_slice(&[0xAA; 40]);
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
    }

    #[test]
    fn leading_zero_padding_is_skipped() {
        let mut bytes = vec![0, 0, 0];
        bytes.extend_from_slice(&silent_frame());
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&bytes))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
    }

    // ================= error paths =================================

    #[test]
    fn tag_packet_is_consumed_without_a_frame() {
        let mut bytes = b"TAG".to_vec();
        bytes.extend_from_slice(&[0u8; 120]);
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&bytes))).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Again)));
    }

    #[test]
    fn short_packet_rejected() {
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        let err = dec.send_packet(Some(&pkt(&[0xFF, 0xFB]))).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidData("packet too small for an MPEG audio frame".into())
        );
    }

    #[test]
    fn bad_header_rejected() {
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        let mut bytes = vec![0xFF, 0x30, 0x00, 0x00];
        bytes.resize(60, 0);
        assert!(matches!(
            dec.send_packet(Some(&pkt(&bytes))),
            Err(Error::InvalidData(_))
        ));
    }

    #[test]
    fn free_format_packet_rejected() {
        let mut w = BitWriter::new();
        w.put(0x7FF, 11); // the 11-bit syncword (mask 0xFFE00000 >> 21)
        w.put(3, 2);
        w.put(1, 2);
        w.put(1, 1);
        w.put(0, 4); // bitrate index 0 → free format
        w.put(0, 2);
        for _ in 0..10 {
            w.put(0, 8);
        }
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        let err = dec.send_packet(Some(&pkt(&w.bytes))).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
        assert!(err.to_string().contains("free-format"));
    }

    #[test]
    fn big_values_too_large_rejected() {
        let g = GranuleSpec {
            part2_3_length: 0,
            big_values: 289,
            global_gain: 0,
            table_select0: 0,
        };
        let mut w = BitWriter::new();
        mono_frame(0, &g, &GranuleSpec::default(), false, &mut w);
        w.pad_to(104);
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        let err = dec.send_packet(Some(&pkt(&w.bytes))).unwrap_err();
        assert_eq!(err, Error::InvalidData("big_values too big".into()));
    }

    #[test]
    fn layer1_2_frames_are_unsupported() {
        // A layer-2 header inside an mp3 stream: C's mp_decode_frame
        // would decode it; the port degrades honestly.
        let mut bytes = vec![0xFF, 0xF5, 0x90, 0x00];
        bytes.resize(64, 0);
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        let err = dec.send_packet(Some(&pkt(&bytes))).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(err.to_string().contains("layer 2"));
    }

    // ================= handshake / init gates ======================

    #[test]
    fn handshake_again_then_frame_then_eof() {
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Again)));
        dec.send_packet(Some(&pkt(&silent_frame()))).unwrap();
        assert!(dec.receive_frame().is_ok());
        assert!(matches!(dec.receive_frame(), Err(Error::Again)));
        dec.send_packet(None).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
        assert!(matches!(
            dec.send_packet(Some(&pkt(&silent_frame()))),
            Err(Error::Eof)
        ));
    }

    #[test]
    fn init_gates() {
        for id in [CodecId::Mp1, CodecId::Mp2, CodecId::Rawvideo] {
            let mut p = params();
            p.codec_id = id;
            let mut dec = Mp3Decoder::new();
            let err = dec.init(&p).unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{id:?}: {err}");
        }
    }

    #[test]
    fn flush_resets_state() {
        let mut dec = Mp3Decoder::new();
        dec.init(&params()).unwrap();
        dec.send_packet(Some(&pkt(&silent_frame()))).unwrap();
        let _ = dec.receive_frame().unwrap();
        dec.flush();
        // after flush the decoder is as-new: a frame decodes again.
        dec.send_packet(Some(&pkt(&silent_frame()))).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.nb_samples, 1152);
    }
}

#[cfg(test)]
mod smoke {
    use super::*;
    use crate::codec::traits::AudioDecoder;

    /// REAL-file smoke: decode /tmp/t.mp3 frame-by-frame (a sync-word scan —
    /// the mp3 demuxer's job later) and compare against system ffmpeg's
    /// decode of the same file. Both decode the SAME encoded stream, so
    /// samples must agree closely (both are float L3 decoders; differences
    /// are LSB-level IMDCT rounding, not signal).
    #[test]
    fn decode_real_file_vs_ffmpeg() {
        let Ok(data) = std::fs::read("/tmp/t_mono.mp3") else {
            eprintln!("skip: no /tmp/t.mp3 fixture");
            return;
        };
        let Ok(ref_pcm) = std::fs::read("/tmp/ref_mono.pcm") else {
            eprintln!("skip: no /tmp/ref.pcm");
            return;
        };

        let mut params = crate::codec::params::CodecParameters::default();
        params.codec_id = CodecId::Mp3;
        let mut dec = Mp3Decoder::new();
        dec.init(&params).unwrap();

        // Interleave the planar output frames into one stereo f32 stream.
        let mut out: Vec<f32> = Vec::new();
        let mut i = 0usize;
        let mut frames = 0u32;
        while i + 4 <= data.len() && frames < 200 {
            let head = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            if ff_mpa_check_header(head).is_err() {
                i += 1;
                continue;
            }
            let mut h = MpaDecodeHeader::default();
            if avpriv_mpegaudio_decode_header(&mut h, head).unwrap() {
                i += 1; // free format: advance one byte and rescan
                continue;
            }
            let fs = h.frame_size as usize;
            if i + fs > data.len() {
                break;
            }
            let pkt = crate::codec::packet::Packet::from_vec(data[i..i + fs].to_vec());
            dec.send_packet(Some(&pkt)).unwrap();
            while let Ok(f) = dec.receive_frame() {
                let ch = f.nb_planes();
                for s in 0..f.nb_samples {
                    for c in 0..ch {
                        let b = &f.plane(c)[s * 4..s * 4 + 4];
                        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    }
                }
                frames += 1;
            }
            i += fs;
        }
        assert!(frames > 20, "decoded only {frames} frames");

        // Middle 90% sample-wise comparison against the s16 reference
        // (scaled to f32): the encoder loss is common to both decoders.
        let n_ref = ref_pcm.len() / 2; // consecutive s16 (mono fixture)
        let n = out.len().min(n_ref);
        let skip = n / 20;
        let mut max_diff = 0.0f32;
        let mut over = 0usize;
        for k in skip..n - skip {
            // MONO compare: ref is plain consecutive s16 samples.
            let r = i16::from_le_bytes([ref_pcm[2 * k], ref_pcm[2 * k + 1]]) as f32 / 32768.0;
            let d = (out[k] - r).abs();
            max_diff = max_diff.max(d);
            if d > 0.02 {
                over += 1;
            }
        }
        eprintln!("SMOKE: frames={frames} samples={n} max_diff={max_diff:.4} over=/{over}");
        let mut dump = Vec::with_capacity(out.len() * 4);
        for v in &out {
            dump.extend_from_slice(&v.to_le_bytes());
        }
        let _ = std::fs::write("/tmp/our.pcm", dump);
        // FIXME : error test
        assert!(
            over * 1000 < (n - 2 * skip),
            "{over} samples (of {}) differ by >0.02, max {max_diff:.4}",
            n - 2 * skip
        );
    }
}

#[cfg(test)]
mod dsp_probe {
    use super::*;
    /// Impulse response: one spectral line in sb_hybrid → the IMDCT+synth
    /// chain must produce a pure tone whose frequency follows the band
    /// index. Verifies the time-domain chain WITHOUT any bitstream.
    #[test]
    fn imdct_synth_impulse_is_pure_tone() {
        let mk = || {
            let mut x = GranuleDef::default();
            x.block_type = 0;
            x.long_end = 22;
            x.short_start = 13;
            x
        };
        let mut granules = [
            [mk(), GranuleDef::default()],
            [GranuleDef::default(), GranuleDef::default()],
        ];
        granules[0][0].sb_hybrid[3 * 18 + 9] = 1.0e5;
        let mut sb_samples = [[0f32; 36 * SBLIMIT]; MPA_MAX_CHANNELS];
        let mut mdct_buf = [[0f32; SBLIMIT * 18]; MPA_MAX_CHANNELS];
        compute_imdct(0, 0, &mut granules, &mut sb_samples, &mut mdct_buf);

        let t = tables();
        let mut synth_buf = vec![0f32; 1024];
        let mut off = 0usize;
        let mut dither = 0i32;
        let mut out = vec![0f32; 36 * 32];
        for i in 0..36 {
            let row = &sb_samples[0][i * SBLIMIT..(i + 1) * SBLIMIT];
            mpa_synth_filter(
                &mut synth_buf,
                &mut off,
                &t.synth_window,
                &mut dither,
                &mut out[i * 32..(i + 1) * 32],
                1,
                row,
            );
        }
        // Tone check: consecutive-sample sign-change count over the middle
        // region. Band 3 ≈ frequencies (3*18+9)/576 * 22050 ≈ 2.4 kHz →
        // ~0.11 zero-crossings/sample.
        let mut zc = 0usize;
        let mut mx = 0f32;
        for k in 64..1088 {
            if out[k - 1] <= 0.0 && out[k] > 0.0 {
                zc += 1;
            }
            mx = mx.max(out[k].abs());
        }
        eprintln!("IMPULSE: zc={zc} max={mx:.1} first16={:?}", &out[64..80]);
        // A pure tone at f gives zc ≈ f/22050 * 1024 samples. Band-3 line
        // → bin 63/576 of Nyquist → f ≈ 63/576*22050 ≈ 2411 Hz → zc ≈ 112.
        assert!(
            zc > 80 && zc < 145,
            "zero crossings {zc} not a ~2.4 kHz tone"
        );
        assert!(
            mx > 1.0 && mx < 1e7,
            "amplitude {mx} plausible for 1e5 line"
        );
        // Smoothness: no sample should jump more than a tone at Nyquist/2.
        let mut maxjump = 0f32;
        for k in 65..1088 {
            maxjump = maxjump.max((out[k] - out[k - 1]).abs());
        }
        eprintln!(
            "IMPULSE: maxjump={maxjump:.1} (mx={mx:.1}, ratio {:.3})",
            maxjump / mx
        );
        assert!(
            maxjump < mx * 0.7,
            "waveform not smooth: jump {maxjump} vs max {mx}"
        );
    }
}

#[cfg(test)]
mod dct32_c_ref {
    use super::*;
    /// Port dct32 vs the C reference vectors (compiled from
    /// FFmpeg/libavcodec/dct32_template.c float arm by /tmp/cprobe/d32).
    #[test]
    fn dct32_matches_c_reference() {
        let Ok(text) = std::fs::read_to_string("/tmp/cprobe/d32ref.txt") else {
            eprintln!("skip: no C reference vectors");
            return;
        };
        let mut checked = 0;
        for line in text.lines().filter(|l| l.starts_with('T')) {
            let t = line[1..2].parse::<u64>().unwrap() as usize;
            let vals: Vec<f32> = line[2..]
                .split_whitespace()
                .map(|v| v.parse::<f32>().unwrap())
                .collect();
            assert_eq!(vals.len(), 32);
            // same LCG as the C harness
            let mut x = (t as u64).wrapping_mul(2654435761).wrapping_add(12345);
            let mut tab = [0f32; 32];
            for v in tab.iter_mut() {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *v = ((x >> 33) as i32 as f32) / 65536.0;
            }
            let mut out = [0f32; 32];
            dct32(&mut out, &tab);
            for k in 0..32usize {
                let expect = vals[k];
                assert!(
                    (out[k] - expect).abs() < 2e-3 * (1.0 + expect.abs()),
                    "t={t} k={k}: port {:+e} vs C {expect:+e}",
                    out[k]
                );
            }
            checked += 1;
        }
        assert!(checked >= 5, "vectors loaded: {checked}");
    }
}

#[cfg(test)]
mod apply_window_c_ref {
    use super::*;
    /// Port apply_window vs C reference (/tmp/cprobe/awref.txt, compiled
    /// from mpegaudiodsp_template.c's ff_mpadsp_apply_window float arm).
    /// Same LCG-filled synth_buf and ramp window as the harness.
    #[test]
    fn apply_window_matches_c_reference() {
        let Ok(text) = std::fs::read_to_string("/tmp/cprobe/awref.txt") else {
            eprintln!("skip: no C reference vectors");
            return;
        };
        let mut synth_buf = vec![0f32; 1024];
        let mut x: u64 = 777;
        for v in synth_buf.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as i32 as f32) / 65536.0;
        }
        let window: Vec<f32> = (0..512 + 256).map(|k| k as f32 * 0.001).collect();
        let mut samples = vec![0f32; 32];
        let mut dither = 0i32;
        apply_window(&mut synth_buf, &window, &mut dither, &mut samples, 1);
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix('S') {
                let c: Vec<f32> = rest
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(c.len(), 32);
                for k in 0..32 {
                    assert!(
                        (samples[k] - c[k]).abs() < 2e-3 * (1.0 + c[k].abs()),
                        "sample {k}: port {:+e} vs C {:+e}",
                        samples[k],
                        c[k]
                    );
                }
            }
            if let Some(rest) = line.strip_prefix("D ") {
                let cd: i32 = rest.trim().parse().unwrap();
                assert_eq!(dither, cd, "dither state");
            }
        }
    }
}

#[cfg(test)]
mod synth_chain_c_ref {
    use super::*;
    /// Full synth chain (36 rows: dct32+apply_window with the REAL
    /// enwindow table and rotation) vs the C reference
    /// (/tmp/cprobe/synthref.txt). Same LCG rows.
    #[test]
    fn synth_chain_matches_c_reference() {
        let Ok(text) = std::fs::read_to_string("/tmp/cprobe/synthref.txt") else {
            eprintln!("skip: no C reference vectors");
            return;
        };
        let t = tables();
        let mut rows = vec![0f32; 36 * 32];
        let mut x: u64 = 4242;
        for v in rows.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as i32 as f32) / 65536.0;
        }
        let mut synth_buf = vec![0f32; 1024];
        let mut off = 0usize;
        let mut dither = 0i32;
        let mut out = vec![0f32; 36 * 32];
        for i in 0..36 {
            mpa_synth_filter(
                &mut synth_buf,
                &mut off,
                &t.synth_window,
                &mut dither,
                &mut out[i * 32..(i + 1) * 32],
                1,
                &rows[i * 32..(i + 1) * 32],
            );
        }
        let mut checked = 0;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix('R') {
                let c: Vec<f32> = rest[2..]
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                let r = rest[..2].parse::<usize>().unwrap();
                assert_eq!(c.len(), 32, "row {r}");
                for k in 0..32 {
                    let got = out[r * 32 + k];
                    let scale =
                        1.0 + c[k].abs() + out[0..r * 32].iter().fold(0f32, |m, v| m.max(v.abs()));
                    assert!(
                        (got - c[k]).abs() < 3e-3 * scale,
                        "row {r} k={k}: port {got:+e} vs C {:+e}",
                        c[k]
                    );
                }
                checked += 1;
            }
        }
        assert!(checked == 36, "rows checked: {checked}");
        // OFF line: final rotation offset
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("OFF ") {
                let mut it = rest.split_whitespace();
                let c_off: usize = it.next().unwrap().parse().unwrap();
                assert_eq!(off, c_off, "final synth_buf offset");
            }
        }
    }
}
