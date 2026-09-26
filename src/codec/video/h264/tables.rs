//! H.264 tables — extracted verbatim from `libavcodec/h264_cavlc.c` and
//! `libavcodec/h264data.c` (baseline/CAVLC subset; triangular VLC rows
//! are zero-padded to their C widths exactly as the C initializers see
//! them). Regenerate with the python extractor noted in h264.rs;
//! do not hand-edit — the C arrays are the spec here.
#![allow(clippy::all)]

/// `coeff_token_len[0]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_LEN_0: [u8; 68] = [
    1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8, 13,
    13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14, 16,
    15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16,
];

/// `coeff_token_len[1]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_LEN_1: [u8; 68] = [
    2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9,
    6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13, 13,
    13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14,
];

/// `coeff_token_len[2]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_LEN_2: [u8; 68] = [
    4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4,
    8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10,
];

/// `coeff_token_len[3]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_LEN_3: [u8; 68] = [
    6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6,
];

/// `coeff_token_bits[0]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_BITS_0: [u8; 68] = [
    1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5,
    4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1, 9, 12, 11,
    14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8,
];

/// `coeff_token_bits[1]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_BITS_1: [u8; 68] = [
    3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5,
    4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 12, 7,
    11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4,
];

/// `coeff_token_bits[2]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_BITS_2: [u8; 68] = [
    15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9, 14,
    13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13,
    8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2,
];

/// `coeff_token_bits[3]` (h264_cavlc.c) — flat [68].
pub static COEFF_TOKEN_BITS_3: [u8; 68] = [
    3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
];

/// `chroma_dc_coeff_token_len` (h264_cavlc.c) — flat [4*5].
pub static CHROMA_DC_TOKEN_LEN: [u8; 20] =
    [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];

/// `chroma_dc_coeff_token_bits` (h264_cavlc.c) — flat [4*5].
pub static CHROMA_DC_TOKEN_BITS: [u8; 20] =
    [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];

/// `total_zeros_len[0]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_0: [u8; 16] = [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9];

/// `total_zeros_len[1]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_1: [u8; 16] = [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 0];

/// `total_zeros_len[2]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_2: [u8; 16] = [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6, 0, 0];

/// `total_zeros_len[3]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_3: [u8; 16] = [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5, 0, 0, 0];

/// `total_zeros_len[4]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_4: [u8; 16] = [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5, 0, 0, 0, 0];

/// `total_zeros_len[5]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_5: [u8; 16] = [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6, 0, 0, 0, 0, 0];

/// `total_zeros_len[6]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_6: [u8; 16] = [6, 5, 3, 3, 3, 2, 3, 4, 3, 6, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[7]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_7: [u8; 16] = [6, 4, 5, 3, 2, 2, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[8]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_8: [u8; 16] = [6, 6, 4, 2, 2, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[9]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_9: [u8; 16] = [5, 5, 3, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[10]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_10: [u8; 16] = [4, 4, 3, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[11]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_11: [u8; 16] = [4, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[12]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_12: [u8; 16] = [3, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[13]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_13: [u8; 16] = [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[14]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_14: [u8; 16] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_len[15]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_LEN_15: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[0]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_0: [u8; 16] = [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1];

/// `total_zeros_bits[1]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_1: [u8; 16] = [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 0];

/// `total_zeros_bits[2]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_2: [u8; 16] = [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0, 0, 0];

/// `total_zeros_bits[3]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_3: [u8; 16] = [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0];

/// `total_zeros_bits[4]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_4: [u8; 16] = [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0];

/// `total_zeros_bits[5]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_5: [u8; 16] = [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[6]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_6: [u8; 16] = [1, 1, 5, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[7]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_7: [u8; 16] = [1, 1, 1, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[8]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_8: [u8; 16] = [1, 0, 1, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[9]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_9: [u8; 16] = [1, 0, 1, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[10]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_10: [u8; 16] = [0, 1, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[11]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_11: [u8; 16] = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[12]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_12: [u8; 16] = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[13]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_13: [u8; 16] = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[14]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_14: [u8; 16] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `total_zeros_bits[15]` (h264_cavlc.c) — zero-padded.
pub static TOTAL_ZEROS_BITS_15: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `chroma_dc_total_zeros_len[0]` (h264_cavlc.c).
pub static CHROMA_DC_TZ_LEN_0: [u8; 4] = [1, 2, 3, 3];

/// `chroma_dc_total_zeros_len[1]` (h264_cavlc.c).
pub static CHROMA_DC_TZ_LEN_1: [u8; 4] = [1, 2, 2, 0];

/// `chroma_dc_total_zeros_len[2]` (h264_cavlc.c).
pub static CHROMA_DC_TZ_LEN_2: [u8; 4] = [1, 1, 0, 0];

/// `chroma_dc_total_zeros_bits[0]` (h264_cavlc.c).
pub static CHROMA_DC_TZ_BITS_0: [u8; 4] = [1, 1, 1, 0];

/// `chroma_dc_total_zeros_bits[1]` (h264_cavlc.c).
pub static CHROMA_DC_TZ_BITS_1: [u8; 4] = [1, 1, 0, 0];

/// `chroma_dc_total_zeros_bits[2]` (h264_cavlc.c).
pub static CHROMA_DC_TZ_BITS_2: [u8; 4] = [1, 0, 0, 0];

/// `run_len[0]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_0: [u8; 16] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_len[1]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_1: [u8; 16] = [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_len[2]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_2: [u8; 16] = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_len[3]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_3: [u8; 16] = [2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_len[4]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_4: [u8; 16] = [2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_len[5]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_5: [u8; 16] = [2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_len[6]` (h264_cavlc.c) — zero-padded.
pub static RUN_LEN_6: [u8; 16] = [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0];

/// `run_bits[0]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_0: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_bits[1]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_1: [u8; 16] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_bits[2]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_2: [u8; 16] = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_bits[3]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_3: [u8; 16] = [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_bits[4]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_4: [u8; 16] = [3, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_bits[5]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_5: [u8; 16] = [3, 0, 1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// `run_bits[6]` (h264_cavlc.c) — zero-padded.
pub static RUN_BITS_6: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0];

/// `ff_h264_golomb_to_pict_type[5]` (h264data.c:37).
pub static GOLOMB_TO_PICT_TYPE: [u8; 5] = [2, 3, 1, 4, 5];

/// `ff_h264_i_mb_type_info` (h264data.c:66) as numeric triplets {type, cbp, pred_mode}; type codes are port-internal (0=4x4, 1=16x16, 25=PCM), 255 encodes C's -1.
pub static I_MB_TYPE_INFO: [u8; 78] = [
    0, 255, 255, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 0, 3, 1, 16, 2, 1, 16, 1, 1, 16, 0, 1, 16, 3, 1, 32,
    2, 1, 32, 1, 1, 32, 0, 1, 32, 3, 1, 15, 2, 1, 15, 1, 1, 15, 0, 1, 15, 3, 1, 31, 2, 1, 31, 1, 1,
    31, 0, 1, 31, 3, 1, 47, 2, 1, 47, 1, 1, 47, 0, 1, 47, 3, 25, 255, 255,
];

/// `ff_h264_golomb_to_intra4x4_cbp[48]` (h264data.c).
pub static GOLOMB_TO_INTRA4X4_CBP: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

/// `ff_h264_golomb_to_inter_cbp[48]` (h264data.c).
pub static GOLOMB_TO_INTER_CBP: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// `ff_h264_dequant4_coeff_init[6][3]` (h264data.c) — flat.
pub static DEQUANT4_COEFF_INIT: [u8; 18] = [
    10, 13, 16, 11, 14, 18, 13, 16, 20, 14, 18, 23, 16, 20, 25, 18, 23, 29,
];

/// `ff_h264_quant_div6[QP_MAX_NUM+1]` (h264data.c).
pub static QUANT_DIV6: [u8; 88] = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14,
    14,
];

/// `ff_h264_quant_rem6[QP_MAX_NUM+1]` (h264data.c).
pub static QUANT_REM6: [u8; 88] = [
    0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
    2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
    4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
];

/// `ff_h264_chroma_dc_scan[4]` (h264data.c:54) — resolved from its (x + y*2)*16 arithmetic.
pub static CHROMA_DC_SCAN: [u8; 4] = [0, 16, 32, 48];
