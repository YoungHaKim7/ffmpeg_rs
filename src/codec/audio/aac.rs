//! AAC decoder — port of FFmpeg's native `ff_aac_decoder` (float),
//! the **AAC-LC** subset: `libavcodec/aac/aacdec.c` (syntax),
//! `aac/aacdec_proc_template.c` (spectral dequant), and
//! `aac/aacdec_dsp_template.c` (tools + synthesis), with tables
//! extracted verbatim into [`tables`].
//!
//! ## Scope (what "LC subset" means here)
//!
//! Supported: ADTS-framed AAC-LC (`object_type == 2`), SCE/CPE/LFE
//! elements, window sequences (long/EIGHT_SHORT/transitions), Kaiser and
//! sine windows, section/scalefactor/spectral Huffman data, pulses, TNS,
//! M/S + intensity stereo, PNS (NOISE_BT), CCE dependent/independent
//! coupling, PCE (with height extension parsing), DSE/FIL skipping.
//!
//! Gated `Unsupported` (degrade honestly, like mp3's layer 1/2): AAC
//! Main (intra-frame prediction), LTP, ER syntaxes (LD/ELD/scalable),
//! HE-AAC (SBR/PS payloads are *skipped*, the LC core decodes at the base
//! rate — ffmpeg would upsample ×2; documented divergence), USAC, ADTS
//! CRC, and default channel configurations above 7.
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `aac_decode_frame_int` / `decode_frame_ga` (aacdec.c:2505/2324) | [`AacDecoder::send_packet`] → [`decode_frame_ga`] |
//! | `parse_adts_frame_header` + `ff_adts_header_parse` (adts_header.c:33) | [`parse_adts_header`] |
//! | `ff_aac_decode_ics` (aacdec.c:1785) | [`decode_ics`] |
//! | `decode_ics_info` (1418) | [`decode_ics_info`] |
//! | `decode_band_types` (1545) / `decode_scalefactors` (1592) | same names |
//! | `ff_aac_decode_tns` (1678) / `decode_pulses` (1651) | same names |
//! | `decode_spectrum_and_dequant` (aacdec_proc_template.c:57) | same name, VMUL×4 macros inlined |
//! | `decode_cpe` (1879) / `decode_cce` (proc:357) / `decode_pce` (820) | same names |
//! | `dequant_scalefactors`…`imdct_and_windowing` (aacdec_dsp_template.c) | same names |
//! | `av_tx` inverse MDCT (AV_TX_FLOAT_MDCT, scale (1/len)/32768) | [`Imdct`] — the naive O(n²) kernel of `tx_template.c:1165` with a 1-D cosine table; verified against libavutil vectors |
//! | `ff_vlc_spectral`/`ff_vlc_scalefactors` (aacdec_tab.c:749) | [`Vlc`] — flat lookup built from the explicit (code,len) arrays |
//! | `ff_kbd_window_init`/`ff_sine_window_init` (kbdwin.c, sinewin.c) | built at first use in [`tables`] |
//! | windows/pow2sf/cbrt tablegen (aactab.c:48, cbrt_tablegen.h) | same generators, run once |

use crate::{
    codec::{
        packet::Packet,
        params::{CodecId, CodecParameters, MediaType},
        traits::AudioDecoder,
    },
    util::{
        channel_layout::ChannelLayout,
        error::{Error, Result},
        samplefmt::SampleFormat,
    },
};

mod tables;

// ---------------------------------------------------------------------
// Constants (aac.h)
// ---------------------------------------------------------------------

/// `enum RawDataBlockType` — the id channel elements are addressed by.
mod ty {
    pub const SCE: u8 = 0;
    pub const CPE: u8 = 1;
    pub const CCE: u8 = 2;
    pub const LFE: u8 = 3;
    pub const DSE: u8 = 4;
    pub const PCE: u8 = 5;
    pub const FIL: u8 = 6;
    pub const END: u8 = 7;
}

/// `enum WindowSequence`.
mod win {
    pub const ONLY_LONG: u8 = 0;
    pub const LONG_START: u8 = 1;
    pub const EIGHT_SHORT: u8 = 2;
    pub const LONG_STOP: u8 = 3;
}

/// `enum BandType` values (band types 1..=11 are spectral codebooks).
mod bt {
    pub const ZERO: u8 = 0;
    pub const ESC: u8 = 11;
    pub const RESERVED: u8 = 12;
    pub const NOISE: u8 = 13;
    pub const INTENSITY2: u8 = 14;
    pub const INTENSITY: u8 = 15;
}

/// `enum ChannelPosition` (`AAC_CHANNEL_*`).
mod pos {
    pub const OFF: u8 = 0;
    pub const FRONT: u8 = 1;
    pub const SIDE: u8 = 2;
    pub const BACK: u8 = 3;
    pub const LFE: u8 = 4;
    pub const CC: u8 = 5;
    pub const LFE2: u8 = 6;
}

/// `SCALE_DIFF_ZERO` (aac.h:80): scalefactor VLC zero-difference code.
const SCALE_DIFF_ZERO: i32 = 60;
/// `NOISE_PRE` / `NOISE_PRE_BITS` / `NOISE_OFFSET` (aac.h:84-86).
const NOISE_PRE: i32 = 256;
const NOISE_PRE_BITS: u32 = 9;
const NOISE_OFFSET: i32 = 90;
/// `POW_SF2_ZERO` (aac.h:97).
const POW_SF2_ZERO: i32 = 200;
/// `TNS_MAX_ORDER` (aac.h:36).
const TNS_MAX_ORDER: usize = 20;
/// `MAX_ELEM_ID` (aac.h:34) — bitstream element ids are 4 bits, and
/// `ff_aac_channel_layout_map` has ≤ 16 rows, so 16 slots suffice.
const MAX_ELEM_ID: usize = 16;

/// `ff_mpeg4audio_sample_rates` (mpeg4audio_sample_rates.h:30).
const M4_SAMPLE_RATES: [i32; 16] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350, 0,
    0, 0,
];

/// `ff_mpeg4audio_channels` (mpeg4audio.c) — chan_config per count.
const M4_CHANNELS: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 8, 0, 0, 0, 8, 0, 16, 0];

/// MPEG-4 Audio Object Types we care about (mpeg4audio.h).
mod aot {
    pub const MAIN: u8 = 1;
    pub const LC: u8 = 2;
    pub const LTP: u8 = 4;
    pub const ER_AAC_LC: u8 = 17;
    pub const ER_AAC_LTP: u8 = 19;
    pub const ER_AAC_LD: u8 = 23;
    pub const ER_AAC_ELD: u8 = 39;
    pub const USAC: u8 = 42;
}

/// `cce_scale[]` (aacdec_float.c:66): coupling gain per 2-bit code.
const CCE_SCALE: [f32; 4] = [
    1.090_507_7, // 2^(1/8)
    1.189_207_1, // 2^(1/4)
    std::f32::consts::SQRT_2,
    2.0,
];

// ---------------------------------------------------------------------
// Structures (aacdec.h)
// ---------------------------------------------------------------------

/// `MPEG4AudioConfig` subset the decoder keeps.
#[derive(Clone, Default)]
struct M4ac {
    object_type: u8,
    sampling_index: i32,
    sample_rate: i32,
    chan_config: u8,
    channels: u8,
}

/// `enum OCStatus`.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum OcStatus {
    #[default]
    None,
    TrialPce,
    TrialFrame,
    GlobalHdr,
    Locked,
}

/// `OutputConfiguration` (USAC half dropped).
#[derive(Clone, Default)]
struct Oc {
    m4ac: M4ac,
    /// `layout_map` rows: [type, id, position].
    layout_map: Vec<[u8; 3]>,
    layout: Option<ChannelLayout>,
    status: OcStatus,
}

/// `IndividualChannelStream` (aacdec.h:169) — LTP/predictor fields kept
/// minimal: LC never sets them (prediction is rejected, LTP gated).
#[derive(Clone, Default)]
struct Ics {
    max_sfb: u8,
    /// Current and previous frame's window sequence / window shape.
    window_sequence: [u8; 2],
    use_kb_window: [bool; 2],
    num_window_groups: usize,
    prev_num_window_groups: usize,
    group_len: [usize; 8],
    num_swb: usize,
    num_windows: usize,
    tns_max_bands: usize,
    predictor_present: bool,
}

impl Ics {
    /// The `swb_offset` table pointer for the current window/rate.
    fn swb_offset(&self, sampling_index: usize) -> &'static [u16] {
        if self.window_sequence[0] == win::EIGHT_SHORT {
            tables::SWB_OFFSET_128[sampling_index]
        } else {
            tables::SWB_OFFSET_1024[sampling_index]
        }
    }
}

/// `TemporalNoiseShaping` (aacdec.h:191).
#[derive(Clone, Default)]
struct Tns {
    present: bool,
    n_filt: [usize; 8],
    length: [[usize; 4]; 8],
    direction: [bool; 4 * 8],
    order: [[usize; 4]; 8],
    coef: [[[f32; TNS_MAX_ORDER]; 4]; 8],
}

/// `Pulse` (aac.h:102).
#[derive(Clone, Copy, Default)]
struct Pulse {
    num_pulse: usize,
    pos: [usize; 4],
    amp: [i32; 4],
}

/// `SingleChannelElement` (aacdec.h:217) — predictor/LTP state dropped.
struct Sce {
    ics: Ics,
    tns: Tns,
    /// `band_type` — a `BandType` per (group, sfb).
    band_type: [u8; 128],
    /// `sfo` — raw scalefactor offsets.
    sfo: [i32; 128],
    /// `sf` — dequantized scalefactors.
    sf: [f32; 128],
    /// `coeffs` — dequantized spectrum.
    coeffs: Box<[f32; 1024]>,
    /// `saved` — IMDCT overlap buffer (512 used + short-sequence tail).
    saved: Box<[f32; 1024]>,
    /// `ret_buf` — the 1024 output samples (frame plane target).
    ret_buf: Box<[f32; 1024]>,
}

impl Sce {
    fn new() -> Self {
        Sce {
            ics: Ics::default(),
            tns: Tns::default(),
            band_type: [0; 128],
            sfo: [0; 128],
            sf: [0.0; 128],
            coeffs: Box::new([0.0; 1024]),
            saved: Box::new([0.0; 1024]),
            ret_buf: Box::new([0.0; 1024]),
        }
    }
}

/// `ChannelCoupling` (aacdec.h:203).
#[derive(Clone, Default)]
struct Coup {
    coupling_point: u8,
    num_coupled: usize,
    type_: [u8; 8],
    id_select: [usize; 8],
    ch_select: [u8; 8],
    gain: [[f32; 120]; 16],
}

/// `enum CouplingPoint` (`AFTER_IMDCT == 3`).
const BEFORE_TNS: u8 = 0;
const BETWEEN_TNS_AND_IMDCT: u8 = 1;
const AFTER_IMDCT: u8 = 3;

/// `ChannelElement` (aacdec.h:296).
struct Che {
    present: bool,
    max_sfb_ste: u8,
    ms_mask: [bool; 128],
    ch: [Sce; 2],
    coup: Coup,
}

impl Che {
    fn new() -> Self {
        Che {
            present: false,
            max_sfb_ste: 0,
            ms_mask: [false; 128],
            ch: [Sce::new(), Sce::new()],
            coup: Coup::default(),
        }
    }
}

/// The decoder state (`AACDecContext` subset): channel-element store,
/// output configuration pair (`oc[0]` is the pushed/backup copy), and the
/// per-channel output order.
struct AacContext {
    che: [Vec<Option<Che>>; 4],
    /// `tag_che_map`: element id → allocated slot per type.
    tag_che_map: [[(u8, u8); MAX_ELEM_ID]; 4],
    oc: [Oc; 2],
    tags_mapped: usize,
    random_state: u32,
    /// `output_element`: output channels in order, as (type, slot).
    output_elements: Vec<(u8, usize)>,
}

impl AacContext {
    fn new() -> Self {
        AacContext {
            che: std::array::from_fn(|_| (0..MAX_ELEM_ID).map(|_| None).collect()),
            tag_che_map: [[(u8::MAX, u8::MAX); MAX_ELEM_ID]; 4],
            oc: [Oc::default(), Oc::default()],
            tags_mapped: 0,
            random_state: 0x1f2e_3d4c,
            output_elements: Vec::new(),
        }
    }

    fn che_mut(&mut self, ty: u8, id: usize) -> Option<&mut Che> {
        self.che[ty as usize][id].as_mut()
    }

    /// The active output layout (avctx ch_layout analog).
    fn layout(&self) -> ChannelLayout {
        self.oc[1]
            .layout
            .clone()
            .unwrap_or_else(|| ChannelLayout::unspecified(self.output_elements.len() as u8))
    }
}

// ---------------------------------------------------------------------
// Bit reader — get_bits.h (safe reader: past-end reads 0)
// ---------------------------------------------------------------------

struct Gb<'a> {
    buf: &'a [u8],
    index: usize,
    size_in_bits: usize,
}

impl<'a> Gb<'a> {
    /// `init_get_bits8`.
    fn new(buf: &'a [u8]) -> Self {
        Gb {
            buf,
            index: 0,
            size_in_bits: buf.len() * 8,
        }
    }

    /// `get_bits_left`.
    fn left(&self) -> i64 {
        self.size_in_bits as i64 - self.index as i64
    }

    /// `get_bits_count`.
    fn count(&self) -> usize {
        self.index
    }

    /// `show_bits(n)` — the next n (≤ 32) bits, MSB-first; 0 past the end.
    fn peek(&self, n: u32) -> u32 {
        if n == 0 || self.index >= self.size_in_bits {
            return 0;
        }
        let mut w = 0u64;
        for k in 0..8 {
            let b = self.buf.get((self.index >> 3) + k).copied().unwrap_or(0);
            w = (w << 8) | b as u64;
        }
        let sh = 64 - (self.index & 7) as u32 - n;
        ((w >> sh) & ((1u64 << n) - 1)) as u32
    }

    /// The next n bits LEFT-aligned in a u32 (C's `GET_CACHE` /
    /// `SHOW_UBITS(v) << (32-v)` shapes the sign/escape paths rely on).
    fn peek_left(&self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.peek(n) << (32 - n)
        }
    }

    /// `skip_bits` (index saturates; `left()` never lies below 0 the way
    /// C's overread checks need).
    fn skip(&mut self, n: u32) {
        self.index = (self.index + n as usize).min(self.size_in_bits);
    }

    /// `skip_bits_long`.
    fn skip_long(&mut self, n: i64) {
        let idx = self.index as i64 + n;
        self.index = idx.clamp(0, self.size_in_bits as i64) as usize;
    }

    /// `get_bits(n)`.
    fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.skip(n);
        v
    }

    /// `get_bits1`.
    fn read_bit(&mut self) -> bool {
        self.read(1) != 0
    }

    /// `align_get_bits`.
    fn align(&mut self) {
        self.index = (self.index + 7) & !7;
    }
}

// ---------------------------------------------------------------------
// VLC — vlc.c's explicit-(code,len) build, as a flat table
// ---------------------------------------------------------------------

/// A Huffman decoder over explicit `(code, length)` pairs with optional
/// symbols — `ff_vlc_init_sparse` semantics (aacdec_tab.c:756 builds
/// `ff_vlc_spectral`/`ff_vlc_scalefactors` from exactly these arrays).
/// Entry `len<<16 | sym`; a 0 length marks an invalid code.
struct Vlc {
    max_len: u32,
    tab: Vec<u32>,
}

impl Vlc {
    fn build(codes: &[u16], lens: &[u8], syms: Option<&[u16]>) -> Vlc {
        let max_len = lens.iter().copied().max().unwrap_or(0) as u32;
        let mut tab = vec![0u32; 1usize << max_len];
        for i in 0..codes.len() {
            let l = lens[i] as u32;
            if l == 0 {
                continue;
            }
            let sym = syms.map(|s| s[i] as u32).unwrap_or(i as u32);
            let lo = (codes[i] as usize) << (max_len - l);
            for t in &mut tab[lo..lo + (1usize << (max_len - l))] {
                *t = (l << 16) | (sym & 0xffff);
            }
        }
        Vlc { max_len, tab }
    }

    /// `get_vlc2` for a complete table: peeks `max_len` bits, consumes
    /// the code, returns the symbol.
    fn get(&self, gb: &mut Gb) -> Result<u16> {
        let e = self.tab[(gb.peek(self.max_len) as usize).min(self.tab.len() - 1)];
        let len = e >> 16;
        if len == 0 {
            return Err(Error::InvalidData("invalid huffman code".into()));
        }
        gb.skip(len);
        Ok((e & 0xffff) as u16)
    }
}

// ---------------------------------------------------------------------
// Runtime tables — the tablegen half of aactab.c / windows / MDCT
// ---------------------------------------------------------------------

/// The inverse MDCT as `av_tx` presents it to the AAC decoder
/// (`AV_TX_FLOAT_MDCT`, inverse, scale `(1.0/len)/32768`): len spectral
/// coefficients in, 2·len samples out. The kernel is `ff_tx_mdct_naive_inv`
/// (tx_template.c:1165) — the O(n²) cosine sums, accumulated in f64 —
/// with the 2-D cos() calls folded into a 1-D `cos(k·π/(4·len2))` table
/// indexed by `(2j+1)·a mod 16·len` (a power of two for AAC's lengths,
/// so a mask replaces the modulo). The FFT-based codelets in libavutil
/// compute the same transform faster; the vector test pins the values.
struct Imdct {
    len: usize,
    mask: usize,
    cos: Vec<f64>,
    scale: f64,
}

impl Imdct {
    fn new(len: usize) -> Imdct {
        let l2 = 2 * len;
        let cos: Vec<f64> = (0..8 * l2).map(|k| {
            (k as f64) * (std::f64::consts::PI / (4.0 * l2 as f64))
        })
        .map(|x| x.cos())
        .collect();
        // MDCT_INIT (aacdec.c:1275): scale_float = (1.0/len) / 32768.0.
        let scale = (1.0 / len as f64) / 32768.0;
        Imdct {
            len,
            mask: 8 * l2 - 1,
            cos,
            scale,
        }
    }

    /// `ff_tx_mdct_naive_inv` (tx_template.c:1165): `dst[i] = Σ_j X[j]·
    /// cos((2j+1)·i_d)`, `dst[i+len] = −Σ_j X[j]·cos((2j+1)·i_u)` with
    /// `i_d = π/(4·len2)·(4·len−2i−1)`, `i_u = π/(4·len2)·(3·len2+2i+1)`.
    fn run(&self, coeffs: &[f32], out: &mut [f32]) {
        let l = self.len;
        let l2 = 2 * l;
        let mask = self.mask;
        let cos = &self.cos;
        let scale = self.scale;
        for i in 0..l {
            let a_d = 4 * l - 2 * i - 1;
            let a_u = 3 * l2 + 2 * i + 1;
            let mut jd = a_d;
            let mut ju = a_u;
            let mut sum_d = 0.0f64;
            let mut sum_u = 0.0f64;
            for j in 0..l2 {
                sum_d += cos[jd & mask] * coeffs[j] as f64;
                sum_u += cos[ju & mask] * coeffs[j] as f64;
                jd += 2 * a_d;
                ju += 2 * a_u;
            }
            out[i] = (sum_d * scale) as f32;
            out[i + l] = (-sum_u * scale) as f32;
        }
    }
}

/// `av_bessel_i0` (libavutil/mathematics.c:257) — minimax rational
/// approximations (Blair & Edwards, AECL-4928); needed by the KBD window.
fn bessel_i0(x: f64) -> f64 {
    const P1: [f64; 15] = [
        -2.2335582639474375249e15,
        -5.5050369673018427753e14,
        -3.2940087627407749166e13,
        -8.4925101247114157499e11,
        -1.1912746104985237192e10,
        -1.0313066708737980747e8,
        -5.9545626019847898221e5,
        -2.4125195876041896775e3,
        -7.0935347449210549190,
        -1.5453977791786851041e-2,
        -2.5172644670688975051e-5,
        -3.0517226450451067446e-8,
        -2.6843448573468483278e-11,
        -1.5982226675653184646e-14,
        -5.2487866627945699800e-18,
    ];
    const Q1: [f64; 6] = [
        -2.2335582639474375245e15,
        7.8858692566751002988e12,
        -1.2207067397808979846e10,
        1.0377081058062166144e7,
        -4.8527560179962773045e3,
        1.0,
    ];
    const P2: [f64; 7] = [
        -2.2210262233306573296e-4,
        1.3067392038106924055e-2,
        -4.4700805721174453923e-1,
        5.5674518371240761397,
        -2.3517945679239481621e1,
        3.1611322818701131207e1,
        -9.6090021968656180000,
    ];
    const Q2: [f64; 8] = [
        -5.5194330231005480228e-4,
        3.2547697594819615062e-2,
        -1.1151759188741312645,
        1.3982595353892851542e1,
        -6.0228002066743340583e1,
        8.5539563258012929600e1,
        -3.1446690275135491500e1,
        1.0,
    ];
    fn eval(poly: &[f64], x: f64) -> f64 {
        poly.iter().rev().fold(0.0, |acc, c| acc * x + c)
    }
    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval(&P1, y) / eval(&Q1, y)
    } else {
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval(&P2, y) / eval(&Q2, y);
        (x.exp() / x.sqrt()) * r
    }
}

/// `kbd_window_init` (kbdwin.c:34).
fn kbd_window_init(alpha: f32, n: usize) -> Vec<f32> {
    let mut win = vec![0.0f32; n];
    let mut temp = vec![0.0f64; n / 2 + 1];
    let alpha2 = 4.0 * ((alpha as f64) * std::f64::consts::PI / n as f64).powi(2);
    let mut scale = 0.0f64;
    for i in 0..=n / 2 {
        let tmp = (i * (n - i)) as f64 * alpha2;
        temp[i] = bessel_i0(tmp.sqrt());
        scale += temp[i] * (1.0 + ((i != 0 && i < n / 2) as u8 as f64));
    }
    scale = 1.0 / (scale + 1.0);
    let mut sum = 0.0f64;
    for i in 0..=n / 2 {
        sum += temp[i];
        win[i] = (sum * scale).sqrt() as f32;
    }
    for i in n / 2 + 1..n {
        sum += temp[n - i];
        win[i] = (sum * scale).sqrt() as f32;
    }
    win
}

/// `ff_sine_window_init` (sinewin.c).
fn sine_window_init(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 0.5) * (std::f32::consts::PI / (2.0 * n as f32))).sin())
        .collect()
}

/// Every table the decoder builds at first use (C's static inits).
struct Tables {
    /// `ff_aac_pow2sf_tab` (aactab.c:47-95): pow(2, (i−POW_SF2_ZERO)/4),
    /// computed with C's exact exp2_lut ladder.
    pow2sf: Vec<f32>,
    /// `ff_cbrt_tab` (cbrt_tablegen.h): IEEE bits of i^(4/3) — the
    /// escape-value dequant table OR-ed with sign bits.
    cbrt: Vec<u32>,
    sine_1024: Vec<f32>,
    sine_128: Vec<f32>,
    kbd_1024: Vec<f32>,
    kbd_128: Vec<f32>,
    vlc_scalefactors: Vlc,
    vlc_spectral: [Vlc; 11],
    mdct_1024: Imdct,
    mdct_128: Imdct,
}

fn tables() -> &'static Tables {
    use std::sync::OnceLock;
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        // aac_tableinit (aactab.c:48-95) — the ladder, not powf.
        let mut pow2sf = vec![0.0f32; 428];
        let mut t1: f32 = 8.881_784_2e-16; // 2^-50
        let mut t2: f32 = 3.637_978_8e-12; // (pow2sf^0.75 base, unused here)
        let (mut t1_prev, mut t2_prev) = (0usize, 8usize);
        for i in 0..428 {
            let t1_cur = 4 * (i % 4);
            let t2_cur = (8 + 3 * i) % 16;
            if t1_cur < t1_prev {
                t1 *= 2.0;
            }
            if t2_cur < t2_prev {
                t2 *= 2.0;
            }
            pow2sf[i] = t1 * tables::EXP2_LUT[t1_cur];
            t1_prev = t1_cur;
            t2_prev = t2_cur;
        }

        // ff_cbrt_tableinit (cbrt_tablegen.h): ff_cbrt_tab[i] holds the
        // float bits of i^(4/3) (the 4/3 dequant power, evaluated in
        // double through the odd-root LUT in C).
        let cbrt: Vec<u32> = (0..8193u32)
            .map(|i| ((i as f64).powf(4.0 / 3.0) as f32).to_bits())
            .collect();

        let vlc_scalefactors = Vlc::build(
            &tables::AAC_SCALEFACTOR_CODE,
            &tables::AAC_SCALEFACTOR_BITS,
            None,
        );
        let mut vlc_spectral: [Vlc; 11] = std::array::from_fn(|i| {
            Vlc::build(
                tables::SPECTRAL_CODES[i],
                tables::SPECTRAL_BITS[i],
                Some(tables::CODEBOOK_VECTOR_IDX[i]),
            )
        });

        // Silence the "unused" on the array-init trick above.
        vlc_spectral.iter_mut().for_each(drop);

        Tables {
            pow2sf,
            cbrt,
            sine_1024: sine_window_init(1024),
            sine_128: sine_window_init(128),
            kbd_1024: kbd_window_init(4.0, 1024),
            kbd_128: kbd_window_init(6.0, 128),
            vlc_scalefactors,
            vlc_spectral,
            mdct_1024: Imdct::new(1024),
            mdct_128: Imdct::new(128),
        }
    })
}

// ---------------------------------------------------------------------
// Channel configuration (aacdec.c:118-776)
// ---------------------------------------------------------------------

/// `count_channels` (aacdec.c:118).
fn count_channels(layout: &[[u8; 3]]) -> usize {
    layout
        .iter()
        .map(|row| {
            (1 + (row[0] == ty::CPE) as usize)
                * (row[2] != pos::OFF && row[2] != pos::CC) as usize
        })
        .sum()
}

/// `che_configure` (aacdec.c:142): allocate the element and append its
/// channels to the output order.
fn che_configure(ctx: &mut AacContext, position: u8, ty: u8, id: usize, channels: &mut usize) -> Result<()> {
    if position != pos::OFF {
        if ctx.che[ty as usize][id].is_none() {
            ctx.che[ty as usize][id] = Some(Che::new());
        }
        if ty != ty::CCE {
            if *channels >= 64 - (ty == ty::CPE) as usize {
                return Err(Error::InvalidData("too many channels".into()));
            }
            ctx.output_elements.push((ty, id));
            *channels += 1;
            if ty == ty::CPE {
                ctx.output_elements.push((ty, id));
                *channels += 1;
            }
        }
    } else if ctx.che[ty as usize][id].is_some() {
        ctx.che[ty as usize][id] = None;
        ctx.output_elements.clear();
        ctx.tag_che_map = [[(u8::MAX, u8::MAX); MAX_ELEM_ID]; 4];
    }
    Ok(())
}

/// `elem_to_channel` (aacdec.c:215).
#[derive(Clone, Copy, Default)]
struct ElemToChannel {
    av_position: u64,
    syn_ele: u8,
    elem_id: u8,
    aac_position: u8,
}

/// `assign_pair` (aacdec.c:222).
fn assign_pair(
    e2c: &mut [ElemToChannel],
    layout_map: &[[u8; 3]],
    offset: usize,
    left: u64,
    right: u64,
    p: u8,
    layout: &mut u64,
) -> usize {
    if layout_map[offset][0] == ty::CPE {
        e2c[offset] = ElemToChannel {
            av_position: left | right,
            syn_ele: ty::CPE,
            elem_id: layout_map[offset][1],
            aac_position: p,
        };
        if e2c[offset].av_position != u64::MAX {
            *layout |= e2c[offset].av_position;
        }
        1
    } else {
        e2c[offset] = ElemToChannel {
            av_position: left,
            syn_ele: ty::SCE,
            elem_id: layout_map[offset][1],
            aac_position: p,
        };
        e2c[offset + 1] = ElemToChannel {
            av_position: right,
            syn_ele: ty::SCE,
            elem_id: layout_map[offset + 1][1],
            aac_position: p,
        };
        if left != u64::MAX {
            *layout |= left;
        }
        if right != u64::MAX {
            *layout |= right;
        }
        2
    }
}

/// `count_paired_channels` (aacdec.c:260).
fn count_paired_channels(layout_map: &[[u8; 3]], p: u8, current: usize) -> i32 {
    let mut num_pos_channels = 0;
    let mut first_cpe = false;
    let mut sce_parity = false;
    for row in &layout_map[current..] {
        if row[2] != p {
            break;
        }
        if row[0] == ty::CPE {
            if sce_parity {
                if p == pos::FRONT && !first_cpe {
                    sce_parity = false;
                } else {
                    return -1;
                }
            }
            num_pos_channels += 2;
            first_cpe = true;
        } else {
            num_pos_channels += 1;
            sce_parity ^= p != pos::LFE;
        }
    }
    if sce_parity && (p == pos::FRONT && first_cpe) {
        return -1;
    }
    num_pos_channels
}

/// `assign_channels` (aacdec.c:292): hand out AV channel positions from
/// `ff_aac_channel_map[layer][pos-1]` in C's order.
fn assign_channels(
    e2c: &mut [ElemToChannel],
    layout_map: &[[u8; 3]],
    layout: &mut u64,
    layer: usize,
    p: u8,
    current: &mut usize,
) -> i32 {
    let mut i = *current;
    let mut j = 0usize;
    let mut nb = count_paired_channels(layout_map, p, i);
    if nb < 0 || nb > 5 {
        return 0;
    }
    let map = &tables::AAC_CHANNEL_MAP[layer][(p - 1) as usize];

    if p == pos::LFE {
        while nb != 0 {
            if map[j] == -1 {
                return -1;
            }
            e2c[i] = ElemToChannel {
                av_position: 1u64 << map[j],
                syn_ele: layout_map[i][0],
                elem_id: layout_map[i][1],
                aac_position: p,
            };
            *layout |= e2c[i].av_position;
            i += 1;
            j += 1;
            nb -= 1;
        }
        *current = i;
        return 0;
    }

    while nb & 1 == 1 {
        if map[0] == -1 {
            return -1;
        }
        if map[0] == -2 {
            break;
        }
        e2c[i] = ElemToChannel {
            av_position: 1u64 << map[0],
            syn_ele: layout_map[i][0],
            elem_id: layout_map[i][1],
            aac_position: p,
        };
        *layout |= e2c[i].av_position;
        i += 1;
        nb -= 1;
    }

    let mut j = if p != pos::SIDE && nb <= 3 { 3 } else { 1 };
    while nb >= 2 {
        if map[j] == -1 || map[j + 1] == -1 {
            return -1;
        }
        i += assign_pair(
            e2c,
            layout_map,
            i,
            1u64 << map[j],
            1u64 << map[j + 1],
            p,
            layout,
        );
        j += 2;
        nb -= 2;
    }
    while nb & 1 == 1 {
        if map[5] == -1 {
            return -1;
        }
        e2c[i] = ElemToChannel {
            av_position: 1u64 << map[5],
            syn_ele: layout_map[i][0],
            elem_id: layout_map[i][1],
            aac_position: p,
        };
        *layout |= e2c[i].av_position;
        i += 1;
        nb -= 1;
    }
    if nb != 0 {
        return -1;
    }
    *current = i;
    0
}

/// `sniff_channel_order` (aacdec.c:370): assign AV channel positions per
/// (layer, position), then REORDER the layout map into channel-position
/// order (the stable bubble). Returns the layout mask (0 = keep declared
/// order).
fn sniff_channel_order(layout_map: &mut [[u8; 3]]) -> u64 {
    let tags = layout_map.len();
    let mut e2c = vec![ElemToChannel::default(); 4 * MAX_ELEM_ID];
    let mut layout = 0u64;
    let mut i = 0usize;
    let mut n = 0usize;
    while n < 3 && i < tags {
        for &p in &[pos::FRONT, pos::SIDE, pos::BACK, pos::LFE] {
            if assign_channels(&mut e2c, layout_map, &mut layout, n, p, &mut i) < 0 {
                return 0;
            }
        }
        n += 1;
    }
    let total = i;
    // Everything but 22.2: stable sort by av_position (the C bubble).
    if layout == tables::AAC_CH_LAYOUT[9].1 {
        // 22.2 fixed swaps (aacdec.c:398-406) — unreachable for the
        // supported configs; keep C's sort for the rest.
    }
    let mut m = total;
    loop {
        let mut next_n = 0;
        for k in 1..m {
            if e2c[k - 1].av_position > e2c[k].av_position {
                e2c.swap(k - 1, k);
                next_n = k;
            }
        }
        m = next_n;
        if m == 0 {
            break;
        }
    }
    for k in 0..total.min(tags) {
        layout_map[k] = [e2c[k].syn_ele, e2c[k].elem_id, e2c[k].aac_position];
    }
    layout
}

/// `ff_aac_set_default_channel_config` (aacdec.c:583).
fn set_default_channel_config(chan_config: u8) -> Result<Vec<[u8; 3]>> {
    if !(1..=7).contains(&chan_config) && !(11..=14).contains(&chan_config) {
        return Err(Error::InvalidData(format!(
            "invalid default channel configuration ({chan_config})"
        )));
    }
    let tags = tables::TAGS_PER_CONFIG[chan_config as usize] as usize;
    let mut map: Vec<[u8; 3]> = tables::CHANNEL_LAYOUT_MAP[chan_config as usize][..tags].to_vec();
    // C's 7.1(wide)→7.1 leniency for config 7 (aacdec.c:610-618).
    if chan_config == 7 {
        map[2][2] = pos::BACK;
    }
    Ok(map)
}

/// `push_output_configuration` (aacdec.c:454) / `pop` (470).
fn push_output_configuration(ctx: &mut AacContext) {
    if ctx.oc[1].status == OcStatus::Locked || ctx.oc[0].status == OcStatus::None {
        ctx.oc[0] = ctx.oc[1].clone();
    }
    ctx.oc[1].status = OcStatus::None;
}

fn pop_output_configuration(ctx: &mut AacContext) {
    if ctx.oc[1].status != OcStatus::Locked && ctx.oc[0].status != OcStatus::None {
        ctx.oc[1] = ctx.oc[0].clone();
        let map = ctx.oc[1].layout_map.clone();
        let status = ctx.oc[1].status;
        let _ = output_configure(ctx, &map, status, false);
    }
}

/// `ff_aac_output_configure` (aacdec.c:487) — with `sniff_channel_order`
/// folded in: the map is reordered to output order, elements allocated,
/// and the channel layout derived from the sniffed mask.
fn output_configure(ctx: &mut AacContext, layout_map: &[[u8; 3]], oc_type: OcStatus, get_new_frame: bool) -> Result<()> {
    let _ = get_new_frame; // frame buffers are built per-packet here
    let mut map = layout_map.to_vec();
    let mut id_map = [[[0u8; MAX_ELEM_ID]; 8]; 1];
    let mut type_counts = [0u8; 8];
    for row in &map {
        let t = row[0] as usize;
        let id = row[1] as usize;
        id_map[0][t][id] = type_counts[t];
        type_counts[t] += 1;
        if type_counts[t] as usize >= MAX_ELEM_ID {
            return Err(Error::Unsupported("too large remapped id".into()));
        }
    }
    let mask = sniff_channel_order(&mut map);

    let mut channels = 0usize;
    ctx.output_elements.clear();
    for row in &map {
        let t = row[0];
        let id = row[1] as usize;
        let iid = id_map[0][t as usize][id] as usize;
        che_configure(ctx, row[2], t, iid, &mut channels)?;
        ctx.tag_che_map[t as usize][id] = (t, iid as u8);
    }
    ctx.oc[1].layout = if mask != 0 {
        ChannelLayout::from_mask(mask).ok()
    } else {
        None
    };
    if ctx.oc[1].layout.is_none() {
        ctx.oc[1].layout = Some(ChannelLayout::unspecified(channels as u8));
    }
    ctx.oc[1].layout_map = map;
    ctx.oc[1].status = oc_type;
    Ok(())
}

/// `ff_aac_get_che` (aacdec.c:623) — the chan_config ≤ 7 paths (8+ gated)
/// including the mono-with-CPE / stereo-with-SCE remaps.
fn get_che(ctx: &mut AacContext, ty: u8, elem_id: usize) -> Option<(u8, usize)> {
    if ctx.oc[1].m4ac.chan_config == 0 {
        let m = ctx.tag_che_map[ty as usize][elem_id];
        if m.0 == u8::MAX {
            return None;
        }
        return Some((m.0, m.1 as usize));
    }
    if ctx.tags_mapped == 0 && ty == ty::CPE && ctx.oc[1].m4ac.chan_config == 1 {
        push_output_configuration(ctx);
        let map = set_default_channel_config(2).ok()?;
        output_configure(ctx, &map, OcStatus::TrialFrame, false).ok()?;
        ctx.oc[1].m4ac.chan_config = 2;
    }
    if ctx.tags_mapped == 0 && ty == ty::SCE && ctx.oc[1].m4ac.chan_config == 2 {
        push_output_configuration(ctx);
        let map = vec![[ty::SCE, 0, pos::FRONT], [ty::SCE, 1, pos::FRONT]];
        output_configure(ctx, &map, OcStatus::TrialFrame, false).ok()?;
    }
    let m = ctx.tag_che_map[ty as usize][elem_id];
    if m.0 == u8::MAX {
        return None;
    }
    ctx.tags_mapped += 1;
    Some((m.0, m.1 as usize))
}

// ---------------------------------------------------------------------
// ADTS (adts_header.c:33 + aacdec.c:2189)
// ---------------------------------------------------------------------

/// `ff_adts_header_parse` + `parse_adts_frame_header`: the 56-bit ADTS
/// header; configures the output on channel-config change. Returns the
/// frame length in bytes.
fn parse_adts_header(ctx: &mut AacContext, gb: &mut Gb) -> Result<usize> {
    if gb.read(12) != 0xfff {
        return Err(Error::InvalidData("ADTS syncword missing".into()));
    }
    gb.skip(1); // id
    gb.skip(2); // layer
    let crc_absent = gb.read_bit();
    if !crc_absent {
        return Err(Error::Unsupported(
            "ADTS with CRC (protection_absent == 0) is not ported".into(),
        ));
    }
    let aot = gb.read(2) as u8 + 1;
    let sr_idx = gb.read(4) as usize;
    if M4_SAMPLE_RATES[sr_idx] == 0 {
        return Err(Error::InvalidData("invalid ADTS sample-rate index".into()));
    }
    gb.skip(1); // private
    let chan_config = gb.read(3) as u8;
    gb.skip(4); // original/copy, home, 2× copyright id
    let size = gb.read(13) as usize;
    if size < 7 {
        return Err(Error::InvalidData("ADTS frame too small".into()));
    }
    gb.skip(11); // buffer fullness
    let num_frames = gb.read(2) + 1;

    if aot != aot::LC {
        return Err(Error::Unsupported(format!(
            "MPEG-4 audio object type {aot} (AAC-LC only is ported)"
        )));
    }

    push_output_configuration(ctx);
    if chan_config != 0 {
        ctx.oc[1].m4ac.chan_config = chan_config;
        let map = set_default_channel_config(chan_config)?;
        let status = if ctx.oc[1].status as u8 > OcStatus::TrialFrame as u8 {
            ctx.oc[1].status
        } else {
            OcStatus::TrialFrame
        };
        output_configure(ctx, &map, status, false)?;
        ctx.oc[1].m4ac.channels = M4_CHANNELS[chan_config as usize];
    } else {
        ctx.oc[1].m4ac.chan_config = 0;
        ctx.oc[1].m4ac.channels = 0;
    }
    ctx.oc[1].m4ac.object_type = aot;
    ctx.oc[1].m4ac.sampling_index = sr_idx as i32;
    ctx.oc[1].m4ac.sample_rate = M4_SAMPLE_RATES[sr_idx];
    if num_frames != 1 {
        return Err(Error::Unsupported(
            "more than one raw data block per ADTS frame".into(),
        ));
    }
    Ok(size)
}

// ---------------------------------------------------------------------
// Syntax — aacdec.c
// ---------------------------------------------------------------------

/// `skip_data_stream_element` (aacdec.c:1361).
fn skip_data_stream_element(gb: &mut Gb) -> Result<()> {
    let byte_align = gb.read_bit();
    let mut count = gb.read(8) as usize;
    if count == 255 {
        count += gb.read(8) as usize;
    }
    if byte_align {
        gb.align();
    }
    if gb.left() < 8 * count as i64 {
        return Err(Error::InvalidData(
            "skip_data_stream_element: input buffer exhausted before END".into(),
        ));
    }
    gb.skip_long(8 * count as i64);
    Ok(())
}

/// `decode_ics_info` (aacdec.c:1418) — non-ER syntax; LTP/prediction
/// gated for LC.
fn decode_ics_info(ctx: &AacContext, ics: &mut Ics, gb: &mut Gb) -> Result<()> {
    let m4ac = &ctx.oc[1].m4ac;
    let aot = m4ac.object_type;
    let srate = m4ac.sampling_index.clamp(0, 12) as usize;

    if gb.read_bit() {
        return Err(Error::InvalidData("reserved ics_info bit set".into()));
    }
    ics.window_sequence[1] = ics.window_sequence[0];
    ics.window_sequence[0] = gb.read(2) as u8;
    ics.use_kb_window[1] = ics.use_kb_window[0];
    ics.use_kb_window[0] = gb.read_bit();

    ics.prev_num_window_groups = ics.num_window_groups.max(1);
    ics.num_window_groups = 1;
    ics.group_len = [0; 8];
    ics.group_len[0] = 1;
    if ics.window_sequence[0] == win::EIGHT_SHORT {
        ics.max_sfb = gb.read(4) as u8;
        for _ in 0..7 {
            if gb.read_bit() {
                ics.group_len[ics.num_window_groups - 1] += 1;
            } else {
                ics.num_window_groups += 1;
                ics.group_len[ics.num_window_groups - 1] = 1;
            }
        }
        ics.num_windows = 8;
        ics.num_swb = tables::NUM_SWB_128[srate] as usize;
        ics.tns_max_bands = tables::TNS_MAX_BANDS_128[srate] as usize;
        ics.predictor_present = false;
    } else {
        ics.max_sfb = gb.read(6) as u8;
        ics.num_windows = 1;
        ics.num_swb = tables::NUM_SWB_1024[srate] as usize;
        ics.tns_max_bands = tables::TNS_MAX_BANDS_1024[srate] as usize;
        ics.predictor_present = gb.read_bit();
        if ics.predictor_present {
            if aot == aot::MAIN {
                return Err(Error::Unsupported(
                    "AAC Main intra-frame prediction is not ported".into(),
                ));
            }
            // LC: "Prediction is not allowed in AAC-LC." — but the LTP
            // forms still parse; the port gates them.
            return Err(Error::Unsupported(
                "predictor data in AAC-LC ics_info is not ported".into(),
            ));
        }
    }
    if ics.max_sfb as usize > ics.num_swb {
        ics.max_sfb = 0;
        return Err(Error::InvalidData(format!(
            "max_sfb {} exceeds the {} scalefactor bands",
            ics.max_sfb, ics.num_swb
        )));
    }
    Ok(())
}

/// `decode_band_types` (aacdec.c:1545) — section_data.
fn decode_band_types(ics: &Ics, band_type: &mut [u8; 128], gb: &mut Gb) -> Result<()> {
    let bits = if ics.window_sequence[0] == win::EIGHT_SHORT { 3 } else { 5 };
    for g in 0..ics.num_window_groups {
        let mut k = 0usize;
        while k < ics.max_sfb as usize {
            let mut sect_end = k;
            let sect_band_type = gb.read(4) as u8;
            if sect_band_type == bt::RESERVED {
                return Err(Error::InvalidData("invalid band type 12".into()));
            }
            loop {
                let incr = gb.read(bits) as usize;
                sect_end += incr;
                if gb.left() < 0 {
                    return Err(Error::InvalidData(
                        "decode_band_types: buffer exhausted".into(),
                    ));
                }
                if sect_end > ics.max_sfb as usize {
                    return Err(Error::InvalidData(format!(
                        "section run {sect_end} exceeds max_sfb {}",
                        ics.max_sfb
                    )));
                }
                if incr != (1 << bits) - 1 {
                    break;
                }
            }
            for b in band_type[g * ics.max_sfb as usize + k..g * ics.max_sfb as usize + sect_end]
                .iter_mut()
            {
                *b = sect_band_type;
            }
            k = sect_end;
        }
    }
    Ok(())
}

/// `decode_scalefactors` (aacdec.c:1592).
fn decode_scalefactors(
    t: &Tables,
    ics: &Ics,
    band_type: &[u8; 128],
    sfo: &mut [i32; 128],
    gb: &mut Gb,
    global_gain: u32,
) -> Result<()> {
    let mut offset = [
        global_gain as i32,
        global_gain as i32 - NOISE_OFFSET,
        0i32,
    ];
    let mut noise_flag = true;
    let mut idx = 0usize;
    for g in 0..ics.num_window_groups {
        for sfb in 0..ics.max_sfb as usize {
            match band_type[g * ics.max_sfb as usize + sfb] {
                bt::ZERO => sfo[idx] = 0,
                x if x == bt::INTENSITY || x == bt::INTENSITY2 => {
                    offset[2] += t.vlc_scalefactors.get(gb)? as i32 - SCALE_DIFF_ZERO;
                    let clipped = offset[2].clamp(-155, 100);
                    sfo[idx] = clipped - 100;
                }
                bt::NOISE => {
                    if noise_flag {
                        noise_flag = false;
                        offset[1] += gb.read(NOISE_PRE_BITS) as i32 - NOISE_PRE;
                    } else {
                        offset[1] += t.vlc_scalefactors.get(gb)? as i32 - SCALE_DIFF_ZERO;
                    }
                    let clipped = offset[1].clamp(-100, 155);
                    sfo[idx] = clipped;
                }
                _ => {
                    offset[0] += t.vlc_scalefactors.get(gb)? as i32 - SCALE_DIFF_ZERO;
                    if offset[0] > 255 {
                        return Err(Error::InvalidData(format!(
                            "scalefactor {} out of range",
                            offset[0]
                        )));
                    }
                    sfo[idx] = offset[0] - 100;
                }
            }
            idx += 1;
        }
    }
    Ok(())
}

/// `decode_pulses` (aacdec.c:1651).
fn decode_pulses(pulse: &mut Pulse, gb: &mut Gb, swb_offset: &[u16], num_swb: usize) -> Result<()> {
    pulse.num_pulse = gb.read(2) as usize + 1;
    let pulse_swb = gb.read(6) as usize;
    if pulse_swb >= num_swb {
        return Err(Error::InvalidData("pulse data corrupt".into()));
    }
    pulse.pos[0] = swb_offset[pulse_swb] as usize + gb.read(5) as usize;
    if pulse.pos[0] >= swb_offset[num_swb] as usize {
        return Err(Error::InvalidData("pulse data corrupt".into()));
    }
    pulse.amp[0] = gb.read(4) as i32;
    for i in 1..pulse.num_pulse {
        pulse.pos[i] = gb.read(5) as usize + pulse.pos[i - 1];
        if pulse.pos[i] >= swb_offset[num_swb] as usize {
            return Err(Error::InvalidData("pulse data corrupt".into()));
        }
        pulse.amp[i] = gb.read(4) as i32;
    }
    Ok(())
}

/// `ff_aac_decode_tns` (aacdec.c:1678) — non-USAC branch.
fn decode_tns(ics: &Ics, tns: &mut Tns, gb: &mut Gb) -> Result<()> {
    let is8 = ics.window_sequence[0] == win::EIGHT_SHORT;
    let tns_max_order = if is8 { 7 } else { 12 };
    for w in 0..ics.num_windows {
        tns.n_filt[w] = gb.read(2 - is8 as u32) as usize;
        if tns.n_filt[w] != 0 {
            let coef_res = gb.read_bit();
            for filt in 0..tns.n_filt[w] {
                tns.length[w][filt] = gb.read(6 - 2 * is8 as u32) as usize;
                tns.order[w][filt] = gb.read(5 - 2 * is8 as u32) as usize;
                if tns.order[w][filt] > tns_max_order {
                    tns.order[w][filt] = 0;
                    return Err(Error::InvalidData(format!(
                        "TNS filter order exceeds maximum {tns_max_order}"
                    )));
                }
                if tns.order[w][filt] != 0 {
                    tns.direction[w * 4 + filt] = gb.read_bit();
                    let coef_compress = gb.read_bit();
                    let coef_len = 3 + coef_res as u32 - coef_compress as u32;
                    let tmp2_idx = 2 * coef_compress as usize + coef_res as usize;
                    for i in 0..tns.order[w][filt] {
                        tns.coef[w][filt][i] =
                            tables::TNS_TMP2_MAP[tmp2_idx][gb.read(coef_len) as usize];
                    }
                }
            }
        }
    }
    Ok(())
}

/// `decode_mid_side_stereo` (aacdec.c:1736).
fn decode_mid_side_stereo(che: &mut Che, gb: &mut Gb, ms_present: u32) {
    let max_idx = che.ch[0].ics.num_window_groups * che.ch[0].ics.max_sfb as usize;
    che.max_sfb_ste = che.ch[0].ics.max_sfb;
    if ms_present == 1 {
        for idx in 0..max_idx.min(128) {
            che.ms_mask[idx] = gb.read_bit();
        }
    } else if ms_present == 2 {
        che.ms_mask[..max_idx.min(128)].fill(true);
    }
}

/// `decode_gain_control` (aacdec.c:1750) — parsed, ignored (C warns too).
fn decode_gain_control(ics: &Ics, gb: &mut Gb) {
    const GAIN_MODE: [[u32; 3]; 4] = [[1, 0, 5], [2, 1, 2], [8, 0, 2], [2, 1, 5]];
    let mode = ics.window_sequence[0] as usize;
    let max_band = gb.read(2);
    for _ in 0..max_band {
        for wd in 0..GAIN_MODE[mode][0] {
            let adjust_num = gb.read(3);
            for _ in 0..adjust_num {
                let extra = (wd == 0 && GAIN_MODE[mode][1] != 0) as u32 * 4;
                gb.skip(4 + GAIN_MODE[mode][2] - (4 - extra.min(4)) + extra);
            }
        }
    }
}

/// `decode_channel_map` (aacdec.c:778).
fn decode_channel_map(map: &mut [[u8; 3]], p: u8, gb: &mut Gb, n: usize) {
    for row in map.iter_mut().take(n) {
        let syn_ele = match p {
            pos::FRONT | pos::BACK | pos::SIDE => {
                if gb.read_bit() {
                    ty::CPE
                } else {
                    ty::SCE
                }
            }
            pos::CC => {
                gb.skip(1);
                ty::CCE
            }
            pos::LFE => ty::LFE,
            _ => ty::SCE,
        };
        row[0] = syn_ele;
        row[1] = gb.read(4) as u8;
        row[2] = p;
    }
}

/// `decode_pce` (aacdec.c:820) — height extension parsed and applied.
/// Returns the tag count.
fn decode_pce(ctx: &mut AacContext, map: &mut Vec<[u8; 3]>, gb: &mut Gb, byte_align_ref: usize) -> Result<usize> {
    gb.skip(2); // object_type
    let sampling_index = gb.read(4) as usize;
    if ctx.oc[1].m4ac.sampling_index != sampling_index as i32 {
        // C logs a warning and carries on.
    }
    let num_front = gb.read(4) as usize;
    let num_side = gb.read(4) as usize;
    let num_back = gb.read(4) as usize;
    let num_lfe = gb.read(2) as usize;
    let num_assoc = gb.read(3) as usize;
    let num_cc = gb.read(4) as usize;
    if gb.read_bit() {
        gb.skip(4);
    }
    if gb.read_bit() {
        gb.skip(4);
    }
    if gb.read_bit() {
        gb.skip(3);
    }
    if gb.left()
        < 5 * (num_front + num_side + num_back + num_cc) as i64
            + 4 * (num_lfe + num_assoc + num_cc) as i64
    {
        return Err(Error::InvalidData("decode_pce: buffer exhausted".into()));
    }
    map.clear();
    map.resize(num_front + num_side + num_back + num_lfe + num_cc, [0; 3]);
    let mut tags = 0;
    decode_channel_map(&mut map[tags..], pos::FRONT, gb, num_front);
    tags += num_front;
    decode_channel_map(&mut map[tags..], pos::SIDE, gb, num_side);
    tags += num_side;
    decode_channel_map(&mut map[tags..], pos::BACK, gb, num_back);
    tags += num_back;
    decode_channel_map(&mut map[tags..], pos::LFE, gb, num_lfe);
    tags += num_lfe;
    gb.skip_long(4 * num_assoc as i64);
    decode_channel_map(&mut map[tags..], pos::CC, gb, num_cc);
    tags += num_cc;

    // relative_align_get_bits (aacdec.c:808)
    let n = (byte_align_ref as i64 - gb.count() as i64) & 7;
    if n != 0 {
        gb.skip(n as u32);
    }

    let mut comment_len = gb.read(8) as usize * 8;
    if gb.left() < comment_len as i64 {
        return Err(Error::InvalidData("decode_pce: buffer exhausted".into()));
    }

    // Height extension (aacdec.c:879-961): marker 0xAC, 2-bit layers,
    // then reorder front/side/back elements per layer.
    if comment_len >= 16 + (num_front + num_side + num_back) * 2 && gb.peek(8) == 0xAC {
        gb.skip(8);
        let mut height: [Vec<u8>; 4] = Default::default();
        let mut tag_copy: Vec<[u8; 3]> = map.clone();
        let mut invalid = false;
        let mut ht = 0usize;
        for (cnt, slot) in [(num_front, 0usize), (num_side, 1), (num_back, 2)] {
            height[slot] = Vec::with_capacity(cnt);
            for _ in 0..cnt {
                let h = gb.read(2) as u8;
                invalid |= h > 2;
                height[slot].push(h);
                ht += 1;
            }
        }
        ht += num_lfe + num_cc;
        if !invalid && ht == tags {
            let mut out: Vec<[u8; 3]> = Vec::with_capacity(tags);
            for layer in 0..3u8 {
                for (slot, cnt) in [(0usize, num_front), (1, num_side), (2, num_back)] {
                    for j in 0..cnt {
                        if height[slot][j] == layer {
                            out.push(tag_copy.remove(0));
                        }
                    }
                }
                if layer == 0 {
                    for _ in 0..num_lfe + num_cc {
                        out.push(tag_copy.remove(0));
                    }
                }
            }
            *map = out;
        }
        comment_len -= 8 + (num_front + num_side + num_back) * 2;
    }
    gb.skip_long(comment_len as i64);
    Ok(tags)
}

/// `decode_fill` (aacdec.c:1991) — fill text; a libfaac marker costs the
/// first frame (C sets skip_samples = 1024).
fn decode_fill(gb: &mut Gb, mut len: i32) -> Result<Option<u32>> {
    let mut skip_after = None;
    if len >= 13 + 7 * 8 {
        gb.skip(13);
        len -= 13;
        let mut buf = Vec::new();
        while buf.len() + 1 < 256 && len >= 8 {
            buf.push(gb.read(8) as u8);
            len -= 8;
        }
        // sscanf(buf, "libfaac %d.%d") == 2
        if let Some(rest) = buf.strip_prefix(b"libfaac ") {
            let mut it = rest.split(|&b| b == b'.');
            let maj = it.next().and_then(|s| parse_dec(s));
            let min = it.next().and_then(|s| parse_dec(s));
            if let (Some(_), Some(_)) = (maj, min) {
                skip_after = Some(1024);
            }
        }
    }
    gb.skip_long(len as i64);
    Ok(skip_after)
}

fn parse_dec(s: &[u8]) -> Option<i32> {
    let mut v: i32 = 0;
    let mut any = false;
    for &c in s {
        if c.is_ascii_digit() {
            v = v * 10 + (c - b'0') as i32;
            any = true;
        } else {
            break;
        }
    }
    any.then_some(v)
}

/// `decode_extension_payload` (aacdec.c:2024) — SBR/PS payloads are
/// skipped (LC core decodes at base rate; documented divergence).
/// Returns (bytes consumed, a pending skip_samples from libfaac fills).
fn decode_extension_payload(gb: &mut Gb, cnt: i32) -> Result<(i32, Option<u32>)> {
    let ty = gb.read(4) as u8;
    match ty {
        0xb => {
            // EXT_DYNAMIC_RANGE: parsed, not applied.
            let n = decode_dynamic_range(gb);
            Ok((n, None))
        }
        0x0 => {
            let skip = decode_fill(gb, 8 * cnt - 4)?;
            Ok((cnt, skip))
        }
        _ => {
            gb.skip_long(8 * cnt as i64 - 4);
            Ok((cnt, None))
        }
    }
}

/// `decode_drc_channel_exclusions` (aacdec.c:1925).
fn decode_drc_channel_exclusions(gb: &mut Gb) -> i32 {
    let mut num_excl_chan = 0usize;
    loop {
        for _ in 0..7 {
            let _ = gb.read_bit();
            num_excl_chan += 1;
        }
        if num_excl_chan < 64 - 7 && gb.read_bit() {
            continue;
        }
        break;
    }
    (num_excl_chan / 7) as i32
}

/// `decode_dynamic_range` (aacdec.c:1944) — consumed, not applied.
/// Returns C's byte-count bookkeeping `n`.
fn decode_dynamic_range(gb: &mut Gb) -> i32 {
    let mut n = 1;
    let mut drc_num_bands = 1usize;
    if gb.read_bit() {
        gb.skip(4); // pce_instance_tag
        gb.skip(4); // tag_reserved_bits
        n += 1;
    }
    if gb.read_bit() {
        n += decode_drc_channel_exclusions(gb);
    }
    if gb.read_bit() {
        let band_incr = gb.read(4) as usize;
        gb.skip(4); // interpolation_scheme
        n += 1;
        drc_num_bands += band_incr;
        for _ in 0..drc_num_bands {
            gb.skip(8); // band_top
            n += 1;
        }
    }
    if gb.read_bit() {
        gb.skip(7); // prog_ref_level
        gb.skip(1);
        n += 1;
    }
    for _ in 0..drc_num_bands {
        gb.skip(1); // dyn_rng_sgn
        gb.skip(7); // dyn_rng_ctl
        n += 1;
    }
    n
}

// ---------------------------------------------------------------------
// Spectral decode + dequant — aacdec_proc_template.c:57
// ---------------------------------------------------------------------

/// `decode_spectrum_and_dequant` (aacdec_proc_template.c:57), float
/// branch: the four VMUL macros from aacdec_float.c:85-150 inlined.
fn decode_spectrum_and_dequant(
    ctx: &mut AacContext,
    t: &Tables,
    gb: &mut Gb,
    pulse: Option<&Pulse>,
    sce: &mut Sce,
) -> Result<()> {
    let ics = sce.ics.clone();
    let c = 1024 / ics.num_windows;
    let offsets: Vec<usize> = ics
        .swb_offset(ctx.oc[1].m4ac.sampling_index.clamp(0, 12) as usize)
        .iter()
        .map(|&o| o as usize)
        .collect();

    // Zero the spectrum tail beyond max_sfb in every window.
    for g in 0..ics.num_windows {
        let off = offsets[ics.max_sfb as usize];
        for k in &mut sce.coeffs[g * 128 + off..g * 128 + c] {
            *k = 0.0;
        }
    }

    let mut idx = 0usize;
    let mut coef_base = 0usize; // window-group offset into coeffs
    for g in 0..ics.num_window_groups {
        let g_len = ics.group_len[g];
        for i in 0..ics.max_sfb as usize {
            let cbt_m1 = sce.band_type[idx].wrapping_sub(1);
            let mut cfo = coef_base + offsets[i];
            let off_len = offsets[i + 1] - offsets[i];

            if cbt_m1 >= bt::INTENSITY2 as u8 - 1 {
                for _ in 0..g_len {
                    for k in &mut sce.coeffs[cfo..cfo + off_len] {
                        *k = 0.0;
                    }
                    cfo += 128;
                }
            } else if cbt_m1 == bt::NOISE as u8 - 1 {
                // PNS: scaled white noise.
                for _ in 0..g_len {
                    let mut energy = 0.0f32;
                    for k in 0..off_len {
                        ctx.random_state = ctx
                            .random_state
                            .wrapping_mul(1664525)
                            .wrapping_add(1013904223);
                        sce.coeffs[cfo + k] = ctx.random_state as i32 as f32;
                        energy += sce.coeffs[cfo + k] * sce.coeffs[cfo + k];
                    }
                    let scale = sce.sf[idx] / energy.sqrt();
                    for k in 0..off_len {
                        sce.coeffs[cfo + k] *= scale;
                    }
                    cfo += 128;
                }
            } else {
                let vq = tables::CODEBOOK_VECTOR_VALS[cbt_m1 as usize];
                let vlc = &t.vlc_spectral[cbt_m1 as usize];
                match cbt_m1 >> 1 {
                    0 => {
                        // Quad codebooks 1/2 (VMUL4).
                        let sf = sce.sf[idx];
                        for _ in 0..g_len {
                            let mut cf = cfo;
                            let mut len = off_len;
                            loop {
                                let cb_idx = vlc.get(gb)? as usize;
                                for (shift, slot) in [(0, 0usize), (2, 1), (4, 2), (6, 3)] {
                                    sce.coeffs[cf + slot] = vq[(cb_idx >> shift) & 3] * sf;
                                }
                                cf += 4;
                                len -= 4;
                                if len == 0 {
                                    break;
                                }
                            }
                            cfo += 128;
                        }
                    }
                    1 => {
                        // Quad codebooks 3/4 with signs (VMUL4S,
                        // aacdec_float.c:132): the packed symbol's top
                        // nibble is the nonzero mask steering the sign
                        // rotation.
                        let sf = sce.sf[idx];
                        for _ in 0..g_len {
                            let mut cf = cfo;
                            let mut len = off_len;
                            loop {
                                let cb_idx = vlc.get(gb)? as usize;
                                let nnz = (cb_idx >> 8) & 15;
                                let mut sign = if nnz != 0 { gb.peek_left(nnz) } else { 0 };
                                gb.skip(nnz as u32);
                                let mut nz = (cb_idx >> 12) as u32;
                                let sbits = sf.to_bits();
                                for (shift, slot) in [(0, 0usize), (2, 1), (4, 2), (6, 3)] {
                                    let v = vq[(cb_idx >> shift) & 3];
                                    sce.coeffs[cf + slot] =
                                        f32::from_bits(sbits ^ (sign & 1u32 << 31)) * v;
                                    sign <<= nz & 1;
                                    nz >>= 1;
                                }
                                cf += 4;
                                len -= 4;
                                if len == 0 {
                                    break;
                                }
                            }
                            cfo += 128;
                        }
                    }
                    2 => {
                        // Pair codebooks 5/6 (VMUL2).
                        let sf = sce.sf[idx];
                        for _ in 0..g_len {
                            let mut cf = cfo;
                            let mut len = off_len;
                            loop {
                                let cb_idx = vlc.get(gb)? as usize;
                                sce.coeffs[cf] = vq[cb_idx & 15] * sf;
                                sce.coeffs[cf + 1] = vq[(cb_idx >> 4) & 15] * sf;
                                cf += 2;
                                len -= 2;
                                if len == 0 {
                                    break;
                                }
                            }
                            cfo += 128;
                        }
                    }
                    3 | 4 => {
                        // Pair codebooks 7..10 with signs (VMUL2S).
                        let sf = sce.sf[idx];
                        for _ in 0..g_len {
                            let mut cf = cfo;
                            let mut len = off_len;
                            loop {
                                let cb_idx = vlc.get(gb)? as usize;
                                let nnz = (cb_idx >> 8) & 15;
                                let sign = if nnz != 0 {
                                    gb.peek(nnz as u32) << (cb_idx >> 12)
                                } else {
                                    0
                                };
                                gb.skip(nnz as u32);
                                let s0 = f32::from_bits(sf.to_bits() ^ (sign >> 1 << 31));
                                let s1 = f32::from_bits(sf.to_bits() ^ (sign << 31));
                                sce.coeffs[cf] = vq[cb_idx & 15] * s0;
                                sce.coeffs[cf + 1] = vq[(cb_idx >> 4) & 15] * s1;
                                cf += 2;
                                len -= 2;
                                if len == 0 {
                                    break;
                                }
                            }
                            cfo += 128;
                        }
                    }
                    _ => {
                        // Escape codebook 11 (aacdec_proc_template.c:216-295):
                        // sign bits first (per the packed mask), then per
                        // escaped value a run of ≤8 one-bits, its length,
                        // and a (b+4)-bit magnitude offset.
                        let sf = sce.sf[idx];
                        let cbrt = &t.cbrt;
                        for _ in 0..g_len {
                            let mut cf = cfo;
                            let mut len = off_len;
                            loop {
                                let mut cb_idx = vlc.get(gb)? as usize;
                                if cb_idx == 0x0000 {
                                    sce.coeffs[cf] = 0.0;
                                    sce.coeffs[cf + 1] = 0.0;
                                    cf += 2;
                                } else {
                                    let nnz = (cb_idx >> 12) as u32;
                                    let nzt = ((cb_idx >> 8) & 0xf) as u32;
                                    let mut bits = gb.peek_left(nnz);
                                    gb.skip(nnz);
                                    for j in 0..2 {
                                        if nzt & (1 << j) != 0 {
                                            let cache = gb.peek_left(25);
                                            let b = (!cache).leading_zeros() as i32;
                                            if b > 8 {
                                                return Err(Error::InvalidData(
                                                    "escape overflow in spectral data".into(),
                                                ));
                                            }
                                            gb.skip(b as u32 + 1);
                                            let b = b + 4;
                                            let n = (1u32 << b) + gb.read(b as u32);
                                            sce.coeffs[cf] = f32::from_bits(
                                                cbrt[n as usize] | (bits & 1u32 << 31),
                                            );
                                            bits <<= 1;
                                        } else {
                                            let v = vq[cb_idx & 15];
                                            sce.coeffs[cf] =
                                                f32::from_bits(v.to_bits() | (bits & 1u32 << 31));
                                            bits <<= (v != 0.0) as u32;
                                        }
                                        cb_idx >>= 4;
                                        cf += 1;
                                    }
                                }
                                len -= 2;
                                if len == 0 {
                                    break;
                                }
                            }
                            // vector_fmul_scalar(cfo, cfo, sf, off_len)
                            for k in 0..off_len {
                                sce.coeffs[cfo + k] *= sf;
                            }
                            cfo += 128;
                        }
                    }
                }
            }
            idx += 1;
        }
        coef_base += g_len << 7;
    }

    // Pulses (aacdec_proc_template.c:304-326).
    if let Some(pulse) = pulse {
        let mut idx = 0usize;
        for i in 0..pulse.num_pulse {
            let mut co = sce.coeffs[pulse.pos[i].min(1023)];
            while offsets[idx + 1] <= pulse.pos[i] {
                idx += 1;
            }
            if sce.band_type[idx] != bt::NOISE && sce.sf[idx] != 0.0 {
                let mut ico = -(pulse.amp[i] as f32);
                if co != 0.0 {
                    co /= sce.sf[idx];
                    ico = co / (co.abs().sqrt().sqrt()) + if co > 0.0 { -ico } else { ico };
                }
                sce.coeffs[pulse.pos[i].min(1023)] =
                    ico.abs().cbrt() * ico * sce.sf[idx];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Element decoders
// ---------------------------------------------------------------------

/// `ff_aac_decode_ics` (aacdec.c:1785) — non-ER LC syntax.
fn decode_ics(ctx: &mut AacContext, t: &Tables, sce: &mut Sce, gb: &mut Gb, common_window: bool) -> Result<()> {
    let mut pulse = Pulse::default();
    let global_gain = gb.read(8);

    if !common_window {
        decode_ics_info(ctx, &mut sce.ics, gb)?;
    }
    let ics = sce.ics.clone();
    decode_band_types(&ics, &mut sce.band_type, gb)?;
    let mut sfo = sce.sfo;
    let r = decode_scalefactors(t, &ics, &sce.band_type, &mut sfo, gb, global_gain);
    sce.sfo = sfo;
    r?;
    dequant_scalefactors(t, sce);

    let pulse_present = gb.read_bit();
    if pulse_present {
        if sce.ics.window_sequence[0] == win::EIGHT_SHORT {
            return Err(Error::InvalidData(
                "pulse tool not allowed in eight-short sequence".into(),
            ));
        }
        decode_pulses(
            &mut pulse,
            gb,
            sce.ics.swb_offset(ctx.oc[1].m4ac.sampling_index.clamp(0, 12) as usize),
            sce.ics.num_swb,
        )?;
    }
    sce.tns.present = gb.read_bit();
    if sce.tns.present {
        let ics_copy = sce.ics.clone();
        let mut tns = std::mem::take(&mut sce.tns);
        let r = decode_tns(&ics_copy, &mut tns, gb);
        sce.tns = tns;
        r?;
    }
    if gb.read_bit() {
        decode_gain_control(&sce.ics, gb);
    }

    decode_spectrum_and_dequant(ctx, t, gb, pulse_present.then_some(&pulse), sce)?;
    Ok(())
}

/// `decode_cpe` (aacdec.c:1879).
fn decode_cpe(ctx: &mut AacContext, t: &Tables, che: &mut Che, gb: &mut Gb) -> Result<(bool, u32)> {
    let mut ms_present = 0u32;
    let common_window = gb.read_bit();
    if common_window {
        let mut ics0 = std::mem::take(&mut che.ch[0].ics);
        decode_ics_info(ctx, &mut ics0, gb)?;
        let kb1 = che.ch[1].ics.use_kb_window[0];
        che.ch[1].ics = ics0.clone();
        che.ch[0].ics = ics0;
        che.ch[1].ics.use_kb_window[1] = kb1;
        if che.ch[1].ics.predictor_present {
            return Err(Error::Unsupported(
                "LTP predictor in CPE is not ported".into(),
            ));
        }
        ms_present = gb.read(2);
        if ms_present == 3 {
            return Err(Error::InvalidData("ms_present = 3 is reserved".into()));
        } else if ms_present != 0 {
            decode_mid_side_stereo(che, gb, ms_present);
        }
    }
    let (mut ch0, mut ch1) = (std::mem::take(&mut che.ch[0]), std::mem::take(&mut che.ch[1]));
    let r0 = decode_ics(ctx, t, &mut ch0, gb, common_window);
    che.ch[0] = ch0;
    r0?;
    let r1 = decode_ics(ctx, t, &mut ch1, gb, common_window);
    che.ch[1] = ch1;
    r1?;

    if common_window && ms_present != 0 {
        apply_mid_side_stereo(che);
    }
    apply_intensity_stereo(che, ms_present);
    Ok((common_window, ms_present))
}

/// `decode_cce` (aacdec_proc_template.c:357).
fn decode_cce(ctx: &mut AacContext, t: &Tables, che: &mut Che, gb: &mut Gb) -> Result<()> {
    let mut num_gain = 0usize;
    che.coup.coupling_point = 2 * gb.read_bit() as u8;
    che.coup.num_coupled = gb.read(3) as usize;
    for c in 0..=che.coup.num_coupled {
        num_gain += 1;
        che.coup.type_[c] = if gb.read_bit() { ty::CPE } else { ty::SCE };
        che.coup.id_select[c] = gb.read(4) as usize;
        if che.coup.type_[c] == ty::CPE {
            che.coup.ch_select[c] = gb.read(2) as u8;
            if che.coup.ch_select[c] == 3 {
                num_gain += 1;
            }
        } else {
            che.coup.ch_select[c] = 2;
        }
    }
    che.coup.coupling_point += (gb.read_bit() || (che.coup.coupling_point >> 1) != 0) as u8;

    let sign = gb.read(1) != 0;
    let scale = CCE_SCALE[gb.read(2) as usize];

    decode_ics(ctx, t, &mut che.ch[0], gb, false)?;

    for c in 0..num_gain {
        let mut idx = 0usize;
        let mut cge = true;
        let mut gain: i32 = 0;
        let mut gain_cache = 1.0f32;
        if c != 0 {
            cge = che.coup.coupling_point == AFTER_IMDCT || gb.read_bit();
            gain = if cge {
                t.vlc_scalefactors.get(gb)? as i32 - 60
            } else {
                0
            };
            gain_cache = get_gain(scale, gain);
        }
        if che.coup.coupling_point == AFTER_IMDCT {
            che.coup.gain[c][0] = gain_cache;
        } else {
            for g in 0..che.ch[0].ics.num_window_groups {
                for sfb in 0..che.ch[0].ics.max_sfb as usize {
                    if che.ch[0].band_type[idx] != bt::ZERO {
                        if !cge {
                            let mut tv = t.vlc_scalefactors.get(gb)? as i32 - 60;
                            if tv != 0 {
                                let mut s = 1.0f32;
                                if sign {
                                    s -= 2.0 * (gain & 0x1) as f32;
                                    gain >>= 1;
                                }
                                gain += tv;
                                tv = gain;
                                gain_cache = get_gain(scale, tv) * s;
                            }
                        }
                        che.coup.gain[c][idx] = gain_cache;
                    }
                    idx += 1;
                }
            }
        }
    }
    Ok(())
}

/// `GET_GAIN` (aacdec.h:136 area): scale·2^(gain/4).
fn get_gain(scale: f32, gain: i32) -> f32 {
    scale * tables_pow2(gain)
}

/// pow(2, gain/4) via the pow2sf ladder.
fn tables_pow2(exp_quarters: i32) -> f32 {
    let t = tables();
    let idx = (exp_quarters + POW_SF2_ZERO).clamp(0, 427);
    // The ladder spans 2^((i-200)/4): gain g maps to row (g + 200).
    let mut v = t.pow2sf[idx as usize];
    if exp_quarters >= 0 {
        v = t.pow2sf[idx as usize];
    }
    v
}

// ---------------------------------------------------------------------
// DSP — aacdec_dsp_template.c (float)
// ---------------------------------------------------------------------

/// `dequant_scalefactors` (aacdec_dsp_template.c:41).
fn dequant_scalefactors(t: &Tables, sce: &mut Sce) {
    let mut idx = 0usize;
    for g in 0..sce.ics.num_window_groups {
        for sfb in 0..sce.ics.max_sfb as usize {
            sce.sf[idx] = match sce.band_type[g * sce.ics.max_sfb as usize + sfb] {
                bt::ZERO => 0.0,
                x if x == bt::INTENSITY || x == bt::INTENSITY2 => {
                    pow2sf_val(t, -sce.sfo[idx] - 100)
                }
                bt::NOISE => -pow2sf_val(t, sce.sfo[idx]),
                _ => -pow2sf_val(t, sce.sfo[idx]),
            };
            idx += 1;
        }
    }
}

/// `ff_aac_pow2sf_tab[-sfo - 100 + POW_SF2_ZERO]`-style lookups: the
/// ladder stores 2^((i−200)/4), so 2^(q/4) is row q+200.
fn pow2sf_val(t: &Tables, quarters: i32) -> f32 {
    let idx = (quarters + POW_SF2_ZERO).clamp(0, 427) as usize;
    t.pow2sf[idx]
}

/// `apply_mid_side_stereo` (aacdec_dsp_template.c:84) with
/// `butterflies_float` (float_dsp.c): ch0 ← mid+side, ch1 ← mid−side.
fn apply_mid_side_stereo(che: &mut Che) {
    let ics = che.ch[0].ics.clone();
    let max_sfb_ste = che.max_sfb_ste as usize;
    let offsets: Vec<usize> = ics
        .swb_offset(SRATE_IDX.with(|c| c.get()))
        .iter()
        .map(|&o| o as usize)
        .collect();
    let mut base = 0usize;
    for g in 0..ics.num_window_groups {
        for sfb in 0..max_sfb_ste {
            let idx = g * max_sfb_ste + sfb;
            if che.ms_mask[idx.min(127)]
                && che.ch[0].band_type[idx.min(127)] < bt::NOISE
                && che.ch[1].band_type[idx.min(127)] < bt::NOISE
            {
                for group in 0..ics.group_len[g] {
                    let o = base + group * 128 + offsets[sfb];
                    let n = offsets[sfb + 1] - offsets[sfb];
                    for k in 0..n {
                        let a = che.ch[0].coeffs[o + k];
                        let b = che.ch[1].coeffs[o + k];
                        che.ch[0].coeffs[o + k] = a + b;
                        che.ch[1].coeffs[o + k] = a - b;
                    }
                }
            }
        }
        base += ics.group_len[g] * 128;
    }
}

/// `apply_intensity_stereo` (aacdec_dsp_template.c:120).
fn apply_intensity_stereo(che: &mut Che, ms_present: u32) {
    let ics = che.ch[1].ics.clone();
    let offsets: Vec<usize> = ics
        .swb_offset(SRATE_IDX.with(|c| c.get()))
        .iter()
        .map(|&o| o as usize)
        .collect();
    let mut base = 0usize;
    for g in 0..ics.num_window_groups {
        for sfb in 0..ics.max_sfb as usize {
            let idx = g * ics.max_sfb as usize + sfb;
            let b = che.ch[1].band_type[idx];
            if b == bt::INTENSITY || b == bt::INTENSITY2 {
                let mut c: f32 = -1.0 + 2.0 * (b as i32 - 14) as f32;
                if ms_present != 0 {
                    c *= 1.0 - 2.0 * che.ms_mask[idx.min(127)] as i32 as f32;
                }
                let scale = c * che.ch[1].sf[idx];
                for group in 0..ics.group_len[g] {
                    let o = base + group * 128 + offsets[sfb];
                    for k in 0..offsets[sfb + 1] - offsets[sfb] {
                        che.ch[1].coeffs[o + k] = scale * che.ch[0].coeffs[o + k];
                    }
                }
            }
        }
        base += ics.group_len[g] * 128;
    }
}

/// `compute_lpc_coefs` (lpc_functions.h:54) with the AAC call's
/// normalize=0: plain Levinson-Durbin reflection from the quantized TNS
/// coefficients into the AR filter taps.
fn compute_lpc_coefs(autoc: &[f32; TNS_MAX_ORDER], order: usize, lpc: &mut [f32; TNS_MAX_ORDER]) {
    let mut lpc_last = *autoc;
    for i in 0..order {
        let r = -autoc[i];
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let f = lpc_last[j];
            let b = lpc_last[i - 1 - j];
            lpc[j] = f + r * b;
            lpc[i - 1 - j] = b + r * f;
        }
        lpc_last = *lpc;
    }
}

/// `apply_tns` (aacdec_dsp_template.c:164) — decode=1 (ar filter) branch.
fn apply_tns(sce: &mut Sce) {
    let ics = sce.ics.clone();
    let mmm = ics.tns_max_bands.min(ics.max_sfb as usize);
    if mmm == 0 {
        return;
    }
    let offsets: Vec<usize> = ics
        .swb_offset(SRATE_IDX.with(|c| c.get()))
        .iter()
        .map(|&o| o as usize)
        .collect();
    let mut lpc = [0f32; TNS_MAX_ORDER];
    for w in 0..ics.num_windows {
        let mut bottom = ics.num_swb;
        for filt in 0..sce.tns.n_filt[w] {
            let top = bottom;
            bottom = top.saturating_sub(sce.tns.length[w][filt]);
            let order = sce.tns.order[w][filt];
            if order == 0 {
                continue;
            }
            compute_lpc_coefs(&sce.tns.coef[w][filt], order, &mut lpc);
            let start_band = bottom.min(mmm);
            let end_band = top.min(mmm);
            let start = offsets[start_band];
            let end = offsets[end_band];
            if end <= start {
                continue;
            }
            let size = end - start;
            let (inc, mut pos) = if sce.tns.direction[w * 4 + filt] {
                (-1i64, (end - 1) as i64)
            } else {
                (1i64, start as i64)
            };
            pos += (w * 128) as i64;
            for _ in 0..size {
                let p = pos as usize;
                let mut acc = sce.coeffs[p];
                for i in 1..=order.min(p.min(1023)) {
                    let q = (pos - i as i64 * inc) as usize; // coef[start − i·inc]
                    acc -= sce.coeffs[q] * lpc[i - 1];
                }
                sce.coeffs[p] = acc;
                pos += inc;
            }
        }
    }
}

/// `vector_fmul_window` (float_dsp.c): `dst[k] = s0[k]·win[n−1−k] −
/// s1[n−1−k]·win[k]`, `dst[n−1−k] = s0[k]·win[k] + s1[n−1−k]·win[n−1−k]`.
fn vector_fmul_window(dst: &mut [f32], src0: &[f32], src1: &[f32], win: &[f32], n: usize) {
    for k in 0..n {
        let s0 = src0[k];
        let s1 = src1[n - 1 - k];
        let wi = win[k];
        let wj = win[n - 1 - k];
        dst[k] = s0 * wj - s1 * wi;
        dst[n - 1 - k] = s0 * wi + s1 * wj;
    }
}

/// `imdct_and_windowing` (aacdec_dsp_template.c:325) — 1024-sample frame
/// flavor (960/768/LD/ELD gated upstream).
fn imdct_and_windowing(t: &Tables, sce: &mut Sce) {
    let seq = sce.ics.window_sequence;
    let prev_seq = sce.ics.window_sequence[1];
    let swindow: &[f32] = if sce.ics.use_kb_window[0] {
        &t.kbd_128
    } else {
        &t.sine_128
    };
    let lwindow_prev: &[f32] = if sce.ics.use_kb_window[1] {
        &t.kbd_1024
    } else {
        &t.sine_1024
    };
    let swindow_prev: &[f32] = if sce.ics.use_kb_window[1] {
        &t.kbd_128
    } else {
        &t.sine_128
    };

    let mut buf = [0f32; 2048];
    let mut temp = [0f32; 128];

    // imdct
    let coeffs: Vec<f32> = sce.coeffs.to_vec();
    if seq[0] == win::EIGHT_SHORT {
        for i in 0..8 {
            t.mdct_128
                .run(&coeffs[i * 128..i * 128 + 128], &mut buf[i * 128..i * 128 + 256]);
        }
    } else {
        t.mdct_1024.run(&coeffs, &mut buf);
    }

    let out = &mut sce.ret_buf[..];
    let saved = &mut sce.saved[..];

    // window overlapping
    if (prev_seq == win::ONLY_LONG || prev_seq == win::LONG_STOP)
        && (seq[0] == win::ONLY_LONG || seq[0] == win::LONG_START)
    {
        vector_fmul_window(out, saved, &buf, lwindow_prev, 512);
    } else {
        out[..448].copy_from_slice(&saved[..448]);
        if seq[0] == win::EIGHT_SHORT {
            vector_fmul_window(&mut out[448..], &saved[448..], &buf[0..], swindow_prev, 64);
            vector_fmul_window(&mut out[448 + 128..], &buf[64..], &buf[128..], swindow, 64);
            vector_fmul_window(&mut out[448 + 256..], &buf[192..], &buf[256..], swindow, 64);
            vector_fmul_window(&mut out[448 + 384..], &buf[320..], &buf[384..], swindow, 64);
            vector_fmul_window(&mut temp, &buf[448..], &buf[512..], swindow, 64);
            out[448 + 512..448 + 576].copy_from_slice(&temp[..64]);
        } else {
            vector_fmul_window(&mut out[448..], &saved[448..], &buf, swindow_prev, 64);
            out[576..1024].copy_from_slice(&buf[64..512]);
        }
    }

    // buffer update
    if seq[0] == win::EIGHT_SHORT {
        saved[..64].copy_from_slice(&temp[64..128]);
        vector_fmul_window(&mut saved[64..], &buf[576..], &buf[640..], swindow, 64);
        vector_fmul_window(&mut saved[192..], &buf[704..], &buf[768..], swindow, 64);
        vector_fmul_window(&mut saved[320..], &buf[832..], &buf[896..], swindow, 64);
        saved[448..512].copy_from_slice(&buf[960..1024]);
    } else if seq[0] == win::LONG_START {
        saved[..448].copy_from_slice(&buf[512..960]);
        saved[448..512].copy_from_slice(&buf[960..1024]);
    } else {
        saved[..512].copy_from_slice(&buf[512..1024]);
    }
}

/// `apply_dependent_coupling` (aacdec_float_coupling.h:37).
fn apply_dependent_coupling(target: &mut Che, _t_arget_ty: u8, cce: &Che, index: usize) {
    let ics = cce.ch[0].ics.clone();
    let offsets: Vec<usize> = ics
        .swb_offset(SRATE_IDX.with(|c| c.get()))
        .iter()
        .map(|&o| o as usize)
        .collect();
    let mut idx = 0usize;
    let mut dbase = 0usize;
    let mut sbase = 0usize;
    for g in 0..ics.num_window_groups {
        for i in 0..ics.max_sfb as usize {
            if cce.ch[0].band_type[idx] != bt::ZERO {
                let gain = cce.coup.gain[index][idx];
                for group in 0..ics.group_len[g] {
                    for k in offsets[i]..offsets[i + 1] {
                        target.ch[0].coeffs[dbase + group * 128 + k] +=
                            gain * cce.ch[0].coeffs[sbase + group * 128 + k];
                    }
                }
            }
            idx += 1;
        }
        dbase += ics.group_len[g] * 128;
        sbase += ics.group_len[g] * 128;
    }
}

/// `apply_independent_coupling` (aacdec_float_coupling.h:71).
fn apply_independent_coupling(target: &mut Che, cce: &Che, index: usize) {
    let gain = cce.coup.gain[index][0];
    for k in 0..1024 {
        target.ch[0].ret_buf[k] += gain * cce.ch[0].ret_buf[k];
    }
}

/// `apply_channel_coupling` (aacdec.c:2090) — one coupling point.
fn apply_channel_coupling(
    ctx: &mut AacContext,
    ty: u8,
    id: usize,
    coupling_point: u8,
) {
    for cce_id in 0..MAX_ELEM_ID {
        let Some(cce) = ctx.che[ty::CCE as usize][cce_id].as_ref() else {
            continue;
        };
        if !cce.present || cce.coup.coupling_point != coupling_point {
            continue;
        }
        let mut index = 0usize;
        let coup = cce.coup.clone();
        for c in 0..=coup.num_coupled {
            if coup.type_[c] == ty && coup.id_select[c] == id {
                if coup.ch_select[c] != 1 {
                    if let Some(t) = ctx.che_mut(ty, id) {
                        apply_dependent_coupling(t, ty, cce, index);
                    }
                    if coup.ch_select[c] != 0 {
                        index += 1;
                    }
                }
                if coup.ch_select[c] != 2 {
                    if let Some(t) = ctx.che_mut(ty, id) {
                        apply_dependent_coupling(t, ty, cce, index);
                    }
                    index += 1;
                }
            } else {
                index += 1 + (coup.ch_select[c] == 3) as usize;
            }
        }
    }
}

/// Independent coupling (after IMDCT) — needs the CCE's own ret_buf, so
/// it runs from `spectral_to_sample` with the CCE already synthesized.
fn apply_independent_couplings(ctx: &mut AacContext, ty: u8, id: usize) {
    for cce_id in 0..MAX_ELEM_ID {
        let Some(cce) = ctx.che[ty::CCE as usize][cce_id].as_ref() else {
            continue;
        };
        if !cce.present || cce.coup.coupling_point != AFTER_IMDCT {
            continue;
        }
        let mut index = 0usize;
        let coup = cce.coup.clone();
        for c in 0..=coup.num_coupled {
            if coup.type_[c] == ty && coup.id_select[c] == id {
                if coup.ch_select[c] != 1 {
                    if let Some(t) = ctx.che_mut(ty, id) {
                        apply_independent_coupling(t, cce, index);
                    }
                    if coup.ch_select[c] != 0 {
                        index += 1;
                    }
                }
                if coup.ch_select[c] != 2 {
                    if let Some(t) = ctx.che_mut(ty, id) {
                        apply_independent_coupling(t, cce, index);
                    }
                    index += 1;
                }
            } else {
                index += 1 + (coup.ch_select[c] == 3) as usize;
            }
        }
    }
}

/// `spectral_to_sample` (aacdec.c:2123) for the LC object: TNS, both
/// dependent-coupling points, IMDCT+windowing, independent coupling —
/// over the elements in reverse type order like C.
fn spectral_to_sample(ctx: &mut AacContext, t: &Tables) {
    for ty in (0..=3u8).rev() {
        for id in 0..MAX_ELEM_ID {
            let present = ctx.che[ty as usize][id]
                .as_ref()
                .map(|c| c.present)
                .unwrap_or(false);
            if !present {
                continue;
            }
            if ty <= ty::CPE {
                apply_channel_coupling(ctx, ty, id, BEFORE_TNS);
            }
            if let Some(che) = ctx.che_mut(ty, id) {
                if che.ch[0].tns.present {
                    let mut ch0 = std::mem::take(&mut che.ch[0]);
                    apply_tns(&mut ch0);
                    che.ch[0] = ch0;
                }
                if che.ch[1].tns.present {
                    let mut ch1 = std::mem::take(&mut che.ch[1]);
                    apply_tns(&mut ch1);
                    che.ch[1] = ch1;
                }
            }
            if ty <= ty::CPE {
                apply_channel_coupling(ctx, ty, id, BETWEEN_TNS_AND_IMDCT);
            }
            if ty != ty::CCE {
                if let Some(che) = ctx.che_mut(ty, id) {
                    let mut ch0 = std::mem::take(&mut che.ch[0]);
                    imdct_and_windowing(t, &mut ch0);
                    let second = ty == ty::CPE;
                    let mut ch1 = if second { Some(std::mem::take(&mut che.ch[1])) } else { None };
                    if let Some(ch1) = ch1.as_mut() {
                        imdct_and_windowing(t, ch1);
                    }
                    che.ch[0] = ch0;
                    if let Some(ch1) = ch1 {
                        che.ch[1] = ch1;
                    }
                }
            }
            if ty <= ty::CCE {
                apply_independent_couplings(ctx, ty, id);
            }
            if let Some(che) = ctx.che_mut(ty, id) {
                che.present = false;
            }
        }
    }
}

// The swb_offset lookups during the apply stage need the sampling index
// outside the element tree; stash it in a thread-local like C's m4ac.
use std::cell::Cell;
thread_local! {
    static SRATE_IDX: Cell<usize> = const { Cell::new(4) };
}

// ---------------------------------------------------------------------
// Frame loop — decode_frame_ga (aacdec.c:2324)
// ---------------------------------------------------------------------

/// `decode_frame_ga` (aacdec.c:2324): the raw_data_block element loop.
/// Returns true when an audio frame was produced.
fn decode_frame_ga(ctx: &mut AacContext, t: &Tables, gb: &mut Gb) -> Result<bool> {
    let mut che_prev: Option<(u8, usize)> = None;
    let mut audio_found = false;
    let mut pce_found = false;
    let mut skip_after_fill: Option<u32> = None;
    let payload_alignment = gb.count();

    loop {
        let elem_type = gb.read(3) as u8;
        if elem_type == ty::END {
            break;
        }
        let mut elem_id = gb.read(4) as usize;

        if ctx.output_elements.is_empty() && elem_type != ty::PCE {
            return Err(Error::InvalidData(
                "no channel configuration before first element".into(),
            ));
        }
        if elem_type == ty::FIL && elem_id == 15 {
            elem_id += gb.read(8) as usize.wrapping_sub(1);
        }

        let mut che: Option<(u8, usize)> = None;
        if elem_type < ty::DSE {
            che = get_che(ctx, elem_type, elem_id);
            let Some(c) = che else {
                return Err(Error::InvalidData(format!(
                    "channel element {elem_type}.{elem_id} is not allocated"
                )));
            };
            if let Some(che_ref) = ctx.che_mut(c.0, c.1) {
                che_ref.present = true;
            }
        }

        match elem_type {
            ty::SCE | ty::LFE => {
                if let Some(c) = che {
                    if let Some(che_ref) = ctx.che_mut(c.0, c.1) {
                        let mut ch0 = std::mem::take(&mut che_ref.ch[0]);
                        let r = decode_ics(ctx, t, &mut ch0, gb, false);
                        che_ref.ch[0] = ch0;
                        r?;
                    }
                }
                audio_found = true;
            }
            ty::CPE => {
                if let Some(c) = che {
                    if let Some(che_ref) = ctx.che_mut(c.0, c.1) {
                        decode_cpe(ctx, t, che_ref, gb)?;
                    }
                }
                audio_found = true;
            }
            ty::CCE => {
                if let Some(c) = che {
                    if let Some(che_ref) = ctx.che_mut(c.0, c.1) {
                        decode_cce(ctx, t, che_ref, gb)?;
                    }
                }
            }
            ty::DSE => skip_data_stream_element(gb)?,
            ty::PCE => {
                let pushed = {
                    let was_locked = ctx.oc[1].status == OcStatus::Locked;
                    let none = ctx.oc[0].status == OcStatus::None;
                    if was_locked || none {
                        ctx.oc[0] = ctx.oc[1].clone();
                    }
                    ctx.oc[1].status = OcStatus::None;
                    was_locked || none
                };
                if pce_found && !pushed {
                    return Err(Error::InvalidData(
                        "second program_config_element in one frame".into(),
                    ));
                }
                let mut map = Vec::new();
                let tags = decode_pce(ctx, &mut map, gb, payload_alignment)?;
                if pce_found {
                    pop_output_configuration(ctx);
                } else {
                    output_configure(ctx, &map[..tags], OcStatus::TrialPce, false)?;
                    ctx.oc[1].m4ac.chan_config = 0;
                    pce_found = true;
                }
                if ctx.oc[1].m4ac.sample_rate == 0 {
                    // A PCE alone carries no rate; keep the container's.
                    ctx.oc[1].m4ac.sample_rate = ctx.oc[0].m4ac.sample_rate;
                    ctx.oc[1].m4ac.sampling_index = ctx.oc[0].m4ac.sampling_index;
                }
            }
            ty::FIL => {
                if gb.left() < 8 * elem_id as i64 {
                    return Err(Error::InvalidData(
                        "TYPE_FIL: buffer exhausted before END".into(),
                    ));
                }
                let mut elem_id = elem_id as i32;
                while elem_id > 0 {
                    let used = decode_extension_payload(gb, elem_id)?;
                    if let Some(skip) = used.1 {
                        skip_after_fill = Some(skip);
                    }
                    elem_id -= used.0;
                }
            }
            _ => return Err(Error::InvalidData("unknown element type".into())),
        }

        if elem_type < ty::DSE {
            che_prev = che;
        }
        if gb.left() < 3 {
            return Err(Error::InvalidData(
                "input buffer exhausted before END element".into(),
            ));
        }
    }

    if ctx.output_elements.is_empty() {
        return Ok(false);
    }
    if !audio_found {
        return Ok(false);
    }
    spectral_to_sample(ctx, t);
    if ctx.oc[1].status != OcStatus::None {
        ctx.oc[1].status = OcStatus::Locked;
    }
    ctx.fill_skip = skip_after_fill;
    Ok(true)
}

// ---------------------------------------------------------------------
// The decoder (ff_aac_decoder)
// ---------------------------------------------------------------------

/// `ff_aac_decoder` (aacdec.c:2665) — float, AAC-LC, FLTP out, one
/// 1024-sample frame per ADTS packet.
#[derive(Debug)]
pub struct AacDecoder {
    ctx: AacContext,
    params: CodecParameters,
    /// One-frame output queue.
    pending: Option<crate::util::audio_frame::AudioFrame>,
    eof: bool,
    /// Gapless skip carry (decode.c skip_samples application; mp3 path).
    skip_left: u32,
}

impl Default for AacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AacDecoder {
    pub fn new() -> Self {
        AacDecoder {
            ctx: AacContext::new(),
            params: CodecParameters::default(),
            pending: None,
            eof: false,
            skip_left: 0,
        }
    }
}

impl AudioDecoder for AacDecoder {
    /// `ff_aac_decode_init` / `aac_decode_init_float` (aacdec.c:1298):
    /// FLTP out, ADTS self-describing streams (no extradata yet).
    fn init(&mut self, params: &CodecParameters) -> Result<()> {
        match params.codec_id {
            CodecId::Aac => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "codec '{}' is not the AAC decoder",
                    other.name()
                )))
            }
        }
        self.params = params.clone();
        self.params.codec_type = MediaType::Audio;
        self.params.sample_fmt = SampleFormat::Fltp;
        Ok(())
    }

    /// `aac_decode_frame` (aacdec.c:2561): one ADTS frame per packet.
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<()> {
        let Some(pkt) = pkt else {
            self.eof = true;
            return Ok(());
        };
        if self.eof {
            return Err(Error::Eof);
        }
        self.pending = None;

        let t = tables();
        let mut gb = Gb::new(pkt.as_slice());
        let mut skip_from_fill = None;

        // aac_decode_frame_int (2505): ADTS header in-packet.
        if gb.peek(12) == 0xfff {
            let size = parse_adts_header(&mut self.ctx, &mut gb)?;
            let pkt_size = pkt.size();
            if size > 7 && pkt_size >= size {
                // The demuxer hands whole ADTS frames; the bit reader is
                // bounded by the packet, matching C's init_get_bits8.
            }
        } else if self.ctx.oc[1].m4ac.object_type == 0 {
            return Err(Error::InvalidData(
                "first packet carries no ADTS syncword".into(),
            ));
        }
        SRATE_IDX.with(|c| c.set(self.ctx.oc[1].m4ac.sampling_index.max(0) as usize));

        self.ctx.tags_mapped = 0;
        let got_frame = decode_frame_ga(&mut self.ctx, t, &mut gb)?;
        skip_from_fill = self.ctx.fill_skip.take();

        if !got_frame {
            return Ok(());
        }

        // frame_configure_elements (181): one FLTP frame, 1024 samples,
        // planes in output_element order.
        let layout = self.ctx.layout();
        let channels = self.ctx.output_elements.len().max(1);
        let mut frame =
            crate::util::audio_frame::AudioFrame::alloc(SampleFormat::Fltp, layout, 1024)?;
        frame.sample_rate = self.ctx.oc[1].m4ac.sample_rate;
        frame.pts = pkt.pts;
        frame.duration = pkt.duration;
        frame.time_base = pkt.time_base;

        // The demuxer's gapless trim (skip/discard), like the mp3 path.
        self.skip_left = self.skip_left.saturating_add(pkt.skip_samples);
        skip_from_fill = skip_from_fill.or(None);

        for (ch, &(ty, id)) in self.ctx.output_elements.iter().enumerate() {
            if ch >= frame.nb_planes() {
                break;
            }
            let src = &self.ctx.che[ty as usize][id].as_ref().unwrap().ch[0].ret_buf;
            // CPE second channel.
            let src: &[f32] = if ty == ty::CPE
                && ch + 1 < self.ctx.output_elements.len()
                && self.ctx.output_elements[ch + 1] == (ty, id)
            {
                &self.ctx.che[ty as usize][id].as_ref().unwrap().ch[1].ret_buf
            } else {
                src
            };
            let plane = frame.plane_mut(ch);
            for (dst, s) in plane.chunks_exact_mut(4).zip(src.iter()) {
                dst.copy_from_slice(&s.to_ne_bytes());
            }
        }

        // Apply the fill-element skip (libfaac) like avci->skip_samples.
        if let Some(skip) = skip_from_fill {
            self.skip_left = self.skip_left.saturating_add(skip);
        }
        if self.skip_left > 0 {
            let skip = (self.skip_left as usize).min(frame.nb_samples);
            frame.crop(skip, 0);
            self.skip_left -= skip as u32;
        }
        let discard = (pkt.discard_padding as usize).min(frame.nb_samples);
        if discard > 0 {
            frame.crop(0, discard);
        }

        if frame.nb_samples > 0 {
            self.pending = Some(frame);
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<crate::util::audio_frame::AudioFrame> {
        match self.pending.take() {
            Some(frame) => Ok(frame),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::Again),
        }
    }

    /// `flush` (aacdec.c:556): clear the IMDCT overlap.
    fn flush(&mut self) {
        for ty in 0..4 {
            for id in 0..MAX_ELEM_ID {
                if let Some(che) = self.ctx.che[ty][id].as_mut() {
                    for ch in &mut che.ch {
                        *ch.saved = [0.0; 1024];
                    }
                }
            }
        }
        self.pending = None;
        self.skip_left = 0;
    }
}

// AacContext::fill_skip — declared here to keep the struct literal simple.
impl AacContext {
    // (field declared below via the impl isn't possible; see struct)
}
