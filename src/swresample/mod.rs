//! libswresample — the audio resampler/converter (Phase 4a).
//!
//! Port map (all files in `FFmpeg/libswresample/`):
//!
//! | file | ports | what it holds |
//! |---|---|---|
//! | [`mod`] | `swresample.c` + `swresample_internal.h` + `options.c` | [`AudioData`], [`SwrContext`], the `swr_*` public API |
//! | [`audioconvert`] | `audioconvert.c` | the sample-format pair conversion matrix |
//! | [`rematrix`] | `rematrix.c` + `rematrix_template.c` | the channel-mix matrix build + apply |
//! | [`resample`] | `resample.c` + `resample_template.c` | the Kaiser-windowed polyphase rate converter |
//!
//! Not ported: soxr (engine 1), DSD (`dsd2pcm.c`), the SIMD dispatch
//! (`*_dsp.c` x86 variants — scalar only), noise shaping (`dither.c` beyond
//! the rectangular default documented at the call sites).

use crate::util::samplefmt::SampleFormat;

/// `SWR_CH_MAX` (`swresample_internal.h:28`).
pub const SWR_CH_MAX: usize = 64;

/// `AudioData` from `swresample_internal.h:47-55`: one `Arc<[u8]>` backing
/// buffer replaces C's `uint8_t *data` + `ch[SWR_CH_MAX]` pointer array,
/// reproducing the aliasing C sets up in `swri_realloc_audio`
/// (`swresample.c:426-456`) and `fill_audiodata` (`swresample.c:471-483`):
///
/// * planar: channel `i` occupies `data[i*count*bps .. (i+1)*count*bps]`
///   (C's 32-byte `ALIGN` stride padding at `swresample.c:31,438` is dropped —
///   it exists only for SIMD and is invisible to every caller);
/// * packed: channel `i` is the alias starting at `data[i*bps ..]` running to
///   the end of the buffer (C `ch[i] = data + i*bps`, `swresample.c:448`).
///
/// `count` is the capacity in samples per channel (C's grow-by-doubling
/// `swri_realloc_audio` stays with the SwrContext zone). `SWR_CH_MAX = 64`
/// (`swresample_internal.h:28`) lives in `mod.rs` there.
#[derive(Clone)]
pub struct AudioData {
    /// Backing buffer shared by all channels (aliases, not copies).
    pub data: std::sync::Arc<[u8]>,
    /// Number of channels (`ch_count`).
    pub ch_count: usize,
    /// Bytes per sample (`bps`).
    pub bps: usize,
    /// Capacity in samples per channel (`count`).
    pub count: usize,
    /// 1 if planar audio, 0 otherwise (`planar`).
    pub planar: bool,
    /// Sample format (`fmt`).
    pub fmt: SampleFormat,
}

impl AudioData {
    /// Zeroed buffer of `ch_count * count * bps` bytes (the `av_calloc`
    /// analog of `swri_realloc_audio`, `swresample.c:438`).
    pub fn new(fmt: SampleFormat, ch_count: usize, count: usize) -> Self {
        let bps = fmt.bytes_per_sample();
        AudioData {
            data: vec![0u8; ch_count * count * bps].into(),
            ch_count,
            bps,
            count,
            planar: fmt.is_planar(),
            fmt,
        }
    }

    /// Channel `ch`'s plane — `None` when the channel's address falls outside
    /// the backing buffer (C's NULL `ch[]` slot, the skip case at
    /// `audioconvert.c:259-260`).
    pub fn plane(&self, ch: usize) -> Option<&[u8]> {
        let start = if self.planar {
            ch * self.count * self.bps
        } else {
            ch * self.bps
        };
        if self.planar {
            self.data.get(start..start + self.count * self.bps)
        } else {
            self.data.get(start..)
        }
    }

    /// Mutable channel `ch` plane, via a single `Arc::make_mut` borrow so
    /// packed channels (which alias one buffer) can be written at their
    /// offsets without aliasing violations.
    pub fn plane_bytes_mut(&mut self, ch: usize) -> Option<&mut [u8]> {
        let data = std::sync::Arc::make_mut(&mut self.data);
        let start = if self.planar {
            ch * self.count * self.bps
        } else {
            ch * self.bps
        };
        if self.planar {
            data.get_mut(start..start + self.count * self.bps)
        } else {
            data.get_mut(start..)
        }
    }

    /// Whole backing buffer, mutable (for the later SwrContext zone).
    pub fn data_mut(&mut self) -> &mut [u8] {
        std::sync::Arc::make_mut(&mut self.data)
    }
}

pub mod audioconvert;
pub mod rematrix;
pub mod resample;
