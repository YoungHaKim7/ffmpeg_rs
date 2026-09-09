//! `AudioFrame` — the audio counterpart of [`super::frame::Frame`].
//!
//! Port of the audio fields of `AVFrame` (`libavutil/frame.h`) plus the
//! audio buffer allocation paths (`frame.c` `get_audio_buffer`,
//! `av_samples_alloc` / `av_samples_fill_arrays` in `samplefmt.c`):
//!
//! * **Plane layout** follows C's `data[]`/`linesize[]` audio semantics:
//!   an interleaved format is ONE plane with
//!   `linesize = nb_samples · channels · bps`; a planar format is one
//!   plane per channel with `linesize = nb_samples · bps`. Each plane
//!   reuses video's [`Plane`] with `rows = 1` (one row of samples).
//! * `nb_channels` is NOT a stored field — C removed `AVFrame.channels`;
//!   the count lives in `ch_layout.nb_channels` and a duplicate would
//!   desync. Use [`AudioFrame::channels`].
//! * **alloc** mirrors `frame.c`'s `get_audio_buffer`: per-plane OWN
//!   buffers (C allocates each `data[]` plane separately for planar
//!   audio), silence-filled (`av_samples_set_silence`), sized with the
//!   `align = 0` auto rule (sample count rounded up to a multiple of 32 —
//!   see [`samplefmt::samples_get_buffer_size`]).
//! * **wrap_buffer** mirrors `av_samples_fill_arrays` over ONE shared
//!   buffer (`align = 1`: no padding beyond the exact sizes).
//!
//! Not ported: `extended_data` (equal to `data` for non-extended layouts —
//! native orders only), `AVFrame.channels` (see above), side data.

use std::sync::Arc;

use super::{
    channel_layout::ChannelLayout,
    frame::Plane,
    rational::Rational,
    samplefmt::{self, SampleFormat},
};

/// `AVFrame` — audio subset.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// `data[]` — one plane (interleaved) or one per channel (planar).
    pub planes: Vec<Plane>,
    /// `nb_samples` — one row of samples across all channels.
    pub nb_samples: usize,
    /// `sample_rate`.
    pub sample_rate: i32,
    /// `format` — the `AVSampleFormat`.
    pub format: SampleFormat,
    /// `ch_layout`.
    pub ch_layout: ChannelLayout,
    /// `pts` — in `time_base` units.
    pub pts: i64,
    /// `duration`.
    pub duration: i64,
    /// `time_base`.
    pub time_base: Rational,
}

impl Default for AudioFrame {
    fn default() -> Self {
        // C's `get_audio_buffer` on a zeroed frame fails; our default is
        // the "unconfigured" shape callers overwrite field by field.
        AudioFrame {
            planes: Vec::new(),
            nb_samples: 0,
            sample_rate: 0,
            format: SampleFormat::U8,
            ch_layout: ChannelLayout::default(),
            pts: crate::NOPTS,
            duration: 0,
            time_base: Rational::UNKNOWN,
        }
    }
}

impl AudioFrame {
    /// `ch_layout.nb_channels` (C removed the duplicate `channels` field).
    pub fn channels(&self) -> usize {
        self.ch_layout.nb_channels
    }

    /// Number of planes: 1 (interleaved) or one per channel (planar).
    pub fn nb_planes(&self) -> usize {
        if self.format.is_planar() {
            self.channels()
        } else {
            1
        }
    }

    /// `frame.c` `get_audio_buffer` (align 0): per-plane OWN buffers,
    /// silence-filled. The planes do not share an `Arc` — each `data[]`
    /// entry is its own allocation in C too.
    pub fn alloc(
        format: SampleFormat,
        ch_layout: ChannelLayout,
        nb_samples: usize,
    ) -> Result<AudioFrame, crate::util::error::Error> {
        let arrays = samplefmt::samples_fill_arrays(ch_layout.nb_channels, nb_samples, format, 0)?;
        // The auto-align (align=0) rounds the SAMPLE count to 32; every
        // plane gets the same linesize (c's per-channel line_size).
        let fill = samplefmt::silence_byte(format);
        let planes = arrays
            .plane_offsets
            .iter()
            .map(|_| {
                let buf: Arc<[u8]> = Arc::from(vec![fill; arrays.linesize]);
                Plane {
                    buf,
                    offset: 0,
                    linesize: arrays.linesize,
                    rows: 1,
                }
            })
            .collect();
        Ok(AudioFrame {
            planes,
            nb_samples,
            sample_rate: 0,
            format,
            ch_layout,
            pts: crate::NOPTS,
            duration: 0,
            time_base: Rational::UNKNOWN,
        })
    }

    /// `av_samples_fill_arrays` over one caller-owned buffer (align 1: the
    /// offsets are exact, no 32-rounding). The buffer's length must equal
    /// the computed size for `nb_samples`.
    pub fn wrap_buffer(
        buf: Arc<[u8]>,
        format: SampleFormat,
        ch_layout: ChannelLayout,
        nb_samples: usize,
    ) -> Result<AudioFrame, crate::util::error::Error> {
        let arrays = samplefmt::samples_fill_arrays(ch_layout.nb_channels, nb_samples, format, 1)?;
        if buf.len() < arrays.buf_size {
            return Err(crate::util::error::Error::BufferTooSmall);
        }
        let planes = arrays
            .plane_offsets
            .iter()
            .map(|&off| Plane {
                buf: Arc::clone(&buf),
                offset: off,
                linesize: arrays.linesize,
                rows: 1,
            })
            .collect();
        Ok(AudioFrame {
            planes,
            nb_samples,
            sample_rate: 0,
            format,
            ch_layout,
            pts: crate::NOPTS,
            duration: 0,
            time_base: Rational::UNKNOWN,
        })
    }

    /// The bytes of plane `i` (interleaved: the whole frame; planar: one
    /// channel's samples).
    pub fn plane(&self, i: usize) -> &[u8] {
        self.planes[i].data()
    }

    pub fn plane_mut(&mut self, i: usize) -> &mut [u8] {
        self.planes[i].data_mut()
    }

    /// `av_frame_copy_props` subset: everything but the samples.
    pub fn copy_props(&mut self, src: &AudioFrame) {
        self.pts = src.pts;
        self.duration = src.duration;
        self.time_base = src.time_base;
        self.sample_rate = src.sample_rate;
        // ch_layout/format describe the buffer shape — NOT copied (C's
        // copy_props skips every format-describing field).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::channel_layout::ChannelLayout;

    fn stereo() -> ChannelLayout {
        ChannelLayout::from_string("stereo").unwrap()
    }

    #[test]
    fn alloc_packed_shape() {
        // s16 interleaved stereo, 10 samples: align=0 rounds to 32
        // samples → linesize 32·2·2 = 128, one plane, silence 0x00.
        let f = AudioFrame::alloc(SampleFormat::S16, stereo(), 10).unwrap();
        assert_eq!(f.nb_planes(), 1);
        assert_eq!(f.plane(0).len(), 128);
        assert!(f.plane(0).iter().all(|&b| b == 0));
        assert_eq!(f.channels(), 2);
        assert_eq!(f.pts, crate::NOPTS);
    }

    #[test]
    fn alloc_planar_shape() {
        // s16p stereo, 10 samples → 2 planes of 32·2 = 64 bytes each.
        let f = AudioFrame::alloc(SampleFormat::S16p, stereo(), 10).unwrap();
        assert_eq!(f.nb_planes(), 2);
        assert_eq!(f.plane(0).len(), 64);
        assert_eq!(f.plane(1).len(), 64);
    }

    #[test]
    fn alloc_u8_silence_is_midpoint() {
        let f = AudioFrame::alloc(SampleFormat::U8, stereo(), 5).unwrap();
        assert!(f.plane(0).iter().all(|&b| b == 0x80));
    }

    #[test]
    fn wrap_buffer_shared_arc_exact_offsets() {
        // 4 samples s16 stereo packed = 16 bytes; align=1 → no rounding.
        let payload: Arc<[u8]> =
            Arc::from((0u16..8).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let f =
            AudioFrame::wrap_buffer(Arc::clone(&payload), SampleFormat::S16, stereo(), 4).unwrap();
        assert_eq!(f.nb_planes(), 1);
        assert_eq!(f.plane(0).len(), 16);
        assert_eq!(f.plane(0), &payload[..]);

        // planar wrap: one shared Arc, channel planes at 0 / linesize
        // (4 samples s16p stereo = 2 planes x 8 bytes).
        let pl: Arc<[u8]> = Arc::from([0u8; 16]);
        let f = AudioFrame::wrap_buffer(Arc::clone(&pl), SampleFormat::S16p, stereo(), 4).unwrap();
        assert_eq!(f.nb_planes(), 2);
        assert!(std::sync::Arc::ptr_eq(&f.planes[0].buf, &f.planes[1].buf));
        assert_eq!(f.planes[1].offset, 8);
        assert_eq!(f.plane(0).len(), 8);
    }

    #[test]
    fn wrap_buffer_rejects_short_buffers() {
        let short: Arc<[u8]> = Arc::from([0u8; 4]);
        assert!(AudioFrame::wrap_buffer(short, SampleFormat::S16, stereo(), 4).is_err());
    }

    #[test]
    fn default_is_the_unconfigured_shape() {
        let f = AudioFrame::default();
        assert_eq!(f.nb_samples, 0);
        assert_eq!(f.channels(), 0);
        assert_eq!(f.nb_planes(), 1); // interleaved shape, zero planes held
    }
}
