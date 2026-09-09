//! Video frames — port of `libavutil/frame.{h,c}` (video subset).
//!
//! ## The C model and its Rust translation
//!
//! An `AVFrame` points into refcounted buffers: `data[8]` plane pointers,
//! `linesize[8]` row strides, and `buf[8]` the owning `AVBufferRef`s, with
//! the rule that a frame is *writable iff exactly one reference exists*
//! (`av_frame_make_writable` is the copy-on-write gate).
//!
//! Rust replaces the pointer soup with ownership:
//!
//! | C | Rust |
//! |---|---|
//! | `AVBufferRef` + `data[i]` pointer + `linesize[i]` | [`Plane`] = Arc buffer + offset + linesize |
//! | `av_frame_ref` (bump refcount) | `frame.clone()` (shallow, Arc bumps) |
//! | `av_buffer_is_writable` | [`Frame::is_writable`] (`Arc::strong_count == 1`) |
//! | `av_frame_make_writable` | [`Frame::make_writable`] (re-buffers shared planes) |
//! | `av_frame_get_buffer` | [`Frame::alloc`] (compact, align 1) |
//!
//! Zero-copy: [`Frame::wrap_buffer`] attaches a packet's `Arc<[u8]>` as the
//! plane backing store — the rawvideo decoder's `av_buffer_ref(avpkt->buf)`
//! path. Multi-plane formats share *one* Arc with per-plane offsets, mirroring
//! how C's single `buf[0]` allocation backs all planes of a raw frame.
//!
//! ## Hard invariants (differences from C, deliberate)
//!
//! * **Linesizes are positive** — no vertically-flipped images via negative
//!   stride (C only produces those from `BottomUp` rawvideo codec tags,
//!   which are not ported).
//! * **No `extended_data`/`extended_buf`** — audio planar layouts arrive in
//!   the audio phase.
//! * **`key_frame` does not exist** — FFmpeg 8 replaced it with
//!   `flags & AV_FRAME_FLAG_KEY`, mirrored by [`FrameFlags::KEY`].
//! * pts is `i64` in `time_base` units; [`crate::util::NOPTS`] when unknown.

use std::sync::Arc;

use super::{
    color::{ChromaLocation, ColorPrimaries, ColorRange, ColorSpace, ColorTrc},
    error::{Error, Result},
    imgutils,
    pixfmt::PixelFormat,
    rational::Rational,
};

/// `AV_NOPTS_VALUE` re-exported for frame users.
pub use super::NOPTS;

/// `enum AVPictureType` (subset).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PictureType {
    /// Undetermined / not set.
    #[default]
    None,
    /// Intra: decoded without references.
    I,
    /// Inter: predicted from previous frames.
    P,
    /// Bidirectionally predicted.
    B,
}

/// `AV_FRAME_FLAG_*` (`frame.h:683`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameFlags(pub u32);

impl FrameFlags {
    /// `AV_FRAME_FLAG_KEY` (1 << 1) — a key frame. Replaces the removed
    /// `key_frame` field of FFmpeg ≤ 7.
    pub const KEY: FrameFlags = FrameFlags(1 << 1);
    /// `AV_FRAME_FLAG_CORRUPT` (1 << 0).
    pub const CORRUPT: FrameFlags = FrameFlags(1 << 0);
    /// `AV_FRAME_FLAG_INTERLACED` (1 << 3).
    pub const INTERLACED: FrameFlags = FrameFlags(1 << 3);

    pub const fn contains(self, other: FrameFlags) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn union(self, other: FrameFlags) -> FrameFlags {
        FrameFlags(self.0 | other.0)
    }
}

/// Row count of plane `i`: planes 1 and 2 (chroma) are subsampled by
/// `2^chroma_h` with ceiling, everything else is full height.
#[inline]
fn plane_rows(i: usize, height: u32, chroma_h: u32) -> usize {
    if i == 1 || i == 2 {
        ((height + (1 << chroma_h) - 1) >> chroma_h) as usize
    } else {
        height as usize
    }
}

/// One plane: a window into a refcounted buffer.
///
/// The visible slice is `buf[offset .. offset + linesize * rows]`. Planes of
/// one frame usually share the same `Arc` at different offsets.
#[derive(Debug, Clone)]
pub struct Plane {
    pub buf: Arc<[u8]>,
    pub offset: usize,
    pub linesize: usize,
    /// Height of this plane in rows (chroma planes of 4:2:0 have half).
    pub rows: usize,
}

impl Plane {
    /// The full contiguous plane contents (row `y` at `y * linesize`).
    pub fn data(&self) -> &[u8] {
        &self.buf[self.offset..self.offset + self.linesize * self.rows]
    }

    /// The mutable plane contents — copy-on-write: a shared buffer is
    /// reallocated privately first (the per-plane slice of make_writable).
    pub fn data_mut(&mut self) -> &mut [u8] {
        if std::sync::Arc::strong_count(&self.buf) > 1 {
            let fresh: std::sync::Arc<[u8]> = std::sync::Arc::from(self.data().to_vec());
            self.buf = fresh;
            self.offset = 0;
        }
        let end = self.offset + self.linesize * self.rows;
        &mut std::sync::Arc::get_mut(&mut self.buf).expect("CoW above") [self.offset..end]
    }
}
/// `AVFrame` — video subset.
#[derive(Debug)]
pub struct Frame {
    /// Plane descriptors, `imgutils::count_planes` many.
    pub planes: Vec<Plane>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub pict_type: PictureType,
    pub sample_aspect_ratio: Rational,
    /// Presentation timestamp in `time_base` units ([`NOPTS`] if unknown).
    pub pts: i64,
    /// Unit of `pts`/`duration`.
    pub time_base: Rational,
    pub duration: i64,
    pub flags: FrameFlags,
    pub color_range: ColorRange,
    pub color_primaries: ColorPrimaries,
    pub color_trc: ColorTrc,
    pub color_space: ColorSpace,
    pub chroma_location: ChromaLocation,
    pub crop_top: u32,
    pub crop_bottom: u32,
    pub crop_left: u32,
    pub crop_right: u32,
}

impl Default for Frame {
    /// The "unallocated" frame, like `av_frame_alloc()`'s zeroed result.
    fn default() -> Self {
        Frame {
            planes: Vec::new(),
            width: 0,
            height: 0,
            format: PixelFormat::Gray8, // meaningless until planes exist
            pict_type: PictureType::None,
            sample_aspect_ratio: Rational::UNKNOWN,
            pts: NOPTS,
            time_base: Rational::UNKNOWN,
            duration: 0,
            flags: FrameFlags(0),
            color_range: ColorRange::Unspecified,
            color_primaries: ColorPrimaries::Unspecified,
            color_trc: ColorTrc::Unspecified,
            color_space: ColorSpace::Unspecified,
            chroma_location: ChromaLocation::Unspecified,
            crop_top: 0,
            crop_bottom: 0,
            crop_left: 0,
            crop_right: 0,
        }
    }
}

impl Frame {
    /// `av_frame_get_buffer(align=1)` — allocate a compact frame with exact
    /// (unpadded) linesizes.
    ///
    /// Each plane gets its own buffer (so a fresh frame is trivially
    /// writable, like C's per-plane `buf[i]`), unlike [`Frame::wrap_buffer`]
    /// where one incoming buffer backs every plane.
    pub fn alloc(format: PixelFormat, width: u32, height: u32) -> Result<Frame> {
        imgutils::check_size(width, height)?;
        let linesizes = imgutils::fill_linesizes(format, width)?;
        let sizes = imgutils::fill_plane_sizes(format, height, &linesizes)?;

        let mut frame = Frame::default();
        frame.format = format;
        frame.width = width;
        frame.height = height;
        frame.planes.reserve(4);

        let nb_planes = super::pixdesc::count_planes(format);
        let chroma_h = super::pixdesc::descriptor(format).log2_chroma_h as u32;
        for i in 0..nb_planes {
            frame.planes.push(Plane {
                buf: Arc::from(vec![0u8; sizes[i]]),
                offset: 0,
                linesize: linesizes[i],
                rows: plane_rows(i, height, chroma_h),
            });
        }
        Ok(frame)
    }

    /// `av_image_fill_arrays` + `av_frame_ref` combined: adopt an existing
    /// compact buffer (e.g. packet data straight off the demuxer) as the
    /// frame's storage. Zero copies, zero allocations.
    pub fn wrap_buffer(
        buf: Arc<[u8]>,
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Result<Frame> {
        imgutils::check_size(width, height)?;
        let linesizes = imgutils::fill_linesizes(format, width)?;
        let sizes = imgutils::fill_plane_sizes(format, height, &linesizes)?;
        let total: usize = sizes
            .iter()
            .try_fold(0usize, |a, &s| a.checked_add(s))
            .ok_or(Error::OutOfRange)?;
        if buf.len() < total {
            return Err(Error::InvalidData(format!(
                "buffer of {} bytes too small for {}x{} {} (needs {total})",
                buf.len(),
                width,
                height,
                format.name()
            )));
        }

        let mut frame = Frame::default();
        frame.format = format;
        frame.width = width;
        frame.height = height;

        let nb_planes = super::pixdesc::count_planes(format);
        let chroma_h = super::pixdesc::descriptor(format).log2_chroma_h as u32;
        let mut offset = 0;
        for i in 0..nb_planes {
            frame.planes.push(Plane {
                buf: buf.clone(),
                offset,
                linesize: linesizes[i],
                rows: plane_rows(i, height, chroma_h),
            });
            offset += sizes[i];
        }
        Ok(frame)
    }

    /// Row stride of plane `i` (`frame->linesize[i]`).
    pub fn linesize(&self, plane: usize) -> usize {
        self.planes[plane].linesize
    }

    /// Borrow plane `i` as a flat slice (rows concatenated at `linesize`).
    pub fn plane(&self, plane: usize) -> &[u8] {
        self.planes[plane].data()
    }

    /// Mutable borrow of plane `i`; panics if the plane is shared
    /// (call [`Frame::make_writable`] first — mirrors the C contract where
    /// writing through a shared buffer is UB instead of a panic).
    pub fn plane_mut(&mut self, plane: usize) -> &mut [u8] {
        let p = &mut self.planes[plane];
        let start = p.offset;
        let end = p.offset + p.linesize * p.rows;
        let buf = Arc::get_mut(&mut p.buf).expect("plane is not writable (shared buffer)");
        &mut buf[start..end]
    }

    /// `av_frame_is_writable` — true iff every plane's buffer is uniquely
    /// referenced by this frame.
    pub fn is_writable(&self) -> bool {
        self.planes.iter().all(|p| Arc::strong_count(&p.buf) == 1)
    }

    /// `av_frame_make_writable` — copy-on-write: for each plane whose buffer
    /// is shared (a cloned frame, or a wrapped packet buffer), reallocate it
    /// privately and copy the contents.
    pub fn make_writable(&mut self) -> Result<()> {
        for plane in &mut self.planes {
            if Arc::strong_count(&plane.buf) > 1 {
                let fresh: Arc<[u8]> = Arc::from(plane.data().to_vec());
                plane.buf = fresh;
                plane.offset = 0;
            }
        }
        Ok(())
    }

    /// `ff_decode_frame_props` analog used by decoders: copy the timestamp
    /// fields from a packet onto the frame (pts/duration/time_base and the
    /// KEY flag — C sets `AV_FRAME_FLAG_KEY` from `AV_PKT_FLAG_KEY`).
    pub fn copy_packet_props(&mut self, pkt: &crate::codec::packet::Packet) {
        self.pts = pkt.pts;
        self.duration = pkt.duration;
        self.time_base = pkt.time_base;
        if pkt.flags.contains(crate::codec::packet::PacketFlags::KEY) {
            self.flags = self.flags.union(FrameFlags::KEY);
        }
    }

    /// `av_frame_copy_props` (frame.c) — copy every metadata field `Frame`
    /// carries from `src` onto `self`, leaving the pixel data and geometry
    /// (`planes`/`width`/`height`/`format`) untouched. Callers that override
    /// a field (e.g. scale stamping the negotiated color metadata) do so
    /// AFTER this call.
    ///
    /// Covers: pts, duration, time_base, pict_type, flags,
    /// sample_aspect_ratio, crop_*, color_range, color_primaries, color_trc,
    /// color_space, chroma_location — everything but the data/geometry
    /// fields. Not ported from C: the side-data, metadata dictionary and
    /// `AV_FRAME_FLAG_*` fields outside our subset (none exist here).
    pub fn copy_props(&mut self, src: &Frame) {
        self.pts = src.pts;
        self.duration = src.duration;
        self.time_base = src.time_base;
        self.pict_type = src.pict_type;
        self.flags = src.flags;
        self.sample_aspect_ratio = src.sample_aspect_ratio;
        self.crop_top = src.crop_top;
        self.crop_bottom = src.crop_bottom;
        self.crop_left = src.crop_left;
        self.crop_right = src.crop_right;
        self.color_range = src.color_range;
        self.color_primaries = src.color_primaries;
        self.color_trc = src.color_trc;
        self.color_space = src.color_space;
        self.chroma_location = src.chroma_location;
    }
}

impl Clone for Frame {
    /// `av_frame_clone` — shallow: bumps plane Arc refcounts. Cheap.
    fn clone(&self) -> Self {
        Frame {
            planes: self.planes.clone(),
            width: self.width,
            height: self.height,
            format: self.format,
            pict_type: self.pict_type,
            sample_aspect_ratio: self.sample_aspect_ratio,
            pts: self.pts,
            time_base: self.time_base,
            duration: self.duration,
            flags: self.flags,
            color_range: self.color_range,
            color_primaries: self.color_primaries,
            color_trc: self.color_trc,
            color_space: self.color_space,
            chroma_location: self.chroma_location,
            crop_top: self.crop_top,
            crop_bottom: self.crop_bottom,
            crop_left: self.crop_left,
            crop_right: self.crop_right,
        }
    }
}
