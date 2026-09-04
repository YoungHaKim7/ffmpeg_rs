//! The Vulkan swscale backend — port of `libswscale/vulkan/` +
//! `vf_scale_vulkan.c`'s execution shape: per frame, upload the source
//! planes to images, dispatch one compute pass, read the result back.
//!
//! `vf_scale_vulkan.c` keeps a frame pool and records into a per-frame
//! command buffer from an `FFVkExecPool`; Phase 2 is the simple correct
//! version — one one-shot command buffer per frame, blocking fence — with
//! images cached per geometry so steady state only allocates staging.
//! (Overlapping frames in flight is a later phase.)
//!
//! The shader (`assets/scale.comp`, see [`crate::shaders`]) implements the
//! exact CPU kernels; the engine-consistency test in `tests/golden.rs` pins
//! the two against each other.

use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{DescriptorImageInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};

use crate::gpu::{self, ComputeGpu};
use crate::shaders::scale_cs;
use crate::util::color::{ChromaLocation, ColorRange};
use crate::util::error::{Error, Result};
use crate::util::frame::Frame;
use crate::util::pixdesc;

use super::{rgb_offsets, ConversionMode, ScaleAlgorithm};

/// Per-geometry image set; rebuilt when the frame geometry changes.
struct Images {
    /// Source planes (SAMPLED + TRANSFER_DST), R8: luma then chroma.
    src: [Arc<ImageView>; 3],
    /// Planar outputs (STORAGE + TRANSFER_SRC), R8: dst luma + dst chroma.
    out_y: Arc<ImageView>,
    out_u: Arc<ImageView>,
    out_v: Arc<ImageView>,
    /// Packed-RGB output (STORAGE + TRANSFER_SRC), RGBA8.
    out_rgb: Arc<ImageView>,
    src_geom: (u32, u32),
    dst_geom: (u32, u32),
}

impl Images {
    /// Output plane views for the planar modes (luma, chroma, chroma).
    fn out_planes(&self) -> [Arc<ImageView>; 3] {
        [self.out_y.clone(), self.out_u.clone(), self.out_v.clone()]
    }
}

/// `FFVkExecPool` + pipeline + image pool for one scaling session.
pub struct GpuScaler {
    gpu: Arc<ComputeGpu>,
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    sampler: Arc<Sampler>,
    /// 1×1 R8 stand-in for bindings the current mode does not use.
    dummy: Arc<ImageView>,
    images: Option<Images>,
    device_name: String,
}

impl GpuScaler {
    /// Create the device + pipeline. Fails cleanly (`Unsupported`/`NotFound`)
    /// so `ScaleEngine::Auto` can fall back to the CPU kernels.
    pub fn new() -> Result<Self> {
        let gpu = Arc::new(ComputeGpu::new()?);
        let entry = unsafe { scale_cs::load(&gpu.device) }
            .map_err(|e| Error::Unsupported(format!("compiling scale.comp: {e}")))?
            .entry_point("main")
            .ok_or_else(|| Error::Unsupported("scale.comp has no main".into()))?;
        let stage = vulkano::pipeline::PipelineShaderStageCreateInfo::new(&entry);
        let layout =
            PipelineLayout::from_stages(&gpu.device, std::slice::from_ref(&stage))
                .map_err(|e| Error::Unsupported(format!("pipeline layout: {e}")))?;
        let pipeline = ComputePipeline::new(
            &gpu.device,
            None,
            &ComputePipelineCreateInfo::new(stage, &layout),
        )
        .map_err(|e| Error::Unsupported(format!("compute pipeline: {e}")))?;

        let sampler = Sampler::new(
            &gpu.device,
            &SamplerCreateInfo {
                // texelFetch never filters; the sampler is only a formality
                // (GLSL texture2D+sampler combination requires one).
                ..Default::default()
            },
        )
        .map_err(|e| Error::Unsupported(format!("sampler: {e}")))?;

        let dummy = plane_image(&gpu.memory_allocator, Format::R8_UNORM, 1, 1, gpu::usage::PLANE_OUT)?;

        let device_name = gpu.device_name.clone();
        crate::log_verbose!(None, "swscale: using Vulkan device '{device_name}'");
        Ok(GpuScaler { gpu, pipeline, layout, sampler, dummy, images: None, device_name })
    }

    /// The picked device name, for the CLI banner.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// `sws_scale` on the GPU. Mirrors `ScaleContext::scale_cpu` mode for
    /// mode; range/siting follow the SOURCE frame like swscale does.
    pub fn scale(
        &mut self,
        src: &Frame,
        dst: &mut Frame,
        mode: ConversionMode,
        algorithm: ScaleAlgorithm,
    ) -> Result<()> {
        self.ensure_images(src, dst, mode)?;

        let full_range = src.color_range == ColorRange::Jpeg;
        let center_siting = src.chroma_location == ChromaLocation::Center;
        let src_chroma = (src.width.div_ceil(2), src.height.div_ceil(2));

        let push = scale_cs::Push {
            src_size: [src.width as i32, src.height as i32],
            dst_size: [dst.width as i32, dst.height as i32],
            chroma_size: [src_chroma.0 as i32, src_chroma.1 as i32],
            mode: shader_mode(mode),
            algorithm: shader_algorithm(algorithm),
            full_range: full_range as i32,
            center_siting: center_siting as i32,
        };

        let mut builder = self.gpu.builder()?;
        let images = self.images.as_ref().expect("ensure_images ran");

        // ---- upload ---------------------------------------------------------
        // Frames in this pipeline are compact (linesize == width), so the
        // staging buffer is the plane bytes as-is.
        match mode {
            ConversionMode::Yuv420pToRgb | ConversionMode::Yuv420pToYuv420p => {
                for p in 0..3 {
                    copy_plane_to_image(&mut builder, &self.gpu, src, p, &images.src[p])?;
                }
            }
            ConversionMode::Gray8ToRgb | ConversionMode::Gray8ToGray8 => {
                copy_plane_to_image(&mut builder, &self.gpu, src, 0, &images.src[0])?;
            }
        }

        // ---- dispatch ---------------------------------------------------------
        let set = self.descriptor_set(images, mode)?;
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .map_err(|e| Error::Unsupported(format!("bind pipeline: {e}")))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.layout.clone(), 0, set)
            .map_err(|e| Error::Unsupported(format!("bind descriptor set: {e}")))?
            .push_constants(self.layout.clone(), 0, push)
            .map_err(|e| Error::Unsupported(format!("push constants: {e}")))?;
        let groups = [dst.width.div_ceil(8), dst.height.div_ceil(8), 1];
        unsafe { builder.dispatch(groups) }.map_err(|e| Error::Unsupported(format!("dispatch: {e}")))?;

        // ---- download ---------------------------------------------------------
        match mode {
            ConversionMode::Yuv420pToRgb | ConversionMode::Gray8ToRgb => {
                let w4 = dst.width as usize * 4;
                let buf = gpu::readback_buffer(&self.gpu.memory_allocator, w4 * dst.height as usize)?;
                builder
                    .copy_image_to_buffer(CopyImageToBufferInfo::new(images.out_rgb.image().clone(), buf.clone()))
                    .map_err(|e| Error::Unsupported(format!("copy to buffer: {e}")))?;
                self.gpu.submit_wait(builder)?;
                pack_rgba_to_dst(&buf, dst)
            }
            ConversionMode::Yuv420pToYuv420p | ConversionMode::Gray8ToGray8 => {
                let out_planes = images.out_planes();
                let nb = if mode == ConversionMode::Yuv420pToYuv420p { 3 } else { 1 };
                let mut bufs: Vec<(Subbuffer<[u8]>, usize, usize)> = Vec::with_capacity(nb);
                for view in out_planes.iter().take(nb) {
                    let ext = view.image().extent();
                    let (w, h) = (ext[0] as usize, ext[1] as usize);
                    let buf = gpu::readback_buffer(&self.gpu.memory_allocator, w * h)?;
                    builder
                        .copy_image_to_buffer(CopyImageToBufferInfo::new(view.image().clone(), buf.clone()))
                        .map_err(|e| Error::Unsupported(format!("copy to buffer: {e}")))?;
                    bufs.push((buf, w, h));
                }
                self.gpu.submit_wait(builder)?;
                for (p, (buf, w, h)) in bufs.iter().enumerate() {
                    let data = buf.read().map_err(|e| Error::Unsupported(format!("readback: {e}")))?;
                    let data: &[u8] = &data[..];
                    let ls = dst.linesize(p);
                    let out = dst.plane_mut(p);
                    for y in 0..*h {
                        out[y * ls..y * ls + w].copy_from_slice(&data[y * w..(y + 1) * w]);
                    }
                }
                Ok(())
            }
        }
    }

    fn descriptor_set(&self, images: &Images, mode: ConversionMode) -> Result<Arc<DescriptorSet>> {
        let planar_out = matches!(mode, ConversionMode::Yuv420pToYuv420p | ConversionMode::Gray8ToGray8);
        let gray = matches!(mode, ConversionMode::Gray8ToRgb | ConversionMode::Gray8ToGray8);
        let dummy = &self.dummy;
        let (y_in, u_in, v_in) = if gray {
            (&images.src[0], dummy, dummy)
        } else {
            (&images.src[0], &images.src[1], &images.src[2])
        };
        let out_y = if planar_out { &images.out_y } else { dummy };
        let out_u = if planar_out { &images.out_u } else { dummy };
        let out_v = if planar_out { &images.out_v } else { dummy };
        let out_rgb = if planar_out { dummy } else { &images.out_rgb };

        // This vulkano rev writes every image-ish descriptor (sampled,
        // storage, or plain sampler) through DescriptorImageInfo.
        let infos = [
            DescriptorImageInfo { image_view: Some(y_in), ..Default::default() },
            DescriptorImageInfo { image_view: Some(u_in), ..Default::default() },
            DescriptorImageInfo { image_view: Some(v_in), ..Default::default() },
            DescriptorImageInfo { image_view: Some(out_y), ..Default::default() },
            DescriptorImageInfo { image_view: Some(out_u), ..Default::default() },
            DescriptorImageInfo { image_view: Some(out_v), ..Default::default() },
            DescriptorImageInfo { image_view: Some(out_rgb), ..Default::default() },
            DescriptorImageInfo { sampler: Some(&self.sampler), image_view: None, ..Default::default() },
        ];
        let writes: Vec<WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| WriteDescriptorSet::image(binding as u32, info))
            .collect();
        DescriptorSet::new(
            &self.gpu.descriptor_set_allocator,
            self.pipeline.layout().set_layouts().first().expect("scale.comp has set 0"),
            &writes,
            &[],
        )
        .map_err(|e| Error::Unsupported(format!("descriptor set: {e}")))
    }

    fn ensure_images(&mut self, src: &Frame, dst: &Frame, _mode: ConversionMode) -> Result<()> {
        let src_geom = (src.width, src.height);
        let dst_geom = (dst.width, dst.height);
        if let Some(imgs) = &self.images {
            if imgs.src_geom == src_geom && imgs.dst_geom == dst_geom {
                return Ok(());
            }
        }
        let (cw, ch) = (src.width.div_ceil(2), src.height.div_ceil(2));
        let (dcw, dch) = (dst.width.div_ceil(2), dst.height.div_ceil(2));
        let mk = |fmt, w, h, usage| plane_image(&self.gpu.memory_allocator, fmt, w, h, usage);
        self.images = Some(Images {
            src: [
                mk(Format::R8_UNORM, src.width, src.height, gpu::usage::PLANE_IN)?,
                mk(Format::R8_UNORM, cw, ch, gpu::usage::PLANE_IN)?,
                mk(Format::R8_UNORM, cw, ch, gpu::usage::PLANE_IN)?,
            ],
            out_y: mk(Format::R8_UNORM, dst.width, dst.height, gpu::usage::PLANE_OUT)?,
            out_u: mk(Format::R8_UNORM, dcw, dch, gpu::usage::PLANE_OUT)?,
            out_v: mk(Format::R8_UNORM, dcw, dch, gpu::usage::PLANE_OUT)?,
            out_rgb: mk(Format::R8G8B8A8_UNORM, dst.width, dst.height, gpu::usage::PLANE_OUT)?,
            src_geom,
            dst_geom,
        });
        Ok(())
    }
}

/// `ConversionMode` → the shader's mode int (kept in one place).
fn shader_mode(mode: ConversionMode) -> i32 {
    match mode {
        ConversionMode::Yuv420pToRgb => 0,
        ConversionMode::Gray8ToRgb => 1,
        ConversionMode::Yuv420pToYuv420p => 2,
        ConversionMode::Gray8ToGray8 => 3,
    }
}

fn shader_algorithm(alg: ScaleAlgorithm) -> i32 {
    match alg {
        ScaleAlgorithm::Nearest => 0,
        ScaleAlgorithm::Bilinear => 1,
        ScaleAlgorithm::Bicubic => 2,
    }
}

fn copy_plane_to_image(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    gpu: &Arc<ComputeGpu>,
    src: &Frame,
    p: usize,
    view: &Arc<ImageView>,
) -> Result<()> {
    let staging = gpu::upload_buffer(&gpu.memory_allocator, src.plane(p))?;
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::new(staging, view.image().clone()))
        .map_err(|e| Error::Unsupported(format!("copy to image: {e}")))?;
    Ok(())
}

/// RGBA8 readback → the caller's packed-RGB frame layout (rgb24/bgr24/…).
fn pack_rgba_to_dst(buf: &Subbuffer<[u8]>, dst: &mut Frame) -> Result<()> {
    let data = buf.read().map_err(|e| Error::Unsupported(format!("readback: {e}")))?;
    let data: &[u8] = &data[..];
    let w4 = dst.width as usize * 4;
    let (r_off, g_off, b_off) = rgb_offsets(dst.format);
    let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
    let ls = dst.linesize(0);
    let dh = dst.height as usize;
    let out = dst.plane_mut(0);
    for y in 0..dh {
        let row = &data[y * w4..(y + 1) * w4];
        let out_row = &mut out[y * ls..];
        for (x, px) in row.chunks_exact(4).enumerate() {
            out_row[x * step + r_off] = px[0];
            out_row[x * step + g_off] = px[1];
            out_row[x * step + b_off] = px[2];
        }
    }
    Ok(())
}

/// `Image` + default view, device-preferred memory (house recipe).
fn plane_image(
    allocator: &Arc<StandardMemoryAllocator>,
    format: Format,
    w: u32,
    h: u32,
    usage: ImageUsage,
) -> Result<Arc<ImageView>> {
    ImageView::new_default(
        &Image::new(
            allocator,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [w, h, 1],
                usage,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .map_err(|e| Error::Unsupported(format!("image {w}x{h}: {e}")))?,
    )
    .map_err(|e| Error::Unsupported(format!("image view: {e}")))
}
