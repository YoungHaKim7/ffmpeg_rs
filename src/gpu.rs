//! Headless Vulkan compute device — the `libavutil/vulkan.c`
//! (`FFVulkanContext`) analog, following the `HeadlessGpu` skeleton of the
//! sibling Navier-Stokes project: instance without surface extensions (so
//! it also works with no display server), a COMPUTE queue family, and the
//! three standard allocators. FFmpeg's queue picking (`ff_vk_exec_pool`)
//! prefers transfer/compute too; here one general compute queue is enough
//! for swscale's single dispatch per frame.
//!
//! Only creation differs from the sibling: we *must* be able to use
//! `R8_UNORM` and `R8G8B8A8_UNORM` images as both sampled and storage
//! (checked per device, `ff_vk_map_usage_to_feats`'s spirit), because the
//! scale shader reads planes as `texture2D` and writes `rgba8`/`r8`
//! storage images.

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    allocator::StandardCommandBufferAllocator,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{
    Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    physical::{PhysicalDevice, PhysicalDeviceType},
};
use vulkano::format::{Format, FormatFeatures};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

use crate::util::error::{Error, Result};

/// Everything a compute-only run needs; no winit, no surface.
pub struct ComputeGpu {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// For diagnostics (av_dump-style "Using device: …" messages).
    pub device_name: String,
}

impl ComputeGpu {
    /// Enumerate devices, pick the best compute-capable one that supports
    /// the formats the scale shader needs.
    pub fn new() -> Result<Self> {
        let library = unsafe { VulkanLibrary::new() }
            .map_err(|e| Error::Unsupported(format!("loading libvulkan: {e}")))?;
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .map_err(|e| Error::Unsupported(format!("creating Vulkan instance: {e}")))?;

        // Preference order like the sibling projects: discrete first.
        let candidates = instance
            .enumerate_physical_devices()
            .map_err(|e| Error::Unsupported(format!("enumerating devices: {e}")))?;
        let mut ranked: Vec<_> = candidates
            .filter(|p| p.queue_family_properties().iter().any(|q| q.queue_flags.intersects(QueueFlags::COMPUTE)))
            .filter(|p| Self::supports_shader_formats(p))
            .map(|p| {
                let rank = match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                };
                (p, rank)
            })
            .collect();
        ranked.sort_by_key(|(_, rank)| *rank);
        let (physical_device, _) = ranked
            .into_iter()
            .next()
            .ok_or_else(|| Error::NotFound("compute-capable Vulkan device with R8/RGBA8 image support".into()))?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
            .expect("filtered above") as u32;

        let device_name = physical_device.properties().device_name.clone();
        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| Error::Unsupported(format!("creating Vulkan device: {e}")))?;
        let queue = queues.next().expect("one queue was requested");

        Ok(ComputeGpu {
            memory_allocator: Arc::new(StandardMemoryAllocator::new(
                &device,
                &Default::default(),
            )),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                &device,
                &Default::default(),
            )),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                &device,
                &Default::default(),
            )),
            device,
            queue,
            device_name,
        })
    }

    /// R8 must be sampleable AND storable (plane in/out), RGBA8 storable
    /// (packed RGB out) — `optimal` tiling, the default for `Image::new`.
    fn supports_shader_formats(p: &PhysicalDevice) -> bool {
        let r8 = p.format_properties(Format::R8_UNORM).optimal_tiling_features;
        let rgba = p.format_properties(Format::R8G8B8A8_UNORM).optimal_tiling_features;
        let need = FormatFeatures::SAMPLED_IMAGE | FormatFeatures::STORAGE_IMAGE;
        r8.intersects(need) && rgba.intersects(FormatFeatures::STORAGE_IMAGE)
    }

    /// One-shot primary command buffer (house pattern).
    pub fn builder(&self) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| Error::Unsupported(format!("allocating command buffer: {e}")))
    }

    /// Submit and block until done (frame-synchronous Phase 2; a
    /// queue of frames with fences is a later optimization).
    pub fn submit_wait(&self, builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> Result<()> {
        let command_buffer = builder
            .build()
            .map_err(|e| Error::Unsupported(format!("building command buffer: {e}")))?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| Error::Unsupported(format!("submitting: {e:?}")))?
            .then_signal_fence_and_flush()
            .map_err(|e| Error::Unsupported(format!("flushing: {e:?}")))?
            .wait(None)
            .map_err(|e| Error::Unsupported(format!("waiting for fence: {e:?}")))
    }
}

/// Staging upload buffer (house recipe: HOST | SEQUENTIAL_WRITE).
pub fn upload_buffer(allocator: &Arc<StandardMemoryAllocator>, data: &[u8]) -> Result<vulkano::buffer::Subbuffer<[u8]>> {
    vulkano::buffer::Buffer::from_iter(
        allocator,
        &vulkano::buffer::BufferCreateInfo {
            usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .map_err(|e| Error::Unsupported(format!("upload buffer: {e}")))
}

/// Readback buffer (HOST | RANDOM_ACCESS).
pub fn readback_buffer(allocator: &Arc<StandardMemoryAllocator>, len: usize) -> Result<vulkano::buffer::Subbuffer<[u8]>> {
    vulkano::buffer::Buffer::from_iter(
        allocator,
        &vulkano::buffer::BufferCreateInfo {
            usage: vulkano::buffer::BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        std::iter::repeat_n(0u8, len),
    )
    .map_err(|e| Error::Unsupported(format!("readback buffer: {e}")))
}

/// Image usage flags the scale shader needs for its inputs/outputs.
pub mod usage {
    use vulkano::image::ImageUsage;

    /// Sampled input planes (`texture2D`, filled by transfer).
    pub const PLANE_IN: ImageUsage = ImageUsage::TRANSFER_DST.union(ImageUsage::SAMPLED);
    /// Storage output planes (r8, read back by transfer).
    pub const PLANE_OUT: ImageUsage =
        ImageUsage::STORAGE.union(ImageUsage::TRANSFER_SRC).union(ImageUsage::TRANSFER_DST);
}
