/// Image usage flags the scale shader needs for its inputs/outputs.
use vulkano::image::ImageUsage;

/// Sampled input planes (`texture2D`, filled by transfer).
pub const PLANE_IN: ImageUsage = ImageUsage::TRANSFER_DST.union(ImageUsage::SAMPLED);
/// Storage output planes (r8, read back by transfer).
pub const PLANE_OUT: ImageUsage = ImageUsage::STORAGE
    .union(ImageUsage::TRANSFER_SRC)
    .union(ImageUsage::TRANSFER_DST);
/// The 1×1 stand-in bound to slots the current mode does not use. It can
/// land on sampled OR storage bindings, so it needs both usages (a view
/// without the descriptor's usage violates VUID-VkDescriptorImageInfo-
/// imageView-00343).
pub const DUMMY: ImageUsage = PLANE_IN.union(ImageUsage::STORAGE);
