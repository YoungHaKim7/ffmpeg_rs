//! Shader modules (house pattern: GLSL files under `assets/`, one module
//! per shader, loaded through `vulkano_shaders`).
//!
//! | module | file | ports |
//! |---|---|---|
//! | [`scale_cs`] | `assets/scale.comp` | `libswscale` kernels + `libavfilter/vulkan/scale.comp.glsl` shape |

pub(crate) mod scale_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "../assets/scale.comp",
    }
}
