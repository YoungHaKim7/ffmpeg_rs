//! `libswscale` — pixel-format conversion and scaling.
//!
//! ## Scope
//!
//! `ScaleContext` is the `sws_getContext` analog; `scale` the `sws_scale`
//! one. The kernels cover YUV420P/GRAY8 → packed RGB at any size (BT.601
//! range-aware) and same-family planar resampling (yuv420p→yuv420p,
//! gray→gray), with three float algorithms ported from
//! `libswscale/filters.c` plus five table-driven kernels ported bit-faithfully
//! from `initFilter` (`libswscale/utils.c:197-612`, see [`filter`]):
//!
//! | algorithm | C kernel | taps | support | path |
//! |---|---|---|---|---|
//! | [`ScaleAlgorithm::Nearest`] | `SWS_SCALE_POINT` (box) | 1 | 0.5 | float |
//! | [`ScaleAlgorithm::Bilinear`] | `triangle` | 2 | 1.0 | float |
//! | [`ScaleAlgorithm::Bicubic`] | `cubic(B=0, C=0.6)` | 4 | 2.0 | float |
//! | [`ScaleAlgorithm::Area`] | `SWS_AREA` trapezoid | 1+1, widened on downscale (`utils.c:287-293`) | ∞ | table, CPU-only |
//! | [`ScaleAlgorithm::Gauss`] | `SWS_GAUSS` `2^(-3d²)` | 1+8, widened | ∞ | table, CPU-only |
//! | [`ScaleAlgorithm::Sinc`] | `SWS_SINC` | 1+20, widened | ∞ | table, CPU-only |
//! | [`ScaleAlgorithm::Lanczos`] | `SWS_LANCZOS` 3-lobe | 1+6, widened | ∞ | table, CPU-only |
//! | [`ScaleAlgorithm::Spline`] | `SWS_SPLINE` recursive Hermite | 1+20, widened | ∞ | table, CPU-only |
//!
//! Bicubic is the default because it is FFmpeg's own default
//! (`-sws_flags bicubic`). Float-path coordinate mapping is the C
//! `(dst_pos + 0.5)·src/dst − 0.5` center alignment (`filters.c:70-78`),
//! weights normalized to sum 1. The float tap window is fixed, so heavy
//! downscaling (beyond ~2×) under-blurs relative to swscale for those three
//! — a documented divergence, now scoped to the float/GPU algorithms only:
//! the five table-driven kernels widen in source space exactly like
//! swscale (`utils.c:287-293`).
//!
//! Chroma siting: on the float path `ChromaLocation::Center` (Y4M
//! `C420jpeg`) samples chroma at the half-pixel-offset `(p−0.5)/2` position,
//! everything else uses LEFT `p/2`; the table path derives positions the C
//! way (`ff_sws_chroma_pos`, see [`filter::chroma_pos`]) — Unspecified means
//! CENTER there. One exception, matching C: identity-geometry yuv420p→RGB
//! takes `ff_get_unscaled_swscale`'s table converter
//! (`swscale_unscaled.c:2425`), which reads chroma cosited with no
//! interpolation and ignores the siting tag — see
//! [`ScaleContext::scale_cpu_unscaled_yuv_rgb`].
//!
//! ## Engines
//!
//! [`ScaleEngine::Vulkan`] runs the port of `vf_scale_vulkan.c`'s shape in
//! [`vulkan`]: a headless compute device (`src/gpu.rs`, the `FFVulkanContext`
//! analog) executing `assets/scale.comp` — the same three float kernels, in
//! GLSL. `Auto` picks the GPU and falls back to the CPU when no suitable
//! device exists; the library default is `Cpu` so unit tests stay hermetic
//! and deterministic. The five table-driven kernels have no shader:
//! `Auto` falls back to the CPU for them (logged at verbose level) and
//! `Vulkan` reports them unsupported at context creation — unless the
//! conversion never runs the algorithm (identity copy, unscaled yuv420p→RGB
//! table converter), where the GPU decision is unchanged. Known corner,
//! accepted: identity-geometry gray8→RGB *does* execute the algorithm
//! (degenerate weights), so `gray8→rgb same-size + lanczos` errors on the
//! Vulkan engine rather than silently ignoring the flag.
//! `tests/golden.rs` pins both engines against system ffmpeg.

pub mod filter;
pub mod vulkan;

use crate::util::{
    color::{ChromaLocation, ColorRange},
    error::{Error, Result},
    frame::Frame,
    pixdesc,
    pixfmt::PixelFormat,
};

/// The packed-RGB outputs `scale` can emit.
const RGB_OUTPUTS: &[PixelFormat] = &[
    PixelFormat::Rgb24,
    PixelFormat::Bgr24,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
];

/// What a `ScaleContext` has been asked to do (drives the shader mode and
/// the CPU kernel selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionMode {
    /// yuv420p → packed RGB (shader mode 0).
    Yuv420pToRgb,
    /// gray8 → packed RGB (shader mode 1).
    Gray8ToRgb,
    /// yuv420p → yuv420p resample (shader mode 2).
    Yuv420pToYuv420p,
    /// gray8 → gray8 resample (shader mode 3).
    Gray8ToGray8,
}

/// Can `ScaleContext` consume `fmt` as a SOURCE? Pure export of the
/// `ConversionMode` gates in `ScaleContext::new` (mod.rs:284-298) — the
/// input side of the conversion matrix, for the filtergraph's format
/// negotiation (`vf_scale` query_formats builds its one-sided input list
/// from this).
///
/// NOTE the deliberately narrower-than-C subset (C's `sws_isSupportedInput`
/// accepts ~every YUV/RGB/gray format): C422/C444/10-bit inputs fail
/// conversion graphs exactly like today's `-pix_fmt` path does; `-vf null`
/// still passes them through untouched.
pub fn supported_input(fmt: PixelFormat) -> bool {
    matches!(fmt, PixelFormat::Yuv420p | PixelFormat::Gray8)
}

/// Can `ScaleContext` produce `fmt` as a DESTINATION? The output side of the
/// same matrix: same-format resampling for the two supported planar inputs,
/// plus every packed-RGB output. See [`supported_input`] for the
/// narrower-than-C caveat.
pub fn supported_output(fmt: PixelFormat) -> bool {
    matches!(fmt, PixelFormat::Yuv420p | PixelFormat::Gray8) || RGB_OUTPUTS.contains(&fmt)
}

/// Which kernel resamples with (`SWS_*` flags subset).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScaleAlgorithm {
    /// `SWS_POINT` — box, 1 tap.
    Nearest,
    /// `SWS_BILINEAR` — triangle, 2 taps.
    Bilinear,
    /// `SWS_BICUBIC` — cubic B=0/C=0.6, 4 taps. FFmpeg's default scaler.
    #[default]
    Bicubic,
    /// `SWS_AREA` (1<<5) — area averaging: downscale trapezoid, upscale
    /// bilinear 2-tap (`utils.c:244-267, 346-354`). CPU-only (table path).
    Area,
    /// `SWS_GAUSS` (1<<7) — `2^(-3·d²)`, sizeFactor 8 (`utils.c:355-357`).
    /// CPU-only (table path).
    Gauss,
    /// `SWS_SINC` (1<<8) — `sin(πd)/(πd)`, sizeFactor 20 (`utils.c:358-359`).
    /// CPU-only (table path).
    Sinc,
    /// `SWS_LANCZOS` (1<<9) — 3-lobe `sinc·sinc/p`, sizeFactor
    /// `ceil(2·3.0) = 6` (`utils.c:278-279, 360-365`). CPU-only (table path).
    Lanczos,
    /// `SWS_SPLINE` (1<<10) — recursive cubic-Hermite with
    /// `p = -2.196152422706632`, sizeFactor 20 (`utils.c:155-166, 371-373`).
    /// One value like ffmpeg's `-sws_flags spline` — no spline16/36/64
    /// tables exist in libswscale; the width character is the tap count.
    /// CPU-only (table path).
    Spline,
}

impl ScaleAlgorithm {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "nearest" | "point" => ScaleAlgorithm::Nearest,
            "bilinear" => ScaleAlgorithm::Bilinear,
            "bicubic" => ScaleAlgorithm::Bicubic,
            "area" => ScaleAlgorithm::Area,
            "gauss" => ScaleAlgorithm::Gauss,
            "sinc" => ScaleAlgorithm::Sinc,
            "lanczos" => ScaleAlgorithm::Lanczos,
            "spline" => ScaleAlgorithm::Spline,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            ScaleAlgorithm::Nearest => "nearest",
            ScaleAlgorithm::Bilinear => "bilinear",
            ScaleAlgorithm::Bicubic => "bicubic",
            ScaleAlgorithm::Area => "area",
            ScaleAlgorithm::Gauss => "gauss",
            ScaleAlgorithm::Sinc => "sinc",
            ScaleAlgorithm::Lanczos => "lanczos",
            ScaleAlgorithm::Spline => "spline",
        }
    }

    /// Tap count per axis (`filters.c` filter_size for upscaling). For the
    /// table-driven variants this is the informational upscale tap count
    /// (1+sizeFactor, `utils.c:287-293`) — never used, because those go
    /// through [`filter::scale_plane`], which widens on downscale and lets
    /// the near-zero reduction pass pick the final size.
    pub const fn taps(self) -> usize {
        match self {
            ScaleAlgorithm::Nearest => 1,
            ScaleAlgorithm::Bilinear => 2,
            ScaleAlgorithm::Bicubic => 4,
            ScaleAlgorithm::Area => 2,
            ScaleAlgorithm::Gauss => 9,
            ScaleAlgorithm::Sinc => 21,
            ScaleAlgorithm::Lanczos => 7,
            ScaleAlgorithm::Spline => 21,
        }
    }

    /// The `scale_algorithms[]` sizeFactor (`utils.c:183-195` + lanczos
    /// override `utils.c:278-279`) — 0 for the float-path algorithms.
    pub const fn size_factor(self) -> i32 {
        match self {
            ScaleAlgorithm::Area => 1,
            ScaleAlgorithm::Gauss => 8,
            ScaleAlgorithm::Sinc => 20,
            ScaleAlgorithm::Lanczos => 6,
            ScaleAlgorithm::Spline => 20,
            _ => 0,
        }
    }

    /// Whether this kernel runs on the [`filter`] table path (CPU-only,
    /// bit-faithful `initFilter` port) instead of the float sampler.
    pub const fn is_table_driven(self) -> bool {
        matches!(
            self,
            ScaleAlgorithm::Area
                | ScaleAlgorithm::Gauss
                | ScaleAlgorithm::Sinc
                | ScaleAlgorithm::Lanczos
                | ScaleAlgorithm::Spline
        )
    }

    /// Kernel weight at distance `x` (source pixels). `filters.c:423`
    /// `cubic()` with the SWS_BICUBIC params {B=0, C=0.6}, `triangle()`
    /// (`filters.c:336`) for bilinear. The table-driven kernels are
    /// fixed-point and live in [`filter`] — this is never called for them
    /// (`sample_plane` guards with `is_table_driven`).
    pub fn weight(self, x: f32) -> f32 {
        match self {
            ScaleAlgorithm::Nearest => {
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            ScaleAlgorithm::Bilinear => (1.0 - x).max(0.0),
            ScaleAlgorithm::Bicubic => {
                // b = 0, c = 0.6, polynomial form from filters.c:423.
                let p0 = 6.0f32;
                let p2 = -18.0 + 6.0 * 0.6;
                let p3 = 12.0 - 6.0 * 0.6;
                let q0 = 24.0 * 0.6;
                let q1 = -48.0 * 0.6;
                let q2 = 30.0 * 0.6;
                let q3 = -6.0 * 0.6;
                if x < 1.0 {
                    (p0 + x * x * (p2 + x * p3)) / p0
                } else if x < 2.0 {
                    (q0 + x * (q1 + x * (q2 + x * q3))) / p0
                } else {
                    0.0
                }
            }
            _ => unreachable!("table-driven kernels are fixed-point (filter.rs)"),
        }
    }
}

/// Where the work happens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScaleEngine {
    /// Try the GPU, fall back to CPU (reason logged at verbose level).
    #[default]
    Auto,
    /// Require the Vulkan compute path; error when unavailable.
    Vulkan,
    /// CPU kernels only — deterministic, no device dependency; the library
    /// default so tests stay hermetic (the CLI passes `Auto`).
    Cpu,
}

/// `sws_getContext` options beyond the geometry.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleOptions {
    pub algorithm: ScaleAlgorithm,
    pub engine: ScaleEngine,
}

/// `SwsContext` — conversion configuration (`sws_getContext`).
pub struct ScaleContext {
    src: (PixelFormat, u32, u32),
    dst: (PixelFormat, u32, u32),
    options: ScaleOptions,
    mode: ConversionMode,
    /// Lazily created GPU session (None on the CPU path).
    gpu: Option<vulkan::GpuScaler>,
    /// Coefficient tables for the table-driven algorithms, built on the
    /// first table-driven `scale()` call (`initFilter` runs per context in
    /// C too, at `ff_sws_init_single_context`).
    filters: Option<filter::FilterPlan>,
    /// The chroma location the cached tables were built for (siting is frame
    /// metadata in C — `ff_sws_chroma_pos` reads the frame — so a changed
    /// location triggers a rebuild).
    filters_loc: ChromaLocation,
}

impl std::fmt::Debug for ScaleContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScaleContext")
            .field("src", &self.src)
            .field("dst", &self.dst)
            .field("options", &self.options)
            .field("mode", &self.mode)
            .field("gpu", &self.gpu.is_some())
            .finish()
    }
}

impl ScaleContext {
    /// `sws_getContext(srcW, srcH, srcFormat, dstW, dstH, dstFormat, flags)`.
    pub fn new(
        src: (PixelFormat, u32, u32),
        dst: (PixelFormat, u32, u32),
        options: ScaleOptions,
    ) -> Result<Self> {
        let mode = if src.0 == dst.0 && src.0 == PixelFormat::Yuv420p {
            ConversionMode::Yuv420pToYuv420p
        } else if src.0 == dst.0 && src.0 == PixelFormat::Gray8 {
            ConversionMode::Gray8ToGray8
        } else if src.0 == PixelFormat::Yuv420p && RGB_OUTPUTS.contains(&dst.0) {
            ConversionMode::Yuv420pToRgb
        } else if src.0 == PixelFormat::Gray8 && RGB_OUTPUTS.contains(&dst.0) {
            ConversionMode::Gray8ToRgb
        } else {
            return Err(Error::Unsupported(format!(
                "cannot convert {} to {}",
                src.0.name(),
                dst.0.name()
            )));
        };

        // Algorithm reachability: mirror scale()'s two unscaled early-outs —
        // copy_identity and the unscaled yuv420p→RGB table converter never
        // execute the algorithm, so those contexts keep the GPU decision
        // unchanged. The five table-driven kernels (filter.rs) have no
        // shader: Auto falls back to the CPU (logged), Vulkan errors —
        // mirroring the hard-fail arm below.
        let algo_used = !(src.0 == dst.0 && src.1 == dst.1 && src.2 == dst.2)
            && !(src.1 == dst.1 && src.2 == dst.2 && mode == ConversionMode::Yuv420pToRgb);

        let gpu = if options.algorithm.is_table_driven() && algo_used {
            match options.engine {
                ScaleEngine::Vulkan => {
                    return Err(Error::Unsupported(format!(
                        "scaling algorithm '{}' is not available on the Vulkan engine \
                         (supported: nearest, bilinear, bicubic)",
                        options.algorithm.name()
                    )));
                }
                _ => {
                    crate::log_verbose!(
                        None,
                        "scaling algorithm '{}' has no Vulkan kernel, using CPU",
                        options.algorithm.name()
                    );
                    None
                }
            }
        } else {
            match options.engine {
                ScaleEngine::Cpu => None,
                engine => match vulkan::GpuScaler::new() {
                    Ok(scaler) => Some(scaler),
                    Err(e) if engine == ScaleEngine::Vulkan => return Err(e),
                    Err(e) => {
                        // Auto: fall back, saying why at verbose level.
                        crate::log_verbose!(
                            None,
                            "Vulkan scaler unavailable ({}), using CPU kernels",
                            e
                        );
                        None
                    }
                },
            }
        };

        Ok(ScaleContext {
            src,
            dst,
            options,
            mode,
            gpu,
            filters: None,
            filters_loc: ChromaLocation::Unspecified,
        })
    }

    /// Whether the GPU path is active (for CLI diagnostics).
    pub fn uses_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// `sws_scale` — convert `src` into a caller-allocated `dst`. The
    /// frames' geometry must match the context's configuration.
    pub fn scale(&mut self, src: &Frame, dst: &mut Frame) -> Result<()> {
        if (src.format, src.width, src.height) != self.src
            || (dst.format, dst.width, dst.height) != self.dst
        {
            return Err(Error::InvalidArgument(
                "frame geometry does not match ScaleContext".into(),
            ));
        }
        if src.format == dst.format && src.width == dst.width && src.height == dst.height {
            self.copy_identity(src, dst);
            return Ok(());
        }

        // `ff_get_unscaled_swscale` (`swscale_unscaled.c:2425-2431`):
        // identity-geometry yuv420p→RGB dispatches to the `yuv2rgb` table
        // converter, which reads each chroma sample for its whole 2×2 luma
        // block — no interpolation, siting tag ignored (`yuv2rgb.c:154-155`).
        // Running the generic resampler instead would bicubic-filter the
        // chroma at fractional siting positions and blur transitions real
        // ffmpeg keeps sharp (measured max 154 vs the ±3 golden). Routed
        // before the engine dispatch so both engines agree here.
        if src.width == dst.width
            && src.height == dst.height
            && self.mode == ConversionMode::Yuv420pToRgb
        {
            return self.scale_cpu_unscaled_yuv_rgb(src, dst);
        }

        if let Some(gpu) = self.gpu.as_mut() {
            return gpu.scale(src, dst, self.mode, self.options.algorithm);
        }
        self.scale_cpu(src, dst)
    }

    /// The unscaled `yuv2rgb` table converter (see `scale`): luma read
    /// directly, chroma at `(xx >> 1, yy >> 1)` — 2×2 replication.
    fn scale_cpu_unscaled_yuv_rgb(&self, src: &Frame, dst: &mut Frame) -> Result<()> {
        let full = src.color_range == ColorRange::Jpeg;
        let (rv, gu, gv, bu) = bt601_coeffs(full);
        let out_ls = dst.linesize(0);
        let (r_off, g_off, b_off) = rgb_offsets(dst.format);
        let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
        let (dw, dh) = (dst.width as usize, dst.height as usize);
        let (y_ls, u_ls, v_ls) = (src.linesize(0), src.linesize(1), src.linesize(2));
        let out = dst.plane_mut(0);
        for yy in 0..dh {
            let out_row = &mut out[yy * out_ls..];
            let y_row = &src.plane(0)[yy * y_ls..];
            let u_row = &src.plane(1)[(yy >> 1) * u_ls..];
            let v_row = &src.plane(2)[(yy >> 1) * v_ls..];
            for xx in 0..dw {
                let fy = range_expand(y_row[xx] as f32, full);
                let u = u_row[xx >> 1] as f32 - 128.0;
                let v = v_row[xx >> 1] as f32 - 128.0;
                let px = xx * step;
                out_row[px + r_off] = (fy + rv * v).round().clamp(0.0, 255.0) as u8;
                out_row[px + g_off] = (fy - gu * u - gv * v).round().clamp(0.0, 255.0) as u8;
                out_row[px + b_off] = (fy + bu * u).round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(())
    }

    /// Unscaled same-format copy (`swscale_unscaled.c`).
    fn copy_identity(&self, src: &Frame, dst: &mut Frame) {
        for p in 0..pixdesc::count_planes(src.format) {
            let s = src.plane(p);
            let d = dst.plane_mut(p);
            let ls = src.linesize(p);
            for (i, row) in s.chunks_exact(ls).enumerate() {
                d[i * ls..i * ls + ls].copy_from_slice(row);
            }
        }
    }

    /// The CPU kernels — the semantic reference the GPU shader mirrors
    /// line-for-line (float path), plus the table-driven dispatch.
    fn scale_cpu(&mut self, src: &Frame, dst: &mut Frame) -> Result<()> {
        if self.options.algorithm.is_table_driven() {
            return self.scale_cpu_c(src, dst);
        }
        let alg = self.options.algorithm;
        // Range and siting follow the SOURCE frame (swscale derives the
        // colorspace details from the input's color tags).
        let full = src.color_range == ColorRange::Jpeg;
        let center = src.chroma_location == ChromaLocation::Center;

        match self.mode {
            ConversionMode::Yuv420pToRgb => {
                let out_ls = dst.linesize(0);
                let (r_off, g_off, b_off) = rgb_offsets(dst.format);
                let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
                let (dw, dh) = (dst.width as usize, dst.height as usize);
                let out = dst.plane_mut(0);
                for yy in 0..dh {
                    let out_row = &mut out[yy * out_ls..];
                    for xx in 0..dw {
                        let (r, g, b) = self.sample_yuv_rgb(src, xx, yy, alg, full, center);
                        let px = xx * step;
                        out_row[px + r_off] = r;
                        out_row[px + g_off] = g;
                        out_row[px + b_off] = b;
                    }
                }
            }
            ConversionMode::Gray8ToRgb => {
                let out_ls = dst.linesize(0);
                let (r_off, g_off, b_off) = rgb_offsets(dst.format);
                let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
                let g_plane = src.plane(0);
                let g_ls = src.linesize(0);
                let (dw, dh) = (dst.width as usize, dst.height as usize);
                let out = dst.plane_mut(0);
                for yy in 0..dh {
                    let out_row = &mut out[yy * out_ls..];
                    for xx in 0..dw {
                        let pos = (
                            dst_to_src(xx, src.width, dw as u32),
                            dst_to_src(yy, src.height, dh as u32),
                        );
                        let v = sample_plane(g_plane, g_ls, (src.width, src.height), pos, alg);
                        let c = range_expand(v, full).round().clamp(0.0, 255.0) as u8;
                        let px = xx * step;
                        out_row[px + r_off] = c;
                        out_row[px + g_off] = c;
                        out_row[px + b_off] = c;
                    }
                }
            }
            ConversionMode::Yuv420pToYuv420p => {
                let (cw_s, ch_s) = (src.width.div_ceil(2), src.height.div_ceil(2));
                let (cw_d, ch_d) = (dst.width.div_ceil(2), dst.height.div_ceil(2));
                for p in 0..3 {
                    let (sw, sh, dw, dh, chroma_shift) = if p == 0 {
                        (src.width, src.height, dst.width, dst.height, false)
                    } else {
                        (cw_s, ch_s, cw_d, ch_d, center)
                    };
                    let s = src.plane(p);
                    let ls_s = src.linesize(p);
                    let out_ls = dst.linesize(p);
                    let (dwu, dhu) = (dw as usize, dh as usize);
                    let out = dst.plane_mut(p);
                    for yy in 0..dhu {
                        let out_row = &mut out[yy * out_ls..];
                        for xx in 0..dwu {
                            // Plane-to-plane: chroma keeps the siting shift
                            // (half a chroma pixel for CENTER siting) so the
                            // resampled value lands where it came from.
                            let shift = if chroma_shift { -0.25 } else { 0.0 };
                            let pos = (
                                dst_to_src(xx, sw, dw) + shift,
                                dst_to_src(yy, sh, dh) + shift,
                            );
                            out_row[xx] = sample_plane(s, ls_s, (sw, sh), pos, alg)
                                .round()
                                .clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
            ConversionMode::Gray8ToGray8 => {
                let s = src.plane(0);
                let ls_s = src.linesize(0);
                let out_ls = dst.linesize(0);
                let (dw, dh) = (dst.width as usize, dst.height as usize);
                let out = dst.plane_mut(0);
                for yy in 0..dh {
                    let out_row = &mut out[yy * out_ls..];
                    for xx in 0..dw {
                        let pos = (
                            dst_to_src(xx, src.width, dw as u32),
                            dst_to_src(yy, src.height, dh as u32),
                        );
                        out_row[xx] = sample_plane(s, ls_s, (src.width, src.height), pos, alg)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        Ok(())
    }

    /// The table-driven path — a two-pass H/V run of the [`filter`]
    /// coefficient plan (the `initFilter` port, `ff_swscale`'s slice
    /// pipeline materialized). Serves the five libswscale kernels the float
    /// sampler doesn't implement; CPU-only (`ScaleContext::new` falls back
    /// or rejects on the Vulkan engine).
    fn scale_cpu_c(&mut self, src: &Frame, dst: &mut Frame) -> Result<()> {
        // Siting is frame metadata in C too (`ff_sws_chroma_pos` reads the
        // frame); rebuild the tables if a later frame carries a different
        // chroma location than the cached plan.
        if self.filters.is_none() || self.filters_loc != src.chroma_location {
            let scaler = filter::TableScaler::from_algorithm(self.options.algorithm);
            self.filters = Some(filter::build_plan(
                scaler,
                (src.width as i32, src.height as i32),
                (dst.width as i32, dst.height as i32),
                src.chroma_location,
            )?);
            self.filters_loc = src.chroma_location;
        }
        let plan = self.filters.as_ref().expect("filter plan just built");

        // Planar modes: per-plane H+V with the luma or chroma tables —
        // exactly C's lum/chr planar passes (slice.c's descriptors).
        let planar = |plane: usize,
                      src: &Frame,
                      dst: &mut Frame,
                      h: &filter::SwsFilter,
                      v: &filter::SwsFilter| {
            let luma = plane == 0;
            let (sw, sh) = if luma {
                (src.width as usize, src.height as usize)
            } else {
                (
                    src.width.div_ceil(2) as usize,
                    src.height.div_ceil(2) as usize,
                )
            };
            let dst_ls = dst.linesize(plane);
            filter::scale_plane(
                src.plane(plane),
                src.linesize(plane),
                (sw as i32, sh as i32),
                dst.plane_mut(plane),
                dst_ls,
                h,
                v,
            );
        };

        match self.mode {
            ConversionMode::Yuv420pToYuv420p => {
                for p in 0..3 {
                    let (h, v) = if p == 0 {
                        (&plan.h_lum, &plan.v_lum)
                    } else {
                        (&plan.h_chr, &plan.v_chr)
                    };
                    planar(p, src, dst, h, v);
                }
                Ok(())
            }
            ConversionMode::Gray8ToGray8 => {
                planar(0, src, dst, &plan.h_lum, &plan.v_lum);
                Ok(())
            }
            ConversionMode::Yuv420pToRgb => {
                // Compose like C's planar intermediate: resample to a
                // yuv420p frame at the destination geometry, then reuse the
                // unscaled table converter. This adds one extra 8-bit
                // rounding on the chroma path vs C's fused yuv2packedX —
                // inside the already-documented ±3 RGB converter divergence
                // (the new-algorithm golden tests assert on planar output).
                let mut inter = Frame::alloc(PixelFormat::Yuv420p, dst.width, dst.height)?;
                inter.color_range = src.color_range;
                for p in 0..3 {
                    let (h, v) = if p == 0 {
                        (&plan.h_lum, &plan.v_lum)
                    } else {
                        (&plan.h_chr, &plan.v_chr)
                    };
                    planar(p, src, &mut inter, h, v);
                }
                self.scale_cpu_unscaled_yuv_rgb(&inter, dst)
            }
            ConversionMode::Gray8ToRgb => {
                // Same composition at identity positions: the existing
                // replicate loop, fed by a resampled gray intermediate.
                let mut inter = Frame::alloc(PixelFormat::Gray8, dst.width, dst.height)?;
                planar(0, src, &mut inter, &plan.h_lum, &plan.v_lum);
                let full = src.color_range == ColorRange::Jpeg;
                let out_ls = dst.linesize(0);
                let (r_off, g_off, b_off) = rgb_offsets(dst.format);
                let step = pixdesc::descriptor(dst.format).comp[0].step as usize;
                let (dw, dh) = (dst.width as usize, dst.height as usize);
                let out = dst.plane_mut(0);
                for yy in 0..dh {
                    let out_row = &mut out[yy * out_ls..];
                    let g_row = &inter.plane(0)[yy * dw..];
                    for xx in 0..dw {
                        let c = range_expand(g_row[xx] as f32, full)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                        let px = xx * step;
                        out_row[px + r_off] = c;
                        out_row[px + g_off] = c;
                        out_row[px + b_off] = c;
                    }
                }
                Ok(())
            }
        }
    }

    /// Sample Y/U/V at dst pixel (xx,yy) and convert to RGB (BT.601).
    fn sample_yuv_rgb(
        &self,
        src: &Frame,
        xx: usize,
        yy: usize,
        alg: ScaleAlgorithm,
        full: bool,
        center: bool,
    ) -> (u8, u8, u8) {
        let (cw, ch) = (src.width.div_ceil(2), src.height.div_ceil(2));
        let sx = dst_to_src(xx, src.width, self.dst.1);
        let sy = dst_to_src(yy, src.height, self.dst.2);
        let y = sample_plane(
            src.plane(0),
            src.linesize(0),
            (src.width, src.height),
            (sx, sy),
            alg,
        );

        // Chroma position in the chroma-plane grid, honoring siting: a
        // CENTER-sited chroma sample i lives at luma position 2i+0.5, so the
        // chroma coordinate of luma position p is (p−0.5)/2.
        let (cx, cy) = if center {
            ((sx - 0.5) / 2.0, (sy - 0.5) / 2.0)
        } else {
            (sx / 2.0, sy / 2.0)
        };
        let u = sample_plane(src.plane(1), src.linesize(1), (cw, ch), (cx, cy), alg) - 128.0;
        let v = sample_plane(src.plane(2), src.linesize(2), (cw, ch), (cx, cy), alg) - 128.0;

        let fy = range_expand(y, full);
        let (rv, gu, gv, bu) = bt601_coeffs(full);
        let r = (fy + rv * v).round().clamp(0.0, 255.0) as u8;
        let g = (fy - gu * u - gv * v).round().clamp(0.0, 255.0) as u8;
        let b = (fy + bu * u).round().clamp(0.0, 255.0) as u8;
        (r, g, b)
    }
}

/// BT.601 YUV→RGB chroma gains `(rv, gu, gv, bu)` — `ff_yuv2rgb_coeffs`
/// (`yuv2rgb.c`): the limited-range set carries the 255/224 scaling; the
/// JPEG/full-range set is the classic 1.402/-0.344136/-0.714136/1.772.
/// Shared by both engines (the shader hardcodes the same two tuples).
#[inline]
pub(crate) fn bt601_coeffs(full: bool) -> (f32, f32, f32, f32) {
    if full {
        (1.402, 0.344136, 0.714136, 1.772)
    } else {
        (1.59603, 0.39176, 0.81297, 2.01723)
    }
}

/// Center-aligned dst→src mapping (`filters.c:78`):
/// `(dst_pos + 0.5) · ratio_inv − 0.5`.
#[inline]
pub(crate) fn dst_to_src(dst_pos: usize, src_size: u32, dst_size: u32) -> f32 {
    (dst_pos as f32 + 0.5) * src_size as f32 / dst_size as f32 - 0.5
}

/// Limited→full luma expansion (`sws_setColorspaceDetails` range handling).
#[inline]
pub(crate) fn range_expand(y: f32, full: bool) -> f32 {
    if full {
        y
    } else {
        (y - 16.0) * (255.0 / 219.0)
    }
}

/// Resample one plane at a float source position: separable weighted sum of
/// `taps²` neighbors with edge clamping and weight normalization. swscale
/// quantizes weights to `SWS_FILTER_SCALE` ints instead of normalizing —
/// the only rounding difference between this and the C (and the GPU shader
/// uses this exact float path so engines agree).
pub(crate) fn sample_plane(
    plane: &[u8],
    linesize: usize,
    size: (u32, u32),
    pos: (f32, f32),
    alg: ScaleAlgorithm,
) -> f32 {
    debug_assert!(
        !alg.is_table_driven(),
        "table algorithms go through filter::scale_plane"
    );
    let (w, h) = (size.0 as i32, size.1 as i32);
    let nearest = |p: (f32, f32)| {
        let x = (p.0.round() as i32).clamp(0, w - 1);
        let y = (p.1.round() as i32).clamp(0, h - 1);
        plane[y as usize * linesize + x as usize] as f32
    };
    if alg == ScaleAlgorithm::Nearest {
        return nearest(pos);
    }

    let taps = alg.taps() as i32;
    // First tap index: bilinear straddles floor(pos) (taps 2), bicubic
    // centers on floor(pos)−1 (taps 4) — the C window for upscales.
    let base_x = pos.0.floor() as i32 - (taps / 2 - 1);
    let base_y = pos.1.floor() as i32 - (taps / 2 - 1);

    let mut wx = [0f32; 4];
    let mut wy = [0f32; 4];
    // Signed sums, like compute_row's wsum_pos + wsum_neg (filters.c:104):
    // kernels with negative lobes (bicubic) must normalize to Σw = 1, not
    // Σ|w|, or constant input would not stay constant.
    let mut wsum_x = 0.0f32;
    let mut wsum_y = 0.0f32;
    for i in 0..taps {
        // filters.c evaluates its kernels on non-negative distances only
        // (the C `x` is offset + i, always ≥ 0 in its domain); Mitchell-
        // Netravali is even, so abs() is the faithful translation.
        wx[i as usize] = alg.weight(((base_x + i) as f32 - pos.0).abs());
        wy[i as usize] = alg.weight(((base_y + i) as f32 - pos.1).abs());
        wsum_x += wx[i as usize];
        wsum_y += wy[i as usize];
    }
    if wsum_x <= f32::EPSILON || wsum_y <= f32::EPSILON {
        return nearest(pos);
    }
    for w in &mut wx {
        *w /= wsum_x;
    }
    for w in &mut wy {
        *w /= wsum_y;
    }

    let mut acc = 0.0f32;
    for j in 0..taps {
        let y = (base_y + j).clamp(0, h - 1) as usize;
        let mut row_acc = 0.0f32;
        for i in 0..taps {
            let x = (base_x + i).clamp(0, w - 1) as usize;
            row_acc += plane[y * linesize + x] as f32 * wx[i as usize];
        }
        acc += row_acc * wy[j as usize];
    }
    acc
}

/// Component byte offsets of R/G/B inside one pixel of a packed RGB format.
pub(crate) fn rgb_offsets(fmt: PixelFormat) -> (usize, usize, usize) {
    let d = pixdesc::descriptor(fmt);
    (
        d.comp[0].offset as usize,
        d.comp[1].offset as usize,
        d.comp[2].offset as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::color::ColorRange;
    use crate::util::frame::Frame;

    fn cpu_opts(alg: ScaleAlgorithm) -> ScaleOptions {
        ScaleOptions {
            algorithm: alg,
            engine: ScaleEngine::Cpu,
        }
    }

    /// 2×1 yuv420p, black then white (limited range): known RGB results at
    /// identity size (kernel weights degenerate → exact passthrough).
    #[test]
    fn bt601_limited_range_endpoints() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 1).unwrap();
        src.color_range = ColorRange::Mpeg;
        src.plane_mut(0)[..2].copy_from_slice(&[16, 235]);
        src.plane_mut(1)[..1].copy_from_slice(&[128]);
        src.plane_mut(2)[..1].copy_from_slice(&[128]);

        for alg in [
            ScaleAlgorithm::Nearest,
            ScaleAlgorithm::Bilinear,
            ScaleAlgorithm::Bicubic,
        ] {
            let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 1).unwrap();
            ScaleContext::new(
                (PixelFormat::Yuv420p, 2, 1),
                (PixelFormat::Rgb24, 2, 1),
                cpu_opts(alg),
            )
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
            let out = dst.plane(0);
            assert_eq!(&out[..3], &[0, 0, 0], "{alg:?} black");
            assert_eq!(&out[3..6], &[255, 255, 255], "{alg:?} white");
        }
    }

    #[test]
    fn bt601_full_range_endpoints() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 1).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0)[..2].copy_from_slice(&[0, 255]);
        src.plane_mut(1)[..1].copy_from_slice(&[128]);
        src.plane_mut(2)[..1].copy_from_slice(&[128]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 1).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 2, 1),
            (PixelFormat::Rgb24, 2, 1),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        let out = dst.plane(0);
        assert_eq!(&out[..3], &[0, 0, 0]);
        assert_eq!(&out[3..6], &[255, 255, 255]);
    }

    /// Full-range chroma gains are the JPEG set (1.402/-0.344136/-0.714136/
    /// 1.772), not the limited-range 255/224-scaled one — Y=90, U=240, V=90
    /// (full range) must land at (37, 79, 255), hand-computed.
    #[test]
    fn bt601_full_range_chroma_gains() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 2).unwrap();
        src.color_range = ColorRange::Jpeg;
        for row in src.plane_mut(0).chunks_exact_mut(2) {
            row.copy_from_slice(&[90, 90]);
        }
        src.plane_mut(1)[..1].copy_from_slice(&[240]);
        src.plane_mut(2)[..1].copy_from_slice(&[90]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 2).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 2, 2),
            (PixelFormat::Rgb24, 2, 2),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        let out = dst.plane(0);
        for px in out.chunks_exact(3) {
            assert_eq!(px, &[37, 79, 255]);
        }
    }

    /// The identity-geometry unscaled path: chroma transitions stay sharp
    /// (2×2 replication, `swscale_unscaled.c`'s table converter) regardless
    /// of the requested algorithm — bicubic must NOT blur across the edge.
    #[test]
    fn unscaled_yuv420p_to_rgb_replicates_chroma() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 4, 2).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0).copy_from_slice(&[100; 8]);
        // One chroma column flips: U = 80 | 200 on the 2-wide chroma grid.
        src.plane_mut(1).copy_from_slice(&[80, 200]);
        src.plane_mut(2).copy_from_slice(&[128, 128]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 4, 2).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 4, 2),
            (PixelFormat::Rgb24, 4, 2),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        // Both luma columns of a chroma column see the SAME U (replication):
        // pixel 0 and 1 identical, 2 and 3 identical, no intermediate value.
        let out = dst.plane(0);
        let px = |i: usize| &out[i * 3..i * 3 + 3];
        assert_eq!(px(0), px(1));
        assert_eq!(px(2), px(3));
        assert_ne!(px(1), px(2), "chroma columns differ");
        // And the replicated U is the raw sample, not an interpolated blend:
        // pixels 0-1 ← chroma col 0 (U=80), pixels 2-3 ← chroma col 1 (U=200).
        let b_col0 = out[2] as i32; // pixel 0 blue — tracks U
        let b_col1 = out[8] as i32; // pixel 2 blue
        let expected_col0 = (100.0f32 + 1.772 * (80.0 - 128.0))
            .round()
            .clamp(0.0, 255.0) as i32;
        let expected_col1 = (100.0f32 + 1.772 * (200.0 - 128.0))
            .round()
            .clamp(0.0, 255.0) as i32;
        assert_eq!((b_col0, b_col1), (expected_col0, expected_col1));
    }

    /// Hand-computed mid-gray + colored chroma (limited range):
    /// Y=126, U=64, V=128 → R=128, G=153, B=0 (see Phase 1).
    #[test]
    fn bt601_colored_pixel_hand_computed() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 2).unwrap();
        src.color_range = ColorRange::Mpeg;
        for row in src.plane_mut(0).chunks_exact_mut(2) {
            row.copy_from_slice(&[126, 126]);
        }
        src.plane_mut(1)[..1].copy_from_slice(&[64]);
        src.plane_mut(2)[..1].copy_from_slice(&[128]);

        let mut dst = Frame::alloc(PixelFormat::Rgb24, 2, 2).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 2, 2),
            (PixelFormat::Rgb24, 2, 2),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        let out = dst.plane(0);
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 128, "R");
            assert_eq!(px[1], 153, "G");
            assert_eq!(px[2], 0, "B");
        }
    }

    #[test]
    fn bgr24_swaps_component_order() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 2, 1).unwrap();
        src.plane_mut(0)[..2].copy_from_slice(&[16, 235]);
        src.plane_mut(1)[..1].copy_from_slice(&[255]);
        src.plane_mut(2)[..1].copy_from_slice(&[128]);
        let mut dst = Frame::alloc(PixelFormat::Bgr24, 2, 1).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 2, 1),
            (PixelFormat::Bgr24, 2, 1),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        let out = dst.plane(0);
        assert_eq!(out[0], 255, "B first in bgr24");
    }

    #[test]
    fn identity_copies_planes() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 4, 2).unwrap();
        src.plane_mut(0)
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = i as u8);
        let mut dst = Frame::alloc(PixelFormat::Yuv420p, 4, 2).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 4, 2),
            (PixelFormat::Yuv420p, 4, 2),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        assert_eq!(src.plane(0), dst.plane(0));
    }

    #[test]
    fn gray8_to_rgb_replicates() {
        let mut src = Frame::alloc(PixelFormat::Gray8, 2, 1).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0)[..2].copy_from_slice(&[0, 200]);
        let mut dst = Frame::alloc(PixelFormat::Rgba, 2, 1).unwrap();
        ScaleContext::new(
            (PixelFormat::Gray8, 2, 1),
            (PixelFormat::Rgba, 2, 1),
            cpu_opts(ScaleAlgorithm::Bicubic),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        let out = dst.plane(0);
        assert_eq!(&out[..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..8], &[200, 200, 200, 0]);
    }

    /// Upscale 2×2 → 4×4 with nearest: every dst pixel maps to a distinct
    /// source quadrant (center-aligned 2× mapping is exact).
    #[test]
    fn nearest_upscale_is_exact_replication() {
        let mut src = Frame::alloc(PixelFormat::Gray8, 2, 2).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0).copy_from_slice(&[10, 20, 30, 40]);
        let mut dst = Frame::alloc(PixelFormat::Gray8, 4, 4).unwrap();
        ScaleContext::new(
            (PixelFormat::Gray8, 2, 2),
            (PixelFormat::Gray8, 4, 4),
            cpu_opts(ScaleAlgorithm::Nearest),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        // dst→src: (x+0.5)/2−0.5 → x=0→−0.25→0, x=1→0.25→0, x=2→0.75→1, x=3→1.25→1.
        let expect = [
            10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
        ];
        assert_eq!(dst.plane(0), &expect);
    }

    /// Bilinear 2×2 → 4×4 midpoint must interpolate: the center pixels of
    /// each quadrant-pair average neighbors.
    #[test]
    fn bilinear_upscale_interpolates() {
        let mut src = Frame::alloc(PixelFormat::Gray8, 2, 2).unwrap();
        src.color_range = ColorRange::Jpeg;
        src.plane_mut(0).copy_from_slice(&[0, 100, 200, 0]);
        let mut dst = Frame::alloc(PixelFormat::Gray8, 4, 4).unwrap();
        ScaleContext::new(
            (PixelFormat::Gray8, 2, 2),
            (PixelFormat::Gray8, 4, 4),
            cpu_opts(ScaleAlgorithm::Bilinear),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        // dst x=1 → src 0.25 → 0.75·0 + 0.25·100 = 25 (row 0).
        assert_eq!(dst.plane(0)[0..4], [0, 25, 75, 100]);
    }

    /// Constant-color input must stay constant under every algorithm —
    /// the normalized-kernel invariant (and kernel weights sum to 1).
    #[test]
    fn constant_color_is_preserved_by_resampling() {
        for alg in [
            ScaleAlgorithm::Nearest,
            ScaleAlgorithm::Bilinear,
            ScaleAlgorithm::Bicubic,
        ] {
            let mut src = Frame::alloc(PixelFormat::Yuv420p, 9, 7).unwrap();
            for p in 0..3 {
                for b in src.plane_mut(p).iter_mut() {
                    *b = (77 + p * 11) as u8;
                }
            }
            let mut dst = Frame::alloc(PixelFormat::Rgb24, 17, 13).unwrap();
            ScaleContext::new(
                (PixelFormat::Yuv420p, 9, 7),
                (PixelFormat::Rgb24, 17, 13),
                cpu_opts(alg),
            )
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap();
            // Compute expected RGB of the constant YUV once.
            let y = 77.0f32;
            let u = (77.0 + 11.0) - 128.0;
            let v = (77.0 + 22.0) - 128.0;
            let fy = (y - 16.0) * (255.0 / 219.0);
            let expected: [u8; 3] = [
                (fy + 1.59603 * v).round().clamp(0.0, 255.0) as u8,
                (fy - 0.39176 * u - 0.81297 * v).round().clamp(0.0, 255.0) as u8,
                (fy + 2.01723 * u).round().clamp(0.0, 255.0) as u8,
            ];
            for px in dst.plane(0).chunks_exact(3) {
                assert_eq!(px, &expected, "{alg:?}");
            }
        }
    }

    /// Planar yuv420p→yuv420p resample keeps plane geometry (odd sizes).
    #[test]
    fn planar_resample_geometry() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 13, 7).unwrap();
        for p in 0..3 {
            for (i, b) in src.plane_mut(p).iter_mut().enumerate() {
                *b = ((i * (p + 1)) % 251) as u8;
            }
        }
        let mut dst = Frame::alloc(PixelFormat::Yuv420p, 32, 24).unwrap();
        ScaleContext::new(
            (PixelFormat::Yuv420p, 13, 7),
            (PixelFormat::Yuv420p, 32, 24),
            cpu_opts(ScaleAlgorithm::Bilinear),
        )
        .unwrap()
        .scale(&src, &mut dst)
        .unwrap();
        // dst chroma is ceil(32/2)×ceil(24/2) = 16×12.
        assert_eq!(dst.plane(1).len(), 16 * 12);
    }

    #[test]
    fn unsupported_pairs_rejected() {
        assert!(
            ScaleContext::new(
                (PixelFormat::Yuv422p, 8, 8),
                (PixelFormat::Rgb24, 8, 8),
                cpu_opts(ScaleAlgorithm::Bicubic),
            )
            .is_err()
        );
    }

    /// Kernel sanity: bicubic weights at 0 = 1 and symmetric; negative lobes
    /// exist (that's what makes it sharper than bilinear).
    #[test]
    fn bicubic_kernel_shape() {
        let bic = |x: f32| ScaleAlgorithm::Bicubic.weight(x);
        assert!((bic(0.0) - 1.0).abs() < 1e-6);
        // The kernel itself is only defined for x >= 0 (filters.c calls it
        // on non-negative distances); callers pass abs() — see sample_plane.
        assert!(bic(1.3) < 0.0, "negative lobe between taps");
        assert_eq!(bic(2.5), 0.0);
        // Partition-of-unity: taps at distances 1.5, 0.5, 0.5, 1.5 (a
        // half-pixel-offset resample) sum to 1.
        let s: f32 = (0..4).map(|i| bic((i as f32 - 1.5).abs())).sum();
        assert!((s - 1.0).abs() < 1e-3, "sum {s}");
    }

    /// The five table-driven kernels keep constant-color input constant in
    /// planar mode — the error-diffused row-sum invariant end-to-end (the
    /// table-path analog of `constant_color_is_preserved_by_resampling`).
    #[test]
    fn table_algorithms_preserve_constant_color_planar() {
        for alg in [
            ScaleAlgorithm::Area,
            ScaleAlgorithm::Gauss,
            ScaleAlgorithm::Sinc,
            ScaleAlgorithm::Lanczos,
            ScaleAlgorithm::Spline,
        ] {
            let mut src = Frame::alloc(PixelFormat::Yuv420p, 9, 7).unwrap();
            for p in 0..3 {
                for b in src.plane_mut(p).iter_mut() {
                    *b = (77 + p * 11) as u8;
                }
            }
            let mut dst = Frame::alloc(PixelFormat::Yuv420p, 17, 13).unwrap();
            ScaleContext::new(
                (PixelFormat::Yuv420p, 9, 7),
                (PixelFormat::Yuv420p, 17, 13),
                cpu_opts(alg),
            )
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap_or_else(|e| panic!("{alg:?}: {e}"));
            for p in 0..3 {
                let want = (77 + p * 11) as u8;
                assert!(
                    dst.plane(p).iter().all(|&b| b == want),
                    "{alg:?} plane {p}: not constant {}",
                    dst.plane(p).iter().min().unwrap()
                );
            }

            // Downscale direction too.
            let mut dst = Frame::alloc(PixelFormat::Yuv420p, 5, 4).unwrap();
            ScaleContext::new(
                (PixelFormat::Yuv420p, 9, 7),
                (PixelFormat::Yuv420p, 5, 4),
                cpu_opts(alg),
            )
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap_or_else(|e| panic!("{alg:?}: {e}"));
            for p in 0..3 {
                let want = (77 + p * 11) as u8;
                assert!(
                    dst.plane(p).iter().all(|&b| b == want),
                    "{alg:?} down plane {p}"
                );
            }
        }
    }

    /// The table path composes RGB conversions through a planar
    /// intermediate (one extra rounding vs C's fused yuv2packedX — kept
    /// inside the ±3 converter divergence; smoke-checked here for shape).
    #[test]
    fn table_algorithm_rgb_compose_smoke() {
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 16, 8).unwrap();
        for p in 0..3 {
            for (i, b) in src.plane_mut(p).iter_mut().enumerate() {
                *b = ((i * (p + 3)) % 191) as u8;
            }
        }
        for alg in [ScaleAlgorithm::Lanczos, ScaleAlgorithm::Area] {
            let mut dst = Frame::alloc(PixelFormat::Rgb24, 8, 4).unwrap();
            ScaleContext::new(
                (PixelFormat::Yuv420p, 16, 8),
                (PixelFormat::Rgb24, 8, 4),
                cpu_opts(alg),
            )
            .unwrap()
            .scale(&src, &mut dst)
            .unwrap_or_else(|e| panic!("{alg:?}: {e}"));
            // 8×4 rgb24 = 96 bytes, all populated.
            assert_eq!(dst.plane(0).len(), 8 * 4 * 3);
            assert!(dst.plane(0).iter().any(|&b| b != 0));
        }
    }

    /// A left-sited source (Y4M `C420mpeg2`) rebuilds the chroma tables with
    /// the C siting positions — chroma output differs from the center
    /// default, luma does not.
    #[test]
    fn table_filters_rebuild_on_chroma_location_change() {
        let mut ctx = ScaleContext::new(
            (PixelFormat::Yuv420p, 13, 7),
            (PixelFormat::Yuv420p, 26, 14),
            cpu_opts(ScaleAlgorithm::Lanczos),
        )
        .unwrap();
        let mut src = Frame::alloc(PixelFormat::Yuv420p, 13, 7).unwrap();
        for p in 0..3 {
            for (i, b) in src.plane_mut(p).iter_mut().enumerate() {
                *b = ((i * 7 * (p + 1)) % 181) as u8;
            }
        }
        let mut dst_center = Frame::alloc(PixelFormat::Yuv420p, 26, 14).unwrap();
        src.chroma_location = ChromaLocation::Center;
        ctx.scale(&src, &mut dst_center).unwrap();
        let mut dst_left = Frame::alloc(PixelFormat::Yuv420p, 26, 14).unwrap();
        src.chroma_location = ChromaLocation::Left;
        ctx.scale(&src, &mut dst_left).unwrap();
        assert_eq!(
            dst_center.plane(0),
            dst_left.plane(0),
            "luma independent of siting"
        );
        assert_ne!(
            dst_center.plane(1),
            dst_left.plane(1),
            "chroma siting must shift the resample"
        );
    }

    /// Engine policy for the CPU-only kernels, decided at context creation:
    /// Vulkan hard-fails (without touching the device), Auto silently takes
    /// the CPU. Conversions that never execute the algorithm (identity
    /// copy, unscaled yuv420p→RGB) keep the GPU decision unchanged.
    #[test]
    fn engine_fallback_for_table_algorithms() {
        let table = ScaleAlgorithm::Lanczos;
        // Real conversion, engine=Vulkan ⇒ Unsupported, no GPU required.
        let err = ScaleContext::new(
            (PixelFormat::Yuv420p, 9, 7),
            (PixelFormat::Rgb24, 17, 13),
            ScaleOptions {
                algorithm: table,
                engine: ScaleEngine::Vulkan,
            },
        )
        .unwrap_err();
        match err {
            Error::Unsupported(msg) => {
                assert!(msg.contains("not available on the Vulkan engine"), "{msg}")
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        // Same conversion, engine=Auto ⇒ CPU, no GPU required.
        let ctx = ScaleContext::new(
            (PixelFormat::Yuv420p, 9, 7),
            (PixelFormat::Rgb24, 17, 13),
            ScaleOptions {
                algorithm: table,
                engine: ScaleEngine::Auto,
            },
        )
        .unwrap();
        assert!(!ctx.uses_gpu());
        // Identity-geometry yuv420p→RGB never runs the algorithm: no early
        // table rejection — the context builds fine and the GPU decision is
        // whatever the host offers (host-dependent, so nothing asserted).
        ScaleContext::new(
            (PixelFormat::Yuv420p, 9, 7),
            (PixelFormat::Rgb24, 9, 7),
            ScaleOptions {
                algorithm: table,
                engine: ScaleEngine::Auto,
            },
        )
        .unwrap();
        // Known corner (documented in the module docs): identity-geometry
        // gray8→rgb DOES run the algorithm today, so Vulkan + lanczos
        // errors even though the geometry is unscaled.
        let err = ScaleContext::new(
            (PixelFormat::Gray8, 9, 7),
            (PixelFormat::Rgb24, 9, 7),
            ScaleOptions {
                algorithm: table,
                engine: ScaleEngine::Vulkan,
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    /// `-scale_algo` names round-trip and the informational sizeFactor
    /// matches `utils.c:183-195`.
    #[test]
    fn algorithm_names_and_size_factors() {
        for (name, alg, sf) in [
            ("area", ScaleAlgorithm::Area, 1),
            ("gauss", ScaleAlgorithm::Gauss, 8),
            ("sinc", ScaleAlgorithm::Sinc, 20),
            ("lanczos", ScaleAlgorithm::Lanczos, 6),
            ("spline", ScaleAlgorithm::Spline, 20),
        ] {
            assert_eq!(ScaleAlgorithm::from_name(name), Some(alg));
            assert_eq!(alg.name(), name);
            assert_eq!(alg.size_factor(), sf);
            assert!(alg.is_table_driven());
        }
        assert_eq!(ScaleAlgorithm::Bicubic.size_factor(), 0);
        assert!(!ScaleAlgorithm::Bicubic.is_table_driven());
    }
}
