//! Command line parsing — the Phase 1 subset of `fftools/ffmpeg_opt.c`.
//!
//! Hand-rolled (matching the repo convention of no heavyweight deps):
//! ffmpeg's ordered-sections grammar without the general option machinery.
//!
//! ```text
//! ffmpeg_rs [global opts] [input opts] -i INPUT [output opts] OUTPUT
//!
//! global:         -y | -n | -v LEVEL | -loglevel LEVEL | -h | --help
//! input opts:     -f FMT | -pixel_format FMT | -video_size WxH | -framerate R
//! output opts:    -f FMT | -pix_fmt FMT | -s WxH
//!                  | -scale_algo nearest|bilinear|bicubic|area|gauss|sinc|lanczos|spline
//!                  | -scale_engine auto|vulkan|cpu
//! ```
//!
//! `-s` resamples through swscale — on the GPU (`vf_scale_vulkan` port)
//! when a Vulkan device is available. Deliberately absent (later phases):
//! `-r`/`-vf` (filtergraph), `-ss` (seeking), `-t`, `-an/-vn`, multiple
//! inputs/outputs.

use crate::swscale::{ScaleAlgorithm, ScaleEngine};
use crate::util::error::{Error, Result};
use crate::util::log::Level;
use crate::util::pixfmt::PixelFormat;
use crate::util::rational::Rational;

/// What `-y`/`-n` decided about clobbering the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overwrite {
    /// Neither flag: refuse if the file exists (ffmpeg prompts; a
    /// non-interactive tool errors out instead).
    #[default]
    Prompt,
    /// `-y`.
    Always,
    /// `-n`.
    Never,
}

/// Parsed command line.
#[derive(Debug, Clone)]
pub struct Cli {
    pub log_level: Level,
    pub overwrite: Overwrite,
    pub input_url: String,
    /// `-f` before `-i`.
    pub input_format: Option<String>,
    /// `-pixel_format` (rawvideo demuxer option).
    pub input_pixel_format: Option<PixelFormat>,
    /// `-video_size WxH` (rawvideo demuxer option).
    pub input_video_size: Option<(u32, u32)>,
    /// `-framerate n[:d]` (rawvideo demuxer option).
    pub input_framerate: Option<Rational>,
    pub output_url: String,
    /// `-f` after `-i`.
    pub output_format: Option<String>,
    /// `-pix_fmt`.
    pub output_pix_fmt: Option<PixelFormat>,
    /// `-s WxH` — resample through swscale (Phase 2).
    pub output_size: Option<(u32, u32)>,
    /// `-scale_algo` (default bicubic, ffmpeg's default).
    pub scale_algorithm: ScaleAlgorithm,
    /// `-scale_engine` (default auto: GPU when possible).
    pub scale_engine: ScaleEngine,
}

const USAGE: &str = "\
usage: ffmpeg_rs [global options] [[input options] -i INPUT] [output options] OUTPUT

global options:
  -y              overwrite output files without asking
  -n              never overwrite output files
  -v LEVEL        set log level (quiet|panic|fatal|error|warning|info|verbose|debug|trace)
  -h, --help      show this help

input options (before -i):
  -f FMT          force input format (yuv4mpegpipe|rawvideo)
  -pixel_format F rawvideo only: pixel format of the samples
  -video_size WXH rawvideo only: frame size, e.g. 128x96
  -framerate R    rawvideo only: frame rate, e.g. 25 or 30000:1001

output options (after -i):
  -f FMT          force output format (yuv4mpegpipe|rawvideo)
  -pix_fmt F      output pixel format
  -s WXH          rescale, e.g. 320x240 (Vulkan compute when available)
  -scale_algo A   nearest | bilinear | bicubic (default) | area | gauss |
                  sinc | lanczos | spline (the latter five are CPU-only,
                  auto-falling back from the Vulkan engine)
  -scale_engine E auto (default) | vulkan | cpu

Pipeline: demux (y4m|rawvideo) -> decode (rawvideo) -> swscale -> encode
(rawvideo) -> mux (y4m|rawvideo). Resampling runs the libswscale kernels on
a Vulkan compute device when one is available, else on the CPU.";

/// `av_parse_video_rate` subset: `n` or `n:d`.
fn parse_rate(s: &str) -> Result<Rational> {
    let bad = || Error::InvalidArgument(format!("Invalid frame rate: {s}"));
    match s.split(':').count() {
        1 => {
            let n: f64 = s.trim().parse().map_err(|_| bad())?;
            Ok(Rational::from_f64(n, i32::MAX))
        }
        2 => {
            let mut it = s.split(':');
            let n: i32 = it.next().unwrap().trim().parse().map_err(|_| bad())?;
            let d: i32 = it.next().unwrap().trim().parse().map_err(|_| bad())?;
            if n <= 0 || d <= 0 {
                return Err(bad());
            }
            Ok(Rational::new(n, d))
        }
        _ => Err(bad()),
    }
}

/// `-video_size WxH`.
fn parse_size(s: &str) -> Result<(u32, u32)> {
    let bad = || Error::InvalidArgument(format!("Invalid video size: {s}"));
    let mut it = s.splitn(2, |c| c == 'x' || c == 'X');
    let w: u32 = it.next().unwrap().trim().parse().map_err(|_| bad())?;
    let h: u32 = it.next().ok_or_else(bad)?.trim().parse().map_err(|_| bad())?;
    if w == 0 || h == 0 {
        return Err(bad());
    }
    Ok((w, h))
}

fn parse_pixfmt(flag: &str, s: &str) -> Result<PixelFormat> {
    PixelFormat::from_name(s).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "Invalid pixel format for {flag}: '{s}' (known: {})",
            PixelFormat::ALL.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
        ))
    })
}

/// Parse argv (`ffmpeg_parse_options` subset).
pub fn parse(args: &[String]) -> Result<Cli> {
    let mut cli = Cli {
        log_level: Level::Info,
        overwrite: Overwrite::Prompt,
        input_url: String::new(),
        input_format: None,
        input_pixel_format: None,
        input_video_size: None,
        input_framerate: None,
        output_url: String::new(),
        output_format: None,
        output_pix_fmt: None,
        output_size: None,
        scale_algorithm: ScaleAlgorithm::Bicubic,
        scale_engine: ScaleEngine::Auto,
    };

    let mut i = 0usize;
    let mut have_input = false;
    let mut positional: Vec<String> = Vec::new();

    while i < args.len() {
        let arg = &args[i];
        let mut next = |what: &str| -> Result<String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| Error::InvalidArgument(format!("Option {what} requires an argument")))
        };

        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-y" => cli.overwrite = Overwrite::Always,
            "-n" => cli.overwrite = Overwrite::Never,
            "-v" | "-loglevel" => {
                let v = next(arg)?;
                cli.log_level =
                    Level::from_name(&v).ok_or_else(|| {
                        Error::InvalidArgument(format!("Unknown log level '{v}'"))
                    })?;
            }
            "-f" => {
                let v = next(arg)?;
                if have_input {
                    cli.output_format = Some(v);
                } else {
                    cli.input_format = Some(v);
                }
            }
            "-i" => {
                let v = next("-i")?;
                if have_input {
                    return Err(Error::InvalidArgument(
                        "multiple inputs are not supported in phase 1".into(),
                    ));
                }
                cli.input_url = v;
                have_input = true;
            }
            "-pixel_format" => {
                if have_input {
                    return Err(Error::InvalidArgument(
                        "-pixel_format is an input (rawvideo) option; put it before -i".into(),
                    ));
                }
                let v = next(arg)?;
                cli.input_pixel_format = Some(parse_pixfmt(arg, &v)?);
            }
            "-video_size" => {
                if have_input {
                    return Err(Error::InvalidArgument(
                        "-video_size is an input (rawvideo) option; put it before -i".into(),
                    ));
                }
                let v = next(arg)?;
                cli.input_video_size = Some(parse_size(&v)?);
            }
            "-framerate" => {
                if have_input {
                    return Err(Error::InvalidArgument(
                        "-framerate is an input (rawvideo) option; put it before -i".into(),
                    ));
                }
                let v = next(arg)?;
                cli.input_framerate = Some(parse_rate(&v)?);
            }
            "-pix_fmt" => {
                if !have_input {
                    return Err(Error::InvalidArgument(
                        "-pix_fmt is an output option; put it after -i".into(),
                    ));
                }
                let v = next(arg)?;
                cli.output_pix_fmt = Some(parse_pixfmt(arg, &v)?);
            }
            "-s" => {
                if !have_input {
                    return Err(Error::InvalidArgument(
                        "-s is an output option; put it after -i".into(),
                    ));
                }
                let v = next(arg)?;
                cli.output_size = Some(parse_size(&v)?);
            }
            "-scale_algo" => {
                let v = next(arg)?;
                cli.scale_algorithm = ScaleAlgorithm::from_name(&v).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "Unknown scaling algorithm '{v}' (nearest|bilinear|bicubic|area|gauss|sinc|lanczos|spline)"
                    ))
                })?;
            }
            "-scale_engine" => {
                let v = next(arg)?;
                cli.scale_engine = match v.as_str() {
                    "auto" => ScaleEngine::Auto,
                    "vulkan" => ScaleEngine::Vulkan,
                    "cpu" => ScaleEngine::Cpu,
                    other => {
                        return Err(Error::InvalidArgument(format!(
                            "Unknown scaling engine '{other}' (auto|vulkan|cpu)"
                        )))
                    }
                };
            }
            "-r" | "-vf" | "-ss" => {
                return Err(Error::Unsupported(format!(
                    "option {arg} arrives with a later phase (filtergraph / seeking)"
                )));
            }
            s if s.starts_with('-') => {
                return Err(Error::InvalidArgument(format!(
                    "Unrecognized option '{s}'\n{USAGE}"
                )));
            }
            s => positional.push(s.to_string()),
        }
        i += 1;
    }

    if !have_input {
        return Err(Error::InvalidArgument(format!(
            "No input given (-i)\n{USAGE}"
        )));
    }
    if positional.len() != 1 {
        return Err(Error::InvalidArgument(format!(
            "Exactly one output file is required\n{USAGE}"
        )));
    }
    cli.output_url = positional.remove(0);
    Ok(cli)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn basic_transcode_invocation() {
        let cli = parse(&args(&["-i", "in.y4m", "-f", "rawvideo", "-pix_fmt", "rgb24", "out.raw"]))
            .unwrap();
        assert_eq!(cli.input_url, "in.y4m");
        assert_eq!(cli.output_url, "out.raw");
        assert_eq!(cli.output_format.as_deref(), Some("rawvideo"));
        assert_eq!(cli.output_pix_fmt, Some(PixelFormat::Rgb24));
        assert_eq!(cli.log_level, Level::Info);
    }

    #[test]
    fn input_options_section_ordering() {
        let cli = parse(&args(&[
            "-f", "rawvideo",
            "-pixel_format", "yuv420p",
            "-video_size", "128x96",
            "-framerate", "25",
            "-i", "in.raw",
            "out.y4m",
        ]))
        .unwrap();
        assert_eq!(cli.input_format.as_deref(), Some("rawvideo"));
        assert_eq!(cli.input_pixel_format, Some(PixelFormat::Yuv420p));
        assert_eq!(cli.input_video_size, Some((128, 96)));
        assert_eq!(cli.input_framerate, Some(Rational::new(25, 1)));
        // No -f after -i → output inferred from extension.
        assert_eq!(cli.output_format, None);
    }

    #[test]
    fn framerate_accepts_colon_and_decimal() {
        assert_eq!(parse_rate("25").unwrap(), Rational::new(25, 1));
        assert_eq!(parse_rate("30000:1001").unwrap(), Rational::new(30000, 1001));
        let ntsc = parse_rate("29.97").unwrap();
        assert!((ntsc.to_f64() - 29.97).abs() < 1e-6);
        assert!(parse_rate("0:0").is_err());
    }

    #[test]
    fn scale_options_parse() {
        let cli = parse(&args(&[
            "-i", "in.y4m", "-s", "320x240",
            "-scale_algo", "bilinear", "-scale_engine", "cpu", "out.raw",
        ]))
        .unwrap();
        assert_eq!(cli.output_size, Some((320, 240)));
        assert_eq!(cli.scale_algorithm, ScaleAlgorithm::Bilinear);
        assert_eq!(cli.scale_engine, ScaleEngine::Cpu);
        // Defaults: bicubic (ffmpeg's default) on the best engine.
        let cli = parse(&args(&["-i", "in.y4m", "out.raw"])).unwrap();
        assert_eq!(cli.scale_algorithm, ScaleAlgorithm::Bicubic);
        assert_eq!(cli.scale_engine, ScaleEngine::Auto);
        assert!(parse(&args(&["-i", "a", "-s", "320", "b"])).is_err());
        assert!(parse(&args(&["-i", "a", "-scale_algo", "fast", "b"])).is_err());
    }

    /// All eight `-scale_algo` names parse (the accepted-value list exists in
    /// 4 places — from_name, the error string, USAGE, the module doc — this
    /// pins the parsing side of the sync).
    #[test]
    fn scale_algo_new_values_parse() {
        for (name, alg) in [
            ("nearest", ScaleAlgorithm::Nearest),
            ("bilinear", ScaleAlgorithm::Bilinear),
            ("bicubic", ScaleAlgorithm::Bicubic),
            ("area", ScaleAlgorithm::Area),
            ("gauss", ScaleAlgorithm::Gauss),
            ("sinc", ScaleAlgorithm::Sinc),
            ("lanczos", ScaleAlgorithm::Lanczos),
            ("spline", ScaleAlgorithm::Spline),
        ] {
            let cli = parse(&args(&["-i", "in.y4m", "-s", "8x8", "-scale_algo", name, "out.raw"]))
                .unwrap_or_else(|_| panic!("{name} should parse"));
            assert_eq!(cli.scale_algorithm, alg);
        }
        assert!(parse(&args(&["-i", "a", "-scale_algo", "bicublin", "b"])).is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&args(&["out.y4m"])).is_err()); // no -i
        assert!(parse(&args(&["-i", "in.y4m"])).is_err()); // no output
        assert!(parse(&args(&["-i", "a.y4m", "-x", "b.raw"])).is_err()); // unknown flag
        assert!(parse(&args(&["-i", "a.y4m", "-pix_fmt", "nope", "b.raw"])).is_err());
        assert!(parse(&args(&["-v", "loud", "-i", "a.y4m", "b.raw"])).is_err());
    }
}
