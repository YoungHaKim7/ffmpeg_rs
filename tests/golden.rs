//! Golden tests against the system ffmpeg binary.
//!
//! Byte-exact comparisons are the whole point of the port: files produced by
//! `ffmpeg_rs` must be indistinguishable from real ffmpeg's for the
//! raw/Y4M paths. The rgb24 conversion is checked to a ±3 tolerance because
//! swscale's table-based rounding is not bit-identical to our float BT.601
//! kernel (measured: max 3, mean 0.66 on testsrc2 — see the swscale module
//! docs).
//!
//! Tolerance-bearing tests pin `-scale_engine cpu` so results do not depend
//! on the host's GPU; the engine-consistency test exercises the Vulkan path
//! explicitly and skips politely when no device is available.
//!
//! Skips (with a notice, exit 0) when `ffmpeg` is not installed — CI boxes
//! without it should not fail on missing tooling.

use std::path::PathBuf;
use std::process::Command;

const W: u32 = 128;
const H: u32 = 96;
const FPS: u32 = 10;
const FRAMES: u32 = 10;

fn system_ffmpeg() -> Option<&'static str> {
    which("ffmpeg")
}

fn which(bin: &str) -> Option<&'static str> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            // Leak once: tests are single-process.
            return Some(Box::leak(candidate.to_str()?.to_string().into_boxed_str()));
        }
    }
    None
}

struct Fixture {
    dir: PathBuf,
    ffmpeg: &'static str,
    ours: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Option<Fixture> {
        let ffmpeg = system_ffmpeg()?;
        let dir = std::env::temp_dir().join(format!("ffmpeg_rs_golden_{name}"));
        std::fs::create_dir_all(&dir).ok()?;
        Some(Fixture {
            dir,
            ffmpeg,
            ours: PathBuf::from(env!("CARGO_BIN_EXE_ffmpeg_rs")),
        })
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn run_ffmpeg(&self, args: &[&str]) {
        let out = Command::new(self.ffmpeg)
            .args(["-hide_banner", "-loglevel", "error"])
            .args(args)
            .output()
            .expect("run system ffmpeg");
        assert!(out.status.success(), "system ffmpeg failed: {args:?}\n{}", String::from_utf8_lossy(&out.stderr));
    }

    fn run_ours(&self, args: &[&str]) -> (bool, String, String) {
        let out = Command::new(&self.ours).args(args).output().expect("run ffmpeg_rs");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// testsrc2 y4m input, ffmpeg-generated (A1:1, C420jpeg).
    fn make_input_y4m(&self) {
        self.run_ffmpeg(&[
            "-f", "lavfi",
            "-i", &format!("testsrc2=duration=1:size={W}x{H}:rate={FPS}"),
            "-pix_fmt", "yuv420p",
            "-f", "yuv4mpegpipe",
            self.path("in.y4m").to_str().unwrap(),
            "-y",
        ]);
    }
}

#[test]
fn golden_y4m_to_rawvideo_is_byte_exact() {
    let Some(fx) = Fixture::new("y4m_raw") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();

    fx.run_ffmpeg(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("ref.raw").to_str().unwrap(), "-y",
    ]);
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("out.raw").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "ffmpeg_rs failed:\n{stderr}");
    assert_eq!(
        std::fs::read(fx.path("out.raw")).unwrap(),
        std::fs::read(fx.path("ref.raw")).unwrap(),
        "rawvideo output differs from system ffmpeg"
    );
    assert_eq!(
        std::fs::metadata(fx.path("out.raw")).unwrap().len(),
        (W * H * 3 / 2) as u64 * FRAMES as u64,
        "expected 10 compact yuv420p frames"
    );
}

#[test]
fn golden_rawvideo_to_y4m_round_trip_is_byte_exact() {
    let Some(fx) = Fixture::new("raw_y4m") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();
    // Reference raw yuv420p payload.
    fx.run_ffmpeg(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("frames.raw").to_str().unwrap(), "-y",
    ]);

    fx.run_ffmpeg(&[
        "-f", "rawvideo",
        "-pixel_format", "yuv420p",
        "-video_size", &format!("{W}x{H}"),
        "-framerate", &FPS.to_string(),
        "-i", fx.path("frames.raw").to_str().unwrap(),
        "-f", "yuv4mpegpipe",
        fx.path("ref.y4m").to_str().unwrap(),
        "-y",
    ]);
    let (ok, _, stderr) = fx.run_ours(&[
        "-f", "rawvideo",
        "-pixel_format", "yuv420p",
        "-video_size", &format!("{W}x{H}"),
        "-framerate", &FPS.to_string(),
        "-i", fx.path("frames.raw").to_str().unwrap(),
        "-f", "yuv4mpegpipe",
        fx.path("out.y4m").to_str().unwrap(),
        "-y",
    ]);
    assert!(ok, "ffmpeg_rs failed:\n{stderr}");

    let ours = std::fs::read(fx.path("out.y4m")).unwrap();
    let theirs = std::fs::read(fx.path("ref.y4m")).unwrap();
    assert_eq!(
        ours, theirs,
        "y4m output differs from system ffmpeg.\nours header: {:?}\ntheirs header: {:?}",
        String::from_utf8_lossy(&ours[..ours.len().min(80)]),
        String::from_utf8_lossy(&theirs[..theirs.len().min(80)]),
    );
}

#[test]
fn golden_y4m_to_rgb24_within_tolerance() {
    let Some(fx) = Fixture::new("y4m_rgb") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();

    fx.run_ffmpeg(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        fx.path("ref.raw").to_str().unwrap(), "-y",
    ]);
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-scale_engine", "cpu",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        fx.path("out.raw").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "ffmpeg_rs failed:\n{stderr}");

    let ours = std::fs::read(fx.path("out.raw")).unwrap();
    let theirs = std::fs::read(fx.path("ref.raw")).unwrap();
    assert_eq!(ours.len(), (W * H * 3) as usize * FRAMES as usize);
    assert_eq!(ours.len(), theirs.len());

    let mut max_diff = 0usize;
    let mut over = 0usize;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = a.abs_diff(*b);
        max_diff = max_diff.max(d as usize);
        if d > 3 {
            over += 1;
        }
    }
    assert_eq!(over, 0, "rgb24 conversion exceeds ±3 tolerance (max {max_diff})");
    eprintln!("rgb24 max byte diff vs swscale: {max_diff}");
}

/// Both engines must agree tap-for-tap (`scale.comp` mirrors the CPU
/// kernels): every conversion mode × algorithm at ½ downscale, Vulkan output
/// vs CPU output within the unorm-store-vs-round() slack (≤2; measured ≤1).
/// Skips politely when no Vulkan device is available.
#[test]
fn golden_engine_consistency_cpu_vs_vulkan() {
    let Some(fx) = Fixture::new("engine") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();
    fx.run_ffmpeg(&[
        "-f", "lavfi",
        "-i", &format!("testsrc2=duration=1:size={W}x{H}:rate={FPS}"),
        "-pix_fmt", "gray8",
        "-f", "rawvideo",
        fx.path("in.gray").to_str().unwrap(), "-y",
    ]);

    let y4m_path = fx.path("in.y4m");
    let gray_path = fx.path("in.gray");
    let y4m = y4m_path.to_str().unwrap();
    let gray = gray_path.to_str().unwrap();
    let size = format!("{}x{}", W / 2, H / 2);
    let vsize = format!("{W}x{H}");
    let fps = FPS.to_string();
    // (label, common args incl. input, output pix_fmt) — one per shader mode.
    let cases: &[(&str, Vec<&str>, &str)] = &[
        ("yuv420p→rgb24 (mode 0)", vec!["-i", y4m], "rgb24"),
        ("yuv420p→yuv420p (mode 2)", vec!["-i", y4m], "yuv420p"),
        (
            "gray8→rgb24 (mode 1)",
            vec![
                "-f", "rawvideo", "-pixel_format", "gray8",
                "-video_size", &vsize, "-framerate", &fps,
                "-i", gray,
            ],
            "rgb24",
        ),
        (
            "gray8→gray8 (mode 3)",
            vec![
                "-f", "rawvideo", "-pixel_format", "gray8",
                "-video_size", &vsize, "-framerate", &fps,
                "-i", gray,
            ],
            "gray8",
        ),
    ];

    for (label, input_args, pix_fmt) in cases {
        for algo in ["nearest", "bilinear", "bicubic"] {
            let run = |engine: &str, out: &str| -> (bool, String) {
                let (ok, _, stderr) = fx.run_ours(&[
                    input_args.as_slice(),
                    &["-s", &size, "-scale_algo", algo, "-scale_engine", engine],
                    &["-f", "rawvideo", "-pix_fmt", pix_fmt, out, "-y"],
                ]
                .concat());
                (ok, stderr)
            };
            let (ok_cpu, err_cpu) = run("cpu", fx.path("cpu.out").to_str().unwrap());
            assert!(ok_cpu, "cpu engine failed ({label}/{algo}):\n{err_cpu}");
            let (ok_gpu, err_gpu) = run("vulkan", fx.path("gpu.out").to_str().unwrap());
            if !ok_gpu {
                eprintln!("skipping: vulkan engine unavailable ({err_gpu})");
                return;
            }

            let cpu = std::fs::read(fx.path("cpu.out")).unwrap();
            let gpu = std::fs::read(fx.path("gpu.out")).unwrap();
            assert_eq!(cpu.len(), gpu.len(), "{label}/{algo}: output sizes differ");
            let mut max_diff = 0usize;
            for (a, b) in cpu.iter().zip(gpu.iter()) {
                max_diff = max_diff.max(a.abs_diff(*b) as usize);
            }
            assert!(
                max_diff <= 2,
                "{label}/{algo}: engines diverge (max {max_diff}, unorm-vs-round slack is ≤2)"
            );
            eprintln!("{label}/{algo}: cpu↔vulkan max byte diff {max_diff}");
        }
    }
}

/// Downscaled output vs system ffmpeg, within the documented fixed-tap
/// divergence of the FLOAT path: swscale widens the kernel in source space
/// on downscale (`utils.c:287-293`), our fixed 4-tap window under-blurs
/// (measured max 89 on testsrc2 at ½ scale — see the swscale module docs).
/// This applies only to nearest/bilinear/bicubic; the table-driven kernels
/// widen like swscale and are pinned tightly by
/// `golden_scaled_new_algorithms_vs_system_ffmpeg` below. Catches gross
/// breakage while the divergence is documented.
#[test]
fn golden_scaled_y4m_vs_system_ffmpeg_tolerance() {
    let Some(fx) = Fixture::new("scaled") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();
    let size = &format!("{}x{}", W / 2, H / 2);

    fx.run_ffmpeg(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-s", size,
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("ref.raw").to_str().unwrap(), "-y",
    ]);
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-s", size, "-scale_engine", "cpu",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("out.raw").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "ffmpeg_rs failed:\n{stderr}");

    let ours = std::fs::read(fx.path("out.raw")).unwrap();
    let theirs = std::fs::read(fx.path("ref.raw")).unwrap();
    assert_eq!(ours.len(), theirs.len());

    let mut max_diff = 0usize;
    let mut over = 0usize;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = a.abs_diff(*b);
        max_diff = max_diff.max(d as usize);
        if d > 3 {
            over += 1;
        }
    }
    assert!(max_diff <= 96, "downscale diverges beyond the documented fixed-tap slack (max {max_diff})");
    assert!(
        (over as f64 / ours.len() as f64) < 0.45,
        "too many bytes outside ±3 ({over}/{})",
        ours.len()
    );
    eprintln!("scaled yuv420p max byte diff vs swscale: {max_diff} ({over}/{} over ±3)", ours.len());
}

/// The five table-driven kernels ([`crate::swscale::filter`] — the
/// `initFilter` port) vs system ffmpeg, which picks the same kernel via
/// `-sws_flags` (its `sws_names` are exactly our `-scale_algo` values).
/// Measured (2026-09, system ffmpeg 8.1.2 x86-64, testsrc2):
///
/// ```text
/// downscale 128x96 → 64x48 yuv420p : area max 0, gauss 1, sinc 2,
///                                     lanczos 1, spline 1; 0 bytes > ±3
/// upscale  128x96 → 256x192 yuv420p : all five max 1;        0 bytes > ±3
/// ```
///
/// The residual ±1-2 is system ffmpeg's own SIMD apply path: a standalone
/// build of FFmpeg's C reference (`initFilter` + `hScale8To15_c` +
/// `yuv2planeX_8_c`, utils.c/swscale.c/output.c) produces byte-identical
/// output to this port for all five kernels in both directions — pinned by
/// the coefficient-row unit tests in `filter.rs`. The distro binary's
/// SSE/AVX2 apply kernels sit ±1-2 off its own `_c` code.
#[test]
fn golden_scaled_new_algorithms_vs_system_ffmpeg() {
    let Some(fx) = Fixture::new("scaled_new") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();
    let in_path = fx.path("in.y4m");
    let in_arg = in_path.to_str().unwrap();

    for algo in ["area", "gauss", "sinc", "lanczos", "spline"] {
        for (label, size) in [("down", format!("{}x{}", W / 2, H / 2)), ("up", format!("{}x{}", W * 2, H * 2))] {
            fx.run_ffmpeg(&[
                "-i", in_arg,
                "-s", &size, "-sws_flags", algo,
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                fx.path("ref.raw").to_str().unwrap(), "-y",
            ]);
            let (ok, _, stderr) = fx.run_ours(&[
                "-i", in_arg,
                "-s", &size, "-scale_algo", algo, "-scale_engine", "cpu",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                fx.path("out.raw").to_str().unwrap(), "-y",
            ]);
            assert!(ok, "ffmpeg_rs failed ({algo}/{label}):\n{stderr}");

            let ours = std::fs::read(fx.path("out.raw")).unwrap();
            let theirs = std::fs::read(fx.path("ref.raw")).unwrap();
            assert_eq!(ours.len(), theirs.len(), "{algo}/{label}");

            let mut max_diff = 0usize;
            let mut over = 0usize;
            for (a, b) in ours.iter().zip(theirs.iter()) {
                let d = a.abs_diff(*b);
                max_diff = max_diff.max(d as usize);
                if d > 3 {
                    over += 1;
                }
            }
            assert!(
                max_diff <= 4,
                "{algo}/{label} diverges beyond the SIMD-vs-C slack (max {max_diff})"
            );
            assert_eq!(over, 0, "{algo}/{label}: {over} bytes outside ±3");
            eprintln!("{algo}/{label}: max byte diff vs swscale: {max_diff}");
        }
    }

    // One rgb24 spot check through the composed planar→RGB path. The bound
    // is dominated by the documented float-BT.601-converter divergence
    // (±3 unscaled, amplified by downscale ringing — the existing float
    // bicubic path measures the same shape: max 178, 37.7% over ±3;
    // lanczos here: max 179, 29.5% over ±3).
    let size = format!("{}x{}", W / 2, H / 2);
    fx.run_ffmpeg(&[
        "-i", in_arg,
        "-s", &size, "-sws_flags", "lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        fx.path("ref.raw").to_str().unwrap(), "-y",
    ]);
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", in_arg,
        "-s", &size, "-scale_algo", "lanczos", "-scale_engine", "cpu",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        fx.path("out.raw").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "ffmpeg_rs failed (lanczos/rgb24):\n{stderr}");
    let ours = std::fs::read(fx.path("out.raw")).unwrap();
    let theirs = std::fs::read(fx.path("ref.raw")).unwrap();
    let mut max_diff = 0usize;
    let mut over = 0usize;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = a.abs_diff(*b);
        max_diff = max_diff.max(d as usize);
        if d > 3 {
            over += 1;
        }
    }
    assert!(max_diff <= 192, "lanczos rgb24 diverges (max {max_diff})");
    assert!(
        (over as f64 / ours.len() as f64) < 0.40,
        "too many bytes outside ±3 ({over}/{})",
        ours.len()
    );
    eprintln!("lanczos/rgb24: max {max_diff}, {over}/{} over ±3", ours.len());
}

/// The five table-driven kernels are CPU-only: `-scale_engine vulkan` must
/// hard-fail at context creation (before touching any device), and
/// `-scale_engine auto` must fall back and produce byte-identical output to
/// the explicit CPU run. Neither leg needs a GPU, so no skip.
#[test]
fn engine_fallback_for_cpu_only_algorithms() {
    let Some(fx) = Fixture::new("fallback") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();
    let in_path = fx.path("in.y4m");
    let in_arg = in_path.to_str().unwrap();
    let size = format!("{}x{}", W / 2, H / 2);

    // (i) explicit Vulkan + lanczos ⇒ nonzero exit, Unsupported message.
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", in_arg,
        "-s", &size, "-scale_algo", "lanczos", "-scale_engine", "vulkan",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("gpu.out").to_str().unwrap(), "-y",
    ]);
    assert!(!ok, "vulkan + lanczos must fail, exit was 0");
    assert!(
        stderr.contains("not available on the Vulkan engine"),
        "stderr lacks the fallback reason:\n{stderr}"
    );

    // (ii) auto ⇒ success, byte-identical to the CPU run.
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", in_arg,
        "-s", &size, "-scale_algo", "lanczos",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("auto.out").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "auto engine failed:\n{stderr}");
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", in_arg,
        "-s", &size, "-scale_algo", "lanczos", "-scale_engine", "cpu",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        fx.path("cpu.out").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "cpu engine failed:\n{stderr}");
    assert_eq!(
        std::fs::read(fx.path("auto.out")).unwrap(),
        std::fs::read(fx.path("cpu.out")).unwrap(),
        "auto-fallback output must be the CPU result"
    );
}

#[test]
fn cli_dump_output_shape() {
    let Some(fx) = Fixture::new("dump") else {
        eprintln!("skipping: system ffmpeg not found");
        return;
    };
    fx.make_input_y4m();
    let (ok, _, stderr) = fx.run_ours(&[
        "-i", fx.path("in.y4m").to_str().unwrap(),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        fx.path("out.raw").to_str().unwrap(), "-y",
    ]);
    assert!(ok, "{stderr}");
    assert!(stderr.contains("Input #0, yuv4mpegpipe, from"), "{stderr}");
    assert!(stderr.contains("Duration: 00:00:01.00, start: 0.000000, bitrate: 1475 kb/s"), "{stderr}");
    assert!(stderr.contains("Stream #0:0: Video: rawvideo (I420 / 0x30323449), yuv420p(progressive), 128x96, SAR 1:1 DAR 4:3, 10 fps, 10 tbr, 10 tbn"), "{stderr}");
    assert!(stderr.contains("Output #0, rawvideo, to"), "{stderr}");
    assert!(stderr.contains("rawvideo (RGB[24] / 0x18424752), rgb24(pc, progressive), 128x96 [SAR 1:1 DAR 4:3], q=2-31, 2949 kb/s, 10 fps, 10 tbn"), "{stderr}");
    assert!(stderr.contains("time=00:00:01.00"), "{stderr}");
}
