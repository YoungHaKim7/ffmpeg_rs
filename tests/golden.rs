//! Golden tests against the system ffmpeg binary.
//!
//! Byte-exact comparisons are the whole point of the port: files produced by
//! `ffmpeg_rs` must be indistinguishable from real ffmpeg's for the
//! raw/Y4M paths. The rgb24 conversion is checked to a ±3 tolerance because
//! swscale's table-based rounding is not bit-identical to our float BT.601
//! kernel (measured: max 3, mean 0.66 on testsrc2 — see the swscale module
//! docs).
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
