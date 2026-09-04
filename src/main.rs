//! Thin shim like every bin in the sibling projects: all logic lives in the
//! library (`fftools::transcode::run`), the binary just forwards argv.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match ffmpeg_rs::fftools::transcode::run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::FAILURE,
    }
}
