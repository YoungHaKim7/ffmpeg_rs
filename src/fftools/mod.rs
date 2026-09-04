//! `fftools` — the command-line front end, port of the fftools layer
//! (`ffmpeg.c` + `ffmpeg_opt.c` + `dump.c`'s printing), reduced to Phase 1's
//! single-stream synchronous pipeline.

pub mod cli;
pub mod dump;
pub mod transcode;
