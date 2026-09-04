//! The transcode driver — port of the `fftools/ffmpeg.c` main loop, minus
//! the scheduler threads (single stream, in-order, no queues needed — the
//! C threads exist for A/V interleaving we don't have yet).
//!
//! ```text
//! demux (InputFormatContext)          ffmpeg_demux.c  input_thread
//!   → decode (RawVideoDecoder)        ffmpeg_dec.c    decoder_thread
//!     → convert (ScaleContext)        ffmpeg_filter.c filter_thread
//!       → encode (RawVideoEncoder)    ffmpeg_enc.c    encoder_thread
//!         → mux (OutputFormatContext) ffmpeg_mux.c    mux_thread
//! ```
//!
//! Frame count / timestamps flow: packets carry `pts` in stream timebase;
//! the decoder copies them onto frames; the encoder rescales into its own
//! timebase; the muxers don't care (both write raw payloads).

use std::time::Instant;

use crate::codec::packet::Packet;
use crate::codec::params::{CodecId, CodecParameters};
use crate::codec::rawvideo::{RawVideoDecoder, RawVideoEncoder};
use crate::codec::traits::{Decoder, Encoder};
use crate::format::demux::DemuxOptions;
use crate::format::{InputFormatContext, OutputFormatContext, Stream};
use crate::log_error;
use crate::log_info;
use crate::swscale::ScaleContext;
use crate::util::error::{Error, Result};
use crate::util::frame::Frame;
use crate::util::imgutils;
use crate::util::log;
use crate::util::pixdesc;
use crate::util::pixfmt::PixelFormat;
use crate::util::rational::Rational;

use super::cli::{parse, Cli, Overwrite};
use super::dump;

/// `main()` — parse, banner, run, and translate the pipeline-faithful
/// `Err(Error::Eof)`-style failures into ffmpeg-like exit behavior.
pub fn run(args: &[String]) -> Result<()> {
    let cli = parse(args)?;
    log::set_level(cli.log_level);

    // Banner (opt_common.c print_banner shape, honestly labeled).
    log_info!(None, "ffmpeg_rs version 0.1.0 Copyright (c) 2026 the ffmpeg_rs authors");
    log_info!(None, "  built with rustc (FFmpeg 8.0.git pipeline-faithful port, phase 1)");
    log_info!(None, "  libavutil / libavformat / libavcodec / libswscale subsets — see the crate docs");

    match transcode(&cli) {
        Ok(stats) => {
            print_summary(&stats);
            Ok(())
        }
        Err(e) => {
            log_error!(None, "{}", e);
            Err(e)
        }
    }
}

/// End-of-run counters for the `frame= … Lsize= …` line.
struct Stats {
    frames: u64,
    bytes: u64,
    time_base: Rational,
    last_pts: i64,
    last_duration: i64,
    started: Instant,
}

/// The whole pipeline, `transcode()` in fftools/ffmpeg.c.
fn transcode(cli: &Cli) -> Result<Stats> {
    // Output-file overwrite policy (ffmpeg_opt.c's assert_file_overwrite).
    if std::path::Path::new(&cli.output_url).exists() {
        match cli.overwrite {
            Overwrite::Always => {}
            Overwrite::Never | Overwrite::Prompt => {
                log_error!(None, "File '{}' exists. Exiting (use -y to overwrite).",
                    cli.output_url);
                return Err(Error::InvalidArgument("output exists".into()));
            }
        }
    }

    // ---- input ------------------------------------------------------------
    let demux_opts = DemuxOptions {
        raw_video: crate::format::demux::RawVideoDemuxOptions {
            pixel_format: cli.input_pixel_format.unwrap_or(PixelFormat::Yuv420p),
            video_size: cli.input_video_size,
            framerate: cli.input_framerate.unwrap_or(Rational::new(25, 1)),
        },
    };
    let mut ictx = InputFormatContext::open(&cli.input_url, cli.input_format.as_deref(), &demux_opts)?;
    ictx.find_stream_info()?;
    dump::dump_input(&ictx);

    let in_st = ictx.streams[0].clone();

    // ---- output stream construction (ffmpeg_mux_init.c) -------------------
    let out_pix_fmt = cli.output_pix_fmt.unwrap_or(in_st.codecpar.format);
    let mut out_par = CodecParameters {
        codec_id: CodecId::Rawvideo,
        format: out_pix_fmt,
        width: in_st.codecpar.width,
        height: in_st.codecpar.height,
        sample_aspect_ratio: in_st.sample_aspect_ratio,
        framerate: in_st.avg_frame_rate,
        field_order: in_st.codecpar.field_order,
        ..in_st.codecpar.clone()
    };
    // RGB outputs are full-range by convention (what the auto-inserted
    // swscale conversion produces); YUV stays as decoded.
    if out_pix_fmt != in_st.codecpar.format
        && pixdesc::descriptor(out_pix_fmt).flags.contains(pixdesc::PixFmtFlags::RGB)
    {
        out_par.color_range = crate::util::color::ColorRange::Jpeg;
    }
    // ff_guess_coded_bitrate: frame bytes · 8 · fps (rounded like C's
    // av_rescale_q nearest).
    let frame_bytes = imgutils::get_buffer_size(out_pix_fmt, out_par.width, out_par.height, 1)? as i64;
    let fps = out_par.framerate;
    out_par.bit_rate = if fps.den > 0 {
        crate::util::mathematics::rescale_q(frame_bytes * 8, Rational::new(fps.num, fps.den), Rational::ONE)
    } else {
        0
    };

    let mut out_st = Stream::new_video(0);
    out_st.codecpar = out_par;
    out_st.sample_aspect_ratio = in_st.sample_aspect_ratio;
    // Output timebase follows the framerate (what ffmpeg does without
    // -video_track_timescale).
    if in_st.avg_frame_rate.num > 0 {
        out_st.set_pts_info(in_st.avg_frame_rate.den as i64, in_st.avg_frame_rate.num as i64);
    } else {
        out_st.set_pts_info(1, 25);
    }

    dump::dump_stream_mapping();

    let mut octx = OutputFormatContext::create(&cli.output_url, cli.output_format.as_deref(), out_st.clone())?;
    octx.write_header()?;
    dump::dump_output(&octx);
    log_info!(None, "Press [q] to stop, [?] for help");

    // ---- codecs + converter ------------------------------------------------
    let mut decoder = RawVideoDecoder::new();
    decoder.init(&in_st.codecpar)?;

    let mut encoder = RawVideoEncoder::new();
    encoder.init(&out_st.codecpar)?;

    let scaler = if in_st.codecpar.format != out_pix_fmt {
        Some(ScaleContext::new(
            (in_st.codecpar.format, in_st.codecpar.width, in_st.codecpar.height),
            (out_pix_fmt, out_st.codecpar.width, out_st.codecpar.height),
        )?)
    } else {
        None // no auto-inserted scale filter — like ffmpeg when formats match
    };

    // ---- the loop ----------------------------------------------------------
    let mut stats = Stats {
        frames: 0,
        bytes: 0,
        time_base: out_st.time_base,
        last_pts: 0,
        last_duration: 1,
        started: Instant::now(),
    };

    let mut eof = false;
    'pipeline: loop {
        if !eof {
            match ictx.read_frame() {
                Ok(pkt) => decoder.send_packet(Some(&pkt))?,
                Err(Error::Eof) => {
                    decoder.send_packet(None)?;
                    eof = true;
                }
                Err(e) => return Err(e),
            }
        }
        loop {
            match decoder.receive_frame() {
                Ok(frame) => {
                    let out_frame = convert_frame(&frame, scaler.as_ref(), &out_st)?;
                    encoder.send_frame(Some(&out_frame))?;
                    loop {
                        match encoder.receive_packet() {
                            Ok(pkt) => write_packet(&mut octx, &pkt, &mut stats)?,
                            Err(Error::Again) => break,
                            Err(Error::Eof) => break,
                            Err(e) => return Err(e),
                        }
                    }
                }
                Err(Error::Again) => break,
                Err(Error::Eof) => break 'pipeline,
                Err(e) => return Err(e),
            }
        }
    }

    // Flush the encoder (encode.c drain).
    encoder.send_frame(None)?;
    loop {
        match encoder.receive_packet() {
            Ok(pkt) => write_packet(&mut octx, &pkt, &mut stats)?,
            Err(Error::Again) => (),
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }

    octx.write_trailer()?;
    Ok(stats)
}

/// The filter-graph stand-in: insert the scale filter only when the formats
/// differ, else pass the frame through untouched (Arc-shared planes).
fn convert_frame(frame: &Frame, scaler: Option<&ScaleContext>, out_st: &Stream) -> Result<Frame> {
    match scaler {
        None => Ok(frame.clone()),
        Some(ctx) => {
            let mut dst = Frame::alloc(out_st.codecpar.format, out_st.codecpar.width, out_st.codecpar.height)?;
            ctx.scale(frame, &mut dst)?;
            // av_frame_copy_props: everything but the pixels.
            dst.pts = frame.pts;
            dst.duration = frame.duration;
            dst.time_base = frame.time_base;
            dst.pict_type = frame.pict_type;
            dst.sample_aspect_ratio = frame.sample_aspect_ratio;
            dst.flags = frame.flags;
            dst.color_range = out_st.codecpar.color_range;
            Ok(dst)
        }
    }
}

fn write_packet(octx: &mut OutputFormatContext, pkt: &Packet, stats: &mut Stats) -> Result<()> {
    octx.write_frame(pkt)?;
    stats.frames += 1;
    stats.bytes += pkt.size() as u64;
    if pkt.pts != crate::NOPTS {
        stats.last_pts = pkt.pts.max(stats.last_pts);
    }
    stats.last_duration = pkt.duration.max(1);
    Ok(())
}

/// The final `print_report(1, …)` line, ffmpeg-shaped:
/// `frame=   10 fps=0.0 q=-0.0 Lsize=     180KiB time=00:00:01.00 bitrate=1475.5kbits/s speed=591x`
fn print_summary(stats: &Stats) {
    let elapsed = stats.started.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 { stats.frames as f64 / elapsed } else { 0.0 };
    // Progress time counts through the END of the last frame (pts+duration),
    // which is why ffmpeg prints 00:00:01.00 for 10 frames at 10 fps.
    let end_pts = stats.last_pts + stats.last_duration;
    let secs = end_pts as f64 * stats.time_base.to_f64();
    // print_report: `size=%8.0fKiB` (rounded, width 8).
    let kib = stats.bytes as f64 / 1024.0;
    let bitrate = if secs > 0.0 { stats.bytes as f64 * 8.0 / secs / 1000.0 } else { 0.0 };
    let speed = if secs > 0.0 { secs / elapsed.max(1e-9) } else { 0.0 };
    log_info!(None,
        "frame={:5} fps={:4.1} q=-0.0 Lsize={kib:8.0}KiB time={} bitrate={bitrate:.1}kbits/s speed={speed:.0}x",
        stats.frames,
        fps,
        dump::progress_time_string(end_pts, stats.time_base),
    );
}
