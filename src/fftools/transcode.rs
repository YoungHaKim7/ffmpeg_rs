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

use crate::{
    codec::{
        packet::Packet,
        params::{CodecId, CodecParameters},
        rawvideo::{RawVideoDecoder, RawVideoEncoder},
        traits::{Decoder, Encoder},
    },
    filter::{FilterGraph, NodeId, buffersink},
    format::{
        demux::DemuxOptions,
        {InputFormatContext, OutputFormatContext, Stream},
    },
    log_error, log_info, log_verbose,
    swscale::{ScaleAlgorithm, ScaleContext, ScaleEngine, ScaleOptions},
    util::{
        color::{ColorRange, ColorSpace},
        error::{Error, Result},
        frame::Frame,
        imgutils, log, pixdesc,
        pixfmt::PixelFormat,
        rational::Rational,
    },
};

use super::{
    cli::{Cli, Overwrite, parse},
    dump,
};

/// `main()` — parse, banner, run, and translate the pipeline-faithful
/// `Err(Error::Eof)`-style failures into ffmpeg-like exit behavior.
pub fn run(args: &[String]) -> Result<()> {
    let cli = parse(args)?;
    log::set_level(cli.log_level);

    // Banner (opt_common.c print_banner shape, honestly labeled).
    log_info!(
        None,
        "ffmpeg_rs version 0.1.0 Copyright (c) 2026 the ffmpeg_rs authors"
    );
    log_info!(
        None,
        "  built with rustc (FFmpeg 8.0.git pipeline-faithful port, phase 1)"
    );
    log_info!(
        None,
        "  libavutil / libavformat / libavcodec / libswscale subsets — see the crate docs"
    );

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

/// The `-vf` filtergraph runner — ffmpeg_filter.c's single-chain shape:
/// `buffersrc` (fed with decoded frames) → parsed description →
/// `buffersink` (pulled into the encoder). The auto-inserted trailing
/// `scale`/`format` (ffmpeg's `insert_filter` for `-s`/`-pix_fmt`) are
/// folded into the description by the caller before construction.
struct VfGraph {
    g: FilterGraph,
    src: NodeId,
    sink: NodeId,
}

/// `av_buffersrc_parameters_set` via the option string (the port's only
/// parameter path) — video fields only, colorspace/range when declared.
fn buffersrc_args(in_st: &Stream) -> String {
    use std::fmt::Write;
    let par = &in_st.codecpar;
    let mut s = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}/{}:frame_rate={}/{}",
        par.width,
        par.height,
        par.format.name(),
        in_st.time_base.num,
        in_st.time_base.den,
        in_st.avg_frame_rate.num,
        in_st.avg_frame_rate.den,
    );
    if in_st.sample_aspect_ratio.num != 0 || in_st.sample_aspect_ratio.den != 0 {
        let _ = write!(
            s,
            ":sar={}/{}",
            in_st.sample_aspect_ratio.num, in_st.sample_aspect_ratio.den
        );
    }
    let csp_name = |c: ColorSpace| match c {
        ColorSpace::Rgb => "gbr", // av_color_space_name(AVCOL_SPC_RGB)
        ColorSpace::Bt709 => "bt709",
        ColorSpace::Fcc => "fcc",
        ColorSpace::Bt470bg => "bt470bg",
        ColorSpace::Smpte170m => "smpte170m",
        ColorSpace::Smpte240m => "smpte240m",
        ColorSpace::Bt2020Ncl => "bt2020nc",
        _ => "",
    };
    if !csp_name(par.color_space).is_empty() {
        let _ = write!(s, ":colorspace={}", csp_name(par.color_space));
    }
    match par.color_range {
        ColorRange::Mpeg => s.push_str(":range=tv"),
        ColorRange::Jpeg => s.push_str(":range=pc"),
        _ => {}
    }
    s
}

impl VfGraph {
    /// `configure_filtergraph` (ffmpeg_filter.c): create the endpoints,
    /// parse the description, attach the open pads, negotiate formats.
    fn new(
        desc: &str,
        in_st: &Stream,
        scale_algorithm: ScaleAlgorithm,
        scale_engine: ScaleEngine,
    ) -> Result<VfGraph> {
        let mut g = FilterGraph::new();
        // `-scale_algo`/`-scale_engine` reach the graph like ffmpeg's
        // `-sws_flags` flows through graph scale_sws_opts (the graph's
        // scale filters read them when their own `flags` is unset).
        g.scale_algorithm = scale_algorithm;
        g.scale_engine = scale_engine;
        let src = g.create_filter("buffer", &buffersrc_args(in_st))?;
        let (open_inputs, open_outputs) = g.parse_ptr(desc)?;
        let sink = g.create_filter("buffersink", "")?;

        // Attach the endpoints to the parsed graph's open pads — exactly one
        // unnamed open pad on each side for this single-chain CLI
        // (ffmpeg_filter.c errors "Too many inputs"/outputs otherwise).
        let attach = |side: &str, pads: &[crate::filter::InOut]| -> Result<(NodeId, usize)> {
            if pads.len() != 1 {
                return Err(Error::InvalidArgument(format!(
                    "Simple filtergraph description has {n} open {side} pads; this CLI supports exactly one",
                    n = pads.len()
                )));
            }
            Ok((pads[0].node, pads[0].pad))
        };
        if open_inputs.is_empty() {
            // degenerate empty description: wire source straight to sink
            g.link(src, 0, sink, 0)?;
        } else {
            let (node, pad) = attach("input", &open_inputs)?;
            g.link(src, 0, node, pad)?;
        }
        let (out_node, out_pad) = if open_outputs.is_empty() {
            (sink, 0)
        } else {
            attach("output", &open_outputs)?
        };
        g.link(out_node, out_pad, sink, 0)?;

        g.config()?;
        Ok(VfGraph { g, src, sink })
    }

    /// One decoded frame into the source (`av_buffersrc_add_frame`).
    fn push(&mut self, frame: &Frame) -> Result<()> {
        self.g.add_frame(self.src, frame)
    }

    /// One filtered frame out, `Err(Again)` when starved.
    fn pull(&mut self) -> Result<Frame> {
        self.g.get_frame(self.sink)
    }

    /// `av_buffersrc_close` — EOF into the source.
    fn close(&mut self) -> Result<()> {
        self.g.close_source(self.src)
    }

    fn sink_format(&self) -> Result<PixelFormat> {
        buffersink::buffersink_get_format(&self.g, self.sink)?
            .ok_or_else(|| Error::InvalidArgument("buffersink link has no format".into()))
    }

    fn sink_w(&self) -> Result<u32> {
        buffersink::buffersink_get_w(&self.g, self.sink)
    }

    fn sink_h(&self) -> Result<u32> {
        buffersink::buffersink_get_h(&self.g, self.sink)
    }

    fn sink_color_range(&self) -> ColorRange {
        buffersink::buffersink_get_color_range(&self.g, self.sink)
            .unwrap_or(ColorRange::Unspecified)
    }
}

/// The whole pipeline, `transcode()` in fftools/ffmpeg.c.
fn transcode(cli: &Cli) -> Result<Stats> {
    // Output-file overwrite policy (ffmpeg_opt.c's assert_file_overwrite).
    if std::path::Path::new(&cli.output_url).exists() {
        match cli.overwrite {
            Overwrite::Always => {}
            Overwrite::Never | Overwrite::Prompt => {
                log_error!(
                    None,
                    "File '{}' exists. Exiting (use -y to overwrite).",
                    cli.output_url
                );
                return Err(Error::InvalidArgument("output exists".into()));
            }
        }
    }

    // ---- input ------------------------------------------------------------
    // Dispatch by media type: open the input, then hand off to the audio
    // loop when stream 0 is audio (ffmpeg's per-stream-type scheduling
    // collapsed to one branch).
    {
        let probe = InputFormatContext::open(
            &cli.input_url,
            cli.input_format.as_deref(),
            &DemuxOptions {
                raw_video: crate::format::demux::RawVideoDemuxOptions::default(),
            },
        )?;
        if probe.streams[0].codecpar.codec_type == crate::codec::params::MediaType::Audio {
            drop(probe);
            return transcode_audio(cli);
        }
    }
    let demux_opts = DemuxOptions {
        raw_video: crate::format::demux::RawVideoDemuxOptions {
            pixel_format: cli.input_pixel_format.unwrap_or(PixelFormat::Yuv420p),
            video_size: cli.input_video_size,
            framerate: cli.input_framerate.unwrap_or(Rational::new(25, 1)),
        },
    };
    let mut ictx =
        InputFormatContext::open(&cli.input_url, cli.input_format.as_deref(), &demux_opts)?;
    ictx.find_stream_info()?;
    dump::dump_input(&ictx);

    let in_st = ictx.streams[0].clone();

    // ---- filtergraph (-vf) -------------------------------------------------
    // ffmpeg folds `-s`/`-pix_fmt` into the graph as trailing filters
    // (`insert_filter`, ffmpeg_filter.c) — same here.
    let mut vf = match &cli.video_filters {
        Some(desc) => {
            let mut full = desc.clone();
            if let Some((w, h)) = cli.output_size {
                full += &format!(",scale={w}x{h}");
            }
            if let Some(fmt) = cli.output_pix_fmt {
                full += &format!(",format={}", fmt.name());
            }
            log_verbose!(None, "filtergraph description: {full}");
            Some(VfGraph::new(
                &full,
                &in_st,
                cli.scale_algorithm,
                cli.scale_engine,
            )?)
        }
        None => None,
    };

    // ---- output stream construction (ffmpeg_mux_init.c) -------------------
    // With a graph running, the output geometry/format come from the
    // negotiated sink link (what C reads off the sink after config).
    let out_pix_fmt = match vf.as_ref() {
        Some(v) => v.sink_format()?,
        None => cli.output_pix_fmt.unwrap_or(in_st.codecpar.format),
    };
    let (out_w, out_h) = match vf.as_ref() {
        Some(v) => (v.sink_w()?, v.sink_h()?),
        None => cli
            .output_size
            .unwrap_or((in_st.codecpar.width, in_st.codecpar.height)),
    };
    let mut out_par = CodecParameters {
        codec_id: CodecId::Rawvideo,
        format: out_pix_fmt,
        width: out_w,
        height: out_h,
        sample_aspect_ratio: in_st.sample_aspect_ratio,
        framerate: in_st.avg_frame_rate,
        field_order: in_st.codecpar.field_order,
        ..in_st.codecpar.clone()
    };
    // RGB outputs are full-range by convention (what the auto-inserted
    // swscale conversion produces); YUV stays as decoded. With a graph
    // running, the sink's negotiated range wins.
    if let Some(v) = vf.as_ref() {
        out_par.color_range = v.sink_color_range();
    } else if out_pix_fmt != in_st.codecpar.format
        && pixdesc::descriptor(out_pix_fmt)
            .flags
            .contains(pixdesc::PixFmtFlags::RGB)
    {
        out_par.color_range = crate::util::color::ColorRange::Jpeg;
    }
    // ff_guess_coded_bitrate: frame bytes · 8 · fps (rounded like C's
    // av_rescale_q nearest).
    let frame_bytes =
        imgutils::get_buffer_size(out_pix_fmt, out_par.width, out_par.height, 1)? as i64;
    let fps = out_par.framerate;
    out_par.bit_rate = if fps.den > 0 {
        crate::util::mathematics::rescale_q(
            frame_bytes * 8,
            Rational::new(fps.num, fps.den),
            Rational::ONE,
        )
    } else {
        0
    };

    let mut out_st = Stream::new_video(0);
    out_st.codecpar = out_par;
    out_st.sample_aspect_ratio = in_st.sample_aspect_ratio;
    // Output timebase follows the framerate (what ffmpeg does without
    // -video_track_timescale).
    if in_st.avg_frame_rate.num > 0 {
        out_st.set_pts_info(
            in_st.avg_frame_rate.den as i64,
            in_st.avg_frame_rate.num as i64,
        );
    } else {
        out_st.set_pts_info(1, 25);
    }

    dump::dump_stream_mapping();

    let mut octx = OutputFormatContext::create(
        &cli.output_url,
        cli.output_format.as_deref(),
        out_st.clone(),
    )?;
    octx.write_header()?;
    dump::dump_output(&octx);
    log_info!(None, "Press [q] to stop, [?] for help");

    // ---- codecs + converter ------------------------------------------------
    let mut decoder = RawVideoDecoder::new();
    decoder.init(&in_st.codecpar)?;

    let mut encoder = RawVideoEncoder::new();
    encoder.init(&out_st.codecpar)?;

    // ffmpeg inserts a scale filter when format OR size differs — but with
    // -vf the graph owns the conversion; no separate scaler runs.
    let needs_scale = vf.is_none()
        && (in_st.codecpar.format != out_pix_fmt
            || (in_st.codecpar.width, in_st.codecpar.height) != (out_w, out_h));
    let mut scaler = if needs_scale {
        Some(ScaleContext::new(
            (
                in_st.codecpar.format,
                in_st.codecpar.width,
                in_st.codecpar.height,
            ),
            (out_pix_fmt, out_w, out_h),
            ScaleOptions {
                algorithm: cli.scale_algorithm,
                engine: cli.scale_engine,
            },
        )?)
    } else {
        None
    };
    if let Some(scaler) = scaler.as_ref() {
        log_verbose!(
            None,
            "auto-inserted scale filter: {}x{} {} -> {}x{} {} ({}, {} engine)",
            in_st.codecpar.width,
            in_st.codecpar.height,
            in_st.codecpar.format.name(),
            out_w,
            out_h,
            out_pix_fmt.name(),
            cli.scale_algorithm.name(),
            match scaler_engine_used(scaler) {
                true => "vulkan",
                false => "cpu",
            }
        );
    }

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
                    // ffmpeg_filter.c: the frame goes into the graph (if
                    // any), else straight through the standalone scaler.
                    let frames: Vec<Frame> = match vf.as_mut() {
                        Some(vf) => {
                            vf.push(&frame)?;
                            let mut out = Vec::new();
                            while let Ok(f) = vf.pull() {
                                out.push(f);
                            }
                            out
                        }
                        None => vec![convert_frame(&frame, &mut scaler, &out_st)?],
                    };
                    for out_frame in frames {
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
                }
                Err(Error::Again) => break,
                Err(Error::Eof) => {
                    // EOF into the graph (av_buffersrc_close), then drain
                    // the sink until it reports Eof.
                    if let Some(vf) = vf.as_mut() {
                        vf.close()?;
                        while let Ok(f) = vf.pull() {
                            encoder.send_frame(Some(&f))?;
                            loop {
                                match encoder.receive_packet() {
                                    Ok(pkt) => write_packet(&mut octx, &pkt, &mut stats)?,
                                    Err(Error::Again) => break,
                                    Err(Error::Eof) => break,
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    }
                    break 'pipeline;
                }
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

/// Whether the scale context ended up on the GPU (for the verbose banner).
fn scaler_engine_used(scaler: &ScaleContext) -> bool {
    scaler.uses_gpu()
}

/// The filter-graph stand-in: insert the scale filter only when format or
/// size differs, else pass the frame through untouched (Arc-shared planes).
fn convert_frame(
    frame: &Frame,
    scaler: &mut Option<ScaleContext>,
    out_st: &Stream,
) -> Result<Frame> {
    match scaler {
        None => Ok(frame.clone()),
        Some(ctx) => {
            let mut dst = Frame::alloc(
                out_st.codecpar.format,
                out_st.codecpar.width,
                out_st.codecpar.height,
            )?;
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
    let fps = if elapsed > 0.0 {
        stats.frames as f64 / elapsed
    } else {
        0.0
    };
    // Progress time counts through the END of the last frame (pts+duration),
    // which is why ffmpeg prints 00:00:01.00 for 10 frames at 10 fps.
    let end_pts = stats.last_pts + stats.last_duration;
    let secs = end_pts as f64 * stats.time_base.to_f64();
    // print_report: `size=%8.0fKiB` (rounded, width 8).
    let kib = stats.bytes as f64 / 1024.0;
    let bitrate = if secs > 0.0 {
        stats.bytes as f64 * 8.0 / secs / 1000.0
    } else {
        0.0
    };
    let speed = if secs > 0.0 {
        secs / elapsed.max(1e-9)
    } else {
        0.0
    };
    log_info!(
        None,
        "frame={:5} fps={:4.1} q=-0.0 Lsize={kib:8.0}KiB time={} bitrate={bitrate:.1}kbits/s speed={speed:.0}x",
        stats.frames,
        fps,
        dump::progress_time_string(end_pts, stats.time_base),
    );
}

// ---------------------------------------------------------------------------
// Audio path (Phase 4b): demux → PcmDecoder → swresample → PcmEncoder → mux
// ---------------------------------------------------------------------------

/// `-sample_fmt`'s packed mapping plus an explicit planar rejection.
fn cli_sample_fmt(
    cli: &Cli,
    fallback: crate::util::samplefmt::SampleFormat,
) -> Result<crate::util::samplefmt::SampleFormat> {
    use crate::util::samplefmt::SampleFormat;
    let fmt = match &cli.output_sample_fmt {
        Some(name) => SampleFormat::from_name(name)
            .ok_or_else(|| Error::InvalidArgument(format!("Unknown sample format '{name}'")))?,
        None => fallback,
    };
    if fmt.is_planar() {
        return Err(Error::InvalidArgument(format!(
            "planar sample format '{}' cannot be stored in WAV; use a packed format",
            fmt.name()
        )));
    }
    Ok(fmt)
}

/// The audio transcode loop — `transcode()`'s AVMEDIA_TYPE_AUDIO branch.
fn transcode_audio(cli: &Cli) -> Result<Stats> {
    use crate::{
        codec::{
            params::MediaType,
            pcm::{PcmDecoder, PcmEncoder, codec_id_for_packed_le},
            traits::{AudioDecoder, AudioEncoder},
        },
        util::{audio_frame::AudioFrame, channel_layout::ChannelLayout},
    };
    // use crate::util::samplefmt::SampleFormat;

    if std::path::Path::new(&cli.output_url).exists() {
        match cli.overwrite {
            Overwrite::Always => {}
            Overwrite::Never | Overwrite::Prompt => {
                log_error!(
                    None,
                    "File '{}' exists. Exiting (use -y to overwrite).",
                    cli.output_url
                );
                return Err(Error::InvalidArgument("output exists".into()));
            }
        }
    }

    let demux_opts = DemuxOptions {
        raw_video: crate::format::demux::RawVideoDemuxOptions::default(),
    };
    let mut ictx =
        InputFormatContext::open(&cli.input_url, cli.input_format.as_deref(), &demux_opts)?;
    ictx.find_stream_info()?;
    let in_st = ictx.streams[0].clone();

    // ---- decoder ----------------------------------------------------------
    let mut decoder = PcmDecoder::new();
    decoder.init(&in_st.codecpar)?;

    // ---- output parameters (ffmpeg_filter.c's ofilter for audio) ----------
    let out_rate = cli.output_sample_rate.unwrap_or(in_st.codecpar.sample_rate);
    let out_layout = match cli.output_channels {
        Some(n) => ChannelLayout::default_for(n),
        None => {
            // The graph-negotiation default again: an UNSPEC input layout
            // must not ride into swr as the OUT side (init normalizes
            // s.out_ch_layout to native; unspec frames then compare as
            // CHANGED, swresample_frame.c:84-89).
            if in_st.codecpar.ch_layout.order
                == crate::util::channel_layout::Order::Unspecified
            {
                ChannelLayout::default_for(in_st.codecpar.ch_layout.nb_channels)
            } else {
                in_st.codecpar.ch_layout
            }
        }
    };
    let out_fmt = cli_sample_fmt(cli, in_st.codecpar.sample_fmt)?;
    let codec_id = codec_id_for_packed_le(out_fmt)
        .ok_or_else(|| Error::Unsupported(format!("no PCM codec for '{}'", out_fmt.name())))?;

    let mut out_par = CodecParameters {
        codec_type: MediaType::Audio,
        codec_id,
        sample_rate: out_rate,
        ch_layout: out_layout,
        sample_fmt: out_fmt,
        ..CodecParameters::default()
    };
    out_par.block_align = (out_layout.nb_channels * out_fmt.bytes_per_sample()) as i32;
    out_par.bit_rate = out_par.block_align as i64 * 8 * out_rate as i64;

    let mut out_st = Stream::new_audio(0);
    out_st.codecpar = out_par.clone();
    out_st.set_pts_info(1, out_rate as i64);

    let mut octx = OutputFormatContext::create(
        &cli.output_url,
        cli.output_format.as_deref(),
        out_st.clone(),
    )?;
    octx.write_header()?;

    // ---- converter + encoder ----------------------------------------------
    let need_swr = out_rate != in_st.codecpar.sample_rate
        || out_layout.nb_channels != in_st.codecpar.ch_layout.nb_channels
        || out_layout.mask != in_st.codecpar.ch_layout.mask
        || out_fmt != in_st.codecpar.sample_fmt;
    let mut swr = if need_swr {
        let mut s = crate::swresample::SwrContext::alloc_set_opts2(
            None,
            &out_layout,
            out_fmt,
            out_rate,
            &in_st.codecpar.ch_layout,
            in_st.codecpar.sample_fmt,
            in_st.codecpar.sample_rate,
        )?;
        s.init()?;
        Some(s)
    } else {
        None
    };

    let mut encoder = PcmEncoder::new();
    encoder.init(&out_par)?;

    let mut stats = Stats {
        frames: 0,
        bytes: 0,
        time_base: out_st.time_base,
        last_pts: 0,
        last_duration: 1,
        started: Instant::now(),
    };

    // The frame pump: decode → (convert) → encode. Output pts run on a
    // sample counter (swr_next_pts's linear-chain equivalent).
    let mut out_pts: i64 = 0;
    let mut push_frame = |swr: &mut Option<crate::swresample::SwrContext>,
                          encoder: &mut PcmEncoder,
                          frame: &AudioFrame,
                          octx: &mut OutputFormatContext,
                          stats: &mut Stats|
     -> Result<()> {
        let converted: Vec<AudioFrame> = match swr {
            Some(s) => {
                let mut out = AudioFrame {
                    format: out_fmt,
                    ch_layout: out_layout,
                    sample_rate: out_rate,
                    time_base: out_st.time_base,
                    ..AudioFrame::default()
                };
                let n = s.convert_frame(Some(&mut out), Some(frame))?;
                out.pts = frame.pts;
                if n > 0 {
                    out.nb_samples = n;
                    vec![out]
                } else {
                    vec![]
                }
            }
            None => vec![frame.clone()],
        };
        for f in &converted {
            let mut f = f.clone();
            f.pts = out_pts;
            out_pts += f.nb_samples as i64;
            f.duration = f.nb_samples as i64;
            encoder.send_frame(Some(&f))?;
            loop {
                match encoder.receive_packet() {
                    Ok(pkt) => write_packet(octx, &pkt, stats)?,
                    Err(Error::Again) => break,
                    Err(Error::Eof) => break,
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
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
                Ok(mut frame) => {
                    // ffmpeg's filtergraph negotiates REAL layouts before
                    // swr sees a frame (aformat defaults); the port's CLI
                    // path normalizes UNSPEC decoder layouts the same way
                    // (av_channel_layout_compare treats unspec-vs-native
                    // as CHANGED, swresample_frame.c:79-81 + chl.c:820).
                    if frame.ch_layout.order
                        == crate::util::channel_layout::Order::Unspecified
                    {
                        frame.ch_layout =
                            crate::util::channel_layout::ChannelLayout::default_for(
                                frame.ch_layout.nb_channels,
                            );
                    }
                    push_frame(&mut swr, &mut encoder, &frame, &mut octx, &mut stats)?;
                }
                Err(Error::Again) => break,
                Err(Error::Eof) => {
                    // Drain the resampler's delay (swr_convert with NULL in).
                    if let Some(s) = swr.as_mut() {
                        loop {
                            let mut out = AudioFrame {
                                format: out_fmt,
                                ch_layout: out_layout,
                                sample_rate: out_rate,
                                time_base: out_st.time_base,
                                ..AudioFrame::default()
                            };
                            let n = s.convert_frame(Some(&mut out), None)?;
                            if n == 0 {
                                break;
                            }
                            out.nb_samples = n;
                            out.pts = out_pts;
                            out.duration = n as i64;
                            out_pts += n as i64;
                            encoder.send_frame(Some(&out))?;
                            loop {
                                match encoder.receive_packet() {
                                    Ok(pkt) => write_packet(&mut octx, &pkt, &mut stats)?,
                                    Err(Error::Again) => break,
                                    Err(Error::Eof) => break,
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    }
                    break 'pipeline;
                }
                Err(e) => return Err(e),
            }
        }
    }

    // Flush the encoder.
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
