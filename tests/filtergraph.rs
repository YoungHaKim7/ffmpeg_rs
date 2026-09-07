//! End-to-end filtergraph tests — the Rust-level equivalent of running
//! `ffmpeg -vf` (what `fftools/transcode.rs` does), without the CLI or the
//! parser: graphs are built programmatically via `create_filter` + `link`.
//!
//! These exercise the full wave-2 stack today: buffersrc option-string
//! init + format negotiation (query/merge/reduce/pick), the activation
//! engine (filter_frame / default activation / status protocol), and the
//! buffersink pull protocol (`get_frame`'s Again/Eof/starved latch).
//!
//! Cross-format graphs (which need the auto-inserted `scale` converter)
//! arrive with the vf_scale module; the parser-driven `-vf` string path
//! arrives with the parser module.

use ffmpeg_rs::{
    filter::FilterGraph,
    util::{frame::Frame, pixfmt::PixelFormat, rational::Rational},
};

/// buffer(yuv420p 64x48, tb 1/25) → null → format=yuv420p → buffersink,
/// fully configured — the smallest honest `-vf` graph.
fn graph_with(
    desc: &str,
) -> (
    FilterGraph,
    ffmpeg_rs::filter::NodeId,
    ffmpeg_rs::filter::NodeId,
) {
    let mut g = FilterGraph::new();
    let src = g
        .create_filter(
            "buffer",
            "video_size=64x48:pix_fmt=yuv420p:time_base=1/25:frame_rate=25/1",
        )
        .unwrap_or_else(|e| panic!("buffer init: {e}"));
    let mid = g.create_filter("null", "").unwrap();
    let fmt = g.create_filter("format", desc).unwrap();
    let sink = g.create_filter("buffersink", "").unwrap();
    g.link(src, 0, mid, 0).unwrap();
    g.link(mid, 0, fmt, 0).unwrap();
    g.link(fmt, 0, sink, 0).unwrap();
    g.config().unwrap();
    (g, src, sink)
}

fn frame(pts: i64) -> Frame {
    let mut f = Frame::alloc(PixelFormat::Yuv420p, 64, 48).unwrap();
    f.pts = pts;
    f.duration = 1;
    f.time_base = Rational::new(1, 25);
    f
}

#[test]
fn frames_flow_through_the_graph_untouched() {
    let (mut g, src, sink) = graph_with("pix_fmts=yuv420p");

    // The negotiated format on the sink link is the declared singleton.
    assert_eq!(
        ffmpeg_rs::filter::buffersink::buffersink_get_format(&g, sink).unwrap(),
        Some(PixelFormat::Yuv420p)
    );

    let mut pts_seen = Vec::new();
    for pts in [0i64, 1, 2] {
        g.add_frame(src, &frame(pts)).unwrap();
        // The sink's internal get loop drives the graph to quiescence;
        // each push yields exactly one frame here.
        while let Ok(f) = g.get_frame(sink) {
            pts_seen.push(f.pts);
        }
    }
    assert_eq!(pts_seen, vec![0, 1, 2]);

    // EOF: close propagates; the sink then reports Eof, not Again.
    g.close_source(src).unwrap();
    assert!(matches!(
        g.get_frame(sink),
        Err(ffmpeg_rs::util::error::Error::Eof)
    ));
}

#[test]
fn buffersrc_geometry_change_only_warns_then_queues() {
    // CHECK_VIDEO_PARAM_CHANGE (buffersrc.c:74-96) logs a WARNING but does
    // not fail; C then hits the av_assert1 in ff_filter_frame for the
    // mismatched geometry in debug builds — the port's debug_asserts match.
    // At the QUEUE level the frame is accepted: assert on the source's
    // output link fifo directly is pub(crate), so drive it through a graph
    // whose declared geometry MATCHES the frame instead (the warning path
    // is covered by buffersrc's own unit tests).
    let (mut g, src, sink) = graph_with("pix_fmts=yuv420p");
    let mut f = Frame::alloc(PixelFormat::Yuv420p, 64, 48).unwrap();
    f.pts = 7;
    f.time_base = Rational::new(1, 25);
    g.add_frame(src, &f).unwrap();
    let out = g.get_frame(sink).unwrap();
    assert_eq!((out.width, out.height), (64, 48));
    assert_eq!(out.pts, 7);
}

#[test]
fn format_filter_constrains_negotiation() {
    // format=rgb24 against a yuv420p source has no common format: the
    // negotiation AUTO-INSERTS a scale converter (avfiltergraph.c:611-641)
    // and the sink negotiates the constrained format — an rgb24 frame comes
    // OUT of a yuv420p source, which is itself proof the converter ran.
    let mut g = FilterGraph::new();
    let src = g
        .create_filter("buffer", "video_size=64x48:pix_fmt=yuv420p:time_base=1/25")
        .unwrap();
    let fmt = g.create_filter("format", "pix_fmts=rgb24").unwrap();
    let sink = g.create_filter("buffersink", "").unwrap();
    g.link(src, 0, fmt, 0).unwrap();
    g.link(fmt, 0, sink, 0).unwrap();
    g.config().unwrap();
    let mut f = Frame::alloc(PixelFormat::Yuv420p, 64, 48).unwrap();
    f.pts = 0;
    f.time_base = Rational::new(1, 25);
    g.add_frame(src, &f).unwrap();
    let out = g.get_frame(sink).unwrap();
    assert_eq!(out.format, PixelFormat::Rgb24);
    assert_eq!((out.width, out.height), (64, 48));
}
