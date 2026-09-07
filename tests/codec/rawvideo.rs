use ffmpeg_rs::{
    Error, Frame, FrameFlags, PictureType, PixelFormat, Rational,
    codec::{
        CodecId, CodecParameters, MediaType, Packet, PacketFlags, RawVideoDecoder, RawVideoEncoder,
        traits::{Decoder, Encoder},
    },
};

fn stream_params() -> CodecParameters {
    let mut p = CodecParameters::default();
    p.codec_type = MediaType::Video;
    p.codec_id = CodecId::Rawvideo;
    p.format = PixelFormat::Yuv420p;
    p.width = 64;
    p.height = 48;
    p.framerate = Rational::new(25, 1);
    p
}

fn sample_packet(data: Vec<u8>, pts: i64) -> Packet {
    let mut pkt = Packet::from_vec(data);
    pkt.pts = pts;
    pkt.duration = 1;
    pkt.time_base = Rational::new(1, 25);
    pkt.flags = pkt.flags.union(PacketFlags::KEY);
    pkt
}

#[test]
fn decode_is_zero_copy_and_carries_props() {
    let mut dec = RawVideoDecoder::new();
    dec.init(&stream_params()).unwrap();

    let payload: Vec<u8> = (0..64 * 48 * 3 / 2).map(|i| (i % 251) as u8).collect();
    let pkt = sample_packet(payload, 7);
    dec.send_packet(Some(&pkt)).unwrap();
    let frame = dec.receive_frame().unwrap();

    assert_eq!((frame.width, frame.height), (64, 48));
    assert_eq!(frame.pts, 7);
    assert_eq!(frame.time_base, Rational::new(1, 25));
    assert!(frame.flags.contains(FrameFlags::KEY));
    assert_eq!(frame.pict_type, PictureType::I);
    // Plane shares the packet buffer.
    assert!(std::sync::Arc::ptr_eq(&frame.planes[0].buf, &pkt.data));
    assert_eq!(frame.plane(0)[5], 5);

    assert!(matches!(dec.receive_frame(), Err(Error::Again)));
    dec.send_packet(None).unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn decode_rejects_short_packets() {
    let mut dec = RawVideoDecoder::new();
    dec.init(&stream_params()).unwrap();
    let pkt = sample_packet(vec![0u8; 100], 0);
    assert!(matches!(
        dec.send_packet(Some(&pkt)),
        Err(Error::InvalidData(_))
    ));
}

#[test]
fn encode_decode_round_trip_is_byte_exact() {
    let params = stream_params();
    let mut enc = RawVideoEncoder::new();
    enc.init(&params).unwrap();

    let mut frame = Frame::alloc(PixelFormat::Yuv420p, 64, 48).unwrap();
    frame
        .plane_mut(0)
        .iter_mut()
        .enumerate()
        .for_each(|(i, b)| *b = (i % 251) as u8);
    frame
        .plane_mut(1)
        .iter_mut()
        .enumerate()
        .for_each(|(i, b)| *b = (i % 13) as u8);
    frame
        .plane_mut(2)
        .iter_mut()
        .enumerate()
        .for_each(|(i, b)| *b = (i % 7) as u8);
    frame.pts = 3;
    frame.duration = 1;
    frame.time_base = Rational::new(1, 25);

    enc.send_frame(Some(&frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();
    assert_eq!(pkt.size(), 64 * 48 * 3 / 2);
    assert_eq!(pkt.pts, 3);
    assert!(pkt.flags.contains(PacketFlags::KEY));
    assert!(matches!(enc.receive_packet(), Err(Error::Again)));
    enc.send_frame(None).unwrap();
    assert!(matches!(enc.receive_packet(), Err(Error::Eof)));

    let mut dec = RawVideoDecoder::new();
    dec.init(&params).unwrap();
    dec.send_packet(Some(&pkt)).unwrap();
    let back = dec.receive_frame().unwrap();
    assert_eq!(back.plane(0), frame.plane(0));
    assert_eq!(back.plane(1), frame.plane(1));
    assert_eq!(back.plane(2), frame.plane(2));
    assert_eq!(back.pts, 3);
}
