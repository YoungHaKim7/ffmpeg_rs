use ffmpeg_rs::{
    Rational,
    codec::{CodecId, CodecParameters, MediaType},
};

#[test]
fn defaults_are_c_unspecified() {
    let p = CodecParameters::default();
    assert_eq!(p.codec_type, MediaType::Unknown);
    assert_eq!(p.codec_id, CodecId::None);
    assert_eq!(p.width, 0);
    assert_eq!(p.sample_aspect_ratio, Rational::UNKNOWN);
    assert_eq!(p.framerate, Rational::UNKNOWN);
}

#[test]
fn codec_names() {
    assert_eq!(CodecId::Rawvideo.name(), "rawvideo");
    assert_eq!(CodecId::WrappedAvframe.name(), "wrapped_avframe");
}
