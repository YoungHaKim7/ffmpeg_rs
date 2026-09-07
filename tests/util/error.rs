use ffmpeg_rs::Error;

#[test]
fn eof_and_again_display_like_av_strerror() {
    // av_strerror(AVERROR_EOF) == "End of file"
    assert_eq!(Error::Eof.to_string(), "End of file");
    // av_strerror(AVERROR(EAGAIN)) starts "Resource temporarily"
    assert!(Error::Again.to_string().starts_with("Resource temporarily"));
}

#[test]
fn invalid_data_carries_context() {
    let e = Error::InvalidData("YUV4MPEG stream contains an unknown pixel format.".into());
    assert!(e.to_string().contains("unknown pixel format"));
}

#[test]
fn io_errors_convert() {
    let e: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "nope").into();
    assert!(matches!(e, Error::Io(_)));
}
