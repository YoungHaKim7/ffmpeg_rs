use ffmpeg_rs::util::log::{Level, get_level, set_level};

#[test]
fn level_names_round_trip() {
    // Canonical names are accepted; "warn" is the one alias.
    for name in [
        "quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug", "trace",
    ] {
        assert!(Level::from_name(name).is_some(), "{name}");
    }
    assert!(Level::from_name("loud").is_none());
    assert_eq!(Level::from_name("warn"), Some(Level::Warning));
}

#[test]
fn set_get_level() {
    let saved = get_level();
    set_level(Level::Debug);
    assert_eq!(get_level(), Level::Debug);
    set_level(saved);
}
