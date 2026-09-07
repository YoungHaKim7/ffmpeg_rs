use ffmpeg_rs::{
    NOPTS, Rational,
    codec::{Packet, PacketFlags},
};

#[test]
fn default_matches_c_zeroing() {
    let p = Packet::default();
    assert_eq!(p.pts, NOPTS);
    assert_eq!(p.dts, NOPTS);
    assert_eq!(p.time_base, Rational::UNKNOWN);
    assert_eq!(p.size(), 0);
}

#[test]
fn shares_payload_without_copy() {
    let p = Packet::from_vec(vec![7u8; 128]);
    let clone = p.clone();
    assert_eq!(clone.size(), 128);
    assert!(std::sync::Arc::ptr_eq(&p.data, &clone.data));
    assert_eq!(p.as_slice()[0], 7);
}

#[test]
fn key_flag_set_and_checked() {
    let mut p = Packet::default();
    assert!(!p.flags.contains(PacketFlags::KEY));
    p.flags = p.flags.union(PacketFlags::KEY);
    assert!(p.flags.contains(PacketFlags::KEY));
}
