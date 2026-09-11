use ffmpeg_rs::Error;
use ffmpeg_rs::util::channel_layout::Channel;
use ffmpeg_rs::util::channel_layout::ChannelLayout;
use ffmpeg_rs::util::channel_layout::Order;
use ffmpeg_rs::util::channel_layout::STANDARD;
use ffmpeg_rs::util::channel_layout::standard;

// ---- from_mask / check (c:253-264, c:785-809) ----
#[test]
fn from_mask_and_check() {
    let stereo = ChannelLayout::from_mask(0x3).unwrap();
    assert_eq!(stereo.order, Order::Native);
    assert_eq!(stereo.nb_channels, 2);
    assert_eq!(stereo.mask, 0x3);
    assert!(stereo.check());
    assert!(matches!(
        ChannelLayout::from_mask(0),
        Err(Error::InvalidArgument(_))
    ));

    // Hand-built inconsistent native layout fails check() (c:792).
    let bad = ChannelLayout {
        order: Order::Native,
        nb_channels: 3,
        mask: 0x3,
    };
    assert!(!bad.check());
    // Unspecified with any positive count is valid (c:804-805)…
    assert!(ChannelLayout::unspecified(2).check());
    // …but 0 channels is invalid for every order (c:787-788).
    assert!(!ChannelLayout::unspecified(0).check());
    assert!(!ChannelLayout::default().check());
}

// ---- layout consts vs the header macros (masks generated from the
// actual FFmpeg headers; see module doc) ----
#[test]
fn const_masks_match_header() {
    let expected: &[(&str, u64, usize)] = &[
        ("MONO", 0x4, 1),
        ("STEREO", 0x3, 2),
        ("TwoPointOne", 0xB, 3),
        ("2_1", 0x103, 3),
        ("SURROUND", 0x7, 3),
        ("ThreePointOne", 0xF, 4),
        ("FourPointZero", 0x107, 4),
        ("FourPointOne", 0x10F, 5),
        ("2_2", 0x603, 4),
        ("QUAD", 0x33, 4),
        ("FivePointZero", 0x607, 5),
        ("FivePointOne", 0x60F, 6),
        ("FivePointZeroBack", 0x37, 5),
        ("FivePointOneBack", 0x3F, 6),
        ("SixPointZero", 0x707, 6),
        ("SixPointZeroFront", 0x6C3, 6),
        ("ThreePointOnePointTwo", 0x500F, 6),
        ("HEXAGONAL", 0x137, 6),
        ("SixPointOne", 0x70F, 7),
        ("SixPointOneBack", 0x13F, 7),
        ("SixPointOneFront", 0x6CB, 7),
        ("SevenPointZero", 0x637, 7),
        ("SevenPointZeroFront", 0x6C7, 7),
        ("SevenPointOne", 0x63F, 8),
        ("SevenPointOneWide", 0x6CF, 8),
        ("SevenPointOneWideBack", 0xFF, 8),
        ("FivePointOnePointTwo", 0x560F, 8),
        ("FivePointOnePointTwoBack", 0x503F, 8),
        ("OCTAGONAL", 0x737, 8),
        ("CUBE", 0x2D033, 8),
        ("FivePointOnePointFour", 0x2D60F, 10),
        ("SevenPointOnePointTwo", 0x563F, 10),
        ("SevenPointOnePointFour", 0x2D63F, 12),
        ("SevenTwoThree", 0x80001563F, 12),
        ("NineOneFour", 0x2D6FF, 14),
        ("NineOneSix", 0x300002D6FF, 16),
        ("HEXADECAGONAL", 0x18003F737, 16),
        ("BINAURAL", 0x6000000000000000, 2),
        ("STEREO_DOWNMIX", 0x60000000, 2),
        ("TwentyTwoTwo", 0x1F80003FFFF, 24),
    ];
    let layouts: &[(&str, ChannelLayout)] = &[
        ("MONO", ChannelLayout::MONO),
        ("STEREO", ChannelLayout::STEREO),
        ("TwoPointOne", ChannelLayout::TwoPointOne),
        ("2_1", ChannelLayout::A2_1),
        ("SURROUND", ChannelLayout::SURROUND),
        ("ThreePointOne", ChannelLayout::ThreePointOne),
        ("FourPointZero", ChannelLayout::FourPointZero),
        ("FourPointOne", ChannelLayout::FourPointOne),
        ("2_2", ChannelLayout::A2_2),
        ("QUAD", ChannelLayout::QUAD),
        ("FivePointZero", ChannelLayout::FivePointZero),
        ("FivePointOne", ChannelLayout::FivePointOne),
        ("FivePointZeroBack", ChannelLayout::FivePointZeroBack),
        ("FivePointOneBack", ChannelLayout::FivePointOneBack),
        ("SixPointZero", ChannelLayout::SixPointZero),
        ("SixPointZeroFront", ChannelLayout::SixPointZeroFront),
        (
            "ThreePointOnePointTwo",
            ChannelLayout::ThreePointOnePointTwo,
        ),
        ("HEXAGONAL", ChannelLayout::HEXAGONAL),
        ("SixPointOne", ChannelLayout::SixPointOne),
        ("SixPointOneBack", ChannelLayout::SixPointOneBack),
        ("SixPointOneFront", ChannelLayout::SixPointOneFront),
        ("SevenPointZero", ChannelLayout::SevenPointZero),
        ("SevenPointZeroFront", ChannelLayout::SevenPointZeroFront),
        ("SevenPointOne", ChannelLayout::SevenPointOne),
        ("SevenPointOneWide", ChannelLayout::SevenPointOneWide),
        (
            "SevenPointOneWideBack",
            ChannelLayout::SevenPointOneWideBack,
        ),
        ("FivePointOnePointTwo", ChannelLayout::FivePointOnePointTwo),
        (
            "FivePointOnePointTwoBack",
            ChannelLayout::FivePointOnePointTwoBack,
        ),
        ("OCTAGONAL", ChannelLayout::OCTAGONAL),
        ("CUBE", ChannelLayout::CUBE),
        (
            "FivePointOnePointFour",
            ChannelLayout::FivePointOnePointFour,
        ),
        (
            "SevenPointOnePointTwo",
            ChannelLayout::SevenPointOnePointTwo,
        ),
        (
            "SevenPointOnePointFour",
            ChannelLayout::SevenPointOnePointFour,
        ),
        ("SevenTwoThree", ChannelLayout::SevenTwoThree),
        ("NineOneFour", ChannelLayout::NineOneFour),
        ("NineOneSix", ChannelLayout::NineOneSix),
        ("HEXADECAGONAL", ChannelLayout::HEXADECAGONAL),
        ("BINAURAL", ChannelLayout::BINAURAL),
        ("STEREO_DOWNMIX", ChannelLayout::STEREO_DOWNMIX),
        ("TwentyTwoTwo", ChannelLayout::TwentyTwoTwo),
    ];
    assert_eq!(expected.len(), 40);
    assert_eq!(layouts.len(), 40);
    for ((name_exp, mask_exp, nb_exp), (name, layout)) in expected.iter().zip(layouts) {
        assert_eq!(name_exp, name);
        assert_eq!(layout.mask, *mask_exp, "{name}");
        assert_eq!(layout.nb_channels, *nb_exp, "{name}");
        assert_eq!(layout.order, Order::Native, "{name}");
        // nb must equal popcount for every const.
        assert_eq!(
            layout.mask.count_ones() as usize,
            layout.nb_channels,
            "{name}"
        );
        assert!(layout.check(), "{name}");
    }
    // Every const must appear in STANDARD and standard() with its
    // table name (order pinned separately below).
    assert_eq!(STANDARD.len(), 40);
    assert_eq!(standard().len(), 40);
    for (name, layout) in STANDARD {
        assert_eq!(
            layout.mask.count_ones() as usize,
            layout.nb_channels,
            "{name}"
        );
    }
    // Spec anchors.
    assert_eq!(ChannelLayout::MONO.mask, 0x4);
    assert_eq!(ChannelLayout::STEREO.mask, 0x3);
    assert_eq!(ChannelLayout::FivePointZeroBack.mask, 0x37);
    assert_eq!(ChannelLayout::FivePointOneBack.mask, 0x3F);
    assert_eq!(ChannelLayout::SixPointOne.mask, 0x70F);
    assert_eq!(ChannelLayout::SixPointOneBack.mask, 0x13F);
    assert_eq!(ChannelLayout::SevenPointOne.mask, 0x63F);
    assert_eq!(ChannelLayout::QUAD.mask, 0x33);
    assert_eq!(ChannelLayout::TwentyTwoTwo.nb_channels, 24);
}

// ---- STANDARD table order is the C channel_layout_map order ----
#[test]
fn standard_table_order() {
    let names: Vec<&str> = STANDARD.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        vec![
            "mono",
            "stereo",
            "2.1",
            "3.0",
            "3.0(back)",
            "4.0",
            "quad",
            "quad(side)",
            "3.1",
            "5.0",
            "5.0(side)",
            "4.1",
            "5.1",
            "5.1(side)",
            "6.0",
            "6.0(front)",
            "3.1.2",
            "hexagonal",
            "6.1",
            "6.1(back)",
            "6.1(front)",
            "7.0",
            "7.0(front)",
            "7.1",
            "7.1(wide)",
            "7.1(wide-side)",
            "5.1.2",
            "5.1.2(back)",
            "octagonal",
            "cube",
            "5.1.4",
            "7.1.2",
            "7.1.4",
            "7.2.3",
            "9.1.4",
            "9.1.6",
            "hexadecagonal",
            "binaural",
            "downmix",
            "22.2",
        ]
    );
    // The swap quirks (c:200, c:215).
    assert_eq!(ChannelLayout::from_string("5.0").unwrap().mask, 0x37);
    assert_eq!(ChannelLayout::from_string("7.1(wide)").unwrap().mask, 0xFF);
    assert_eq!(
        ChannelLayout::from_string("7.1(wide-side)").unwrap().mask,
        0x6CF
    );
    for (i, ((name, layout), std)) in STANDARD.iter().zip(standard()).enumerate() {
        assert_eq!(std, layout, "standard() index {i} ({name})");
    }
}

// ---- describe (c:600-652) ----
#[test]
fn describe_native() {
    assert_eq!(ChannelLayout::MONO.describe(), "mono");
    assert_eq!(ChannelLayout::STEREO.describe(), "stereo");
    assert_eq!(ChannelLayout::FivePointOneBack.describe(), "5.1");
    assert_eq!(ChannelLayout::FivePointOne.describe(), "5.1(side)");
    assert_eq!(ChannelLayout::QUAD.describe(), "quad");
    assert_eq!(ChannelLayout::A2_2.describe(), "quad(side)");
    assert_eq!(ChannelLayout::TwentyTwoTwo.describe(), "22.2");
    // Non-standard masks: "{nb} channels (A+B+…)", ascending bits.
    assert_eq!(
        ChannelLayout::from_mask(0x63).unwrap().describe(),
        "4 channels (FL+FR+BR+FLC)"
    );
    assert_eq!(
        ChannelLayout::from_mask(0x5).unwrap().describe(),
        "2 channels (FL+FC)"
    );
    // Count 0 falls through to the unspecified text (c:643-645).
    assert_eq!(
        ChannelLayout {
            order: Order::Native,
            nb_channels: 0,
            mask: 0
        }
        .describe(),
        "0 channels"
    );
}

#[test]
fn describe_unspecified() {
    assert_eq!(ChannelLayout::unspecified(2).describe(), "2 channels");
    assert_eq!(ChannelLayout::unspecified(7).describe(), "7 channels");
    // Display is describe().
    assert_eq!(ChannelLayout::unspecified(2).to_string(), "2 channels");
}

// ---- from_string (c:313-441) ----
#[test]
fn from_string_names() {
    assert_eq!(
        ChannelLayout::from_string("mono").unwrap(),
        ChannelLayout::MONO
    );
    assert_eq!(
        ChannelLayout::from_string("stereo").unwrap(),
        ChannelLayout::STEREO
    );
    assert_eq!(
        ChannelLayout::from_string("5.1").unwrap(),
        ChannelLayout::FivePointOneBack
    );
    assert_eq!(
        ChannelLayout::from_string("quad(side)").unwrap(),
        ChannelLayout::A2_2
    );
    assert!(matches!(
        ChannelLayout::from_string("Mono"),
        Err(Error::InvalidArgument(_))
    )); // case-sensitive
}

#[test]
fn from_string_channel_lists() {
    let l = ChannelLayout::from_string("FL+FR+LFE").unwrap();
    assert_eq!(l.mask, 0xB); // == TwoPointOne
    assert_eq!(l, ChannelLayout::TwoPointOne);
    let l = ChannelLayout::from_string("FL+FC").unwrap();
    assert_eq!(l.mask, 0x5);
    assert_eq!(l.nb_channels, 2);
    // Wrapped form with matching count parses (c:398, c:406).
    let l = ChannelLayout::from_string("2 channels (FL+FC)").unwrap();
    assert_eq!(l.mask, 0x5);
    // Count mismatch is terminal EINVAL (c:406-409).
    assert!(matches!(
        ChannelLayout::from_string("3 channels (FL+FC)"),
        Err(Error::InvalidArgument(_))
    ));
    // Out-of-order / duplicate lists are CUSTOM order in C -> not
    // ported (c:307 + c:463-474).
    assert!(matches!(
        ChannelLayout::from_string("FR+FL"),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string("FL+FL"),
        Err(Error::Unsupported(_))
    ));
    // Custom '@' names are CUSTOM-only (c:275, c:286).
    assert!(matches!(
        ChannelLayout::from_string("FL@Left"),
        Err(Error::Unsupported(_))
    ));
    // UNK in a list: C builds a CUSTOM map; not ported.
    assert!(matches!(
        ChannelLayout::from_string("FL+UNK"),
        Err(Error::Unsupported(_))
    ));
    // AMBI* tokens: unresolvable in the closed enum, C would accept —
    // comes out as plain invalid after fall-through.
    assert!(matches!(
        ChannelLayout::from_string("FL+AMBI0"),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn from_string_masks_and_numbers() {
    // Hex mask.
    let l = ChannelLayout::from_string("0x63").unwrap();
    assert_eq!(l.order, Order::Native);
    assert_eq!(l.mask, 0x63);
    assert_eq!(l.nb_channels, 4); // FL|FR|BR|FLC
    assert_eq!(l.describe(), "4 channels (FL+FR+BR+FLC)");
    // Decimal mask: 4 = bit 2 = FC == mono layout.
    assert_eq!(
        ChannelLayout::from_string("4").unwrap(),
        ChannelLayout::MONO
    );
    // Octal via strtoull base 0: "010" == 8 == bit 3 == LFE.
    let l = ChannelLayout::from_string("010").unwrap();
    assert_eq!(l.mask, 0x8);
    assert_eq!(l.nb_channels, 1);
    assert_eq!(l.describe(), "1 channels (LFE)");
    // "09" is not a valid octal number and not fully decimal-parsable.
    assert!(matches!(
        ChannelLayout::from_string("09"),
        Err(Error::InvalidArgument(_))
    ));
    // "0" / "0x0" reject the zero mask (c:417 mask != 0).
    assert!(matches!(
        ChannelLayout::from_string("0"),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string("0x0"),
        Err(Error::InvalidArgument(_))
    ));
    // "0x" and trailing junk do not fully consume.
    assert!(matches!(
        ChannelLayout::from_string("0x"),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string("0x63junk"),
        Err(Error::InvalidArgument(_))
    ));
    // '-' anywhere rejects the mask branch (c:417) and nothing else
    // matches.
    assert!(matches!(
        ChannelLayout::from_string("-4"),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string("0x-3"),
        Err(Error::InvalidArgument(_))
    ));
    // Overflow behaves like C's ERANGE (branch skipped).
    assert!(matches!(
        ChannelLayout::from_string("0xFFFFFFFFFFFFFFFFF"),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn from_string_channel_counts() {
    // "{n}c": default layout, only when native (c:426-429).
    assert_eq!(
        ChannelLayout::from_string("6c").unwrap(),
        ChannelLayout::FivePointOneBack
    );
    assert_eq!(
        ChannelLayout::from_string("2c").unwrap(),
        ChannelLayout::STEREO
    );
    // "11c" has no standard layout -> NOT unspecified; real FFmpeg
    // falls through to EINVAL. Pin so nobody "fixes" it.
    assert!(matches!(
        ChannelLayout::from_string("11c"),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string("9c"),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string("0c"),
        Err(Error::InvalidArgument(_))
    ));
    // "{n}C" and "{n} channels": unspecified (c:433-437).
    assert_eq!(
        ChannelLayout::from_string("2C").unwrap(),
        ChannelLayout::unspecified(2)
    );
    assert_eq!(
        ChannelLayout::from_string("4 channels").unwrap(),
        ChannelLayout::unspecified(4)
    );
    // Channel-list parse of "4 channels" fails (unresolvable token)
    // and falls through to branch 6, exactly like C.
    assert_eq!(
        ChannelLayout::from_string("2 channels").unwrap(),
        ChannelLayout::unspecified(2)
    );
    // Ambisonic dropped (c:334-391).
    assert!(matches!(
        ChannelLayout::from_string("ambisonic 1"),
        Err(Error::Unsupported(_))
    ));
    // Final EINVAL (c:440).
    assert!(matches!(
        ChannelLayout::from_string("bogus"),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ChannelLayout::from_string(""),
        Err(Error::InvalidArgument(_))
    ));
}

// ---- describe/from_string round-trip over every standard entry ----
#[test]
fn describe_from_string_round_trip() {
    for (name, layout) in STANDARD {
        assert_eq!(&layout.describe(), name, "{name}");
        assert_eq!(ChannelLayout::from_string(name).unwrap(), *layout, "{name}");
    }
    // …including the "N channels (…)" re-parse for non-standard masks.
    for mask in [0x5u64, 0x63, 0x1] {
        let layout = ChannelLayout::from_mask(mask).unwrap();
        assert_eq!(
            ChannelLayout::from_string(&layout.describe()).unwrap(),
            layout,
            "mask {mask:#x}"
        );
    }
    // Unspecified round-trips via "N channels".
    let u = ChannelLayout::unspecified(5);
    assert_eq!(ChannelLayout::from_string(&u.describe()).unwrap(), u);
}

// ---- default_for (c:841-852) ----
#[test]
fn default_for_first_match() {
    assert_eq!(ChannelLayout::default_for(1).mask, 0x4);
    assert_eq!(ChannelLayout::default_for(2).mask, 0x3);
    assert_eq!(ChannelLayout::default_for(3).mask, 0xB); // 2.1
    assert_eq!(ChannelLayout::default_for(4).mask, 0x107); // 4.0, NOT quad
    assert_eq!(ChannelLayout::default_for(5).mask, 0x37); // 5.0 = _BACK
    assert_eq!(ChannelLayout::default_for(6).mask, 0x3F); // 5.1 = _BACK
    assert_eq!(ChannelLayout::default_for(7).mask, 0x70F); // 6.1 (side)
    assert_eq!(ChannelLayout::default_for(8).mask, 0x63F); // 7.1
    assert_eq!(
        ChannelLayout::default_for(10),
        ChannelLayout::FivePointOnePointFour
    );
    assert_eq!(
        ChannelLayout::default_for(12),
        ChannelLayout::SevenPointOnePointFour
    );
    assert_eq!(ChannelLayout::default_for(14), ChannelLayout::NineOneFour);
    assert_eq!(ChannelLayout::default_for(16), ChannelLayout::NineOneSix);
    assert_eq!(ChannelLayout::default_for(24), ChannelLayout::TwentyTwoTwo);
    // No standard layout: unspecified (c:850-851).
    assert_eq!(ChannelLayout::default_for(9), ChannelLayout::unspecified(9));
    assert_eq!(ChannelLayout::default_for(0), ChannelLayout::unspecified(0));
    assert!(!ChannelLayout::default_for(0).check());
}

// ---- index lookups (c:673-747) and subset (c:867-885) ----
#[test]
fn index_round_trip() {
    let stereo = ChannelLayout::STEREO;
    assert_eq!(stereo.index_from_channel(Channel::FrontLeft), Ok(0));
    assert_eq!(stereo.index_from_channel(Channel::FrontRight), Ok(1));
    assert!(stereo.index_from_channel(Channel::LowFrequency).is_err());
    assert_eq!(stereo.channel_from_index(0), Some(Channel::FrontLeft));
    assert_eq!(stereo.channel_from_index(1), Some(Channel::FrontRight));
    assert_eq!(stereo.channel_from_index(2), None);

    let five1 = ChannelLayout::FivePointOneBack;
    assert_eq!(five1.index_from_channel(Channel::LowFrequency), Ok(3));
    assert!(five1.index_from_channel(Channel::SideLeft).is_err());

    // channel_from_index <-> index_from_channel inverse over all bits.
    for layout in standard() {
        for i in 0..layout.nb_channels {
            let ch = layout.channel_from_index(i).unwrap();
            assert_eq!(layout.index_from_channel(ch), Ok(i));
        }
    }

    // None channel and unspecified order are EINVAL (c:720-721,
    // c:744-745).
    assert!(stereo.index_from_channel(Channel::None).is_err());
    assert!(
        ChannelLayout::unspecified(2)
            .index_from_channel(Channel::FrontLeft)
            .is_err()
    );
    assert_eq!(ChannelLayout::unspecified(2).channel_from_index(0), None);

    // Ascending bit order for a sparse mask (c:692-696).
    let m = ChannelLayout::from_mask(0x63).unwrap();
    assert_eq!(m.channel_from_index(0), Some(Channel::FrontLeft));
    assert_eq!(m.channel_from_index(2), Some(Channel::BackRight));
    assert_eq!(m.channel_from_index(3), Some(Channel::FrontLeftOfCenter));
}

#[test]
fn string_index_lookups() {
    let stereo = ChannelLayout::STEREO;
    assert_eq!(stereo.index_from_string("FR"), Ok(1));
    assert_eq!(stereo.channel_from_string("FR"), Some(Channel::FrontRight));
    assert!(stereo.index_from_string("bogus").is_err());
    // '@' needs CUSTOM order -> Unsupported (C's native arm would
    // EINVAL; documented divergence).
    assert!(matches!(
        stereo.index_from_string("FL@L"),
        Err(Error::Unsupported(_))
    ));
    // Unspecified has no positional info (c:749-783 -> c:782 EINVAL).
    assert!(
        ChannelLayout::unspecified(2)
            .index_from_string("FL")
            .is_err()
    );
    assert_eq!(
        ChannelLayout::unspecified(2).channel_from_string("FL"),
        None
    );
}

#[test]
fn subset_and_union() {
    assert_eq!(ChannelLayout::FivePointOneBack.subset(0xF), 0xF);
    assert_eq!(ChannelLayout::STEREO.subset(0xFFFFFFFFFFFFFFFF), 0x3);
    assert_eq!(ChannelLayout::unspecified(2).subset(0xF), 0);
    // Union (not a C API): mask OR, count = popcount.
    let u = ChannelLayout::STEREO.union(&ChannelLayout::MONO).unwrap();
    assert_eq!(u.mask, 0x7);
    assert_eq!(u.nb_channels, 3);
    assert!(
        ChannelLayout::unspecified(2)
            .union(&ChannelLayout::STEREO)
            .is_err()
    );
}

// ---- PartialEq == av_channel_layout_compare (c:811-839) ----
#[test]
fn compare_semantics() {
    assert_ne!(ChannelLayout::STEREO, ChannelLayout::unspecified(2));
    assert_eq!(ChannelLayout::unspecified(2), ChannelLayout::unspecified(2));
    assert_eq!(
        ChannelLayout::from_mask(0x3).unwrap(),
        ChannelLayout::STEREO
    );
    assert_ne!(ChannelLayout::STEREO, ChannelLayout::MONO);
    assert_ne!(
        ChannelLayout::FivePointZero,
        ChannelLayout::FivePointZeroBack
    );
}

// ---- Channel names/descriptions (c:48-133) ----
#[test]
fn channel_names() {
    assert_eq!(Channel::FrontLeft.name(), "FL");
    assert_eq!(Channel::LowFrequency.name(), "LFE");
    // Dolby downmix pair, bits 29/30 — NOT front left/right.
    assert_eq!(Channel::StereoLeft.name(), "DL");
    assert_eq!(Channel::StereoRight as i32, 30);
    assert_eq!(Channel::StereoLeft as i32, 29);
    assert_eq!(Channel::BinauralLeft as i32, 61);
    assert_eq!(Channel::Unused as i32, 0x200);
    assert_eq!(Channel::Unknown as i32, 0x300);
    assert_eq!(Channel::AmbisonicBase as i32, 0x400);
    assert_eq!(Channel::AmbisonicEnd as i32, 0x7ff);
    assert_eq!(Channel::Unknown.name(), "UNK");
    assert_eq!(Channel::Unused.name(), "UNSD");
    assert_eq!(Channel::None.name(), "NONE");
    assert_eq!(Channel::FrontLeft.description(), "front left");
    assert_eq!(Channel::StereoLeft.description(), "downmix left");
    assert_eq!(Channel::Unknown.description(), "unknown");

    // from_name exact + case-sensitive (c:153-183).
    assert_eq!(Channel::from_name("FL"), Some(Channel::FrontLeft));
    assert_eq!(Channel::from_name("LFE2"), Some(Channel::LowFrequency2));
    assert_eq!(Channel::from_name("UNK"), Some(Channel::Unknown));
    assert_eq!(Channel::from_name("UNSD"), Some(Channel::Unused));
    assert_eq!(Channel::from_name("fl"), None);
    assert_eq!(Channel::from_name("FL "), None);
    // AMBI*/USR* dropped (c:159-164, c:175-178).
    assert_eq!(Channel::from_name("AMBI0"), None);
    assert_eq!(Channel::from_name("USR5"), None);

    // mask_bit: positional only (c:468, c:739).
    assert_eq!(Channel::FrontLeft.mask_bit(), Some(1));
    assert_eq!(Channel::BackCenter.mask_bit(), Some(1 << 8));
    assert_eq!(Channel::BinauralRight.mask_bit(), Some(1 << 62));
    assert_eq!(Channel::None.mask_bit(), None);
    assert_eq!(Channel::Unknown.mask_bit(), None);
    assert_eq!(Channel::Unused.mask_bit(), None);
    assert_eq!(Channel::AmbisonicBase.mask_bit(), None);

    // Display is name().
    assert_eq!(Channel::FrontLeft.to_string(), "FL");
}
