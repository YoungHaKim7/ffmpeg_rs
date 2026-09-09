//! Audio channel layouts — port of `libavutil/channel_layout.{h,c}`
//! (native-order core).
//!
//! ## C → Rust map
//!
//! | C | here |
//! |---|---|
//! | `enum AVChannel` (`channel_layout.h:47-112`) | [`Channel`] |
//! | `AV_CH_*` bit macros (`h:175-210`) | [`Channel::mask_bit`] |
//! | `channel_names[]` (`c:48-85`) | [`Channel::name`]/[`Channel::description`] |
//! | `av_channel_name` (`c:87-118`) | [`Channel::name`] (bprint plumbing obsolete) |
//! | `av_channel_description` (`c:120-151`) | [`Channel::description`] |
//! | `av_channel_from_string` (`c:153-183`) | [`Channel::from_name`] |
//! | `enum AVChannelOrder` (`h:114-160`) | [`Order`] |
//! | `AVChannelLayout` (`h:328-386`) | [`ChannelLayout`] |
//! | `AV_CH_LAYOUT_*` (`h:217-256`) + `AV_CHANNEL_LAYOUT_*` (`h:403-442`) | [`ChannelLayout`] consts (`MONO` … `TwentyTwoTwo`) |
//! | `channel_layout_map[]` (`c:190-231`) | [`STANDARD`] (order is behavior) |
//! | `av_channel_layout_from_mask` (`c:253-264`) | [`ChannelLayout::from_mask`] |
//! | `av_channel_layout_default` (`c:841-852`) | [`ChannelLayout::default_for`] |
//! | `av_channel_layout_standard` (`c:854-865`) | [`standard`] |
//! | `av_channel_layout_describe` (`c:600-671`) | [`ChannelLayout::describe`] |
//! | `av_channel_layout_from_string` (`c:313-441`) | [`ChannelLayout::from_string`] |
//! | `av_channel_layout_channel_from_index` (`c:673-701`) | [`ChannelLayout::channel_from_index`] |
//! | `av_channel_layout_index_from_channel` (`c:715-747`) | [`ChannelLayout::index_from_channel`] |
//! | `av_channel_layout_index_from_string` (`c:749-783`) | [`ChannelLayout::index_from_string`] |
//! | `av_channel_layout_channel_from_string` (`c:703-713`) | [`ChannelLayout::channel_from_string`] |
//! | `av_channel_layout_subset` (`c:867-885`) | [`ChannelLayout::subset`] |
//! | `av_channel_layout_check` (`c:785-809`) | [`ChannelLayout::check`] |
//! | `av_channel_layout_compare` (`c:811-839`) | derived `PartialEq` (see struct doc) |
//!
//! ## Not ported (every producer returns `Error::Unsupported`)
//!
//! * **`AV_CHANNEL_ORDER_CUSTOM`** (`h:132-131`… `h:126-131`): explicit
//!   per-channel maps (`u.map`, `AVChannelCustom`, `h:292-296`) are
//!   unrepresentable in a closed enum + mask. C's
//!   `av_channel_layout_from_string` *succeeds* with a custom map for
//!   non-increasing channel lists ("FR+FL"), duplicate channels,
//!   `UNK`/`UNSD`/`AMBI*` list entries, `@`-named channels
//!   (`c:266-311`) and `av_channel_layout_custom_init` (`c:233-251`); this
//!   port returns [`Error::Unsupported`] at exactly those sites, per call.
//! * **`AV_CHANNEL_ORDER_AMBISONIC`** (`h:133-155`): `ambisonic N` strings
//!   (`c:334-391`), `av_channel_layout_ambisonic_order` (`c:486-526`),
//!   `try_describe_ambisonic` (`c:560-598`), `av_channel_layout_retype`
//!   (`c:887-979`) and `AV_CHANNEL_LAYOUT_AMBISONIC_FIRST_ORDER`
//!   (`h:455-459`) are dropped. [`Channel::AmbisonicBase`]/[`End`] exist
//!   only to keep C's id domain complete (they never enter a mask).
//! * `Channel::from_name`'s `"AMBI<n>"` (`c:159-164`) and `"USR<n>"`
//!   (`c:175-178`) forms return `None`: the ids they produce are
//!   unrepresentable here. In C they are valid channel ids; callers that
//!   must distinguish that case match on the error/`None` — documented
//!   degradation.
//! * Deprecated aliases `AV_CH_LAYOUT_*_BACK` / `AV_CHANNEL_LAYOUT_*_BACK`
//!   (`h:258-267`, `h:444-453`) — spelling aliases only.
//! * `av_channel_layout_copy` (`c:450-461`) / `av_channel_layout_uninit`
//!   (`c:443-448`) — Rust move semantics (the struct is `Copy`).
//! * `enum AVMatrixEncoding` (`h:273-282`) — rematrix phase concern.
//!
//! ## Divergences from C's string parsing (documented per call)
//!
//! C's `strtoull`/`strtol` skip leading ASCII whitespace and accept a `+`
//! sign (`" 4"` parses as 4); [`ChannelLayout::from_string`] does not —
//! no ported caller produces such strings. Everything else follows
//! `strtoull(base 0)` semantics exactly, including octal `"010"` = 8.

use super::error::{Error, Result};

/// `enum AVChannel` (`channel_layout.h:47-112`) — exact C discriminants,
/// including the unused gaps 19–28 (before `StereoLeft = 29`, `h:69`) and
/// 45–60 (before `BinauralLeft = 61`, `h:87`), and the special ids past
/// the mask range. There is **no** `AV_CHAN_NB` sentinel in C — none here.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    /// `AV_CHAN_NONE` = −1 — invalid channel index.
    None = -1,
    /// `AV_CHAN_FRONT_LEFT` = 0.
    FrontLeft = 0,
    /// `AV_CHAN_FRONT_RIGHT` = 1.
    FrontRight = 1,
    /// `AV_CHAN_FRONT_CENTER` = 2.
    FrontCenter = 2,
    /// `AV_CHAN_LOW_FREQUENCY` = 3.
    LowFrequency = 3,
    /// `AV_CHAN_BACK_LEFT` = 4.
    BackLeft = 4,
    /// `AV_CHAN_BACK_RIGHT` = 5.
    BackRight = 5,
    /// `AV_CHAN_FRONT_LEFT_OF_CENTER` = 6.
    FrontLeftOfCenter = 6,
    /// `AV_CHAN_FRONT_RIGHT_OF_CENTER` = 7.
    FrontRightOfCenter = 7,
    /// `AV_CHAN_BACK_CENTER` = 8.
    BackCenter = 8,
    /// `AV_CHAN_SIDE_LEFT` = 9.
    SideLeft = 9,
    /// `AV_CHAN_SIDE_RIGHT` = 10.
    SideRight = 10,
    /// `AV_CHAN_TOP_CENTER` = 11.
    TopCenter = 11,
    /// `AV_CHAN_TOP_FRONT_LEFT` = 12.
    TopFrontLeft = 12,
    /// `AV_CHAN_TOP_FRONT_CENTER` = 13.
    TopFrontCenter = 13,
    /// `AV_CHAN_TOP_FRONT_RIGHT` = 14.
    TopFrontRight = 14,
    /// `AV_CHAN_TOP_BACK_LEFT` = 15.
    TopBackLeft = 15,
    /// `AV_CHAN_TOP_BACK_CENTER` = 16.
    TopBackCenter = 16,
    /// `AV_CHAN_TOP_BACK_RIGHT` = 17.
    TopBackRight = 17,
    /// `AV_CHAN_STEREO_LEFT` = 29 — Dolby downmix left (gap 19–28 unused).
    StereoLeft = 29,
    /// `AV_CHAN_STEREO_RIGHT` = 30 — Dolby downmix right.
    StereoRight = 30,
    /// `AV_CHAN_WIDE_LEFT` = 31.
    WideLeft = 31,
    /// `AV_CHAN_WIDE_RIGHT` = 32.
    WideRight = 32,
    /// `AV_CHAN_SURROUND_DIRECT_LEFT` = 33.
    SurroundDirectLeft = 33,
    /// `AV_CHAN_SURROUND_DIRECT_RIGHT` = 34.
    SurroundDirectRight = 34,
    /// `AV_CHAN_LOW_FREQUENCY_2` = 35.
    LowFrequency2 = 35,
    /// `AV_CHAN_TOP_SIDE_LEFT` = 36.
    TopSideLeft = 36,
    /// `AV_CHAN_TOP_SIDE_RIGHT` = 37.
    TopSideRight = 37,
    /// `AV_CHAN_BOTTOM_FRONT_CENTER` = 38.
    BottomFrontCenter = 38,
    /// `AV_CHAN_BOTTOM_FRONT_LEFT` = 39.
    BottomFrontLeft = 39,
    /// `AV_CHAN_BOTTOM_FRONT_RIGHT` = 40.
    BottomFrontRight = 40,
    /// `AV_CHAN_SIDE_SURROUND_LEFT` = 41 (+90°, Lss/SiL).
    SideSurroundLeft = 41,
    /// `AV_CHAN_SIDE_SURROUND_RIGHT` = 42 (−90°, Rss/SiR).
    SideSurroundRight = 42,
    /// `AV_CHAN_TOP_SURROUND_LEFT` = 43 (+110°, Lvs/TpLS).
    TopSurroundLeft = 43,
    /// `AV_CHAN_TOP_SURROUND_RIGHT` = 44 (−110°, Rvs/TpRS).
    TopSurroundRight = 44,
    /// `AV_CHAN_BINAURAL_LEFT` = 61 (gap 45–60 unused).
    BinauralLeft = 61,
    /// `AV_CHAN_BINAURAL_RIGHT` = 62.
    BinauralRight = 62,
    /// `AV_CHAN_UNUSED` = 0x200 — channel is empty, safely skipped.
    Unused = 0x200,
    /// `AV_CHAN_UNKNOWN` = 0x300 — has data, position unknown.
    Unknown = 0x300,
    /// `AV_CHAN_AMBISONIC_BASE` = 0x400 — first ambisonic ACN id
    /// (CUSTOM-order only in C; ambisonic not ported).
    AmbisonicBase = 0x400,
    /// `AV_CHAN_AMBISONIC_END` = 0x7ff — last ambisonic ACN id.
    AmbisonicEnd = 0x7ff,
}

impl Channel {
    /// The `AV_CH_*` bit of this channel (`channel_layout.h:175-210`):
    /// `Some(1 << position)` for the 36 positional variants (values 0..=62 —
    /// C only allows `ch < 63` in masks, `c:468`, `c:739`), `None` for the
    /// specials `None`/`Unused`/`Unknown`/`AmbisonicBase`/`AmbisonicEnd`.
    pub const fn mask_bit(self) -> Option<u64> {
        let v = self as i32;
        if v >= 0 && v < 63 {
            Some(1u64 << v)
        } else {
            None
        }
    }

    /// `av_channel_name` (`c:87-118`) — `channel_names[].name` (`c:48-85`)
    /// for positional channels; `"NONE"`/`"UNK"`/`"UNSD"` for the specials
    /// (`c:95-100`). C's `"USR%d"` and `"AMBI%d"` fallbacks are
    /// unrepresentable here (closed enum — see module doc).
    pub const fn name(self) -> &'static str {
        match self {
            Channel::FrontLeft => "FL",
            Channel::FrontRight => "FR",
            Channel::FrontCenter => "FC",
            Channel::LowFrequency => "LFE",
            Channel::BackLeft => "BL",
            Channel::BackRight => "BR",
            Channel::FrontLeftOfCenter => "FLC",
            Channel::FrontRightOfCenter => "FRC",
            Channel::BackCenter => "BC",
            Channel::SideLeft => "SL",
            Channel::SideRight => "SR",
            Channel::TopCenter => "TC",
            Channel::TopFrontLeft => "TFL",
            Channel::TopFrontCenter => "TFC",
            Channel::TopFrontRight => "TFR",
            Channel::TopBackLeft => "TBL",
            Channel::TopBackCenter => "TBC",
            Channel::TopBackRight => "TBR",
            // Dolby downmix pair — NOT front left/right.
            Channel::StereoLeft => "DL",
            Channel::StereoRight => "DR",
            Channel::WideLeft => "WL",
            Channel::WideRight => "WR",
            Channel::SurroundDirectLeft => "SDL",
            Channel::SurroundDirectRight => "SDR",
            Channel::LowFrequency2 => "LFE2",
            Channel::TopSideLeft => "TSL",
            Channel::TopSideRight => "TSR",
            Channel::BottomFrontCenter => "BFC",
            Channel::BottomFrontLeft => "BFL",
            Channel::BottomFrontRight => "BFR",
            Channel::SideSurroundLeft => "SSL",
            Channel::SideSurroundRight => "SSR",
            Channel::TopSurroundLeft => "TTL",
            Channel::TopSurroundRight => "TTR",
            Channel::BinauralLeft => "BIL",
            Channel::BinauralRight => "BIR",
            // c:95-100
            Channel::None => "NONE",
            Channel::Unknown => "UNK",
            Channel::Unused => "UNSD",
            // Unreachable in ported paths (ambisonic dropped); kept total
            // so name() needs no Option.
            Channel::AmbisonicBase | Channel::AmbisonicEnd => "",
        }
    }

    /// `av_channel_description` (`c:120-151`) — `channel_names[].description`
    /// (`c:48-85`); `"none"`/`"unknown"`/`"unused"` for the specials
    /// (`c:128-133`).
    pub const fn description(self) -> &'static str {
        match self {
            Channel::FrontLeft => "front left",
            Channel::FrontRight => "front right",
            Channel::FrontCenter => "front center",
            Channel::LowFrequency => "low frequency",
            Channel::BackLeft => "back left",
            Channel::BackRight => "back right",
            Channel::FrontLeftOfCenter => "front left-of-center",
            Channel::FrontRightOfCenter => "front right-of-center",
            Channel::BackCenter => "back center",
            Channel::SideLeft => "side left",
            Channel::SideRight => "side right",
            Channel::TopCenter => "top center",
            Channel::TopFrontLeft => "top front left",
            Channel::TopFrontCenter => "top front center",
            Channel::TopFrontRight => "top front right",
            Channel::TopBackLeft => "top back left",
            Channel::TopBackCenter => "top back center",
            Channel::TopBackRight => "top back right",
            Channel::StereoLeft => "downmix left",
            Channel::StereoRight => "downmix right",
            Channel::WideLeft => "wide left",
            Channel::WideRight => "wide right",
            Channel::SurroundDirectLeft => "surround direct left",
            Channel::SurroundDirectRight => "surround direct right",
            Channel::LowFrequency2 => "low frequency 2",
            Channel::TopSideLeft => "top side left",
            Channel::TopSideRight => "top side right",
            Channel::BottomFrontCenter => "bottom front center",
            Channel::BottomFrontLeft => "bottom front left",
            Channel::BottomFrontRight => "bottom front right",
            Channel::SideSurroundLeft => "side surround left",
            Channel::SideSurroundRight => "side surround right",
            Channel::TopSurroundLeft => "top surround left",
            Channel::TopSurroundRight => "top surround right",
            Channel::BinauralLeft => "binaural left",
            Channel::BinauralRight => "binaural right",
            // c:128-133
            Channel::None => "none",
            Channel::Unknown => "unknown",
            Channel::Unused => "unused",
            Channel::AmbisonicBase | Channel::AmbisonicEnd => "",
        }
    }

    /// `av_channel_from_string` (`c:153-183`), supported subset: exact,
    /// case-sensitive match on [`Channel::name`] over the 36 positional
    /// names (`c:166-169`), plus `"UNK"` → [`Unknown`] (`c:170-171`) and
    /// `"UNSD"` → [`Unused`] (`c:172-173`). C's `"AMBI<n>"` (`c:159-164`)
    /// and `"USR<n>"` (`c:175-178`) forms yield ids this closed enum cannot
    /// represent → `None` (documented degradation, module doc).
    pub fn from_name(s: &str) -> Option<Channel> {
        Some(match s {
            "FL" => Channel::FrontLeft,
            "FR" => Channel::FrontRight,
            "FC" => Channel::FrontCenter,
            "LFE" => Channel::LowFrequency,
            "BL" => Channel::BackLeft,
            "BR" => Channel::BackRight,
            "FLC" => Channel::FrontLeftOfCenter,
            "FRC" => Channel::FrontRightOfCenter,
            "BC" => Channel::BackCenter,
            "SL" => Channel::SideLeft,
            "SR" => Channel::SideRight,
            "TC" => Channel::TopCenter,
            "TFL" => Channel::TopFrontLeft,
            "TFC" => Channel::TopFrontCenter,
            "TFR" => Channel::TopFrontRight,
            "TBL" => Channel::TopBackLeft,
            "TBC" => Channel::TopBackCenter,
            "TBR" => Channel::TopBackRight,
            "DL" => Channel::StereoLeft,
            "DR" => Channel::StereoRight,
            "WL" => Channel::WideLeft,
            "WR" => Channel::WideRight,
            "SDL" => Channel::SurroundDirectLeft,
            "SDR" => Channel::SurroundDirectRight,
            "LFE2" => Channel::LowFrequency2,
            "TSL" => Channel::TopSideLeft,
            "TSR" => Channel::TopSideRight,
            "BFC" => Channel::BottomFrontCenter,
            "BFL" => Channel::BottomFrontLeft,
            "BFR" => Channel::BottomFrontRight,
            "SSL" => Channel::SideSurroundLeft,
            "SSR" => Channel::SideSurroundRight,
            "TTL" => Channel::TopSurroundLeft,
            "TTR" => Channel::TopSurroundRight,
            "BIL" => Channel::BinauralLeft,
            "BIR" => Channel::BinauralRight,
            // c:170-173
            "UNK" => Channel::Unknown,
            "UNSD" => Channel::Unused,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The positional channel at mask bit `pos` — inverse of
/// [`Channel::mask_bit`] for the 36 values C allows in masks.
const fn channel_at(pos: u32) -> Option<Channel> {
    Some(match pos {
        0 => Channel::FrontLeft,
        1 => Channel::FrontRight,
        2 => Channel::FrontCenter,
        3 => Channel::LowFrequency,
        4 => Channel::BackLeft,
        5 => Channel::BackRight,
        6 => Channel::FrontLeftOfCenter,
        7 => Channel::FrontRightOfCenter,
        8 => Channel::BackCenter,
        9 => Channel::SideLeft,
        10 => Channel::SideRight,
        11 => Channel::TopCenter,
        12 => Channel::TopFrontLeft,
        13 => Channel::TopFrontCenter,
        14 => Channel::TopFrontRight,
        15 => Channel::TopBackLeft,
        16 => Channel::TopBackCenter,
        17 => Channel::TopBackRight,
        29 => Channel::StereoLeft,
        30 => Channel::StereoRight,
        31 => Channel::WideLeft,
        32 => Channel::WideRight,
        33 => Channel::SurroundDirectLeft,
        34 => Channel::SurroundDirectRight,
        35 => Channel::LowFrequency2,
        36 => Channel::TopSideLeft,
        37 => Channel::TopSideRight,
        38 => Channel::BottomFrontCenter,
        39 => Channel::BottomFrontLeft,
        40 => Channel::BottomFrontRight,
        41 => Channel::SideSurroundLeft,
        42 => Channel::SideSurroundRight,
        43 => Channel::TopSurroundLeft,
        44 => Channel::TopSurroundRight,
        61 => Channel::BinauralLeft,
        62 => Channel::BinauralRight,
        _ => return None,
    })
}

/// `enum AVChannelOrder` (`channel_layout.h:114-160`) — the two orders
/// representable without an explicit channel map.
///
/// `AV_CHANNEL_ORDER_CUSTOM` and `AV_CHANNEL_ORDER_AMBISONIC` (`h:126-155`)
/// are **not ported**: every C site that can produce them returns
/// [`Error::Unsupported`] instead (module doc lists them).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Order {
    /// `AV_CHANNEL_ORDER_UNSPEC` — only `nb_channels` is meaningful; the
    /// `mask` field is undefined and must not be used (`h:341-343`) —
    /// conventionally 0 here.
    #[default]
    Unspecified,
    /// `AV_CHANNEL_ORDER_NATIVE` — channels in [`Channel`] enum order;
    /// the mask is the source of truth.
    Native,
}

/// `AV_CH_FOO` (`channel_layout.h:175-210`) for the positional channels.
const fn bit(c: Channel) -> u64 {
    1u64 << (c as i32)
}

/// `AV_CHANNEL_LAYOUT_MASK(nb, m)` (`h:393-397`).
const fn native(nb_channels: usize, mask: u64) -> ChannelLayout {
    ChannelLayout {
        order: Order::Native,
        nb_channels,
        mask,
    }
}

/// `AVChannelLayout` (`channel_layout.h:328-386`) — native-order core.
///
/// Field map: `order` (`h:333`), `nb_channels` (`h:338`, `int` → `usize`),
/// `u.mask` (`h:360`). The union's `u.map` arm (`h:361-379`,
/// `AV_CHANNEL_ORDER_CUSTOM`) and `opaque` (`h:385`) are dropped. When
/// `order == Unspecified` the mask is undefined in C (`h:341-343`) and
/// conventionally 0 here — constructors keep it 0; do not read it.
///
/// **Equality** (derived `PartialEq`) *is* C's `av_channel_layout_compare`
/// (`c:811-839`) for every layout constructible through this API: counts
/// differ → `!=`; exactly one `Unspecified` → `!=` (orders differ); both
/// `Unspecified` → `==` iff counts equal; both `Native` → mask equality
/// (count is the mask's popcount). Do not add a `compare()` method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelLayout {
    /// Channel order used in this layout (mandatory field, `h:333`).
    pub order: Order,
    /// Number of channels (mandatory field, `h:338`).
    pub nb_channels: usize,
    /// Native-order bitmask — one bit per positional channel (`h:345-360`).
    /// Meaningless (conventionally 0) when `order == Unspecified`.
    pub mask: u64,
}

/// C's `{0}` initializer / `memset`-0 (`h:312-313`): the "unset" layout.
/// Written out (not derived) so `Order`'s `Default` can never disagree
/// with `nb_channels`/`mask`.
impl Default for ChannelLayout {
    fn default() -> Self {
        ChannelLayout {
            order: Order::Unspecified,
            nb_channels: 0,
            mask: 0,
        }
    }
}

impl ChannelLayout {
    // ------------------------------------------------------------------
    // The 40 AV_CHANNEL_LAYOUT_* constants (h:403-442), masks written as
    // the OR expressions of the AV_CH_LAYOUT_* macros they alias
    // (h:217-256) so they audit against the header line for line.
    // ------------------------------------------------------------------

    /// `AV_CHANNEL_LAYOUT_MONO` — `FC`.
    pub const MONO: ChannelLayout = native(1, bit(Channel::FrontCenter));
    /// `AV_CHANNEL_LAYOUT_STEREO` — `FL|FR`.
    pub const STEREO: ChannelLayout = native(2, bit(Channel::FrontLeft) | bit(Channel::FrontRight));
    /// `AV_CHANNEL_LAYOUT_2POINT1` — `STEREO|LFE`.
    pub const TwoPointOne: ChannelLayout =
        native(3, ChannelLayout::STEREO.mask | bit(Channel::LowFrequency));
    /// `AV_CHANNEL_LAYOUT_2_1` — `STEREO|BC`.
    pub const A2_1: ChannelLayout =
        native(3, ChannelLayout::STEREO.mask | bit(Channel::BackCenter));
    /// `AV_CHANNEL_LAYOUT_SURROUND` — `STEREO|FC`.
    pub const SURROUND: ChannelLayout =
        native(3, ChannelLayout::STEREO.mask | bit(Channel::FrontCenter));
    /// `AV_CHANNEL_LAYOUT_3POINT1` — `SURROUND|LFE`.
    pub const ThreePointOne: ChannelLayout =
        native(4, ChannelLayout::SURROUND.mask | bit(Channel::LowFrequency));
    /// `AV_CHANNEL_LAYOUT_4POINT0` — `SURROUND|BC`.
    pub const FourPointZero: ChannelLayout =
        native(4, ChannelLayout::SURROUND.mask | bit(Channel::BackCenter));
    /// `AV_CHANNEL_LAYOUT_4POINT1` — `FourPointZero|LFE`.
    pub const FourPointOne: ChannelLayout = native(
        5,
        ChannelLayout::FourPointZero.mask | bit(Channel::LowFrequency),
    );
    /// `AV_CHANNEL_LAYOUT_2_2` — `STEREO|SL|SR`.
    pub const A2_2: ChannelLayout = native(
        4,
        ChannelLayout::STEREO.mask | bit(Channel::SideLeft) | bit(Channel::SideRight),
    );
    /// `AV_CHANNEL_LAYOUT_QUAD` — `STEREO|BL|BR`.
    pub const QUAD: ChannelLayout = native(
        4,
        ChannelLayout::STEREO.mask | bit(Channel::BackLeft) | bit(Channel::BackRight),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT0` — `SURROUND|SL|SR` (side surrounds).
    pub const FivePointZero: ChannelLayout = native(
        5,
        ChannelLayout::SURROUND.mask | bit(Channel::SideLeft) | bit(Channel::SideRight),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT1` — `FivePointZero|LFE` (side surrounds).
    pub const FivePointOne: ChannelLayout = native(
        6,
        ChannelLayout::FivePointZero.mask | bit(Channel::LowFrequency),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT0_BACK` — `SURROUND|BL|BR` (back surrounds).
    pub const FivePointZeroBack: ChannelLayout = native(
        5,
        ChannelLayout::SURROUND.mask | bit(Channel::BackLeft) | bit(Channel::BackRight),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT1_BACK` — `FivePointZeroBack|LFE`.
    pub const FivePointOneBack: ChannelLayout = native(
        6,
        ChannelLayout::FivePointZeroBack.mask | bit(Channel::LowFrequency),
    );
    /// `AV_CHANNEL_LAYOUT_6POINT0` — `FivePointZero|BC`.
    pub const SixPointZero: ChannelLayout = native(
        6,
        ChannelLayout::FivePointZero.mask | bit(Channel::BackCenter),
    );
    /// `AV_CHANNEL_LAYOUT_6POINT0_FRONT` — `2_2|FLC|FRC`.
    pub const SixPointZeroFront: ChannelLayout = native(
        6,
        ChannelLayout::A2_2.mask
            | bit(Channel::FrontLeftOfCenter)
            | bit(Channel::FrontRightOfCenter),
    );
    /// `AV_CHANNEL_LAYOUT_3POINT1POINT2` — `ThreePointOne|TFL|TFR`.
    pub const ThreePointOnePointTwo: ChannelLayout = native(
        6,
        ChannelLayout::ThreePointOne.mask
            | bit(Channel::TopFrontLeft)
            | bit(Channel::TopFrontRight),
    );
    /// `AV_CHANNEL_LAYOUT_HEXAGONAL` — `FivePointZeroBack|BC`.
    pub const HEXAGONAL: ChannelLayout = native(
        6,
        ChannelLayout::FivePointZeroBack.mask | bit(Channel::BackCenter),
    );
    /// `AV_CHANNEL_LAYOUT_6POINT1` — `FivePointOne|BC`.
    pub const SixPointOne: ChannelLayout = native(
        7,
        ChannelLayout::FivePointOne.mask | bit(Channel::BackCenter),
    );
    /// `AV_CHANNEL_LAYOUT_6POINT1_BACK` — `FivePointOneBack|BC`.
    pub const SixPointOneBack: ChannelLayout = native(
        7,
        ChannelLayout::FivePointOneBack.mask | bit(Channel::BackCenter),
    );
    /// `AV_CHANNEL_LAYOUT_6POINT1_FRONT` — `SixPointZeroFront|LFE`.
    pub const SixPointOneFront: ChannelLayout = native(
        7,
        ChannelLayout::SixPointZeroFront.mask | bit(Channel::LowFrequency),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT0` — `FivePointZero|BL|BR`.
    pub const SevenPointZero: ChannelLayout = native(
        7,
        ChannelLayout::FivePointZero.mask | bit(Channel::BackLeft) | bit(Channel::BackRight),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT0_FRONT` — `FivePointZero|FLC|FRC`.
    pub const SevenPointZeroFront: ChannelLayout = native(
        7,
        ChannelLayout::FivePointZero.mask
            | bit(Channel::FrontLeftOfCenter)
            | bit(Channel::FrontRightOfCenter),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT1` — `FivePointOne|BL|BR`.
    pub const SevenPointOne: ChannelLayout = native(
        8,
        ChannelLayout::FivePointOne.mask | bit(Channel::BackLeft) | bit(Channel::BackRight),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT1_WIDE` — `FivePointOne|FLC|FRC`.
    pub const SevenPointOneWide: ChannelLayout = native(
        8,
        ChannelLayout::FivePointOne.mask
            | bit(Channel::FrontLeftOfCenter)
            | bit(Channel::FrontRightOfCenter),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT1_WIDE_BACK` — `FivePointOneBack|FLC|FRC`.
    pub const SevenPointOneWideBack: ChannelLayout = native(
        8,
        ChannelLayout::FivePointOneBack.mask
            | bit(Channel::FrontLeftOfCenter)
            | bit(Channel::FrontRightOfCenter),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT1POINT2` — `FivePointOne|TFL|TFR`.
    pub const FivePointOnePointTwo: ChannelLayout = native(
        8,
        ChannelLayout::FivePointOne.mask | bit(Channel::TopFrontLeft) | bit(Channel::TopFrontRight),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT1POINT2_BACK` — `FivePointOneBack|TFL|TFR`.
    pub const FivePointOnePointTwoBack: ChannelLayout = native(
        8,
        ChannelLayout::FivePointOneBack.mask
            | bit(Channel::TopFrontLeft)
            | bit(Channel::TopFrontRight),
    );
    /// `AV_CHANNEL_LAYOUT_OCTAGONAL` — `FivePointZero|BL|BC|BR`.
    pub const OCTAGONAL: ChannelLayout = native(
        8,
        ChannelLayout::FivePointZero.mask
            | bit(Channel::BackLeft)
            | bit(Channel::BackCenter)
            | bit(Channel::BackRight),
    );
    /// `AV_CHANNEL_LAYOUT_CUBE` — `QUAD|TFL|TFR|TBL|TBR`.
    pub const CUBE: ChannelLayout = native(
        8,
        ChannelLayout::QUAD.mask
            | bit(Channel::TopFrontLeft)
            | bit(Channel::TopFrontRight)
            | bit(Channel::TopBackLeft)
            | bit(Channel::TopBackRight),
    );
    /// `AV_CHANNEL_LAYOUT_5POINT1POINT4` — `FivePointOnePointTwo|TBL|TBR`.
    pub const FivePointOnePointFour: ChannelLayout = native(
        10,
        ChannelLayout::FivePointOnePointTwo.mask
            | bit(Channel::TopBackLeft)
            | bit(Channel::TopBackRight),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT1POINT2` — `SevenPointOne|TFL|TFR`.
    pub const SevenPointOnePointTwo: ChannelLayout = native(
        10,
        ChannelLayout::SevenPointOne.mask
            | bit(Channel::TopFrontLeft)
            | bit(Channel::TopFrontRight),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT1POINT4` — `SevenPointOnePointTwo|TBL|TBR`.
    pub const SevenPointOnePointFour: ChannelLayout = native(
        12,
        ChannelLayout::SevenPointOnePointTwo.mask
            | bit(Channel::TopBackLeft)
            | bit(Channel::TopBackRight),
    );
    /// `AV_CHANNEL_LAYOUT_7POINT2POINT3` — `SevenPointOnePointTwo|TBC|LFE2`.
    pub const SevenTwoThree: ChannelLayout = native(
        12,
        ChannelLayout::SevenPointOnePointTwo.mask
            | bit(Channel::TopBackCenter)
            | bit(Channel::LowFrequency2),
    );
    /// `AV_CHANNEL_LAYOUT_9POINT1POINT4` — `SevenPointOnePointFour|FLC|FRC`.
    pub const NineOneFour: ChannelLayout = native(
        14,
        ChannelLayout::SevenPointOnePointFour.mask
            | bit(Channel::FrontLeftOfCenter)
            | bit(Channel::FrontRightOfCenter),
    );
    /// `AV_CHANNEL_LAYOUT_9POINT1POINT6` — `NineOneFour|TSL|TSR`.
    pub const NineOneSix: ChannelLayout = native(
        16,
        ChannelLayout::NineOneFour.mask | bit(Channel::TopSideLeft) | bit(Channel::TopSideRight),
    );
    /// `AV_CHANNEL_LAYOUT_HEXADECAGONAL` —
    /// `OCTAGONAL|WL|WR|TBL|TBR|TBC|TFC|TFL|TFR` (`h:253`).
    pub const HEXADECAGONAL: ChannelLayout = native(
        16,
        ChannelLayout::OCTAGONAL.mask
            | bit(Channel::WideLeft)
            | bit(Channel::WideRight)
            | bit(Channel::TopBackLeft)
            | bit(Channel::TopBackRight)
            | bit(Channel::TopBackCenter)
            | bit(Channel::TopFrontCenter)
            | bit(Channel::TopFrontLeft)
            | bit(Channel::TopFrontRight),
    );
    /// `AV_CHANNEL_LAYOUT_BINAURAL` — `BIL|BIR`.
    pub const BINAURAL: ChannelLayout =
        native(2, bit(Channel::BinauralLeft) | bit(Channel::BinauralRight));
    /// `AV_CHANNEL_LAYOUT_STEREO_DOWNMIX` — `DL|DR`.
    pub const STEREO_DOWNMIX: ChannelLayout =
        native(2, bit(Channel::StereoLeft) | bit(Channel::StereoRight));
    /// `AV_CHANNEL_LAYOUT_22POINT2` —
    /// `NineOneSix|BC|LFE2|TFC|TC|TBC|BFC|BFL|BFR` (`h:256`).
    pub const TwentyTwoTwo: ChannelLayout = native(
        24,
        ChannelLayout::NineOneSix.mask
            | bit(Channel::BackCenter)
            | bit(Channel::LowFrequency2)
            | bit(Channel::TopFrontCenter)
            | bit(Channel::TopCenter)
            | bit(Channel::TopBackCenter)
            | bit(Channel::BottomFrontCenter)
            | bit(Channel::BottomFrontLeft)
            | bit(Channel::BottomFrontRight),
    );

    /// `av_channel_layout_from_mask` (`c:253-264`): native layout with
    /// `nb_channels = popcount(mask)` (`av_popcount64`, c:260).
    /// `mask == 0` → `EINVAL` (c:256-257).
    pub fn from_mask(mask: u64) -> Result<ChannelLayout> {
        if mask == 0 {
            return Err(Error::InvalidArgument("invalid channel mask 0".into()));
        }
        Ok(native(mask.count_ones() as usize, mask))
    }

    /// The `AV_CHANNEL_ORDER_UNSPEC` initializer: only a channel count,
    /// no positional information (`h:119-121`).
    pub const fn unspecified(nb_channels: usize) -> ChannelLayout {
        ChannelLayout {
            order: Order::Unspecified,
            nb_channels,
            mask: 0,
        }
    }

    /// `av_channel_layout_check` (`c:785-809`): `nb_channels == 0` →
    /// invalid (c:787-788); `Native` → popcount must equal `nb_channels`
    /// (c:792); `Unspecified` → valid (c:804-805). (`Custom`/`Ambisonic`
    /// arms dropped with the orders.)
    pub fn check(&self) -> bool {
        if self.nb_channels == 0 {
            return false;
        }
        match self.order {
            Order::Native => self.mask.count_ones() as usize == self.nb_channels,
            Order::Unspecified => true,
        }
    }

    /// `av_channel_layout_default` (`c:841-852`): first [`STANDARD`] entry
    /// (table order!) whose count matches, else `Unspecified{nb}`.
    ///
    /// First-match consequences pinned by tests: 4 → "4.0" (NOT quad —
    /// "4.0" precedes "quad", c:196-197), 5 → "5.0" = `FivePointZeroBack`,
    /// 6 → "5.1" = `FivePointOneBack`, 7 → "6.1" = `SixPointOne`, 8 → "7.1".
    /// `default_for(0)` returns an invalid layout (`check() == false`),
    /// exactly like C.
    pub fn default_for(nb_channels: usize) -> ChannelLayout {
        for (_, layout) in STANDARD {
            if layout.nb_channels == nb_channels {
                return *layout;
            }
        }
        ChannelLayout::unspecified(nb_channels)
    }

    /// `av_channel_layout_describe_bprint`, native/unspecified arms
    /// (`c:600-652`; the `char buf`/`ERANGE` plumbing of c:654-671 is
    /// obsolete).
    ///
    /// Native order: (a) first [`STANDARD`] entry whose **mask** matches →
    /// its name (c:607-611); (b) otherwise `"{nb} channels (A+B+…)"`
    /// with channels in ascending bit order via
    /// [`ChannelLayout::channel_from_index`] (c:627-641); (c) count 0 →
    /// falls through to the unspecified text `"0 channels"` (c:643-645).
    /// Unspecified order → `"{nb} channels"` (c:644-645).
    pub fn describe(&self) -> String {
        match self.order {
            Order::Native => {
                for (name, layout) in STANDARD {
                    if layout.mask == self.mask {
                        return (*name).to_string();
                    }
                }
                if self.nb_channels > 0 {
                    // c:629-641 — beyond the set bits channel_from_index
                    // yields NONE ("NONE") just like C; only possible for
                    // layouts that fail check().
                    let mut out = format!("{} channels (", self.nb_channels);
                    for i in 0..self.nb_channels {
                        if i > 0 {
                            out.push('+');
                        }
                        out.push_str(self.channel_from_index(i).unwrap_or(Channel::None).name());
                    }
                    out.push(')');
                    out
                } else {
                    format!("{} channels", self.nb_channels)
                }
            }
            Order::Unspecified => format!("{} channels", self.nb_channels),
        }
    }

    /// `av_channel_layout_channel_from_index` (`c:673-701`).
    ///
    /// `idx >= nb_channels` → `None` (c:679-680); `Unspecified` → `None`
    /// (c:682 default arm); `Native` → walk bit positions 0..64 ascending,
    /// counting down `idx` on set bits (c:693-696). (C's `CUSTOM`/`AMBISONIC`
    /// arms dropped with the orders.)
    pub fn channel_from_index(&self, idx: usize) -> Option<Channel> {
        if idx >= self.nb_channels {
            return None;
        }
        match self.order {
            Order::Unspecified => None,
            Order::Native => {
                let mut idx = idx;
                for pos in 0..64u32 {
                    if self.mask & (1u64 << pos) != 0 {
                        if idx == 0 {
                            return channel_at(pos);
                        }
                        idx -= 1;
                    }
                }
                None
            }
        }
    }

    /// `av_channel_layout_index_from_channel` (`c:715-747`).
    ///
    /// `Channel::None` → `EINVAL` (c:720-721); `Unspecified` → `EINVAL`
    /// (c:744-745 default arm); `Native`: special channel (no mask bit,
    /// `> 63` in C, c:739) or bit not set → `EINVAL`, else the number of
    /// set bits *below* the channel's bit (c:741-742; C's `ambi_channels`
    /// term is 0 for native layouts). Error text carries the debug form of
    /// the channel.
    pub fn index_from_channel(&self, channel: Channel) -> Result<usize> {
        if channel == Channel::None {
            return Err(Error::InvalidArgument(format!(
                "channel {channel:?} not present in layout"
            )));
        }
        match self.order {
            Order::Unspecified => Err(Error::InvalidArgument(format!(
                "channel {channel:?} not present in layout"
            ))),
            Order::Native => {
                let Some(bit) = channel.mask_bit() else {
                    return Err(Error::InvalidArgument(format!(
                        "channel {channel:?} not present in layout"
                    )));
                };
                if self.mask & bit == 0 {
                    return Err(Error::InvalidArgument(format!(
                        "channel {channel:?} not present in layout"
                    )));
                }
                Ok((self.mask & (bit - 1)).count_ones() as usize)
            }
        }
    }

    /// `av_channel_layout_index_from_string` (`c:749-783`).
    ///
    /// `Unspecified` → `EINVAL` (C's switch has no UNSPEC case and falls
    /// out to c:782). A name containing `'@'` → [`Error::Unsupported`]:
    /// custom names exist only in `AV_CHANNEL_ORDER_CUSTOM` maps
    /// (c:757-767, dropped) — note C's native arm would instead fail the
    /// name lookup with `EINVAL`. Otherwise [`Channel::from_name`] →
    /// `None` → `EINVAL`, `Some` → [`ChannelLayout::index_from_channel`]
    /// (c:776-779).
    pub fn index_from_string(&self, name: &str) -> Result<usize> {
        match self.order {
            Order::Unspecified => Err(Error::InvalidArgument(format!(
                "invalid channel name '{name}'"
            ))),
            Order::Native => {
                if name.contains('@') {
                    return Err(Error::Unsupported(
                        "channel names with '@' require AV_CHANNEL_ORDER_CUSTOM, which is not ported"
                            .into(),
                    ));
                }
                match Channel::from_name(name) {
                    None => Err(Error::InvalidArgument(format!(
                        "invalid channel name '{name}'"
                    ))),
                    Some(ch) => self.index_from_channel(ch),
                }
            }
        }
    }

    /// `av_channel_layout_channel_from_string` (`c:703-713`): index lookup,
    /// then channel lookup.
    pub fn channel_from_string(&self, name: &str) -> Option<Channel> {
        self.index_from_string(name)
            .ok()
            .and_then(|i| self.channel_from_index(i))
    }

    /// `av_channel_layout_subset` (`c:867-885`): `Native` →
    /// `self.mask & mask` (c:874-876); `Unspecified` → 0 (C's switch has
    /// no case — `ret` stays 0).
    pub fn subset(&self, mask: u64) -> u64 {
        match self.order {
            Order::Native => self.mask & mask,
            Order::Unspecified => 0,
        }
    }

    /// Not a C API (Rust-side convenience for mask composition, the
    /// `AV_CH_LAYOUT_X | AV_CH_LAYOUT_Y` idiom): union of two native
    /// layouts; `nb_channels` = popcount of the OR ([`from_mask`]
    /// guarantees consistency). Any `Unspecified` operand →
    /// [`Error::Unsupported`] (undefined).
    pub fn union(&self, other: &ChannelLayout) -> Result<ChannelLayout> {
        if self.order != Order::Native || other.order != Order::Native {
            return Err(Error::Unsupported(
                "union of unspecified channel layouts is undefined".into(),
            ));
        }
        ChannelLayout::from_mask(self.mask | other.mask)
    }

    /// `av_channel_layout_from_string` (`c:313-441`). Branches in C order:
    ///
    /// 1. **Exact standard name** — scan [`STANDARD`], case-sensitive
    ///    (c:322-327).
    /// 2. **`"ambisonic …"`** → [`Error::Unsupported`] (c:334-391 dropped).
    /// 3. **Channel list** `"A[+B…]"` / `"N channels (A+B…)"` (c:397-411):
    ///    tokens resolved via [`Channel::from_name`]; every channel must
    ///    carry a mask bit and the bits must be **strictly increasing**
    ///    (`masked_description`, c:463-474) — duplicates or out-of-order
    ///    lists are `AV_CHANNEL_ORDER_CUSTOM` in C →
    ///    [`Error::Unsupported`]; `'@'`-named channels likewise (c:275,
    ///    c:286). Unresolvable tokens make the branch fail like C's
    ///    `EINVAL` (c:290-292) and **fall through** to branches 4-6 — so
    ///    `"4"` reaches the mask branch, `"4 channels"` branch 6. For the
    ///    wrapped form the parsed count must match the leading number
    ///    (c:406). C would *succeed* with a custom map for `AMBI*`/`USR*`
    ///    tokens (c:159-164, c:175-178); here they are simply unresolvable
    ///    → fall through → final `EINVAL`.
    /// 4. **Mask number** (`c:413-420`): full-string `strtoull` base-0
    ///    parse — `0x`/`0X` hex, leading `0` octal, else decimal; the
    ///    entire string must be digits in that base, no `'-'` anywhere,
    ///    no overflow (`ERANGE`), value ≠ 0.
    /// 5. **`"{n}c"`** — default layout, returned only if native
    ///    (c:426-429); `"11c"`/`"0c"` fall through (real FFmpeg errors).
    /// 6. **`"{n}C"` / `"{n} channels"`** — `Unspecified{n}` (c:433-437).
    /// 7. Else `EINVAL` (c:440) → [`Error::InvalidArgument`].
    ///
    /// Divergence: C's `strtoull`/`strtol` skip leading whitespace and
    /// accept `'+'`; this port does not (no ported caller produces such
    /// strings).
    pub fn from_string(s: &str) -> Result<ChannelLayout> {
        // (1) channel layout names (c:322-327)
        for (name, layout) in STANDARD {
            if s == *name {
                return Ok(*layout);
            }
        }

        // (2) ambisonic (c:334-391) — dropped
        if s.starts_with("ambisonic ") {
            return Err(Error::Unsupported(
                "ambisonic channel layouts are not ported".into(),
            ));
        }

        // (3) channel list (c:397-411) — failures fall through to 4-6
        // exactly like C's parse_channel_list EINVAL (c:290-292, 300-301
        // + c:401-402); Unsupported outcomes are terminal.
        match parse_channel_list(s) {
            ListParse::Ok(layout) => return Ok(layout),
            ListParse::Unsupported(msg) => return Err(Error::Unsupported(msg)),
            ListParse::NoMatch => {}
            ListParse::Invalid(msg) => return Err(Error::InvalidArgument(msg)),
        }

        // (4) channel layout mask (c:413-420)
        if !s.contains('-') {
            if let Some(mask) = parse_u64_base0(s) {
                if mask != 0 {
                    // from_mask cannot fail: mask != 0.
                    return Ok(ChannelLayout::from_mask(mask).unwrap());
                }
            }
        }

        // (5) number of channels -> default layout (c:422-430)
        if let Some(channels) = split_leading_int(s, "c") {
            if channels > 0 {
                let layout = ChannelLayout::default_for(channels);
                if layout.order == Order::Native {
                    return Ok(layout);
                }
            }
        }

        // (6) number of unordered channels (c:432-438)
        if let Some(channels) = split_leading_int(s, "C") {
            if channels > 0 {
                return Ok(ChannelLayout::unspecified(channels));
            }
        }
        if let Some(channels) = split_leading_int(s, " channels") {
            if channels > 0 {
                return Ok(ChannelLayout::unspecified(channels));
            }
        }

        // (7) c:440
        Err(Error::InvalidArgument(format!(
            "invalid channel layout '{s}'"
        )))
    }
}

impl std::fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

/// `channel_layout_map[]` (`c:190-231`) — the 40 `(name, layout)` pairs
/// **in C table order**. The order is behavior: `describe()` first-match
/// and `default_for()` first-match both depend on it (e.g. `"5.0"` →
/// `FivePointZeroBack`, c:200; `"7.1(wide)"` → `SevenPointOneWideBack`, c:215).
/// Do not sort.
pub const STANDARD: &[(&str, ChannelLayout)] = &[
    ("mono", ChannelLayout::MONO),
    ("stereo", ChannelLayout::STEREO),
    ("2.1", ChannelLayout::TwoPointOne),
    ("3.0", ChannelLayout::SURROUND),
    ("3.0(back)", ChannelLayout::A2_1),
    ("4.0", ChannelLayout::FourPointZero),
    ("quad", ChannelLayout::QUAD),
    ("quad(side)", ChannelLayout::A2_2),
    ("3.1", ChannelLayout::ThreePointOne),
    ("5.0", ChannelLayout::FivePointZeroBack),
    ("5.0(side)", ChannelLayout::FivePointZero),
    ("4.1", ChannelLayout::FourPointOne),
    ("5.1", ChannelLayout::FivePointOneBack),
    ("5.1(side)", ChannelLayout::FivePointOne),
    ("6.0", ChannelLayout::SixPointZero),
    ("6.0(front)", ChannelLayout::SixPointZeroFront),
    ("3.1.2", ChannelLayout::ThreePointOnePointTwo),
    ("hexagonal", ChannelLayout::HEXAGONAL),
    ("6.1", ChannelLayout::SixPointOne),
    ("6.1(back)", ChannelLayout::SixPointOneBack),
    ("6.1(front)", ChannelLayout::SixPointOneFront),
    ("7.0", ChannelLayout::SevenPointZero),
    ("7.0(front)", ChannelLayout::SevenPointZeroFront),
    ("7.1", ChannelLayout::SevenPointOne),
    ("7.1(wide)", ChannelLayout::SevenPointOneWideBack),
    ("7.1(wide-side)", ChannelLayout::SevenPointOneWide),
    ("5.1.2", ChannelLayout::FivePointOnePointTwo),
    ("5.1.2(back)", ChannelLayout::FivePointOnePointTwoBack),
    ("octagonal", ChannelLayout::OCTAGONAL),
    ("cube", ChannelLayout::CUBE),
    ("5.1.4", ChannelLayout::FivePointOnePointFour),
    ("7.1.2", ChannelLayout::SevenPointOnePointTwo),
    ("7.1.4", ChannelLayout::SevenPointOnePointFour),
    ("7.2.3", ChannelLayout::SevenTwoThree),
    ("9.1.4", ChannelLayout::NineOneFour),
    ("9.1.6", ChannelLayout::NineOneSix),
    ("hexadecagonal", ChannelLayout::HEXADECAGONAL),
    ("binaural", ChannelLayout::BINAURAL),
    ("downmix", ChannelLayout::STEREO_DOWNMIX),
    ("22.2", ChannelLayout::TwentyTwoTwo),
];

const STANDARD_LAYOUTS: &[ChannelLayout] = &[
    ChannelLayout::MONO,
    ChannelLayout::STEREO,
    ChannelLayout::TwoPointOne,
    ChannelLayout::SURROUND,
    ChannelLayout::A2_1,
    ChannelLayout::FourPointZero,
    ChannelLayout::QUAD,
    ChannelLayout::A2_2,
    ChannelLayout::ThreePointOne,
    ChannelLayout::FivePointZeroBack,
    ChannelLayout::FivePointZero,
    ChannelLayout::FourPointOne,
    ChannelLayout::FivePointOneBack,
    ChannelLayout::FivePointOne,
    ChannelLayout::SixPointZero,
    ChannelLayout::SixPointZeroFront,
    ChannelLayout::ThreePointOnePointTwo,
    ChannelLayout::HEXAGONAL,
    ChannelLayout::SixPointOne,
    ChannelLayout::SixPointOneBack,
    ChannelLayout::SixPointOneFront,
    ChannelLayout::SevenPointZero,
    ChannelLayout::SevenPointZeroFront,
    ChannelLayout::SevenPointOne,
    ChannelLayout::SevenPointOneWideBack,
    ChannelLayout::SevenPointOneWide,
    ChannelLayout::FivePointOnePointTwo,
    ChannelLayout::FivePointOnePointTwoBack,
    ChannelLayout::OCTAGONAL,
    ChannelLayout::CUBE,
    ChannelLayout::FivePointOnePointFour,
    ChannelLayout::SevenPointOnePointTwo,
    ChannelLayout::SevenPointOnePointFour,
    ChannelLayout::SevenTwoThree,
    ChannelLayout::NineOneFour,
    ChannelLayout::NineOneSix,
    ChannelLayout::HEXADECAGONAL,
    ChannelLayout::BINAURAL,
    ChannelLayout::STEREO_DOWNMIX,
    ChannelLayout::TwentyTwoTwo,
];

/// `av_channel_layout_standard` (`c:854-865`) — iteration over the
/// standard layouts, in `channel_layout_map` order.
pub fn standard() -> &'static [ChannelLayout] {
    STANDARD_LAYOUTS
}

/// Outcome of the channel-list parse (`parse_channel_list`, c:266-311).
enum ListParse {
    /// A valid strictly-increasing positional list → native layout.
    Ok(ChannelLayout),
    /// C would build an `AV_CHANNEL_ORDER_CUSTOM`/ambisonic layout here;
    /// not ported.
    Unsupported(String),
    /// The branch failed the way C's `EINVAL` does (c:290-292, 300-301) —
    /// caller falls through to the mask/number branches (c:401-402).
    NoMatch,
    /// Terminal EINVAL (c:406): the wrapped count mismatches the list —
    /// C returns AVERROR(EINVAL) without falling through.
    Invalid(String),
}

/// `parse_channel_list` + the `"N channels (…)"` wrapper handling of
/// `av_channel_layout_from_string` (c:397-411).
fn parse_channel_list(s: &str) -> ListParse {
    // C's sscanf "%d channels (%[^)]" (c:398): leading digits, literal
    // " channels (", capture up to the FIRST ')'. matches == 2 marks the
    // wrapped form; c:405-409 then requires that ')' be the last char and
    // the count match. Capture is attempted even when the tail checks
    // would fail (see below).
    let mut wrapped_count: Option<usize> = None;
    let mut inner = s;
    if let Some((num, rest)) = leading_digits(s) {
        if let Some(after) = rest.strip_prefix(" channels (") {
            let captured = match after.find(')') {
                Some(close) => &after[..close],
                None => after, // %[^)] runs to end of string
            };
            inner = captured;
            // matches == 2 (c:398): the count check only applies when a
            // ')' exists and terminates the string (c:405-406).
            if let Some(close) = after.find(')') {
                if close == after.len() - 1 {
                    wrapped_count = Some(num);
                }
            }
        }
    }

    // c:300-301: empty list -> EINVAL -> fall through.
    if inner.is_empty() {
        return ListParse::NoMatch;
    }

    let mut mask: u64 = 0;
    let mut nb = 0usize;
    for token in inner.split('+') {
        // c:275 + c:286: "name@custom" pairs are CUSTOM-only.
        if token.contains('@') {
            return ListParse::Unsupported(format!(
                "channel order for '{s}' is AV_CHANNEL_ORDER_CUSTOM, which is not ported"
            ));
        }
        // c:287-292: unknown channel name -> EINVAL -> fall through.
        // (AMBI*/USR* are unresolvable in the closed enum — C would
        // accept them into a custom map; documented degradation.)
        let Some(channel) = Channel::from_name(token) else {
            return ListParse::NoMatch;
        };
        // masked_description (c:463-474): every channel must map to a
        // fresh bit above all previously set bits — rejects UNK/UNSD
        // (no mask bit) and, with them, duplicates and non-increasing
        // orders (which C keeps as CUSTOM order, c:307 retype).
        let Some(bit) = channel.mask_bit() else {
            return ListParse::Unsupported(format!(
                "channel order for '{s}' is AV_CHANNEL_ORDER_CUSTOM, which is not ported"
            ));
        };
        if mask >= bit {
            return ListParse::Unsupported(format!(
                "channel order for '{s}' is AV_CHANNEL_ORDER_CUSTOM, which is not ported"
            ));
        }
        mask |= bit;
        nb += 1;
    }

    // c:406: wrapped form must agree on the count (terminal EINVAL in C).
    if let Some(expected) = wrapped_count {
        if expected != nb {
            return ListParse::Invalid(format!(
                "invalid channel layout '{s}' (channel count mismatch)"
            ));
        }
    }

    // mask != 0: at least one token set a bit.
    ChannelLayout::from_mask(mask)
        .map(ListParse::Ok)
        .unwrap_or(ListParse::NoMatch)
}

/// Leading ASCII decimal digits of `s` + the remainder (the `%d` of C's
/// sscanf/strtol calls). Overflow and no-digits → `None` (C: `ERANGE`
/// skips the branch).
fn leading_digits(s: &str) -> Option<(usize, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    let n = s[..end].parse::<usize>().ok()?;
    Some((n, &s[end..]))
}

/// `strtol(str, &end, 10)` + `strcmp(end, suffix)` (c:423-437): leading
/// decimal digits followed by exactly `suffix`.
fn split_leading_int(s: &str, suffix: &str) -> Option<usize> {
    let (n, rest) = leading_digits(s)?;
    rest.strip_prefix(suffix).map(|_| n)
}

/// `strtoull(str, &end, 0)` with full consumption (`*end == '\0'`),
/// base-0 semantics (`c:413-414`): `0x`/`0X` → hex, a leading `0` with
/// more characters → octal, else decimal. `None` = not a fully-consumed
/// number in any base (also `ERANGE` overflow, empty, `0x` with no hex
/// digits, `09` …). The caller rejects `'-'` separately (C's `strchr`
/// check, c:417). Divergence: leading whitespace / `'+'` are not skipped
/// (C's strtoull would).
fn parse_u64_base0(s: &str) -> Option<u64> {
    let (digits, radix) = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        (hex, 16)
    } else if s.len() > 1 && s.starts_with('0') {
        (&s[1..], 8)
    } else {
        (s, 10)
    };
    if digits.is_empty() {
        return None;
    }
    u64::from_str_radix(digits, radix).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
