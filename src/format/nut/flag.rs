/// `nut.h:43-56` (`Flag`) — frame-code / frame-header flags.
/// if set, frame is keyframe
pub const KEY: u32 = 1;
/// if set, stream has no relevance on presentation. (EOR)
pub const EOR: u32 = 2;
/// if set, coded_pts is in the frame header
pub const CODED_PTS: u32 = 8;
/// if set, stream_id is coded in the frame header
pub const STREAM_ID: u32 = 16;
/// if set, data_size_msb is at frame header, otherwise data_size_msb is 0
pub const SIZE_MSB: u32 = 32;
/// if set, the frame header contains a checksum
pub const CHECKSUM: u32 = 64;
/// if set, reserved_count is coded in the frame header
pub const RESERVED: u32 = 128;
/// if set, side / meta data is stored in the frame header.
pub const SM_DATA: u32 = 256;
/// If set, header_idx is coded in the frame header.
pub const HEADER_IDX: u32 = 1024;
/// If set, match_time_delta is coded in the frame header
pub const MATCH_TIME: u32 = 2048;
/// if set, coded_flags are stored in the frame header
pub const CODED: u32 = 4096;
/// if set, frame_code is invalid
pub const INVALID: u32 = 8192;
