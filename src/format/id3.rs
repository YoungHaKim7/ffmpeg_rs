//! ID3v2 tag plumbing — the `libavformat/id3v2.c` subset both raw-audio
//! demuxers need (mp3, aac): magic check, syncsafe length, skip.
//!
//! Contents are DROPPED (no metadata dictionary in the port) — exactly the
//! `ff_id3v2_read`-with-no-callbacks shape. `libavformat/aacdec.c` also
//! meets ID3v2 tags *between* frames and re-syncs past them.

use crate::util::error::{Error, Result};

use super::io::IoContext;

/// `ff_id3v2_match` (id3v2.c): "ID3" + version bytes ≠ 0xff.
pub fn id3v2_match(buf: &[u8]) -> bool {
    buf.len() >= 10
        && buf[0] == b'I'
        && buf[1] == b'D'
        && buf[2] == b'3'
        && buf[3] != 0xff
        && buf[4] != 0xff
}

/// `get_size` (id3v2.c:210-217): 7-bit-per-byte syncsafe size.
pub fn id3v2_tag_len(buf: &[u8]) -> usize {
    (((buf[6] as usize) & 0x7f) << 21)
        | (((buf[7] as usize) & 0x7f) << 14)
        | (((buf[8] as usize) & 0x7f) << 7)
        | ((buf[9] as usize) & 0x7f) + 10
}

/// `ff_id3v2_skip` shape: consume the whole tag (header + body + footer).
/// Returns the total tag length.
pub fn skip_id3v2(io: &mut IoContext) -> Result<usize> {
    let mut head = [0u8; 10];
    read_full(io, &mut head)?;
    let len = id3v2_tag_len(&head);
    let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
    io.seek(io.tell() + (len - 10 + footer) as u64)?;
    Ok(len + footer)
}

/// `io.read` until the buffer is full; Err(Eof) if it runs dry mid-way.
pub(crate) fn read_full(io: &mut IoContext, buf: &mut [u8]) -> Result<()> {
    let mut got = 0;
    while got < buf.len() {
        match io.read(&mut buf[got..])? {
            0 => return Err(Error::Eof),
            n => got += n,
        }
    }
    Ok(())
}
