use ffmpeg_rs::{
    Error,
    format::io::{IoContext, IoHandler},
};

/// In-memory handler for tests (the `data:` protocol, roughly).
struct Mem {
    data: Vec<u8>,
    pos: usize,
}

impl IoHandler for Mem {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (self.data.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.data.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        self.pos = (pos as usize).min(self.data.len());
        Ok(self.pos as u64)
    }
    fn size(&self) -> u64 {
        self.data.len() as u64
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn mem_io(data: &[u8]) -> IoContext {
    IoContext::new(Box::new(Mem {
        data: data.to_vec(),
        pos: 0,
    }))
}

#[test]
fn r8_returns_zero_and_latches_eof() {
    let mut io = mem_io(b"AB");
    assert_eq!(io.r8().unwrap(), b'A');
    assert_eq!(io.r8().unwrap(), b'B');
    assert_eq!(io.r8().unwrap(), 0); // EOF reads as 0
    assert!(io.is_eof());
    assert_eq!(io.r8().unwrap(), 0); // stays 0
}

#[test]
fn tell_tracks_position_across_refills() {
    let data = vec![7u8; 10_000];
    let mut io = mem_io(&data);
    assert_eq!(io.tell(), 0);
    for _ in 0..5000 {
        assert_eq!(io.r8().unwrap(), 7);
    }
    assert_eq!(io.tell(), 5000);
}

#[test]
fn get_packet_reads_exact_or_errors_at_zero() {
    let mut io = mem_io(b"FRAME\nXXXXX");
    io.skip(6).unwrap();
    let pkt = io.get_packet(5).unwrap();
    assert_eq!(pkt, b"XXXXX");

    let mut short = mem_io(b"abc");
    let pkt = short.get_packet(5).unwrap(); // partial, not error
    assert_eq!(pkt, b"abc");
    // Like C's avio_read, the partial read attempted one more fill, so
    // eof is latched — that's what y4m's short-frame check relies on.
    assert!(short.is_eof());

    let mut empty = mem_io(b"");
    assert!(matches!(empty.get_packet(5), Err(Error::Eof)));
}

#[test]
fn seek_resets_buffer_and_eof() {
    let mut io = mem_io(b"0123456789");
    io.get_packet(10).unwrap();
    // Consuming everything does not by itself latch EOF.
    assert!(!io.is_eof());
    assert!(matches!(io.get_packet(1), Err(Error::Eof)));
    assert!(io.is_eof());
    io.seek(4).unwrap();
    assert!(!io.is_eof());
    assert_eq!(io.tell(), 4);
    assert_eq!(io.get_packet(2).unwrap(), b"45");
}

#[test]
fn peek_does_not_consume() {
    let mut io = mem_io(b"YUV4MPEG2 W10");
    let head = io.peek(9).unwrap();
    assert_eq!(&head, b"YUV4MPEG2");
    assert_eq!(io.tell(), 0);
    assert_eq!(io.r8().unwrap(), b'Y');
}

#[test]
fn skip_moves_position() {
    let mut io = mem_io(b"0123456789");
    io.skip(4).unwrap();
    assert_eq!(io.tell(), 4);
    assert_eq!(io.get_packet(2).unwrap(), b"45");
    assert!(matches!(io.skip(1000), Err(Error::Eof)));
}
