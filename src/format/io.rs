//! Buffered I/O — port of `libavformat/avio.h` + the `aviobuf.c` core.
//!
//! `AVIOContext` is FFmpeg's buffered byte stream with user-supplied
//! callbacks (`read_packet`/`write_packet`/`seek`); file/pipe/http are just
//! different callback sets. The Rust port keeps the same split:
//! [`IoHandler`] is the callback vtable, [`IoContext`] the buffered wrapper
//! every demuxer/muxer reads through.
//!
//! Semantics that the y4m demuxer depends on, preserved exactly:
//!
//! * `r8()` at clean EOF returns `Ok(0)` *and* latches `eof_reached` —
//!   y4m's frame-header loop checks `error` first, then `eof_reached`, then
//!   "overlong header" (yuv4mpegdec.c:281-286); that ordering must survive.
//! * `get_packet(n)` returns `Err(Eof)` only when *nothing* was read; a
//!   short partial read is `Ok(short)` and the caller decides (y4m: EOF vs
//!   corrupt; rawvideo: passes short packets downstream).
//! * `seek()` drops the buffer — reads after a seek refill from the handler.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use crate::util::error::{Error, Result};

/// The `URLProtocol` callback set (aviobuf.c's ffurl layer).
pub trait IoHandler {
    /// Read up to `buf.len()` bytes; `Ok(0)` = EOF.
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize>;
    /// Reposition; returns the new absolute offset.
    fn seek(&mut self, pos: u64) -> std::io::Result<u64>;
    /// Total size when knowable (files), else 0.
    fn size(&self) -> u64;
    /// Flush buffered writes.
    fn flush(&mut self) -> std::io::Result<()>;
}

/// File-backed handler (the `file` protocol). Read and write sides are
/// separate handles so one `IoContext` is either an input or an output,
/// like `AVIO_FLAG_READ`/`WRITE`.
enum FileHandler {
    Input(BufReader<File>),
    Output(BufWriter<File>),
}

impl IoHandler for FileHandler {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            FileHandler::Input(r) => r.read(buf),
            FileHandler::Output(_) => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "read on write-only IoContext",
            )),
        }
    }

    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            FileHandler::Input(_) => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "write on read-only IoContext",
            )),
            FileHandler::Output(w) => w.write(buf),
        }
    }

    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        match self {
            FileHandler::Input(r) => r.seek(SeekFrom::Start(pos)),
            FileHandler::Output(w) => w.get_ref().seek(SeekFrom::Start(pos)),
        }
    }

    fn size(&self) -> u64 {
        match self {
            FileHandler::Input(r) => r.get_ref().metadata().map(|m| m.len()).unwrap_or(0),
            FileHandler::Output(w) => w.get_ref().metadata().map(|m| m.len()).unwrap_or(0),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            FileHandler::Input(_) => Ok(()),
            FileHandler::Output(w) => w.flush(),
        }
    }
}

/// `AVIOContext` — buffered byte stream (aviobuf.c).
pub struct IoContext {
    handler: Box<dyn IoHandler + Send>,
    /// The read buffer (`AVIOContext.buffer`).
    buffer: Box<[u8; 4096]>,
    /// Index of the next unconsumed byte in `buffer` (`buf_ptr - buffer`).
    buf_start: usize,
    /// One past the last valid byte in `buffer` (`buf_end - buffer`).
    buf_end: usize,
    /// Absolute position of the next byte to be read (`avio_tell`).
    pos: u64,
    eof_reached: bool,
    error: Option<std::io::Error>,
}

impl IoContext {
    /// `avio_open(path, AVIO_FLAG_READ)`.
    pub fn open_input<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(Error::Io)?;
        Ok(IoContext::new(Box::new(FileHandler::Input(
            BufReader::new(file),
        ))))
    }

    /// `avio_open(path, AVIO_FLAG_WRITE)`.
    pub fn open_output<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path).map_err(Error::Io)?;
        Ok(IoContext::new(Box::new(FileHandler::Output(
            BufWriter::new(file),
        ))))
    }

    /// `avio_alloc_context` over custom callbacks.
    pub fn new(handler: Box<dyn IoHandler + Send>) -> Self {
        IoContext {
            handler,
            buffer: Box::new([0u8; 4096]),
            buf_start: 0,
            buf_end: 0,
            pos: 0,
            eof_reached: false,
            error: None,
        }
    }

    /// `fill_buffer` (aviobuf.c) — pull one handler read into the buffer.
    fn fill_buffer(&mut self) -> std::io::Result<usize> {
        self.buf_start = 0;
        self.buf_end = 0;
        let n = self.handler.read(&mut self.buffer[..])?;
        self.buf_end = n;
        Ok(n)
    }

    /// `avio_r8` — one byte; `Ok(0)` at EOF (with `eof_reached` latched, the
    /// C contract).
    pub fn r8(&mut self) -> Result<u8> {
        if self.buf_start == self.buf_end {
            if self.eof_reached {
                return Ok(0);
            }
            match self.fill_buffer() {
                Ok(0) => {
                    self.eof_reached = true;
                    return Ok(0);
                }
                Ok(_) => {}
                Err(e) => {
                    self.error = Some(std::io::Error::new(e.kind(), e.to_string()));
                    return Err(Error::Io(std::io::Error::new(e.kind(), e.to_string())));
                }
            }
        }
        let b = self.buffer[self.buf_start];
        self.buf_start += 1;
        self.pos += 1;
        Ok(b)
    }

    /// `avio_read` — up to `buf.len()` bytes; short only at EOF/error.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let mut filled = 0usize;
        while filled < buf.len() {
            if self.buf_start == self.buf_end {
                if self.eof_reached {
                    break;
                }
                match self.fill_buffer() {
                    Ok(0) => {
                        self.eof_reached = true;
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        self.error = Some(std::io::Error::new(e.kind(), e.to_string()));
                        return Err(Error::Io(std::io::Error::new(e.kind(), e.to_string())));
                    }
                }
            }
            let avail = self.buf_end - self.buf_start;
            let want = buf.len() - filled;
            let n = avail.min(want);
            buf[filled..filled + n]
                .copy_from_slice(&self.buffer[self.buf_start..self.buf_start + n]);
            self.buf_start += n;
            self.pos += n as u64;
            filled += n;
        }
        Ok(filled)
    }

    /// `av_get_packet` — exactly `n` bytes or what's left. `Err(Eof)` only
    /// when zero bytes could be read.
    pub fn get_packet(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        let got = self.read(&mut out)?;
        out.truncate(got);
        if got == 0 {
            // Nothing read: clean EOF (C returns AVERROR_EOF).
            return Err(Error::Eof);
        }
        Ok(out)
    }

    /// `avio_skip`.
    pub fn skip(&mut self, n: u64) -> Result<()> {
        let mut remaining = n;
        let mut chunk = [0u8; 4096];
        while remaining > 0 {
            let take = remaining.min(chunk.len() as u64) as usize;
            let got = self.read(&mut chunk[..take])?;
            if got == 0 {
                return Err(Error::Eof);
            }
            remaining -= got as u64;
        }
        Ok(())
    }

    /// `avio_tell`.
    pub fn tell(&self) -> u64 {
        self.pos
    }

    /// `avio_size`.
    pub fn size(&self) -> u64 {
        self.handler.size()
    }

    /// `avio_seek` — absolute; drops the buffer (C drops unless the target
    /// is inside it, a fast path we don't need) and clears EOF.
    pub fn seek(&mut self, pos: u64) -> Result<()> {
        self.handler.seek(pos).map_err(Error::Io)?;
        self.buf_start = 0;
        self.buf_end = 0;
        self.pos = pos;
        self.eof_reached = false;
        Ok(())
    }

    /// `avio_write` (`avio_write` is all-or-error; BufWriter flushes on drop
    /// or explicit flush below).
    pub fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        self.handler.write(buf).map_err(Error::Io)?;
        Ok(())
    }

    /// Flush the write side (`avio_flush`).
    pub fn flush(&mut self) -> Result<()> {
        self.handler.flush().map_err(Error::Io)?;
        Ok(())
    }

    /// `pb->eof_reached`.
    pub fn is_eof(&self) -> bool {
        self.eof_reached
    }

    /// `pb->error` — sticky first I/O error, if any.
    pub fn take_error(&mut self) -> Option<std::io::Error> {
        self.error.take()
    }

    /// Peek the first `n` bytes without consuming (used by format probing;
    /// implemented as read + rewind).
    pub fn peek(&mut self, n: usize) -> Result<Vec<u8>> {
        let save = self.pos;
        let mut buf = vec![0u8; n];
        let got = self.read(&mut buf)?;
        buf.truncate(got);
        self.seek(save)?;
        Ok(buf)
    }
}
