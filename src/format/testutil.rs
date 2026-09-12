//! Shared test fixtures for the format layer: an in-memory `IoHandler`
//! (roughly the `data:` protocol). Read and write sides share one buffer so
//! a test can write through an `IoContext` and read the bytes back.

use std::sync::{Arc, Mutex};

use super::io::{IoContext, IoHandler};

pub struct MemHandler {
    pub data: Arc<Mutex<Vec<u8>>>,
    pos: usize,
}

impl MemHandler {
    pub fn io(data: &[u8]) -> IoContext {
        IoContext::new(Box::new(MemHandler {
            data: Arc::new(Mutex::new(data.to_vec())),
            pos: 0,
        }))
    }

    /// [`io`] plus the shared buffer handle — muxer tests write through
    /// the IoContext and read the produced bytes back through the Arc.
    pub fn shared(data: &[u8]) -> (IoContext, Arc<Mutex<Vec<u8>>>) {
        let data = Arc::new(Mutex::new(data.to_vec()));
        let io = IoContext::new(Box::new(MemHandler {
            data: Arc::clone(&data),
            pos: 0,
        }));
        (io, data)
    }
}

impl IoHandler for MemHandler {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let data = self.data.lock().unwrap();
        let n = (data.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.data.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        self.pos = (pos as usize).min(self.data.lock().unwrap().len());
        Ok(self.pos as u64)
    }
    fn size(&self) -> u64 {
        self.data.lock().unwrap().len() as u64
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
