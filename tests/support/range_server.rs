// Minimal HTTP/1.1 file server with `Range` support for the URL-constant tests and benchmark.
//
// It serves one file under any path, answers `GET` with `Range: bytes=a-b` as
// `206 Partial Content`, keeps connections alive, counts requests, checks `If-Match` against a
// fixed entity tag and can throttle the body to emulate a network link. Test support only.

#![allow(dead_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Entity tag of the served file.
pub const ETAG: &str = "\"v1\"";

pub struct RangeServer {
    pub port: u16,
    requests: Arc<AtomicUsize>,
    bytes_sent: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
}

impl RangeServer {
    /// Serve `file` on an ephemeral localhost port. `bytes_per_second == 0` disables throttling;
    /// otherwise the rate is shared by all connections.
    pub fn start(file: &Path, bytes_per_second: u64) -> std::io::Result<Self> {
        Self::start_on(file, bytes_per_second, 0)
    }

    pub fn start_on(file: &Path, bytes_per_second: u64, port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let port = listener.local_addr()?.port();
        let requests = Arc::new(AtomicUsize::new(0));
        let bytes_sent = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared {
            file: file.to_path_buf(),
            bytes_per_second,
            next_send: std::sync::Mutex::new(None),
            requests: Arc::clone(&requests),
            bytes_sent: Arc::clone(&bytes_sent),
        });
        let stop_flag = Arc::clone(&stop);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_flag.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let _ = serve_connection(stream, &shared);
                });
            }
        });
        Ok(Self {
            port,
            requests,
            bytes_sent,
            stop,
        })
    }

    pub fn url(&self, name: &str) -> String {
        format!("http://127.0.0.1:{}/{name}", self.port)
    }

    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::SeqCst)
    }
}

impl Drop for RangeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock `accept`.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

struct Shared {
    file: PathBuf,
    bytes_per_second: u64,
    /// Earliest time the next chunk may be sent; shared by all connections. At most 50 ms
    /// of idle time are banked, so a paused download does not resume unthrottled.
    next_send: std::sync::Mutex<Option<Instant>>,
    requests: Arc<AtomicUsize>,
    bytes_sent: Arc<AtomicU64>,
}

fn serve_connection(stream: TcpStream, shared: &Shared) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    loop {
        let mut request_line = String::new();
        if reader.read_line(&mut request_line)? == 0 {
            return Ok(());
        }
        let mut range: Option<(u64, Option<u64>)> = None;
        let mut if_match: Option<String> = None;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                return Ok(());
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            if name.eq_ignore_ascii_case("range") {
                range = parse_range(value);
            } else if name.eq_ignore_ascii_case("if-match") {
                if_match = Some(value.to_string());
            }
        }
        if !request_line.starts_with("GET ") {
            writer.write_all(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")?;
            continue;
        }
        shared.requests.fetch_add(1, Ordering::SeqCst);
        if if_match.as_deref().is_some_and(|tag| tag != ETAG) {
            writer.write_all(b"HTTP/1.1 412 Precondition Failed\r\nContent-Length: 0\r\n\r\n")?;
            continue;
        }
        let mut file = File::open(&shared.file)?;
        let size = file.metadata()?.len();
        let (status, start, len) = match range {
            Some((start, end)) if start < size => {
                let end = end.unwrap_or(size - 1).min(size - 1);
                ("206 Partial Content", start, end + 1 - start)
            }
            Some(_) => {
                write!(
                    writer,
                    "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{size}\r\nContent-Length: 0\r\n\r\n"
                )?;
                continue;
            }
            None => ("200 OK", 0, size),
        };
        write!(
            writer,
            "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\nAccept-Ranges: bytes\r\nETag: {ETAG}\r\nContent-Length: {len}\r\n"
        )?;
        if status.starts_with("206") {
            write!(
                writer,
                "Content-Range: bytes {start}-{}/{size}\r\n",
                start + len - 1
            )?;
        }
        writer.write_all(b"\r\n")?;
        file.seek(SeekFrom::Start(start))?;
        let mut remaining = len;
        let mut chunk = vec![0u8; 64 * 1024];
        while remaining > 0 {
            let n = remaining.min(chunk.len() as u64) as usize;
            file.read_exact(&mut chunk[..n])?;
            writer.write_all(&chunk[..n])?;
            remaining -= n as u64;
            let sent = shared.bytes_sent.fetch_add(n as u64, Ordering::SeqCst) + n as u64;
            let _ = sent;
            if shared.bytes_per_second > 0 {
                // Global pacing: every chunk reserves its transmission time on a shared clock.
                let cost = Duration::from_secs_f64(n as f64 / shared.bytes_per_second as f64);
                let wait_until = {
                    let mut next = shared.next_send.lock().unwrap();
                    let now = Instant::now();
                    // Catch up after an overslept chunk (timer granularity), but bank at most
                    // 50 ms of idle time.
                    let floor = now.checked_sub(Duration::from_millis(50)).unwrap_or(now);
                    let start = next.map_or(now, |t| t.max(floor));
                    *next = Some(start + cost);
                    start + cost
                };
                let now = Instant::now();
                if wait_until > now {
                    std::thread::sleep(wait_until - now);
                }
            }
        }
        writer.flush()?;
    }
}

fn parse_range(value: &str) -> Option<(u64, Option<u64>)> {
    let spec = value.strip_prefix("bytes=")?;
    let (start, end) = spec.split_once('-')?;
    let start = start.trim().parse().ok()?;
    let end = match end.trim() {
        "" => None,
        end => Some(end.parse().ok()?),
    };
    Some((start, end))
}
