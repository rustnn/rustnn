//! Persistent on-disk caches used by backends (currently TensorRT engines and its runtime
//! cache).
//!
//! Entries live under `<platform cache dir>/rustnn/<category>/<key>`, for example
//! `~/.cache/rustnn/trtx` on Linux or `%LOCALAPPDATA%\rustnn\trtx` on Windows. Writes go
//! through a temporary file and an atomic rename. With the `zstd-cache-compression` feature
//! entries are stored compressed with a `.zstd` suffix.

use std::borrow::Cow;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use log::debug;

/// The cache category, which is the last component of `<cache dir>/rustnn/<category>`.
fn cache_category(root_path: &Path) -> impl std::fmt::Display + '_ {
    root_path
        .file_name()
        .map(|name| Path::new(name).display())
        .unwrap_or_else(|| Path::new("unknown").display())
}

/// Report what a cache read found, as an event inside its `get` span: a hit with its byte count, a
/// miss, or an error. Emitted here so the event carries the same target as the span it belongs to.
/// A read cannot use `err` on its attribute for that, because a missing entry arrives as an `Err`
/// too, and a miss is a result rather than a failure.
fn emit_cache_outcome(result: &CacheResult<Vec<u8>>) {
    match cache_outcome(result) {
        CacheOutcome::Hit(bytes) => tracing::debug!(outcome = "hit", bytes = bytes),
        CacheOutcome::Miss => tracing::debug!(outcome = "miss"),
        CacheOutcome::Error(error) => tracing::error!(outcome = "error", error = %error),
    }
}

/// What a cache read found. A missing entry arrives as an `Err` with `NotFound`, which is how a
/// miss is told apart from a real I/O error.
#[derive(Debug)]
enum CacheOutcome<'a> {
    Hit(usize),
    Miss,
    Error(&'a CacheError),
}

fn cache_outcome(result: &CacheResult<Vec<u8>>) -> CacheOutcome<'_> {
    match result {
        Ok(data) => CacheOutcome::Hit(data.len()),
        Err(CacheError::FailedToReadCacheFile { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            CacheOutcome::Miss
        }
        Err(error) => CacheOutcome::Error(error),
    }
}

/// Failures of the on-disk caches.
#[derive(Debug, Error)]
pub enum CacheError {
    /// The platform reports no cache directory.
    #[error("Failed to create cache path")]
    FailedToDetermineCachePath,

    /// The cache directory could not be created.
    #[error("Failed to create cache path: {path:?} ({source})")]
    FailedToCreateCachePath {
        /// Directory that could not be created.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// An entry could not be written.
    #[error("Failed to write cache file: {path:?} ({source})")]
    FailedToWriteCacheFile {
        /// File that could not be written.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// An entry could not be read (including a plain cache miss).
    #[error("Failed to read cache file: {path:?} ({source})")]
    FailedToReadCacheFile {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

/// Result of cache operations.
pub type CacheResult<T> = std::result::Result<T, CacheError>;

/// A key-value store of byte blobs under `<cache_dir>/rustnn/<category>`.
pub trait PersistentCache<'cache>: Sized + Send + Sync + std::fmt::Debug {
    /// Open (and create) the cache directory for `category`.
    fn new(category: &str) -> CacheResult<Self>;
    /// Read the entry stored under `key`.
    fn get(&self, key: &str) -> CacheResult<Cow<'cache, [u8]>>;
    /// Write `data` under `key`, replacing an existing entry atomically.
    fn set(&self, key: &str, data: &[u8]) -> CacheResult<()>;
}

/// One file per key, uncompressed.
// should probably be a SQLite data base that
// where it is easy to evict old data, do size management and also allow to work without file access
#[derive(Debug)]
pub struct SimpleFileCache {
    root_path: PathBuf,
}

impl<'cache> PersistentCache<'cache> for SimpleFileCache {
    fn new(category: &str) -> CacheResult<Self> {
        let root_path = dirs::cache_dir()
            .map(|dir| dir.join("rustnn").join(category))
            .ok_or(CacheError::FailedToDetermineCachePath)?;
        std::fs::create_dir_all(&root_path).map_err(|e| CacheError::FailedToCreateCachePath {
            path: root_path.clone(),
            source: e,
        })?;
        Ok(Self { root_path })
    }

    #[tracing::instrument(
        skip_all,
        level = "debug",
        fields(
            category = %cache_category(&self.root_path),
            key = %key,
            compressed = false,
        )
    )]
    fn get(&self, key: &str) -> CacheResult<Cow<'cache, [u8]>> {
        debug!("Looking up cache key: {key}");
        let cache_path = self.root_path.join(key);
        let result = read_cache_file(&cache_path).map_err(|e| CacheError::FailedToReadCacheFile {
            path: cache_path.clone(),
            source: e,
        });
        emit_cache_outcome(&result);
        result.map(Cow::Owned)
    }

    #[tracing::instrument(
        skip_all,
        err,
        level = "debug",
        fields(
            category = %cache_category(&self.root_path),
            key = %key,
            bytes = data.len(),
            compressed = false,
        )
    )]
    fn set(&self, key: &str, data: &[u8]) -> CacheResult<()> {
        debug!("Setting cache key: {key} with {} bytes", data.len());
        let cache_path = self.root_path.join(key);
        // TODO: currently, blocking, especially the exclusive lock on the file
        write_cache_file(&cache_path, data).map_err(|e| CacheError::FailedToWriteCacheFile {
            path: cache_path.clone(),
            source: e,
        })
    }
}

/// Cache used when the `zstd-cache-compression` feature is off.
pub type DefaultCache = SimpleFileCache;

/// One zstd-compressed file per key (`<key>.zstd`).
#[cfg(feature = "zstd-cache-compression")]
#[derive(Debug)]
pub struct ZstdCompressedFileCache {
    root_path: PathBuf,
}

#[cfg(feature = "zstd-cache-compression")]
impl ZstdCompressedFileCache {
    fn cache_path(&self, key: &str) -> PathBuf {
        self.root_path.join(format!("{key}.zstd"))
    }
}

#[cfg(feature = "zstd-cache-compression")]
impl<'cache> PersistentCache<'cache> for ZstdCompressedFileCache {
    fn new(category: &str) -> CacheResult<Self> {
        let root_path = dirs::cache_dir()
            .map(|dir| dir.join("rustnn").join(category))
            .ok_or(CacheError::FailedToDetermineCachePath)?;
        std::fs::create_dir_all(&root_path).map_err(|e| CacheError::FailedToCreateCachePath {
            path: root_path.clone(),
            source: e,
        })?;
        Ok(Self { root_path })
    }

    #[tracing::instrument(
        skip_all,
        level = "debug",
        fields(
            category = %cache_category(&self.root_path),
            key = %key,
            compressed = true,
        )
    )]
    fn get(&self, key: &str) -> CacheResult<Cow<'cache, [u8]>> {
        debug!("Looking up compressed cache key: {key}");
        let cache_path = self.cache_path(key);
        let result =
            read_zstd_cache_file(&cache_path).map_err(|e| CacheError::FailedToReadCacheFile {
                path: cache_path.clone(),
                source: e,
            });
        emit_cache_outcome(&result);
        result.map(Cow::Owned)
    }

    #[tracing::instrument(
        skip_all,
        err,
        level = "debug",
        fields(
            category = %cache_category(&self.root_path),
            key = %key,
            bytes = data.len(),
            compressed = true,
        )
    )]
    fn set(&self, key: &str, data: &[u8]) -> CacheResult<()> {
        debug!(
            "Setting compressed cache key: {key} with {} bytes",
            data.len()
        );
        let cache_path = self.cache_path(key);
        write_zstd_cache_file(&cache_path, data).map_err(|e| CacheError::FailedToWriteCacheFile {
            path: cache_path.clone(),
            source: e,
        })
    }
}

pub(crate) fn read_cache_file(cache_path: &Path) -> std::io::Result<Vec<u8>> {
    debug!("Reading cache file {cache_path:?}");
    let mut file = File::open(cache_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

pub(crate) fn write_cache_file(cache_path: &Path, content: &[u8]) -> std::io::Result<()> {
    debug!(
        "Trying to write {} bytes to cache file {cache_path:?}",
        content.len()
    );
    write_cache_file_atomically(cache_path, |file| file.write_all(content))
}

fn write_cache_file_atomically(
    cache_path: &Path,
    write: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let parent = cache_path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("cache path has no parent: {cache_path:?}"),
        )
    })?;
    let mut temporary_file = tempfile::Builder::new()
        .prefix(".rustnn-cache-")
        .tempfile_in(parent)?;
    write(temporary_file.as_file_mut())?;
    temporary_file.as_file_mut().sync_all()?;
    temporary_file
        .persist(cache_path)
        .map(|_| ())
        .map_err(|error| error.error)
}

#[cfg(feature = "zstd-cache-compression")]
pub(crate) fn read_zstd_cache_file(cache_path: &Path) -> std::io::Result<Vec<u8>> {
    debug!("Reading compressed cache file {cache_path:?}");
    let file = File::open(cache_path)?;
    let mut decoder = zstd::stream::read::Decoder::new(file)?;
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[cfg(feature = "zstd-cache-compression")]
pub(crate) fn write_zstd_cache_file(cache_path: &Path, content: &[u8]) -> std::io::Result<()> {
    debug!(
        "Trying to write {} bytes to compressed cache file {cache_path:?}",
        content.len()
    );
    write_cache_file_atomically(cache_path, |file| {
        let mut encoder = zstd::stream::write::Encoder::new(file, 0)?;
        encoder.write_all(content)?;
        encoder.finish()?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use tracing::Level;

    use super::*;
    use crate::instrumentation::test_support::events;

    #[cfg(feature = "zstd-cache-compression")]
    #[test]
    fn zstd_cache_round_trip_uses_zstd_file_extension() {
        let directory = tempfile::tempdir().unwrap();
        let cache = ZstdCompressedFileCache {
            root_path: directory.path().to_path_buf(),
        };
        let content = b"repeated cache data repeated cache data repeated cache data";

        cache.set("engine", content).unwrap();

        assert!(!directory.path().join("engine").exists());
        assert!(directory.path().join("engine.zstd").exists());
        assert_eq!(cache.get("engine").unwrap().as_ref(), content);
    }

    fn read_error(kind: std::io::ErrorKind) -> CacheError {
        CacheError::FailedToReadCacheFile {
            path: "entry.cache".into(),
            source: std::io::Error::from(kind),
        }
    }

    #[test]
    fn a_hit_reports_the_bytes_it_found() {
        // Called inside the `get` span, the way the cache reads do it.
        let reported = events(|| {
            tracing::debug_span!("get").in_scope(|| emit_cache_outcome(&Ok(vec![0; 8])));
        });

        assert_eq!(reported.len(), 1, "{reported:?}");
        assert_eq!(reported[0].level, Level::DEBUG);
        assert_eq!(reported[0].target, "rustnn::backends::caching");
        assert_eq!(reported[0].within.as_deref(), Some("get"));
        assert_eq!(reported[0].field("outcome"), Some("outcome=hit"));
        assert_eq!(reported[0].field("bytes"), Some("bytes=8"));
    }

    #[test]
    fn a_missing_file_is_a_miss() {
        let reported = events(|| {
            let missing = Err(read_error(std::io::ErrorKind::NotFound));
            tracing::debug_span!("get").in_scope(|| emit_cache_outcome(&missing));
        });

        assert_eq!(reported.len(), 1, "{reported:?}");
        assert_eq!(reported[0].level, Level::DEBUG);
        assert_eq!(reported[0].target, "rustnn::backends::caching");
        assert_eq!(reported[0].within.as_deref(), Some("get"));
        assert_eq!(reported[0].field("outcome"), Some("outcome=miss"));
        assert_eq!(reported[0].field("error"), None, "a miss is not an error");
    }

    #[test]
    fn a_failed_read_is_an_error() {
        let reported = events(|| {
            let denied = Err(read_error(std::io::ErrorKind::PermissionDenied));
            tracing::debug_span!("get").in_scope(|| emit_cache_outcome(&denied));
        });

        assert_eq!(reported.len(), 1, "{reported:?}");
        assert_eq!(reported[0].level, Level::ERROR);
        assert_eq!(reported[0].target, "rustnn::backends::caching");
        assert_eq!(reported[0].within.as_deref(), Some("get"));
        assert_eq!(reported[0].field("outcome"), Some("outcome=error"));
        assert!(
            reported[0].field("error").is_some(),
            "the error itself is reported: {reported:?}"
        );
    }
}
