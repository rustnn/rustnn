use std::borrow::Cow;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use log::debug;

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Failed to create cache path")]
    FailedToDetermineCachePath,

    #[error("Failed to create cache path: {path:?} ({source})")]
    FailedToCreateCachePath {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Failed to write cache file: {path:?} ({source})")]
    FailedToWriteCacheFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Failed to read cache file: {path:?} ({source})")]
    FailedToReadCacheFile {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub type CacheResult<T> = std::result::Result<T, CacheError>;

pub trait PersistentCache<'cache>: Sized + Send + Sync + std::fmt::Debug {
    fn new(category: &str) -> CacheResult<Self>;
    fn get(&self, key: &str) -> CacheResult<Cow<'cache, [u8]>>;
    fn set(&self, key: &str, data: &[u8]) -> CacheResult<()>;
}

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

    fn get(&self, key: &str) -> CacheResult<Cow<'cache, [u8]>> {
        debug!("Looking up cache key: {key}");
        let cache_path = self.root_path.join(key);
        read_cache_file(&cache_path)
            .map_err(|e| CacheError::FailedToReadCacheFile {
                path: cache_path.clone(),
                source: e,
            })
            .map(Cow::Owned)
    }

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

pub type DefaultCache = SimpleFileCache;

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

    fn get(&self, key: &str) -> CacheResult<Cow<'cache, [u8]>> {
        debug!("Looking up compressed cache key: {key}");
        let cache_path = self.cache_path(key);
        read_zstd_cache_file(&cache_path)
            .map_err(|e| CacheError::FailedToReadCacheFile {
                path: cache_path.clone(),
                source: e,
            })
            .map(Cow::Owned)
    }

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

#[cfg(all(test, feature = "zstd-cache-compression"))]
mod tests {
    use super::*;

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
}
