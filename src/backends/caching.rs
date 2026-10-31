//! Persistent on-disk caches used by backends: TensorRT engines and its runtime cache as byte
//! blobs ([`PersistentCache`]), and CoreML compiled bundles as directories
//! ([`BundleCacheEntry`]).
//!
//! Blob entries live under `<platform cache dir>/rustnn/<category>/<key>`, for example
//! `~/.cache/rustnn/trtx` on Linux or `%LOCALAPPDATA%\rustnn\trtx` on Windows. Writes go
//! through a temporary file and an atomic rename. With the `zstd-cache-compression` feature
//! entries are stored compressed with a `.zstd` suffix.
//!
//! Directory entries live under a caller-chosen directory as `<cache_dir>/<key>/` and hold a
//! bundle directory plus a `manifest.json`. They are written into a staging directory beside
//! the entry and published with one atomic rename.

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
    match read_outcome(result) {
        ReadOutcome::Hit(bytes) => tracing::debug!(outcome = "hit", bytes = bytes),
        ReadOutcome::Miss => tracing::debug!(outcome = "miss"),
        ReadOutcome::Error(error) => tracing::error!(outcome = "error", error = %error),
    }
}

/// What a blob cache read found. A missing entry arrives as an `Err` with `NotFound`, which is
/// how a miss is told apart from a real I/O error.
#[derive(Debug)]
enum ReadOutcome<'a> {
    Hit(usize),
    Miss,
    Error(&'a CacheError),
}

fn read_outcome(result: &CacheResult<Vec<u8>>) -> ReadOutcome<'_> {
    match result {
        Ok(data) => ReadOutcome::Hit(data.len()),
        Err(CacheError::FailedToReadCacheFile { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            ReadOutcome::Miss
        }
        Err(error) => ReadOutcome::Error(error),
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

// ---------------------------------------------------------------------------
// Directory entries (compiled bundles)
// ---------------------------------------------------------------------------

/// Keys are path components chosen by the caller: 1 to 128 ASCII letters, digits, `-`, `_`
/// or `.`, not starting with `.`.
pub fn validate_cache_key(key: &str) -> std::io::Result<()> {
    let valid = !key.is_empty()
        && key.len() <= 128
        && !key.starts_with('.')
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid cache key {key:?}"),
        ))
    }
}

/// What a [`BundleCacheEntry`] records next to its bundle, as `manifest.json`.
///
/// The manifest is the receipt of the write: the inputs the key was derived from (so a key
/// collision is detected instead of loading another graph's bundle), the crate and OS that
/// produced it, when, and how many bytes the bundle holds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BundleManifest {
    /// Named inputs of the key, in the order the backend hashed them.
    pub key_inputs: Vec<(String, String)>,
    /// `CARGO_PKG_VERSION` of the writer.
    pub crate_version: String,
    /// Unix time (seconds) of the write.
    pub created_unix_seconds: u64,
    /// Operating system build of the writer, as the backend reports it.
    pub os_build: String,
    /// Total size of the regular files in the bundle, in bytes.
    pub bundle_bytes: u64,
}

/// One directory-valued cache entry: `<cache_dir>/<key>/` holding `<bundle>` (a directory)
/// and [`BundleManifest`] as `manifest.json`.
///
/// Writers fill a staging directory `<cache_dir>/<key>.partial-<pid>` and publish it with one
/// rename, so a reader sees either no entry, a partial entry without a manifest (a crashed or
/// concurrent writer) or a complete one. A partial entry is treated as a miss and replaced by
/// the next publish. The cache never evicts other keys and never deletes a complete entry on
/// its own; the owner of the directory decides when to remove entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleCacheEntry {
    key: String,
    dir: PathBuf,
    bundle_name: String,
}

impl BundleCacheEntry {
    const MANIFEST: &'static str = "manifest.json";

    /// The entry for `key` under `cache_dir`, whose bundle directory is named `bundle_name`
    /// (CoreML needs the `.mlmodelc` suffix to load one). Fails for an invalid key.
    pub fn new(cache_dir: &Path, key: &str, bundle_name: &str) -> std::io::Result<Self> {
        validate_cache_key(key)?;
        Ok(Self {
            key: key.to_owned(),
            dir: cache_dir.join(key),
            bundle_name: bundle_name.to_owned(),
        })
    }

    /// The key this entry was created with.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The entry directory `<cache_dir>/<key>`.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The bundle directory inside the entry.
    pub fn bundle(&self) -> PathBuf {
        self.dir.join(&self.bundle_name)
    }

    /// A complete entry has its manifest and its bundle directory. The manifest is written
    /// last inside the staging directory, so a partial entry never has one.
    pub fn is_complete(&self) -> bool {
        self.dir.join(Self::MANIFEST).is_file() && self.bundle().is_dir()
    }

    /// Read the manifest of a complete entry.
    pub fn manifest(&self) -> std::io::Result<BundleManifest> {
        let path = self.dir.join(Self::MANIFEST);
        let bytes = std::fs::read(&path)?;
        serde_json::from_slice(&bytes).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid cache manifest {}: {error}", path.display()),
            )
        })
    }

    /// Remove the entry (complete or partial). A missing entry is not an error.
    pub fn remove(&self) -> std::io::Result<()> {
        match std::fs::remove_dir_all(&self.dir) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            result => result,
        }
    }

    /// Fill a staging directory beside the entry through `write_bundle`, which receives the
    /// path of the staged bundle directory (not yet created), then record `manifest` with the
    /// bundle's byte count and publish the entry with one atomic rename.
    ///
    /// Stale staging directories of this key from earlier processes are removed first, and so
    /// is a partial entry at the final path. When another process published the same key
    /// meanwhile, its complete entry wins and this one is discarded. On any failure the
    /// staging directory is removed and the final path is left as it was.
    pub fn publish(
        &self,
        mut manifest: BundleManifest,
        write_bundle: impl FnOnce(&Path) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let parent = self.dir.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("cache entry has no parent: {}", self.dir.display()),
            )
        })?;
        std::fs::create_dir_all(parent)?;
        let prefix = format!("{}.partial-", self.key);
        if let Ok(entries) = std::fs::read_dir(parent) {
            for stale in entries.flatten() {
                if stale.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = std::fs::remove_dir_all(stale.path());
                }
            }
        }
        let staging = parent.join(format!("{prefix}{}", std::process::id()));
        std::fs::create_dir_all(&staging)?;
        let staged_bundle = staging.join(&self.bundle_name);
        let written = write_bundle(&staged_bundle).and_then(|()| {
            if !staged_bundle.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("bundle was not written to {}", staged_bundle.display()),
                ));
            }
            manifest.bundle_bytes = directory_byte_count(&staged_bundle)?;
            let json = serde_json::to_vec_pretty(&manifest).map_err(std::io::Error::other)?;
            write_cache_file(&staging.join(Self::MANIFEST), &json)
        });
        if let Err(error) = written {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error);
        }
        // A partial entry at the final path (crashed writer) would make the rename fail.
        if self.dir.exists() && !self.is_complete() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
        match std::fs::rename(&staging, &self.dir) {
            Ok(()) => {
                debug!("Published cache entry {}", self.dir.display());
                Ok(())
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(&staging);
                if self.is_complete() {
                    debug!(
                        "Cache entry {} was published concurrently; keeping it",
                        self.dir.display()
                    );
                    Ok(())
                } else {
                    Err(error)
                }
            }
        }
    }
}

/// Sum of the sizes of the regular files below `dir`.
fn directory_byte_count(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            total += directory_byte_count(&entry.path())?;
        } else if file_type.is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

/// Copy the directory tree `src` to `dst`, replacing an existing `dst`.
pub fn copy_directory(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_directory(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
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

    fn manifest() -> BundleManifest {
        BundleManifest {
            key_inputs: vec![("program".into(), "abc".into())],
            crate_version: "0.0.0".into(),
            created_unix_seconds: 7,
            os_build: "test".into(),
            bundle_bytes: 0,
        }
    }

    fn write_bundle(bytes: &[u8]) -> impl FnOnce(&Path) -> std::io::Result<()> + '_ {
        move |bundle: &Path| {
            std::fs::create_dir_all(bundle.join("nested"))?;
            std::fs::write(bundle.join("model.mil"), bytes)?;
            std::fs::write(bundle.join("nested").join("weight.bin"), b"wx")
        }
    }

    #[test]
    fn cache_keys_are_single_safe_path_components() {
        for key in ["a", "abc-DEF_0.9", &"k".repeat(128)] {
            validate_cache_key(key).unwrap();
        }
        for key in [
            "",
            ".",
            "..",
            ".hidden",
            "a/b",
            "a\\b",
            "a b",
            "ключ",
            &"k".repeat(129),
        ] {
            assert!(validate_cache_key(key).is_err(), "{key:?}");
        }
        let cache = tempfile::tempdir().unwrap();
        assert!(BundleCacheEntry::new(cache.path(), "../escape", "b").is_err());
        let entry = BundleCacheEntry::new(cache.path(), "abc", "bundle.mlmodelc").unwrap();
        assert_eq!(entry.key(), "abc");
        assert_eq!(entry.dir(), cache.path().join("abc"));
        assert_eq!(
            entry.bundle(),
            cache.path().join("abc").join("bundle.mlmodelc")
        );
    }

    #[test]
    fn bundle_entries_are_complete_only_with_manifest_and_bundle() {
        let cache = tempfile::tempdir().unwrap();
        let entry = BundleCacheEntry::new(cache.path(), "k", "b").unwrap();
        assert!(!entry.is_complete());
        std::fs::create_dir_all(entry.bundle()).unwrap();
        assert!(!entry.is_complete(), "bundle without manifest is partial");
        std::fs::write(entry.dir().join("manifest.json"), b"{}").unwrap();
        assert!(entry.is_complete());
        assert!(
            entry.manifest().is_err(),
            "an unreadable manifest is an error"
        );
        std::fs::remove_dir_all(entry.bundle()).unwrap();
        assert!(!entry.is_complete(), "manifest without bundle is partial");
        entry.remove().unwrap();
        assert!(!entry.dir().exists());
        entry.remove().unwrap();
    }

    #[test]
    fn publish_stages_renames_and_removes_stale_staging() {
        let cache = tempfile::tempdir().unwrap();
        let root = cache.path().join("compiled");
        let entry = BundleCacheEntry::new(&root, "tiny", "bundle.mlmodelc").unwrap();
        let stale = root.join("tiny.partial-1");
        std::fs::create_dir_all(stale.join("bundle.mlmodelc")).unwrap();
        let other = root.join("other.partial-1");
        std::fs::create_dir_all(&other).unwrap();

        entry.publish(manifest(), write_bundle(b"program")).unwrap();

        assert!(entry.is_complete());
        assert_eq!(
            std::fs::read(entry.bundle().join("model.mil")).unwrap(),
            b"program"
        );
        let recorded = entry.manifest().unwrap();
        assert_eq!(recorded.key_inputs, manifest().key_inputs);
        assert_eq!(
            recorded.bundle_bytes,
            7 + 2,
            "regular files below the bundle"
        );
        assert!(!stale.exists(), "stale staging of this key removed");
        assert!(other.exists(), "other keys untouched");
        let names: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            !names.iter().any(|n| n.starts_with("tiny.partial-")),
            "{names:?}"
        );
        // Publishing again onto a complete entry keeps a complete entry.
        entry.publish(manifest(), write_bundle(b"second")).unwrap();
        assert!(entry.is_complete());
    }

    #[test]
    fn publish_replaces_a_partial_entry() {
        let cache = tempfile::tempdir().unwrap();
        let entry = BundleCacheEntry::new(cache.path(), "k", "b").unwrap();
        std::fs::create_dir_all(entry.bundle()).unwrap();
        std::fs::write(entry.bundle().join("garbage"), b"x").unwrap();
        assert!(!entry.is_complete());

        entry.publish(manifest(), write_bundle(b"fresh")).unwrap();

        assert!(entry.is_complete());
        assert!(!entry.bundle().join("garbage").exists());
        assert_eq!(
            std::fs::read(entry.bundle().join("model.mil")).unwrap(),
            b"fresh"
        );
    }

    #[test]
    fn publish_failure_leaves_no_entry() {
        let cache = tempfile::tempdir().unwrap();
        let entry = BundleCacheEntry::new(cache.path(), "bad", "b").unwrap();
        let error = entry
            .publish(manifest(), |_bundle| {
                Err(std::io::Error::other("compiler failed"))
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "compiler failed");
        assert!(!entry.dir().exists());
        let leftovers: Vec<_> = std::fs::read_dir(cache.path()).unwrap().collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");

        // A writer that returns Ok without creating the bundle is a failure too.
        assert!(entry.publish(manifest(), |_bundle| Ok(())).is_err());
        assert!(!entry.dir().exists());
    }

    #[test]
    fn copy_directory_copies_nested_trees() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        write_bundle(b"abc")(&src).unwrap();
        let dst = dir.path().join("dst");
        std::fs::create_dir_all(dst.join("old")).unwrap();
        copy_directory(&src, &dst).unwrap();
        assert!(!dst.join("old").exists(), "destination is replaced");
        assert_eq!(std::fs::read(dst.join("model.mil")).unwrap(), b"abc");
        assert_eq!(
            std::fs::read(dst.join("nested").join("weight.bin")).unwrap(),
            b"wx"
        );
        assert_eq!(directory_byte_count(&dst).unwrap(), 5);
    }
}
