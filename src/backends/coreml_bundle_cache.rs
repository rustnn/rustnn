//! Directory entries of the CoreML compiled-bundle cache (`CoremlOptions::compiled_bundle_cache`).
//!
//! The on-disk layout (`<cache_dir>/<key>/` with a bundle directory and a `manifest.json`) is an
//! implementation detail of the CoreML backend, not a format for other crates to read; the
//! blob caches in [`crate::backends::caching`] are unrelated and unchanged by it. Entries are
//! filled in a staging directory beside the entry and published with one atomic rename.

use std::path::{Path, PathBuf};

use log::debug;

use crate::backends::caching::write_cache_file;

/// Keys are path components chosen by the caller: 1 to 128 ASCII letters, digits, `-`, `_`
/// or `.`, not starting with `.`.
pub(crate) fn validate_cache_key(key: &str) -> std::io::Result<()> {
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
pub(crate) struct BundleManifest {
    /// Named inputs of the key, in the order the backend hashed them.
    pub(crate) key_inputs: Vec<(String, String)>,
    /// `CARGO_PKG_VERSION` of the writer.
    pub(crate) crate_version: String,
    /// Unix time (seconds) of the write.
    pub(crate) created_unix_seconds: u64,
    /// Operating system build of the writer, as the backend reports it.
    pub(crate) os_build: String,
    /// Total size of the regular files in the bundle, in bytes.
    pub(crate) bundle_bytes: u64,
}

/// One directory-valued cache entry: `<cache_dir>/<key>/` holding `<bundle>` (a directory)
/// and [`BundleManifest`] as `manifest.json`.
///
/// Writers fill a staging directory `<cache_dir>/<key>.partial-<pid>-<random>` of their own
/// and publish it with one rename, so a reader sees either no entry, a partial entry without a manifest (a crashed or
/// concurrent writer) or a complete one. A partial entry is treated as a miss and replaced by
/// the next publish. The cache never evicts other keys and never deletes a complete entry on
/// its own; the owner of the directory decides when to remove entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundleCacheEntry {
    key: String,
    dir: PathBuf,
    bundle_name: String,
}

impl BundleCacheEntry {
    const MANIFEST: &'static str = "manifest.json";

    /// The entry for `key` under `cache_dir`, whose bundle directory is named `bundle_name`
    /// (CoreML needs the `.mlmodelc` suffix to load one). Fails for an invalid key.
    pub(crate) fn new(cache_dir: &Path, key: &str, bundle_name: &str) -> std::io::Result<Self> {
        validate_cache_key(key)?;
        Ok(Self {
            key: key.to_owned(),
            dir: cache_dir.join(key),
            bundle_name: bundle_name.to_owned(),
        })
    }

    /// The entry directory `<cache_dir>/<key>`.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// The bundle directory inside the entry.
    pub(crate) fn bundle(&self) -> PathBuf {
        self.dir.join(&self.bundle_name)
    }

    /// A complete entry has its manifest and its bundle directory. The manifest is written
    /// last inside the staging directory, so a partial entry never has one.
    pub(crate) fn is_complete(&self) -> bool {
        self.dir.join(Self::MANIFEST).is_file() && self.bundle().is_dir()
    }

    /// Read the manifest of a complete entry.
    pub(crate) fn manifest(&self) -> std::io::Result<BundleManifest> {
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
    pub(crate) fn remove(&self) -> std::io::Result<()> {
        match std::fs::remove_dir_all(&self.dir) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            result => result,
        }
    }

    fn staging_prefix(&self) -> String {
        format!("{}.partial-", self.key)
    }

    /// Remove staging directories of this key whose writer is provably gone: the name
    /// carries the writer's process id, and the directory goes only when no process has
    /// that id. A live writer's directory (another process, or another publish of this
    /// one) is never touched, so two writers of one key cannot empty each other's bundle.
    fn remove_stale_staging(&self, parent: &Path) {
        let prefix = self.staging_prefix();
        let Ok(entries) = std::fs::read_dir(parent) else {
            return;
        };
        for candidate in entries.flatten() {
            let name = candidate.file_name();
            let Some(rest) = name.to_str().and_then(|name| name.strip_prefix(&prefix)) else {
                continue;
            };
            let writer = rest
                .split('-')
                .next()
                .and_then(|pid| pid.parse::<u32>().ok());
            if writer.is_some_and(|pid| pid != std::process::id() && process_is_gone(pid)) {
                let _ = std::fs::remove_dir_all(candidate.path());
            }
        }
    }

    /// Fill a staging directory beside the entry through `write_bundle`, which receives the
    /// path of the staged bundle directory (not yet created), then record `manifest` with the
    /// bundle's byte count and publish the entry with one atomic rename.
    ///
    /// Staging directories of this key left by processes that no longer exist are removed
    /// first, and so is a partial entry at the final path. When another writer published the
    /// same key
    /// meanwhile, its complete entry wins and this one is discarded. On any failure the
    /// staging directory is removed and the final path is left as it was.
    pub(crate) fn publish(
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
        self.remove_stale_staging(parent);
        // A directory of this publish alone: the process id says who owns it, the random
        // suffix keeps two publishes of one process apart. Dropping it removes it.
        let staging_dir = tempfile::Builder::new()
            .prefix(&format!("{}{}-", self.staging_prefix(), std::process::id()))
            .tempdir_in(parent)?;
        let staging = staging_dir.path().to_path_buf();
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
        written?;
        // A partial entry at the final path (crashed writer) would make the rename fail.
        if self.dir.exists() && !self.is_complete() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
        match std::fs::rename(&staging, &self.dir) {
            Ok(()) => {
                // The directory now lives at the entry path; nothing is left to remove.
                let _ = staging_dir.keep();
                debug!("Published cache entry {}", self.dir.display());
                Ok(())
            }
            Err(error) => {
                drop(staging_dir);
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

/// Whether no process has this id: `kill(pid, 0)` fails with `ESRCH`. A process this one may
/// not signal (`EPERM`) exists, and so does any id the call cannot classify.
fn process_is_gone(pid: u32) -> bool {
    const ESRCH: i32 = 3;
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    // Signal 0 performs the existence and permission checks and sends nothing.
    pid > 0
        && unsafe { kill(pid, 0) } == -1
        && std::io::Error::last_os_error().raw_os_error() == Some(ESRCH)
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The id of a process that has exited and been reaped.
    fn dead_process_id() -> u32 {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        pid
    }

    #[test]
    fn publish_stages_renames_and_removes_only_dead_writers_staging() {
        let cache = tempfile::tempdir().unwrap();
        let root = cache.path().join("compiled");
        let entry = BundleCacheEntry::new(&root, "tiny", "bundle.mlmodelc").unwrap();
        let dead = dead_process_id();
        let stale = [
            root.join(format!("tiny.partial-{dead}")),
            root.join(format!("tiny.partial-{dead}-AbC123")),
        ];
        // Process 1 always exists, and so does this one: both are live writers.
        let live = [
            root.join("tiny.partial-1-AbC123"),
            root.join(format!("tiny.partial-{}-AbC123", std::process::id())),
            root.join("tiny.partial-unparsed"),
        ];
        let other = root.join(format!("other.partial-{dead}-AbC123"));
        for dir in stale.iter().chain(&live).chain([&other]) {
            std::fs::create_dir_all(dir.join("bundle.mlmodelc")).unwrap();
        }

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
        for dir in &stale {
            assert!(!dir.exists(), "staging of a dead writer removed: {dir:?}");
        }
        for dir in &live {
            assert!(dir.exists(), "staging of a live writer kept: {dir:?}");
        }
        assert!(other.exists(), "other keys untouched");
        let mut names: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names.len(),
            1 + live.len() + 1,
            "no staging of this publish: {names:?}"
        );
        // Publishing again onto a complete entry keeps a complete entry.
        entry.publish(manifest(), write_bundle(b"second")).unwrap();
        assert!(entry.is_complete());
    }

    #[test]
    fn concurrent_publishes_of_one_key_leave_a_whole_entry() {
        let cache = tempfile::tempdir().unwrap();
        for round in 0..20 {
            let key = format!("k{round}");
            let barrier = std::sync::Barrier::new(8);
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    scope.spawn(|| {
                        let entry = BundleCacheEntry::new(cache.path(), &key, "b").unwrap();
                        barrier.wait();
                        entry.publish(manifest(), write_bundle(b"program")).unwrap();
                    });
                }
            });
            let entry = BundleCacheEntry::new(cache.path(), &key, "b").unwrap();
            assert!(entry.is_complete(), "round {round}");
            assert_eq!(
                std::fs::read(entry.bundle().join("model.mil")).unwrap(),
                b"program"
            );
            assert_eq!(
                std::fs::read(entry.bundle().join("nested").join("weight.bin")).unwrap(),
                b"wx"
            );
            assert_eq!(entry.manifest().unwrap().bundle_bytes, 9);
        }
        let leftovers: Vec<_> = std::fs::read_dir(cache.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".partial-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
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
    fn bundle_bytes_count_nested_regular_files() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("bundle");
        write_bundle(b"abc")(&bundle).unwrap();
        assert_eq!(directory_byte_count(&bundle).unwrap(), 5);
    }
}
