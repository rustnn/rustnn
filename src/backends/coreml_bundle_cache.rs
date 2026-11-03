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
/// Writers fill a staging directory `<cache_dir>/<key>.partial-<pid>` and publish it with one
/// rename, so a reader sees either no entry, a partial entry without a manifest (a crashed or
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

    /// Fill a staging directory beside the entry through `write_bundle`, which receives the
    /// path of the staged bundle directory (not yet created), then record `manifest` with the
    /// bundle's byte count and publish the entry with one atomic rename.
    ///
    /// Stale staging directories of this key from earlier processes are removed first, and so
    /// is a partial entry at the final path. When another process published the same key
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
pub(crate) fn copy_directory(src: &Path, dst: &Path) -> std::io::Result<()> {
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
