//! Load WPT conformance tests from upstream `.https.any.js` files via a Node.js bridge.
//!
//! Mirrors pywebnn/tests/wpt_js_loader.py: one Node process dumps the full corpus.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::wpt_types::{WptCorpus, WptLoadedCase};

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn default_wpt_dir() -> PathBuf {
    std::env::var("WPT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join(".cache").join("wpt"))
}

pub fn bridge_dir() -> PathBuf {
    repo_root().join("scripts").join("wpt_bridge")
}

pub fn dump_corpus_script() -> PathBuf {
    bridge_dir().join("dump_corpus.mjs")
}

pub fn fetch_wpt_script() -> PathBuf {
    repo_root().join("scripts").join("fetch_wpt.mjs")
}

pub fn conformance_tests_dir(wpt_dir: &Path) -> PathBuf {
    wpt_dir.join("webnn").join("conformance_tests")
}

pub fn wpt_cache_available(wpt_dir: &Path) -> bool {
    conformance_tests_dir(wpt_dir).is_dir()
}

pub fn node_available() -> bool {
    Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_node(script: &Path, args: &[&str]) -> Result<String, String> {
    if !node_available() {
        return Err(
            "Node.js is required for WPT conformance tests but was not found on PATH".to_string(),
        );
    }
    let output = Command::new("node")
        .arg(script)
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn node {}: {}", script.display(), e))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(if output.stderr.is_empty() {
            &output.stdout
        } else {
            &output.stderr
        });
        return Err(format!(
            "Node WPT bridge failed (node {} {}): {}",
            script.display(),
            args.join(" "),
            detail.trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|e| format!("Node WPT bridge returned invalid UTF-8: {e}"))
}

pub fn fetch_wpt_cache(_wpt_dir: &Path) -> Result<(), String> {
    let script = fetch_wpt_script();
    if !script.is_file() {
        return Err(format!("WPT fetch script not found: {}", script.display()));
    }
    run_node(&script, &[])?;
    Ok(())
}

pub fn ensure_wpt_cache(wpt_dir: &Path) -> Result<(), String> {
    if wpt_cache_available(wpt_dir) {
        return Ok(());
    }
    fetch_wpt_cache(wpt_dir)
}

/// Load the full WPT conformance corpus in one Node.js invocation.
pub fn load_wpt_corpus(wpt_dir: &Path) -> Result<WptCorpus, String> {
    ensure_wpt_cache(wpt_dir)?;

    let script = dump_corpus_script();
    if !script.is_file() {
        return Err(format!("WPT dump script not found: {}", script.display()));
    }

    let wpt_arg = wpt_dir.to_string_lossy().to_string();
    let stdout = run_node(&script, &["--wpt-dir", &wpt_arg])?;
    let corpus: WptCorpus = serde_json::from_str(&stdout)
        .map_err(|e| format!("invalid WPT corpus JSON from {}: {e}", script.display()))?;

    if !corpus.file_errors.is_empty() {
        eprintln!(
            "[WPT] warning: {} conformance file(s) failed to parse:",
            corpus.file_errors.len()
        );
        for fe in corpus.file_errors.iter().take(5) {
            eprintln!("  {}: {}", fe.file_name, fe.error);
        }
        if corpus.file_errors.len() > 5 {
            eprintln!("  ... and {} more", corpus.file_errors.len() - 5);
        }
    }

    if corpus.cases.is_empty() {
        return Err(format!(
            "no WPT conformance cases loaded from {} (check WPT cache at {})",
            script.display(),
            wpt_dir.display()
        ));
    }

    Ok(corpus)
}

/// Sanitize a test name for use as a libtest filter id.
pub fn sanitize_test_id(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn trial_name(backend: &str, case: &WptLoadedCase) -> String {
    format!(
        "{}::{}::{}",
        backend,
        case.operation,
        sanitize_test_id(&case.name)
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanitize_replaces_filename_unsafe_characters() {
        assert_eq!(
            super::sanitize_test_id("relu float32: 2D <tensor> / \\ | ? *"),
            "relu_float32__2D__tensor___________"
        );
    }
}
