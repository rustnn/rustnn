"""Generate and embed the C/C++ API reference after each MkDocs build."""

import logging
from pathlib import Path
import shutil
import subprocess


log = logging.getLogger("mkdocs.hooks.capi")


def on_post_build(config):
    """Keep the website's C/C++ reference in sync with the source headers."""
    root = Path(__file__).resolve().parents[1]
    log.info("Building C and C++ API documentation")
    subprocess.run(["make", "docs-capi"], cwd=root, check=True)
    destination = Path(config["site_dir"]) / "c-api"
    shutil.copytree(root / "target/doxygen/html", destination, dirs_exist_ok=True)
    log.info("Embedded C and C++ API documentation in %s", destination)
