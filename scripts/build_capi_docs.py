"""Generate and register the C/C++ API reference in each MkDocs build."""

import logging
from pathlib import Path
import shutil
import subprocess

from mkdocs.structure.files import File


log = logging.getLogger("mkdocs.hooks.capi")


def on_pre_build(config):
    """Generate the reference before MkDocs resolves navigation and page links."""
    root = Path(__file__).resolve().parents[1]
    log.info("Building C and C++ API documentation")
    subprocess.run(["make", "docs-capi"], cwd=root, check=True)
    destination = root / "target/doxygen/website/c-api"
    shutil.copytree(root / "target/doxygen/html", destination, dirs_exist_ok=True)


def on_files(files, config):
    """Register Doxygen HTML as site files so relative links resolve normally."""
    source = Path(__file__).resolve().parents[1] / "target/doxygen/website"
    for path in sorted((source / "c-api").rglob("*")):
        if path.is_file():
            files.append(
                File(
                    path.relative_to(source).as_posix(),
                    str(source),
                    config["site_dir"],
                    config["use_directory_urls"],
                )
            )
    return files
