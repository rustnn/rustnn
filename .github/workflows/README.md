# GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push, pull request | Cargo.lock consistency, `cargo fmt --check`, `cargo check` per feature (ONNX Runtime, TensorRT, LiteRT, CANN, CoreML on macOS, wasm32 with `webnn-runtime`), `cargo test --lib` on Linux and macOS plus the CANN mock, rustdoc with warnings denied (`make docs-api`), operator report drift check (`make docs-backend-ops-check`) with the generator's unit tests, MkDocs strict build, version check on release tags |
| `wpt-conformance.yml` | push, pull request | WPT conformance suites: ONNX Runtime and LiteRT on Linux (LiteRT non-blocking), CoreML on macOS; uploads JSON and HTML reports |
| `wpt-conformance-nightly.yml` | schedule, manual | Full WPT run with reports, then builds the documentation site with rustdoc under `/api/` and the conformance dashboard under `/wpt-conformance/`, and deploys to GitHub Pages |
| `snapshot-sync.yml` | weekly (Monday 03:00 UTC), manual | Regenerates PASS snapshots and expected-failure lists for LiteRT, ONNX Runtime and CoreML against the pinned WPT revision and opens a pull request with the diff |
| `rustnnpt-gate.yml` | pull request | Runs the external rustnnpt conformance runner against the PR's rustnn revision and enforces a minimum pass rate |
| `docs.yml` | push to `main` (docs, `mkdocs.yml`, `src/`, `Cargo.toml`, `Makefile`), pull request, manual | MkDocs strict build, rustdoc embedded under `/api/`, cached WPT report embedded, deploy to GitHub Pages from `main` |
| `docs-pr.yml` | pull request touching docs | MkDocs strict build, rustdoc build, link check, status comment on the PR |
| `publish.yml` | GitHub release, manual | fmt, clippy, tests, `cargo publish` to crates.io |

## Conventions

- The Rust version is pinned in `rust-toolchain.toml`; every workflow that installs Rust pins
  the same version. Bump them together (the toolchain file lists the workflows).
- `protoc` is installed in every job; `flatc` in jobs that build the `litert-runtime` feature.
- TensorRT-RTX has no GPU runner. CI compiles the backend (`cargo check -F trtx-runtime
  --all-targets`); its WPT snapshots are regenerated locally with `make wpt-sync-trtx`.
- The documentation site combines three generated parts: MkDocs pages from `docs/`, rustdoc from
  `make docs-api`, and the WPT dashboard cached by the nightly workflow. Test a docs change
  locally with `make ci-docs` and `make docs-api`.

## Pages deployment

GitHub Pages is configured with "GitHub Actions" as the source. `docs.yml` deploys on pushes to
`main`; the nightly workflow redeploys with fresh conformance data. If a deployment fails with a
permission error, check Settings -> Actions -> General -> Workflow permissions (read and write).
