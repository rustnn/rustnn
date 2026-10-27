# GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push, pull request | Cargo.lock consistency, `cargo fmt --check`, `cargo check` per feature (ONNX Runtime, TensorRT, LiteRT, CANN, CoreML on macOS, wasm32 with `webnn-runtime`), `cargo test --lib` on Linux and macOS plus the CANN mock, rustdoc with warnings denied (`make docs-api`), operator report drift check (`make docs-backend-ops-check`) with the generator's unit tests, MkDocs strict build |
| `wpt-conformance.yml` | push, pull request | WPT conformance suites: ONNX Runtime, Burn CPU and LiteRT on Linux (LiteRT non-blocking), CoreML on macOS; uploads JSON and HTML reports |
| `wpt-conformance-nightly.yml` | schedule, manual | Full WPT run with reports, then builds the documentation site with rustdoc under `/api/` and the conformance dashboard under `/wpt-conformance/`, and deploys to GitHub Pages |
| `wpt-sync.yml` | weekly (Monday 03:00 UTC), manual | Bumps `WPT_REVISION` to the current upstream WPT revision, regenerates the expected-failure lists for LiteRT, ONNX Runtime and CoreML against it and opens a pull request with the diff |
| `rustnnpt-gate.yml` | pull request | Runs the external rustnnpt conformance runner against the PR's rustnn revision and enforces a minimum pass rate |
| `docs.yml` | push to `main` (docs, `mkdocs.yml`, `src/`, `Cargo.toml`, `Makefile`), pull request, manual | MkDocs strict build, rustdoc embedded under `/api/`, cached WPT report embedded, deploy to GitHub Pages from `main` |
| `docs-pr.yml` | pull request touching docs | MkDocs strict build, rustdoc build, link check, status comment on the PR |
| `publish.yml` | GitHub release, manual | release tag must match the `Cargo.toml` version, then fmt, clippy, tests, `cargo publish` to crates.io |

## Conventions

- Specification data type metadata is refreshed manually with `make fetch-webnn-spec`
  and `make generate-operation-data-types`. `make check-operation-data-types` checks
  against the local cached source; CI does not fetch the moving specification.
  Run the extractor regressions with
  `python3 -m unittest scripts/test_generate_operation_data_types.py`.

- The Rust version is pinned in `rust-toolchain.toml`; every workflow that installs Rust pins
  the same version. Bump them together (the toolchain file lists the workflows).
- `protoc` is installed in every job; `flatc` in jobs that build the `litert-runtime` feature.
- TensorRT-RTX has no GPU runner. CI compiles the backend (`cargo check -F trtx-runtime
  --all-targets`); its WPT snapshots are regenerated locally with `make wpt-sync-trtx`.
- macOS CI explicitly runs `make test-coreml-dtypes` with and without dynamic inputs.
  `make test-coreml-gelu` covers exact GELU precision, shapes, and compositions under
  both feature configurations, including all FP16 encodings under CPU/GPU/NPU requests.
  `make test-coreml-tensor-reuse` also runs in both configurations to check retained
  storage ownership, conversion, aliasing and resizing.
  It also builds and tests CoreML with and without dynamic inputs using `make build-coreml`
  and `make test-coreml`; both explicitly enable `cli` so CLI integration tests build and
  execute the CoreML binary. CLI tests are excluded when `cli` is disabled.
  `make test-coreml-gather` also runs the focused gather bounds and
  scalar-index shape regressions; numerical checks remain strict.
- `make test-coreml-precision` qualifies explicit float16 rounding in native Pipeline
  stages, including scalar/dynamic interfaces, fanout, masks, shared weights and special values.
  These ordinary integration tests also run through `make test-coreml`.
- The documentation site combines three generated parts: MkDocs pages from `docs/`, rustdoc from
  `make docs-api`, and the WPT dashboard cached by the nightly workflow. Test a docs change
- The documentation site combines four generated parts: MkDocs pages from `docs/`, the C/C++
  Doxygen reference under `/c-api/`, rustdoc from `make docs-api`, and the WPT dashboard cached
  by the nightly workflow. MkDocs runs `scripts/build_capi_docs.py` to regenerate the C header
  and embed Doxygen output. Website jobs install Doxygen and cbindgen. Test a docs change
  locally with `make ci-docs` and `make docs-api`.
- `benchmark-coreml-kv` runs self-contained attention/cache chaining by default; an external
  model/reference fixture is optional. macOS CI runs its numerical and cache-content gates
  via `make test-coreml-kv-benchmark`, without asserting a performance threshold.

## Pages deployment

GitHub Pages is configured with "GitHub Actions" as the source. `docs.yml` deploys on pushes to
`main`; the nightly workflow redeploys with fresh conformance data. If a deployment fails with a
permission error, check Settings -> Actions -> General -> Workflow permissions (read and write).

Before deploying, both workflows read the repository's Pages configuration and skip the deploy
step with a notice unless the source is "GitHub Actions". A fork without Pages runs every other
step, including the nightly's report cache save, and the `github-pages` artifact stays
downloadable from the run; a fork that sets the Pages source to "GitHub Actions" deploys its own
site. Any API error other than 404 fails the job instead of skipping the deployment.

The nightly saves its `reports/` directory to the Actions cache under
`wpt-conformance-pages-<os>-<run id>`; `docs.yml` and `docs-pr.yml` restore the newest entry by
the `wpt-conformance-pages-<os>-` prefix, so a docs deploy republishes the latest nightly
dashboard rather than a placeholder. Cache entries are immutable: never reuse a fixed key here,
because the save would fail on every later run and each docs deploy would republish the first
report forever. Entries that are not restored for seven days are evicted; a docs deploy then
falls back to the placeholder page until the next successful nightly run.

Burn CPU also runs full release-mode WPT conformance in `wpt-conformance.yml` on Linux.
The job is blocking, checks the pinned WPT revision, and uploads JSON and HTML reports.
It enables only `burn-runtime-cpu` and requires no external backend runtime library.
