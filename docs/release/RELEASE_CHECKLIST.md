# hSNN Release Checklist (Phases 7/8 Finalized)

This is the end-to-end, actionable release flow for the multi-crate workspace (core, CLI, FFI, storage, etc.). It codifies semantic versioning, CHANGELOG rules, cbindgen header generation/packaging, CI artifacts, docs publishing, and a smoke validation matrix with exact commands.

## 0) Prerequisites

- Toolchain: stable Rust per [rust-toolchain.toml](../../rust-toolchain.toml)
- Clean working tree; CI green on main
- Ensure crate interdependencies use compatible versions

## 1) Semantic Versioning (Workspace)

- [ ] Decide version bump per semver:
  - PATCH: bug fixes and docs only; no public API changes
  - MINOR: backward-compatible features (e.g., new CLI subcommands, additive FFI)
  - MAJOR: breaking API changes (CLI flags removal/rename, FFI signature changes)
- [ ] Update versions:
  - [ ] Root workspace version (if used) and each published crate’s version in Cargo.toml
  - [ ] Synchronize dependent crate version requirements
- [ ] Run a workspace dry check:
  - cargo metadata -q > /dev/null

## 2) Changelog Discipline

- [ ] Convert “Unreleased” to a dated section in [CHANGELOG.md](../../CHANGELOG.md)
- [ ] Group entries by Added/Changed/Fixed/Removed/Docs/CI
- [ ] Mention migration notes and any determinism-impacting changes
- [ ] Link key docs:
  - [docs/release/RELEASE_CHECKLIST.md](docs/release/RELEASE_CHECKLIST.md)
  - [docs/development/FFI-ABI.md](docs/development/FFI-ABI.md)

## 3) Build & Test Matrix

- [ ] Core (skip legacy tests if policy requires):
  - cargo check -p shnn-core --no-default-features --features "std math plastic-sum"
- [ ] Storage:
  - cargo test -p shnn-storage -- --nocapture
- [ ] CLI:
  - cargo build -p shnn-cli
- [ ] FFI (test with C ABI while skipping cbindgen in test jobs):
  - HSNN_SKIP_CBINDGEN=1 cargo test -p shnn-ffi --features c-bindings -- --nocapture
- [ ] Optional:
  - cargo clippy --workspace --all-features -- -D warnings
  - cargo fmt --all -- --check

Notes:
- HSNN_SKIP_CBINDGEN=1 disables header generation in build.rs to avoid toolchain constraints; a dedicated job generates headers (see section 4).

## 4) FFI Header Generation & Packaging (cbindgen)

- [ ] Dedicated CI job on modern toolchain:
  - Install cbindgen (cargo install cbindgen)
  - Generate header into repo path [crates/shnn-ffi/include/shnn_ffi.h](../../crates/shnn-ffi/include/shnn_ffi.h)
- [ ] Upload header as CI artifact:
  - Artifact name: shnn_ffi.h
  - Path: crates/shnn-ffi/include/shnn_ffi.h
- [ ] Local verification:
  - cargo build -p shnn-ffi --release
  - test -f crates/shnn-ffi/include/shnn_ffi.h

## 5) Release Artifacts (per target triple)

If distributing binaries or headers separately:

- [ ] Create per-target archives containing:
  - CLI binaries (optional)
  - Header: crates/shnn-ffi/include/shnn_ffi.h
  - LICENSE and top-level README.md
- [ ] Example packaging (macOS host shown):
  - mkdir -p dist && cp LICENSE README.md dist/
  - cp crates/shnn-ffi/include/shnn_ffi.h dist/
  - # add built binaries if applicable under dist/bin/<triple>/
  - tar -czf hsnn-artifacts-$TARGET.tar.gz -C dist .
- [ ] Attach archives to GitHub Release:
  - cbindgen-generated header (shnn_ffi.h)
  - Optional: prebuilt binaries for CLI (if distributing)
  - Docs archive (if applicable)

## 6) Docs Publishing

- [ ] Ensure updated docs in:
  - [docs/architecture/CLI_FIRST_ARCHITECTURE.md](../architecture/CLI_FIRST_ARCHITECTURE.md)
  - [docs/architecture/BINARY_SCHEMAS.md](../architecture/BINARY_SCHEMAS.md)
  - [docs/development/FFI-ABI.md](../development/FFI-ABI.md)
  - [docs/getting-started/README.md](../getting-started/README.md)
  - [docs/release/RELEASE_CHECKLIST.md](docs/release/RELEASE_CHECKLIST.md)
- [ ] Publish documentation
  - If using GitHub Pages via Actions:
    - Enable Pages for the repository with source "GitHub Actions"
    - Verify workflow exists: .github/workflows/docs.yml
    - Push to default branch to trigger build; confirm deployment URL in workflow summary
  - Otherwise: publish docs per repository policy (e.g., internal site or manual upload)

## 7) Smoke Validation Matrix (exact commands)

- Core:
  - cargo check -p shnn-core --no-default-features --features "std math plastic-sum"
- Storage:
  - cargo test -p shnn-storage -- --nocapture
- CLI (compile + run + snapshot):
  - cargo run -p shnn-cli -- nir compile -o out/model.nirt --neurons lif --plasticity stdp --inputs 4 --hidden 4 --outputs 2 --topology fully-connected --steps 100 --dt-us 100 --stimulus poisson --stimulus-rate 5.0
  - cargo run -p shnn-cli -- nir run out/model.nirt --output results/spikes.json --spikes-format json
  - cargo run -p shnn-cli -- snapshot export --backend graph --inputs 4 --hidden 4 --outputs 2 --weight 1.0 --format json --out results/weights.json
  - cargo run -p shnn-cli -- snapshot import --backend graph --inputs 4 --hidden 4 --outputs 2 --format json --input results/weights.json
- FFI:
  - HSNN_SKIP_CBINDGEN=1 cargo test -p shnn-ffi --features c-bindings -- --nocapture
- Viz (local):
  - cargo run -p shnn-cli -- viz serve --port 7878 --results-dir results --results-file results/spikes.json

## 8) Tag & Publish

- [ ] Tag git (e.g., vX.Y.Z) and push tags
- [ ] Ensure CI artifacts (including shnn_ffi.h) are uploaded
- [ ] Publish crates in dependency order:
  - cargo publish -p shnn-core
  - cargo publish -p shnn-storage
  - cargo publish -p shnn-cli
  - cargo publish -p shnn-ffi
  - … others as needed

## 9) Post-Release

- [ ] Install via Cargo and re-run CLI snapshot/NIR smoke
- [ ] Validate C integration using the published header (compile a tiny client)
- [ ] Open follow-up issues for regressions or docs clarifications

---

Notes:
- Use HSNN_SKIP_CBINDGEN=1 in generic test jobs; run a dedicated cbindgen header job to produce the distributable header for artifacts.
- Additive FFI/CLI changes are MINOR; removing/renaming symbols/flags is MAJOR.