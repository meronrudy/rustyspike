# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and this project adheres to Semantic Versioning.

## [Unreleased]

No changes yet.

## [0.1.1] - 2025-08-29

### Added
- Phase 7/8 documentation consolidation and release process:
  - Roadmap and architecture docs synchronized with current behavior.
  - Comprehensive release checklist covering semver, artifacts, CI, and publishing.
- CI workflows:
  - Workspace CI with help-sync checks for CLI subcommands (viz/snapshot/nir).
  - Dedicated FFI header generation workflow (cbindgen) with C header smoke-compile.
  - Release artifacts packaging workflow; archives tarball(s) and uploads to Releases and as CI artifacts.
  - Docs publish workflow (GitHub Pages) from `docs/`.
- CLI tests:
  - NIR compile/verify/run smoke tests.
  - Negative-path tests for invalid flags/topologies/formats.
  - Snapshot roundtrip tests (JSON and bincode) with deterministic sort-by-(pre, post, weight_bits).
- Storage examples:
  - VEVT roundtrip example at `examples/storage-vevt`.
  - VMSK placeholder example at `examples/storage-vmsk`.

### Changed
- Release checklist clarifications and specifics for attaching archives to GitHub Releases.
- Fixed/updated documentation links across architecture and getting-started.

### Fixed
- FFI doctest mode set to `rust,no_run` to avoid spurious failures.
- Snapshot JSON determinism test made robust by parsing/sorting instead of raw byte compare.

## [0.1.0] - YYYY-MM-DD
Initial workspace release (placeholder).
