# Contributing to hSNN

Thank you for your interest in contributing to hSNN. This document defines the baseline contribution workflow, coding standards, determinism policies, quality gates, and CI requirements. The goal is to keep builds green and reproducible across Linux, macOS, and Windows.

This project follows a CLI‑first, thin‑waist architecture. See docs for background:
- docs/CURRENT_STATE.md
- docs/ROADMAP.md
- docs/development/RFC-0001-Determinism-Modes.md

## Table of Contents
- Prerequisites
- Project layout
- MSRV and toolchain policy
- Build, lint, and test
- Determinism and reproducibility
- Feature flags and targets
- Conventional commits and PRs
- Code review checklist
- RFC process
- CI matrix and quality gates
- Security and licensing

## Prerequisites

- Rust (pinned via rust-toolchain.toml). This repo uses a pinned toolchain to ensure reproducible builds.
- Git and a GitHub account.
- Python (optional) for working with the Python bindings (later phases).
- Node (optional) for WASM demo (later phases).

## Project layout (high level)

- crates/shnn-core: thin‑waist core (data model, neuron/connectivity abstractions, time), designed to compile with and without std.
- crates/shnn-runtime: reference simulation engine used by the CLI.
- crates/shnn-ir / crates/shnn-compiler: textual Neuromorphic IR and compiler (verify, passes, lower).
- crates/shnn-cli: CLI entrypoints (nir compile/verify/run, viz server).
- crates/shnn-storage / crates/shnn-serialize: binary formats and zero‑copy serialization.
- crates/shnn-async-runtime, crates/shnn-lockfree, crates/shnn-math: performance/parallelization scaffolds.
- crates/shnn-ffi, crates/shnn-python, crates/shnn-wasm, crates/shnn-embedded: cross‑language and cross‑platform targets.

## MSRV and Toolchain Policy

- Minimum supported Rust version (MSRV): defined in rust-toolchain.toml.
- Toolchain components required: rustfmt, clippy.
- CI enforces MSRV and fails on new warnings.

## Build, Lint, and Test

Run the following before every commit:

- Format:
  - cargo fmt --all
- Lint (deny warnings):
  - cargo clippy --all-targets --all-features -D warnings
- Build:
  - cargo build --workspace --all-features
- Test:
  - cargo test --workspace --all-features

Notes:
- Prefer small, incremental PRs that keep CI green.
- Add unit tests and integration tests along with code changes.

## Determinism and Reproducibility

Determinism policy is specified in docs/development/RFC-0001-Determinism-Modes.md.

Quick summary:
- Fixed-step runs with a constant dt_ns and a set seed must produce stable spike streams on the same platform (OS/arch).
- Core ordering rules:
  - Input spikes sorted by (timestamp, source_id) when determinism is enabled.
  - Routing targets processed in ascending NeuronId when determinism is enabled.
  - Pending spike queue ties broken by (delivery_time, source_id) when determinism is enabled.

To run determinism smoke tests:
- cargo test -p shnn-core -- --nocapture
- cargo test -p shnn-runtime -- --nocapture

## Feature Flags and Targets

- Build with all features during CI to detect clashes: --all-features
- Exercise no-std compatibility in targeted crates (shnn-core, shnn-embedded, shnn-serialize) using feature gates in local testing.
- Use feature gates for heavy dependencies (parallel, simd, wasm) and keep defaults minimal.

Guidelines:
- Keep default features light.
- Document new feature flags in the crate README and lib.rs docs.
- Avoid accidental default-on performance flags that might hinder portability.

## Conventional Commits and PRs

Commit message format:
- type(scope): short description
  - types: feat, fix, perf, refactor, docs, test, chore, ci, build
  - scope: crate-name or subsystem (e.g., core, runtime, cli)

Examples:
- feat(core): add DeterminismConfig and stable target ordering
- fix(runtime): prevent overflow in delay computation
- perf(lockfree): optimize MPMC backoff strategy

PR guidelines:
- Link associated issues.
- Include rationale and highlights of tests added.
- Call out any API or ABI change explicitly.
- Keep PRs focused; split large changes.

## Code Review Checklist

- API surface:
  - Does the change keep thin‑waist stable (IR and public traits)?
  - Are Result types used instead of panics on error paths?
- Determinism:
  - If determinism is impacted, are RFC rules followed and tests added?
- Safety:
  - Any unsafe blocks have a clear justification and tests.
- Portability:
  - std/no‑std gates respected where appropriate.
- Tests:
  - Unit tests cover happy path and error path.
  - Determinism and ordering tests where applicable.
- Docs:
  - Public items documented; examples added where helpful.

## RFC Process

Propose API, ABI, or format changes via an RFC in docs/development/:
- RFC-XXXX-Title.md format.
- Include background, proposal, alternatives, migration plan, and acceptance tests.
- Add cross-references to code files (paths and lines) in the PR description.

## CI Matrix and Quality Gates

This repo uses GitHub Actions (see .github/workflows/ci.yml):

- OS: Linux, macOS, Windows
- Toolchain: stable
- Steps:
  - cargo fmt --check
  - cargo clippy -D warnings
  - cargo build --workspace --all-features
  - cargo test --workspace --all-features

Contributors should run the same steps locally. CI must be green before merge.

## Security and Licensing

- Dual-license: MIT OR Apache-2.0. All contributions are licensed under the same terms.
- Avoid introducing incompatible dependencies.
- Use cargo-audit and cargo-deny locally for dependency checks where feasible (to be integrated in CI later).
- Do not check secrets or credentials into the repository.

## Filing Issues

When filing a bug:
- Include OS, toolchain version (rustc -V), feature flags used, and reproduction steps.
- Attach IR snippets or minimal code examples if possible.

When requesting features:
- Provide a rationale, expected benefits, and impact on determinism, API, or ABI.

## Getting Help

Open a discussion or issue for questions. Please include relevant code pointers and expected outcomes.

Thank you for contributing to hSNN!