//! hSNN CLI crate
//!
//! Purpose:
//! - Provide a CLI-first interface to the hSNN neuromorphic platform.
//! - Expose user-facing commands to compile, verify, and run NIR programs, list registered ops,
//!   and serve a minimal visualization UI backed by simple JSON endpoints.
//!
//! Public responsibilities (library view):
//! - Re-export the primary CLI entry (HsnnCli) for integration in binary and testing contexts.
//! - Expose command modules as a library so they can be invoked programmatically in tests or
//!   downstream automation if desired.
//!
//! Major commands (see [commands]):
//! - nir: compile (TOML/CLI → textual NIR), verify (parse + verify), run (parse → verify → compile_with_passes → run),
//!        op-list (dynamic registry introspection).
//! - viz: serve a minimal SPA (static files) and JSON endpoints (/api/health, /api/list, /api/spikes)
//!        to visualize spike rasters exported by nir run.
//! - study (scaffolded runner) and ttr (JSON mask placeholder) are present but not the current focus.
//!
//! Integration points:
//! - shnn_ir: parse_text/to_text for textual NIR serialization.
//! - shnn_compiler: verify_module, list_ops, compile_with_passes.
//!
//! Notes:
//! - The binary (src/main.rs) wires up logging and argument parsing, calling HsnnCli::execute().
//! - The library surface re-exports command modules to support integration testing without invoking
//!   an external process.

pub mod commands;
pub mod config;
pub mod error;
pub mod workspace;

pub use commands::HsnnCli;