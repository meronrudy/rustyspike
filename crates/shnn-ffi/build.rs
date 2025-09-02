use std::{env, fs, path::Path};

fn main() {
    // Re-run if FFI surface or config changes
    println!("cargo:rerun-if-changed=src/c_bindings.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    // Allow skipping cbindgen during tests/older cargo toolchains
    if env::var("HSNN_SKIP_CBINDGEN").ok().as_deref() == Some("1") {
        println!("cargo:warning=Skipping cbindgen due to HSNN_SKIP_CBINDGEN=1");
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");

    // Where Cargo will place build artifacts (always available)
    let header_out = Path::new(&out_dir).join("shnn_ffi.h");

    // Try to load cbindgen config if present; otherwise default
    let config_path = Path::new(&crate_dir).join("cbindgen.toml");
    let config = match cbindgen::Config::from_file(&config_path) {
        Ok(cfg) => cfg,
        Err(_) => cbindgen::Config::default(),
    };

    // Generate C header from the crate's public FFI surface
    let builder = cbindgen::Builder::new().with_crate(&crate_dir).with_config(config);

    let bindings = match builder.generate() {
        Ok(b) => b,
        Err(e) => {
            println!("cargo:warning=Skipping cbindgen header generation: {}", e);
            return;
        }
    };

    // 1) Emit into target OUT_DIR for build consumers
    bindings
        .write_to_file(&header_out);

    // 2) Emit a distributable header into the crate include/ directory
    let include_dir = Path::new(&crate_dir).join("include");
    let _ = fs::create_dir_all(&include_dir);
    let dist_header = include_dir.join("shnn_ffi.h");
    bindings
        .write_to_file(&dist_header);

    println!("cargo:warning=Generated C header: {}", dist_header.display());
}