#![allow(clippy::single_match)]
//! Visualization and analysis commands (completed minimal layer)
//! - Static server for SPA assets
//! - JSON endpoints:
//!   - GET /api/health
//!   - GET /api/list (list result files in results_dir)
//!   - GET /api/spikes[?file=relative.json] (serve spikes JSON)
//! - Frontend draws a basic spike raster on a Canvas using WebGL-like 2D primitives (Canvas 2D for portability)

use clap::{Args, Subcommand, ValueEnum};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use tracing::{error, info, warn};

use crate::error::{CliError, CliResult};

/// Visualization and analysis tools
#[derive(Args, Debug)]
pub struct VizCommand {
    #[command(subcommand)]
    pub sub: VizSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum VizSubcommand {
    /// Serve static viz and JSON endpoints
    Serve(VizServe),
    /// Export network topology (stub)
    Export(VizExport),
    /// Generate static plots (stub)
    Plot(VizPlot),
}

/// Start visualization server
#[derive(Args, Debug)]
pub struct VizServe {
    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to listen on
    #[arg(long, default_value = "7878")]
    pub port: u16,
    /// Directory containing simulation results (JSON)
    #[arg(long)]
    pub results_dir: Option<PathBuf>,
    /// Default results JSON to serve (if provided)
    #[arg(long)]
    pub results_file: Option<PathBuf>,
    /// Run in background (not implemented; logs note)
    #[arg(long)]
    pub background: bool,
}

#[derive(Args, Debug)]
pub struct VizExport {
    /// Input network (NIR or runtime export)
    pub input: PathBuf,
    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,
    /// Format
    #[arg(short, long, default_value = "graphml")]
    pub format: ExportFormat,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ExportFormat {
    Graphml,
    Dot,
    Json,
    Svg,
}

#[derive(Args, Debug)]
pub struct VizPlot {
    /// Input simulation results JSON
    pub input: PathBuf,
    /// Output directory
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Plot types
    #[arg(short, long, value_delimiter = ',')]
    pub plots: Vec<PlotType>,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum PlotType {
    Raster,
    Potential,
    Weight,
    Activity,
}

struct ServerState {
    static_root: PathBuf,
    results_dir: Option<PathBuf>,
    default_results: Option<PathBuf>,
}

impl VizCommand {
    pub async fn execute(self, _workspace: PathBuf, _config: Option<PathBuf>) -> CliResult<()> {
        match self.sub {
            VizSubcommand::Serve(cmd) => cmd.execute().await,
            VizSubcommand::Export(cmd) => cmd.execute().await,
            VizSubcommand::Plot(cmd) => cmd.execute().await,
        }
    }
}

impl VizServe {
    pub async fn execute(self) -> CliResult<()> {
        // Ensure static assets exist
        let static_root = ensure_static_assets()?;

        let addr = format!("{}:{}", self.host, self.port);
        info!("Starting viz server at http://{}", addr);
        if self.background {
            warn!("--background not implemented; running in foreground");
        }

        let state = Arc::new(ServerState {
            static_root,
            results_dir: self.results_dir.clone(),
            default_results: self.results_file.clone(),
        });

        let listener = TcpListener::bind(&addr)
            .map_err(|e| CliError::Generic(anyhow::anyhow!("bind {} failed: {}", addr, e)))?;

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let st = state.clone();
                    thread::spawn(move || {
                        if let Err(e) = handle_client(stream, st) {
                            error!("client error: {}", e);
                        }
                    });
                }
                Err(e) => error!("accept error: {}", e),
            }
        }
        Ok(())
    }
}

impl VizExport {
    pub async fn execute(self) -> CliResult<()> {
        info!(
            "Export stub: {} -> {} ({:?})",
            self.input.display(),
            self.output.display(),
            self.format
        );
        std::fs::write(&self.output, b"<!-- export not implemented yet -->")?;
        Ok(())
    }
}

impl VizPlot {
    pub async fn execute(self) -> CliResult<()> {
        let out = self.output.unwrap_or_else(|| PathBuf::from("plots"));
        std::fs::create_dir_all(&out)?;
        info!("Plot stub from {} -> {}", self.input.display(), out.display());
        Ok(())
    }
}

fn handle_client(mut stream: TcpStream, state: Arc<ServerState>) -> CliResult<()> {
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf)?;
    if n == 0 {
        return Ok(());
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let mut lines = req.lines();
    let request_line = lines.next().unwrap_or("");
    let (method, full_path) = parse_request_line(request_line);
    let (path, query) = split_path_query(&full_path);

    match (method, path.as_str()) {
        ("GET", "/api/health") => {
            let has_results = state.default_results.as_ref().map(|p| p.exists()).unwrap_or(false)
                || state
                    .results_dir
                    .as_ref()
                    .map(|d| d.exists())
                    .unwrap_or(false);
            let json = serde_json::json!({
                "ok": true,
                "has_results": has_results
            });
            respond_json(&mut stream, &serde_json::to_string(&json).unwrap())?;
        }
        ("GET", "/api/list") => {
            let mut entries: Vec<String> = Vec::new();
            if let Some(dir) = &state.results_dir {
                if dir.exists() {
                    if let Ok(read_dir) = std::fs::read_dir(dir) {
                        for e in read_dir.flatten() {
                            let p = e.path();
                            if matches!(p.extension().and_then(|s| s.to_str()), Some("json") | Some("vevt")) {
                                if let Ok(rel) = p.strip_prefix(dir) {
                                    if let Some(s) = rel.to_str() {
                                        entries.push(s.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            entries.sort();
            let json = serde_json::json!({ "files": entries });
            respond_json(&mut stream, &serde_json::to_string(&json).unwrap())?;
        }
        ("GET", "/api/spikes") => {
            // optional query ?file=relative.(json|vevt)
            let mut body = serde_json::json!({ "results": { "spike_count": 0, "spikes": [] }});
            if let Some(path_to_serve) = resolve_results_path(&state, query.as_deref()) {
                match path_to_serve.extension().and_then(|s| s.to_str()) {
                    Some("vevt") => {
                        match std::fs::read(&path_to_serve) {
                            Ok(bytes) => {
                                match shnn_storage::vevt::decode_vevt(&bytes) {
                                    Ok((_hdr, events)) => {
                                        let mut spikes = Vec::new();
                                        for ev in events {
                                            if ev.event_type == 0 {
                                                let time_ns = ev.timestamp;
                                                let neuron_id = ev.source_id;
                                                spikes.push(serde_json::json!({
                                                    "neuron_id": neuron_id,
                                                    "time_ns": time_ns,
                                                    "time_ms": time_ns as f64 / 1_000_000.0,
                                                }));
                                            }
                                        }
                                        let spike_count = spikes.len();
                                        body = serde_json::json!({ "results": { "spike_count": spike_count, "spikes": spikes }});
                                    }
                                    Err(e) => {
                                        warn!("failed to decode VEVT {}: {}", path_to_serve.display(), e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("failed to read {}: {}", path_to_serve.display(), e);
                            }
                        }
                    }
                    _ => {
                        match std::fs::read_to_string(&path_to_serve) {
                            Ok(text) => {
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                                    body = val;
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "failed to read results {}: {}",
                                    path_to_serve.display(),
                                    e
                                );
                            }
                        }
                    }
                }
            }
            respond_json(&mut stream, &serde_json::to_string(&body).unwrap())?;
        }
        ("GET", "/") => {
            let index = state.static_root.join("index.html");
            serve_path(&mut stream, &index)?;
        }
        ("GET", p) => {
            // sanitize path to prevent traversal
            let rel = p.trim_start_matches('/');
            let safe = Path::new(rel);
            if safe
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                respond_404(&mut stream)?;
            } else {
                let candidate = state.static_root.join(safe);
                if candidate.exists() {
                    serve_path(&mut stream, &candidate)?;
                } else {
                    respond_404(&mut stream)?;
                }
            }
        }
        _ => respond_404(&mut stream)?,
    }
    Ok(())
}

fn parse_request_line(line: &str) -> (&str, String) {
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let path = parts.next().unwrap_or("/");
    (method, path.to_string())
}

fn split_path_query(p: &str) -> (String, Option<String>) {
    if let Some(i) = p.find('?') {
        (p[..i].to_string(), Some(p[i + 1..].to_string()))
    } else {
        (p.to_string(), None)
    }
}

fn query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        if k == key {
            return Some(percent_decode(v));
        }
    }
    None
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                if let (Some(a), Some(b)) = (from_hex(bytes[i + 1]), from_hex(bytes[i + 2])) {
                    out.push((a << 4) | b);
                    i += 3;
                    continue;
                } else {
                    // invalid escape; copy literal '%'
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn from_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        b'A'..=b'F' => Some(10 + b - b'A'),
        _ => None,
    }
}

fn serve_path(stream: &mut TcpStream, path: &Path) -> CliResult<()> {
    let content = std::fs::read(path)?;
    let mime = mime_for_path(path);
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        mime,
        content.len()
    )?;
    stream.write_all(&content)?;
    Ok(())
}

fn respond_json(stream: &mut TcpStream, body: &str) -> CliResult<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    Ok(())
}

fn respond_404(stream: &mut TcpStream) -> CliResult<()> {
    let body = b"Not Found";
    write!(
        stream,
        "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    Ok(())
}

fn mime_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

fn resolve_results_path(state: &ServerState, query: Option<&str>) -> Option<PathBuf> {
    if let Some(q) = query {
        if let Some(file_rel) = query_param(q, "file") {
            if let Some(dir) = &state.results_dir {
                let rel = Path::new(&file_rel);
                // Disallow traversal
                if rel.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
                    return None;
                }
                let candidate = dir.join(rel);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }
    // Fall back to default file if present
    state.default_results.clone()
}

fn ensure_static_assets() -> CliResult<PathBuf> {
    let dir = Path::new("crates/shnn-cli/static/viz");
    std::fs::create_dir_all(dir)?;
    let index = dir.join("index.html");
    let css = dir.join("style.css");
    let js = dir.join("app.js");

    if !index.exists() {
        std::fs::write(&index, DEFAULT_INDEX)?;
    }
    if !css.exists() {
        std::fs::write(&css, DEFAULT_CSS)?;
    }
    if !js.exists() {
        std::fs::write(&js, DEFAULT_JS)?;
    }
    Ok(dir.to_path_buf())
}

const DEFAULT_INDEX: &str = r#"<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>hSNN Viz</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="stylesheet" href="/style.css">
  </head>
  <body>
    <header><h1>hSNN Visualization</h1></header>
    <main>
      <section class="controls">
        <button id="refresh">Refresh</button>
        <label>File:
          <select id="fileSelect"></select>
        </label>
        <span class="status">Status: <span id="status">checking...</span></span>
      </section>
      <canvas id="viz" width="1000" height="500"></canvas>
    </main>
    <script src="/app.js"></script>
  </body>
</html>
"#;

const DEFAULT_CSS: &str = r#"
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 0; }
header { background: #0b7285; color: white; padding: 0.75rem 1rem; }
main { padding: 1rem; }
.controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 0.5rem; }
.status { margin-left: auto; }
canvas { border: 1px solid #ccc; background: #000; width: 100%; height: auto; max-height: 70vh; }
label select { margin-left: 0.5rem; }
button { padding: 0.25rem 0.75rem; }
"#;

const DEFAULT_JS: &str = r#"
const statusEl = document.getElementById('status');
const fileSelect = document.getElementById('fileSelect');
const canvas = document.getElementById('viz');
const ctx = canvas.getContext('2d');
const refreshBtn = document.getElementById('refresh');

function setStatus(s) { statusEl.textContent = s; }

async function getJSON(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function refreshHealth() {
  try {
    const health = await getJSON('/api/health');
    setStatus(health.ok ? 'OK' : 'Error');
  } catch (e) {
    setStatus('Unavailable');
  }
}

async function listFiles() {
  try {
    const data = await getJSON('/api/list');
    fileSelect.innerHTML = '';
    for (const f of data.files || []) {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      fileSelect.appendChild(opt);
    }
  } catch {
    // ignore
  }
}

function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawRaster(spikes) {
  clearCanvas();
  if (!spikes || spikes.length === 0) {
    ctx.fillStyle = '#aaa';
    ctx.fillText('No spikes', 20, 20);
    return;
  }
  // Determine bounds
  let tMin = Infinity, tMax = -Infinity, nMin = Infinity, nMax = -Infinity;
  for (const s of spikes) {
    const t = s.time_ms ?? (s.time_ns ? s.time_ns / 1_000_000.0 : 0.0);
    const n = s.neuron_id ?? 0;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
    if (n < nMin) nMin = n;
    if (n > nMax) nMax = n;
  }
  if (!isFinite(tMin) || !isFinite(tMax)) { tMin = 0; tMax = 1; }
  if (!isFinite(nMin) || !isFinite(nMax)) { nMin = 0; nMax = 1; }
  const w = canvas.width, h = canvas.height;
  const padL = 40, padB = 20, padT = 10, padR = 10;
  const innerW = w - padL - padR, innerH = h - padT - padB;

  // axes
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + innerH);
  ctx.lineTo(padL + innerW, padT + innerH);
  ctx.stroke();

  // spikes
  ctx.fillStyle = '#0cf';
  const toX = t => padL + ((t - tMin) / Math.max(1e-6, (tMax - tMin))) * innerW;
  const toY = n => padT + innerH - ((n - nMin) / Math.max(1e-6, (nMax - nMin))) * innerH;

  const r = 1.5;
  for (const s of spikes) {
    const t = s.time_ms ?? (s.time_ns ? s.time_ns / 1_000_000.0 : 0.0);
    const n = s.neuron_id ?? 0;
    const x = toX(t);
    const y = toY(n);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // labels
  ctx.fillStyle = '#bbb';
  ctx.fillText(`${(tMax - tMin).toFixed(1)} ms`, padL + innerW - 60, padT + innerH + 14);
  ctx.save();
  ctx.translate(padL - 28, padT + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`Neuron IDs`, 0, 0);
  ctx.restore();
}

async function loadAndDraw(selectedFile=null) {
  let url = '/api/spikes';
  if (selectedFile) {
    url += `?file=${encodeURIComponent(selectedFile)}`;
  }
  try {
    const data = await getJSON(url);
    const spikes = (data.results && data.results.spikes) ? data.results.spikes : [];
    drawRaster(spikes);
  } catch {
    clearCanvas();
    ctx.fillStyle = '#f66';
    ctx.fillText('Failed to load spikes', 20, 20);
  }
}

refreshBtn.addEventListener('click', () => {
  const f = fileSelect.value || null;
  loadAndDraw(f);
});

fileSelect.addEventListener('change', () => {
  const f = fileSelect.value || null;
  loadAndDraw(f);
});

async function boot() {
  await refreshHealth();
  await listFiles();
  const f = fileSelect.value || null;
  await loadAndDraw(f);
}

boot();
"#;