//! GPU support: detection for the UI and, later, the compute device layer (see docs/GPU-PLAN.md).
//!
//! - `discovery`: adapter enumeration through wgpu (G1). Cargo feature `gpu` (on by default).
//! - `select`: the adapter preference rule (pure). `device`: device, buffers, pool, pipeline cache (G2).
//! - `kernels`: first compute kernels vs CPU references (G2). `bench`: memory bandwidth (G3, `--gpu-bench`).
//! - `pack` (pure): matvec formats, repacking, row-aligned splitting, reference. `matvec`: the f32/F16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K
//!   kernels (G4, G5, G5b). `mvbench`: `--gpu-matvec` throughput. `ops` (G6): RMSNorm, LayerNorm, RoPE, SiLU/GELU, softmax, add, embedding gather.
//!   `attention` (G7): the f16 KV cache in video memory, its append kernel and the decode attention kernel (GQA, sliding window).
//!   `model` (G8): a whole decoder model on the GPU, one token per submission (`GpuModel`).
//!   `gemm` (G9): the batched matrix product for prefill, one kernel for all weight formats. `batch` (G9): the batched RoPE, bias, KV
//!   append and causal attention.
//! - this file: the old `nvidia-smi` detection (kept as the source of VRAM size, which wgpu does not
//!   report), the merge of both sources for `/api/system`, and the `--gpu-info` CLI entry point.

#[cfg(feature = "gpu")]
pub mod attention;
pub mod bench;
#[cfg(feature = "gpu")]
pub mod batch;
pub mod choose;
#[cfg(feature = "gpu")]
pub mod gemm;
#[cfg(feature = "gpu")]
pub mod matvec;
#[cfg(feature = "gpu")]
pub mod model;
pub mod mvbench;
#[cfg(feature = "gpu")]
pub mod ops;
pub mod pack;
#[cfg(feature = "gpu")]
pub mod device;
pub mod discovery;
#[cfg(feature = "gpu")]
pub mod kernels;
pub mod select;

use discovery::Discovery;
use serde::Serialize;
use std::time::Duration;
use tokio::process::Command;

/// CLI flag handled in `main` before the server starts (like `--pdf-extract`).
pub const GPU_INFO_ARG: &str = "--gpu-info";

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: Option<u64>,
    /// Suggested acceleration backend(s): `CUDA / Vulkan` for nvidia-smi only, otherwise the wgpu backends found.
    pub hint: String,
}

/// `buriti --gpu-info [--json]`: prints the adapters and returns the exit code (0 even with no GPU).
pub fn cli_main() -> i32 {
    let json = std::env::args_os().any(|a| a == "--json");
    let d = discovery::discover();
    if json {
        println!("{}", discovery::format_json(&d));
    } else {
        print!("{}", discovery::format_table(&d));
    }
    0
}

/// `buriti --gpu-probe <selector>` (task G10, D69): opens the adapter in this throw-away process and reports what it opened. The server runs it
/// BEFORE opening a GPU in its own process: a driver that crashes in `request_device` (the Intel HD 620 in Vulkan does, D64) then kills only
/// this child, never the server. Prints one JSON line; the exit code is 0 only when the device opened and passed a health check.
pub const GPU_PROBE_ARG: &str = "--gpu-probe";

#[derive(Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct ProbeResult {
    pub ok: bool,
    pub adapter: String,
    pub backend: String,
    pub error: Option<String>,
}

pub fn probe_main() -> i32 {
    let selector = std::env::args().nth(2).unwrap_or_else(|| "best".into());
    let r = probe_here(&selector);
    println!("{}", serde_json::to_string(&r).unwrap_or_default());
    if r.ok {
        0
    } else {
        1
    }
}

#[cfg(feature = "gpu")]
fn probe_here(selector: &str) -> ProbeResult {
    match device::GpuDevice::open(&select::AdapterSelector::parse(selector)) {
        Ok(d) => match d.check_health() {
            Ok(()) => ProbeResult { ok: true, adapter: d.name().to_string(), backend: d.report.backend.clone(), error: None },
            Err(e) => ProbeResult { ok: false, adapter: d.name().to_string(), backend: d.report.backend.clone(), error: Some(e) },
        },
        Err(e) => ProbeResult { ok: false, adapter: String::new(), backend: String::new(), error: Some(e) },
    }
}

#[cfg(not(feature = "gpu"))]
fn probe_here(_selector: &str) -> ProbeResult {
    ProbeResult { ok: false, adapter: String::new(), backend: String::new(), error: Some("this build has no GPU support".into()) }
}

/// Runs the probe in a child process (the binary itself) with a time limit. `Ok` carries the adapter that opened; `Err` carries why it did not
/// (including a crash of the child, which is the point of the isolation).
pub async fn probe_isolated(selector: &str, limit: Duration) -> Result<ProbeResult, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate the executable: {e}"))?;
    let mut cmd = Command::new(exe);
    cmd.args([GPU_PROBE_ARG, selector]).stdin(std::process::Stdio::null()).kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    match tokio::time::timeout(limit, cmd.output()).await {
        Err(_) => Err("the GPU check timed out".into()),
        Ok(Err(e)) => Err(format!("could not start the GPU check: {e}")),
        Ok(Ok(out)) => match serde_json::from_slice::<ProbeResult>(&out.stdout) {
            Ok(r) if r.ok => Ok(r),
            Ok(r) => Err(r.error.unwrap_or_else(|| "the GPU did not open".into())),
            Err(_) => Err(format!("the GPU check ended abnormally (exit code {:?}): the driver may have crashed", out.status.code())),
        },
    }
}

/// Best-effort detection through `nvidia-smi` (name and VRAM). Failure = empty list.
pub async fn detect() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // NVIDIA: nvidia-smi (Windows/Linux)
    let mut cmd = Command::new("nvidia-smi");
    cmd.args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    if let Ok(Ok(out)) = tokio::time::timeout(Duration::from_secs(3), cmd.output()).await {
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let mut parts = line.split(',');
                let name = parts.next().unwrap_or("").trim().to_string();
                let vram = parts.next().and_then(|v| v.trim().parse::<u64>().ok());
                if !name.is_empty() {
                    gpus.push(GpuInfo { name, vram_mb: vram, hint: "CUDA / Vulkan".into() });
                }
            }
        }
    }

    // Apple Silicon: Metal already comes with the default llama.cpp build.
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        gpus.push(GpuInfo { name: "Apple Silicon".into(), vram_mb: None, hint: "Metal".into() });
    }

    gpus
}

/// Runs the discovery in a child process (the binary itself with `--gpu-info --json`) and caches the result.
/// Loading GPU drivers can crash or hang; like the PDF reader (D55) this must not take the server down.
pub async fn discover_isolated() -> &'static Discovery {
    static CACHE: tokio::sync::OnceCell<Discovery> = tokio::sync::OnceCell::const_new();
    CACHE.get_or_init(run_child).await
}

async fn run_child() -> Discovery {
    let fail = |why: String| Discovery { adapters: vec![], reason: Some(why) };
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return fail(format!("cannot locate the executable: {e}")),
    };
    let mut cmd = Command::new(exe);
    cmd.args([GPU_INFO_ARG, "--json"]).stdin(std::process::Stdio::null()).kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    match tokio::time::timeout(Duration::from_secs(20), cmd.output()).await {
        Err(_) => fail("GPU discovery timed out".into()),
        Ok(Err(e)) => fail(format!("could not start GPU discovery: {e}")),
        Ok(Ok(out)) if !out.status.success() => fail("GPU discovery process failed (driver crash?)".into()),
        Ok(Ok(out)) => serde_json::from_slice::<Discovery>(&out.stdout).unwrap_or_else(|_| fail("GPU discovery returned unreadable output".into())),
    }
}

fn norm(s: &str) -> String {
    s.to_ascii_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn same_gpu(a: &str, b: &str) -> bool {
    let (a, b) = (norm(a), norm(b));
    !a.is_empty() && !b.is_empty() && (a == b || a.contains(&b) || b.contains(&a))
}

/// Builds the legacy `gpus` list (name, vram_mb, hint) from both sources, without duplicates: one entry per
/// physical GPU (the same GPU shows up once per backend in wgpu), VRAM taken from `nvidia-smi` when the names match.
pub fn merge(legacy: Vec<GpuInfo>, d: &Discovery) -> Vec<GpuInfo> {
    let real: Vec<_> = d.adapters.iter().filter(|a| !a.software).collect();
    if real.is_empty() {
        return legacy;
    }
    let mut out: Vec<GpuInfo> = Vec::new();
    let mut used = vec![false; legacy.len()];
    for a in real {
        if let Some(e) = out.iter_mut().find(|e| same_gpu(&e.name, &a.name)) {
            if !e.hint.split(" / ").any(|h| h == a.backend) {
                e.hint = format!("{} / {}", e.hint, a.backend);
            }
            continue;
        }
        let mut vram = None;
        for (i, l) in legacy.iter().enumerate() {
            if same_gpu(&l.name, &a.name) {
                used[i] = true;
                vram = vram.or(l.vram_mb);
            }
        }
        out.push(GpuInfo { name: a.name.clone(), vram_mb: vram, hint: a.backend.clone() });
    }
    // Legacy entries that wgpu did not see (for example CUDA-only drivers). The Metal placeholder is redundant once wgpu sees it.
    for (i, l) in legacy.into_iter().enumerate() {
        if !used[i] && l.hint != "Metal" {
            out.push(l);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::discovery::*;
    use super::*;

    fn adapter(name: &str, backend: &str, software: bool) -> AdapterReport {
        AdapterReport {
            name: name.into(),
            vendor_id: 0,
            vendor: "x".into(),
            device_id: 0,
            device_type: "discrete".into(),
            backend: backend.into(),
            driver: String::new(),
            driver_info: String::new(),
            software,
            limits: LimitsReport {
                max_buffer_size: 1,
                max_storage_buffer_binding_size: 1,
                max_compute_workgroup_size_x: 1,
                max_compute_workgroup_size_y: 1,
                max_compute_workgroup_size_z: 1,
                max_compute_invocations_per_workgroup: 1,
                max_compute_workgroups_per_dimension: 1,
                max_compute_workgroup_storage_size: 1,
                min_subgroup_size: 0,
                max_subgroup_size: 0,
            },
            features: FeaturesReport { shader_f16: false, subgroup: false },
        }
    }

    fn smi(name: &str, vram: u64) -> GpuInfo {
        GpuInfo { name: name.into(), vram_mb: Some(vram), hint: "CUDA / Vulkan".into() }
    }

    #[test]
    fn merge_without_adapters_keeps_legacy() {
        let d = Discovery { adapters: vec![], reason: Some("none".into()) };
        let m = merge(vec![smi("NVIDIA GeForce 940MX", 4096)], &d);
        assert_eq!(m, vec![smi("NVIDIA GeForce 940MX", 4096)]);
    }

    #[test]
    fn merge_dedups_backends_and_takes_vram() {
        let d = Discovery {
            adapters: vec![
                adapter("NVIDIA GeForce 940MX", "Vulkan", false),
                adapter("Intel(R) HD Graphics 620", "Vulkan", false),
                adapter("NVIDIA GeForce 940MX", "DX12", false),
                adapter("Microsoft Basic Render Driver", "DX12", true),
            ],
            reason: None,
        };
        let m = merge(vec![smi("NVIDIA GeForce 940MX", 4096)], &d);
        assert_eq!(m.len(), 2, "{m:?}");
        assert_eq!(m[0].name, "NVIDIA GeForce 940MX");
        assert_eq!(m[0].vram_mb, Some(4096));
        assert_eq!(m[0].hint, "Vulkan / DX12");
        assert_eq!(m[1].vram_mb, None);
    }

    #[test]
    fn merge_keeps_unmatched_legacy_but_drops_metal_placeholder() {
        let d = Discovery { adapters: vec![adapter("Apple M1", "Metal", false)], reason: None };
        let metal = GpuInfo { name: "Apple Silicon".into(), vram_mb: None, hint: "Metal".into() };
        let m = merge(vec![smi("Tesla T4", 15360), metal], &d);
        assert_eq!(m.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(), vec!["Apple M1", "Tesla T4"]);
    }

    #[test]
    fn software_only_falls_back_to_legacy() {
        let d = Discovery { adapters: vec![adapter("llvmpipe", "Vulkan", true)], reason: None };
        assert!(merge(vec![], &d).is_empty());
    }
}
