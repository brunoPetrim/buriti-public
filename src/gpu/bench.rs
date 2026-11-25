//! Memory bandwidth microbenchmark (task G3, D63): `buriti --gpu-bench [--json] [--adapter <sel>] [--sizes 64,256,512] [--model-mib N]`.
//!
//! Token generation reads every weight once per token, so `bandwidth / bytes of weights` is an upper bound
//! for tokens per second. This module measures that bandwidth on each GPU adapter (a read kernel and a copy
//! kernel streaming a big storage buffer) and, for comparison, on the CPU (one pass over the same number of
//! bytes with 1, 2 and 4 threads).
//!
//! Process model: like the discovery behind `/api/system` (D62), the adapter list and every adapter run in a
//! child process (this binary again), each with a time limit, so a hanging or crashing driver costs one
//! adapter, not the run. The parent always exits 0 (like `--gpu-info`).
//!
//! Conventions: GB/s = 1e9 bytes per second. "read" counts the bytes read; "copy" counts bytes read plus
//! bytes written (2 x size), the way memory traffic is usually quoted. GPU time comes from timestamp queries
//! when the adapter has TIMESTAMP_QUERY, otherwise from the wall clock around `queue.submit` + poll.

use super::discovery::Discovery;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const GPU_BENCH_ARG: &str = "--gpu-bench";
pub const GPU_BENCH_CHILD_ARG: &str = "--gpu-bench-child";

pub const DEFAULT_SIZES_MIB: [u64; 3] = [64, 256, 512];
const PASSES: usize = 5;
/// Per-adapter limit in the parent (the child also has per-job timeouts inside).
const CHILD_TIMEOUT: Duration = Duration::from_secs(240);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BwPoint {
    /// `read` or `copy`.
    pub kernel: String,
    pub size_mib: u64,
    /// Median over the timed passes (the metric chosen by `timing`).
    pub gbps_median: f64,
    pub gbps_min: f64,
    pub gbps_max: f64,
    /// Same median computed from the wall clock (submit to completion); equals `gbps_median` without timestamps.
    pub wall_gbps_median: f64,
    pub passes: u32,
    /// Dispatches inside one timed pass.
    pub iters: u32,
    /// `timestamp` or `wall`.
    pub timing: String,
    /// The kernel output matched the CPU expectation (so the loads were really executed).
    pub verified: bool,
    /// Why this size was not run (does not fit the adapter limits, ...).
    pub skipped: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdapterBench {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub software: bool,
    pub timestamp_query: bool,
    pub points: Vec<BwPoint>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuPoint {
    pub threads: usize,
    pub size_mib: u64,
    pub gbps_median: f64,
    pub gbps_min: f64,
    pub gbps_max: f64,
    pub passes: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchReport {
    pub adapters: Vec<AdapterBench>,
    pub cpu: Vec<CpuPoint>,
    pub notes: Vec<String>,
}

// ------------------------------------------------------------------ pure helpers

pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

pub fn gbps(bytes: f64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        bytes / seconds / 1e9
    } else {
        0.0
    }
}

/// Upper bound for generation speed: every token reads all the weights once.
pub fn ceiling_tok_s(gbps: f64, model_bytes: f64) -> f64 {
    if model_bytes > 0.0 {
        gbps * 1e9 / model_bytes
    } else {
        0.0
    }
}

pub fn parse_sizes(s: &str) -> Result<Vec<u64>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let v: u64 = part.trim().parse().map_err(|_| format!("'{part}' is not a size in MiB"))?;
        if v == 0 || v > 4096 {
            return Err(format!("size {v} MiB is outside 1..4096"));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err("no sizes".into());
    }
    Ok(out)
}

/// Value of element `j` of the test pattern: `j % 7`.
fn pattern_u32(j: u64) -> u32 {
    (j % 7) as u32
}

/// Sum of the pattern over `n` elements, in closed form.
pub fn pattern_sum(n: u64) -> u64 {
    (n / 7) * 21 + (0..(n % 7)).sum::<u64>()
}

fn mib(n: u64) -> u64 {
    n << 20
}

// ------------------------------------------------------------------ CPU side

fn sum_u32_1t(data: &[u32]) -> u32 {
    let mut acc = [0u32; 8];
    let mut it = data.chunks_exact(8);
    for c in &mut it {
        for k in 0..8 {
            acc[k] = acc[k].wrapping_add(c[k]);
        }
    }
    let mut t = acc.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    for &v in it.remainder() {
        t = t.wrapping_add(v);
    }
    t
}

/// One-pass read bandwidth of `size_mib` MiB with `threads` threads: median, min and max GB/s over `PASSES` passes.
pub fn cpu_read_bandwidth(size_mib: u64, threads: usize) -> Result<CpuPoint, String> {
    let n = (mib(size_mib) / 4) as usize;
    let data: Vec<u32> = (0..n as u64).map(pattern_u32).collect();
    let expect = (pattern_sum(n as u64) & 0xFFFF_FFFF) as u32;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().map_err(|e| format!("thread pool: {e}"))?;
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let one_pass = |d: &[u32]| -> u32 {
        if threads <= 1 {
            sum_u32_1t(d)
        } else {
            use rayon::prelude::*;
            pool.install(|| d.par_chunks(chunk).map(sum_u32_1t).reduce(|| 0u32, |a, b| a.wrapping_add(b)))
        }
    };
    let mut rates = Vec::new();
    for pass in 0..=PASSES {
        let t = Instant::now();
        let s = std::hint::black_box(one_pass(std::hint::black_box(&data)));
        let dt = t.elapsed().as_secs_f64();
        if s != expect {
            return Err(format!("CPU read check failed: {s} instead of {expect}"));
        }
        if pass > 0 {
            rates.push(gbps(mib(size_mib) as f64, dt)); // pass 0 is the warm-up (page faults, caches)
        }
    }
    Ok(CpuPoint {
        threads,
        size_mib,
        gbps_median: median(&rates),
        gbps_min: rates.iter().cloned().fold(f64::MAX, f64::min),
        gbps_max: rates.iter().cloned().fold(0.0, f64::max),
        passes: PASSES as u32,
    })
}

// ------------------------------------------------------------------ GPU side

#[cfg(feature = "gpu")]
pub(crate) mod gpu_side {
    use super::*;
    use crate::gpu::device::{GpuBuffer, GpuDevice, DEFAULT_TIMEOUT};
    use crate::gpu::kernels::grid_2d;
    use crate::gpu::select::AdapterSelector;

    pub const READ_WGSL: &str = include_str!("shaders/bw_read.wgsl");
    pub const COPY_WGSL: &str = include_str!("shaders/bw_copy.wgsl");
    const CHUNK: u64 = 16 << 20;

    fn pattern_bytes(first: u64, count: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(count as usize * 4);
        for j in first..first + count {
            v.extend_from_slice(&(pattern_u32(j) as f32).to_le_bytes());
        }
        v
    }

    fn fill_pattern(dev: &GpuDevice, buf: &GpuBuffer) -> Result<(), String> {
        let total = buf.len / 4;
        let per = CHUNK / 4;
        let mut at = 0;
        while at < total {
            let n = per.min(total - at);
            dev.upload_bytes(buf, at * 4, &pattern_bytes(at, n))?;
            at += n;
        }
        Ok(())
    }

    struct Measured {
        gbps: Vec<f64>,
        wall: Vec<f64>,
        iters: u32,
        timing: &'static str,
    }

    /// Calibrates the number of dispatches per pass (about 150 ms, always below the driver watchdog), then runs `PASSES` passes.
    fn measure(dev: &GpuDevice, pipeline: &wgpu::ComputePipeline, bufs: &[&GpuBuffer], groups: [u32; 3], bytes_per_dispatch: f64) -> Result<Measured, String> {
        let warm = dev.run(pipeline, bufs, groups, 1, DEFAULT_TIMEOUT)?;
        let t1 = warm.gpu_seconds.unwrap_or(warm.wall_seconds).max(1e-6);
        let iters = ((0.15 / t1).ceil() as u32).clamp(1, 64);
        let (mut g, mut w) = (Vec::new(), Vec::new());
        let mut used_ts = true;
        for _ in 0..PASSES {
            let t = dev.run(pipeline, bufs, groups, iters, DEFAULT_TIMEOUT)?;
            let total = bytes_per_dispatch * iters as f64;
            w.push(gbps(total, t.wall_seconds));
            match t.gpu_seconds {
                Some(s) => g.push(gbps(total, s)),
                None => {
                    used_ts = false;
                    g.push(gbps(total, t.wall_seconds));
                }
            }
        }
        Ok(Measured { gbps: g, wall: w, iters, timing: if used_ts { "timestamp" } else { "wall" } })
    }

    fn point(kernel: &str, size: u64, m: Measured, verified: bool) -> BwPoint {
        BwPoint {
            kernel: kernel.into(),
            size_mib: size,
            gbps_median: median(&m.gbps),
            gbps_min: m.gbps.iter().cloned().fold(f64::MAX, f64::min),
            gbps_max: m.gbps.iter().cloned().fold(0.0, f64::max),
            wall_gbps_median: median(&m.wall),
            passes: PASSES as u32,
            iters: m.iters,
            timing: m.timing.into(),
            verified,
            skipped: None,
        }
    }

    fn skipped(kernel: &str, size: u64, why: String) -> BwPoint {
        BwPoint {
            kernel: kernel.into(),
            size_mib: size,
            gbps_median: 0.0,
            gbps_min: 0.0,
            gbps_max: 0.0,
            wall_gbps_median: 0.0,
            passes: 0,
            iters: 0,
            timing: String::new(),
            verified: false,
            skipped: Some(why),
        }
    }

    fn bench_read(dev: &GpuDevice, size: u64) -> Result<BwPoint, String> {
        let bytes = mib(size);
        let src = dev.create_storage(bytes, "bw.src")?;
        fill_pattern(dev, &src)?;
        let n_vec4 = bytes / 16;
        let max = dev.caps.max_compute_workgroups_per_dimension;
        let groups = (n_vec4 / (256 * 16)).clamp(1, 4096).min(max as u64) as u32;
        let sink = dev.create_storage(groups as u64 * 256 * 4, "bw.sink")?;
        let pipeline = dev.pipeline("bw_read", READ_WGSL, "main")?;
        let m = measure(dev, &pipeline, &[&src, &sink], [groups, 1, 1], bytes as f64)?;
        // The sink holds one sum per thread: together they must add up to the whole pattern.
        let sums = dev.read_back_f32(&sink, groups as usize * 256, DEFAULT_TIMEOUT)?;
        let got: f64 = sums.iter().map(|&s| s as f64).sum();
        let want = pattern_sum(bytes / 4) as f64;
        let verified = got == want;
        if !verified {
            return Err(format!("read kernel output {got} does not match the expected {want}: the loads were not performed as intended"));
        }
        Ok(point("read", size, m, verified))
    }

    fn bench_copy(dev: &GpuDevice, size: u64) -> Result<BwPoint, String> {
        let bytes = mib(size);
        let src = dev.create_storage(bytes, "bw.copy.src")?;
        let dst = dev.create_storage(bytes, "bw.copy.dst")?;
        fill_pattern(dev, &src)?;
        let n_vec4 = bytes / 16;
        let groups = (n_vec4 / (256 * 16)).clamp(1, 4096).min(dev.caps.max_compute_workgroups_per_dimension as u64) as u32;
        let _ = grid_2d(groups as u64, dev.caps.max_compute_workgroups_per_dimension)?;
        let pipeline = dev.pipeline("bw_copy", COPY_WGSL, "main")?;
        let m = measure(dev, &pipeline, &[&src, &dst], [groups, 1, 1], 2.0 * bytes as f64)?;
        // Check the first and last 64 KiB of the destination.
        let edge = 64 << 10;
        let head = dev.read_back(&dst, 0, edge, DEFAULT_TIMEOUT)?;
        let tail = dev.read_back(&dst, bytes - edge, edge, DEFAULT_TIMEOUT)?;
        let verified = head == pattern_bytes(0, edge / 4) && tail == pattern_bytes((bytes - edge) / 4, edge / 4);
        if !verified {
            return Err("copy kernel output does not match the source pattern".into());
        }
        Ok(point("copy", size, m, verified))
    }

    /// Median read bandwidth (GB/s) of a `size_mib` MiB buffer: the ceiling the matvec kernels are compared with.
    pub fn read_ceiling(dev: &GpuDevice, size_mib: u64) -> Result<f64, String> {
        bench_read(dev, size_mib).map(|p| p.gbps_median)
    }

    /// Runs the benchmark on the adapter at `index` of the discovery list (checking that its name is `expect_name`).
    pub fn bench_adapter(index: usize, expect_name: &str, sizes: &[u64]) -> AdapterBench {
        let mut out = AdapterBench::default();
        let dev = match GpuDevice::open(&AdapterSelector::Index(index)) {
            Ok(d) => d,
            Err(e) => {
                out.error = Some(e);
                return out;
            }
        };
        out.name = dev.report.name.clone();
        out.backend = dev.report.backend.clone();
        out.device_type = dev.report.device_type.clone();
        out.software = dev.report.software;
        out.timestamp_query = dev.caps.timestamp_query;
        if !expect_name.is_empty() && out.name != expect_name {
            out.error = Some(format!("adapter {index} is '{}' now, expected '{expect_name}' (the adapter order changed)", out.name));
            return out;
        }
        for &size in sizes {
            let bytes = mib(size);
            let fits = bytes <= dev.caps.max_buffer_size && bytes <= dev.caps.max_storage_buffer_binding_size;
            if !fits {
                let why = format!("{size} MiB does not fit max_buffer_size / max_storage_buffer_binding_size ({} / {} MiB)", dev.caps.max_buffer_size >> 20, dev.caps.max_storage_buffer_binding_size >> 20);
                out.points.push(skipped("read", size, why.clone()));
                out.points.push(skipped("copy", size, why));
                continue;
            }
            for kernel in ["read", "copy"] {
                let r = if kernel == "read" { bench_read(&dev, size) } else { bench_copy(&dev, size) };
                match r {
                    Ok(p) => out.points.push(p),
                    Err(e) => {
                        let fatal = dev.check_health().is_err() || e.contains("did not finish") || e.contains("hang");
                        out.points.push(skipped(kernel, size, e));
                        if fatal {
                            out.error = Some("the device stopped answering; the remaining sizes were not run".into());
                            return out;
                        }
                    }
                }
            }
            dev.clear_pool();
        }
        out
    }
}

#[cfg(feature = "gpu")]
pub use gpu_side::bench_adapter;

#[cfg(not(feature = "gpu"))]
pub fn bench_adapter(_index: usize, _expect_name: &str, _sizes: &[u64]) -> AdapterBench {
    AdapterBench { error: Some("this build was compiled without the `gpu` feature".into()), ..Default::default() }
}

// ------------------------------------------------------------------ child process plumbing

/// Runs this binary again with `args`, returns its stdout. Kills the child (only this one) at `limit`.
pub(super) fn run_self(args: &[String], limit: Duration) -> Result<Vec<u8>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate the executable: {e}"))?;
    let mut child = Command::new(exe).args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().map_err(|e| format!("could not start the child process: {e}"))?;
    let mut stdout = child.stdout.take().ok_or("no stdout")?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if start.elapsed() > limit {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = reader.join();
                    return Err(format!("timed out after {} s (driver hang?)", limit.as_secs()));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("waiting for the child failed: {e}")),
        }
    };
    let out = reader.join().unwrap_or_default();
    if !status.success() {
        return Err(format!("the child process failed ({status}); driver crash?"));
    }
    Ok(out)
}

pub(super) fn arg_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// `--gpu-bench-child <index> <name> <sizes>`: runs one adapter and prints its `AdapterBench` as JSON.
pub fn child_main() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let pos = args.iter().position(|a| a == GPU_BENCH_CHILD_ARG).unwrap_or(0);
    let index: usize = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let name = args.get(pos + 2).cloned().unwrap_or_default();
    let sizes = args.get(pos + 3).and_then(|s| parse_sizes(s).ok()).unwrap_or_else(|| DEFAULT_SIZES_MIB.to_vec());
    let r = std::panic::catch_unwind(|| bench_adapter(index, &name, &sizes)).unwrap_or_else(|_| AdapterBench { error: Some("the GPU library panicked".into()), ..Default::default() });
    println!("{}", serde_json::to_string(&r).unwrap_or_else(|_| "{}".into()));
    0
}

/// Parent side. Exit code is always 0.
pub fn cli_main() -> i32 {
    let json = std::env::args_os().any(|a| a == "--json");
    let sizes = match arg_value("--sizes") {
        Some(s) => match parse_sizes(&s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("--sizes: {e}");
                return 0;
            }
        },
        None => DEFAULT_SIZES_MIB.to_vec(),
    };
    let model_mib = arg_value("--model-mib").and_then(|s| s.parse::<f64>().ok());
    let only = arg_value("--adapter");
    let has = |f: &str| std::env::args_os().any(|a| a == f);
    let (no_cpu, no_gpu) = (has("--no-cpu"), has("--no-gpu"));
    let mut report = BenchReport::default();

    // 1. Adapter list, in a child process.
    let disc: Discovery = match run_self(&["--gpu-info".into(), "--json".into()], Duration::from_secs(30)).and_then(|b| serde_json::from_slice(&b).map_err(|e| format!("unreadable discovery output: {e}"))) {
        Ok(d) => d,
        Err(e) => {
            report.notes.push(format!("GPU discovery failed: {e}"));
            Discovery::default()
        }
    };
    let list: Vec<(usize, &super::discovery::AdapterReport)> = if no_gpu { vec![] } else { match &only {
        None => disc.adapters.iter().enumerate().collect(),
        Some(sel) => {
            let sel = super::select::AdapterSelector::parse(sel);
            match super::select::select_adapter(&disc.adapters, &sel) {
                Ok(i) => vec![(i, &disc.adapters[i])],
                Err(e) => {
                    report.notes.push(format!("adapter selection: {e}"));
                    vec![]
                }
            }
        }
    } };
    if disc.adapters.is_empty() {
        if let Some(r) = &disc.reason {
            report.notes.push(format!("no GPU adapter: {r}"));
        }
    }

    // 2. One child per adapter.
    let sizes_arg = sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
    for (i, a) in list {
        let args = vec![GPU_BENCH_CHILD_ARG.to_string(), i.to_string(), a.name.clone(), sizes_arg.clone()];
        let res = run_self(&args, CHILD_TIMEOUT).and_then(|b| serde_json::from_slice::<AdapterBench>(&b).map_err(|e| format!("unreadable result: {e}")));
        report.adapters.push(match res {
            Ok(b) => b,
            Err(e) => AdapterBench { name: a.name.clone(), backend: a.backend.clone(), device_type: a.device_type.clone(), software: a.software, error: Some(e), ..Default::default() },
        });
    }

    // 3. CPU, same sizes, in this process (no driver involved).
    let threads = [1usize, 2, 4];
    for &s in &sizes {
        if no_cpu {
            break;
        }
        for &t in &threads {
            match cpu_read_bandwidth(s, t) {
                Ok(p) => report.cpu.push(p),
                Err(e) => report.notes.push(format!("CPU {s} MiB x {t} thread(s): {e}")),
            }
        }
    }
    report.notes.push("GB/s = 1e9 bytes/s. read = bytes read / time; copy = (bytes read + bytes written) / time. Median and range over the timed passes of one run.".into());

    if json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into()));
    } else {
        print!("{}", format_report(&report, model_mib));
    }
    0
}

pub fn format_report(r: &BenchReport, model_mib: Option<f64>) -> String {
    let mut s = String::new();
    s.push_str("Memory bandwidth (GB/s = 1e9 bytes/s)\n\n");
    let ceil_hdr = model_mib.map(|m| format!("  tok/s ceiling ({m} MiB model)")).unwrap_or_default();
    s.push_str(&format!("{:<34} {:<7} {:<5} {:>8}  {:>20}  {:<10} {:>10}{ceil_hdr}\n", "adapter", "backend", "kernel", "size MiB", "GB/s median (min-max)", "timing", "wall GB/s"));
    for a in &r.adapters {
        if a.points.is_empty() {
            s.push_str(&format!("{:<34} {:<7} no result: {}\n", a.name, a.backend, a.error.as_deref().unwrap_or("?")));
            continue;
        }
        for p in &a.points {
            if let Some(why) = &p.skipped {
                s.push_str(&format!("{:<34} {:<7} {:<5} {:>8}  skipped: {why}\n", a.name, a.backend, p.kernel, p.size_mib));
                continue;
            }
            let ceil = match (model_mib, p.kernel.as_str()) {
                (Some(m), "read") => format!("  {:>8.1}", ceiling_tok_s(p.gbps_median, m * 1048576.0)),
                _ => String::new(),
            };
            s.push_str(&format!(
                "{:<34} {:<7} {:<5} {:>8}  {:>7.2} ({:>5.2}-{:>5.2})  {:<10} {:>10.2}{ceil}\n",
                a.name, a.backend, p.kernel, p.size_mib, p.gbps_median, p.gbps_min, p.gbps_max, p.timing, p.wall_gbps_median
            ));
        }
        if let Some(e) = &a.error {
            s.push_str(&format!("    note: {e}\n"));
        }
    }
    for c in &r.cpu {
        let ceil = model_mib.map(|m| format!("  {:>8.1}", ceiling_tok_s(c.gbps_median, m * 1048576.0))).unwrap_or_default();
        s.push_str(&format!(
            "{:<34} {:<7} {:<5} {:>8}  {:>7.2} ({:>5.2}-{:>5.2})  {:<10} {:>10}{ceil}\n",
            format!("CPU ({} thread{})", c.threads, if c.threads == 1 { "" } else { "s" }),
            "-",
            "read",
            c.size_mib,
            c.gbps_median,
            c.gbps_min,
            c.gbps_max,
            "wall",
            "-"
        ));
    }
    for n in &r.notes {
        s.push_str(&format!("\n{n}"));
    }
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_rates() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
        assert_eq!(gbps(2e9, 1.0), 2.0);
        assert_eq!(gbps(1.0, 0.0), 0.0);
        // 462 MiB model at 14.4 GB/s: about 29.7 tokens per second.
        let c = ceiling_tok_s(14.4, 462.0 * 1048576.0);
        assert!((c - 29.72).abs() < 0.05, "{c}");
    }

    #[test]
    fn pattern_sum_matches_brute_force() {
        for n in [0u64, 1, 6, 7, 8, 13, 14, 100, 1001] {
            let brute: u64 = (0..n).map(|j| pattern_u32(j) as u64).sum();
            assert_eq!(pattern_sum(n), brute, "n={n}");
        }
    }

    #[test]
    fn sizes_parse_and_reject() {
        assert_eq!(parse_sizes("64, 256,512").unwrap(), vec![64, 256, 512]);
        assert!(parse_sizes("0").is_err());
        assert!(parse_sizes("abc").is_err());
        assert!(parse_sizes("99999").is_err());
    }

    #[test]
    fn cpu_bandwidth_runs_and_checks() {
        let p = cpu_read_bandwidth(8, 1).unwrap();
        assert!(p.gbps_median > 0.0 && p.gbps_min <= p.gbps_median && p.gbps_median <= p.gbps_max);
        let p2 = cpu_read_bandwidth(8, 2).unwrap();
        assert!(p2.gbps_median > 0.0);
    }

    #[test]
    fn single_thread_sum_has_teeth() {
        // 13 elements: the SIMD-friendly loop and the remainder must both be counted.
        let v: Vec<u32> = (0..13).map(pattern_u32).collect();
        assert_eq!(sum_u32_1t(&v) as u64, pattern_sum(13));
        assert_ne!(sum_u32_1t(&v[..12]) as u64, pattern_sum(13));
    }

    #[test]
    fn report_table_and_json() {
        let r = BenchReport {
            adapters: vec![AdapterBench {
                name: "Test GPU".into(),
                backend: "Vulkan".into(),
                device_type: "discrete".into(),
                points: vec![BwPoint {
                    kernel: "read".into(),
                    size_mib: 64,
                    gbps_median: 14.4,
                    gbps_min: 14.0,
                    gbps_max: 14.6,
                    wall_gbps_median: 13.9,
                    passes: 5,
                    iters: 3,
                    timing: "timestamp".into(),
                    verified: true,
                    skipped: None,
                }],
                ..Default::default()
            }],
            cpu: vec![CpuPoint { threads: 2, size_mib: 64, gbps_median: 12.0, gbps_min: 11.0, gbps_max: 12.5, passes: 5 }],
            notes: vec![],
        };
        let t = format_report(&r, Some(462.0));
        assert!(t.is_ascii());
        assert!(t.contains("Test GPU") && t.contains("14.40") && t.contains("CPU (2 threads)") && t.contains("462 MiB model"));
        let j: serde_json::Value = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(j["adapters"][0]["points"][0]["kernel"], "read");
    }

    /// Software adapter, small sizes: checks that the kernels run and verify. Skips when no adapter can be created.
    #[cfg(feature = "gpu")]
    #[test]
    fn software_adapter_bandwidth_smoke() {
        let d = crate::gpu::discovery::discover();
        let Some((i, a)) = d.adapters.iter().enumerate().find(|(_, a)| a.software) else {
            println!("SKIP: no software adapter on this machine");
            return;
        };
        let b = bench_adapter(i, &a.name, &[8]);
        if let Some(e) = &b.error {
            println!("SKIP: software adapter could not run the benchmark: {e}");
            return;
        }
        assert_eq!(b.points.len(), 2, "{b:?}");
        for p in &b.points {
            assert!(p.skipped.is_none(), "{p:?}");
            assert!(p.verified && p.gbps_median > 0.0, "{p:?}");
        }
    }

    /// A kernel that skips most of the buffer must fail the verification (teeth for the read check).
    #[cfg(feature = "gpu")]
    #[test]
    fn read_check_rejects_a_kernel_that_skips_data() {
        use crate::gpu::device::GpuDevice;
        use crate::gpu::select::AdapterSelector;
        let Ok(dev) = GpuDevice::open(&AdapterSelector::Software) else {
            println!("SKIP: no software adapter");
            return;
        };
        let bytes = mib(4);
        let src = dev.create_storage(bytes, "t.src").unwrap();
        let mut at = 0;
        while at < bytes / 4 {
            let n = (1u64 << 20).min(bytes / 4 - at);
            let chunk: Vec<u8> = (at..at + n).flat_map(|j| (pattern_u32(j) as f32).to_le_bytes()).collect();
            dev.upload_bytes(&src, at * 4, &chunk).unwrap();
            at += n;
        }
        let groups = 16u32;
        let sink = dev.create_storage(groups as u64 * 256 * 4, "t.sink").unwrap();
        let sum = |wgsl: &str, key: &str| -> f64 {
            let p = dev.pipeline(key, wgsl, "main").unwrap();
            dev.run(&p, &[&src, &sink], [groups, 1, 1], 1, std::time::Duration::from_secs(30)).unwrap();
            dev.read_back_f32(&sink, groups as usize * 256, std::time::Duration::from_secs(30)).unwrap().iter().map(|&x| x as f64).sum()
        };
        let want = pattern_sum(bytes / 4) as f64;
        assert_eq!(sum(super::gpu_side::READ_WGSL, "t_read"), want);
        let buggy = super::gpu_side::READ_WGSL.replace("i += total;", "i += total * 2u;");
        assert_ne!(buggy, super::gpu_side::READ_WGSL, "planted-bug marker missing");
        assert_ne!(sum(&buggy, "t_read_bug"), want, "planted stride bug was NOT detected");
    }
}
