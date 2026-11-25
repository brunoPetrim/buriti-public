//! Throughput of the matvec kernels (tasks G4 and G5, D64):
//! `buriti --gpu-matvec [--json] [--adapter <sel>] [--formats f32,f16,q8_0,q4_0,q4_1,q5_0,q5_1,q4_k,q5_k,q6_k] [--shapes 4864x896,...] [--wg 32,64] [--rpg 1,2,4 (0 = default)] [--runs 3] [--rest <s>]`.
//!
//! Same process model as `--gpu-bench` (D63): every run on every adapter is a child process (this binary
//! again, `--gpu-matvec-child`) with a time limit, so a driver crash costs one adapter; the parent always
//! exits 0. Each child measures the adapter's read-bandwidth ceiling first (the G3 kernel on a 256 MiB buffer),
//! then every format x shape: upload, check against the exact float reference on sampled rows, timing.
//! The parent repeats the child `--runs` times and reports the median of the runs with the range.
//!
//! GB/s = 1e9 bytes/s of weight data read, from timestamp queries when available. Small matrices are
//! uploaded several times (about 64 MiB in total) and dispatched round robin, so the timing does not
//! measure the cache of the GPU.

use super::discovery::Discovery;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const GPU_MATVEC_ARG: &str = "--gpu-matvec";
pub const GPU_MATVEC_CHILD_ARG: &str = "--gpu-matvec-child";

const DEFAULT_FORMATS: &str = "f32,f16,q8_0,q4_0,q4_1,q5_0,q5_1,q4_k,q5_k,q6_k";
/// Qwen 2.5 0.5B-like (hidden 896, FFN 4864, vocabulary 151936) and one large matrix.
const DEFAULT_SHAPES: &str = "4864x896,896x4864,151936x896,8192x8192";
const CHILD_TIMEOUT: Duration = Duration::from_secs(420);
/// A software adapter is skipped for matrices above this many weights (it reads ~2 GB/s).
const SOFTWARE_MAX_WEIGHTS: u64 = 8_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvPoint {
    pub format: String,
    pub rows: u64,
    pub cols: u64,
    pub wg: u32,
    pub rpg: u32,
    pub resident_bytes: u64,
    pub native_bytes: u64,
    pub copies: u32,
    pub iters: u32,
    pub gbps_median: f64,
    pub gbps_min: f64,
    pub gbps_max: f64,
    pub wall_gbps_median: f64,
    pub timing: String,
    pub rows_checked: u32,
    pub err_gpu_max: f64,
    pub err_cpu_max: f64,
    pub err_ratio_rms: f64,
    pub skipped: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MvAdapter {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub software: bool,
    pub ceiling_gbps: Option<f64>,
    pub ceiling_mib: u64,
    pub points: Vec<MvPoint>,
    pub error: Option<String>,
}

/// One row of the final table: the same measurement over the runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvRow {
    pub adapter: String,
    pub backend: String,
    pub software: bool,
    pub format: String,
    pub rows: u64,
    pub cols: u64,
    pub wg: u32,
    pub rpg: u32,
    pub resident_mib: f64,
    pub native_mib: f64,
    pub runs: usize,
    pub gbps_median: f64,
    pub gbps_lo: f64,
    pub gbps_hi: f64,
    pub ceiling_gbps: Option<f64>,
    pub fraction_of_ceiling: Option<f64>,
    pub err_gpu_max: f64,
    pub err_cpu_max: f64,
    pub err_ratio_rms: f64,
    pub note: Option<String>,
}

pub fn parse_shapes(s: &str) -> Result<Vec<(u64, u64)>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let (a, b) = part.trim().split_once('x').ok_or_else(|| format!("'{part}' is not ROWSxCOLS"))?;
        let r: u64 = a.parse().map_err(|_| format!("'{a}' is not a number"))?;
        let c: u64 = b.parse().map_err(|_| format!("'{b}' is not a number"))?;
        if r == 0 || c == 0 || r > 1_000_000 || c > 100_000 {
            return Err(format!("shape {r}x{c} is outside 1..1000000 rows and 1..100000 columns"));
        }
        out.push((r, c));
    }
    if out.is_empty() {
        return Err("no shapes".into());
    }
    Ok(out)
}

pub fn parse_formats(s: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let d = super::pack::parse_dtype(part).ok_or_else(|| format!("'{part}' is not a matvec format (f32, f16, q8_0, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k)"))?;
        out.push(super::pack::name(d).to_string());
    }
    Ok(out)
}

pub fn parse_wgs(s: &str) -> Result<Vec<u32>, String> {
    s.split(',').map(|p| p.trim().parse::<u32>().map_err(|_| format!("'{p}' is not a workgroup size"))).collect()
}

fn skipped(format: &str, rows: u64, cols: u64, wg: u32, rpg: u32, why: String) -> MvPoint {
    MvPoint {
        format: format.into(),
        rows,
        cols,
        wg,
        rpg,
        resident_bytes: 0,
        native_bytes: 0,
        copies: 0,
        iters: 0,
        gbps_median: 0.0,
        gbps_min: 0.0,
        gbps_max: 0.0,
        wall_gbps_median: 0.0,
        timing: String::new(),
        rows_checked: 0,
        err_gpu_max: 0.0,
        err_cpu_max: 0.0,
        err_ratio_rms: 0.0,
        skipped: Some(why),
    }
}

// ------------------------------------------------------------------ child (one adapter, one run)

#[cfg(feature = "gpu")]
mod gpu_side {
    use super::*;
    use crate::gpu::bench::{gpu_side::read_ceiling, median};
    use crate::gpu::device::GpuDevice;
    use crate::gpu::matvec::{error_stats, make_io, matvec, sample_rows, time_matvec, upload_matrix, MvKernel};
    use crate::gpu::pack;
    use crate::gpu::select::AdapterSelector;

    const WORKING_SET: u64 = 64 << 20;
    const MAX_RESIDENT: u64 = 1 << 30;
    const SEED: u64 = 1;

    fn run_point(dev: &GpuDevice, format: &str, rows: u64, cols: u64, wg: u32, rpg_req: u32) -> MvPoint {
        let fail = |why: String| skipped(format, rows, cols, wg, rpg_req, why);
        let Some(d) = pack::parse_dtype(format) else { return fail("unknown format".into()) };
        if let Err(e) = pack::check_shape(d, rows as usize, cols as usize) {
            return fail(format!("{e} (real models store such tensors in another format)"));
        }
        if dev.report.software && rows * cols > SOFTWARE_MAX_WEIGHTS {
            return fail("too big for a software adapter".into());
        }
        let rpg = if rpg_req == 0 { pack::default_rows(d) } else { rpg_req };
        let k = match MvKernel::config(d, wg, rpg) {
            Ok(k) => k,
            Err(e) => return fail(e),
        };
        let one: u64 = pack::array_row_bytes(d, cols as usize).iter().sum::<u64>() * rows;
        let copies = (WORKING_SET.div_ceil(one)).min((MAX_RESIDENT / one).max(1)).clamp(1, 64) as usize;
        let src = move |s: u64, n: u64| pack::gen_rows(d, s, n, cols as usize, SEED);
        let mut mats = Vec::new();
        for _ in 0..copies {
            match upload_matrix(dev, d, rows, cols, &src, None) {
                Ok(m) => mats.push(m),
                Err(e) => return fail(e),
            }
        }
        let x = pack::gen_x(cols as usize, SEED + 1);
        let r = (|| -> Result<MvPoint, String> {
            let y = matvec(dev, &k, &mats[0], &x)?;
            let check = sample_rows(&mats[0], 64);
            let st = error_stats(d, cols as usize, &y, &x, &src, &check);
            if !st.passes(pack::TOL) {
                return Err(format!("the result does not match the reference (max error {:.3e} of sum|wx|, limit {:.0e}); timing not reported", st.max_gpu, pack::TOL));
            }
            let (bx, by) = make_io(dev, &x, rows)?;
            let refs: Vec<_> = mats.iter().collect();
            let t = time_matvec(dev, &k, &refs, &bx, &by, 5, 0.15)?;
            Ok(MvPoint {
                format: format.into(),
                rows,
                cols,
                wg,
                rpg,
                resident_bytes: mats[0].resident_bytes,
                native_bytes: mats[0].native_bytes,
                copies: copies as u32,
                iters: t.iters,
                gbps_median: median(&t.gbps),
                gbps_min: t.gbps.iter().cloned().fold(f64::MAX, f64::min),
                gbps_max: t.gbps.iter().cloned().fold(0.0, f64::max),
                wall_gbps_median: median(&t.wall_gbps),
                timing: if t.timestamp { "timestamp" } else { "wall" }.into(),
                rows_checked: st.rows_checked,
                err_gpu_max: st.max_gpu,
                err_cpu_max: st.max_cpu,
                err_ratio_rms: st.ratio(),
                skipped: None,
            })
        })();
        drop(mats);
        dev.clear_pool();
        r.unwrap_or_else(fail)
    }

    pub fn run_adapter(index: usize, expect_name: &str, formats: &[String], shapes: &[(u64, u64)], wgs: &[u32], rpgs: &[u32]) -> MvAdapter {
        let mut out = MvAdapter::default();
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
        if !expect_name.is_empty() && out.name != expect_name {
            out.error = Some(format!("adapter {index} is '{}' now, expected '{expect_name}'", out.name));
            return out;
        }
        out.ceiling_mib = if dev.report.software { 16 } else { 256 };
        match read_ceiling(&dev, out.ceiling_mib) {
            Ok(g) => out.ceiling_gbps = Some(g),
            Err(e) => out.error = Some(format!("bandwidth ceiling: {e}")),
        }
        for &wg in wgs {
            for &rpg in rpgs {
                for f in formats {
                    // only the quantized kernels have a rows-per-workgroup setting
                    if rpg > 1 && !pack::parse_dtype(f).is_some_and(pack::multi_row) {
                        continue;
                    }
                    for &(r, c) in shapes {
                        let p = run_point(&dev, f, r, c, wg, rpg);
                        let dead = p.skipped.as_deref().is_some_and(|s| s.contains("did not finish") || s.contains("hang")) || dev.check_health().is_err();
                        out.points.push(p);
                        if dead {
                            out.error = Some("the device stopped answering; the remaining points were not run".into());
                            return out;
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(feature = "gpu")]
use gpu_side::run_adapter;

#[cfg(not(feature = "gpu"))]
fn run_adapter(_index: usize, _expect_name: &str, _formats: &[String], _shapes: &[(u64, u64)], _wgs: &[u32], _rpgs: &[u32]) -> MvAdapter {
    MvAdapter { error: Some("this build was compiled without the `gpu` feature".into()), ..Default::default() }
}

/// `--gpu-matvec-child <index> <name> <formats> <shapes> <wgs>`: runs one adapter, prints an `MvAdapter` as JSON.
pub fn child_main() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let pos = args.iter().position(|a| a == GPU_MATVEC_CHILD_ARG).unwrap_or(0);
    let get = |i: usize| args.get(pos + i).cloned().unwrap_or_default();
    let index: usize = get(1).parse().unwrap_or(0);
    let name = get(2);
    let formats = parse_formats(&get(3)).unwrap_or_default();
    let shapes = parse_shapes(&get(4)).unwrap_or_default();
    let wgs = parse_wgs(&get(5)).unwrap_or_else(|_| vec![super::pack::DEFAULT_WG]);
    let rpgs = parse_wgs(&get(6)).unwrap_or_else(|_| vec![0]);
    let r = std::panic::catch_unwind(|| run_adapter(index, &name, &formats, &shapes, &wgs, &rpgs)).unwrap_or_else(|_| MvAdapter { error: Some("the GPU library panicked".into()), ..Default::default() });
    println!("{}", serde_json::to_string(&r).unwrap_or_else(|_| "{}".into()));
    0
}

// ------------------------------------------------------------------ parent

/// Combines the runs of one adapter: median of the per-run medians, with the lowest and highest.
pub fn summarize(runs: &[MvAdapter]) -> Vec<MvRow> {
    use super::bench::median;
    let Some(first) = runs.iter().find(|r| !r.points.is_empty()).or(runs.first()) else { return vec![] };
    let ceilings: Vec<f64> = runs.iter().filter_map(|r| r.ceiling_gbps).collect();
    let ceiling = (!ceilings.is_empty()).then(|| median(&ceilings));
    let mut rows = Vec::new();
    for (i, p) in first.points.iter().enumerate() {
        let same: Vec<&MvPoint> = runs.iter().filter_map(|r| r.points.get(i)).filter(|q| q.format == p.format && q.rows == p.rows && q.cols == p.cols && q.wg == p.wg && q.rpg == p.rpg && q.skipped.is_none()).collect();
        let base = MvRow {
            adapter: first.name.clone(),
            backend: first.backend.clone(),
            software: first.software,
            format: p.format.clone(),
            rows: p.rows,
            cols: p.cols,
            wg: p.wg,
            rpg: p.rpg,
            resident_mib: p.resident_bytes as f64 / 1048576.0,
            native_mib: p.native_bytes as f64 / 1048576.0,
            runs: same.len(),
            gbps_median: 0.0,
            gbps_lo: 0.0,
            gbps_hi: 0.0,
            ceiling_gbps: ceiling,
            fraction_of_ceiling: None,
            err_gpu_max: 0.0,
            err_cpu_max: 0.0,
            err_ratio_rms: 0.0,
            note: None,
        };
        if same.is_empty() {
            rows.push(MvRow { note: Some(p.skipped.clone().unwrap_or_else(|| "no run finished".into())), resident_mib: 0.0, native_mib: 0.0, ..base });
            continue;
        }
        let g: Vec<f64> = same.iter().map(|q| q.gbps_median).collect();
        let m = median(&g);
        rows.push(MvRow {
            gbps_median: m,
            gbps_lo: g.iter().cloned().fold(f64::MAX, f64::min),
            gbps_hi: g.iter().cloned().fold(0.0, f64::max),
            fraction_of_ceiling: ceiling.filter(|c| *c > 0.0).map(|c| m / c),
            err_gpu_max: same.iter().map(|q| q.err_gpu_max).fold(0.0, f64::max),
            err_cpu_max: same.iter().map(|q| q.err_cpu_max).fold(0.0, f64::max),
            err_ratio_rms: median(&same.iter().map(|q| q.err_ratio_rms).collect::<Vec<_>>()),
            ..base
        });
    }
    rows
}

pub fn format_table(rows: &[MvRow], notes: &[String]) -> String {
    let mut s = String::new();
    s.push_str("Matvec kernels: weight bytes read per second (GB/s = 1e9 bytes/s), median of the runs [lowest-highest]\n\n");
    s.push_str(&format!("{:<30} {:<6} {:<5} {:<11} {:>3} {:>2} {:>9}  {:>22} {:>7} {:>9} {:>9} {:>6}\n", "adapter", "back.", "fmt", "shape", "wg", "rg", "MiB", "GB/s [lo-hi]", "% ceil", "err GPU", "err CPU", "ratio"));
    for r in rows {
        let shape = format!("{}x{}", r.rows, r.cols);
        if let Some(n) = &r.note {
            s.push_str(&format!("{:<30} {:<6} {:<5} {:<11} {:>3}  skipped: {n}\n", r.adapter, r.backend, r.format, shape, r.wg));
            continue;
        }
        let pct = r.fraction_of_ceiling.map(|f| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "-".into());
        s.push_str(&format!(
            "{:<30} {:<6} {:<5} {:<11} {:>3} {:>2} {:>9.1}  {:>7.2} [{:>5.2}-{:>5.2}] {:>7} {:>9.1e} {:>9.1e} {:>6.2}\n",
            r.adapter, r.backend, r.format, shape, r.wg, r.rpg, r.resident_mib, r.gbps_median, r.gbps_lo, r.gbps_hi, pct, r.err_gpu_max, r.err_cpu_max, r.err_ratio_rms
        ));
    }
    for n in notes {
        s.push_str(&format!("\n{n}"));
    }
    s.push('\n');
    s
}

/// Parent side. Exit code is always 0.
pub fn cli_main() -> i32 {
    use super::bench::{arg_value, run_self};
    let json = std::env::args_os().any(|a| a == "--json");
    let mut notes: Vec<String> = Vec::new();
    let formats_s = arg_value("--formats").unwrap_or_else(|| DEFAULT_FORMATS.into());
    let shapes_s = arg_value("--shapes").unwrap_or_else(|| DEFAULT_SHAPES.into());
    let wgs_s = arg_value("--wg").unwrap_or_else(|| super::pack::DEFAULT_WG.to_string());
    let rpgs_s = arg_value("--rpg").unwrap_or_else(|| "0".into());
    if let Err(e) = parse_formats(&formats_s).map(|_| ()).and(parse_shapes(&shapes_s).map(|_| ())).and(parse_wgs(&wgs_s).map(|_| ())).and(parse_wgs(&rpgs_s).map(|_| ())) {
        eprintln!("{e}");
        return 0;
    }
    let runs: usize = arg_value("--runs").and_then(|s| s.parse().ok()).unwrap_or(3).clamp(1, 9);
    let rest: u64 = arg_value("--rest").and_then(|s| s.parse().ok()).unwrap_or(0).min(300);
    let disc: Discovery = match run_self(&["--gpu-info".into(), "--json".into()], Duration::from_secs(30)).and_then(|b| serde_json::from_slice(&b).map_err(|e| format!("unreadable discovery output: {e}"))) {
        Ok(d) => d,
        Err(e) => {
            notes.push(format!("GPU discovery failed: {e}"));
            Discovery::default()
        }
    };
    let list: Vec<(usize, &super::discovery::AdapterReport)> = match arg_value("--adapter") {
        None => disc.adapters.iter().enumerate().collect(),
        Some(sel) => match super::select::select_adapter(&disc.adapters, &super::select::AdapterSelector::parse(&sel)) {
            Ok(i) => vec![(i, &disc.adapters[i])],
            Err(e) => {
                notes.push(format!("adapter selection: {e}"));
                vec![]
            }
        },
    };
    let mut rows = Vec::new();
    for (n, (i, a)) in list.iter().enumerate() {
        if n > 0 && rest > 0 {
            std::thread::sleep(Duration::from_secs(rest));
        }
        let args = vec![GPU_MATVEC_CHILD_ARG.to_string(), i.to_string(), a.name.clone(), formats_s.clone(), shapes_s.clone(), wgs_s.clone(), rpgs_s.clone()];
        let mut results = Vec::new();
        for r in 0..runs {
            if r > 0 && rest > 0 {
                std::thread::sleep(Duration::from_secs(rest.min(10)));
            }
            match run_self(&args, CHILD_TIMEOUT).and_then(|b| serde_json::from_slice::<MvAdapter>(&b).map_err(|e| format!("unreadable result: {e}"))) {
                Ok(b) => results.push(b),
                Err(e) => {
                    notes.push(format!("{} ({}): run {} failed: {e}", a.name, a.backend, r + 1));
                    break;
                }
            }
        }
        if let Some(e) = results.iter().find_map(|r| r.error.clone()) {
            notes.push(format!("{} ({}): {e}", a.name, a.backend));
        }
        let mut sum = summarize(&results);
        if sum.is_empty() {
            sum.push(MvRow {
                adapter: a.name.clone(),
                backend: a.backend.clone(),
                software: a.software,
                format: "-".into(),
                rows: 0,
                cols: 0,
                wg: 0,
                rpg: 0,
                resident_mib: 0.0,
                native_mib: 0.0,
                runs: 0,
                gbps_median: 0.0,
                gbps_lo: 0.0,
                gbps_hi: 0.0,
                ceiling_gbps: None,
                fraction_of_ceiling: None,
                err_gpu_max: 0.0,
                err_cpu_max: 0.0,
                err_ratio_rms: 0.0,
                note: Some("no result (see notes)".into()),
            });
        }
        if let Some(c) = sum.iter().find_map(|r| r.ceiling_gbps) {
            notes.push(format!("{} ({}): read-bandwidth ceiling {:.2} GB/s (median of {} runs, same session)", a.name, a.backend, c, results.len()));
        }
        rows.extend(sum);
    }
    notes.push("err = largest error of the sampled rows relative to sum|w*x| against the exact float reference (f64 over the f32-decoded weights); 'err CPU' is the CPU engine's float kernel (quant::dot_row) on the same rows; ratio = rms GPU error / rms CPU error. Activations stay f32 on the GPU (the CPU engine quantizes them to int8).".into());
    notes.push("MiB = weight buffers on the GPU after repacking. GB/s counts those bytes. Small matrices are uploaded several times (about 64 MiB) and dispatched round robin.".into());
    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "rows": rows, "notes": notes })).unwrap_or_else(|_| "{}".into()));
    } else {
        print!("{}", format_table(&rows, &notes));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsers() {
        assert_eq!(parse_shapes("4864x896, 8x4").unwrap(), vec![(4864, 896), (8, 4)]);
        assert!(parse_shapes("4864").is_err());
        assert!(parse_shapes("0x4").is_err());
        assert!(parse_shapes("axb").is_err());
        assert_eq!(parse_formats("F32,q4_k").unwrap(), vec!["F32", "Q4_K"]);
        assert!(parse_formats("bf16").is_err());
        assert_eq!(parse_formats("q5_k, Q4_0").unwrap(), vec!["Q5_K", "Q4_0"]);
        assert_eq!(parse_wgs("32, 64").unwrap(), vec![32, 64]);
        assert!(parse_wgs("x").is_err());
    }

    fn pt(gbps: f64) -> MvPoint {
        MvPoint { gbps_median: gbps, gbps_min: gbps, gbps_max: gbps, wall_gbps_median: gbps, ..skipped("F32", 4, 8, 64, 1, String::new()) }
    }

    #[test]
    fn summary_takes_the_median_of_the_runs() {
        let run = |g: f64, c: f64| MvAdapter {
            name: "GPU".into(),
            backend: "Vulkan".into(),
            ceiling_gbps: Some(c),
            points: vec![MvPoint { skipped: None, resident_bytes: 1 << 20, ..pt(g) }],
            ..Default::default()
        };
        let rows = summarize(&[run(10.0, 40.0), run(30.0, 40.0), run(20.0, 40.0)]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].gbps_median, 20.0);
        assert_eq!((rows[0].gbps_lo, rows[0].gbps_hi), (10.0, 30.0));
        assert_eq!(rows[0].fraction_of_ceiling, Some(0.5));
        assert_eq!(rows[0].runs, 3);
        // a skipped point is reported with its reason, not as a zero
        let sk = MvAdapter { name: "GPU".into(), points: vec![skipped("Q4_K", 4, 896, 64, 1, "no".into())], ..Default::default() };
        let rows = summarize(&[sk.clone(), sk]);
        assert_eq!(rows[0].note.as_deref(), Some("no"));
        assert!(format_table(&rows, &[]).is_ascii());
    }
}
