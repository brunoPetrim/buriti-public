//! First compute kernels on the device layer (task G2, D63): elementwise f32 add and a sum of squares with a
//! workgroup reduction, each with a CPU reference. The runners take the WGSL text as a parameter so the
//! tests can run deliberately broken variants and prove that the checks have teeth.

#![allow(dead_code)]

use super::device::{GpuDevice, DEFAULT_TIMEOUT};

pub const VEC_ADD_WGSL: &str = include_str!("shaders/vec_add.wgsl");
pub const SUM_SQUARES_WGSL: &str = include_str!("shaders/sum_squares.wgsl");

const VEC_ADD_WG: u64 = 64;
const SUM_WG: u32 = 256;

/// Splits `groups` workgroups over x and y so that neither exceeds `max`.
pub fn grid_2d(groups: u64, max: u32) -> Result<[u32; 3], String> {
    let max = max as u64;
    if groups == 0 {
        return Err("empty dispatch".into());
    }
    let x = groups.min(max);
    let y = groups.div_ceil(x);
    if y > max {
        return Err(format!("{groups} workgroups do not fit in a {max} x {max} grid"));
    }
    Ok([x as u32, y as u32, 1])
}

/// `out[i] = a[i] + b[i]` on the GPU. `wgsl` is the shader text, `key` its pipeline-cache key.
pub fn vec_add(dev: &GpuDevice, key: &str, wgsl: &str, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err("vec_add: the inputs have different lengths".into());
    }
    if a.is_empty() {
        return Ok(Vec::new());
    }
    let pipeline = dev.pipeline(key, wgsl, "main")?;
    let ba = dev.storage_from_f32(a, "vec_add.a")?;
    let bb = dev.storage_from_f32(b, "vec_add.b")?;
    let out = dev.create_storage(a.len() as u64 * 4, "vec_add.out")?;
    let groups = grid_2d((a.len() as u64).div_ceil(VEC_ADD_WG), dev.caps.max_compute_workgroups_per_dimension)?;
    dev.run(&pipeline, &[&ba, &bb, &out], groups, 1, DEFAULT_TIMEOUT)?;
    dev.read_back_f32(&out, a.len(), DEFAULT_TIMEOUT)
}

pub fn cpu_vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// IEEE addition is exact and identical on both sides: the check is bit equality.
pub fn check_vec_add(got: &[f32], want: &[f32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("length {} instead of {}", got.len(), want.len()));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(format!("element {i}: GPU {g} vs CPU {w}"));
        }
    }
    Ok(())
}

/// Sum of squares on the GPU: one partial per workgroup, added on the CPU in f64.
pub fn sum_squares(dev: &GpuDevice, key: &str, wgsl: &str, x: &[f32]) -> Result<f64, String> {
    if x.is_empty() {
        return Ok(0.0);
    }
    let pipeline = dev.pipeline(key, wgsl, "main")?;
    let bx = dev.storage_from_f32(x, "sum_squares.x")?;
    let groups = ((x.len() as u64).div_ceil(SUM_WG as u64)).min(256).max(1) as u32;
    let bp = dev.create_storage(groups as u64 * 4, "sum_squares.partial")?;
    dev.run(&pipeline, &[&bx, &bp], [groups, 1, 1], 1, DEFAULT_TIMEOUT)?;
    let partial = dev.read_back_f32(&bp, groups as usize, DEFAULT_TIMEOUT)?;
    Ok(partial.iter().map(|&p| p as f64).sum())
}

pub fn cpu_sum_squares(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// The GPU adds in f32 in another order: relative tolerance against the f64 reference.
pub fn check_sum(got: f64, want: f64, rel_tol: f64) -> Result<(), String> {
    let err = (got - want).abs() / want.abs().max(1e-30);
    if err <= rel_tol {
        Ok(())
    } else {
        Err(format!("sum {got} vs reference {want} (relative error {err:.3e} > {rel_tol:.0e})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::select::AdapterSelector;

    /// Deterministic values in [-1, 1) (a simple LCG; no denormals).
    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// `None` (with the reason printed) when no device can be created, for example on CI without a driver.
    fn open_or_skip(sel: &AdapterSelector, what: &str) -> Option<GpuDevice> {
        match GpuDevice::open(sel) {
            Ok(d) => {
                println!("[{what}] adapter: {} ({}, {})", d.name(), d.report.backend, d.report.device_type);
                Some(d)
            }
            Err(e) => {
                println!("SKIP [{what}]: no GPU device could be created: {e}");
                None
            }
        }
    }

    fn replace_once(src: &str, from: &str, to: &str) -> String {
        let src = &src.replace("\r\n", "\n");
        assert_eq!(src.matches(from).count(), 1, "planted-bug marker '{from}' must appear exactly once");
        src.replace(from, to)
    }

    /// The whole battery for one device. Used for the software adapter (CI) and the real adapters.
    fn exercise(dev: &GpuDevice) {
        // --- vec_add: correct kernel matches the CPU bit for bit (sizes: not a multiple of the workgroup, one and several groups).
        for n in [1usize, 63, 64, 65, 1000, 100_003] {
            let (a, b) = (data(n, 1), data(n, 2));
            let got = vec_add(dev, "vec_add", VEC_ADD_WGSL, &a, &b).expect("vec_add");
            check_vec_add(&got, &cpu_vec_add(&a, &b)).unwrap_or_else(|e| panic!("n={n}: {e}"));
        }
        // 2D grid: more groups than one dimension allows only if the limit is small; force it with a big n when cheap.
        let max_1d = dev.caps.max_compute_workgroups_per_dimension as u64 * VEC_ADD_WG;
        if max_1d < 8_000_000 {
            let n = max_1d as usize + 1000;
            let (a, b) = (data(n, 3), data(n, 4));
            let got = vec_add(dev, "vec_add", VEC_ADD_WGSL, &a, &b).expect("vec_add 2D");
            check_vec_add(&got, &cpu_vec_add(&a, &b)).expect("vec_add 2D grid");
        }
        // --- planted bug: wrong index on b. Must be caught by the check.
        let buggy = replace_once(VEC_ADD_WGSL, "out[i] = a[i] + b[i];", "out[i] = a[i] + b[i + 1u];");
        let (a, b) = (data(1000, 1), data(1000, 2));
        let got = vec_add(dev, "vec_add_bug_index", &buggy, &a, &b).expect("buggy shader still runs");
        assert!(check_vec_add(&got, &cpu_vec_add(&a, &b)).is_err(), "planted index bug was NOT detected");

        // --- sum_squares: tolerance against the f64 reference, a few sizes (partial last workgroup, several groups, grid stride).
        for n in [1usize, 255, 256, 257, 10_000, 1_000_003] {
            let x = data(n, 5);
            let got = sum_squares(dev, "sum_squares", SUM_SQUARES_WGSL, &x).expect("sum_squares");
            check_sum(got, cpu_sum_squares(&x), 1e-4).unwrap_or_else(|e| panic!("n={n}: {e}"));
        }
        // --- planted bugs: each must be caught.
        let x = data(10_000, 5);
        let want = cpu_sum_squares(&x);
        let bugs = [
            ("wrong index in the reduction", replace_once(SUM_SQUARES_WGSL, "scratch[lid.x] += scratch[lid.x + s];", "scratch[lid.x] += scratch[lid.x + s + 1u];")),
            ("not squared", replace_once(SUM_SQUARES_WGSL, "acc += v * v;", "acc += v;")),
            ("reduction starts at 64 (half of the partials dropped)", replace_once(SUM_SQUARES_WGSL, "var s = 128u;", "var s = 64u;")),
        ];
        for (i, (what, src)) in bugs.iter().enumerate() {
            let got = sum_squares(dev, &format!("sum_squares_bug{i}"), src, &x).expect("buggy shader still runs");
            assert!(check_sum(got, want, 1e-4).is_err(), "planted bug '{what}' was NOT detected");
        }
        // Missing barrier: whether it breaks depends on the hardware (lockstep execution can hide it), so it is
        // reported, not asserted.
        let no_barrier = replace_once(SUM_SQUARES_WGSL, "        workgroupBarrier();\n        s = s >> 1u;", "        s = s >> 1u;");
        let got = sum_squares(dev, "sum_squares_no_barrier", &no_barrier, &x).expect("no-barrier shader runs");
        println!("  planted 'missing barrier' bug: {}", if check_sum(got, want, 1e-4).is_err() { "detected" } else { "NOT visible on this adapter (lockstep hides it)" });

        // --- pipeline cache: same key + same text compiles once; changed text recompiles.
        let before = dev.compile_count();
        dev.pipeline("vec_add", VEC_ADD_WGSL, "main").unwrap();
        dev.pipeline("vec_add", VEC_ADD_WGSL, "main").unwrap();
        assert_eq!(dev.compile_count(), before, "cached pipeline was recompiled");
        dev.pipeline("vec_add", &buggy, "main").unwrap();
        assert_eq!(dev.compile_count(), before + 1, "changed source behind the same key must recompile");
        dev.pipeline("vec_add", VEC_ADD_WGSL, "main").unwrap();

        // --- bad shader: an error with the compiler message, not a panic.
        let e = dev.pipeline("broken", "this is not wgsl", "main").err().expect("broken shader must fail");
        println!("  broken shader error: {}", e.lines().next().unwrap_or(""));
        assert!(e.contains("broken"));
        let e = dev.pipeline("no_entry", VEC_ADD_WGSL, "nope").err().expect("missing entry point must fail");
        assert!(!e.is_empty());
        dev.check_health().expect("device must still be healthy after rejected shaders");

        // --- buffers: roundtrip, offsets, limits, pool reuse.
        let b = dev.create_storage(4096, "rt").unwrap();
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
        dev.upload_bytes(&b, 0, &bytes).unwrap();
        assert_eq!(dev.read_back(&b, 0, 4096, DEFAULT_TIMEOUT).unwrap(), bytes);
        assert_eq!(dev.read_back(&b, 1024, 8, DEFAULT_TIMEOUT).unwrap(), bytes[1024..1032]);
        assert!(dev.upload_bytes(&b, 0, &bytes[..3]).is_err(), "unaligned upload must be an error");
        assert!(dev.upload_bytes(&b, 4096, &[0u8; 4]).is_err(), "upload past the end must be an error");
        assert!(dev.read_back(&b, 4092, 8, DEFAULT_TIMEOUT).is_err(), "readback past the end must be an error");
        let too_big = dev.caps.max_buffer_size.saturating_add(4);
        let e = dev.create_storage(too_big, "huge").err().expect("buffer above max_buffer_size must fail");
        assert!(e.contains("max_buffer_size"), "{e}");
        assert!(dev.create_storage(0, "zero").is_err());
        let p1 = dev.acquire_storage(1000, "pool").unwrap();
        dev.release(p1);
        let s0 = dev.pool_stats();
        let p2 = dev.acquire_storage(900, "pool").unwrap(); // same power-of-two bucket (1024)
        assert_eq!(dev.pool_stats().reused, s0.reused + 1, "pool did not reuse the buffer");
        assert_eq!(p2.len, 900);
        dev.release(p2);
        dev.clear_pool();
        assert_eq!(dev.pool_stats().retained_bytes, 0);
        dev.check_health().expect("healthy at the end");
    }

    #[test]
    fn grid_2d_splits_and_rejects() {
        assert_eq!(grid_2d(10, 65535).unwrap(), [10, 1, 1]);
        assert_eq!(grid_2d(65536, 65535).unwrap(), [65535, 2, 1]);
        assert_eq!(grid_2d(65535 * 3, 65535).unwrap(), [65535, 3, 1]);
        assert!(grid_2d(0, 10).is_err());
        assert!(grid_2d(101, 10).is_err());
    }

    #[test]
    fn checks_reject_wrong_results() {
        assert!(check_vec_add(&[1.0, 2.0], &[1.0, 2.0]).is_ok());
        assert!(check_vec_add(&[1.0, 2.0], &[1.0, 2.5]).is_err());
        assert!(check_vec_add(&[1.0], &[1.0, 2.0]).is_err());
        assert!(check_sum(100.0, 100.001, 1e-4).is_ok());
        assert!(check_sum(100.0, 101.0, 1e-4).is_err());
    }

    /// Runs on the software adapter (WARP, lavapipe, SwiftShader). Skips cleanly when there is none (CI, no driver).
    #[test]
    fn software_adapter_kernels() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        exercise(&dev);
    }

    /// Needs a real GPU: `cargo test real_adapter_kernels -- --ignored --nocapture`. Uses the best compute
    /// adapter, or the one in `BURITI_TEST_ADAPTER` (`best`, an index of `--gpu-info`, `backend:dx12`, a name).
    /// One adapter per process on purpose: a driver that crashes in device creation (the Intel HD 620 on
    /// Vulkan does, D63) must not hide the result of the others.
    #[test]
    #[ignore]
    fn real_adapter_kernels() {
        let sel = std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best);
        let dev = open_or_skip(&sel, "real").expect("no adapter could be opened");
        println!("caps: {:?}", dev.caps);
        exercise(&dev);
    }
}
