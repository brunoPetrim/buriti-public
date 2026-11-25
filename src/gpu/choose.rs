//! Where the own engine computes (task G10, D69): the pure decision between the CPU and a GPU adapter. No driver is touched here; the
//! caller passes the adapters that the isolated discovery found (`gpu::discover_isolated`) and the memory of the adapters when it is known.
//!
//! The rule, in plain words:
//! - `cpu`: always the CPU.
//! - `gpu`: the adapter named by the selector, unless it is a software renderer, or its memory is known and the model does not fit.
//! - `auto` (heuristic, measured on one machine only, see D69): a GPU only if the selected adapter is a **discrete** GPU of a real vendor,
//!   its memory is **known**, and the model (weights + KV cache + working buffers) takes at most 80% of it. An integrated GPU, an unknown
//!   memory size or a software renderer means the CPU: the conservative answer when nothing was measured on that hardware.

use super::discovery::AdapterReport;
use super::select::{select_adapter, AdapterSelector};

/// Why the CPU was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    /// The setting asked for the CPU.
    Requested,
    /// No adapter matched (none found, or the selector names none).
    NoGpu,
    /// The only match is a software renderer (WARP, lavapipe, ...).
    SoftwareOnly,
    /// `auto` and the adapter is not a discrete GPU.
    NotDiscrete,
    /// `auto` and the adapter's memory is not known.
    UnknownMemory,
    /// The model does not fit in the adapter's memory.
    DoesNotFit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    Cpu(CpuReason),
    Gpu { index: usize, name: String },
}

/// Fraction of the adapter's memory the model may take in `auto` (the rest is for the driver, the display and other programs).
pub const AUTO_MEMORY_FRACTION: f64 = 0.8;

/// `memory_mb_of`: the memory of an adapter in MiB when known (the nvidia-smi figure merged by `gpu::merge`). `need_bytes`: weights plus KV
/// cache plus working buffers of the model at the configured context (`need_bytes` below).
pub fn choose(mode: &str, device: &str, adapters: &[AdapterReport], memory_mb_of: &dyn Fn(&AdapterReport) -> Option<u64>, need_bytes: u64) -> Choice {
    if mode == "cpu" {
        return Choice::Cpu(CpuReason::Requested);
    }
    let index = match select_adapter(adapters, &AdapterSelector::parse(device)) {
        Ok(i) => i,
        Err(_) => return Choice::Cpu(CpuReason::NoGpu),
    };
    let a = &adapters[index];
    if a.software {
        return Choice::Cpu(CpuReason::SoftwareOnly);
    }
    let mem = memory_mb_of(a).map(|mb| mb * 1024 * 1024);
    if mode == "auto" {
        if a.device_type != "discrete" {
            return Choice::Cpu(CpuReason::NotDiscrete);
        }
        let Some(m) = mem else { return Choice::Cpu(CpuReason::UnknownMemory) };
        if need_bytes as f64 > m as f64 * AUTO_MEMORY_FRACTION {
            return Choice::Cpu(CpuReason::DoesNotFit);
        }
    } else if let Some(m) = mem {
        // forced GPU: only a model that cannot fit at all is refused
        if need_bytes > m {
            return Choice::Cpu(CpuReason::DoesNotFit);
        }
    }
    Choice::Gpu { index, name: a.name.clone() }
}

/// Memory the GPU path needs: the weights as they sit in the file (the GPU keeps them in the same quantized format, repacked to nearly the
/// same size), the f16 KV cache of the whole context, and a margin for the activation buffers, the batch buffers of the prefill and the
/// attention scratch (about 100 MiB at the default settings).
pub fn need_bytes(weights_file_bytes: u64, n_layer: usize, kv_dim: usize, ctx: usize) -> u64 {
    let kv = n_layer as u64 * 2 * ctx as u64 * kv_dim as u64 * 2;
    weights_file_bytes + kv + 100 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::super::discovery::{FeaturesReport, LimitsReport};
    use super::*;

    fn ad(name: &str, backend: &str, ty: &str, software: bool) -> AdapterReport {
        AdapterReport {
            name: name.into(),
            vendor_id: 0,
            vendor: "x".into(),
            device_id: 0,
            device_type: ty.into(),
            backend: backend.into(),
            driver: String::new(),
            driver_info: String::new(),
            software,
            limits: LimitsReport {
                max_buffer_size: 1 << 30,
                max_storage_buffer_binding_size: 1 << 30,
                max_compute_workgroup_size_x: 256,
                max_compute_workgroup_size_y: 256,
                max_compute_workgroup_size_z: 64,
                max_compute_invocations_per_workgroup: 256,
                max_compute_workgroups_per_dimension: 65535,
                max_compute_workgroup_storage_size: 16384,
                min_subgroup_size: 0,
                max_subgroup_size: 0,
            },
            features: FeaturesReport { shader_f16: false, subgroup: false },
        }
    }

    const MIB: u64 = 1024 * 1024;
    fn mem_940(a: &AdapterReport) -> Option<u64> {
        if a.name.contains("940MX") {
            Some(4096)
        } else {
            None
        }
    }
    fn machine() -> Vec<AdapterReport> {
        vec![ad("Intel HD 620", "Vulkan", "integrated", false), ad("NVIDIA 940MX", "Vulkan", "discrete", false), ad("NVIDIA 940MX", "DX12", "discrete", false), ad("WARP", "DX12", "cpu", true)]
    }

    #[test]
    fn cpu_when_asked() {
        assert_eq!(choose("cpu", "best", &machine(), &mem_940, 100 * MIB), Choice::Cpu(CpuReason::Requested));
    }

    #[test]
    fn auto_takes_the_discrete_gpu_that_fits() {
        let c = choose("auto", "best", &machine(), &mem_940, 600 * MIB);
        assert_eq!(c, Choice::Gpu { index: 1, name: "NVIDIA 940MX".into() }, "the best adapter is the discrete Vulkan one, not the integrated or the software one");
    }

    #[test]
    fn auto_refuses_what_does_not_fit_in_eighty_percent() {
        // 4096 MiB * 0.8 = 3276.8 MiB
        assert!(matches!(choose("auto", "best", &machine(), &mem_940, 3200 * MIB), Choice::Gpu { .. }));
        assert_eq!(choose("auto", "best", &machine(), &mem_940, 3300 * MIB), Choice::Cpu(CpuReason::DoesNotFit));
    }

    #[test]
    fn auto_is_conservative_without_a_discrete_gpu_or_memory_figure() {
        let only_integrated = vec![ad("Intel HD 620", "Vulkan", "integrated", false)];
        assert_eq!(choose("auto", "best", &only_integrated, &|_| Some(8192), 100 * MIB), Choice::Cpu(CpuReason::NotDiscrete));
        let unknown = vec![ad("Some GPU", "Vulkan", "discrete", false)];
        assert_eq!(choose("auto", "best", &unknown, &|_| None, 100 * MIB), Choice::Cpu(CpuReason::UnknownMemory));
    }

    #[test]
    fn forced_gpu_accepts_integrated_and_unknown_memory_but_not_software_or_impossible_models() {
        let only_integrated = vec![ad("Intel HD 620", "DX12", "integrated", false)];
        assert!(matches!(choose("gpu", "best", &only_integrated, &|_| None, 500 * MIB), Choice::Gpu { index: 0, .. }));
        assert_eq!(choose("gpu", "best", &machine(), &mem_940, 5000 * MIB), Choice::Cpu(CpuReason::DoesNotFit), "more than the whole memory is refused even when forced");
        assert!(matches!(choose("gpu", "best", &machine(), &mem_940, 3900 * MIB), Choice::Gpu { .. }), "between 80% and 100% is allowed when forced");
        let software = vec![ad("WARP", "DX12", "cpu", true)];
        assert_eq!(choose("gpu", "best", &software, &|_| None, MIB), Choice::Cpu(CpuReason::SoftwareOnly));
    }

    #[test]
    fn the_device_setting_picks_the_adapter() {
        assert!(matches!(choose("gpu", "backend:dx12", &machine(), &mem_940, 100 * MIB), Choice::Gpu { index: 2, .. }));
        assert!(matches!(choose("gpu", "intel", &machine(), &mem_940, 100 * MIB), Choice::Gpu { index: 0, .. }));
        assert_eq!(choose("gpu", "radeon", &machine(), &mem_940, 100 * MIB), Choice::Cpu(CpuReason::NoGpu));
        assert_eq!(choose("gpu", "7", &machine(), &mem_940, 100 * MIB), Choice::Cpu(CpuReason::NoGpu));
        assert_eq!(choose("auto", "best", &[], &mem_940, 100 * MIB), Choice::Cpu(CpuReason::NoGpu));
    }

    #[test]
    fn need_counts_weights_kv_and_a_margin() {
        // Qwen 2.5 0.5B: 24 layers, kv_dim 128, 2048 positions: 24 * 2 * 2048 * 128 * 2 = 25.2 MB of KV
        let n = need_bytes(463 * MIB, 24, 128, 2048);
        assert_eq!(n, 463 * MIB + 24 * 2 * 2048 * 128 * 2 + 100 * MIB);
        assert!(need_bytes(0, 24, 128, 4096) > need_bytes(0, 24, 128, 2048), "a longer context needs more");
    }
}
