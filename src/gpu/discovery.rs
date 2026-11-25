//! GPU adapter discovery (task G1, D62).
//!
//! Enumerates the adapters that `wgpu` can see (Vulkan, DX12, Metal) and reports what matters for compute.
//! Reading adapter info and limits needs no device. Everything here is best effort: a missing driver or a
//! failure inside wgpu yields an empty list plus a reason string, never a panic.
//!
//! The data types do not depend on wgpu, so they (and the formatting) are testable without a GPU and the
//! crate still builds with `--no-default-features`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterReport {
    pub name: String,
    /// PCI vendor id (0 when unknown).
    pub vendor_id: u32,
    pub vendor: String,
    pub device_id: u32,
    /// `discrete`, `integrated`, `virtual`, `cpu` or `other`.
    pub device_type: String,
    /// `Vulkan`, `DX12` or `Metal`.
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    /// True for a CPU/software renderer (lavapipe, WARP, SwiftShader, ...): not a real GPU.
    pub software: bool,
    pub limits: LimitsReport,
    pub features: FeaturesReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LimitsReport {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaturesReport {
    /// `SHADER_F16`: half-precision arithmetic in WGSL.
    pub shader_f16: bool,
    /// `SUBGROUP`: subgroup (warp/wave) operations in WGSL.
    pub subgroup: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Discovery {
    pub adapters: Vec<AdapterReport>,
    /// Why the list is empty (or short), when it is. `None` when at least one adapter was found.
    pub reason: Option<String>,
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x8086 => "Intel",
        0x1002 | 0x1022 => "AMD",
        0x106B => "Apple",
        0x5143 => "Qualcomm",
        0x13B5 => "ARM",
        0x1010 => "Imagination",
        0x1414 => "Microsoft",
        0x10005 => "Mesa",
        0 => "unknown",
        _ => "other",
    }
}

/// Name heuristics for software renderers that do not report `DeviceType::Cpu`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn looks_like_software(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["llvmpipe", "lavapipe", "swiftshader", "softpipe", "microsoft basic render", "warp"]
        .iter()
        .any(|k| n.contains(k))
}

pub fn mib(bytes: u64) -> String {
    if bytes >= 1 << 30 && bytes % (1 << 30) == 0 {
        format!("{} GiB", bytes >> 30)
    } else {
        format!("{} MiB", bytes >> 20)
    }
}

/// Human-readable report (ASCII only: safe in any console).
pub fn format_table(d: &Discovery) -> String {
    let mut s = String::new();
    if d.adapters.is_empty() {
        s.push_str("No GPU adapter found.\n");
        if let Some(r) = &d.reason {
            s.push_str(&format!("Reason: {r}\n"));
        }
        return s;
    }
    s.push_str(&format!("{} adapter(s) found.\n", d.adapters.len()));
    for (i, a) in d.adapters.iter().enumerate() {
        let l = &a.limits;
        s.push_str(&format!("\n[{i}] {}\n", a.name));
        s.push_str(&format!(
            "    vendor      : {} (0x{:04x})   device id 0x{:04x}\n",
            a.vendor, a.vendor_id, a.device_id
        ));
        s.push_str(&format!(
            "    type        : {}{}\n",
            a.device_type,
            if a.software { " (software renderer, not a real GPU)" } else { "" }
        ));
        s.push_str(&format!("    backend     : {}\n", a.backend));
        s.push_str(&format!("    driver      : {} {}\n", a.driver, a.driver_info).trim_end().to_string());
        s.push('\n');
        s.push_str(&format!("    max buffer  : {}   max storage binding: {}\n", mib(l.max_buffer_size), mib(l.max_storage_buffer_binding_size)));
        s.push_str(&format!(
            "    workgroup   : size {} x {} x {}, {} invocations, {} per dimension, {} B shared\n",
            l.max_compute_workgroup_size_x,
            l.max_compute_workgroup_size_y,
            l.max_compute_workgroup_size_z,
            l.max_compute_invocations_per_workgroup,
            l.max_compute_workgroups_per_dimension,
            l.max_compute_workgroup_storage_size
        ));
        s.push_str(&format!(
            "    f16 shaders : {}   subgroups: {}{}\n",
            if a.features.shader_f16 { "yes" } else { "no" },
            if a.features.subgroup { "yes" } else { "no" },
            if a.features.subgroup { format!(" (size {} to {})", l.min_subgroup_size, l.max_subgroup_size) } else { String::new() }
        ));
    }
    if let Some(r) = &d.reason {
        s.push_str(&format!("\nNote: {r}\n"));
    }
    s.push_str("\nVRAM size is not reported by wgpu; the web UI gets it from nvidia-smi when available.\n");
    s
}

pub fn format_json(d: &Discovery) -> String {
    serde_json::to_string_pretty(d).unwrap_or_else(|_| "{\"adapters\":[],\"reason\":\"serialization failed\"}".into())
}

/// Enumerate adapters. Never panics; see the module docs.
#[cfg(feature = "gpu")]
pub fn discover() -> Discovery {
    match std::panic::catch_unwind(enumerate) {
        Ok(d) => d,
        Err(_) => Discovery { adapters: vec![], reason: Some("the GPU library panicked while enumerating adapters".into()) },
    }
}

#[cfg(not(feature = "gpu"))]
pub fn discover() -> Discovery {
    Discovery { adapters: vec![], reason: Some("this build was compiled without the `gpu` feature".into()) }
}

/// Builds the report of one wgpu adapter (`None` for a backend this project does not use, such as GL).
#[cfg(feature = "gpu")]
pub(crate) fn report_of(a: &wgpu::Adapter) -> Option<AdapterReport> {
    use wgpu::{Backend, DeviceType, Features};
    let info = a.get_info();
    let lim = a.limits();
    let feat = a.features();
    let backend = match info.backend {
        Backend::Vulkan => "Vulkan",
        Backend::Dx12 => "DX12",
        Backend::Metal => "Metal",
        _ => return None,
    };
    let device_type = match info.device_type {
        DeviceType::DiscreteGpu => "discrete",
        DeviceType::IntegratedGpu => "integrated",
        DeviceType::VirtualGpu => "virtual",
        DeviceType::Cpu => "cpu",
        DeviceType::Other => "other",
    };
    let software = info.device_type == DeviceType::Cpu || looks_like_software(&info.name);
    Some(AdapterReport {
        vendor: vendor_name(info.vendor).to_string(),
        vendor_id: info.vendor,
        device_id: info.device,
        device_type: device_type.to_string(),
        backend: backend.to_string(),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
        software,
        limits: LimitsReport {
            max_buffer_size: lim.max_buffer_size,
            max_storage_buffer_binding_size: lim.max_storage_buffer_binding_size,
            max_compute_workgroup_size_x: lim.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: lim.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: lim.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: lim.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: lim.max_compute_workgroups_per_dimension,
            max_compute_workgroup_storage_size: lim.max_compute_workgroup_storage_size,
            min_subgroup_size: info.subgroup_min_size,
            max_subgroup_size: info.subgroup_max_size,
        },
        features: FeaturesReport { shader_f16: feat.contains(Features::SHADER_F16), subgroup: feat.contains(Features::SUBGROUP) },
        name: info.name,
    })
}

/// The wgpu instance and its adapters (the ones `report_of` accepts), in the order of `discover()`.
#[cfg(feature = "gpu")]
pub(crate) fn enumerate_wgpu() -> (wgpu::Instance, Vec<(wgpu::Adapter, AdapterReport)>) {
    use wgpu::{Backends, InstanceDescriptor, InstanceFlags};
    let mut desc = InstanceDescriptor::new_without_display_handle();
    let backends = Backends::VULKAN | Backends::DX12 | Backends::METAL;
    desc.backends = backends;
    desc.flags = InstanceFlags::empty(); // no validation layers
    let instance = wgpu::Instance::new(desc);
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let list = adapters.into_iter().filter_map(|a| report_of(&a).map(|r| (a, r))).collect();
    (instance, list)
}

#[cfg(feature = "gpu")]
fn enumerate() -> Discovery {
    let (_instance, list) = enumerate_wgpu();
    let out: Vec<AdapterReport> = list.into_iter().map(|(_, r)| r).collect();
    let reason = if out.is_empty() { Some("wgpu found no Vulkan, DX12 or Metal adapter (no GPU or no driver)".to_string()) } else { None };
    Discovery { adapters: out, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(name: &str, backend: &str, software: bool) -> AdapterReport {
        AdapterReport {
            name: name.into(),
            vendor_id: 0x10DE,
            vendor: vendor_name(0x10DE).into(),
            device_id: 0x134D,
            device_type: "discrete".into(),
            backend: backend.into(),
            driver: "NVIDIA".into(),
            driver_info: "512.00".into(),
            software,
            limits: LimitsReport {
                max_buffer_size: 4 << 30,
                max_storage_buffer_binding_size: 2 << 30,
                max_compute_workgroup_size_x: 1024,
                max_compute_workgroup_size_y: 1024,
                max_compute_workgroup_size_z: 64,
                max_compute_invocations_per_workgroup: 1024,
                max_compute_workgroups_per_dimension: 65535,
                max_compute_workgroup_storage_size: 49152,
                min_subgroup_size: 32,
                max_subgroup_size: 32,
            },
            features: FeaturesReport { shader_f16: false, subgroup: true },
        }
    }

    #[test]
    fn empty_discovery_formats_with_reason() {
        let d = Discovery { adapters: vec![], reason: Some("no driver".into()) };
        let t = format_table(&d);
        assert!(t.contains("No GPU adapter found."));
        assert!(t.contains("Reason: no driver"));
        let j: serde_json::Value = serde_json::from_str(&format_json(&d)).unwrap();
        assert_eq!(j["adapters"].as_array().unwrap().len(), 0);
        assert_eq!(j["reason"], "no driver");
    }

    #[test]
    fn table_lists_limits_and_features() {
        let d = Discovery { adapters: vec![sample("GeForce Test", "Vulkan", false)], reason: None };
        let t = format_table(&d);
        assert!(t.contains("1 adapter(s) found."));
        assert!(t.contains("GeForce Test"));
        assert!(t.contains("NVIDIA (0x10de)"));
        assert!(t.contains("max buffer  : 4 GiB"));
        assert!(t.contains("storage binding: 2 GiB"));
        assert!(t.contains("size 1024 x 1024 x 64"));
        assert!(t.contains("f16 shaders : no   subgroups: yes (size 32 to 32)"));
        assert!(t.is_ascii());
        let sw = Discovery { adapters: vec![sample("llvmpipe", "Vulkan", true)], reason: None };
        assert!(format_table(&sw).contains("software renderer"));
    }

    #[test]
    fn json_shape_is_stable() {
        let d = Discovery { adapters: vec![sample("GeForce Test", "DX12", false)], reason: None };
        let j: serde_json::Value = serde_json::from_str(&format_json(&d)).unwrap();
        let a = &j["adapters"][0];
        for k in ["name", "vendor_id", "vendor", "device_id", "device_type", "backend", "driver", "driver_info", "software", "limits", "features"] {
            assert!(a.get(k).is_some(), "missing key {k}");
        }
        for k in [
            "max_buffer_size",
            "max_storage_buffer_binding_size",
            "max_compute_workgroup_size_x",
            "max_compute_workgroup_size_y",
            "max_compute_workgroup_size_z",
            "max_compute_invocations_per_workgroup",
            "max_compute_workgroups_per_dimension",
            "min_subgroup_size",
            "max_subgroup_size",
        ] {
            assert!(a["limits"].get(k).is_some(), "missing limit {k}");
        }
        assert_eq!(a["features"]["shader_f16"], false);
        assert_eq!(a["features"]["subgroup"], true);
        assert_eq!(a["backend"], "DX12");
        assert!(j["reason"].is_null());
    }

    #[test]
    fn software_name_heuristics() {
        assert!(looks_like_software("llvmpipe (LLVM 15.0.7, 256 bits)"));
        assert!(looks_like_software("Microsoft Basic Render Driver"));
        assert!(looks_like_software("Microsoft Direct3D12 (WARP)"));
        assert!(!looks_like_software("NVIDIA GeForce 940MX"));
        assert!(!looks_like_software("Intel(R) HD Graphics 620"));
    }

    #[test]
    fn vendor_names() {
        assert_eq!(vendor_name(0x10DE), "NVIDIA");
        assert_eq!(vendor_name(0x8086), "Intel");
        assert_eq!(vendor_name(0x1002), "AMD");
        assert_eq!(vendor_name(0), "unknown");
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn without_feature_returns_empty_with_reason() {
        let d = discover();
        assert!(d.adapters.is_empty());
        assert!(d.reason.unwrap().contains("gpu"));
    }

    /// Needs real drivers: `cargo test real_adapters -- --ignored --nocapture`.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn real_adapters() {
        let d = discover();
        println!("{}", format_table(&d));
        for a in &d.adapters {
            assert!(!a.name.is_empty());
            assert!(a.limits.max_buffer_size > 0);
            assert!(a.limits.max_compute_invocations_per_workgroup >= 256);
            assert!(a.limits.max_storage_buffer_binding_size >= 128 << 20);
        }
        assert_eq!(d.adapters.is_empty(), d.reason.is_some());
    }
}
