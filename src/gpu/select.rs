//! Adapter selection rule (task G2, D63). Pure functions over `AdapterReport`: testable without a GPU.
//!
//! "Best compute adapter" preference, highest first:
//! 1. discrete GPU on Vulkan (or Metal, which is the native API where Vulkan does not exist)
//! 2. integrated GPU on Vulkan/Metal
//! 3. virtual or unknown GPU type on Vulkan/Metal
//! 4. a real GPU on DX12 (fallback: the Vulkan path is the one that gets subgroups here, see D62)
//! 5. software renderers (Vulkan first, then DX12): last resort, useful for CI and for tests
//!
//! Ties are broken by the larger `max_buffer_size`, then by the lower index (stable order).

use super::discovery::AdapterReport;

#[derive(Debug, Clone, PartialEq)]
pub enum AdapterSelector {
    /// The best compute adapter by `score`.
    Best,
    /// Position in the discovery list (the same order as `buriti --gpu-info`).
    Index(usize),
    /// Best adapter of one backend (`Vulkan`, `DX12`, `Metal`), case-insensitive.
    Backend(String),
    /// Best adapter whose name contains the text, case-insensitive.
    Name(String),
    /// Best software renderer (WARP, lavapipe, SwiftShader, ...).
    Software,
}

impl AdapterSelector {
    /// Parses a command-line value: `best`, `software`, a number (index), `backend:<name>`, or any other text (name).
    pub fn parse(s: &str) -> AdapterSelector {
        let t = s.trim();
        let l = t.to_ascii_lowercase();
        if l.is_empty() || l == "best" || l == "auto" {
            return AdapterSelector::Best;
        }
        if l == "software" {
            return AdapterSelector::Software;
        }
        if let Ok(i) = l.parse::<usize>() {
            return AdapterSelector::Index(i);
        }
        if let Some(b) = l.strip_prefix("backend:") {
            return AdapterSelector::Backend(b.trim().to_string());
        }
        AdapterSelector::Name(t.to_string())
    }
}

/// Preference score of one adapter (higher is better). See the module docs for the order.
pub fn score(a: &AdapterReport) -> u32 {
    let native = a.backend == "Vulkan" || a.backend == "Metal";
    if a.software {
        return match a.backend.as_str() {
            "Vulkan" => 10,
            "DX12" => 5,
            _ => 1,
        };
    }
    let kind = match a.device_type.as_str() {
        "discrete" => 0,
        "integrated" => 1,
        "virtual" => 2,
        _ => 3,
    };
    if native {
        [100, 80, 60, 50][kind]
    } else if a.backend == "DX12" {
        [40, 30, 25, 20][kind]
    } else {
        [15, 14, 13, 12][kind]
    }
}

fn best_of(adapters: &[AdapterReport], candidates: impl Iterator<Item = usize>) -> Option<usize> {
    candidates.min_by_key(|&i| {
        let a = &adapters[i];
        // min_by_key: negate the score and the buffer size so that bigger wins; the index breaks the tie.
        (std::cmp::Reverse(score(a)), std::cmp::Reverse(a.limits.max_buffer_size), i)
    })
}

/// Index of the adapter that matches the selector, or an error that says why nothing matched.
pub fn select_adapter(adapters: &[AdapterReport], sel: &AdapterSelector) -> Result<usize, String> {
    if adapters.is_empty() {
        return Err("no GPU adapter is available".into());
    }
    let all = 0..adapters.len();
    match sel {
        AdapterSelector::Best => Ok(best_of(adapters, all).expect("non-empty")),
        AdapterSelector::Index(i) => {
            if *i < adapters.len() {
                Ok(*i)
            } else {
                Err(format!("adapter index {i} does not exist ({} adapter(s), valid 0 to {})", adapters.len(), adapters.len() - 1))
            }
        }
        AdapterSelector::Backend(b) => best_of(adapters, all.filter(|&i| adapters[i].backend.eq_ignore_ascii_case(b)))
            .ok_or_else(|| format!("no adapter on backend '{b}'")),
        AdapterSelector::Name(n) => {
            let n = n.to_ascii_lowercase();
            best_of(adapters, all.filter(|&i| adapters[i].name.to_ascii_lowercase().contains(&n)))
                .ok_or_else(|| format!("no adapter whose name contains '{n}'"))
        }
        AdapterSelector::Software => best_of(adapters, all.filter(|&i| adapters[i].software)).ok_or_else(|| "no software adapter found".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::discovery::{FeaturesReport, LimitsReport};
    use super::*;

    fn ad(name: &str, backend: &str, ty: &str, software: bool, max_buf: u64) -> AdapterReport {
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
                max_buffer_size: max_buf,
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

    fn this_machine() -> Vec<AdapterReport> {
        vec![
            ad("Intel HD 620", "Vulkan", "integrated", false, 1 << 30),
            ad("NVIDIA 940MX", "Vulkan", "discrete", false, 4 << 30),
            ad("Intel HD 620", "DX12", "integrated", false, 2 << 30),
            ad("NVIDIA 940MX", "DX12", "discrete", false, 2 << 30),
            ad("Microsoft Basic Render Driver", "DX12", "cpu", true, 2 << 30),
        ]
    }

    #[test]
    fn best_prefers_discrete_vulkan() {
        assert_eq!(select_adapter(&this_machine(), &AdapterSelector::Best), Ok(1));
    }

    #[test]
    fn best_falls_back_integrated_then_dx12_then_software() {
        let v = this_machine();
        // Without the discrete Vulkan: integrated Vulkan beats discrete DX12.
        let no_dv = vec![v[0].clone(), v[2].clone(), v[3].clone(), v[4].clone()];
        assert_eq!(select_adapter(&no_dv, &AdapterSelector::Best), Ok(0));
        // Only DX12 real GPUs and software: discrete DX12 wins over software.
        let dx = vec![v[4].clone(), v[2].clone(), v[3].clone()];
        assert_eq!(select_adapter(&dx, &AdapterSelector::Best), Ok(2));
        // Only software.
        let sw = vec![v[4].clone()];
        assert_eq!(select_adapter(&sw, &AdapterSelector::Best), Ok(0));
        // Software Vulkan (lavapipe) beats software DX12 (WARP).
        let both = vec![v[4].clone(), ad("llvmpipe", "Vulkan", "cpu", true, 1 << 30)];
        assert_eq!(select_adapter(&both, &AdapterSelector::Best), Ok(1));
    }

    #[test]
    fn ties_prefer_bigger_buffer_then_lower_index() {
        let v = vec![ad("A", "Vulkan", "discrete", false, 1 << 30), ad("B", "Vulkan", "discrete", false, 2 << 30), ad("C", "Vulkan", "discrete", false, 2 << 30)];
        assert_eq!(select_adapter(&v, &AdapterSelector::Best), Ok(1));
    }

    #[test]
    fn explicit_selectors() {
        let v = this_machine();
        assert_eq!(select_adapter(&v, &AdapterSelector::Index(3)), Ok(3));
        assert!(select_adapter(&v, &AdapterSelector::Index(5)).is_err());
        assert_eq!(select_adapter(&v, &AdapterSelector::Backend("dx12".into())), Ok(3));
        assert_eq!(select_adapter(&v, &AdapterSelector::Name("intel".into())), Ok(0));
        assert_eq!(select_adapter(&v, &AdapterSelector::Software), Ok(4));
        assert!(select_adapter(&v, &AdapterSelector::Backend("metal".into())).is_err());
        assert!(select_adapter(&v, &AdapterSelector::Name("radeon".into())).is_err());
        assert!(select_adapter(&[], &AdapterSelector::Best).is_err());
    }

    #[test]
    fn parse_selector_text() {
        assert_eq!(AdapterSelector::parse(""), AdapterSelector::Best);
        assert_eq!(AdapterSelector::parse("Best"), AdapterSelector::Best);
        assert_eq!(AdapterSelector::parse("software"), AdapterSelector::Software);
        assert_eq!(AdapterSelector::parse("2"), AdapterSelector::Index(2));
        assert_eq!(AdapterSelector::parse("backend:Vulkan"), AdapterSelector::Backend("vulkan".into()));
        assert_eq!(AdapterSelector::parse("940MX"), AdapterSelector::Name("940MX".into()));
    }
}
