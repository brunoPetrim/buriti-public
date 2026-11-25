//! Modelo dedicado de autocompletar (FIM) ao lado do modelo do chat (D78).
//!
//! Hoje há UMA sessão por modelo carregado: um turno de chat e um pedido de autocompletar não compartilham prefixo, então cada
//! um joga fora o KV cache do outro (D76). Com um segundo modelo pequeno, residente, o autocompletar tem sessão e cache próprios.
//!
//! Política de memória (função pura `fim_plan`, testada em perfis de 4, 8, 16 e 32 GB): RAM de sobra = `dedicated` (residente);
//! pouca = `dedicated_on_demand` (carregado no primeiro pedido FIM e descartado depois de 10 min parado); quase nada =
//! `shared` (o próprio modelo do chat serve aos dois, e a recomendação diz qual modelo pequeno instalar). Os números são
//! ESTIMATIVAS (não medições), com margens conservadoras como em `embeddings::plan`.

use std::sync::Mutex;

use axum::{extract::State, response::IntoResponse, response::Response, Json};
use serde_json::{json, Value};

use crate::server::{active, Shared};

const MIB: f64 = 1_048_576.0;
/// Referências para a recomendação (tamanhos de arquivo Q4_K_M): o Qwen2.5-Coder 1.5B é o do catálogo (986 MB, medido no
/// cabeçalho do Hugging Face); o de 0,5B NÃO está no catálogo e o tamanho abaixo é uma estimativa (não conferida).
pub const REF_FIM_SMALL_BYTES: u64 = 400 * 1_048_576;
pub const REF_FIM_CATALOG_BYTES: u64 = 986_048_800;
/// Estimativa conservadora do KV cache f16 por token de um modelo de 0,5B a 1,5B (o Qwen2.5 0,5B usa 12 KiB, o 1,5B 28 KiB).
const KV_KIB_PER_TOKEN: f64 = 32.0;
/// Depois de tanto tempo sem pedido FIM, o modelo `dedicated_on_demand` é descartado.
pub const ON_DEMAND_IDLE_SECS: u64 = 600;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FimMode {
    Shared,
    Dedicated,
    DedicatedOnDemand,
}

impl FimMode {
    pub fn name(self) -> &'static str {
        match self {
            FimMode::Shared => "shared",
            FimMode::Dedicated => "dedicated",
            FimMode::DedicatedOnDemand => "dedicated_on_demand",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FimInput {
    pub ram_total_mb: u64,
    pub ram_available_mb: Option<u64>,
    /// memória de vídeo da GPU do chat (ou a maior), se conhecida
    pub vram_mb: Option<u64>,
    pub chat_on_gpu: bool,
    pub chat_bytes: u64,
    /// arquivo do modelo FIM escolhido, se há um
    pub fim_bytes: Option<u64>,
    /// contexto configurado (tokens)
    pub ctx: u32,
    /// `fim_mode` = "auto" (true) ou "off" (false)
    pub enabled: bool,
}

#[derive(Clone, Debug)]
pub struct FimPlan {
    pub mode: FimMode,
    /// `disabled`, `no_fim_model`, `fits_resident`, `fits_on_demand`, `does_not_fit` (a interface traduz)
    pub reason_code: &'static str,
    /// razão legível (pt-BR)
    pub reason: String,
    /// `none`, `use_dedicated`, `use_dedicated_on_demand`, `keep_shared`, `install_fim_model`, `install_small_fim_model`
    pub recommendation_code: &'static str,
    pub recommendation: String,
    /// RAM que sobra (MiB, pode ser negativa) depois da margem do sistema e do modelo do chat
    pub free_mb: i64,
    /// o modelo FIM caberia com folga na memória de vídeo ao lado do chat (informativo: hoje o modelo FIM roda SEMPRE na CPU)
    pub gpu_would_fit: bool,
}

impl FimPlan {
    pub fn to_json(&self) -> Value {
        json!({ "mode": self.mode.name(), "reason_code": self.reason_code, "reason": self.reason,
            "recommendation_code": self.recommendation_code, "recommendation": self.recommendation,
            "free_mb": self.free_mb, "processor": "cpu", "gpu_would_fit": self.gpu_would_fit })
    }
}

fn margin_mb(total_mb: f64) -> f64 {
    (total_mb * 0.15).max(2048.0)
}

/// RAM (MiB) do modelo do chat: pesos + KV e buffers (~15% + 256 MiB); na GPU (com a VRAM conhecida) ~25% dos pesos + 300 MiB.
fn chat_resident_mb(i: &FimInput) -> f64 {
    let w = i.chat_bytes as f64 / MIB;
    if i.chat_on_gpu && i.vram_mb.is_some() {
        w * 0.25 + 300.0
    } else {
        w * 1.15 + 256.0
    }
}

fn kv_mb(ctx: u32) -> f64 {
    ctx as f64 * KV_KIB_PER_TOKEN / 1024.0
}

/// Custo estimado do modelo FIM residente (pesos reempacotados + KV + buffers) e só durante o uso.
pub fn resident_need_mb(bytes: u64, ctx: u32) -> f64 {
    bytes as f64 / MIB * 1.3 + kv_mb(ctx) + 400.0
}
pub fn on_demand_need_mb(bytes: u64, ctx: u32) -> f64 {
    bytes as f64 / MIB * 1.15 + kv_mb(ctx) + 150.0
}

/// Função pura: o modo recomendado para a máquina.
pub fn fim_plan(i: &FimInput) -> FimPlan {
    let total = i.ram_total_mb as f64;
    let mut free = total - margin_mb(total) - chat_resident_mb(i);
    if let Some(a) = i.ram_available_mb {
        free = free.min(a as f64 - 512.0);
    }
    let free_mb = free.round() as i64;
    let gb = |mb: f64| format!("{:.1} GB", mb / 1024.0);

    // o modelo FIM cabe na placa ao lado do chat com 25% da memória de vídeo ainda livre?
    let gpu_would_fit = match (i.chat_on_gpu, i.vram_mb, i.fim_bytes) {
        (true, Some(v), Some(b)) => {
            let chat_v = i.chat_bytes as f64 / MIB * 1.1 + kv_mb(i.ctx) + 300.0;
            let fim_v = b as f64 / MIB * 1.1 + kv_mb(i.ctx) + 150.0;
            v as f64 - chat_v - fim_v >= v as f64 * 0.25
        }
        _ => false,
    };
    let done = |mode, reason_code, reason: String, recommendation_code, recommendation: String| FimPlan { mode, reason_code, reason, recommendation_code, recommendation, free_mb, gpu_would_fit };

    if !i.enabled {
        return done(FimMode::Shared, "disabled", "O modelo dedicado de autocompletar está desligado nas configurações: o modelo do chat atende aos dois.".into(), "none", "Nenhuma ação necessária.".into());
    }
    match i.fim_bytes {
        None => {
            let (code, rec) = if free >= resident_need_mb(REF_FIM_CATALOG_BYTES, i.ctx) {
                ("install_fim_model", format!("Instale um modelo de código pequeno (por exemplo o Qwen2.5-Coder 1.5B do catálogo, cerca de {}): cabe residente ao lado do chat e o autocompletar para de apagar o cache da conversa.", gb(REF_FIM_CATALOG_BYTES as f64 / MIB)))
            } else if free >= resident_need_mb(REF_FIM_SMALL_BYTES, i.ctx) {
                ("install_small_fim_model", format!("Instale um modelo de código bem pequeno (0,5B, cerca de {}, por exemplo um Qwen2.5-Coder 0,5B): o de 1,5B não caberia com folga nesta máquina.", gb(REF_FIM_SMALL_BYTES as f64 / MIB)))
            } else if free >= on_demand_need_mb(REF_FIM_SMALL_BYTES, i.ctx) {
                ("install_small_fim_model", format!("Um modelo de código de 0,5B (cerca de {}) só caberia carregado sob demanda (o primeiro autocompletar depois de uma pausa é lento).", gb(REF_FIM_SMALL_BYTES as f64 / MIB)))
            } else {
                ("keep_shared", "Nesta máquina um único modelo atende ao chat e ao autocompletar. Para o autocompletar funcionar, esse modelo precisa ter tokens FIM (um modelo de código, como o Qwen2.5-Coder 0,5B a 1,5B); alternar entre conversa e autocompletar recalcula o contexto.".to_string())
            };
            done(FimMode::Shared, "no_fim_model", "Nenhum modelo de autocompletar dedicado foi escolhido: o modelo do chat atende aos dois e cada um apaga o cache do outro.".into(), code, rec)
        }
        Some(b) if free >= resident_need_mb(b, i.ctx) => done(
            FimMode::Dedicated,
            "fits_resident",
            format!("O modelo de autocompletar ({}) cabe ao lado do modelo do chat com folga (sobram cerca de {} de RAM): fica residente, com sessão e cache próprios.", gb(b as f64 / MIB), gb(free)),
            "use_dedicated",
            "Use o modelo de autocompletar dedicado (residente).".into(),
        ),
        Some(b) if free >= on_demand_need_mb(b, i.ctx) => done(
            FimMode::DedicatedOnDemand,
            "fits_on_demand",
            format!("O modelo de autocompletar ({}) cabe, mas sem folga para ficar residente (sobram cerca de {} de RAM): é carregado no primeiro pedido e descartado após 10 minutos sem uso. Enquanto carrega, o modelo do chat atende.", gb(b as f64 / MIB), gb(free)),
            "use_dedicated_on_demand",
            "Use o modelo dedicado sob demanda; para mantê-lo residente, feche outros programas ou use um modelo de chat menor.".into(),
        ),
        Some(b) => done(
            FimMode::Shared,
            "does_not_fit",
            format!("O modelo de autocompletar ({}) não cabe ao lado do modelo do chat (sobram cerca de {} de RAM): o modelo do chat atende aos dois.", gb(b as f64 / MIB), gb(free)),
            "keep_shared",
            "Escolha um modelo de código menor (0,5B a 1,5B) ou um modelo de chat menor; enquanto isso, o autocompletar usa o modelo do chat e apaga o cache da conversa.".into(),
        ),
    }
}

// ------------------------------------------------------------------ memória de vídeo

/// Nome da GPU que o motor informou em "Calculando na GPU: <nome>." (a mesma fonte que a escolha do processador, D69).
pub fn gpu_name_from_status(msg: Option<&str>) -> Option<String> {
    let rest = msg?.strip_prefix("Calculando na GPU")?;
    let name = rest.trim_start_matches(':').trim().trim_end_matches('.').trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// Memória de vídeo (MiB) pela mesma fonte do plano de GPU do motor (D69): a descoberta isolada de adaptadores mesclada com o
/// `nvidia-smi`. Com `name` (a GPU do chat) devolve a dela; sem nome, a maior GPU real. `None` se desconhecida.
pub async fn read_vram_mb(name: Option<&str>) -> Option<u64> {
    let d = crate::gpu::discover_isolated().await;
    let infos = crate::gpu::merge(crate::gpu::detect().await, d);
    match name {
        Some(n) => infos.iter().find(|g| crate::gpu::same_gpu(&g.name, n)).and_then(|g| g.vram_mb),
        None => infos.iter().filter_map(|g| g.vram_mb).max(),
    }
}

// ------------------------------------------------------------------ estado ao vivo (para a rota do plano)

/// O que o motor próprio está fazendo com o modelo FIM agora. Escrito por `NativeEngine`, lido pela rota.
#[derive(Clone, Debug, Default)]
pub struct Live {
    /// `off` (modo compartilhado), `loading`, `ready`, `idle` (sob demanda, ainda não carregado), `failed`
    pub state: &'static str,
    /// aviso em pt-BR quando caiu para o modo compartilhado (a interface o traduz por padrão de texto)
    pub note: String,
    pub model: String,
    pub plan: Option<Value>,
}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);

pub fn set_live(l: Option<Live>) {
    *LIVE.lock().unwrap_or_else(|e| e.into_inner()) = l;
}

fn live() -> Option<Live> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// `GET /v1/fim/plan`: o plano para a máquina agora (RAM e VRAM medidas), o estado real do modelo FIM e os candidatos da pasta de
/// modelos (arquivos `.gguf` com "coder" ou "fim" no nome). Os mesmos campos servem à tela do Buriti.
pub async fn plan_handler(State(st): State<Shared>) -> Response {
    let s = st.settings.read().await.clone();
    let chat_file = s.model_file(&st.paths).ok();
    let chat_name = chat_file.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let status = active(&st).await.status().await;
    let gpu_name = gpu_name_from_status(status.message.as_deref());
    let chat_on_gpu = gpu_name.is_some();
    let vram_mb = read_vram_mb(gpu_name.as_deref()).await;
    let (ram_total_mb, ram_available_mb) = crate::embeddings::measure_ram();
    let chat_bytes = chat_file.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    let fim_file = s.fim_file(&st.paths).ok().flatten();
    let fim_bytes = fim_file.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
    let base = FimInput { ram_total_mb, ram_available_mb, vram_mb, chat_on_gpu, chat_bytes, fim_bytes, ctx: s.ctx_size, enabled: s.fim_mode != "off" };
    let current = fim_plan(&base);
    let models_dir = st.paths.models.clone();
    let chat_name2 = chat_name.clone();
    let candidates = tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&models_dir) else { return out };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".gguf") || lower.contains(".mmproj") || name == chat_name2 || !(lower.contains("coder") || lower.contains("fim")) {
                continue;
            }
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            let pl = fim_plan(&FimInput { fim_bytes: Some(size), ..base });
            out.push(json!({ "name": name, "size_bytes": size, "plan": pl.to_json() }));
        }
        out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        out
    })
    .await
    .unwrap_or_default();
    let live = live().map(|l| json!({ "state": l.state, "note": l.note, "model": l.model, "plan": l.plan }));
    Json(json!({
        "ram_total_mb": ram_total_mb, "ram_available_mb": ram_available_mb, "vram_mb": vram_mb,
        "chat_model": chat_name, "chat_on_gpu": chat_on_gpu,
        "fim_model": fim_file.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()),
        "fim_mode_setting": s.fim_mode,
        "current": current.to_json(),
        "live": live,
        "reference": { "catalog_fim_model_mb": REF_FIM_CATALOG_BYTES / 1_048_576, "small_fim_model_mb": REF_FIM_SMALL_BYTES / 1_048_576 },
        "candidates": candidates,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT_SMALL: u64 = 400 * 1_048_576; // 0,5B Q4_K_M
    const CHAT_1_5B: u64 = 1126 * 1_048_576;
    const CHAT_7B: u64 = 4608 * 1_048_576;
    const FIM_05B: u64 = 400 * 1_048_576;
    const FIM_15B: u64 = 986_048_800;

    fn p(total_gb: u64, chat: u64, fim: Option<u64>) -> FimPlan {
        fim_plan(&FimInput { ram_total_mb: total_gb * 1024, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: chat, fim_bytes: fim, ctx: 2048, enabled: true })
    }

    #[test]
    fn profile_4gb_is_shared_with_a_1_5b_chat_and_dedicated_with_a_tiny_one() {
        let a = p(4, CHAT_1_5B, Some(FIM_05B));
        assert_eq!(a.mode, FimMode::Shared);
        assert_eq!(a.reason_code, "does_not_fit");
        assert_eq!(a.recommendation_code, "keep_shared");
        assert!(a.free_mb < 700, "{}", a.free_mb);
        assert_eq!(p(4, CHAT_SMALL, Some(FIM_05B)).mode, FimMode::Dedicated);
        // sem modelo FIM: honesto sobre o que fazer
        let n = p(4, CHAT_1_5B, None);
        assert_eq!((n.mode, n.reason_code, n.recommendation_code), (FimMode::Shared, "no_fim_model", "keep_shared"));
        assert!(n.recommendation.contains("tokens FIM"));
    }

    #[test]
    fn profile_8gb() {
        assert_eq!(p(8, CHAT_1_5B, Some(FIM_15B)).mode, FimMode::Dedicated);
        // com um 7B de chat sobra pouco: o modelo pequeno só cabe sob demanda ou nem isso
        let a = p(8, CHAT_7B, Some(FIM_05B));
        assert_eq!(a.mode, FimMode::Shared, "free={}", a.free_mb);
        let b = fim_plan(&FimInput { ram_total_mb: 8192, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: 4300 * 1_048_576, fim_bytes: Some(FIM_05B), ctx: 2048, enabled: true });
        assert_eq!(b.mode, FimMode::DedicatedOnDemand, "free={}", b.free_mb);
        assert_eq!(b.reason_code, "fits_on_demand");
    }

    #[test]
    fn profiles_16_and_32gb_are_dedicated_even_with_a_7b_chat() {
        for gb in [16, 32] {
            let a = p(gb, CHAT_7B, Some(FIM_15B));
            assert_eq!((a.mode, a.reason_code, a.recommendation_code), (FimMode::Dedicated, "fits_resident", "use_dedicated"), "{gb} GB");
        }
        // sem modelo FIM em máquina folgada: recomenda instalar o do catálogo
        let n = p(32, CHAT_7B, None);
        assert_eq!((n.mode, n.recommendation_code), (FimMode::Shared, "install_fim_model"));
        assert!(n.recommendation.contains("Qwen2.5-Coder 1.5B"));
        // 8 GB com chat 1,5B e sem modelo: o 1,5B cabe; num perfil mais justo recomenda o de 0,5B
        assert_eq!(p(8, CHAT_1_5B, None).recommendation_code, "install_fim_model");
        let tight = fim_plan(&FimInput { ram_total_mb: 5000, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_1_5B, fim_bytes: None, ctx: 2048, enabled: true });
        assert_eq!(tight.recommendation_code, "install_small_fim_model", "free={}", tight.free_mb);
    }

    #[test]
    fn chat_on_gpu_leaves_the_ram_to_the_fim_model_which_stays_on_the_cpu() {
        let base = FimInput { ram_total_mb: 8192, ram_available_mb: None, vram_mb: Some(12288), chat_on_gpu: true, chat_bytes: CHAT_7B, fim_bytes: Some(FIM_15B), ctx: 2048, enabled: true };
        let g = fim_plan(&base);
        assert_eq!(g.mode, FimMode::Dedicated); // com o chat na placa a RAM sobra...
        assert_eq!(p(8, CHAT_7B, Some(FIM_15B)).mode, FimMode::Shared); // ...e sem GPU não sobraria
        assert!(g.gpu_would_fit); // 12 GB de VRAM: 7B (5,3 GB) + 1,5B (1,2 GB) deixam mais de 25% livre
        assert_eq!(g.to_json()["processor"], "cpu");
        // placa justa: o FIM não vai para a GPU
        let tight = fim_plan(&FimInput { vram_mb: Some(8192), ..base }); // 8 GB: sobrariam 1,5 GB (18%), menos de 25%
        assert!(!tight.gpu_would_fit);
        // VRAM desconhecida: o chat conta como RAM (conservador) e a GPU nunca é sugerida
        let unk = fim_plan(&FimInput { vram_mb: None, ..base });
        assert_eq!(unk.mode, FimMode::Shared);
        assert!(!unk.gpu_would_fit);
    }

    #[test]
    fn disabled_and_no_model_and_measured_availability() {
        let off = fim_plan(&FimInput { ram_total_mb: 32768, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_SMALL, fim_bytes: Some(FIM_05B), ctx: 2048, enabled: false });
        assert_eq!((off.mode, off.reason_code), (FimMode::Shared, "disabled"));
        // 32 GB com a RAM quase toda em uso agora: o plano segue o que está livre de fato
        let busy = fim_plan(&FimInput { ram_total_mb: 32768, ram_available_mb: Some(900), vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_SMALL, fim_bytes: Some(FIM_05B), ctx: 2048, enabled: true });
        assert_eq!(busy.mode, FimMode::Shared, "free={}", busy.free_mb);
        // com 1200 MiB livres agora (688 MiB depois da folga) ainda cabe sob demanda
        let some = fim_plan(&FimInput { ram_available_mb: Some(1200), ..FimInput { ram_total_mb: 32768, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_SMALL, fim_bytes: Some(FIM_05B), ctx: 2048, enabled: true } });
        assert_eq!(some.mode, FimMode::DedicatedOnDemand, "free={}", some.free_mb);
    }

    #[test]
    fn json_carries_the_ui_fields() {
        let j = p(16, CHAT_7B, Some(FIM_05B)).to_json();
        for k in ["mode", "reason_code", "reason", "recommendation_code", "recommendation", "free_mb", "processor", "gpu_would_fit"] {
            assert!(j.get(k).is_some(), "falta {k}");
        }
        assert_eq!(j["mode"], "dedicated");
    }

    #[test]
    fn gpu_name_comes_from_the_engine_status() {
        assert_eq!(gpu_name_from_status(Some("Calculando na GPU: NVIDIA GeForce 940MX.")).as_deref(), Some("NVIDIA GeForce 940MX"));
        assert_eq!(gpu_name_from_status(Some("O modelo não cabe na memória da GPU; calculando na CPU.")), None);
        assert_eq!(gpu_name_from_status(None), None);
    }
}
