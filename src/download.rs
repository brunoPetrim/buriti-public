//! Downloads iniciados pela interface: modelos do catálogo e o motor (llama.cpp).
//! Tudo é verificado por SHA-256 antes de ser aceito:
//!  - modelos: hash embutido no catálogo (lido do Hugging Face);
//!  - motor: campo `digest` do asset no release oficial do GitHub (ggml-org/llama.cpp).
//! O motor é só extraído (nunca executado aqui); quem o executa é o botão "Iniciar".

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::{catalog::CatalogEntry, config::Paths, errcode::coded};

const RELEASES_URL: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8";

#[derive(Serialize, Clone)]
pub struct Progress {
    pub id: String,
    /// "model" | "engine"
    pub kind: &'static str,
    /// "running" | "done" | "error"
    pub state: &'static str,
    pub done: u64,
    pub total: u64,
    pub message: Option<String>,
}

pub struct Downloads {
    items: Mutex<HashMap<String, (Progress, Arc<AtomicBool>)>>,
    http: reqwest::Client,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl Downloads {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            items: Mutex::new(HashMap::new()),
            http: reqwest::Client::builder()
                .user_agent("buriti")
                .connect_timeout(Duration::from_secs(15))
                .build()
                .expect("cliente http"),
        })
    }

    pub fn list(&self) -> Vec<Progress> {
        self.items.lock().unwrap().values().map(|(p, _)| p.clone()).collect()
    }

    pub fn cancel(&self, id: &str) {
        if let Some((_, flag)) = self.items.lock().unwrap().get(id) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    fn begin(&self, id: &str, kind: &'static str, total: u64) -> Result<Arc<AtomicBool>, String> {
        let mut m = self.items.lock().unwrap();
        if matches!(m.get(id), Some((p, _)) if p.state == "running") {
            return Err(coded("err_in_progress", &[], "Already in progress."));
        }
        let flag = Arc::new(AtomicBool::new(false));
        m.insert(
            id.to_string(),
            (Progress { id: id.to_string(), kind, state: "running", done: 0, total, message: None }, flag.clone()),
        );
        Ok(flag)
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Progress)) {
        if let Some((p, _)) = self.items.lock().unwrap().get_mut(id) {
            f(p);
        }
    }

    fn finish(&self, id: &str, res: Result<(), String>) {
        self.update(id, |p| match res {
            Ok(()) => {
                p.state = "done";
                p.message = None;
            }
            Err(e) => {
                p.state = "error";
                p.message = Some(e);
            }
        });
    }

    // ---------------------------------------------------------------- modelos

    pub fn start_model(self: &Arc<Self>, paths: Paths, e: &'static CatalogEntry) -> Result<(), String> {
        // modelos com visão: o mmproj é baixado logo depois do modelo, com o mesmo id de progresso e a mesma conferência de SHA-256
        let mm = crate::catalog::mmproj_for(e.id);
        let cancel = self.begin(e.id, "model", e.size + mm.map_or(0, |m| m.size))?;
        let this = self.clone();
        tokio::spawn(async move {
            let dest = paths.models.join(e.file);
            let part = paths.models.join(format!("{}.part", e.file));
            let res = async {
                this.fetch(e.id, e.url, &part, e.sha256, &cancel, 0, mm.map_or(0, |m| m.size)).await?;
                if let Some(m) = mm {
                    let mm_dest = paths.models.join(crate::catalog::mmproj_file(e));
                    let mm_part = paths.models.join(format!("{}.part", crate::catalog::mmproj_file(e)));
                    let r = this.fetch(e.id, m.url, &mm_part, m.sha256, &cancel, e.size, 0).await;
                    if r.is_err() {
                        let _ = tokio::fs::remove_file(&mm_part).await;
                    }
                    r?;
                    tokio::fs::rename(&mm_part, &mm_dest).await.map_err(|x| x.to_string())?;
                }
                tokio::fs::rename(&part, &dest).await.map_err(|x| x.to_string())
            }
            .await;
            if res.is_err() {
                let _ = tokio::fs::remove_file(&part).await;
            }
            this.finish(e.id, res);
        });
        Ok(())
    }

    /// Baixa `url` para `dest` em streaming, calculando o SHA-256 no caminho.
    /// `offset`: bytes já baixados de arquivos anteriores do mesmo item; `extra`: bytes dos arquivos seguintes (só para o total).
    async fn fetch(&self, id: &str, url: &str, dest: &Path, expect_sha: &str, cancel: &AtomicBool, offset: u64, extra: u64) -> Result<(), String> {
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| coded("err_network", &[&e.to_string()], format!("network failure: {e}")))?
            .error_for_status()
            .map_err(|e| coded("err_server_error", &[&e.to_string()], format!("the server answered with an error: {e}")))?;
        if let Some(len) = resp.content_length() {
            self.update(id, |p| p.total = offset + len + extra);
        }
        let mut file = tokio::fs::File::create(dest).await.map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut stream = resp.bytes_stream();
        let mut done = 0u64;
        while let Some(chunk) = stream.next().await {
            if cancel.load(Ordering::Relaxed) {
                return Err(coded("err_cancelled", &[], "cancelled"));
            }
            let c = chunk.map_err(|e| coded("err_network", &[&e.to_string()], format!("network failure: {e}")))?;
            hasher.update(&c);
            file.write_all(&c).await.map_err(|e| e.to_string())?;
            done += c.len() as u64;
            self.update(id, |p| p.done = offset + done);
        }
        file.flush().await.map_err(|e| e.to_string())?;
        drop(file);
        let got = hex(&hasher.finalize());
        if !got.eq_ignore_ascii_case(expect_sha) {
            return Err(coded("err_sha", &[expect_sha, &got], format!("SHA-256 does not match (expected {expect_sha}, got {got}). File discarded.")));
        }
        Ok(())
    }

    // ------------------------------------------------------------------ motor

    pub fn start_engine(self: &Arc<Self>, paths: Paths, backend: String) -> Result<(), String> {
        let suffix = engine_asset_suffix(&backend)?;
        let cancel = self.begin("engine", "engine", 0)?;
        let this = self.clone();
        tokio::spawn(async move {
            let res = this.engine_job(&paths, suffix, &cancel).await;
            this.finish("engine", res);
        });
        Ok(())
    }

    async fn engine_job(&self, paths: &Paths, suffix: &'static str, cancel: &AtomicBool) -> Result<(), String> {
        // Os releases do llama.cpp são todos "pré-release": /latest não serve; listamos os recentes.
        let releases: Vec<serde_json::Value> = self
            .http
            .get(RELEASES_URL)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|e| coded("err_github", &[&e.to_string()], format!("failed to query GitHub: {e}")))?
            .error_for_status()
            .map_err(|e| coded("err_github_resp", &[&e.to_string()], format!("GitHub answered with an error: {e}")))?
            .json()
            .await
            .map_err(|e| e.to_string())?;

        let mut found: Option<(String, String, String)> = None; // (nome, url, sha256)
        'outer: for r in &releases {
            if r["draft"].as_bool() == Some(true) {
                continue;
            }
            for a in r["assets"].as_array().into_iter().flatten() {
                let name = a["name"].as_str().unwrap_or("");
                if name.starts_with("llama-") && name.ends_with(suffix) {
                    let url = a["browser_download_url"].as_str().unwrap_or("");
                    let digest = a["digest"].as_str().unwrap_or("");
                    if let (true, Some(sha)) = (!url.is_empty(), digest.strip_prefix("sha256:")) {
                        found = Some((name.to_string(), url.to_string(), sha.to_string()));
                        break 'outer;
                    }
                }
            }
        }
        let (name, url, sha) = found.ok_or_else(|| {
            coded("err_no_release", &[suffix], format!("No recent release with '{suffix}' and a SHA-256 hash was found. Download it manually from github.com/ggml-org/llama.cpp/releases"))
        })?;
        if !url.starts_with("https://github.com/ggml-org/llama.cpp/") {
            return Err(coded("err_bad_url", &[], "Unexpected download URL; aborted."));
        }

        let archive = paths.engine_dir.join(format!("{name}.part"));
        let res = self.fetch("engine", &url, &archive, &sha, cancel, 0, 0).await;
        if let Err(e) = res {
            let _ = tokio::fs::remove_file(&archive).await;
            return Err(e);
        }

        let dest = paths.engine_dir.clone();
        let arc = archive.clone();
        let is_zip = name.ends_with(".zip");
        let ex = tokio::task::spawn_blocking(move || if is_zip { extract_zip(&arc, &dest) } else { extract_tar_gz(&arc, &dest) })
            .await
            .map_err(|e| e.to_string())?;
        let _ = tokio::fs::remove_file(&archive).await;
        ex?;

        if !paths.engine_binary().is_file() {
            return Err(coded("err_no_server_pkg", &[], "Extracted, but llama-server was not found in the package."));
        }
        Ok(())
    }
}

/// Sufixo do asset do llama.cpp para este sistema. `backend`: "cpu" ou "vulkan".
fn engine_asset_suffix(backend: &str) -> Result<&'static str, String> {
    let gpu = match backend {
        "cpu" => false,
        "vulkan" => true,
        _ => return Err(coded("err_bad_backend", &[], "invalid backend")),
    };
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    Ok(match (os, arch, gpu) {
        ("windows", "x86_64", false) => "bin-win-cpu-x64.zip",
        ("windows", "x86_64", true) => "bin-win-vulkan-x64.zip",
        ("windows", "aarch64", false) => "bin-win-cpu-arm64.zip",
        ("windows", "aarch64", true) => "bin-win-vulkan-arm64.zip",
        ("linux", "x86_64", false) => "bin-ubuntu-x64.tar.gz",
        ("linux", "x86_64", true) => "bin-ubuntu-vulkan-x64.tar.gz",
        ("linux", "aarch64", false) => "bin-ubuntu-arm64.tar.gz",
        ("linux", "aarch64", true) => "bin-ubuntu-vulkan-arm64.tar.gz",
        ("macos", "aarch64", _) => "bin-macos-arm64.tar.gz", // Metal já incluso
        ("macos", "x86_64", _) => "bin-macos-x64.tar.gz",
        _ => return Err(coded("err_no_auto_dl", &[&format!("{os}/{arch}")], format!("No automatic download for {os}/{arch}. Download it manually."))),
    })
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
    for i in 0..z.len() {
        let mut entry = z.by_index(i).map_err(|e| e.to_string())?;
        // enclosed_name rejeita caminhos com ".." / absolutos (zip-slip)
        let Some(rel) = entry.enclosed_name() else { continue };
        let out: PathBuf = dest.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let mut o = std::fs::File::create(&out).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut o).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode));
        }
    }
    Ok(())
}

fn extract_tar_gz(archive: &Path, dest: &Path) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    tar::Archive::new(flate2::read::GzDecoder::new(f)).unpack(dest).map_err(|e| e.to_string())
}
