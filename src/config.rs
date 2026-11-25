use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Configurações editáveis pela interface web.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Nome do arquivo .gguf dentro da pasta de modelos ("" = nenhum).
    pub model: String,
    /// Caminho absoluto de um .gguf em qualquer lugar do disco ("" = usar `model`).
    /// O arquivo é usado onde está: nunca é enviado nem copiado.
    pub model_path: String,
    /// Camadas descarregadas na GPU. 0 = só CPU/RAM.
    pub gpu_layers: u32,
    /// Threads de CPU. 0 = automático.
    pub threads: u32,
    /// Tamanho do contexto em tokens.
    pub ctx_size: u32,
    /// Porta interna do motor llama.cpp (só 127.0.0.1).
    pub engine_port: u16,
    /// Qual motor usar: "llama" (llama.cpp baixado) ou "native" (motor próprio do Buriti).
    pub engine_kind: String,
    /// Motor próprio: onde calcular. "auto" (a GPU dedicada, se o modelo couber nela), "cpu" ou "gpu" (D69).
    pub processor: String,
    /// Motor próprio: qual adaptador de GPU usar. "best" = o melhor; ou o nome, o índice ou "backend:vulkan" como em `buriti --gpu-info`.
    pub gpu_device: String,
    /// Modelo dedicado de autocompletar (FIM, D78): nome de um .gguf da pasta models ("" = nenhum). Só no motor próprio.
    pub fim_model: String,
    /// Como `model_path`, para o modelo FIM: caminho absoluto de um .gguf em qualquer lugar ("" = usar `fim_model`).
    pub fim_model_path: String,
    /// "auto" (padrão: o plano de memória decide entre compartilhado, residente e sob demanda) ou "off" (um modelo só).
    pub fim_mode: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: String::new(),
            model_path: String::new(),
            gpu_layers: 0,
            threads: 0,
            ctx_size: 2048,
            engine_port: 8081,
            engine_kind: "llama".into(),
            processor: "auto".into(),
            gpu_device: "best".into(),
            fim_model: String::new(),
            fim_model_path: String::new(),
            fim_mode: "auto".into(),
        }
    }
}

impl Settings {
    /// Valida e normaliza valores vindos da web (nunca confiar no cliente).
    pub fn sanitize(mut self) -> Result<Self, String> {
        if !self.model.is_empty() && !valid_model_name(&self.model) {
            return Err(crate::errcode::coded("err_model_name", &[], "Invalid model name (use a .gguf file from the models folder)."));
        }
        self.model_path = self.model_path.trim().trim_matches('"').to_string();
        if !self.model_path.is_empty() {
            let p = Path::new(&self.model_path);
            if self.model_path.contains('\0')
                || !p.is_absolute()
                || !self.model_path.to_ascii_lowercase().ends_with(".gguf")
            {
                return Err(crate::errcode::coded("err_model_path", &[], "Invalid path: use the full path of a .gguf file."));
            }
        }
        self.gpu_layers = self.gpu_layers.min(999);
        self.threads = self.threads.min(256);
        self.ctx_size = self.ctx_size.clamp(256, 32768);
        if self.engine_port < 1024 {
            return Err(crate::errcode::coded("err_port_min", &[], "The engine port must be >= 1024."));
        }
        if self.engine_kind != "llama" && self.engine_kind != "native" {
            return Err(crate::errcode::coded("err_engine_kind", &[], "Invalid engine (use \"llama\" or \"native\")."));
        }
        // processador e dispositivo (D69): valores desconhecidos viram o padrão em vez de erro (a interface só oferece os válidos)
        self.processor = self.processor.trim().to_ascii_lowercase();
        if !["auto", "cpu", "gpu"].contains(&self.processor.as_str()) {
            self.processor = "auto".into();
        }
        self.gpu_device = self.gpu_device.chars().filter(|c| !c.is_control()).collect::<String>().trim().chars().take(100).collect();
        if self.gpu_device.is_empty() {
            self.gpu_device = "best".into();
        }
        // modelo FIM (D78): validado como o modelo do chat
        if !self.fim_model.is_empty() && !valid_model_name(&self.fim_model) {
            return Err(crate::errcode::coded("err_fim_name", &[], "Invalid autocomplete model name (use a .gguf file from the models folder)."));
        }
        self.fim_model_path = self.fim_model_path.trim().trim_matches('"').to_string();
        if !self.fim_model_path.is_empty() {
            let p = Path::new(&self.fim_model_path);
            if self.fim_model_path.contains('\0') || !p.is_absolute() || !self.fim_model_path.to_ascii_lowercase().ends_with(".gguf") {
                return Err(crate::errcode::coded("err_fim_path", &[], "Invalid autocomplete model path: use the full path of a .gguf file."));
            }
        }
        self.fim_mode = self.fim_mode.trim().to_ascii_lowercase();
        if self.fim_mode != "auto" && self.fim_mode != "off" {
            self.fim_mode = "auto".into();
        }
        Ok(self)
    }

    /// Arquivo .gguf do modelo FIM, se escolhido (caminho livre tem prioridade). `Ok(None)` = nenhum escolhido.
    pub fn fim_file(&self, paths: &Paths) -> Result<Option<PathBuf>, String> {
        let p = if !self.fim_model_path.is_empty() {
            PathBuf::from(&self.fim_model_path)
        } else if !self.fim_model.is_empty() {
            paths.models.join(&self.fim_model)
        } else {
            return Ok(None);
        };
        if !p.is_file() {
            return Err(crate::errcode::coded("err_fim_missing", &[&p.display().to_string()], format!("Autocomplete model not found: {}", p.display())));
        }
        if !is_gguf(&p) {
            return Err(crate::errcode::coded("err_fim_not_gguf", &[&p.display().to_string()], format!("The autocomplete model file is not a valid GGUF: {}", p.display())));
        }
        Ok(Some(p))
    }

    pub fn has_model(&self) -> bool {
        !self.model.is_empty() || !self.model_path.is_empty()
    }

    /// Arquivo .gguf escolhido (caminho livre tem prioridade sobre o da pasta models/).
    pub fn model_file(&self, paths: &Paths) -> Result<PathBuf, String> {
        let p = if !self.model_path.is_empty() {
            PathBuf::from(&self.model_path)
        } else if !self.model.is_empty() {
            paths.models.join(&self.model)
        } else {
            return Err(crate::errcode::coded("err_no_model", &[], "No model selected."));
        };
        if !p.is_file() {
            return Err(crate::errcode::coded("err_model_missing", &[&p.display().to_string()], format!("Model not found: {}", p.display())));
        }
        if !is_gguf(&p) {
            return Err(crate::errcode::coded("err_not_gguf", &[&p.display().to_string()], format!("The file is not a valid GGUF: {}", p.display())));
        }
        Ok(p)
    }
}

/// Porta padrão da interface web e da API `/v1`. Escolhida por ser pouco usada por ferramentas de
/// desenvolvimento (a 8080 e a 8081 são de Jenkins, Tomcat, proxies, servidores de desenvolvimento...).
pub const DEFAULT_WEB_PORT: u16 = 8421;
/// Quantas portas seguintes tentar quando a preferida está ocupada, antes de pedir uma qualquer ao sistema.
const PORT_SCAN: u16 = 50;

/// Abre o `TcpListener` da interface web em 127.0.0.1: tenta `preferred` e as `PORT_SCAN - 1` seguintes;
/// se todas estiverem ocupadas, deixa o sistema escolher uma livre. Devolve o listener e a porta real.
pub async fn bind_from(preferred: u16) -> Result<(tokio::net::TcpListener, u16), String> {
    let last = preferred.saturating_add(PORT_SCAN - 1);
    for p in preferred..=last {
        if let Ok(l) = tokio::net::TcpListener::bind(("127.0.0.1", p)).await {
            return Ok((l, p));
        }
    }
    let l = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.map_err(|e| crate::errcode::coded("err_no_port", &[&e.to_string()], format!("could not open any local port: {e}")))?;
    let p = l.local_addr().map_err(|e| e.to_string())?.port();
    Ok((l, p))
}

/// Porta livre para um processo filho (o llama-server): a `start` se estiver livre, senão a próxima
/// livre, senão uma qualquer. Só a sondagem: há uma janela mínima entre testar e o filho abrir a porta.
pub fn free_local_port(start: u16) -> u16 {
    for p in start..=start.saturating_add(PORT_SCAN - 1) {
        if std::net::TcpListener::bind(("127.0.0.1", p)).is_ok() {
            return p;
        }
    }
    std::net::TcpListener::bind(("127.0.0.1", 0)).and_then(|l| l.local_addr()).map(|a| a.port()).unwrap_or(start)
}

/// Nome de arquivo .gguf simples (sem diretórios).
pub fn valid_model_name(m: &str) -> bool {
    !m.is_empty()
        && !m.contains('/')
        && !m.contains('\\')
        && !m.contains("..")
        && !m.contains('\0')
        && m.to_ascii_lowercase().ends_with(".gguf")
}

/// Confere o "número mágico" GGUF nos 4 primeiros bytes.
pub fn is_gguf(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == b"GGUF")
        .unwrap_or(false)
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub home: PathBuf,
    pub models: PathBuf,
    pub engine_dir: PathBuf,
    pub history: PathBuf,
    pub settings_file: PathBuf,
}

impl Paths {
    /// Dados ficam ao lado do executável (`<pasta do exe>/buriti-data`),
    /// a menos que BURITI_HOME aponte para outro lugar.
    pub fn from_env() -> Self {
        let home = std::env::var_os("BURITI_HOME").map(PathBuf::from).unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."))
                .join("buriti-data")
        });
        Self {
            models: home.join("models"),
            engine_dir: home.join("engine"),
            history: home.join("history"),
            settings_file: home.join("settings.json"),
            home,
        }
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.models)?;
        std::fs::create_dir_all(&self.engine_dir)?;
        std::fs::create_dir_all(&self.history)
    }

    /// A chave do histórico NÃO fica na pasta do executável, e sim no perfil do usuário
    /// (copiar/compartilhar a pasta do Buriti não leva a chave junto).
    pub fn key_file(&self) -> PathBuf {
        if let Some(p) = std::env::var_os("BURITI_KEY_FILE") {
            return PathBuf::from(p);
        }
        let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
        let base = if cfg!(windows) {
            env("APPDATA")
        } else if cfg!(target_os = "macos") {
            env("HOME").map(|h| h.join("Library").join("Application Support"))
        } else {
            env("XDG_CONFIG_HOME").or_else(|| env("HOME").map(|h| h.join(".config")))
        };
        base.unwrap_or_else(|| self.home.clone()).join("Buriti").join("key.bin")
    }

    /// Última porta web usada (os editores guardam a URL da API: a porta não deve mudar à toa).
    pub fn port_file(&self) -> PathBuf {
        self.home.join("port.txt")
    }

    /// Escolhe e abre a porta da interface web. `BURITI_PORT` manda (e falha com mensagem clara se estiver
    /// ocupada); sem ela, usa a última porta salva (ou o padrão) e, se estiver ocupada, a próxima livre.
    /// Devolve também um aviso a mostrar quando a porta preferida estava ocupada.
    pub async fn bind_web(&self) -> Result<(tokio::net::TcpListener, u16, Option<String>), String> {
        if let Ok(v) = std::env::var("BURITI_PORT") {
            let port: u16 = v.trim().parse().map_err(|_| format!("BURITI_PORT inválida: \"{v}\" (use um número de 1024 a 65535)"))?;
            let l = tokio::net::TcpListener::bind(("127.0.0.1", port))
                .await
                .map_err(|e| format!("A porta {port} (BURITI_PORT) não está disponível: {e}. Escolha outra ou remova a variável."))?;
            return Ok((l, port, None));
        }
        let saved = std::fs::read_to_string(self.port_file()).ok().and_then(|s| s.trim().parse::<u16>().ok()).filter(|p| *p >= 1024);
        let preferred = saved.unwrap_or(DEFAULT_WEB_PORT);
        let (l, port) = bind_from(preferred).await?;
        let note = (port != preferred).then(|| format!("A porta {preferred} está ocupada; o Buriti usa a {port}. Se algum editor estava configurado com a URL antiga da API, atualize-a."));
        if saved != Some(port) {
            let _ = std::fs::write(self.port_file(), port.to_string());
        }
        Ok((l, port, note))
    }

    /// Chave da API compatível com a OpenAI (`/v1`), no mesmo lugar da chave do histórico: no perfil do
    /// usuário, não ao lado do executável.
    pub fn api_key_file(&self) -> PathBuf {
        self.key_file().with_file_name("api-key.txt")
    }

    /// Lê a chave da API ou cria uma nova (32 bytes aleatórios em hexadecimal, prefixo `buriti-`).
    pub fn load_or_create_api_key(&self) -> Result<String, String> {
        let file = self.api_key_file();
        if let Ok(s) = std::fs::read_to_string(&file) {
            let s = s.trim().to_string();
            if s.len() >= 32 && s.is_ascii() {
                return Ok(s);
            }
        }
        self.new_api_key()
    }

    /// Gera e grava uma chave nova (a antiga deixa de valer).
    pub fn new_api_key(&self) -> Result<String, String> {
        let mut raw = [0u8; 32];
        getrandom::fill(&mut raw).map_err(|e| format!("sem fonte de aleatoriedade: {e}"))?;
        let key = format!("buriti-{}", raw.iter().map(|b| format!("{b:02x}")).collect::<String>());
        let file = self.api_key_file();
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("pasta da chave: {e}"))?;
        }
        std::fs::write(&file, &key).map_err(|e| crate::errcode::coded("err_key_write", &[&e.to_string()], format!("could not write the API key: {e}")))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600));
        }
        Ok(key)
    }

    /// Caminho do binário do motor. NÃO é editável pela web (evita que a API execute
    /// binários arbitrários); só via variável BURITI_ENGINE.
    pub fn engine_binary(&self) -> PathBuf {
        if let Some(p) = std::env::var_os("BURITI_ENGINE") {
            return PathBuf::from(p);
        }
        let name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
        let direct = self.engine_dir.join(name);
        if direct.is_file() {
            return direct;
        }
        find_file(&self.engine_dir, name, 3).unwrap_or(direct)
    }

    pub fn load_settings(&self) -> Settings {
        std::fs::read_to_string(&self.settings_file)
            .ok()
            .and_then(|s| serde_json::from_str::<Settings>(&s).ok())
            .and_then(|s| s.sanitize().ok())
            .unwrap_or_default()
    }

    pub fn save_settings(&self, s: &Settings) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(s).map_err(std::io::Error::other)?;
        std::fs::write(&self.settings_file, json)
    }
}

fn find_file(dir: &Path, name: &str, depth: u8) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let mut subdirs = Vec::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_file() && e.file_name().to_string_lossy() == name {
            return Some(p);
        }
        if p.is_dir() {
            subdirs.push(p);
        }
    }
    subdirs.into_iter().find_map(|d| find_file(&d, name, depth - 1))
}

#[cfg(test)]
mod port_tests {
    use super::*;

    #[tokio::test]
    async fn skips_busy_port() {
        // ocupa uma porta qualquer; pedir exatamente essa tem de devolver OUTRA, já aberta e usável
        let busy = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let busy_port = busy.local_addr().unwrap().port();
        let (l, p) = bind_from(busy_port).await.unwrap();
        assert_ne!(p, busy_port);
        assert_eq!(l.local_addr().unwrap().port(), p);
        // libera a ocupada: abrir de novo a partir dela funciona (não exige que seja a mesma, outros testes usam portas)
        drop(l);
        drop(busy);
        assert!(bind_from(busy_port).await.is_ok());
    }

    #[test]
    fn free_port_skips_busy_port() {
        let busy = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let busy_port = busy.local_addr().unwrap().port();
        let p = free_local_port(busy_port);
        assert_ne!(p, busy_port);
        assert!(p >= 1024);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_and_device_are_normalized() {
        let ok = |p: &str, d: &str| Settings { processor: p.into(), gpu_device: d.into(), ..Settings::default() }.sanitize().unwrap();
        assert_eq!((ok("GPU", " backend:vulkan ").processor, ok("GPU", " backend:vulkan ").gpu_device), ("gpu".to_string(), "backend:vulkan".to_string()));
        assert_eq!(ok("cpu", "best").processor, "cpu");
        assert_eq!(ok("quantum", "best").processor, "auto", "an unknown processor falls back to auto, not to an error");
        assert_eq!(ok("auto", "   ").gpu_device, "best");
        assert_eq!(ok("auto", "NVIDIA\u{0}\nGeForce").gpu_device, "NVIDIAGeForce", "control characters are dropped");
        assert_eq!(ok("auto", &"x".repeat(500)).gpu_device.len(), 100);
        let d = Settings::default();
        assert_eq!((d.processor.as_str(), d.gpu_device.as_str()), ("auto", "best"));
    }

    #[test]
    fn old_settings_files_without_the_new_fields_still_load() {
        let s: Settings = serde_json::from_str(r#"{"model":"a.gguf","engine_kind":"native"}"#).unwrap();
        assert_eq!((s.processor.as_str(), s.gpu_device.as_str()), ("auto", "best"));
        assert_eq!((s.fim_model.as_str(), s.fim_model_path.as_str(), s.fim_mode.as_str()), ("", "", "auto"));
    }

    #[test]
    fn fim_settings_are_validated_like_model_paths() {
        let ok = |m: &str, p: &str, mode: &str| Settings { fim_model: m.into(), fim_model_path: p.into(), fim_mode: mode.into(), ..Settings::default() }.sanitize();
        assert_eq!(ok("", "", "OFF").unwrap().fim_mode, "off");
        assert_eq!(ok("", "", "turbo").unwrap().fim_mode, "auto", "unknown mode falls back to auto");
        assert!(ok("../x.gguf", "", "auto").is_err());
        assert!(ok("x.bin", "", "auto").is_err());
        assert!(ok("", "relative/x.gguf", "auto").is_err());
        assert!(ok("", "C:\\m\\x.txt", "auto").is_err());
        assert!(ok("coder.gguf", "", "auto").is_ok());
        let paths = Paths::from_env();
        let s = Settings::default();
        assert_eq!(s.fim_file(&paths), Ok(None));
        let missing = Settings { fim_model: "nao-existe-fim.gguf".into(), ..Settings::default() };
        assert!(missing.fim_file(&paths).is_err());
    }
}
