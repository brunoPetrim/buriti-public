# Buriti

A **lightweight, friendly local runner for AI models**, built to be accessible and to adapt to different computing, financial and social realities. Its goal is to put AI models within reach of everyone, especially the people who are usually left out for lack of resources: it runs open-source LLMs (GGUF files such as Qwen, Llama, Gemma, Mistral and Phi) on modest machines, with a clean web interface served by a small local server written in Rust, and it adapts to the memory and the processor you have.

[Ollama](https://ollama.com) was the inspiration and a guide for the start of the project.

You pick the engine that runs the model:

- **llama.cpp** (`llama-server`), downloaded for you from the UI with a SHA-256 check: the fastest and the most compatible option.
- **Buriti's own engine**, written from scratch in Rust and built into the executable: no download, no extra process, and it matches llama.cpp's speed on several models (see [Performance](#performance)).

Everything is configured from the browser, in **34 languages**. Your chats are stored **encrypted** on your machine, and nothing leaves it unless you click a download button.

> **Status:** working and in active development. Tested on Windows 10 x64. Linux and macOS builds are expected to work but have **not been tested yet**. No pre-built binaries are published yet (see [Building](#building-from-source)).

## Features

- Chat UI with a ChatGPT-style sidebar of past conversations (encrypted at rest).
- **Model catalog** with provider, release date, size, license, strengths and "best for" notes, plus a "recommended for your PC" hint. Or point Buriti at any `.gguf` file already on your disk. Files are used where they are: **nothing is uploaded or copied**.
- Step-by-step first-run guide: choose an engine, choose a model, start chatting.
- Settings in the UI: engine, model, CPU threads, context size, GPU layers, port.
- **Attachments:** drag, paste or pick text, code and PDF files in the chat. Files are read on your computer (PDF text is extracted by an isolated child process of Buriti itself) and are never uploaded or saved in the history; only the file name, size and type are kept. Scanned PDFs (no OCR) and password-protected PDFs are refused with a clear message.
- **Buriti Code:** an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions` with fill-in-the-middle, `/v1/models`) on `127.0.0.1`, so editors such as VS Code and JetBrains IDEs can use the loaded model for chat and autocomplete. See [Buriti Code](#buriti-code-api-for-code-editors).
- Light and dark themes (light is the default).
- Built-in tutorial and warnings for the usual pain points (weak PC, low RAM, SmartScreen, firewall, losing the history key).
- One executable: the web UI is embedded in the binary.

## Models and engines

The catalog ships these models (all Q4_K_M, with verified SHA-256):

| Model | Provider | Approx. size |
|---|---|---|
| Qwen 2.5 0.5B / 1.5B / 3B Instruct | Alibaba Cloud (Qwen) | 0.5 / 1.1 / 2.1 GB |
| Qwen 3 0.6B (with a "thinking" mode) | Alibaba Cloud (Qwen) | 0.4 GB |
| Gemma 3 1B Instruct | Google DeepMind | 0.8 GB |
| Llama 3.2 1B / 3B Instruct | Meta | 0.8 / 2.0 GB |
| Phi-3.1 mini 4k Instruct | Microsoft | 2.4 GB |
| Phi-3.5 mini Instruct | Microsoft | 2.4 GB |
| Phi-4-mini Instruct | Microsoft | 2.5 GB |
| Mistral 7B Instruct v0.3 | Mistral AI | 4.4 GB |

Each model has its own license, separate from Buriti's (see [License](#license)). Check it before commercial use: the Qwen 2.5 3B card is marked **non-commercial** (Qwen Research License) and Gemma has its own terms.

**Buriti's own engine** currently runs these architectures, each checked against llama.cpp: **Qwen 2 / 2.5, Qwen 3 (dense), Llama 3.x (validated with 3.2), Gemma 3 (text only), Mistral 7B Instruct v0.3 and Phi-3 mini 4k**. Qwen, Llama 3.2 and Gemma 3 were checked against an unquantized (F16) reference; Mistral 7B and Phi-3 only with quantized files, so their evidence is weaker. Supported weight formats: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `F16`, `BF16`, `F32`. It runs on the **CPU** (AVX2/F16C on x86-64, NEON on ARM64) and, for Qwen 2 / 2.5 so far, optionally on the **GPU** through `wgpu` (Vulkan, DX12 or Metal): Settings > Processor chooses Automatic, CPU only or GPU. Automatic uses a dedicated GPU only when the model fits in its memory; on the one low-end GPU tested the prompt is read several times faster and the answer is generated at about the CPU speed. The GPU path is new and was tested on one machine.

Not supported by the own engine yet (use the llama.cpp engine instead): `Q2_K`, `Q3_K` and the `IQ*` formats, Gemma 1 and 2, Mistral v0.1/v0.2 and Llama 2, vision models, mixture-of-experts models. Phi-3.5 mini and Phi-4-mini are supported (validated by tokens and logits against llama.cpp).

**GPU:** the llama.cpp engine can be downloaded as a Vulkan build from the setup card (Windows and Linux) and the number of GPU layers is a setting. This path has **not been tested on a real GPU by the maintainer**.

## Quick start

1. [Build](#building-from-source) the program (or run a binary if you have one) and start it:

   ```
   buriti
   ```

2. Your browser opens at `http://127.0.0.1:8421/#token=...` (the console prints the same URL). The token changes on every start; if a tab stops working after a restart, open the new URL.
3. Follow the **"Let's get started"** card:
   - **Engine:** download llama.cpp (verified with SHA-256) or choose Buriti's own engine.
   - **Model:** pick one from the catalog (the recommended one is highlighted) or choose a `.gguf` already on your disk.
   - **Start** the engine and chat.
4. Next time the engine starts by itself.

**Where things are stored**

- Data lives in `buriti-data/` **next to the executable** (override with `BURITI_HOME`): `models/`, `engine/`, `history/`, `settings.json`, `engine.log`.
- The **history encryption key** is kept in your user profile, not next to the chats, so copying the Buriti folder does not copy the key: `%APPDATA%\Buriti\key.bin` (Windows), `~/.config/Buriti/key.bin` (Linux), `~/Library/Application Support/Buriti/key.bin` (macOS). **If you lose this file, old conversations cannot be recovered.**

**Tips**

- Weak PC? Start with Qwen 2.5 0.5B/1.5B, Qwen 3 0.6B or Gemma 3 1B. Models above ~3B are slow on CPUs with few cores, and Mistral 7B needs at least 8 GB of RAM.
- The default context size is 2048 tokens (a setting); larger contexts use more RAM.

## Performance

Speed of Buriti's own engine **relative to llama.cpp** (build b12606, same model, same 4 threads, CPU only, interleaved runs, 3 rounds, medians, a prompt of about 560 tokens), on an Intel Core i7-7500U (2 cores / 4 threads, AVX2 only), Q4_K_M unless noted, development build, measured on 2026-11-24:

| Model | Prefill tok/s (llama.cpp / Buriti) | Prefill ratio | Generation tok/s (llama.cpp / Buriti) | Generation ratio |
|---|---|---|---|---|
| Qwen 2.5 0.5B | 88.2 / 162.1 | 184% | 30.8 / 33.6 | 109% |
| Qwen3 0.6B | 143.3 / 144.6 | 101% | 33.2 / 30.4 | 91% |
| Gemma 3 1B | 48.6 / 87.5 | 180% | 16.3 / 16.8 | 103% |
| Llama 3.2 1B | 79.2 / 91.4 | 115% | 17.4 / 17.4 | 100% |
| Qwen 2.5 1.5B | 57.2 / 68.6 | 120% | 13.7 / 13.8 | 101% |
| Qwen2.5-Coder 1.5B (Q8_0) | 31.8 / 55.2 | 174% | 8.85 / 8.77 | 99% |
| StarCoder2 3B | 26.0 / 32.0 | 123% | 7.36 / 7.46 | 101% |
| Llama 3.2 3B | 27.0 / 33.1 | 122% | 7.05 / 6.89 | 98% |
| Phi-4-mini | 23.5 / 25.5 | 109% | 5.62 / 5.58 | 99% |
| Qwen3 4B | 20.1 / 25.0 | 124% | 5.42 / 5.53 | 102% |
| Mistral 7B v0.3 | 10.3 / 14.0 | 135% | 3.23 / 3.29 | 102% |
| Qwen2.5-Coder 7B | 11.5 / 15.5 | 135% | 3.23 / 3.25 | 101% |

- **Prefill** (prompt processing) is at or above llama.cpp on all 12 models (101 to 184%). **Generation** is within 91 to 109%: close, but not ahead.
- The weights of the Q4_K, Q6_K, Q8_0 and Q5_0 formats are repacked into interleaved rows when the model loads, which is where the prefill gains come from.
- **GPU** (own engine, wgpu): on an NVIDIA GeForce 940MX (4 GiB, Vulkan) with Qwen 2.5 0.5B Q4_K_M, 202 tok/s of prefill against 167 tok/s for the llama.cpp Vulkan build (121%), and 22.1 tok/s of generation against 24.4 (91%). One GPU and one model: indicative only.
- Phi-3.5 mini is supported but is not in the table: the measurements of its Q5_K path were too noisy to publish.
- **Not measured:** other CPUs (no AVX-VNNI or AVX-512 here), ARM64 speed, long contexts, Ollama's own overhead, and, beyond a small sample, the quality of the repacked Q8_0 and Q5_0 paths against llama.cpp. The measuring laptop throttles when hot, so every comparison is interleaved and run on a rested machine.

## Building from source

**Requirements:** Rust (stable, via [rustup](https://rustup.rs)) and a linker/C toolchain for your OS. There are no system libraries to install: TLS uses `rustls` (pure Rust).

```bash
git clone <this repository>
cd buriti
cargo build --release      # optimized binary with LTO: target/release/buriti
```

- **Windows:** either the MSVC build tools, or the GNU toolchain with MinGW-w64 (no Visual Studio needed), which is what the maintainer uses:
  ```powershell
  rustup default stable-x86_64-pc-windows-gnu
  winget install BrechtSanders.WinLibs.POSIX.UCRT   # then add its mingw64\bin folder to PATH
  cargo build --release
  ```
  The executable icon (`web/images/buriti_favicon.ico`) is embedded at build time by `build.rs` (it needs `windres` from MinGW, or `rc.exe` with MSVC). If the icon step fails the build still succeeds, just without the icon.
- **Linux:** a C linker is enough (`build-essential` on Debian/Ubuntu). No OpenSSL or other system libraries are needed.
- **macOS:** Xcode command line tools.

The first release build takes a few minutes (about 5.5 minutes on a 2-core laptop, because of LTO) and produces a **~5 MB executable** that runs from any folder: the web UI and images are embedded in it.

`cargo build` (without `--release`) is also optimized (`opt-level = 3`), because the engine is about 20 times slower without optimization. It builds faster; release adds LTO and stripping.

### Running

```bash
cargo run --release
# or run the binary directly:
./target/release/buriti          # Linux / macOS
.\target\release\buriti.exe      # Windows
```

Stop it with `Ctrl+C` (the engines are stopped too).

### Configuration

Everything day-to-day is in the UI. These environment variables are for deployment and development:

| Variable | Effect |
|---|---|
| `BURITI_HOME` | Data folder (default: `buriti-data` next to the executable). |
| `BURITI_PORT` | Force the web UI / API port (default `8421`, fails with a clear message if it is busy). Without it, Buriti reuses the last port it used (saved in `port.txt` in the data folder) and, if that one is busy, takes the next free port and says so in the console. The internal llama.cpp port (`8081` by default, editable in Settings) also moves to a free one automatically. |
| `BURITI_ENGINE` | Path to a `llama-server` binary to use instead of the downloaded one. **Deliberately not editable from the web UI.** |
| `BURITI_KEY_FILE` | Path of the history encryption key. |
| `BURITI_NO_BROWSER` | If set, do not open the browser on start. |
| `BURITI_DIAG` | If set (`1`), the own engine prints timing per weight format and per phase to stderr. |

PowerShell: `$env:BURITI_PORT = "9000"; .\buriti.exe`. Bash: `BURITI_PORT=9000 ./buriti`.

## Development

```
Cargo.toml, build.rs, buriti.rc     package, icon embedding
src/main.rs                         entry point: token, auto-start engine, open browser
src/server.rs                       /api routes, security (token + Host check), file browser
src/config.rs                       settings (editable from the web UI) and paths
src/history.rs                      encrypted history (ChaCha20-Poly1305)
src/catalog.rs                      model catalog (metadata + SHA-256)
src/download.rs                     verified downloads (models and the llama.cpp engine)
src/gpu.rs                          best-effort GPU detection
src/engine/                         `Engine` trait + the llama.cpp engine (child process)
src/native/                         Buriti's own engine:
    gguf.rs                           GGUF reader (mmap, bounds-checked)
    tokenizer.rs                      BPE and SentencePiece tokenizers, chat templates
    model.rs                          transformer forward pass (single token and batched prefill)
    kernels.rs, neon.rs, vmath.rs     int8 dot-product kernels (AVX2 / NEON), exp, attention
    quant.rs, f16.rs                  dequantization reference, f16 helpers (KV cache)
    mod.rs                            engine glue, sampling, streaming
web/                                UI (HTML/JS, 34 languages), embedded in the binary
tools/                              translation checks (Node.js)
```

### Tests

```bash
cargo test            # 30 unit tests plus 7 ignored diagnostics
```

The tests compare every kernel with a float reference, check that the batched prefill gives **bit-identical** results to token-by-token processing, and cover the tokenizers. One test uses a small model if available (`BURITI_TEST_MODEL`, default `buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`) and skips itself otherwise.

Diagnostics (ignored by default), for example:

```bash
cargo test bench_kernels    -- --ignored --nocapture    # kernel micro-benchmark
cargo test bench_kernels_mt -- --ignored --nocapture    # sustained throughput with 1-4 threads
BURITI_TEST_MODEL=model.gguf cargo test dump_gguf -- --ignored --nocapture   # inspect a GGUF
```

### Contributing

- UI text goes in `web/i18n*.js` in **all** 34 languages (a missing key falls back to English). Translations are AI-generated and need review by native speakers, especially Yoruba, Amharic, Oromo, Igbo, Shona, Hausa, Zulu, Khmer, Papiamento and Haitian Creole.

## Buriti Code (API for code editors)

Buriti serves an OpenAI-compatible API at `http://127.0.0.1:8421/v1` (the port is the one printed at start-up; see `BURITI_PORT`), loopback only. Start the engine with a model first; requests made while it is not `ready` get HTTP 409.

- **Key:** a stable API key is created on first run and stored next to the history key (`%APPDATA%\Buriti\api-key.txt` on Windows). The web UI shows it on the **API** page, where you can also regenerate it. Send it as `Authorization: Bearer <key>`. The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`.
- **Endpoints:** `GET /v1/models`; `POST /v1/chat/completions` and `POST /v1/completions` (`stream` or not, `temperature`, `max_tokens`, `stop`, and the sampling options `top_p`, `top_k`, `min_p`, `repeat_penalty`, `frequency_penalty`, `presence_penalty` and `seed`; the same seed, prompt and options give the same text on the own engine). `/v1/completions` takes an optional `suffix`: with it, the model fills the gap between `prompt` (code before the cursor) and `suffix` (code after), which is how inline autocomplete works.
- **Token usage:** non-streaming responses always carry `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`; `prompt_tokens` counts the whole prompt, including the part the own engine reuses from its KV cache). Streaming responses carry it only when the request sets `"stream_options": {"include_usage": true}`, as one extra final chunk with `choices: []`, just before `data: [DONE]`; without that option no `usage` is sent. With llama.cpp the counts are the ones `llama-server` reports (on `/infill`, its `tokens_evaluated` and `tokens_predicted`).
- **Tool calling (`tools`), own engine only:** `tools` (type `function`), `tool_choice` (`auto`, `none`, `required` or a named function), assistant `tool_calls` and `role: "tool"` messages for multi-turn, streaming or not. Buriti writes the prompt exactly as the model's own chat template does (Qwen2.5/Qwen3 and Llama 3.1+/3.2 are implemented) and reads the answer back into `tool_calls`. **There is no guarantee of valid JSON** (no constrained decoding): when the model produces a malformed call, an unknown function or non-object arguments, the raw text comes back as `content` with `finish_reason` unchanged and a top-level `buriti_tool_call_error` field, never an invented call. `tool_choice` `required`/named is a hint (named only offers that one function to the model); a text-only answer gets the same flag. In streaming, plain text flows live and a call is emitted whole in one chunk at the end. `GET /v1/models` reports `buriti_tools` for the loaded model (format, or why it is not supported and what to use). Models without tools in their template (Gemma, Phi, StarCoder2, Llama 3.0, Mistral for now) and the llama.cpp engine get HTTP 400 with the recommendation. **Quality:** tiny models fail often (Qwen2.5 0.5B and 1.5B got a valid call with the right function and argument in 5 of 5 easy single-call prompts, which says nothing about similar tools, nested schemas or several calls); for agents use Qwen2.5/Qwen3 or Llama 3.1/3.2 at 7B/8B or more (expected, not measured here).
- **Constrained JSON output (`response_format`), own engine only:** `{"type": "json_object"}` (any JSON object) or `{"type": "json_schema", "json_schema": {"schema": {...}}}`. The text is built under a grammar, so it is valid JSON that obeys the schema (supported: `type`, `enum`, `const`, `properties`/`required`, `items` with `minItems`/`maxItems`, `anyOf`/`oneOf`, `$ref`). Keywords that cannot be enforced (`pattern`, `minimum`, `maxLength`, `allOf`, ...) are rejected with an error, never ignored. Members come out required-first, alphabetical. Tell the model in the prompt that you want JSON; set a `max_tokens` large enough, because a cut answer is not valid JSON. Not combined with `tools`; the llama.cpp engine rejects it.
- **Not supported (rejected with a clear error, never ignored):** `functions` (the old format), `tools` on `/v1/completions`, `n` > 1, image content. Requests are processed one at a time.
- **Embeddings:** `POST /v1/embeddings` on the own engine (decoder models; `float` or `base64`; L2-normalized; `input` is a text or a list of up to 256 texts). Buriti picks the mode from your RAM and says so in the response (`buriti` field) and at `GET /v1/embeddings/plan`: a dedicated embedding model resident next to the chat model when memory is roomy (set `BURITI_EMBED_MODEL` to a .gguf in the models folder, or send its name in `model`), loaded only for the request on a tight machine, or the chat model itself (lower quality) when nothing else fits. BERT-style encoders, token-id inputs and `dimensions` are refused with a clear error.
- **Fill-in-the-middle** needs a model with FIM tokens: the Qwen2.5-Coder family works in both engines. Instruct variants tend to keep generating after the gap; base models stop cleanly. On the own engine the server also ends the completion when the model starts rewriting the text that already follows the cursor, or closes a block that the suffix already closes, and stops at `<|fim_pad|>`, `<|file_sep|>` and similar markers. A prompt that already contains the FIM markers as text (`<|fim_prefix|>...<|fim_middle|>`, as some plugins send it without `suffix`) is understood too. The model to prefer for autocomplete is **Qwen2.5-Coder 1.5B Base** (in the catalog; the Instruct one is for chat).
- **Prompt budget for autocomplete:** every keystroke recomputes the prompt (the KV cache only helps up to the edit point), and prefill on a CPU is slow, so a completion with `suffix` is limited to about **1024 prompt tokens** by default: the prefix is trimmed from its start and the suffix from its end, in whole lines (the text near the cursor is kept). Override per request with the optional field `max_prompt_tokens` (an extension of Buriti, clamped to 64..262144; it also applies to requests without `suffix`, which otherwise are limited only by the context). The cut points move in steps of about 1/8 of the budget, so they do not change on every keystroke (a moving start would invalidate the whole KV cache). On the own engine the budget counts real tokens; with llama.cpp it is an estimate (3 characters per token, so the real prompt ends up a bit smaller).
- **Requests and cancellation:** the own engine serves one request at a time; a request whose client already disconnected while waiting is dropped without any prefill. For `/v1/completions` **with `suffix`** (never chat) the rule is "latest wins": when a new request arrives from the same client (same `User-Agent`), older requests of that client that are still waiting or have not produced any text yet are ended with an empty, valid completion (editors fire one request per keystroke). Risk: two different programs with the same `User-Agent` asking for autocomplete at the same time would cancel each other. Turn it off with `BURITI_FIM_LATEST_WINS=0`.
- **Try it:**

```bash
curl http://127.0.0.1:8421/v1/chat/completions \
  -H "Authorization: Bearer $BURITI_KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a Python function that reverses a string."}]}'
```

For VS Code and JetBrains setup instructions, open the **Buriti Code** page in the web UI.

*Buriti Code is not affiliated with, endorsed by or sponsored by OpenAI, Microsoft/GitHub, JetBrains, Continue or Cline. Those names are used only to say which tools it works with.*

## Security and privacy

- The server listens on **127.0.0.1 only**. Each start creates a random session token (sent in the URL fragment and as an `X-Buriti-Token` header), and the `Host` header is checked to block DNS-rebinding attacks.
- Downloads (models and the llama.cpp engine) are accepted **only if the SHA-256 matches**: catalog hash for models, the release `digest` for the engine. Archives are extracted without running anything (zip-slip is blocked).
- The `/v1` API additionally requires the API key (compared in constant time), sends no CORS headers (a web page cannot call it from your browser), and refuses unsupported features instead of ignoring them.
- The path to the engine binary cannot be changed from the web UI, only through `BURITI_ENGINE`.
- Model files are read as untrusted input: the GGUF parser checks bounds everywhere, so a malformed file gives an error, not a crash.
- Chat history is encrypted with ChaCha20-Poly1305. The key is in your user profile. This protects the files from being opened by mistake or from another account or machine; it does **not** protect against someone who already controls your user session.
- Text you type is never interpreted as a special token (no injection of chat delimiters).
- Buriti has no telemetry. The network is used only when you click a download.

## Troubleshooting

- **Windows SmartScreen / antivirus warning:** the executable is not code-signed yet, so Windows may warn on first run. Release binaries built in CI with MSVC are planned.
- **Firewall prompt:** both servers bind to `127.0.0.1`, so you can safely block outside access.
- **Errors or an empty page after restarting Buriti:** the session token changed; open the new URL printed in the console.
- **No favicon:** browsers cache favicons per address for a long time; try a private window or clear site data for `127.0.0.1:8421`.
- **Port already in use:** Buriti takes the next free port by itself and prints the one it chose (it remembers it in `port.txt`). Set `BURITI_PORT` to force a specific port. If you use the API from an editor, update the URL when the port changes.
- **Slow replies:** choose a smaller model (see Tips), lower the context size, or use the llama.cpp engine.
- **The own engine refuses a model:** the message says why (unsupported architecture or weight format). Use the llama.cpp engine for that file.

## Limitations

- CPU-only for Buriti's own engine; GPU only through llama.cpp (untested on real hardware).
- Own engine: see [Models and engines](#models-and-engines) for what it does not run yet. On ARM64 the NEON kernels have only been validated by emulation, cross-compilation and a built-in self-test (no ARM machine was available), so their speed is unmeasured.
- No pre-built binaries, installer or code signing yet.
- Tested on one Windows machine so far.

## License

Buriti is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) (SPDX: `PolyForm-Noncommercial-1.0.0`). This is a *source-available* license, **not** an OSI-approved "open source" license, because it does not allow commercial use.

In plain words (the license text is what counts; this summary is not legal advice):

- You may use, study, modify and share Buriti for any **noncommercial purpose**: personal use, research, experiments, hobby projects, personal study, and use by charities, educational institutions, public research, public safety or health, environmental and government organizations.
- You may **not** use it commercially (for example, selling it, offering it as a paid service, or using it inside a for-profit business) without separate written permission from the author.
- If you share it, keep the license text (or its URL) and the `Required Notice:` line from the [LICENSE](LICENSE) file.
- It comes as is, without any warranty.

Commercial use is possible by agreement with the licensor, Petrim Tecnologia da Informação ([petrim.com.br](https://petrim.com.br), contato@petrim.com.br).

### Third-party licenses

- **Models have their own licenses**, independent of Buriti's, and Buriti's license does not change them. Some require attribution, restrict certain uses or forbid commercial use (the catalog marks these as non-commercial). The catalog shows each model's license as a summary; read the full terms on the model's card before using it commercially. Models you point Buriti at are used where they are and keep their own terms.
- **Dependencies:** on 2026-11-23, the license field of all 254 crates from the registry listed in `Cargo.lock` was checked (read from their `Cargo.toml`): every one is permissive (MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0 and similar, or a dual license that includes a permissive option). This is a check of the declared licenses, not a legal audit, and redistributing a build requires carrying their notices.
- **llama.cpp** (MIT license) is downloaded by the app on request and is not bundled in this repository.

## Acknowledgements

Buriti builds on the ideas and file formats of [llama.cpp](https://github.com/ggml-org/llama.cpp) and the GGUF format, and its engine is checked against llama.cpp. The name comes from the buriti palm.
