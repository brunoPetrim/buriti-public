//! Reaproveitamento do prefixo no KV cache (D76): o resultado com reuso tem de ser IDÊNTICO ao de um prefill completo.

use super::*;

fn model_path() -> Option<String> {
    let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR")));
    if !std::path::Path::new(&path).exists() {
        eprintln!("modelo de teste ausente, pulando: {path}");
        return None;
    }
    Some(path)
}

fn greedy() -> Sampling {
    Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0, seed: None }
}

/// Roda um pedido de texto cru e devolve tudo o que foi enviado ao cliente (texto, `finish_reason` e `usage`).
fn go(model: &Model, sess: &mut Session, prompt: &[u32], max_new: usize, gpu: Option<&Gpu>) -> String {
    let (tx, mut rx) = mpsc::channel::<Bytes>(256);
    run(model, sess, prompt.to_vec(), max_new, greedy(), &Stops { texts: &[], ids: &[], suffix: None }, Kind::Text, &tx, None, gpu, None);
    drop(tx);
    let mut out = String::new();
    while let Ok(b) = rx.try_recv() {
        out.push_str(&String::from_utf8_lossy(&b));
    }
    out
}

fn code_text(n: usize) -> String {
    (0..n).map(|i| format!("fn item_{i}(x: u32) -> u32 {{ x * {} + {} }}\n", i % 7 + 2, i * 13 % 101)).collect()
}

type Case = (&'static str, Vec<u32>, Vec<u32>, usize);

/// Os pares (anterior, novo): acrescentar, editar no meio, encurtar, repetir igual, trocar tudo.
fn cases(model: &Model) -> Vec<Case> {
    let a = model.tok.encode_plain(&code_text(14));
    let other = model.tok.encode_plain("Completely different text: the quick brown fox jumps over the lazy dog, again and again.");
    assert!(a.len() > 120, "prompt A curto demais: {}", a.len());
    let mut append = a.clone();
    append.extend(model.tok.encode_plain("fn tail() -> u32 { 7 }"));
    let mut middle = a[..70].to_vec();
    middle.extend(&other);
    let mut disjoint = other.clone();
    disjoint[0] = a[0].wrapping_add(1) % 1000 + 1; // não começa igual a A
    vec![
        ("append", a.clone(), append, 1),
        ("edit-middle", a.clone(), middle, 1),
        ("shorter", a.clone(), a[..60].to_vec(), 1),
        ("identical", a.clone(), a.clone(), 1),
        ("disjoint", a.clone(), disjoint, 0),
    ]
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// `prev` e depois `next` na MESMA sessão (com reuso) contra `next` numa sessão nova (prefill completo). Devolve quantos
/// tokens foram reaproveitados. Reaproveitar não pode mudar nada: texto enviado, tokens, bytes do KV e logits.
fn check_case(model: &Model, name: &str, prev: &[u32], next: &[u32], ctx: usize) -> usize {
    let mut fresh = model.new_session(ctx);
    let want = go(model, &mut fresh, next, 4, None);

    let mut reused = model.new_session(ctx);
    go(model, &mut reused, prev, 4, None);
    // o cache tem o prompt anterior mais o que foi gerado: o reuso real é o prefixo comum com isso
    let expect_reuse = lcp(&reused.tokens, next).min(next.len() - 1);
    let got = go(model, &mut reused, next, 4, None);

    assert_eq!(got, want, "{name}: o que foi enviado ao cliente difere com reuso");
    assert_eq!(reused.tokens, fresh.tokens, "{name}: tokens do cache diferem");
    assert_eq!(reused.logits, fresh.logits, "{name}: logits diferem com reuso (igualdade estrita)");
    let (rk, rv) = reused.kv();
    let (fk, fv) = fresh.kv();
    let kv_dim = rk[0].len() / ctx;
    let n = reused.tokens.len() * kv_dim;
    for l in 0..rk.len() {
        assert!(rk[l][..n] == fk[l][..n], "{name}: KV (K) da camada {l} difere com reuso");
        assert!(rv[l][..n] == fv[l][..n], "{name}: KV (V) da camada {l} difere com reuso");
    }
    expect_reuse
}

#[test]
fn reuse_equals_full_prefill_on_the_cpu() {
    let Some(path) = model_path() else { return };
    // nos dois caminhos de pesos: reempacotado (o padrão) e layout original
    for repack in [true, false] {
        let model = Model::load_with(std::path::Path::new(&path), repack).expect("carregar modelo");
        for (name, prev, next, want_reuse) in cases(&model) {
            let reused = check_case(&model, name, &prev, &next, 512);
            eprintln!("repack={repack} {name}: {reused} de {} tokens reaproveitados", next.len());
            if want_reuse > 0 {
                assert!(reused > 20, "{name}: o teste só vale se houver reuso real (reaproveitou {reused})");
            } else {
                assert_eq!(reused, 0, "{name}");
            }
        }
    }
}

/// Teste com dentes: reaproveitar UM token a mais (o KV daquela posição é do prompt antigo) tem de reprovar.
#[test]
#[should_panic(expected = "difere")]
fn planted_off_by_one_reuse_is_caught() {
    let Some(path) = model_path() else { panic!("difere (sem modelo o teste não prova nada)") };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("carregar modelo");
    let (name, prev, next, _) = cases(&model).into_iter().find(|c| c.0 == "edit-middle").unwrap();
    let mut fresh = model.new_session(512);
    let want = go(&model, &mut fresh, &next, 4, None);
    let mut reused = model.new_session(512);
    go(&model, &mut reused, &prev, 4, None);
    PLANT_EXTRA_REUSE.with(|c| c.set(1));
    let got = go(&model, &mut reused, &next, 4, None);
    PLANT_EXTRA_REUSE.with(|c| c.set(0));
    assert_eq!(got, want, "{name}: o que foi enviado ao cliente difere com reuso");
    assert_eq!(reused.logits, fresh.logits, "{name}: logits diferem com reuso");
    let (rk, _) = reused.kv();
    let (fk, _) = fresh.kv();
    assert!(rk == fk, "{name}: KV difere com reuso");
}

#[test]
fn cache_is_dropped_when_the_processor_changes() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("carregar modelo");
    let a = model.tok.encode_plain(&code_text(6));
    let mut s = model.new_session(512);
    go(&model, &mut s, &a, 2, None);
    assert!(!s.tokens.is_empty() && s.cache_owner == 1);
    // mesmo processador: reaproveita tudo menos o último token
    assert_eq!(reusable_prefix(&mut s, &a, false, None), a.len() - 1);
    // o KV foi escrito pela CPU e o pedido agora é da GPU: nada dele vale
    let mut s2 = model.new_session(512);
    go(&model, &mut s2, &a, 2, None);
    assert_eq!(reusable_prefix(&mut s2, &a, true, None), 0);
    assert!(s2.tokens.is_empty() && s2.cache_owner == 2);
    // imagem diferente: o KV das posições de imagem não vale
    let mut s3 = model.new_session(512);
    go(&model, &mut s3, &a, 2, None);
    s3.img_key = 5;
    assert_eq!(reusable_prefix(&mut s3, &a, false, Some((6, 10))), 10);
    // (a chamada acima truncou o cache em 10 tokens; refaz a sessão para a mesma imagem)
    let mut s4 = model.new_session(512);
    go(&model, &mut s4, &a, 2, None);
    s4.img_key = 5;
    assert_eq!(reusable_prefix(&mut s4, &a, false, Some((5, 10))), a.len() - 1);
}

/// Medição (ignorada): tempo até o primeiro token (prefill + 1 token), com e sem reuso, intercalado.
/// `BURITI_TEST_MODEL=... [BURITI_TEST_LINES=60] [BURITI_TEST_ROUNDS=3] cargo test bench_prefix_reuse_cpu -- --ignored --nocapture`
#[test]
#[ignore = "medição: precisa de um modelo e de uma máquina descansada"]
fn bench_prefix_reuse_cpu() {
    let Some(path) = model_path() else { return };
    let model = Model::load(std::path::Path::new(&path)).expect("carregar modelo");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2)).build().unwrap();
    pool.install(|| bench_reuse(&model, None));
}

fn bench_reuse(model: &Model, gpu: Option<&Gpu>) {
    let lines: usize = std::env::var("BURITI_TEST_LINES").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let rounds: usize = std::env::var("BURITI_TEST_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let ctx = 2048;
    let a = model.tok.encode_plain(&code_text(lines));
    // 1) "tecla no fim": o prompt cresce alguns tokens; 2) "edição a 80%": muda uma linha e o resto (o sufixo do FIM) é recalculado
    let mut typed = a.clone();
    typed.extend(model.tok.encode_plain("let y"));
    let cut = a.len() * 8 / 10;
    let mut edited = a[..cut].to_vec();
    edited.extend(model.tok.encode_plain("// edited\n"));
    edited.extend(&a[cut..]);
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    eprintln!("prompt de {} tokens (contexto {ctx}), backend {}", a.len(), if gpu.is_some() { "GPU" } else { "CPU" });
    for (name, next) in [("tecla no fim", &typed), ("edição a 80%", &edited)] {
        let (mut full, mut reuse) = (Vec::new(), Vec::new());
        for _ in 0..rounds {
            let mut s = model.new_session(ctx);
            let t = Instant::now();
            go(model, &mut s, next, 1, gpu);
            full.push(ms(t));
            let mut s = model.new_session(ctx);
            go(model, &mut s, &a, 1, gpu);
            let n = lcp(&s.tokens, next);
            let t = Instant::now();
            go(model, &mut s, next, 1, gpu);
            reuse.push(ms(t));
            eprintln!("  {name}: {n} de {} tokens reaproveitados", next.len());
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        eprintln!(
            "{name}: sem reuso {:?} ms (mediana {:.0}), com reuso {:?} ms (mediana {:.0})",
            full.iter().map(|x| x.round()).collect::<Vec<_>>(),
            med(&mut full),
            reuse.iter().map(|x| x.round()).collect::<Vec<_>>(),
            med(&mut reuse)
        );
    }
}

/// Mesmo teste de igualdade na GPU: o KV fica na placa (só comparamos o que sai: texto, tokens e logits). Ignorado: precisa de GPU.
/// `BURITI_TEST_MODEL=... [BURITI_TEST_ADAPTER=...] cargo test reuse_equals_full_prefill_on_the_gpu -- --ignored --nocapture`
#[cfg(feature = "gpu")]
#[test]
#[ignore = "precisa de uma GPU real e de um modelo"]
fn reuse_equals_full_prefill_on_the_gpu() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("carregar modelo");
    let mut gm = open_gpu(&model, 512);
    gm.enable_prefill(64).expect("prefill na GPU");
    for (name, prev, next, want_reuse) in cases(&model) {
        let mut fresh = model.new_session(512);
        let want = go(&model, &mut fresh, &next, 4, Some(&gm));
        let mut reused = model.new_session(512);
        go(&model, &mut reused, &prev, 4, Some(&gm));
        let expect = lcp(&reused.tokens, &next).min(next.len() - 1);
        let got = go(&model, &mut reused, &next, 4, Some(&gm));
        let (mean_top, max_all) = logprob_diff(&reused.logits, &fresh.logits);
        eprintln!("GPU {name}: {expect} de {} tokens reaproveitados; logits identicos: {}; diferenca de logprob top-10 media {mean_top:.5}, maxima no vocabulario {max_all:.5}", next.len(), reused.logits == fresh.logits);
        assert_eq!(got, want, "{name}: o que foi enviado ao cliente difere com reuso na GPU");
        assert_eq!(reused.tokens, fresh.tokens, "{name}: tokens do cache diferem");
        assert_eq!(argmax(&reused.logits), argmax(&fresh.logits), "{name}: o token mais provavel difere");
        assert!(mean_top < 0.005, "{name}: logits diferem com reuso na GPU ({mean_top})");
        if want_reuse > 0 {
            assert!(expect > 20, "{name}: sem reuso real ({expect})");
        }
    }
}

#[cfg(feature = "gpu")]
fn open_gpu(model: &Model, ctx: usize) -> Gpu {
    let sel = std::env::var("BURITI_TEST_ADAPTER").map(|s| crate::gpu::select::AdapterSelector::parse(&s)).unwrap_or(crate::gpu::select::AdapterSelector::Best);
    Gpu::open(model, &sel, ctx).expect("abrir a GPU")
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "medição na GPU: precisa de uma GPU real e de um modelo"]
fn bench_prefix_reuse_gpu() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("carregar modelo");
    let mut gm = open_gpu(&model, 2048);
    gm.enable_prefill(64).expect("prefill na GPU");
    bench_reuse(&model, Some(&gm));
}

/// Na GPU (f32) o KV de posicoes calculadas antes pode vir de outro caminho (GEMM do prefill ou matvec da geracao), com outra ordem de
/// soma: a igualdade bit a bit nao vale como na CPU; vale a mesma medida dos outros testes da GPU (diferenca de logprob no top-10).
#[cfg(feature = "gpu")]
fn logprob_diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let ls = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
        v.iter().map(|&x| x as f64 - m - s.ln()).collect::<Vec<f64>>()
    };
    let (la, lb) = (ls(a), ls(b));
    let mut order: Vec<usize> = (0..a.len()).collect();
    order.sort_by(|&x, &y| b[y].partial_cmp(&b[x]).unwrap());
    let mean = order[..10].iter().map(|&i| (la[i] - lb[i]).abs()).sum::<f64>() / 10.0;
    (mean, la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max))
}

#[cfg(feature = "gpu")]
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
}
