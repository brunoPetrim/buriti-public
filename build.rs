// Embute o ícone no .exe (só no Windows). Em outros sistemas não faz nada.
fn main() {
    println!("cargo:rerun-if-changed=buriti.rc");
    println!("cargo:rerun-if-changed=web/images/buriti_favicon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let _ = embed_resource::compile("buriti.rc", embed_resource::NONE);
    }
}
