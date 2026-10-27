// The command-line interface and its dependencies are native-only.
#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
