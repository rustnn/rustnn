//! SmolLM2-135M-Instruct on Core ML, from the Hugging Face checkpoint to a device receipt.
//!
//! `--checkpoint DIR` authors the decode graph from `model.safetensors` and writes it to the
//! `--bundle` directory (`model.webnn`, its weights and `tokenizer.json`). Without
//! `--checkpoint` the bundle is loaded as it is, which is what an iPhone or an Apple Watch
//! does. The graph is then compiled through `MLContext` and decoded one token per dispatch.
//!
//! The key/value window is carried inside the model as a Core ML `MLState` (`--carry state`)
//! or bound as tensors (`--carry tensors`). The receipt records build time, decode rate, peak
//! resident memory and a hash of the token stream, so two runs can be compared exactly.
//! `docs/user-guide/smollm2-on-apple-devices.md` has the build and packaging steps.

#[path = "smollm2_coreml/decode.rs"]
mod decode;

fn main() {
    // Graph conversion recurses deeply; give it more than the main thread's stack.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(decode::entry)
        .unwrap()
        .join()
        .unwrap();
}
