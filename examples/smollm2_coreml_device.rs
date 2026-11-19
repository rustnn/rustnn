//! `smollm2_coreml` as a static library for the iOS and watchOS app host.
//!
//! An app must run the system's application loop, so on a device the example is linked into
//! `scripts/apple_device_host.swift`, which owns that loop and calls `rustnn_example_main` on
//! a thread with a large stack. Arguments come from the process, as in the command-line
//! example. `scripts/apple_device_app.py` builds and signs the app.

#[path = "smollm2_coreml/decode.rs"]
mod decode;

#[unsafe(no_mangle)]
pub extern "C" fn rustnn_example_main() {
    decode::entry();
}
