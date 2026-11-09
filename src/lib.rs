#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{prelude::*, UnwrapThrowExt};

pub mod rd;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn setup_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap_throw();
    log::info!("Loaded");
    Ok(())
}
