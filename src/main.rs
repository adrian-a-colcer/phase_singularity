use phase_singularity::rd::{NX, NY, run_headless, save_png};

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("png") => {
            let steps: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12_000);
            let state = pollster::block_on(run_headless(steps))?;
            let path = format!("fhn_{steps}.png");
            save_png(&path, &state, NX, NY)?;
            println!("saved {path}");
            Ok(())
        }
        _ => phase_singularity::viewer::run(),
    }
}