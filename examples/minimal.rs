//! Minimal example app using `custom-game-engine`.
//!
//! Run with: `cargo run --example minimal`

use custom_game_engine::{App, AppBuilder, RunContext};

struct MinimalGame {
    frames: u32,
}

impl App for MinimalGame {
    fn update(&mut self, ctx: &mut RunContext) {
        self.frames += 1;
        for key in &ctx.input.keys_just_pressed {
            if matches!(
                key,
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape)
            ) {
                eprintln!("escape pressed");
            }
        }
        if self.frames % 60 == 0 {
            eprintln!(
                "frame {}  delta={:.3}  (engine running)",
                self.frames, ctx.delta
            );
        }
    }

    fn render(&mut self, _ctx: &mut RunContext) {
        // The engine clears the screen and draws a quad each frame.
    }
}

fn main() {
    AppBuilder::new()
        .title("custom-game-engine minimal")
        .size(800, 600)
        .vsync(false)
        .memory_budget_mb(100)
        .build(MinimalGame { frames: 0 })
        .run();
}
