# custom-game-engine

A custom, low-end-optimized game engine targeting **<=100 MB RAM for a full game**.

The engine and the game are separate repositories (separate projects). The game depends on
this engine crate (path or git dependency) and implements the [`App`] trait.

## Design goals

- **Low-end friendly**: no garbage collector, deterministic memory, bounded asset cache,
  low-power GPU preference, and a hard memory budget.
- **Small footprint**: the engine itself is a tiny Rust library; a full game built on it
  should fit in <=100 MB RAM.
- **Separate repos**: `custom-game-engine` (this repo) and `custom-game` (the game repo)
  are independent projects.
- **Verified**: memory is tracked per frame and asserted in tests/CI.

## Core modules

| Module | Responsibility |
|--------|----------------|
| `core` | App trait, window, input, timing, run loop |
| `render` | Minimal wgpu renderer (fixed pipelines, bounded buffers) |
| `ecs` | Thin data-oriented wrapper over `hecs` |
| `asset` | Bounded asset cache with handles |
| `audio` | Minimal `rodio` playback |
| `memory` | RSS tracking + budget enforcement (<=100 MB) |

## Usage

```rust
use custom_game_engine::{App, AppBuilder, RunContext};

struct MyGame;

impl App for MyGame {
    fn update(&mut self, ctx: &mut RunContext) {
        // game logic
    }
    fn render(&mut self, ctx: &mut RunContext) {
        // drawing
    }
}

fn main() {
    AppBuilder::new()
        .title("My Game")
        .size(800, 600)
        .memory_budget_mb(100)
        .build(MyGame)
        .run();
}
```

## Verification

```bash
cargo test        # memory budget + RSS reporting tests
cargo check       # confirm the engine compiles clean
```

The game repo (`custom-game`) depends on this engine and is verified separately to use
<=100 MB RAM at runtime.
