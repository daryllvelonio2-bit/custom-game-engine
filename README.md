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

## Fast dev loop

| Command | What it does |
|---|---|
| `cargo q` | `check --lib` |
| `cargo t` | `test --lib` |
| `cargo l` | `clippy --lib -D warnings` |
| `cargo c` | `check --all-targets` |

Dependencies are trimmed for compile speed (`image`: PNG only, `rodio`: WAV only,
`wgpu`: no DX12/Metal/SPIR-V backends). This crate and `custom-game` share one target dir
(`Documents/GitHub/custom-target`), so the graph is compiled once. Override with
`CARGO_TARGET_DIR` if you ever need to.

## Verification

```bash
cargo t           # memory budget + render/font tests
cargo c           # confirm the engine (and its example) compiles clean
```

The game repo (`custom-game`) depends on this engine and is verified separately to use
<=100 MB RAM at runtime.
