//! # custom-game-engine
//!
//! A custom, low-end-optimized game engine targeting <=100 MB RAM for a full game.
//!
//! The engine and the game live in separate repositories. The game depends on
//! this engine crate (path or git dependency) and implements the [`App`] trait.

pub mod asset;
pub mod audio;
pub mod core;
pub mod ecs;
pub mod memory;
pub mod render;

pub use core::{App, AppBuilder, Engine, InitContext, RunContext};
