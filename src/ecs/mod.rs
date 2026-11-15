//! ECS: a thin, data-oriented wrapper around `hecs`.
//!
//! Games build entities with components and register systems that run each frame.

use hecs::{Entity, World};

/// A system runs once per frame with access to the ECS world.
pub type System<'a> = Box<dyn FnMut(&mut World) + 'a>;

/// The ECS world container. Games may also use `hecs::World` directly.
pub struct Ecs {
    pub world: World,
}

impl Default for Ecs {
    fn default() -> Self {
        Self::new()
    }
}

impl Ecs {
    pub fn new() -> Self {
        Self {
            world: World::new(),
        }
    }

    pub fn spawn(&mut self, bundle: impl hecs::Bundle) -> Entity {
        self.world.spawn(bundle)
    }

    pub fn run_system<F: FnMut(&mut World)>(&mut self, mut system: F) {
        system(&mut self.world);
    }
}
