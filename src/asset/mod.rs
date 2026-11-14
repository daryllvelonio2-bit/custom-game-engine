//! Asset: minimal asset loader with a fixed-size cache.
//!
//! Assets are loaded on demand and referenced by handle. The cache has a hard
//! entry cap so the engine cannot grow memory unboundedly.

use std::collections::HashMap;
use std::sync::Arc;

/// Opaque handle to a loaded asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle(pub u64);

/// Loaded asset payload.
#[derive(Clone)]
pub enum Asset {
    Image(Arc<image::RgbaImage>),
    Bytes(Arc<Vec<u8>>),
}

/// Asset loader with a bounded cache.
pub struct AssetLoader {
    next_id: u64,
    cache: HashMap<AssetHandle, Asset>,
    path_to_handle: HashMap<String, AssetHandle>,
    max_entries: usize,
}

impl Default for AssetLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetLoader {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            cache: HashMap::new(),
            path_to_handle: HashMap::new(),
            max_entries: 256,
        }
    }

    pub fn with_capacity(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Load a texture from a file path, returning a handle. Cached by path.
    pub fn load_image(&mut self, path: &str) -> Option<AssetHandle> {
        if let Some(&h) = self.path_to_handle.get(path) {
            return Some(h);
        }
        if self.cache.len() >= self.max_entries {
            return None; // budget exhausted: refuse to grow
        }
        let img = image::open(path).ok()?.into_rgba8();
        let handle = AssetHandle(self.next_id);
        self.next_id += 1;
        self.cache.insert(handle, Asset::Image(Arc::new(img)));
        self.path_to_handle.insert(path.to_string(), handle);
        Some(handle)
    }

    pub fn get(&self, handle: AssetHandle) -> Option<&Asset> {
        self.cache.get(&handle)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}
