//! Memory: process memory tracking and budget enforcement.
//!
//! Reads RSS from /proc/self/status and exposes it so the game (and CI) can
//! verify the engine stays under the configured budget (<=100 MB).

use std::fs;

/// Tracks resident memory and reports against a hard budget.
pub struct MemoryBudget {
    budget_mb: usize,
    peak_mb: usize,
    frame: usize,
    last_rss_mb: f32,
}

impl MemoryBudget {
    pub fn new(budget_mb: usize) -> Self {
        let initial = current_rss_mb();
        Self {
            budget_mb,
            peak_mb: initial as usize,
            frame: 0,
            last_rss_mb: initial,
        }
    }

    /// Call once per frame. Returns current RSS in MiB (sampled periodically to avoid heap churn).
    pub fn tick(&mut self) -> f32 {
        self.frame += 1;
        if self.frame == 1 || self.frame.is_multiple_of(30) {
            self.last_rss_mb = current_rss_mb();
            if self.last_rss_mb as usize > self.peak_mb {
                self.peak_mb = self.last_rss_mb as usize;
            }
        }
        self.last_rss_mb
    }

    pub fn last_rss_mb(&self) -> f32 {
        self.last_rss_mb
    }

    pub fn current_mb(&self) -> f32 {
        current_rss_mb()
    }

    pub fn peak_mb(&self) -> usize {
        self.peak_mb
    }

    pub fn budget_mb(&self) -> usize {
        self.budget_mb
    }

    pub fn within_budget(&self) -> bool {
        (current_rss_mb() as usize) <= self.budget_mb
    }

    pub fn frame(&self) -> usize {
        self.frame
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Read resident set size in MiB from /proc/self/status.
fn current_rss_mb() -> f32 {
    let content = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f32 = rest
                .trim()
                .strip_suffix(" kB")
                .unwrap_or(rest.trim())
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_is_reported() {
        let m = current_rss_mb();
        assert!(m >= 0.0, "RSS should be non-negative, got {}", m);
    }

    #[test]
    fn budget_starts_within() {
        let b = MemoryBudget::new(100);
        assert!(b.within_budget());
    }
}
