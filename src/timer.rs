use std::time::{Duration, Instant};

pub struct Timer {
    begin: Instant,
    duration: Duration,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            begin: Instant::now(),
            duration: Duration::ZERO,
        }
    }

    pub fn tic(&mut self) {
        self.begin = Instant::now();
    }

    pub fn toc(&mut self) -> f32 {
        self.duration += Instant::now().duration_since(self.begin);
        self.get()
    }

    pub fn get(&self) -> f32 {
        self.duration.as_millis() as f32 / 1000.0
    }
}
