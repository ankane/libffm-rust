use crate::disk::{ProblemLoader, ProblemOnDisk};
use crate::error::Error;
use crate::model::Model;
use crate::timer::Timer;
use std::io::{Read, Seek, Write};
use std::path::Path;

/// A set of parameters.
#[derive(Debug)]
pub struct Params {
    pub(crate) eta: f32,
    pub(crate) lambda: f32,
    nr_iters: i32,
    pub(crate) k: i32,
    pub(crate) normalization: bool,
    auto_stop: bool,
    quiet: bool,
    on_disk: bool,
}

impl Params {
    /// Returns a new set of parameters.
    pub fn new() -> Self {
        Self {
            eta: 0.2,
            lambda: 0.00002,
            nr_iters: 15,
            k: 4,
            normalization: true,
            auto_stop: false,
            quiet: false,
            on_disk: false,
        }
    }

    /// Sets the learning rate.
    pub fn learning_rate(&mut self, value: f32) -> &mut Self {
        self.eta = value;
        self
    }

    /// Sets the regularization parameter.
    pub fn lambda(&mut self, value: f32) -> &mut Self {
        self.lambda = value;
        self
    }

    /// Sets the number of iterations.
    pub fn iterations(&mut self, value: i32) -> &mut Self {
        self.nr_iters = value;
        self
    }

    /// Sets the number of latent factors.
    pub fn factors(&mut self, value: i32) -> &mut Self {
        self.k = value;
        self
    }

    /// Sets whether to use instance-wise normalization.
    pub fn normalization(&mut self, value: bool) -> &mut Self {
        self.normalization = value;
        self
    }

    /// Sets whether to stop at the iteration that achieves the best validation loss.
    pub fn auto_stop(&mut self, value: bool) -> &mut Self {
        self.auto_stop = value;
        self
    }

    /// Sets whether to use quiet mode (no output).
    pub fn quiet(&mut self, value: bool) -> &mut Self {
        self.quiet = value;
        self
    }

    /// Sets whether to use on-disk training.
    pub fn on_disk(&mut self, value: bool) -> &mut Self {
        self.on_disk = value;
        self
    }

    /// Trains a model.
    pub fn train<P: AsRef<Path>>(&self, tr_path: P) -> Result<Model, Error> {
        let mut tr_loader = ProblemLoader::new(tr_path, self.quiet)?;

        if self.on_disk {
            let tr_problem = tr_loader.read_to_disk()?;
            self.train_core(tr_problem)
        } else {
            let tr_problem = tr_loader.read_to_memory()?;
            self.train_core(tr_problem)
        }
    }

    /// Trains a model and performs cross-validation.
    pub fn train_eval<P: AsRef<Path>, Q: AsRef<Path>>(&self, tr_path: P, va_path: Q) -> Result<Model, Error> {
        // open both files first to fail fast
        let mut tr_loader = ProblemLoader::new(tr_path, self.quiet)?;
        let mut va_loader = ProblemLoader::new(va_path, self.quiet)?;

        if self.on_disk {
            let tr_problem = tr_loader.read_to_disk()?;
            let va_problem = va_loader.read_to_disk()?;
            self.train_eval_core(tr_problem, va_problem)
        } else {
            let tr_problem = tr_loader.read_to_memory()?;
            let va_problem = va_loader.read_to_memory()?;
            self.train_eval_core(tr_problem, va_problem)
        }
    }

    fn train_core<W: Read + Write + Seek>(&self, mut tr: ProblemOnDisk<W>) -> Result<Model, Error> {
        let mut model = Model::new(tr.meta.n, tr.meta.m, self);

        self.logln(format!("{:>4}{:>13}{:>13}", "iter", "tr_logloss", "tr_time"));

        let mut timer = Timer::new();

        for iter in 1..=self.nr_iters {
            timer.tic();
            let tr_loss = model.one_epoch(&mut tr, true, self)?;
            timer.toc();
            self.logln(format!("{:>4}{:>13.5}{:>13.1}", iter, tr_loss, timer.get()));
        }

        Ok(model)
    }

    fn train_eval_core<W: Read + Write + Seek>(&self, mut tr: ProblemOnDisk<W>, mut va: ProblemOnDisk<W>) -> Result<Model, Error> {
        let mut model = Model::new(tr.meta.n, tr.meta.m, self);
        let mut prev_w = Vec::new();
        let mut best_va_loss = f32::MAX;

        self.logln(format!("{:>4}{:>13}{:>13}{:>13}", "iter", "tr_logloss", "va_logloss", "tr_time"));

        let mut timer = Timer::new();

        for iter in 1..=self.nr_iters {
            timer.tic();
            let tr_loss = model.one_epoch(&mut tr, true, self)?;
            timer.toc();

            let va_loss = model.one_epoch(&mut va, false, self)?;
            if self.auto_stop {
                if va_loss > best_va_loss {
                    model.w = prev_w;
                    self.logln(format!("Auto-stop. Use model at {}th iteration.", iter - 1));
                    break;
                } else {
                    prev_w = model.w.clone();
                    best_va_loss = va_loss;
                }
            }

            self.logln(format!("{:>4}{:>13.5}{:>13.5}{:>13.1}", iter, tr_loss, va_loss, timer.get()));
        }

        Ok(model)
    }

    fn logln(&self, msg: String) {
        if !self.quiet {
            println!("{}", msg);
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Self::new()
    }
}
