use clap::{AppSettings, ColorChoice, Parser};
use libffm::{Error, Model};
use std::path::PathBuf;
use std::process;

#[derive(Debug, Parser)]
#[clap(name = "ffm-train", version, color = ColorChoice::Never)]
#[clap(global_setting(AppSettings::DeriveDisplayOrder))]
struct Opt {
    /// Set regularization parameter
    #[clap(short, default_value = "0.00002")]
    lambda: f32,

    /// Set number of latent factors
    #[clap(short = 'k', default_value = "4")]
    factor: i32,

    /// Set number of iterations
    #[clap(short = 't', default_value = "15")]
    iteration: i32,

    /// Set learning rate
    #[clap(short = 'r', default_value = "0.2")]
    eta: f32,

    /// Set number of threads
    #[clap(short = 's', default_value = "1")]
    nr_threads: i32,

    /// Set path to the validation set
    #[clap(short = 'p', parse(from_os_str))]
    va_path: Option<PathBuf>,

    /// Quiet mode (no output)
    #[clap(long)]
    quiet: bool,

    /// Disable instance-wise normalization
    #[clap(long)]
    no_norm: bool,

    /// Stop at the iteration that achieves the best validation loss (must be used with -p)
    #[clap(long)]
    auto_stop: bool,

    /// Enable in-memory training
    #[clap(long)]
    in_memory: bool,

    #[clap(name = "train-file", parse(from_os_str))]
    tr_path: PathBuf,

    #[clap(name = "model-file", parse(from_os_str))]
    model_path: Option<PathBuf>,
}

fn train_on_disk(opt: &Opt) -> Result<(), Error> {
    let mut params = Model::params();
    params
        .learning_rate(opt.eta)
        .lambda(opt.lambda)
        .iterations(opt.iteration)
        .factors(opt.factor)
        .normalization(!opt.no_norm)
        .auto_stop(opt.auto_stop)
        .quiet(opt.quiet)
        .on_disk(!opt.in_memory);

    let model = match &opt.va_path {
        Some(p) => params.train_eval(&opt.tr_path, &p)?,
        None => params.train(&opt.tr_path)?,
    };

    let model_path = opt.model_path.clone().unwrap_or_else(|| {
        let mut filename = opt.tr_path.file_name().unwrap().to_os_string();
        filename.push(".model");
        PathBuf::from(filename)
    });
    model.save(&model_path)
}

fn main() {
    let opt = Opt::parse();

    if opt.auto_stop && opt.va_path.is_none() {
        println!("To use auto-stop, you need to assign a validation set");
        process::exit(1);
    }

    if let Err(err) = train_on_disk(&opt) {
        println!("{}", err);
        process::exit(1);
    }
}
