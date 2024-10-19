use clap::{ColorChoice, Parser};
use libffm::{Error, Model};
use std::path::PathBuf;
use std::process;

#[derive(Debug, Parser)]
#[command(name = "ffm-train", version, color = ColorChoice::Never)]
struct Args {
    /// Set regularization parameter
    #[arg(short, default_value_t = 0.00002)]
    lambda: f32,

    /// Set number of latent factors
    #[arg(short = 'k', default_value_t = 4)]
    factor: i32,

    /// Set number of iterations
    #[arg(short = 't', default_value_t = 15)]
    iteration: i32,

    /// Set learning rate
    #[arg(short = 'r', default_value_t = 0.2)]
    eta: f32,

    /// Set number of threads
    #[arg(short = 's', default_value_t = 1)]
    nr_threads: i32,

    /// Set path to the validation set
    #[arg(short = 'p', value_parser)]
    va_path: Option<PathBuf>,

    /// Quiet mode (no output)
    #[arg(long)]
    quiet: bool,

    /// Disable instance-wise normalization
    #[arg(long)]
    no_norm: bool,

    /// Stop at the iteration that achieves the best validation loss (must be used with -p)
    #[arg(long)]
    auto_stop: bool,

    /// Enable in-memory training
    #[arg(long)]
    in_memory: bool,

    #[arg(name = "train-file", value_parser)]
    tr_path: PathBuf,

    #[arg(name = "model-file", value_parser)]
    model_path: Option<PathBuf>,
}

fn train_on_disk(args: &Args) -> Result<(), Error> {
    let mut params = Model::params();
    params
        .learning_rate(args.eta)
        .lambda(args.lambda)
        .iterations(args.iteration)
        .factors(args.factor)
        .normalization(!args.no_norm)
        .auto_stop(args.auto_stop)
        .quiet(args.quiet)
        .on_disk(!args.in_memory);

    let model = match &args.va_path {
        Some(p) => params.train_eval(&args.tr_path, p)?,
        None => params.train(&args.tr_path)?,
    };

    let model_path = args.model_path.clone().unwrap_or_else(|| {
        let mut filename = args.tr_path.file_name().unwrap().to_os_string();
        filename.push(".model");
        PathBuf::from(filename)
    });
    model.save(&model_path)
}

fn main() {
    let args = Args::parse();

    if args.auto_stop && args.va_path.is_none() {
        println!("To use auto-stop, you need to assign a validation set");
        process::exit(1);
    }

    if let Err(err) = train_on_disk(&args) {
        println!("{}", err);
        process::exit(1);
    }
}
