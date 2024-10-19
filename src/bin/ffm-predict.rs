use clap::{ColorChoice, Parser};
use libffm::Model;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process;

#[derive(Debug, Parser)]
#[command(name = "ffm-predict", version, color = ColorChoice::Never)]
struct Args {
    #[arg(value_parser)]
    test_file: PathBuf,

    #[arg(value_parser)]
    model_file: PathBuf,

    #[arg(value_parser)]
    output_file: PathBuf,
}

fn predict(
    test_path: &PathBuf,
    model_path: &PathBuf,
    output_path: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let model = Model::load(model_path)?;
    let (predictions, loss) = model.predict(test_path)?;

    let f_out = File::create(output_path)?;
    let mut writer = BufWriter::new(f_out);
    for prediction in predictions {
        writer.write_all(format!("{:.6}\n", prediction).as_bytes())?;
    }
    writer.flush()?;

    println!("logloss = {:.5}", loss);

    Ok(())
}

fn main() {
    let args = Args::parse();

    if let Err(err) = predict(&args.test_file, &args.model_file, &args.output_file) {
        println!("{}", err);
        process::exit(1);
    }
}
