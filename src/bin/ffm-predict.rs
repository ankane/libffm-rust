use clap::{AppSettings, ColorChoice, Parser};
use libffm::Model;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process;

#[derive(Debug, Parser)]
#[clap(name = "ffm-predict", version, color = ColorChoice::Never)]
#[clap(global_setting(AppSettings::DeriveDisplayOrder))]
struct Opt {
    #[clap(parse(from_os_str))]
    test_file: PathBuf,

    #[clap(parse(from_os_str))]
    model_file: PathBuf,

    #[clap(parse(from_os_str))]
    output_file: PathBuf,
}

fn predict(test_path: &PathBuf, model_path: &PathBuf, output_path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let model = Model::load(model_path)?;
    let (predictions, loss) = model.predict(test_path)?;

    let f_out = File::create(output_path)?;
    let mut writer = BufWriter::new(f_out);
    for prediction in predictions {
        writer.write(format!("{:.6}\n", prediction).as_bytes())?;
    }
    writer.flush()?;

    println!("logloss = {:.5}", loss);

    Ok(())
}

fn main() {
    let opt = Opt::parse();

    if let Err(err) = predict(&opt.test_file, &opt.model_file, &opt.output_file) {
        println!("{}", err);
        process::exit(1);
    }
}
