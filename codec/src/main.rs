use std::path::PathBuf;

use clap::Parser;

fn main() {
    let args = Args::parse();
    let input_file = std::fs::read(args.input).unwrap();

    match args.mode {
        Mode::Compress => {
            let compressed = codec::compress(&input_file);
            std::fs::write(args.output, compressed).unwrap();
        }
        Mode::Decompress => {
            let decompressed = codec::decompress(&input_file);
            std::fs::write(args.output, decompressed).unwrap();
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    mode: Mode,
    input: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum Mode {
    Compress,
    Decompress,
}
