use clap::Parser;
use std::path::PathBuf;
mod ac;

fn main() {
    let args = Args::parse();
    let input_file = std::fs::read(args.input).unwrap();

    match args.mode {
        Mode::HMCompress => {
            let compressed = codec::compress(&input_file);
            std::fs::write(args.output, compressed).unwrap();
        }
        Mode::HMDecompress => {
            let decompressed = codec::decompress(&input_file);
            std::fs::write(args.output, decompressed).unwrap();
        }
        Mode::ACCompress => {
            let compressed = crate::ac::compress(&input_file);
            std::fs::write(args.output, compressed).unwrap();
        }
        Mode::ACDecompress => {
            let decompressed =
                crate::ac::decompress(&input_file).expect_err("Could not decompress!");
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
    HMCompress,
    HMDecompress,
    ACCompress,
    ACDecompress,
}
