use clap::Parser;
use std::path::PathBuf;

fn main() {
    let args = Args::parse();
    let input_file = std::fs::read(args.input).unwrap();

    match args.mode {
        Mode::HMCompress => {
            let compressed = codec::hm::compress(&input_file);
            std::fs::write(args.output, compressed).unwrap();
        }
        Mode::HMDecompress => {
            let decompressed = codec::hm::decompress(&input_file);
            std::fs::write(args.output, decompressed).unwrap();
        }
        Mode::ACCompress => {
            let compressed = codec::ac::compress(&input_file);
            std::fs::write(args.output, compressed).unwrap();
        }
        Mode::ACDecompress => {
            let decompressed = codec::ac::decompress(&input_file).expect("Could not decompress!");
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
