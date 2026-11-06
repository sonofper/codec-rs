use clap::Parser;
use std::path::PathBuf;

fn main() {
    let args = Args::parse();
    let input_file = std::fs::read(args.input).unwrap();

    match args.mode {
        Mode::DeflateCompress => {
            let compressed = codec::deflate::compress(&input_file);
            std::fs::write(args.output, compressed).unwrap();
        }
        Mode::DeflateDecompress => {
            let decompressed = codec::deflate::decompress(&input_file);
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
    DeflateCompress,
    DeflateDecompress,
}
