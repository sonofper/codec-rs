//! Claude's try — a context-mixing file compressor.
//!
//! Usage:
//!   claudes_try compress   <input> <output>
//!   claudes_try decompress <input> <output>
//!   claudes_try bench      <files...>      (compress + decompress + verify in memory)
//!
//! Compression and decompression are always separate runs; the decompressor
//! only needs the compressed file.

mod coder;
mod math;
mod model;

use coder::{Decoder, Encoder};
use model::Model;
use std::time::Instant;
use std::{env, fs, process};

const MAGIC: &[u8; 4] = b"CLT1";

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn read_varint(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 3 + 16);
    out.extend_from_slice(MAGIC);
    write_varint(&mut out, data.len() as u64);
    if data.is_empty() {
        return out;
    }
    let mut enc = Encoder::new(out);
    let mut m = Model::new(data.len());
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            enc.encode(bit, m.p());
            m.update(bit);
        }
    }
    enc.finish()
}

pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 5 || &data[..4] != MAGIC {
        return Err("not a claudes_try file (bad magic)".into());
    }
    let mut pos = 4;
    let len = read_varint(data, &mut pos).ok_or("corrupt header")? as usize;
    let mut out = Vec::with_capacity(len);
    if len == 0 {
        return Ok(out);
    }
    let mut dec = Decoder::new(&data[pos..]);
    let mut m = Model::new(len);
    for _ in 0..len {
        let mut c = 0u32;
        for _ in 0..8 {
            let bit = dec.decode(m.p());
            m.update(bit);
            c = (c << 1) | bit;
        }
        out.push(c as u8);
    }
    Ok(out)
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    process::exit(1);
}

fn usage() -> ! {
    eprintln!("Claude's try - context-mixing compressor");
    eprintln!("usage:");
    eprintln!("  claudes_try compress   <input> <output>");
    eprintln!("  claudes_try decompress <input> <output>");
    eprintln!("  claudes_try bench      <files...>");
    process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "compress" | "c" => {
            if args.len() != 4 {
                usage();
            }
            let data = fs::read(&args[2]).unwrap_or_else(|e| die(&format!("reading {}: {e}", args[2])));
            let t = Instant::now();
            let out = compress(&data);
            fs::write(&args[3], &out).unwrap_or_else(|e| die(&format!("writing {}: {e}", args[3])));
            println!(
                "{} -> {}: {} -> {} bytes ({:.3} bits/byte, {:.2}%) in {:.2}s",
                args[2],
                args[3],
                data.len(),
                out.len(),
                out.len() as f64 * 8.0 / data.len().max(1) as f64,
                out.len() as f64 * 100.0 / data.len().max(1) as f64,
                t.elapsed().as_secs_f64()
            );
        }
        "decompress" | "d" => {
            if args.len() != 4 {
                usage();
            }
            let data = fs::read(&args[2]).unwrap_or_else(|e| die(&format!("reading {}: {e}", args[2])));
            let t = Instant::now();
            let out = decompress(&data).unwrap_or_else(|e| die(&e));
            fs::write(&args[3], &out).unwrap_or_else(|e| die(&format!("writing {}: {e}", args[3])));
            println!(
                "{} -> {}: {} -> {} bytes in {:.2}s",
                args[2],
                args[3],
                data.len(),
                out.len(),
                t.elapsed().as_secs_f64()
            );
        }
        "bench" => {
            let files = &args[2..];
            let handles: Vec<_> = files
                .iter()
                .cloned()
                .map(|f| {
                    std::thread::spawn(move || {
                        let data = fs::read(&f).unwrap_or_else(|e| die(&format!("reading {f}: {e}")));
                        let t = Instant::now();
                        let c = compress(&data);
                        let tc = t.elapsed().as_secs_f64();
                        let t = Instant::now();
                        let d = decompress(&c).unwrap();
                        let td = t.elapsed().as_secs_f64();
                        let ok = d == data;
                        format!(
                            "{:<28} {:>10} -> {:>9}  {:.4} bpc  c {:.2}s d {:.2}s  {}",
                            f,
                            data.len(),
                            c.len(),
                            c.len() as f64 * 8.0 / data.len().max(1) as f64,
                            tc,
                            td,
                            if ok { "OK" } else { "MISMATCH!" }
                        )
                    })
                })
                .collect();
            for h in handles {
                println!("{}", h.join().unwrap());
            }
        }
        _ => usage(),
    }
}
