use crate::{hm, lz77};

// Picks the best of three ways to store the data, and writes which one it used in the
// first byte:
//
//     [method][compressed data]
//
//     STORED:  the input copied as is. Used when nothing else makes it smaller,
//              like random bytes or data that is already compressed (zip, jpg).
//     HUFFMAN: word Huffman (hm.rs). One leaf per word.
//     LZ77:    LZ77 + Huffman (lz77.rs). Usually the smallest.
//
// We just run both compressors and keep the smaller result. That costs about twice
// the time of running one, but it can never pick the wrong one.

const STORED: u8 = 0;
const HUFFMAN: u8 = 1;
const LZ77: u8 = 2;

pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut best_method = STORED;
    let mut best = data.to_vec();

    // Word Huffman can't build a tree from fewer than 2 different words.
    if let Some(compressed) = hm::compress(data) {
        if compressed.len() < best.len() {
            best_method = HUFFMAN;
            best = compressed;
        }
    }

    let compressed = lz77::compress(data);
    if compressed.len() < best.len() {
        best_method = LZ77;
        best = compressed;
    }

    let mut buffer = Vec::with_capacity(1 + best.len());
    buffer.push(best_method);
    buffer.extend_from_slice(&best);
    buffer
}

pub fn decompress(data: &[u8]) -> Vec<u8> {
    let (&method, data) = data.split_first().expect("Empty file, no method byte");
    match method {
        STORED => data.to_vec(),
        HUFFMAN => hm::decompress(data),
        LZ77 => lz77::decompress(data),
        _ => panic!("Unknown compression method {method}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(input: &[u8]) -> u8 {
        let compressed = compress(input);
        assert_eq!(input, decompress(&compressed).as_slice());
        compressed[0]
    }

    #[test]
    fn small_inputs() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"hello hello");
    }

    #[test]
    fn random_bytes_are_stored() {
        // A simple pseudo random generator, so the test needs no extra crates.
        let mut state: u32 = 12345;
        let input: Vec<u8> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect();
        assert_eq!(round_trip(&input), STORED);
    }

    #[test]
    fn text_uses_lz77() {
        let input = std::fs::read("../test_files/17kb.txt").unwrap();
        assert_eq!(round_trip(&input), LZ77);
    }
}
