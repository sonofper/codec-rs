use std::collections::HashMap;

use super::hm::{
    Node, NodeVariant, TreeData, deserialize_huffman_tree, encode_huffman_tree, find_occurences,
    serialize_huffman_tree, traverse_tree,
};
use crate::bit_stream::BitStream;

// LZ77 + Huffman, the same idea as DEFLATE (zip, gzip, png).
//
// Step 1, LZ77: walk through the input. At every position, look back in the last
// WINDOW_SIZE bytes for the longest earlier copy of the bytes that come next.
//
//     "the cat and the hat"
//      ^^^^        ^^^^ "the " was seen 12 bytes ago, 4 bytes long
//
//     Literal('t') Literal('h') Literal('e') Literal(' ') ... Match { length: 4, distance: 12 } ...
//
// Step 2, Huffman: the tokens are turned into symbols and Huffman coded.
// There are two trees:
//
//     literal tree:  every literal byte, and the "class" of every match length
//     distance tree: the "class" of every match distance
//
// A class is just how many bits a number needs. 12 = 0b1100 needs 4 bits, so its class
// is 4. The top bit of a number is always 1, so only the 3 bits below it ("100") are
// written raw after the symbol. This keeps the trees small (at most ~16 classes)
// instead of one leaf for every possible distance.
//
// File layout:
//
//     [u32 literal tree length][literal tree]
//     [u32 distance tree length][distance tree]
//     [u64 symbol count][u64 bit count][bits]

const WINDOW_SIZE: usize = 32 * 1024;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
// How many earlier positions we try per byte. Higher = better compression, slower.
const MAX_CHAIN: usize = 64;

const HASH_BITS: u32 = 15;
const NO_POSITION: usize = usize::MAX;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Lz77Token {
    Literal(u8),
    Match { length: usize, distance: usize },
}

// Leaves in the literal tree.
//   literal byte:  1 byte leaf  [byte]
//   length class:  2 byte leaf  [LENGTH_MARKER, class]
// Leaves in the distance tree.
//   distance class: 1 byte leaf [class]
// Both trees are padded with 3 byte leaves so they always have at least 2 leaves.
const LENGTH_MARKER: u8 = 0;
const PADDING_LEAVES: [&[u8]; 2] = [b"pad", b"PAD"];

pub fn compress(data: &[u8]) -> Vec<u8> {
    let tokens = find_tokens(data);

    // Turn the tokens into leaves so we can count them and build the trees.
    let mut literal_leaves: Vec<Vec<u8>> = Vec::new();
    let mut distance_leaves: Vec<Vec<u8>> = Vec::new();
    for &token in &tokens {
        match token {
            Lz77Token::Literal(byte) => literal_leaves.push(vec![byte]),
            Lz77Token::Match { length, distance } => {
                let (length_class, _, _) = split_into_class(length - MIN_MATCH + 1);
                let (distance_class, _, _) = split_into_class(distance);
                literal_leaves.push(vec![LENGTH_MARKER, length_class]);
                distance_leaves.push(vec![distance_class]);
            }
        }
    }
    let literal_tree = build_tree(&literal_leaves);
    let distance_tree = build_tree(&distance_leaves);

    let mut buffer = Vec::new();
    write_tree(&literal_tree, &mut buffer);
    write_tree(&distance_tree, &mut buffer);

    // Leaf -> (code, code length), like in compress_huffman.
    let mut literal_codes = HashMap::new();
    traverse_tree(&literal_tree, &mut literal_codes, 0, 0);
    let mut distance_codes = HashMap::new();
    traverse_tree(&distance_tree, &mut distance_codes, 0, 0);

    let mut bit_stream = BitStream::new();
    for &token in &tokens {
        match token {
            Lz77Token::Literal(byte) => {
                let (code, code_length) = literal_codes[&[byte][..]];
                bit_stream.push_bits(code, code_length);
            }
            Lz77Token::Match { length, distance } => {
                let (class, extra, extra_bits) = split_into_class(length - MIN_MATCH + 1);
                let (code, code_length) = literal_codes[&[LENGTH_MARKER, class][..]];
                bit_stream.push_bits(code, code_length);
                bit_stream.push_bits(extra, extra_bits);

                let (class, extra, extra_bits) = split_into_class(distance);
                let (code, code_length) = distance_codes[&[class][..]];
                bit_stream.push_bits(code, code_length);
                bit_stream.push_bits(extra, extra_bits);
            }
        }
    }

    buffer.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    buffer.extend_from_slice(&(bit_stream.len() as u64).to_le_bytes());
    buffer.extend_from_slice(bit_stream.as_bytes());
    buffer
}

pub fn decompress(data: &[u8]) -> Vec<u8> {
    let (literal_tree, read) = read_tree(data);
    let data = &data[read..];
    let (distance_tree, read) = read_tree(data);
    let data = &data[read..];

    let token_count = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let bit_stream_length = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let bit_stream = BitStream::from_buffer(&data[16..], bit_stream_length as usize);
    let mut bits = bit_stream.iter();

    let mut decompressed = Vec::new();
    for _ in 0..token_count {
        let leaf = read_leaf(&mut bits, &literal_tree);
        match leaf {
            [byte] => decompressed.push(*byte),
            [LENGTH_MARKER, class] => {
                let extra = read_bits(&mut bits, class - 1);
                let length = join_class(*class, extra) + MIN_MATCH - 1;

                let [class] = read_leaf(&mut bits, &distance_tree) else {
                    panic!("Bad distance leaf");
                };
                let extra = read_bits(&mut bits, class - 1);
                let distance = join_class(*class, extra);

                // Copy byte by byte: the copy may overlap what it is writing.
                // ("aaaaaa" is Literal('a') + Match { length: 5, distance: 1 })
                let start = decompressed.len() - distance;
                for i in 0..length {
                    decompressed.push(decompressed[start + i]);
                }
            }
            _ => panic!("Bad literal leaf {leaf:?}"),
        }
    }

    decompressed
}

// LZ77 match finding with hash chains.
//
// `head[hash]` is the latest position whose next 3 bytes have that hash.
// `previous[position]` is the position before it with the same hash.
// Following previous[] from head[] visits every earlier candidate, newest first.
fn find_tokens(data: &[u8]) -> Vec<Lz77Token> {
    let mut head = vec![NO_POSITION; 1 << HASH_BITS];
    let mut previous = vec![NO_POSITION; data.len()];
    let mut tokens = Vec::new();

    let mut position = 0;
    while position < data.len() {
        let (length, distance) = longest_match(data, position, &head, &previous);

        let step = if length >= MIN_MATCH {
            tokens.push(Lz77Token::Match { length, distance });
            length
        } else {
            tokens.push(Lz77Token::Literal(data[position]));
            1
        };

        // Remember every position we step over, also the ones inside a match,
        // so later matches can point into them.
        for p in position..position + step {
            if p + MIN_MATCH <= data.len() {
                let hash = hash_three_bytes(&data[p..]);
                previous[p] = head[hash];
                head[hash] = p;
            }
        }
        position += step;
    }

    tokens
}

fn longest_match(
    data: &[u8],
    position: usize,
    head: &[usize],
    previous: &[usize],
) -> (usize, usize) {
    if position + MIN_MATCH > data.len() {
        return (0, 0);
    }
    let max_length = MAX_MATCH.min(data.len() - position);
    let mut best_length = 0;
    let mut best_distance = 0;

    let mut candidate = head[hash_three_bytes(&data[position..])];
    let mut tries = 0;
    while candidate != NO_POSITION && position - candidate <= WINDOW_SIZE && tries < MAX_CHAIN {
        let length = data[candidate..]
            .iter()
            .zip(&data[position..position + max_length])
            .take_while(|(a, b)| a == b)
            .count();
        if length > best_length {
            best_length = length;
            best_distance = position - candidate;
            if length == max_length {
                break;
            }
        }
        candidate = previous[candidate];
        tries += 1;
    }

    (best_length, best_distance)
}

fn hash_three_bytes(bytes: &[u8]) -> usize {
    let value = (bytes[0] as u32) << 16 | (bytes[1] as u32) << 8 | bytes[2] as u32;
    // Multiply by a large odd number and keep the top bits: spreads similar inputs apart.
    (value.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize
}

// 12 = 0b1100 -> class 4, extra 0b100, 3 extra bits.
// `value` must be at least 1.
fn split_into_class(value: usize) -> (u8, u64, u8) {
    let class = usize::BITS - value.leading_zeros();
    let extra_bits = class - 1;
    let extra = value as u64 & ((1 << extra_bits) - 1);
    (class as u8, extra, extra_bits as u8)
}

// Class 4, extra 0b100 -> 0b1100 = 12. Puts the top bit back.
fn join_class(class: u8, extra: u64) -> usize {
    (1 << (class - 1)) | extra as usize
}

fn build_tree(leaves: &[Vec<u8>]) -> Box<Node> {
    let mut words: Vec<TreeData> = leaves.iter().map(|leaf| leaf.as_slice()).collect();
    // A tree needs at least two leaves. These are never used, they only fill the tree.
    words.extend(PADDING_LEAVES);
    let mut occurences = find_occurences(&words);
    for (count, leaf) in &mut occurences {
        if let NodeVariant::Leaf(leaf) = leaf {
            if PADDING_LEAVES.contains(&leaf.as_slice()) {
                *count = 0; // never used, so give it the longest code
            }
        }
    }
    encode_huffman_tree(occurences).unwrap()
}

fn write_tree(tree: &Node, buffer: &mut Vec<u8>) {
    let start = buffer.len();
    buffer.extend_from_slice(&[0; 4]);
    serialize_huffman_tree(tree, buffer);
    let tree_length = (buffer.len() - start - 4) as u32;
    buffer[start..start + 4].copy_from_slice(&tree_length.to_le_bytes());
}

// Returns the tree and how many bytes it used, including the length header.
fn read_tree(data: &[u8]) -> (Box<Node>, usize) {
    let tree_length = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let (tree, _) = deserialize_huffman_tree(&data[4..4 + tree_length]).unwrap();
    match tree {
        NodeVariant::Leaf(_) => panic!("That's not a tree! That's a leaf."),
        NodeVariant::Node(node) => (node, 4 + tree_length),
    }
}

// Walk from the root, one bit at a time, until we land on a leaf.
fn read_leaf<'a>(bits: &mut impl Iterator<Item = bool>, tree: &'a Node) -> &'a [u8] {
    let mut current_node = tree;
    loop {
        let bit = bits
            .next()
            .expect("bit stream ended in the middle of a symbol");
        let variant = if bit {
            &current_node.right
        } else {
            &current_node.left
        };
        match variant {
            NodeVariant::Node(node) => current_node = node,
            NodeVariant::Leaf(leaf) => return leaf,
        }
    }
}

// Read `count` raw bits, first bit is the most significant (same order as push_bits).
fn read_bits(bits: &mut impl Iterator<Item = bool>, count: u8) -> u64 {
    let mut value = 0;
    for _ in 0..count {
        let bit = bits
            .next()
            .expect("bit stream ended in the middle of extra bits");
        value = (value << 1) | bit as u64;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(input: &[u8]) {
        let compressed = compress(input);
        let decompressed = decompress(&compressed);
        assert_eq!(input, decompressed.as_slice());
    }

    #[test]
    fn small_inputs() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"ab");
        round_trip(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        round_trip(b"the cat and the hat and the bat");
    }

    #[test]
    fn finds_match() {
        let tokens = find_tokens(b"the cat and the hat");
        assert!(tokens.contains(&Lz77Token::Match {
            length: 4,
            distance: 12
        }));
    }

    #[test]
    fn class_split_and_join() {
        for value in [1, 2, 3, 12, 255, 256, 32768] {
            let (class, extra, _) = split_into_class(value);
            assert_eq!(join_class(class, extra), value);
        }
    }

    #[test]
    fn all_bytes() {
        let input: Vec<u8> = (0..10_000u32).map(|i| (i * 7919 % 251) as u8).collect();
        round_trip(&input);
    }

    #[test]
    fn full_round_trip() {
        let input = std::fs::read("../test_files/17kb.txt").unwrap();
        round_trip(&input);
    }
}
