use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::bit_stream::BitStream;

const HEADER_SIZE: u32 = 4;

pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut buffer = vec![0; 4];
    let tokens = tokenize(data);
    let occurences = find_occurences(&tokens);
    let tree = encode_huffman_tree(occurences).unwrap();

    serialize_huffman_tree(&tree, &mut buffer);
    let huffman_tree_length = buffer.len() as u32 - HEADER_SIZE;
    buffer[0..4].copy_from_slice(&huffman_tree_length.to_le_bytes());
    let bit_stream = compress_huffman(&tokens, &tree);
    buffer.extend_from_slice(&(bit_stream.len() as u64).to_le_bytes());
    buffer.extend_from_slice(bit_stream.as_bytes());
    buffer
}

pub fn decompress(data: &[u8]) -> Vec<u8> {
    let huffman_tree_length = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let (tree, _) = deserialize_huffman_tree(&data[4..huffman_tree_length as usize + 4]).unwrap();
    let tree = match tree {
        NodeVariant::Leaf(_) => panic!("That's not a tree! That's a leaf."),
        NodeVariant::Node(node) => node,
    };
    let data = &data[huffman_tree_length as usize + 4..];
    let bit_stream_length = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let bit_stream = BitStream::from_buffer(&data[8..], bit_stream_length as usize);
    decompress_huffman_tree(&bit_stream, &tree)
}

#[derive(Debug, PartialEq, Eq)]
struct Node {
    left: NodeVariant,
    right: NodeVariant,
}

#[derive(Debug, PartialEq, Eq)]
enum NodeVariant {
    Node(Box<Node>),
    Leaf(Vec<u8>),
}

type TreeData<'a> = &'a [u8];

fn tokenize(data: &[u8]) -> Vec<TreeData<'_>> {
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80;
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let start = i;
        if is_word(data[i]) {
            while i < data.len() && is_word(data[i]) {
                i += 1;
            }
        } else if data[i] == b' ' {
            while i < data.len() && data[i] == b' ' {
                i += 1;
            }
        } else {
            i += 1;
        }
        tokens.push(&data[start..i]);
    }
    tokens
}

struct HeapEntry(usize, NodeVariant);
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

fn encode_huffman_tree(occurences: Vec<(usize, NodeVariant)>) -> Option<Box<Node>> {
    let mut heap: BinaryHeap<HeapEntry> = occurences
        .into_iter()
        .map(|(count, node)| HeapEntry(count, node))
        .collect();
    while heap.len() >= 2 {
        let last = heap.pop()?;
        let second_last = heap.pop()?;
        heap.push(HeapEntry(
            last.0 + second_last.0,
            NodeVariant::Node(Box::new(Node {
                left: last.1,
                right: second_last.1,
            })),
        ));
    }
    match heap.pop()?.1 {
        NodeVariant::Node(n) => Some(n),
        NodeVariant::Leaf(_) => None,
    }
}

fn serialize_huffman_tree(tree: &Node, writer: &mut Vec<u8>) {
    writer.push(0);
    match &tree.left {
        NodeVariant::Leaf(leaf) => serialize_leaf_node(leaf, writer),
        NodeVariant::Node(node) => serialize_huffman_tree(node, writer),
    }
    match &tree.right {
        NodeVariant::Leaf(leaf) => serialize_leaf_node(leaf, writer),
        NodeVariant::Node(node) => serialize_huffman_tree(node, writer),
    }
}

fn serialize_leaf_node(leaf: &[u8], writer: &mut Vec<u8>) {
    let length = leaf.len() as u64;
    if length < 128 {
        writer.push(0x80 | length as u8);
        writer.extend_from_slice(leaf);
        return;
    }
    let bit_length = length.ilog2() + 1;
    let byte_length = bit_length.div_ceil(8);
    let byte = (byte_length - 1) | 8;
    writer.push(byte as u8);
    writer.extend_from_slice(&length.to_le_bytes()[..byte_length as usize]);
    writer.extend_from_slice(leaf);
}

fn deserialize_huffman_tree(mut data: &[u8]) -> Option<(NodeVariant, usize)> {
    let is_leaf = data[0] != 0;
    if is_leaf {
        if data[0] & 0x80 != 0 {
            let length = (data[0] & 0x7f) as usize;
            return Some((NodeVariant::Leaf(data[1..1 + length].to_vec()), 1 + length));
        }
        let mut bytes_read = 0;
        let byte_length = (data[0] & 7) + 1;
        let mut length_slice = [0u8; 8];
        for (byte, length) in length_slice
            .iter_mut()
            .zip(&data[1..byte_length as usize + 1])
        {
            *byte = *length;
        }
        let length = u64::from_le_bytes(length_slice) as usize;
        data = &data[byte_length as usize + 1..];
        bytes_read += byte_length as usize + 1;
        let message = &data[0..length];
        let node = NodeVariant::Leaf(message.to_vec());
        bytes_read += length;
        Some((node, bytes_read))
    } else {
        let mut bytes_read = 1;
        let (left, read) = deserialize_huffman_tree(&data[bytes_read..]).unwrap();
        bytes_read += read;
        data = &data[bytes_read..];
        let (right, read) = deserialize_huffman_tree(data).unwrap();
        bytes_read += read;
        let node = Node { left, right };
        Some((NodeVariant::Node(Box::new(node)), bytes_read))
    }
}

fn find_occurences(tokens: &[TreeData]) -> Vec<(usize, NodeVariant)> {
    let mut map: HashMap<TreeData, usize> = HashMap::new();
    for &word in tokens {
        *map.entry(word).or_default() += 1;
    }
    map.into_iter()
        .map(|(word, count)| (count, NodeVariant::Leaf(word.to_vec())))
        .collect()
}

fn compress_huffman(data: &[TreeData], tree: &Node) -> BitStream {
    let mut map: HashMap<TreeData, (u64, u8)> = HashMap::new();
    traverse_tree(tree, &mut map, 0, 0);
    let mut out_stream = BitStream::new();
    for &word in data {
        let (code, length) = map[word];
        out_stream.push_bits(code, length);
    }
    out_stream
}

fn traverse_tree<'a>(
    node: &'a Node,
    map: &mut HashMap<TreeData<'a>, (u64, u8)>,
    code: u64,
    length: u8,
) {
    assert!(length < 64, "Huffman code longer than 64 bits");
    for (child, bit) in [(&node.left, 0), (&node.right, 1)] {
        let child_code = (code << 1) | bit;
        match child {
            NodeVariant::Leaf(leaf) => {
                map.insert(leaf, (child_code, length + 1));
            }
            NodeVariant::Node(node) => traverse_tree(node, map, child_code, length + 1),
        }
    }
}

fn decompress_huffman_tree(bit_stream: &BitStream, tree: &Node) -> Vec<u8> {
    let mut current_node = tree;
    let mut decompressed = Vec::new();
    for bit in bit_stream {
        let variant = if bit {
            &current_node.right
        } else {
            &current_node.left
        };
        match variant {
            NodeVariant::Node(node) => current_node = node,
            NodeVariant::Leaf(leaf) => {
                current_node = tree;
                decompressed.extend_from_slice(leaf);
            }
        }
    }
    decompressed
}

// SUDO code
// input(pattern) for eks. words or repeated sentences
// give every pattern a number representing the amount of times shown
// put the words in a sorted list, sort by the number
// make a new list with the two least common patterns then re-add to the list
// then resort and repeat until the final list only have 2 items
// BOOM you got a huffman tree represented with a nested list
