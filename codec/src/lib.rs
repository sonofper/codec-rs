use std::collections::HashMap;

mod bit_stream;

use bit_stream::BitStream;

const HEADER_SIZE: u32 = 4;
const ASCII_SPACE: u8 = 0x20;

pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut buffer = vec![0; 4];
    // u32
    let occurences = find_occurences(data);
    let tree = encode_huffman_tree(occurences).unwrap();

    serialize_huffman_tree(&tree, &mut buffer);
    let huffman_tree_length = buffer.len() as u32 - HEADER_SIZE;
    buffer[0..4].copy_from_slice(&huffman_tree_length.to_le_bytes());
    let bit_stream = compress_huffman(
        &data.split(|&byte| byte == ASCII_SPACE).collect::<Vec<_>>(),
        &tree,
    );
    buffer.extend_from_slice(&(bit_stream.len() as u64).to_le_bytes());
    buffer.extend_from_slice(bit_stream.as_bytes());

    // .lgm
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
    // Leaf(String),
}

// Huffman tree
// is 6 very 4 gigachad 3 huffman 2
// (is 6 ((gigachad 3 huffman 2) 5 very 4) 9) 15

//              15
//      is                   9
//                 very                 5
//                              huffman    gigachad

//  0   10    110     111
//  is  very  huffman gigachad

// "huffman is very very is is is gigachad is huffman is very very gigachad gigachad"

type TreeData<'a> = &'a [u8];

fn encode_huffman_tree(mut occurences: Vec<(usize, NodeVariant)>) -> Option<Box<Node>> {
    while occurences.len() >= 2 {
        let last = occurences.pop()?;
        let second_last = occurences.pop()?;

        let node = (
            last.0 + second_last.0,
            NodeVariant::Node(Box::new(Node {
                left: last.1,
                right: second_last.1,
            })),
        );
        occurences.push(node);
        sort_by_occurences(&mut occurences);
    }
    let (_, node) = occurences.pop()?;
    match node {
        NodeVariant::Node(n) => Some(n),
        NodeVariant::Leaf(_) => None,
    }
}

// So for each node, starting at root:

//     If leaf-node: Output 1-bit + N-bit character/byte
//     If not leaf-node, output 0-bit. Then encode both child nodes (left first then right) the same way

// To read, do this:

//     Read bit. If 1, then read N-bit character/byte, return new node around it with no children

//     If bit was 0, decode left and right child-nodes the same way, and return new node around them
//     with those children, but no value

// 0 7 llll ; 0, 1, 2, 3, 4
// 0000_tlll

fn serialize_huffman_tree(tree: &Node, writer: &mut Vec<u8>) {
    writer.push(0);
    match &tree.left {
        NodeVariant::Leaf(leaf) => {
            serialize_leaf_node(leaf, writer);
        }
        NodeVariant::Node(node) => {
            serialize_huffman_tree(node, writer);
        }
    }
    match &tree.right {
        NodeVariant::Leaf(leaf) => {
            serialize_leaf_node(leaf, writer);
        }
        NodeVariant::Node(node) => {
            serialize_huffman_tree(node, writer);
        }
    }
}

fn serialize_leaf_node(leaf: &[u8], writer: &mut Vec<u8>) {
    let length = leaf.len() as u64;
    let bit_length = if length == 0 { 1 } else { length.ilog2() + 1 };
    let byte_length = bit_length.div_ceil(8);
    let byte = (byte_length - 1) | 8;
    writer.push(byte as u8);
    writer.extend_from_slice(&length.to_le_bytes()[..byte_length as usize]);
    writer.extend_from_slice(leaf);
}

fn deserialize_huffman_tree(mut data: &[u8]) -> Option<(NodeVariant, usize)> {
    // TODO: Update for bit streams.
    // let mut left: Option<NodeVariant> = None;
    // let mut right: Option<NodeVariant> = None;
    let is_leaf = data[0] != 0;
    if is_leaf {
        let mut bytes_read = 0;
        let byte_length = (data[0] & 7) + 1;
        let mut length_slice = [0u8; 8];
        // length_slice.
        // length_slice.copy_from_slice(&data[1..byte_length as usize + 1]);
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
        // data = &data[length..];
        //
    } else {
        let mut bytes_read = 1;
        let (left, read) = deserialize_huffman_tree(&data[bytes_read..]).unwrap();
        bytes_read += read;
        data = &data[bytes_read..];
        let (right, read) = deserialize_huffman_tree(data).unwrap();
        bytes_read += read;
        // data = &data[bytes_read..];
        let node = Node { left, right };
        Some((NodeVariant::Node(Box::new(node)), bytes_read))
        // todo!()
    }

    // todo!()
}

fn find_occurences(data: TreeData) -> Vec<(usize, NodeVariant)> {
    let mut occurences: Vec<(usize, NodeVariant)> = Vec::new();
    let mut map: HashMap<TreeData, usize> = HashMap::new();

    for word in data.split(|&byte| byte == ASCII_SPACE) {
        // for word in data.split(' ') {
        *map.entry(word).or_default() += 1;
        // if map.contains_key(word) {
        //     map.entry(word).and_modify(|count| *count += 1);
        // } else {
        //     map.insert(word, 1);
        // }
    }

    for (word, count) in map {
        let node = NodeVariant::Leaf(word.to_vec());
        // let node = NodeVariant::Leaf(word.to_string());
        occurences.push((count, node));
    }

    sort_by_occurences(&mut occurences);

    occurences
}

fn sort_by_occurences(occurences: &mut [(usize, NodeVariant)]) {
    occurences.sort_by_key(|a| std::cmp::Reverse(a.0));
}

fn compress_huffman(data: &[TreeData], tree: &Node) -> BitStream {
    let mut map: HashMap<TreeData, BitStream> = HashMap::new();
    traverse_tree(tree, &mut map, &BitStream::new());
    let mut out_stream = BitStream::new();
    for &word in data {
        let stream = map.get(word).unwrap();
        for bit in stream {
            out_stream.push(bit);
        }
    }

    out_stream
    // data.iter()
    //     .flat_map(|word| {
    //         let string = std::str::from_utf8(word).unwrap();
    //         let stream = map.get(word).unwrap();
    //         println!("word: {string}, stream: {stream:?}");
    //         stream
    //     })
    //     // .copied()
    //     .collect()
    // todo!()
}

fn traverse_tree<'a>(
    node: &'a Node,
    map: &mut HashMap<TreeData<'a>, BitStream>,
    directions: &BitStream,
) {
    let mut left_directions = directions.clone();
    left_directions.push(false);
    match &node.left {
        NodeVariant::Leaf(node) => {
            map.insert(node, left_directions);
        }
        NodeVariant::Node(node) => traverse_tree(node, map, &left_directions),
    }

    let mut right_directions = directions.clone();
    right_directions.push(true);
    match &node.right {
        NodeVariant::Leaf(node) => {
            map.insert(node, right_directions);
        }
        NodeVariant::Node(node) => traverse_tree(node, map, &right_directions),
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
                decompressed.push(ASCII_SPACE);
            }
        }
    }
    decompressed.pop();

    decompressed
}

// SUDO code
// input(pattern) for eks. words or repeated sentences
// give every pattern a number representing the amount of times shown
// put the words in a sorted list, sort by the number
// make a new list with the two least common patterns then re-add to the list
// then resort and repeat until the final list only have 2 items
// BOOM you got a huffman tree represented with a nested list

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn sort_order() {
    //     let mut internal = vec![(3, 0), (5, 0), (1, 0)];
    //     sort_by_occurences(&mut internal);
    //     assert_eq!(internal, [(5, 0), (3, 0), (1, 0)]);
    // }

    #[test]
    fn huffman_tree() {
        // is 6 very 4 gigachad 3 huffman 2
        // (is 6 ((gigachad 3 huffman 2) 5 very 4) 9) 15
        let data =
            "huffman is very very is is is gigachad is huffman is very very gigachad gigachad";
        // let occurences = vec![(6, "is"), (4, "very"), (3, "gigachad"), (2, "huffman")];
        let occurences = find_occurences(data.as_bytes());
        println!("{occurences:?}");
        let tree = encode_huffman_tree(occurences).unwrap();
        // let mut map = HashMap::new();
        // traverse_tree(&tree, &mut map, &[]);

        println!("{tree:?}");
        let data = compress_huffman(
            &data.split(' ').map(|s| s.as_bytes()).collect::<Vec<_>>(),
            // &data.split(' ').collect::<Vec<_>>(),
            &tree,
        );
        panic!("{data:?}");
        // panic!("{map:?}");
    }

    #[test]
    fn round_trip() {
        // is 6 very 4 gigachad 3 huffman 2
        // (is 6 ((gigachad 3 huffman 2) 5 very 4) 9) 15
        let data =
            "huffman is very very is is is gigachad is huffman is very very gigachad gigachad";
        // let occurences = vec![(6, "is"), (4, "very"), (3, "gigachad"), (2, "huffman")];
        let occurences = find_occurences(data.as_bytes());
        println!("{occurences:?}");
        let tree = encode_huffman_tree(occurences).unwrap();
        // let mut map = HashMap::new();
        // traverse_tree(&tree, &mut map, &[]);

        println!("{tree:?}");
        let bit_stream = compress_huffman(
            &data.split(' ').map(|s| s.as_bytes()).collect::<Vec<_>>(),
            // &data.split(' ').collect::<Vec<_>>(),
            &tree,
        );
        let data = decompress_huffman_tree(&bit_stream, &tree);
        let string = std::str::from_utf8(&data).unwrap();
        panic!("'{string}'");
        // panic!("{data:?}");
        // panic!("{map:?}");
    }

    #[test]
    fn serde_huffman_tree() {
        let data =
            "huffman is very very is is is gigachad is huffman is very very gigachad gigachad";
        let occurences = find_occurences(data.as_bytes());
        let tree = encode_huffman_tree(occurences).unwrap();
        let mut buffer = Vec::new();
        serialize_huffman_tree(&tree, &mut buffer);
        let (deserialized, _) = deserialize_huffman_tree(&buffer).unwrap();
        match deserialized {
            NodeVariant::Leaf(_) => panic!("unexpected leaf"),
            NodeVariant::Node(node) => assert_eq!(tree, node),
        }
    }

    #[test]
    fn full_round_trip() {
        let input = std::fs::read("../test_files/17kb.txt").unwrap();
        let compressed = compress(&input);
        let uncompressed = decompress(&compressed);
        assert_eq!(input, uncompressed);
    }

    // fn to_leaf(string: &str) -> NodeVariant {
    //     NodeVariant::Leaf(string.as_bytes().to_vec())
    //     // NodeVariant::Leaf(string.to_string())
    // }
}
