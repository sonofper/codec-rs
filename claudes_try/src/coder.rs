//! Binary arithmetic coder (32-bit, carry-less).
//!
//! Instead of Huffman codes (which cost at least 1 bit per symbol), every
//! *bit* of the file is coded with the probability the model predicts for
//! it. A bit predicted with probability 0.99 costs only ~0.015 bits.

pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    pub fn new(out: Vec<u8>) -> Self {
        Encoder { x1: 0, x2: u32::MAX, out }
    }

    /// Encode `bit` where `p` = P(bit == 1) scaled to 16 bits (1..=65535).
    #[inline(always)]
    pub fn encode(&mut self, bit: u32, p: u32) {
        let range = (self.x2 - self.x1) as u64;
        let xmid = self.x1 + ((range * p as u64) >> 16) as u32;
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        // One byte is enough to land inside [x1, x2]: the decoder pads with zeros.
        // Use x1's top byte + 1 unless that would exceed x2; fall back to all 4 bytes.
        let b = (self.x1 >> 24) + 1;
        if b <= 0xFF && (b << 24) <= self.x2 && (b << 24) >= self.x1 {
            self.out.push(b as u8);
        } else {
            self.out.extend_from_slice(&self.x1.to_be_bytes());
        }
        self.out
    }
}

pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: u32::MAX, x: 0, data, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next_byte();
        }
        d
    }

    #[inline(always)]
    fn next_byte(&mut self) -> u32 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b as u32
    }

    #[inline(always)]
    pub fn decode(&mut self, p: u32) -> u32 {
        let range = (self.x2 - self.x1) as u64;
        let xmid = self.x1 + ((range * p as u64) >> 16) as u32;
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | self.next_byte();
        }
        bit
    }
}
