#[derive(Debug, Clone)]
pub struct BitStream {
    buffer: Vec<u8>,
    length: usize,
}

impl BitStream {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            length: 0,
        }
    }

    pub fn from_buffer(buffer: &[u8], bits: usize) -> Self {
        Self {
            buffer: buffer.to_vec(),
            length: bits,
        }
    }

    pub fn push(&mut self, bit: bool) {
        let index = self.length / 8;
        if self.length.is_multiple_of(8) {
            self.buffer.push(0);
        }
        let mut byte = self.buffer[index];
        byte = (byte << 1) | bit as u8;
        self.buffer[index] = byte;
        self.length += 1;
    }

    pub fn pop(&mut self) -> Option<bool> {
        if self.length == 0 {
            return None;
        }
        let index = (self.length - 1) / 8;
        let mut byte = self.buffer[index];
        let bit = byte & 1 != 0;
        byte >>= 1;
        self.buffer[index] = byte;
        self.length -= 1;
        if self.length.is_multiple_of(8) {
            self.buffer.pop();
        }

        Some(bit)
    }

    /// Returns length in bits.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    pub fn iter(&self) -> BitStreamIter<'_> {
        BitStreamIter {
            buffer: &self.buffer,
            length: self.length,
            bit_index: 0,
        }
    }
}

pub struct BitStreamIter<'a> {
    buffer: &'a [u8],
    length: usize,
    bit_index: usize,
}

impl<'a> IntoIterator for &'a BitStream {
    type Item = bool;
    type IntoIter = BitStreamIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Iterator for BitStreamIter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bit_index >= self.length {
            return None;
        }
        let index = self.bit_index / 8;
        let pos = self.bit_index % 8;
        let byte = self.buffer[index];
        let bit = if self.bit_index / 8 == self.buffer.len() - 1 && !self.length.is_multiple_of(8) {
            let shift = 8 - self.length % 8 + pos;
            byte << shift & 0x80 != 0
        } else {
            byte << pos & 0x80 != 0
        };
        self.bit_index += 1;

        Some(bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstream() {
        let mut stream = BitStream::new();
        let data = [
            false, false, false, true, true, false, true, false, true, true, false, false, true,
            false,
        ];
        for bit in data {
            stream.push(bit);
        }
        let bits: Vec<bool> = stream.iter().collect();
        assert_eq!(bits, data);
        let mut reversed = Vec::new();
        while let Some(bit) = stream.pop() {
            reversed.push(bit);
        }
        assert_eq!(reversed, data.into_iter().rev().collect::<Vec<_>>());
    }
}
