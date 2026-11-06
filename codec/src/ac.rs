// DECLARATIONS USED FOR ARITHMETIC ENCODING AND DECODING

// SIZE OF ARITHMETIC CODE VALUES

const CODE_VALUE_BITS: u32 = 16; // Number of bits in a code value
type CodeValue = i64; // Type of an arithmetic code value
const TOP_VALUE: CodeValue = (1 << CODE_VALUE_BITS) - 1; // Largest code value

// HALF AND QUARTER POINTS IN THE CODE VALUE RANGE

const FIRST_QUARTER: CodeValue = TOP_VALUE / 4 + 1; // Point after first quarter
const HALF: CodeValue = 2 * FIRST_QUARTER; // Point after first half
const THIRD_QUARTER: CodeValue = 3 * FIRST_QUARTER; // Point after third quarter

// INTERFACE TO THE MODEL

// THE SET OF SYMBOLS THAT MAY BE ENCODED

const NUMBER_OF_CHARS: usize = 256; // Number of character symbols
const EOF_SYMBOL: usize = NUMBER_OF_CHARS + 1; // Index of EOF symbol

const NUMBER_OF_SYMBOLS: usize = NUMBER_OF_CHARS + 1; // Total number of symbols

// CUMULATIVE FREQUENCY TABLE

const MAX_FREQUENCY: CodeValue = 16383; // Maximum allowed frequency count (2^14 - 1)

// THE ADAPTIVE SOURCE MODEL

struct Model {
    char_to_index: [usize; NUMBER_OF_CHARS], // To index from character
    index_to_char: [u8; NUMBER_OF_SYMBOLS + 1], // To character from index
    freq: [CodeValue; NUMBER_OF_SYMBOLS + 1], // Symbol frequencies
    cum_freq: [CodeValue; NUMBER_OF_SYMBOLS + 1], // Cumulative symbol frequencies
}

impl Model {
    /// INITIALIZE THE MODEL (start_model)
    fn new() -> Self {
        let mut model = Model {
            char_to_index: [0; NUMBER_OF_CHARS],
            index_to_char: [0; NUMBER_OF_SYMBOLS + 1],
            freq: [1; NUMBER_OF_SYMBOLS + 1],
            cum_freq: [0; NUMBER_OF_SYMBOLS + 1],
        };
        // Set up tables that translate between symbol indexes and characters.
        for i in 0..NUMBER_OF_CHARS {
            model.char_to_index[i] = i + 1;
            model.index_to_char[i + 1] = i as u8;
        }
        // Set up initial frequency counts to be one for all symbols.
        for i in 0..=NUMBER_OF_SYMBOLS {
            model.cum_freq[i] = (NUMBER_OF_SYMBOLS - i) as CodeValue;
        }
        model.freq[0] = 0; // freq[0] must not be the same as freq[1].
        model
    }

    /// UPDATE THE MODEL TO ACCOUNT FOR A NEW SYMBOL (update_model)
    fn update(&mut self, symbol: usize) {
        // See if frequency counts are at their maximum; if so, halve all counts
        // (keeping them non-zero).
        if self.cum_freq[0] == MAX_FREQUENCY {
            let mut cum = 0;
            for i in (0..=NUMBER_OF_SYMBOLS).rev() {
                self.freq[i] = (self.freq[i] + 1) / 2;
                self.cum_freq[i] = cum;
                cum += self.freq[i];
            }
        }

        // Find symbol's new index.
        let mut i = symbol;
        while self.freq[i] == self.freq[i - 1] {
            i -= 1;
        }

        // Update the translation tables if the symbol has moved.
        if i < symbol {
            let char_index = self.index_to_char[i];
            let symbol_index = self.index_to_char[symbol];
            self.index_to_char[i] = symbol_index;
            self.index_to_char[symbol] = char_index;
            self.char_to_index[char_index as usize] = symbol;
            self.char_to_index[symbol_index as usize] = i;
        }

        // Increment the frequency count for the symbol and update the
        // cumulative frequencies.
        self.freq[i] += 1;
        while i > 0 {
            i -= 1;
            self.cum_freq[i] += 1;
        }
    }
}

// BIT OUTPUT ROUTINES

struct BitOutput {
    out: Vec<u8>,
    buffer: u8,      // Bits buffered for output
    bits_to_go: u32, // Number of free bits in buffer
}

impl BitOutput {
    /// INITIALIZE FOR BIT OUTPUT (start_outputing_bits)
    fn new() -> Self {
        BitOutput {
            out: Vec::new(),
            buffer: 0,
            bits_to_go: 8,
        } // Buffer starts out with no bits in it.
    }

    /// OUTPUT A BIT (output_bit)
    fn output_bit(&mut self, bit: bool) {
        self.buffer >>= 1; // Put bit in top of buffer.
        if bit {
            self.buffer |= 0x80;
        }
        self.bits_to_go -= 1;
        if self.bits_to_go == 0 {
            // Output buffer if it is now full.
            self.out.push(self.buffer);
            self.bits_to_go = 8;
        }
    }

    /// FLUSH OUT THE LAST BITS (done_outputing_bits)
    fn done(mut self) -> Vec<u8> {
        // Only flush a partial byte; the C version always writes one more
        // byte, which is harmless but useless when bits_to_go == 8.
        if self.bits_to_go < 8 {
            self.out.push(self.buffer >> self.bits_to_go);
        }
        self.out
    }
}

// BIT INPUT ROUTINES

struct BitInput<'a> {
    input: &'a [u8],
    pos: usize,
    buffer: u8,        // Bits waiting to be input
    bits_to_go: u32,   // Number of bits still in buffer
    garbage_bits: u32, // Number of bits past end-of-file
}

impl<'a> BitInput<'a> {
    /// INITIALIZE BIT INPUT (start_inputing_bits)
    fn new(input: &'a [u8]) -> Self {
        BitInput {
            input,
            pos: 0,
            buffer: 0,
            bits_to_go: 0,
            garbage_bits: 0,
        } // Buffer starts out with no bits in it.
    }

    /// INPUT A BIT (input_bit)
    fn input_bit(&mut self) -> Result<CodeValue, &'static str> {
        if self.bits_to_go == 0 {
            // Read the next byte if no bits are left in buffer.
            match self.input.get(self.pos) {
                Some(&b) => {
                    self.buffer = b;
                    self.pos += 1;
                }
                None => {
                    // Return arbitrary bits after eof, but check for too many
                    // such.
                    self.buffer = 0;
                    self.garbage_bits += 1;
                    if self.garbage_bits > CODE_VALUE_BITS - 2 {
                        return Err("Bad input file");
                    }
                }
            }
            self.bits_to_go = 8;
        }
        let t = (self.buffer & 1) as CodeValue; // Return the next bit from the bottom of the byte.
        self.buffer >>= 1;
        self.bits_to_go -= 1;
        Ok(t)
    }
}

// ARITHMETIC ENCODING ALGORITHM

struct Encoder {
    low: CodeValue, // Ends of the current code region
    high: CodeValue,
    bits_to_follow: u64, // Number of opposite bits to output after the next bit
    bits: BitOutput,
}

impl Encoder {
    /// START ENCODING A STREAM OF SYMBOLS (start_encoding)
    fn new() -> Self {
        Encoder {
            low: 0, // Full code range.
            high: TOP_VALUE,
            bits_to_follow: 0, // No bits to follow next.
            bits: BitOutput::new(),
        }
    }

    /// ENCODE A SYMBOL (encode_symbol)
    fn encode_symbol(&mut self, symbol: usize, cum_freq: &[CodeValue]) {
        // Narrow the code region to that allotted to this symbol.
        let range = (self.high - self.low) + 1;
        self.high = self.low + (range * cum_freq[symbol - 1]) / cum_freq[0] - 1;
        self.low += (range * cum_freq[symbol]) / cum_freq[0];

        loop {
            if self.high < HALF {
                self.bit_plus_follow(false); // Output 0 if in low half.
            } else if self.low >= HALF {
                // Output 1 if in high half.
                self.bit_plus_follow(true);
                self.low -= HALF;
                self.high -= HALF; // Subtract offset to top.
            } else if self.low >= FIRST_QUARTER && self.high < THIRD_QUARTER {
                // Output an opposite bit later if in middle half.
                self.bits_to_follow += 1;
                self.low -= FIRST_QUARTER; // Subtract offset to middle.
                self.high -= FIRST_QUARTER;
            } else {
                break; // Otherwise exit loop.
            }
            // Scale up code range.
            self.low *= 2;
            self.high = 2 * self.high + 1;
        }
    }

    /// FINISH ENCODING THE STREAM (done_encoding)
    fn done(mut self) -> Vec<u8> {
        // Output two bits that select the quarter that the current code range
        // contains.
        self.bits_to_follow += 1;
        if self.low < FIRST_QUARTER {
            self.bit_plus_follow(false);
        } else {
            self.bit_plus_follow(true);
        }
        self.bits.done()
    }

    /// OUTPUT BITS PLUS FOLLOWING OPPOSITE BITS (bit_plus_follow)
    fn bit_plus_follow(&mut self, bit: bool) {
        self.bits.output_bit(bit); // Output the bit.
        while self.bits_to_follow > 0 {
            self.bits.output_bit(!bit); // Output bits_to_follow opposite bits.
            self.bits_to_follow -= 1;
        }
    }
}

// ARITHMETIC DECODING ALGORITHM

struct Decoder<'a> {
    value: CodeValue, // Currently-seen code value
    low: CodeValue,   // Ends of current code region
    high: CodeValue,
    bits: BitInput<'a>,
}

impl<'a> Decoder<'a> {
    /// START DECODING A STREAM OF SYMBOLS (start_decoding)
    fn new(input: &'a [u8]) -> Result<Self, &'static str> {
        let mut bits = BitInput::new(input);
        let mut value = 0;
        // Input bits to fill the code value.
        for _ in 0..CODE_VALUE_BITS {
            value = 2 * value + bits.input_bit()?;
        }
        Ok(Decoder {
            value,
            low: 0,
            high: TOP_VALUE,
            bits,
        }) // Full code range.
    }

    /// DECODE THE NEXT SYMBOL (decode_symbol)
    fn decode_symbol(&mut self, cum_freq: &[CodeValue]) -> Result<usize, &'static str> {
        let range = (self.high - self.low) + 1;
        // Find cum freq for value.
        let cum = ((self.value - self.low + 1) * cum_freq[0] - 1) / range;
        // Then find symbol.
        let mut symbol = 1;
        while cum_freq[symbol] > cum {
            symbol += 1;
        }

        // Narrow the code region to that allotted to this symbol.
        self.high = self.low + (range * cum_freq[symbol - 1]) / cum_freq[0] - 1;
        self.low += (range * cum_freq[symbol]) / cum_freq[0];

        loop {
            if self.high < HALF {
                // nothing — expand low half.
            } else if self.low >= HALF {
                // Expand high half.
                self.value -= HALF;
                self.low -= HALF; // Subtract offset to top.
                self.high -= HALF;
            } else if self.low >= FIRST_QUARTER && self.high < THIRD_QUARTER {
                // Expand middle half.
                self.value -= FIRST_QUARTER;
                self.low -= FIRST_QUARTER; // Subtract offset to middle.
                self.high -= FIRST_QUARTER;
            } else {
                break; // Otherwise exit loop.
            }
            // Scale up code range.
            self.low *= 2;
            self.high = 2 * self.high + 1;
            self.value = 2 * self.value + self.bits.input_bit()?; // Move in next input bit.
        }
        Ok(symbol)
    }
}

// MAIN PROGRAM FOR ENCODING

pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut model = Model::new(); // Set up other modules.
    let mut enc = Encoder::new();
    for &ch in data {
        // Loop through characters.
        let symbol = model.char_to_index[ch as usize]; // Translate to an index.
        enc.encode_symbol(symbol, &model.cum_freq); // Encode that symbol.
        model.update(symbol); // Update the model.
    }
    enc.encode_symbol(EOF_SYMBOL, &model.cum_freq); // Encode the EOF symbol.
    enc.done() // Send the last few bits.
}

// MAIN PROGRAM FOR DECODING

pub fn decompress(data: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut model = Model::new(); // Set up other modules.
    let mut dec = Decoder::new(data)?;
    let mut out = Vec::new();
    loop {
        // Loop through characters.
        let symbol = dec.decode_symbol(&model.cum_freq)?; // Decode next symbol.
        if symbol == EOF_SYMBOL {
            break; // Exit loop if EOF symbol.
        }
        out.push(model.index_to_char[symbol]); // Translate to a character and write it out.
        model.update(symbol); // Update the model.
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let compress = compress(data);
        let decompress = decompress(&compress).expect("decode failed");
        assert_eq!(decompress, data, "roundtrip mismatch (len {})", data.len());
    }

    #[test]
    fn empty() {
        roundtrip(b"");
    }

    #[test]
    fn short_strings() {
        roundtrip(b"l");
        roundtrip(b"Ligma balls!");
        roundtrip(b"hello ligma, hello ligma balls");
    }

    #[test]
    fn all_bytes() {
        let vector: Vec<u8> = (0..=255u8).cycle().take(10_000).collect();
        roundtrip(&vector);
    }

    #[test]
    fn skewed_compresses() {
        // Long run forces frequency halving (MAX_FREQUENCY) many times.
        let vector = vec![b'x'; 200_000];
        let compress = compress(&vector);
        assert!(
            compress.len() < 2_000,
            "compressed to {} bytes",
            compress.len()
        );
        roundtrip(&vector);
    }

    #[test]
    fn pseudo_random() {
        let mut x: u32 = 12345;
        let vector: Vec<u8> = (0..100_000)
            .map(|_| {
                x = x.wrapping_mul(1103515245).wrapping_add(12345);
                // Skewed distribution over a handful of letters.
                b"aaaabbbcdeeeeeefg"[(x >> 16) as usize % 17]
            })
            .collect();
        let compress = compress(&vector);
        assert!(compress.len() < vector.len() / 2);
        roundtrip(&vector);
    }
}
