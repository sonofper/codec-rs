# Claude's try

A context-mixing (CM) compressor, written from scratch in Rust with no dependencies.
It exists as a **benchmark** to beat in the Øving 6 compression competition, which is
ranked on `diverse.lyx` (= `test_files/180kb.lyx`).

## Results

| File | Original | gzip -9 | bzip2 -9 | xz -9e | **Claude's try** | bits/byte |
|---|---:|---:|---:|---:|---:|---:|
| `180kb.lyx` (diverse.lyx) | 184 300 | 10 750 | 8 894 | 8 996 | **6 368** | 0.276 |
| `34kb.lyx` (opg6-kompr.lyx) | 34 751 | 9 264 | 8 628 | 8 820 | **6 905** | 1.590 |
| `17kb.txt` (diverse.txt) | 16 973 | 5 410 | 5 243 | 5 184 | **4 100** | 1.933 |
| `595kb.txt` (Jules Verne) | 609 116 | 230 936 | 171 095 | 191 552 | **143 111** | 1.880 |
| `98mb.txt` (enwik8) | 100 000 000 | – | – | – | **19 949 726** | 1.596 |

enwik8 took 15 min to compress and 11 min to decompress, and it also roundtrips byte-identical.

Every file roundtrips byte-identical through separate compress and decompress runs.
Speed is roughly 80–100 KB/s each way: about 2 s for diverse.lyx.
Memory use is up to ~200 MB.

## Usage

```bash
cargo build --release
./target/release/claudes_try compress   diverse.lyx diverse.ct
./target/release/claudes_try decompress diverse.ct  diverse.out
./target/release/claudes_try bench file1 file2 ...   # compress+decompress+verify in memory
```

## How it works

Instead of emitting LZ references and Huffman codes, it predicts the file **one bit at a time**
and codes each bit with an arithmetic coder. A bit predicted with probability 0.99 costs about 0.015 bits.
All the work goes into predicting well:

1. **24 contexts** are hashed once per nibble into a shared table:
   * order 0–4, 8, 12, 20 (the last n bytes);
   * words (current word, plus the previous one or two words);
   * column / "character above" in the previous line;
   * sparse context;
   * the byte the match model expects.
   Each context stores a 1-byte *bit-history state* (bounded counts of 0s/1s + last bit)
   and a direct adaptive probability.
2. **LyX-aware contexts.** These gave the biggest single gain, about 2.5%:
   * it tracks the current `\begin_layout` / `\begin_inset` nesting (Itemize, Formula, table Text, …)
     and the table cell index, and uses them as contexts;
   * a "text-only stream" skips markup lines and line breaks, so prose split across lines
     reads as running text.
3. **Match model** (the LZ77 idea used as a predictor). It finds the last occurrence of the
   previous 5+ bytes and predicts the byte that followed it. Its confidence grows with match length.
4. **Mixing.** 75 inputs (stretched probabilities) feed 6 small neural networks (logistic mixing).
   Each network picks its weight set by a different context: partial byte, match length,
   previous byte, character classes, number of known orders, and document structure.
   A final mixer combines the six, and 4 APM/SSE stages refine the result.
5. Every model **learns online**. The decompressor runs the identical model, so nothing but the
   file length is stored. All floating-point math uses only + − × ÷ (exp and ln are computed
   with series), so results are bit-identical on any machine.

Source: `src/coder.rs` (arithmetic coder), `src/math.rs` (stretch/squash), `src/model.rs` (all modelling).

## Ideas if you want to beat it

* A classic LZ77 + Huffman lands around 9–11 KB on diverse.lyx (gzip, which is LZ77 + Huffman, gets 10 750).
  Beating 6 368 basically requires probability modelling + arithmetic coding (or ANS).
  Check with the teacher that this counts as "algorithms we have seen" before betting on it.
* Where the bits still go in diverse.lyx: ~73% is the Norwegian prose lines. The LyX markup is nearly free,
  and the novel preamble at the start costs ~1.8 KB.
* Unexplored directions: an LSTM/GRU byte model as an extra mixer input (cmix-style);
  2-pass parameter search stored in the header; better state tables; a second, longer match model;
  SSE on the match model's expected byte.
