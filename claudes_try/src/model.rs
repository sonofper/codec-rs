//! The predictor: given everything seen so far, estimate P(next bit = 1).
//!
//! Architecture (a small member of the PAQ family, written from scratch):
//!
//!   24 contexts (order-1..order-20, words, columns, LyX structure, ...)
//!        |  each context hashed (per nibble) into a table of 1-byte
//!        |  "bit-history states" + a direct adaptive probability
//!        v
//!   StateMaps: bit-history state -> adaptive probability
//!        |                                   match model (long repeats)
//!        v                                          |
//!   6 neural mixers (weight sets chosen by small contexts)
//!        v
//!   final mixer -> 4 APM / SSE refinement stages -> arithmetic coder

use crate::math::Logistic;

// ---------------------------------------------------------------------------
// Tuned constants (tuned on diverse.lyx, checked on the other test files).
// ---------------------------------------------------------------------------

const MIXER_LR: f32 = 0.005; // learning rate of the first-layer mixers
const MIXER_W0: f32 = 0.02; // initial weight of the first-layer mixers
const FINAL_LR: f32 = 0.001; // learning rate of the final mixer
const FINAL_W0: f32 = 0.25; // initial weight of the final mixer
const LR_BOOST: f32 = 2.0; // weight sets learn (1 + boost) times faster when new ...
const LR_DECAY: f32 = 300.0; // ... decaying over roughly this many updates
const APM_RATE: f32 = 0.02;
const STATEMAP_LIMIT: u16 = 1023;
const DIRECT_LIMIT: usize = 30;
const MATCH_MIN: usize = 5;

// ---------------------------------------------------------------------------
// Bit-history states.
//
// One byte summarises what a context has seen: (n0, n1, last bit), where the
// counts are bounded and the "opposite" count is discounted when a new bit
// arrives (so the state adapts when a context's behaviour changes).
// ---------------------------------------------------------------------------

pub struct States {
    pub next: [[u8; 2]; 256],
    pub n0: [u16; 256],
    pub n1: [u16; 256],
    pub total: [u8; 256],
}

/// Largest allowed count for the bigger counter when the smaller one is `a`.
fn state_limit(a: u32) -> u32 {
    match a {
        0 => 40,
        1 => 24,
        2 => 12,
        3 => 8,
        4 => 6,
        5 => 5,
        _ => 0,
    }
}

fn state_allowed(n0: u32, n1: u32) -> bool {
    let (a, b) = if n0 < n1 { (n0, n1) } else { (n1, n0) };
    b <= state_limit(a)
}

impl States {
    pub fn new() -> Self {
        use std::collections::HashMap;
        let key = |n0: u32, n1: u32, last: u32| -> (u32, u32, u32) {
            (n0, n1, if n0 > 0 && n1 > 0 { last } else { 0 })
        };
        // Breadth-first enumeration of all reachable states from (0, 0).
        let mut list = vec![(0u32, 0u32, 0u32)];
        let mut map: HashMap<(u32, u32, u32), usize> = HashMap::new();
        map.insert((0, 0, 0), 0);
        let mut next = [[0u8; 2]; 256];
        let mut i = 0;
        while i < list.len() {
            let (n0, n1, _) = list[i];
            for y in 0..2u32 {
                // a = opposite count, b = count being incremented
                let (mut a, mut b) = if y == 1 { (n0, n1) } else { (n1, n0) };
                b += 1;
                if a > 2 {
                    a = a / 2 + 1;
                }
                loop {
                    let (t0, t1) = if y == 1 { (a, b) } else { (b, a) };
                    if state_allowed(t0, t1) {
                        break;
                    }
                    if b > a { b -= 1 } else { a -= 1 }
                }
                let (t0, t1) = if y == 1 { (a, b) } else { (b, a) };
                let k = key(t0, t1, y);
                let idx = *map.entry(k).or_insert_with(|| {
                    list.push(k);
                    list.len() - 1
                });
                next[i][y as usize] = idx as u8;
            }
            i += 1;
        }
        assert!(list.len() <= 256, "too many states: {}", list.len());
        let mut s = States { next, n0: [0; 256], n1: [0; 256], total: [0; 256] };
        for (i, &(n0, n1, _)) in list.iter().enumerate() {
            s.n0[i] = n0 as u16;
            s.n1[i] = n1 as u16;
            s.total[i] = (n0 + n1).min(255) as u8;
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Adaptive probability maps.
// ---------------------------------------------------------------------------

/// RECIP[n] = 65536 / (n + 1.5): learning rate after n observations.
fn recip_table() -> Vec<u32> {
    (0..1024).map(|n| (65536.0 * 2.0 / (2 * n + 3) as f64) as u32).collect()
}

/// Maps a small context to an adaptive probability (32-bit precision).
/// Learning rate starts at 1/1.5 and decays to 1/limit.
struct AdaptiveMap {
    p: Vec<u32>,
    n: Vec<u16>,
    limit: u16,
}

impl AdaptiveMap {
    fn new(size: usize, limit: u16) -> Self {
        AdaptiveMap { p: vec![1 << 31; size], n: vec![0; size], limit }
    }
    #[inline(always)]
    fn p16(&self, i: usize) -> u32 {
        self.p[i] >> 16
    }
    #[inline(always)]
    fn update(&mut self, i: usize, bit: u32, rec: &[u32]) {
        let n = self.n[i];
        let p = self.p[i] as i64;
        let target: i64 = if bit != 0 { 0xFFFF_FFFF } else { 0 };
        let np = p + (((target - p) * rec[n as usize] as i64) >> 16);
        self.p[i] = np.clamp(0, 0xFFFF_FFFF) as u32;
        if n < self.limit {
            self.n[i] = n + 1;
        }
    }
}

/// Adaptive Probability Map (a.k.a. SSE): refines a probability given a
/// context, by interpolating between 33 buckets over the stretched input.
struct Apm {
    t: Vec<f32>,
    idx: usize,
    rate: f32,
}

impl Apm {
    fn new(contexts: usize, rate: f32, lg: &Logistic) -> Self {
        let mut t = vec![0.0f32; contexts * 33];
        for c in 0..contexts {
            for j in 0..33 {
                t[c * 33 + j] = lg.squash((j as f32 - 16.0) * 0.5);
            }
        }
        Apm { t, idx: 0, rate }
    }
    #[inline(always)]
    fn pp(&mut self, st: f32, ctx: usize) -> f32 {
        let s = (st.clamp(-7.999, 7.999) + 8.0) * 2.0;
        let lo = s as usize;
        let w = s - lo as f32;
        self.idx = ctx * 33 + lo + if w > 0.5 { 1 } else { 0 };
        let base = ctx * 33 + lo;
        self.t[base] * (1.0 - w) + self.t[base + 1] * w
    }
    #[inline(always)]
    fn update(&mut self, bit: u32) {
        let y = bit as f32;
        let v = &mut self.t[self.idx];
        *v += (y - *v) * self.rate;
    }
}

// ---------------------------------------------------------------------------
// Neural mixer: logistic regression on stretched probabilities, with the
// weight vector chosen by a small context, trained online by gradient descent
// on coding cost. Each weight set learns faster while it is still new.
// ---------------------------------------------------------------------------

struct Mixer {
    n: usize,
    w: Vec<f32>,
    cnt: Vec<u32>,
    set: usize,
    dot: f32,
    p: f32,
    lr: f32,
}

const SQ: f32 = 15.0;

impl Mixer {
    fn new(n: usize, nsel: usize, lr: f32, w0: f32) -> Self {
        Mixer { n, w: vec![w0; n * nsel], cnt: vec![0; nsel], set: 0, dot: 0.0, p: 0.5, lr }
    }
    #[inline(always)]
    fn mix(&mut self, x: &[f32], sel: usize, lg: &Logistic) -> f32 {
        self.set = sel;
        let w = &self.w[sel * self.n..(sel + 1) * self.n];
        let mut s = 0.0f32;
        for i in 0..self.n {
            s += w[i] * x[i];
        }
        self.dot = s.clamp(-SQ, SQ);
        self.p = lg.squash(self.dot);
        self.dot
    }
    #[inline(always)]
    fn update(&mut self, x: &[f32], bit: u32) {
        let c = self.cnt[self.set];
        self.cnt[self.set] = c.saturating_add(1);
        let err = (bit as f32 - self.p) * self.lr * (1.0 + LR_BOOST * LR_DECAY / (LR_DECAY + c as f32));
        let w = &mut self.w[self.set * self.n..(self.set + 1) * self.n];
        for i in 0..self.n {
            w[i] += err * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Hash table of bit-history states.
//
// Contexts are looked up once per nibble. Each 16-byte slot holds a 1-byte
// checksum + 15 states (one per node of the 4-level binary tree of a nibble).
// 4 slots share a 64-byte cache line; on a miss the least used one is evicted.
// A parallel array holds a direct adaptive probability for every node.
// ---------------------------------------------------------------------------

struct HashTable {
    t: Vec<u8>,
    dp: Vec<u16>,
    mask: usize, // mask for slot-group index (in units of 4 slots)
}

impl HashTable {
    fn new(bits: u32) -> Self {
        let slots = 1usize << bits;
        HashTable { t: vec![0u8; slots * 16], dp: vec![32768u16; slots * 16], mask: (slots >> 2) - 1 }
    }

    /// Returns byte offset of the slot (its checksum byte) for hash `h`.
    #[inline(always)]
    fn find(&mut self, h: u64, st: &States) -> usize {
        let chk = (h >> 56) as u8;
        let base = ((h as usize) & self.mask) * 64;
        let mut best = base;
        let mut best_pri = u32::MAX;
        for i in 0..4 {
            let off = base + i * 16;
            if self.t[off] == chk {
                return off;
            }
            let pri = st.total[self.t[off + 1] as usize] as u32;
            if pri < best_pri {
                best_pri = pri;
                best = off;
            }
        }
        self.t[best..best + 16].fill(0);
        self.dp[best..best + 16].fill(32768);
        self.t[best] = chk;
        best
    }
}

#[inline(always)]
fn hash2(a: u64, b: u64) -> u64 {
    let mut h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_add(0x632B_E59B_D9B4_E019).rotate_left(23);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 29;
    h
}

// ---------------------------------------------------------------------------
// Match model: finds the most recent occurrence of the last MATCH_MIN bytes
// and predicts that the byte which followed it will follow again.
// (This is the Lempel-Ziv idea, used as a predictor instead of a copy command.)
// ---------------------------------------------------------------------------

struct MatchModel {
    table: Vec<u32>,
    mask: usize,
    ptr: usize,
    len: usize,
    miss: bool, // predicted byte already contradicted in the current byte
    sm: AdaptiveMap,
    ctx: usize,
}

impl MatchModel {
    fn new(bits: u32) -> Self {
        MatchModel {
            table: vec![0; 1 << bits],
            mask: (1 << bits) - 1,
            ptr: 0,
            len: 0,
            miss: false,
            sm: AdaptiveMap::new(64, 1023),
            ctx: 0,
        }
    }

    fn update_byte(&mut self, buf: &[u8]) {
        let pos = buf.len();
        if self.len > 0 && !self.miss && self.ptr < pos {
            self.len = (self.len + 1).min(65535);
            self.ptr += 1;
        } else {
            self.len = 0;
        }
        self.miss = false;
        if pos >= MATCH_MIN {
            let mut h: u64 = 0;
            for &b in &buf[pos - MATCH_MIN..] {
                h = (h.wrapping_add(b as u64 + 1)).wrapping_mul(0x2F0F_3D7B_9A3C_6E15);
            }
            let hi = (h >> 40) as usize & self.mask;
            if self.len == 0 {
                let cand = self.table[hi] as usize;
                if cand > 0 {
                    let mut l = 0;
                    while l < cand && l < 65535 && buf[cand - 1 - l] == buf[pos - 1 - l] {
                        l += 1;
                    }
                    if l >= MATCH_MIN {
                        self.len = l;
                        self.ptr = cand;
                    }
                }
            }
            self.table[hi] = pos as u32;
        }
    }

    #[inline(always)]
    fn len_bucket(&self) -> usize {
        match self.len {
            _ if self.miss => 0,
            0..16 => self.len,
            16..32 => 16 + (self.len - 16) / 4,
            32..64 => 20 + (self.len - 32) / 8,
            64..512 => 24 + (self.len - 64) / 64,
            _ => 31,
        }
    }

    /// Expected next byte (if a match is active).
    #[inline(always)]
    fn expected(&self, buf: &[u8]) -> Option<u8> {
        if self.len > 0 { Some(buf[self.ptr]) } else { None }
    }
}

// ---------------------------------------------------------------------------
// The full model.
// ---------------------------------------------------------------------------

const N_CTX: usize = 24;
const N_INPUTS: usize = N_CTX * 3 + 3;
const N_MIXERS: usize = 6;

pub struct Model {
    lg: Logistic,
    st: States,
    rec: Vec<u32>,
    ht: HashTable,
    sms: Vec<AdaptiveMap>,
    mm: MatchModel,
    mixers: Vec<Mixer>,
    final_mixer: Mixer,
    apms: Vec<Apm>,

    buf: Vec<u8>,
    c0: u32, // bits of the current byte so far, with a leading 1
    bitpos: u32,
    c4: u32, // last 4 bytes
    c8: u64, // last 8 bytes
    ctx: [u64; N_CTX],
    slot: [usize; N_CTX],
    node: usize, // position in the current nibble's binary tree (1..15)
    x: Vec<f32>,
    x2: Vec<f32>,
    pr: u32,

    // word / line state
    word: u64,
    word1: u64,
    word2: u64,
    line_start: usize,
    prev_line_start: usize,
    first_of_line: u8,

    // LyX document structure
    layouts: Vec<u64>,
    insets: Vec<u64>,
    cell: u32,

    // "text only" stream: prose lines with markup lines and line breaks removed
    in_text: bool,
    t8: u64,
    tword: u64,
    tword1: u64,
    tword2: u64,
}

impl Model {
    pub fn new(len_hint: usize) -> Self {
        let lg = Logistic::new();
        // Size the hash table to the input: ~8 slots per byte, 2^18..2^22 slots (max ~192 MB).
        let bits = (len_hint.max(1) * 8).next_power_of_two().trailing_zeros().clamp(18, 22);
        let mbits = (len_hint.max(1) * 2).next_power_of_two().trailing_zeros().clamp(16, 24);
        let mixers = vec![
            Mixer::new(N_INPUTS, 256, MIXER_LR, MIXER_W0),         // partial byte c0
            Mixer::new(N_INPUTS, 32 * 8, MIXER_LR, MIXER_W0),      // match length x bit position
            Mixer::new(N_INPUTS, 256 * 8, MIXER_LR, MIXER_W0),     // previous byte x bit position
            Mixer::new(N_INPUTS, 11 * 11 * 8, MIXER_LR, MIXER_W0), // char classes of last 2 bytes
            Mixer::new(N_INPUTS, 21 * 2 * 8, MIXER_LR, MIXER_W0),  // #known orders x match x bitpos
            Mixer::new(N_INPUTS, 128 * 8, MIXER_LR, MIXER_W0),     // document structure
        ];
        let apms = vec![
            Apm::new(256, APM_RATE, &lg),
            Apm::new(65536, APM_RATE, &lg),
            Apm::new(65536, APM_RATE, &lg),
            Apm::new(65536, APM_RATE, &lg),
        ];
        let mut m = Model {
            lg,
            st: States::new(),
            rec: recip_table(),
            ht: HashTable::new(bits),
            sms: (0..N_CTX).map(|_| AdaptiveMap::new(256, STATEMAP_LIMIT)).collect(),
            mm: MatchModel::new(mbits),
            mixers,
            final_mixer: Mixer::new(N_MIXERS + 1, 1, FINAL_LR, FINAL_W0),
            apms,
            buf: Vec::with_capacity(len_hint),
            c0: 1,
            bitpos: 0,
            c4: 0,
            c8: 0,
            ctx: [0; N_CTX],
            slot: [0; N_CTX],
            node: 1,
            x: vec![0.0; N_INPUTS],
            x2: vec![0.0; N_MIXERS + 1],
            pr: 32768,
            word: 0,
            word1: 0,
            word2: 0,
            line_start: 0,
            prev_line_start: 0,
            first_of_line: 0,
            layouts: Vec::new(),
            insets: Vec::new(),
            cell: 0,
            in_text: false,
            t8: 0,
            tword: 0,
            tword1: 0,
            tword2: 0,
        };
        m.compute_contexts();
        m.lookup_slots();
        m.predict();
        m
    }

    /// Probability that the next bit is 1, as 16-bit integer in 1..=65535.
    #[inline(always)]
    pub fn p(&self) -> u32 {
        self.pr
    }

    fn char_class(c: u8) -> usize {
        match c {
            b'a'..=b'z' => 1,
            b'A'..=b'Z' => 2,
            b'0'..=b'9' => 3,
            b' ' => 4,
            b'\n' => 5,
            b'\\' => 6,
            b'.' | b',' | b':' | b';' | b'!' | b'?' => 7,
            b'{' | b'}' | b'[' | b']' | b'(' | b')' => 8,
            128..=255 => 9,
            _ => 10,
        }
    }

    /// Hash describing where we are in the LyX document structure:
    /// innermost layout + innermost inset + nesting depth.
    fn struct_ctx(&self) -> u64 {
        let l = self.layouts.last().copied().unwrap_or(0);
        let i = self.insets.last().copied().unwrap_or(0);
        hash2(hash2(l, i), (self.layouts.len() + 8 * self.insets.len()).min(63) as u64)
    }

    /// Track \begin_layout / \begin_inset nesting (and table cells) from a completed line.
    fn structure_line(&mut self, start: usize, end: usize) {
        let line = &self.buf[start..end];
        let hash_bytes = |b: &[u8]| b.iter().fold(0x1234u64, |h, &c| hash2(h, c as u64));
        if let Some(rest) = line.strip_prefix(br"\begin_layout ") {
            self.layouts.push(hash_bytes(rest) | 1);
        } else if line == br"\end_layout" {
            self.layouts.pop();
        } else if let Some(rest) = line.strip_prefix(br"\begin_inset ") {
            // inset type = first word, e.g. "Formula", "Argument", "Text"
            let kind: &[u8] = rest.split(|&c| c == b' ').next().unwrap_or(rest);
            self.insets.push(hash_bytes(kind) | 1);
        } else if line == br"\end_inset" {
            self.insets.pop();
        } else if line.starts_with(b"<row") {
            self.cell = 0;
        } else if line.starts_with(b"<cell") {
            self.cell += 1;
        }
        if self.layouts.len() > 64 {
            self.layouts.remove(0);
        }
        if self.insets.len() > 64 {
            self.insets.remove(0);
        }
    }

    /// Compute the hash of every context; called once per byte.
    fn compute_contexts(&mut self) {
        let buf = &self.buf;
        let pos = buf.len();
        let back = |i: usize| -> u64 { if i <= pos { buf[pos - i] as u64 } else { 0 } };
        let c4 = self.c4 as u64;
        let c8 = self.c8;

        let col = (pos - self.line_start) as u64;
        let prev_len = self.line_start - self.prev_line_start;
        let above = if (col as usize) < prev_len { buf[self.prev_line_start + col as usize] as u64 } else { 256 };
        let sc = self.struct_ctx();

        let mut h = [0u64; N_CTX];
        // plain order-n contexts
        h[0] = 0;
        h[1] = c4 & 0xFF;
        h[2] = c4 & 0xFFFF;
        h[3] = c4 & 0xFF_FFFF;
        h[4] = c4;
        h[7] = c8;
        let mut o12 = c8;
        for i in 9..=12 {
            o12 = hash2(o12, back(i));
        }
        h[8] = o12;
        let mut o20 = o12;
        for i in 13..=20 {
            o20 = hash2(o20, back(i));
        }
        h[9] = o20;
        // words (case-folded letters, UTF-8 bytes count as letters)
        h[10] = hash2(self.word, c4 & 0xFF);
        h[11] = hash2(self.word, self.word1);
        h[12] = hash2(hash2(self.word, self.word1), self.word2);
        // column / layout contexts
        h[13] = hash2(above, col.min(63));
        h[14] = hash2(above, c4 & 0xFF);
        h[15] = c4 & 0xFFFF_0000; // sparse: skips the last byte
        h[16] = hash2(self.first_of_line as u64, hash2(col.min(31), c4 & 0xFFFF));
        // byte predicted by the match model
        h[17] = match self.mm.expected(buf) {
            Some(e) => hash2(e as u64, self.mm.len.min(15) as u64) ^ 1,
            None => 0,
        };
        // LyX structure contexts
        h[5] = hash2(sc, c4 & 0xFFFF);
        h[6] = hash2(sc, hash2(self.word, c4 & 0xFF));
        h[18] = hash2(sc, c4 & 0xFF_FFFF);
        h[19] = hash2(hash2(sc, self.cell as u64), c4 & 0xFFFF);
        h[20] = hash2(hash2(sc, self.word), self.word1);
        // text-only stream contexts
        h[21] = self.t8 & 0xFF_FFFF;
        h[22] = hash2(self.tword, self.tword1);
        h[23] = hash2(hash2(self.tword, self.tword1), self.tword2);

        for i in 0..N_CTX {
            self.ctx[i] = hash2(h[i], (i as u64 + 1) << 44);
        }
    }

    /// Find the hash table slots for the current nibble; called twice per byte.
    #[inline(always)]
    fn lookup_slots(&mut self) {
        let nib = if self.bitpos == 0 { 0 } else { self.c0 as u64 };
        for i in 0..N_CTX {
            let h = if nib == 0 { self.ctx[i] } else { hash2(self.ctx[i], nib) };
            self.slot[i] = self.ht.find(h, &self.st);
        }
        self.node = 1;
    }

    fn predict(&mut self) {
        let lg = &self.lg;

        // --- context model inputs: 3 per context ---
        let mut known = 0usize;
        for i in 0..N_CTX {
            let off = self.slot[i] + self.node;
            let s = self.ht.t[off] as usize;
            if (1..=9).contains(&i) && s != 0 {
                known += 1;
            }
            // 1) what this bit-history usually leads to (learned per context type)
            self.x[3 * i] = lg.st16(self.sms[i].p16(s));
            // 2) confidence boost for deterministic histories (only 0s or only 1s seen)
            let (n0, n1) = (self.st.n0[s] as f32, self.st.n1[s] as f32);
            self.x[3 * i + 1] = if n0 == 0.0 && n1 > 0.0 {
                soft_count(n1 + 1.0)
            } else if n1 == 0.0 && n0 > 0.0 {
                -soft_count(n0 + 1.0)
            } else {
                0.0
            };
            // 3) direct adaptive probability stored in this context's slot
            self.x[3 * i + 2] = lg.st16(self.ht.dp[off] as u32);
        }

        // --- match model inputs ---
        let base = 3 * N_CTX;
        let lb = self.mm.len_bucket();
        if lb > 0 {
            let e = self.buf[self.mm.ptr] as u32 | 0x100;
            let expected_bit = (e >> (7 - self.bitpos)) & 1;
            self.mm.ctx = lb * 2 + expected_bit as usize;
            self.x[base] = lg.st16(self.mm.sm.p16(self.mm.ctx));
            let strength = lb as f32 / 8.0;
            self.x[base + 1] = if expected_bit == 1 { strength } else { -strength };
        } else {
            self.mm.ctx = 0;
            self.x[base] = 0.0;
            self.x[base + 1] = 0.0;
        }
        self.x[base + 2] = 0.5; // bias

        // --- mixers ---
        let bp = self.bitpos as usize;
        let c1 = (self.c4 & 0xFF) as usize;
        let sel = [
            self.c0 as usize,
            lb * 8 + bp,
            c1 * 8 + bp,
            (Self::char_class(c1 as u8) * 11 + Self::char_class((self.c4 >> 8) as u8)) * 8 + bp,
            (known * 2 + (lb > 0) as usize) * 8 + bp,
            (self.struct_ctx() as usize & 127) * 8 + bp,
        ];
        for (k, m) in self.mixers.iter_mut().enumerate() {
            self.x2[k] = m.mix(&self.x, sel[k], lg);
        }
        self.x2[N_MIXERS] = 0.5;
        let dot = self.final_mixer.mix(&self.x2, 0, lg);
        let pm = lg.squash(dot);

        // --- APM / SSE stages ---
        let c0 = self.c0 as usize;
        let a0 = self.apms[0].pp(dot, c0);
        let a1 = self.apms[1].pp(dot, c0 | (c1 << 8));
        let h2 = (hash2(self.c4 as u64 & 0xFFFF, c0 as u64) & 0xFFFF) as usize;
        let a2 = self.apms[2].pp(dot, h2);
        let h3 = (hash2(self.struct_ctx() ^ (c1 as u64) << 8, c0 as u64) & 0xFFFF) as usize;
        let a3 = self.apms[3].pp(dot, h3);
        let p = (pm * 2.0 + a0 + a1 * 2.0 + a2 * 2.0 + a3) / 8.0;
        self.pr = ((p * 65536.0) as i32).clamp(1, 65535) as u32;
    }

    /// Learn from the actual bit, then prepare the prediction for the next one.
    pub fn update(&mut self, bit: u32) {
        let rec = &self.rec;
        for i in 0..N_CTX {
            let off = self.slot[i] + self.node;
            let s = self.ht.t[off] as usize;
            self.sms[i].update(s, bit, rec);
            let n = (self.st.n0[s] + self.st.n1[s]) as usize;
            let r = rec[n.min(DIRECT_LIMIT)] as i32;
            let d = self.ht.dp[off] as i32;
            let target = if bit != 0 { 65535 } else { 0 };
            self.ht.dp[off] = (d + (((target - d) * r) >> 16)).clamp(16, 65519) as u16;
            self.ht.t[off] = self.st.next[s][bit as usize];
        }
        if self.mm.ctx > 0 {
            self.mm.sm.update(self.mm.ctx, bit, rec);
        }
        if self.mm.len > 0 && !self.mm.miss {
            let e = self.buf[self.mm.ptr] as u32 | 0x100;
            if (e >> (7 - self.bitpos)) & 1 != bit {
                self.mm.miss = true;
            }
        }
        for m in self.mixers.iter_mut() {
            m.update(&self.x, bit);
        }
        self.final_mixer.update(&self.x2, bit);
        for a in self.apms.iter_mut() {
            a.update(bit);
        }

        self.c0 = (self.c0 << 1) | bit;
        self.bitpos += 1;
        self.node = self.node * 2 + bit as usize;
        if self.bitpos == 8 {
            let c = (self.c0 & 0xFF) as u8;
            self.byte_update(c);
            self.c0 = 1;
            self.bitpos = 0;
            self.compute_contexts();
            self.lookup_slots();
        } else if self.bitpos == 4 {
            self.lookup_slots();
        }
        self.predict();
    }

    fn byte_update(&mut self, c: u8) {
        self.buf.push(c);
        self.c4 = (self.c4 << 8) | c as u32;
        self.c8 = (self.c8 << 8) | c as u64;
        let pos = self.buf.len();

        let is_letter = c.is_ascii_alphabetic() || c >= 128;
        if is_letter {
            self.word = hash2(self.word, c.to_ascii_lowercase() as u64);
        } else if self.word != 0 {
            self.word2 = self.word1;
            self.word1 = self.word;
            self.word = 0;
        }

        // text-only stream: bytes of prose lines, without the line breaks
        if pos == self.line_start + 1 {
            self.in_text = !matches!(c, b'\\' | b'<' | b'#' | b'%' | b'\n') && !self.layouts.is_empty();
        }
        if self.in_text && c != b'\n' {
            self.t8 = (self.t8 << 8) | c as u64;
            if is_letter {
                self.tword = hash2(self.tword, c.to_ascii_lowercase() as u64);
            } else if self.tword != 0 {
                self.tword2 = self.tword1;
                self.tword1 = self.tword;
                self.tword = 0;
            }
        }

        if c == b'\n' {
            self.structure_line(self.line_start, pos - 1);
            self.prev_line_start = self.line_start;
            self.line_start = pos;
            self.first_of_line = 0;
        } else if pos == self.line_start + 1 {
            self.first_of_line = c;
        }
        self.mm.update_byte(&self.buf);
    }
}

/// Saturating curve for small counts (uses only + * /, so it is deterministic).
#[inline(always)]
fn soft_count(x: f32) -> f32 {
    x / (1.0 + x * 0.15)
}
