//! Logistic helper functions (stretch / squash).
//!
//! Everything here is built from + - * / only, so the tables come out
//! bit-identical on every machine. That matters: the compressor and the
//! decompressor must compute *exactly* the same probabilities, otherwise
//! decoding derails after the first difference.

const LN2: f64 = 0.693_147_180_559_945_309_417_232_121_458_18;

/// exp(x) using range reduction + Taylor series (deterministic).
pub fn exp_det(x: f64) -> f64 {
    let k = (x / LN2).round();
    let r = x - k * LN2;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..30 {
        term *= r / i as f64;
        sum += term;
    }
    let mut k = k as i64;
    while k > 0 {
        sum *= 2.0;
        k -= 1;
    }
    while k < 0 {
        sum *= 0.5;
        k += 1;
    }
    sum
}

/// ln(x) for x > 0 using the atanh series (deterministic).
pub fn ln_det(x: f64) -> f64 {
    let mut m = x;
    let mut e = 0i64;
    while m > 1.5 {
        m *= 0.5;
        e += 1;
    }
    while m < 0.75 {
        m *= 2.0;
        e -= 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..40 {
        sum += term / k;
        term *= s2;
        k += 2.0;
    }
    2.0 * sum + e as f64 * LN2
}

pub struct Logistic {
    /// stretch(p) = ln(p / (1 - p)) for 16-bit probabilities.
    stretch: Vec<f32>,
    /// squash(x) = 1 / (1 + e^-x) sampled on [-16, 16] in steps of 1/128.
    squash: Vec<f32>,
}

pub const SQ_RANGE: f32 = 15.99;

impl Logistic {
    pub fn new() -> Self {
        let stretch = (0..65536)
            .map(|i| {
                let p = (i as f64 + 0.5) / 65536.0;
                ln_det(p / (1.0 - p)) as f32
            })
            .collect();
        let squash = (0..=4096)
            .map(|i| {
                let x = (i as f64 - 2048.0) / 128.0;
                (1.0 / (1.0 + exp_det(-x))) as f32
            })
            .collect();
        Logistic { stretch, squash }
    }

    /// Stretch of a probability given with 16 bits of precision.
    #[inline(always)]
    pub fn st16(&self, p16: u32) -> f32 {
        self.stretch[p16 as usize]
    }

    #[inline(always)]
    pub fn squash(&self, x: f32) -> f32 {
        let x = x.clamp(-SQ_RANGE, SQ_RANGE);
        let t = (x + 16.0) * 128.0;
        let i = t as usize;
        let f = t - i as f32;
        self.squash[i] * (1.0 - f) + self.squash[i + 1] * f
    }
}
