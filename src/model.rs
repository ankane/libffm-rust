use crate::disk::*;
use crate::error::Error;
use crate::node::Node;
use crate::params::Params;
use byteorder::{NativeEndian, ReadBytesExt};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::mem;
use std::path::Path;

#[cfg(target_arch = "x86")]
#[cfg(target_feature = "sse3")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "sse3")]
use std::arch::x86_64::*;

#[cfg(target_feature = "sse3")]
const ALIGN_BYTE: usize = 16;

#[cfg(not(target_feature = "sse3"))]
const ALIGN_BYTE: usize = 4;

const ALIGN: usize = ALIGN_BYTE / mem::size_of::<f32>();

#[inline(always)]
fn get_k_aligned(k: i32) -> usize {
    ((k as f32) / (ALIGN as f32)).ceil() as usize * ALIGN
}

// Memory appears to be aligned for SIMD
// since operations will segfault if not aligned
// https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm_load_ps.html
//
// Vec<T> is automatically aligned for T
// https://users.rust-lang.org/t/how-can-i-allocate-aligned-memory-in-rust/33293
fn malloc_aligned_float(size: usize) -> Vec<f32> {
    vec![0.0; size]
}

/// A model.
#[derive(Debug, Default)]
pub struct Model {
    pub(crate) n: i32,
    pub(crate) m: i32,
    pub(crate) k: i32,
    pub(crate) w: Vec<f32>,
    pub(crate) normalization: bool,
}

impl Model {
    /// Returns a new set of parameters.
    pub fn params() -> Params {
        Params::new()
    }

    pub(crate) fn new(n: i32, m: i32, param: &Params) -> Self {
        let k_aligned = get_k_aligned(param.k);

        let mut model = Self {
            n,
            k: param.k,
            m,
            w: malloc_aligned_float(n as usize * m as usize * k_aligned * 2),
            normalization: param.normalization,
        };

        let coef = 1.0 / (model.k as f32).sqrt();
        let w = &mut model.w;

        let mut rng = thread_rng();

        let mut w_offset = 0;

        for _ in 0..model.n {
            for _ in 0..model.m {
                let mut d = 0;
                while d < k_aligned {
                    let mut s = 0;
                    while s < ALIGN {
                        w[w_offset] = if d < model.k as usize { coef * rng.gen_range(0.0..1.0) } else { 0.0 };
                        w[w_offset + ALIGN] = 1.0;
                        s += 1;
                        w_offset += 1;
                        d += 1;
                    }
                    w_offset += ALIGN;
                }
            }
        }

        model
    }

    /// Loads a model from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut f_in = File::open(path)?;

        let mut model = Self {
            n: f_in.read_i32::<NativeEndian>()?,
            m: f_in.read_i32::<NativeEndian>()?,
            k: f_in.read_i32::<NativeEndian>()?,
            normalization: f_in.read_u8()? != 0,
            ..Self::default()
        };

        let w_size = model.get_w_size();
        model.w = vec![0.0; w_size];
        f_in.read_f32_into::<NativeEndian>(&mut model.w)?;

        Ok(model)
    }

    /// Saves the model to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut f_out = File::create(path)?;
        f_out.write_all(&self.n.to_ne_bytes())?;
        f_out.write_all(&self.m.to_ne_bytes())?;
        f_out.write_all(&self.k.to_ne_bytes())?;
        f_out.write_all(&(self.normalization as u8).to_ne_bytes())?;
        f_out.write_all(&self.w.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>())?;
        Ok(())
    }

    /// Trains a model.
    pub fn train<P: AsRef<Path>>(tr_path: P) -> Result<Self, Error> {
        Params::new().train(tr_path)
    }

    /// Trains a model and performs cross-validation.
    pub fn train_eval<P: AsRef<Path>, Q: AsRef<Path>>(tr_path: P, va_path: Q) -> Result<Self, Error> {
        Params::new().train_eval(tr_path, va_path)
    }

    /// Returns predictions.
    pub fn predict<P: AsRef<Path>>(&self, path: P) -> Result<(Vec<f32>, f64), Error> {
        let mut predictions = Vec::new();
        let mut loss: f64 = 0.0;
        let mut x: Vec<Node> = Vec::new();

        let f_in = File::open(path)?;
        for (i, line_option) in BufReader::new(f_in).lines().enumerate() {
            let line = line_option?;
            x.clear();

            let mut tokens = line.split(&[' ', '\t'][..]);
            let y = parse_y(tokens.next(), i)?;

            for (j, token) in tokens.enumerate() {
                if token.is_empty() {
                    break;
                }
                x.push(Node::parse(token, i, j)?);
            }

            let y_bar = self.predict_nodes(&x);

            loss -= if y == 1.0 { y_bar.ln() } else { (1.0 - y_bar).ln() } as f64;

            predictions.push(y_bar);
        }

        loss /= predictions.len() as f64;

        Ok((predictions, loss))
    }

    fn predict_nodes(&self, nodes: &[Node]) -> f32 {
        let mut r = 1.0;
        if self.normalization {
            r = 0.0;
            for n in nodes {
                r += n.v * n.v;
            }
            r = 1.0 / r;
        }

        let t = self.wtx(nodes, r);
        1.0 / (1.0 + (-t).exp())
    }

    pub(crate) fn one_epoch<W: Read + Write + Seek>(&mut self, prob: &mut ProblemOnDisk<W>, do_update: bool, param: &Params) -> Result<f32, Error> {
        let mut rng = thread_rng();

        let mut loss = 0.0;

        let mut outer_order: Vec<usize> = (0..prob.meta.num_blocks as usize).collect();
        outer_order.shuffle(&mut rng);

        for blk in outer_order {
            let l = prob.load_block(blk as i32)?;

            let mut inner_order: Vec<usize> = (0..l).collect();
            inner_order.shuffle(&mut rng);

            for ii in inner_order.iter().take(l) {
                let i = *ii;
                let y = prob.y[i];

                let nodes = &prob.x[prob.p[i] as usize..prob.p[i + 1] as usize];

                let r = if param.normalization { prob.r[i] } else { 1.0 };

                let t = self.wtx(nodes, r);

                let expnyt = (-y * t).exp();

                loss += (1.0 + expnyt).ln();

                if do_update {
                    let kappa = -y * expnyt / (1.0 + expnyt);
                    self.wtx_update(nodes, r, kappa, param.eta, param.lambda);
                }
            }
        }

        Ok(loss / prob.meta.l as f32)
    }

    #[cfg(target_feature = "sse3")]
    fn wtx(&self, nodes: &[Node], r: f32) -> f32 {
        let align0 = 2 * get_k_aligned(self.k);
        let align1 = self.m as usize * align0;

        unsafe {
            let mut xmm_t = _mm_setzero_ps();

            for i1 in 0..nodes.len() {
                let n1 = &nodes[i1];
                let j1 = n1.j;
                let f1 = n1.f;
                let v1 = n1.v;
                if j1 >= self.n || f1 >= self.m {
                    continue;
                }

                for i2 in i1 + 1..nodes.len() {
                    let n2 = &nodes[i2];
                    let j2 = n2.j;
                    let f2 = n2.f;
                    let v2 = n2.v;
                    if j2 >= self.n || f2 >= self.m {
                        continue;
                    }

                    let w = &self.w;

                    let w1_base_offset = j1 as usize * align1 + f2 as usize * align0;
                    let w2_base_offset = j2 as usize * align1 + f1 as usize * align0;

                    let xmm_v = _mm_set1_ps(v1 * v2 * r);

                    let mut d = 0;
                    while d < align0 {
                        let xmm_w1 = _mm_load_ps(&w[w1_base_offset + d]);
                        let xmm_w2 = _mm_load_ps(&w[w2_base_offset + d]);

                        xmm_t = _mm_add_ps(xmm_t, _mm_mul_ps(_mm_mul_ps(xmm_w1, xmm_w2), xmm_v));

                        d += ALIGN * 2;
                    }
                }
            }

            xmm_t = _mm_hadd_ps(xmm_t, xmm_t);
            xmm_t = _mm_hadd_ps(xmm_t, xmm_t);
            let mut t = 0.0;
            _mm_store_ss(&mut t, xmm_t);

            t
        }
    }

    #[cfg(target_feature = "sse3")]
    fn wtx_update(&mut self, nodes: &[Node], r: f32, kappa: f32, eta: f32, lambda: f32) -> f32 {
        let align0 = 2 * get_k_aligned(self.k);
        let align1 = self.m as usize * align0;

        unsafe {
            let xmm_kappa = _mm_set1_ps(kappa);
            let xmm_eta = _mm_set1_ps(eta);
            let xmm_lambda = _mm_set1_ps(lambda);

            for i1 in 0..nodes.len() {
                let n1 = &nodes[i1];
                let j1 = n1.j;
                let f1 = n1.f;
                let v1 = n1.v;
                if j1 >= self.n || f1 >= self.m {
                    continue;
                }

                for i2 in i1 + 1..nodes.len() {
                    let n2 = &nodes[i2];
                    let j2 = n2.j;
                    let f2 = n2.f;
                    let v2 = n2.v;
                    if j2 >= self.n || f2 >= self.m {
                        continue;
                    }

                    let w = &mut self.w;

                    let w1_base_offset = j1 as usize * align1 + f2 as usize * align0;
                    let w2_base_offset = j2 as usize * align1 + f1 as usize * align0;

                    let xmm_v = _mm_set1_ps(v1 * v2 * r);

                    let xmm_kappav = _mm_mul_ps(xmm_kappa, xmm_v);

                    let mut d = 0;
                    while d < align0 {
                        let w1_offset = w1_base_offset + d;
                        let w2_offset = w2_base_offset + d;

                        let wg1_offset = w1_base_offset + d + ALIGN;
                        let wg2_offset = w2_base_offset + d + ALIGN;

                        let mut xmm_w1 = _mm_load_ps(&w[w1_offset]);
                        let mut xmm_w2 = _mm_load_ps(&w[w2_offset]);

                        let mut xmm_wg1 = _mm_load_ps(&w[wg1_offset]);
                        let mut xmm_wg2 = _mm_load_ps(&w[wg2_offset]);

                        let xmm_g1 = _mm_add_ps(
                            _mm_mul_ps(xmm_lambda, xmm_w1),
                            _mm_mul_ps(xmm_kappav, xmm_w2)
                        );
                        let xmm_g2 = _mm_add_ps(
                            _mm_mul_ps(xmm_lambda, xmm_w2),
                            _mm_mul_ps(xmm_kappav, xmm_w1)
                        );

                        xmm_wg1 = _mm_add_ps(xmm_wg1, _mm_mul_ps(xmm_g1, xmm_g1));
                        xmm_wg2 = _mm_add_ps(xmm_wg2, _mm_mul_ps(xmm_g2, xmm_g2));

                        xmm_w1 = _mm_sub_ps(
                            xmm_w1,
                            _mm_mul_ps(xmm_eta, _mm_mul_ps(_mm_rsqrt_ps(xmm_wg1), xmm_g1))
                        );
                        xmm_w2 = _mm_sub_ps(
                            xmm_w2,
                            _mm_mul_ps(xmm_eta, _mm_mul_ps(_mm_rsqrt_ps(xmm_wg2), xmm_g2))
                        );

                        _mm_store_ps(&mut w[w1_offset], xmm_w1);
                        _mm_store_ps(&mut w[w2_offset], xmm_w2);

                        _mm_store_ps(&mut w[wg1_offset], xmm_wg1);
                        _mm_store_ps(&mut w[wg2_offset], xmm_wg2);

                        d += ALIGN * 2;
                    }
                }
            }

            0.0
        }
    }

    #[cfg(not(target_feature = "sse3"))]
    fn wtx(&self, nodes: &[Node], r: f32) -> f32 {
        let align0 = 2 * get_k_aligned(self.k);
        let align1 = self.m as usize * align0;

        let mut t = 0.0;
        for i1 in 0..nodes.len() {
            let n1 = &nodes[i1];
            let j1 = n1.j;
            let f1 = n1.f;
            let v1 = n1.v;
            if j1 >= self.n || f1 >= self.m {
                continue;
            }

            for n2 in nodes.iter().skip(i1 + 1) {
                let j2 = n2.j;
                let f2 = n2.f;
                let v2 = n2.v;
                if j2 >= self.n || f2 >= self.m {
                    continue;
                }

                let w = &self.w;

                let w1_offset = j1 as usize * align1 + f2 as usize * align0;
                let w2_offset = j2 as usize * align1 + f1 as usize * align0;

                let v = v1 * v2 * r;

                let mut d = 0;
                while d < align0 {
                    t += w[w1_offset + d] * w[w2_offset + d] * v;
                    d += ALIGN * 2;
                }
            }
        }

        t
    }

    #[cfg(not(target_feature = "sse3"))]
    fn wtx_update(&mut self, nodes: &[Node], r: f32, kappa: f32, eta: f32, lambda: f32) -> f32 {
        let align0 = 2 * get_k_aligned(self.k);
        let align1 = self.m as usize * align0;

        for i1 in 0..nodes.len() {
            let n1 = &nodes[i1];
            let j1 = n1.j as usize;
            let f1 = n1.f as usize;
            let v1 = n1.v;
            if j1 >= self.n as usize || f1 >= self.m as usize {
                continue;
            }

            for n2 in nodes.iter().skip(i1 + 1) {
                let j2 = n2.j as usize;
                let f2 = n2.f as usize;
                let v2 = n2.v;
                if j2 >= self.n as usize || f2 >= self.m as usize {
                    continue;
                }

                let w = &mut self.w;

                let w1_offset = j1 * align1 + f2 * align0;
                let w2_offset = j2 * align1 + f1 * align0;

                let v = v1 * v2 * r;

                let wg1_offset = w1_offset + ALIGN;
                let wg2_offset = w2_offset + ALIGN;

                let mut d = 0;
                while d < align0 {
                    let g1 = lambda * w[w1_offset + d] + kappa * w[w2_offset + d] * v;
                    let g2 = lambda * w[w2_offset + d] + kappa * w[w1_offset + d] * v;

                    w[wg1_offset + d] += g1 * g1;
                    w[wg2_offset + d] += g2 * g2;

                    w[w1_offset + d] -= eta / w[wg1_offset + d].sqrt() * g1;
                    w[w2_offset + d] -= eta / w[wg2_offset + d].sqrt() * g2;

                    d += ALIGN * 2;
                }
            }
        }

        0.0
    }

    fn get_w_size(&self) -> usize {
        let k_aligned = get_k_aligned(self.k);
        self.n as usize * self.m as usize * k_aligned * 2
    }
}
