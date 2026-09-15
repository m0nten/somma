use rayon::{iter::ParallelIterator, slice::ParallelSlice};

#[inline(always)]
pub fn par_nrm2_f32(x: &[f32], chunk_size: usize) -> f32 {
    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut scale = 0.0f32;
            for &ax in chunk.iter() {
                scale = scale.max(ax.abs());
            }

            if scale == 0.0 { (0.0f32, 0.0f32) } else { (scale, raw_nrm2_f32(chunk, scale)) }
        })
        .reduce(
            || (0.0f32, 0.0f32),
            |(s1, sum1), (s2, sum2)| {
                if s1 == 0.0 {
                    return (s2, sum2);
                }
                if s2 == 0.0 {
                    return (s1, sum1);
                }

                if s1 >= s2 {
                    let r = s2 / s1;
                    (s1, sum1 + sum2 * (r * r))
                } else {
                    let r = s1 / s2;
                    (s2, sum2 + sum1 * (r * r))
                }
            },
        );

    scale * sq_sum.sqrt()
}

#[inline(always)]
pub fn par_nrm2_f64(x: &[f64], chunk_size: usize) -> f64 {
    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut scale = 0.0f64;
            for &ax in chunk.iter() {
                scale = scale.max(ax.abs());
            }

            if scale == 0.0 { (0.0f64, 0.0f64) } else { (scale, raw_nrm2_f64(chunk, scale)) }
        })
        .reduce(
            || (0.0f64, 0.0f64),
            |(s1, sum1), (s2, sum2)| {
                if s1 == 0.0 {
                    return (s2, sum2);
                }
                if s2 == 0.0 {
                    return (s1, sum1);
                }

                if s1 >= s2 {
                    let r = s2 / s1;
                    (s1, sum1 + sum2 * (r * r))
                } else {
                    let r = s1 / s2;
                    (s2, sum2 + sum1 * (r * r))
                }
            },
        );

    scale * sq_sum.sqrt()
}

#[inline(always)]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let mut scale = 0.0f32;
    for ax in x.iter() {
        scale = scale.max(ax.abs());
    }

    if scale == 0.0 {
        return 0.0;
    }

    scale * raw_nrm2_f32(x, scale).sqrt()
}

#[inline(always)]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    for ax in x.iter() {
        scale = scale.max(ax.abs());
    }

    if scale == 0.0 {
        return 0.0;
    }

    scale * raw_nrm2_f64(x, scale).sqrt()
}

#[inline(always)]
fn raw_nrm2_f32(x: &[f32], scale: f32) -> f32 {
    let mut x = x;
    let mut acc = [0.0f32; 8];
    let inv_scale = 1.0 / scale;

    while x.len() >= 8 {
        let v0 = x[0] * inv_scale;
        let v1 = x[1] * inv_scale;
        let v2 = x[2] * inv_scale;
        let v3 = x[3] * inv_scale;
        let v4 = x[4] * inv_scale;
        let v5 = x[5] * inv_scale;
        let v6 = x[6] * inv_scale;
        let v7 = x[7] * inv_scale;

        acc[0] += v0 * v0;
        acc[1] += v1 * v1;
        acc[2] += v2 * v2;
        acc[3] += v3 * v3;
        acc[4] += v4 * v4;
        acc[5] += v5 * v5;
        acc[6] += v6 * v6;
        acc[7] += v7 * v7;

        x = &x[8..];
    }

    let mut acc = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);

    for ax in x.iter() {
        let v = ax * inv_scale;
        acc += v * v;
    }

    acc
}

#[inline(always)]
fn raw_nrm2_f64(x: &[f64], scale: f64) -> f64 {
    let mut x = x;
    let mut acc = [0.0f64; 8];
    let inv_scale = 1.0 / scale;

    while x.len() >= 8 {
        let v0 = x[0] * inv_scale;
        let v1 = x[1] * inv_scale;
        let v2 = x[2] * inv_scale;
        let v3 = x[3] * inv_scale;
        let v4 = x[4] * inv_scale;
        let v5 = x[5] * inv_scale;
        let v6 = x[6] * inv_scale;
        let v7 = x[7] * inv_scale;

        acc[0] += v0 * v0;
        acc[1] += v1 * v1;
        acc[2] += v2 * v2;
        acc[3] += v3 * v3;
        acc[4] += v4 * v4;
        acc[5] += v5 * v5;
        acc[6] += v6 * v6;
        acc[7] += v7 * v7;

        x = &x[8..];
    }

    let mut acc = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);

    for ax in x.iter() {
        let v = ax * inv_scale;
        acc += v * v;
    }

    acc
}
