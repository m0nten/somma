use crate::arch::{avx2::Avx2, SimdArch};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

#[inline(always)]
pub unsafe fn nrm2_f32<S: SimdArch>(x: &[f32]) -> f32 {
    unsafe {
        let scale = raw_scale_nrm2_f32::<S>(x);
        if scale == 0.0 {
            return 0.0;
        }
        scale * raw_nrm2_f32::<S>(x, scale).sqrt()
    }
}

#[inline(always)]
pub unsafe fn nrm2_f64<S: SimdArch>(x: &[f64]) -> f64 {
    unsafe {
        let scale = raw_scale_nrm2_f64::<S>(x);
        if scale == 0.0 {
            return 0.0;
        }
        scale * raw_nrm2_f64::<S>(x, scale).sqrt()
    }
}

#[inline(always)]
pub unsafe fn par_nrm2_f32<S: SimdArch>(x: &[f32], chunk_size: usize) -> f32 {
    if x.len() < 32_768 {
        return unsafe { nrm2_f32::<S>(x) };
    }

    let chunk_size = chunk_size.clamp(2048, 8192);

    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| unsafe {
            let s = raw_scale_nrm2_f32::<S>(chunk);
            if s == 0.0 {
                (0.0f32, 0.0f32)
            } else {
                (s, raw_nrm2_f32::<S>(chunk, s))
            }
        })
        .reduce(
            || (0.0f32, 0.0f32),
            |(s1, sum1), (s2, sum2)| {
                if s1 == 0.0 { return (s2, sum2); }
                if s2 == 0.0 { return (s1, sum1); }

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
pub unsafe fn par_nrm2_f64<S: SimdArch>(x: &[f64], chunk_size: usize) -> f64 {
    if x.len() < 32_768 {
        return unsafe { nrm2_f64::<S>(x) };
    }

    let chunk_size = chunk_size.clamp(2048, 8192);

    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| unsafe {
            let s = raw_scale_nrm2_f64::<S>(chunk);
            if s == 0.0 {
                (0.0f64, 0.0f64)
            } else {
                (s, raw_nrm2_f64::<S>(chunk, s))
            }
        })
        .reduce(
            || (0.0f64, 0.0f64),
            |(s1, sum1), (s2, sum2)| {
                if s1 == 0.0 { return (s2, sum2); }
                if s2 == 0.0 { return (s1, sum1); }

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

// Вспомогательные функции с атрибутом target_feature для безопасного вызова внутри замыканий Rayon
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn helper_scale_avx2_f32(chunk: &[f32]) -> f32 {
    unsafe { raw_scale_nrm2_f32::<Avx2>(chunk) }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn helper_nrm2_avx2_f32(chunk: &[f32], scale: f32) -> f32 {
    unsafe { raw_nrm2_f32::<Avx2>(chunk, scale) }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn helper_scale_avx2_f64(chunk: &[f64]) -> f64 {
    unsafe { raw_scale_nrm2_f64::<Avx2>(chunk) }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn helper_nrm2_avx2_f64(chunk: &[f64], scale: f64) -> f64 {
    unsafe { raw_nrm2_f64::<Avx2>(chunk, scale) }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_nrm2_avx2_f32(x: &[f32], chunk_size: usize) -> f32 {
    if x.len() < 32_768 {
        return unsafe { nrm2_avx2_f32(x) };
    }

    let chunk_size = chunk_size.clamp(2048, 8192);

    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| unsafe {
            let s = helper_scale_avx2_f32(chunk);
            if s == 0.0 {
                (0.0f32, 0.0f32)
            } else {
                (s, helper_nrm2_avx2_f32(chunk, s))
            }
        })
        .reduce(
            || (0.0f32, 0.0f32),
            |(s1, sum1), (s2, sum2)| {
                if s1 == 0.0 { return (s2, sum2); }
                if s2 == 0.0 { return (s1, sum1); }

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

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_nrm2_avx2_f64(x: &[f64], chunk_size: usize) -> f64 {
    if x.len() < 32_768 {
        return unsafe { nrm2_avx2_f64(x) };
    }

    let chunk_size = chunk_size.clamp(2048, 8192);

    let (scale, sq_sum) = x
        .par_chunks(chunk_size)
        .map(|chunk| unsafe {
            let s = helper_scale_avx2_f64(chunk);
            if s == 0.0 {
                (0.0f64, 0.0f64)
            } else {
                (s, helper_nrm2_avx2_f64(chunk, s))
            }
        })
        .reduce(
            || (0.0f64, 0.0f64),
            |(s1, sum1), (s2, sum2)| {
                if s1 == 0.0 { return (s2, sum2); }
                if s2 == 0.0 { return (s1, sum1); }

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

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn nrm2_avx2_f32(x: &[f32]) -> f32 {
    unsafe { nrm2_f32::<Avx2>(x) }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn nrm2_avx2_f64(x: &[f64]) -> f64 {
    unsafe { nrm2_f64::<Avx2>(x) }
}

#[inline(always)]
unsafe fn raw_scale_nrm2_f32<S: SimdArch>(x: &[f32]) -> f32 {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        let mut acc0 = S::setzero_m256_f32();
        let mut acc1 = S::setzero_m256_f32();
        let mut acc2 = S::setzero_m256_f32();
        let mut acc3 = S::setzero_m256_f32();

        let mut i = 0;
        while i + 32 <= len {
            let x0 = S::loadu_m256_f32(ptr_x.add(i));
            let x1 = S::loadu_m256_f32(ptr_x.add(i + 8));
            let x2 = S::loadu_m256_f32(ptr_x.add(i + 16));
            let x3 = S::loadu_m256_f32(ptr_x.add(i + 24));

            let v0 = S::abs_f32(x0);
            let v1 = S::abs_f32(x1);
            let v2 = S::abs_f32(x2);
            let v3 = S::abs_f32(x3);

            acc0 = S::max_f32(acc0, v0);
            acc1 = S::max_f32(acc1, v1);
            acc2 = S::max_f32(acc2, v2);
            acc3 = S::max_f32(acc3, v3);

            i += 32;
        }

        let acc0 = S::max_f32(acc0, acc1);
        let acc1 = S::max_f32(acc2, acc3);
        let acc = S::max_f32(acc0, acc1);

        let mut scale = S::reduce_max_f32(acc);

        while i < len {
            scale = scale.max(x[i].abs());
            i += 1;
        }

        scale
    }
}

#[inline(always)]
unsafe fn raw_scale_nrm2_f64<S: SimdArch>(x: &[f64]) -> f64 {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        let mut acc0 = S::setzero_m256_f64();
        let mut acc1 = S::setzero_m256_f64();
        let mut acc2 = S::setzero_m256_f64();
        let mut acc3 = S::setzero_m256_f64();
        let mut acc4 = S::setzero_m256_f64();
        let mut acc5 = S::setzero_m256_f64();
        let mut acc6 = S::setzero_m256_f64();
        let mut acc7 = S::setzero_m256_f64();

        let mut i = 0;
        while i + 32 <= len {
            let x0 = S::loadu_m256_f64(ptr_x.add(i));
            let x1 = S::loadu_m256_f64(ptr_x.add(i + 4));
            let x2 = S::loadu_m256_f64(ptr_x.add(i + 8));
            let x3 = S::loadu_m256_f64(ptr_x.add(i + 12));
            let x4 = S::loadu_m256_f64(ptr_x.add(i + 16));
            let x5 = S::loadu_m256_f64(ptr_x.add(i + 20));
            let x6 = S::loadu_m256_f64(ptr_x.add(i + 24));
            let x7 = S::loadu_m256_f64(ptr_x.add(i + 28));

            let v0 = S::abs_f64(x0);
            let v1 = S::abs_f64(x1);
            let v2 = S::abs_f64(x2);
            let v3 = S::abs_f64(x3);
            let v4 = S::abs_f64(x4);
            let v5 = S::abs_f64(x5);
            let v6 = S::abs_f64(x6);
            let v7 = S::abs_f64(x7);

            acc0 = S::max_f64(acc0, v0);
            acc1 = S::max_f64(acc1, v1);
            acc2 = S::max_f64(acc2, v2);
            acc3 = S::max_f64(acc3, v3);
            acc4 = S::max_f64(acc4, v4);
            acc5 = S::max_f64(acc5, v5);
            acc6 = S::max_f64(acc6, v6);
            acc7 = S::max_f64(acc7, v7);

            i += 32;
        }

        let acc0 = S::max_f64(acc0, acc1);
        let acc1 = S::max_f64(acc2, acc3);
        let acc2 = S::max_f64(acc4, acc5);
        let acc3 = S::max_f64(acc6, acc7);

        let acc0 = S::max_f64(acc0, acc1);
        let acc1 = S::max_f64(acc2, acc3);
        let acc = S::max_f64(acc0, acc1);

        let mut scale = S::reduce_max_f64(acc);

        while i < len {
            scale = scale.max(x[i].abs());
            i += 1;
        }

        scale
    }
}

#[inline(always)]
unsafe fn raw_nrm2_f32<S: SimdArch>(x: &[f32], scale: f32) -> f32 {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();
        let mem_scale = S::set1_m256_f32(1.0 / scale);

        let mut acc0 = S::setzero_m256_f32();
        let mut acc1 = S::setzero_m256_f32();
        let mut acc2 = S::setzero_m256_f32();
        let mut acc3 = S::setzero_m256_f32();

        let mut i = 0;
        while i + 32 <= len {
            let x0 = S::loadu_m256_f32(ptr_x.add(i));
            let x1 = S::loadu_m256_f32(ptr_x.add(i + 8));
            let x2 = S::loadu_m256_f32(ptr_x.add(i + 16));
            let x3 = S::loadu_m256_f32(ptr_x.add(i + 24));

            let v0 = S::mul_f32(x0, mem_scale);
            let v1 = S::mul_f32(x1, mem_scale);
            let v2 = S::mul_f32(x2, mem_scale);
            let v3 = S::mul_f32(x3, mem_scale);

            acc0 = S::fmadd_f32(v0, v0, acc0);
            acc1 = S::fmadd_f32(v1, v1, acc1);
            acc2 = S::fmadd_f32(v2, v2, acc2);
            acc3 = S::fmadd_f32(v3, v3, acc3);

            i += 32;
        }

        let acc0 = S::add_f32(acc0, acc1);
        let acc1 = S::add_f32(acc2, acc3);
        let acc = S::add_f32(acc0, acc1);

        let s1 = S::extract_m128_f32(acc);
        let s2 = S::cast_m256f32_m128f32(acc);
        let sum128 = S::add_m128_f32(s1, s2);

        let iv64 = S::movehl_f32(sum128, sum128);
        let sum64 = S::add_m128_f32(sum128, iv64);
        let shuf64 = S::shuffle_f32(sum64, sum64);
        let sum = S::add_m128_f32(sum64, shuf64);

        let mut sum = S::cvtss_f32(sum);

        let inv_scale = 1.0 / scale;
        while i < len {
            let v = x[i] * inv_scale;
            sum += v * v;
            i += 1;
        }

        sum
    }
}

#[inline(always)]
unsafe fn raw_nrm2_f64<S: SimdArch>(x: &[f64], scale: f64) -> f64 {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();
        let mem_scale = S::set1_m256_f64(1.0 / scale);

        let mut acc0 = S::setzero_m256_f64();
        let mut acc1 = S::setzero_m256_f64();
        let mut acc2 = S::setzero_m256_f64();
        let mut acc3 = S::setzero_m256_f64();
        let mut acc4 = S::setzero_m256_f64();
        let mut acc5 = S::setzero_m256_f64();
        let mut acc6 = S::setzero_m256_f64();
        let mut acc7 = S::setzero_m256_f64();

        let mut i = 0;
        while i + 32 <= len {
            let x0 = S::loadu_m256_f64(ptr_x.add(i));
            let x1 = S::loadu_m256_f64(ptr_x.add(i + 4));
            let x2 = S::loadu_m256_f64(ptr_x.add(i + 8));
            let x3 = S::loadu_m256_f64(ptr_x.add(i + 12));
            let x4 = S::loadu_m256_f64(ptr_x.add(i + 16));
            let x5 = S::loadu_m256_f64(ptr_x.add(i + 20));
            let x6 = S::loadu_m256_f64(ptr_x.add(i + 24));
            let x7 = S::loadu_m256_f64(ptr_x.add(i + 28));

            let v0 = S::mul_f64(x0, mem_scale);
            let v1 = S::mul_f64(x1, mem_scale);
            let v2 = S::mul_f64(x2, mem_scale);
            let v3 = S::mul_f64(x3, mem_scale);
            let v4 = S::mul_f64(x4, mem_scale);
            let v5 = S::mul_f64(x5, mem_scale);
            let v6 = S::mul_f64(x6, mem_scale);
            let v7 = S::mul_f64(x7, mem_scale);

            acc0 = S::fmadd_f64(v0, v0, acc0);
            acc1 = S::fmadd_f64(v1, v1, acc1);
            acc2 = S::fmadd_f64(v2, v2, acc2);
            acc3 = S::fmadd_f64(v3, v3, acc3);
            acc4 = S::fmadd_f64(v4, v4, acc4);
            acc5 = S::fmadd_f64(v5, v5, acc5);
            acc6 = S::fmadd_f64(v6, v6, acc6);
            acc7 = S::fmadd_f64(v7, v7, acc7);

            i += 32;
        }

        let acc0 = S::add_f64(acc0, acc1);
        let acc1 = S::add_f64(acc2, acc3);
        let acc2 = S::add_f64(acc4, acc5);
        let acc3 = S::add_f64(acc6, acc7);

        let acc0 = S::add_f64(acc0, acc1);
        let acc1 = S::add_f64(acc2, acc3);
        let acc = S::add_f64(acc0, acc1);

        let hi = S::extract_m128_f64(acc);
        let lo = S::cast_m256f64_m128f64(acc);

        let sum128 = S::add_m128_f64(hi, lo);
        let hi128 = S::npackhi_f64(sum128, sum128);
        let sum = S::add_sd(sum128, hi128);
        let mut sum = S::cvtsd_f64(sum);

        let inv_scale = 1.0 / scale;
        while i < len {
            let v = x[i] * inv_scale;
            sum += v * v;
            i += 1;
        }

        sum
    }
}