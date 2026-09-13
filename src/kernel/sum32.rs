use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use std::arch::x86_64::*;

/// # Safety
/// This functon have custom SIMD construction.
#[target_feature(enable = "avx2")]
pub unsafe fn par_sum_avx2(x: &[f32], chunk_size: usize) -> f32 {
    x.par_chunks(chunk_size).map(|x| unsafe { sum_avx2(x) }).sum()
}

#[inline(always)]
pub fn par_sum(x: &[f32], chunk_size: usize) -> f32 {
    x.par_chunks(chunk_size).map(sum).sum()
}

/// # Safety
/// This functon have custom SIMD construction.
#[target_feature(enable = "avx2")]
pub unsafe fn sum_avx2(x: &[f32]) -> f32 {
    let len = x.len();
    let mut i = 0;

    let ptr_x = x.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= len {
        acc0 = _mm256_add_ps(unsafe { _mm256_loadu_ps(ptr_x.add(i)) }, acc0);
        acc1 = _mm256_add_ps(unsafe { _mm256_loadu_ps(ptr_x.add(i + 8)) }, acc1);
        acc2 = _mm256_add_ps(unsafe { _mm256_loadu_ps(ptr_x.add(i + 16)) }, acc2);
        acc3 = _mm256_add_ps(unsafe { _mm256_loadu_ps(ptr_x.add(i + 24)) }, acc3);

        i += 32;
    }

    while i + 8 <= len {
        acc0 = _mm256_add_ps(unsafe { _mm256_loadu_ps(ptr_x.add(i)) }, acc0);
        i += 8;
    }

    let acc0 = _mm256_add_ps(acc0, acc1);
    let acc1 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc0, acc1);

    let s1 = _mm256_extractf128_ps(acc, 1);
    let s2 = _mm256_castps256_ps128(acc);
    let acc = _mm_add_ps(s1, s2);

    let shuf = _mm_movehdup_ps(acc);
    let sums = _mm_add_ps(acc, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_vec = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(final_vec);

    while i < len {
        unsafe { sum += *ptr_x.add(i) };
        i += 1;
    }

    sum
}

#[inline(always)]
pub fn sum(x: &[f32]) -> f32 {
    let mut x = x;
    let (mut acc0, mut acc1, mut acc2, mut acc3, mut acc4, mut acc5, mut acc6, mut acc7) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    while x.len() >= 8 {
        acc0 += x[0];
        acc1 += x[1];
        acc2 += x[2];
        acc3 += x[3];
        acc4 += x[4];
        acc5 += x[5];
        acc6 += x[6];
        acc7 += x[7];

        x = &x[8..];
    }

    let mut sum = (acc0 + acc4) + (acc1 + acc5) + (acc2 + acc6) + (acc3 + acc7);

    for x in x.iter() {
        sum += x;
    }

    sum
}
