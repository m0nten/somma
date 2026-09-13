use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use std::arch::x86_64::*;

/// # Safety
/// Only on devices that support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn par_sum_avx2(x: &[f64], chunk_size: usize) -> f64 {
    x.par_chunks(chunk_size).map(|x| unsafe { sum_avx2(x) }).sum()
}

/// # Safety
/// Only on devices that support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn sum_avx2(x: &[f64]) -> f64 {
    let len = x.len();
    let mut i = 0;

    let ptr_x = x.as_ptr();

    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut acc4 = _mm256_setzero_pd();
    let mut acc5 = _mm256_setzero_pd();
    let mut acc6 = _mm256_setzero_pd();
    let mut acc7 = _mm256_setzero_pd();

    while i + 32 <= len {
        acc0 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i)) }, acc0);
        acc1 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 4)) }, acc1);
        acc2 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 8)) }, acc2);
        acc3 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 12)) }, acc3);
        acc4 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 16)) }, acc4);
        acc5 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 20)) }, acc5);
        acc6 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 24)) }, acc6);
        acc7 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 28)) }, acc7);

        i += 32;
    }

    while i + 8 <= len {
        acc0 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i)) }, acc0);
        acc1 = _mm256_add_pd(unsafe { _mm256_loadu_pd(ptr_x.add(i + 4)) }, acc1);
        i += 8;
    }

    let acc0 = _mm256_add_pd(acc0, acc1);
    let acc1 = _mm256_add_pd(acc2, acc3);
    let acc2 = _mm256_add_pd(acc4, acc5);
    let acc3 = _mm256_add_pd(acc6, acc7);

    let acc0 = _mm256_add_pd(acc0, acc1);
    let acc1 = _mm256_add_pd(acc2, acc3);
    let acc = _mm256_add_pd(acc0, acc1);

    let s1 = _mm256_extractf128_pd(acc, 1);
    let s2 = _mm256_castpd256_pd128(acc);
    let acc = _mm_add_pd(s1, s2);

    let hi = _mm_unpackhi_pd(acc, acc);
    let var = _mm_add_sd(acc, hi);
    let mut sum = _mm_cvtsd_f64(var);

    while i < len {
        unsafe { sum += *ptr_x.add(i) };
        i += 1;
    }

    sum
}
