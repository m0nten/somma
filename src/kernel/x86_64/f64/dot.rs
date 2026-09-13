use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use std::arch::x86_64::*;

/// # Safety
/// This functon have custom SIMD construction.
/// The length of vector X and the length of vector Y must not differ.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_dot_avx2(x: &[f64], y: &[f64], chunk_size: usize) -> f64 {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| unsafe { dot_avx2(x, y) }).sum()
}

/// # Safety
/// This functon have custom SIMD construction.
/// The length of vector X and the length of vector Y must not differ.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());

    let len = x.len();
    let mut i = 0;

    let ptr_x = x.as_ptr();
    let ptr_y = y.as_ptr();

    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut acc4 = _mm256_setzero_pd();
    let mut acc5 = _mm256_setzero_pd();
    let mut acc6 = _mm256_setzero_pd();
    let mut acc7 = _mm256_setzero_pd();

    while i + 32 <= len {
        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i)) };
        acc0 = _mm256_fmadd_pd(part_x, part_y, acc0);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 4)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 4)) };
        acc1 = _mm256_fmadd_pd(part_x, part_y, acc1);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 8)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 8)) };
        acc2 = _mm256_fmadd_pd(part_x, part_y, acc2);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 12)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 12)) };
        acc3 = _mm256_fmadd_pd(part_x, part_y, acc3);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 16)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 16)) };
        acc4 = _mm256_fmadd_pd(part_x, part_y, acc4);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 20)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 20)) };
        acc5 = _mm256_fmadd_pd(part_x, part_y, acc5);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 24)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 24)) };
        acc6 = _mm256_fmadd_pd(part_x, part_y, acc6);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 28)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 28)) };
        acc7 = _mm256_fmadd_pd(part_x, part_y, acc7);

        i += 32;
    }

    while i + 8 <= len {
        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i)) };
        acc0 = _mm256_fmadd_pd(part_x, part_y, acc0);

        let part_x = unsafe { _mm256_loadu_pd(ptr_x.add(i + 4)) };
        let part_y = unsafe { _mm256_loadu_pd(ptr_y.add(i + 4)) };
        acc1 = _mm256_fmadd_pd(part_x, part_y, acc1);

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
    let sum128 = _mm_add_pd(s1, s2);

    let var = _mm_unpackhi_pd(sum128, sum128);
    let var = _mm_add_sd(sum128, var);
    let mut sum = _mm_cvtsd_f64(var);

    while i < len {
        sum += unsafe { *ptr_x.add(i) } * unsafe { *ptr_y.add(i) };
        i += 1;
    }

    sum
}
