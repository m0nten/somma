use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::arch::x86_64::*;

/// # Safety
/// This functon have custom SIMD construction.
/// The length of vector X and the length of vector Y must not differ.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_dot_avx2(x: &[f32], y: &[f32], chunk_size: usize) -> f32 {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| unsafe { dot_avx2(x, y) }).sum()
}

/// # Safety
/// This functon have custom SIMD construction.
/// The length of vector X and the length of vector Y must not differ.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());

    let len = x.len();
    let mut i = 0;

    let ptr_x = x.as_ptr();
    let ptr_y = y.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= len {
        let slice_x = unsafe { _mm256_loadu_ps(ptr_x.add(i)) };
        let slice_y = unsafe { _mm256_loadu_ps(ptr_y.add(i)) };
        acc0 = _mm256_fmadd_ps(slice_x, slice_y, acc0);

        let slice_x = unsafe { _mm256_loadu_ps(ptr_x.add(i + 8)) };
        let slice_y = unsafe { _mm256_loadu_ps(ptr_y.add(i + 8)) };
        acc1 = _mm256_fmadd_ps(slice_x, slice_y, acc1);

        let slice_x = unsafe { _mm256_loadu_ps(ptr_x.add(i + 16)) };
        let slice_y = unsafe { _mm256_loadu_ps(ptr_y.add(i + 16)) };
        acc2 = _mm256_fmadd_ps(slice_x, slice_y, acc2);

        let slice_x = unsafe { _mm256_loadu_ps(ptr_x.add(i + 24)) };
        let slice_y = unsafe { _mm256_loadu_ps(ptr_y.add(i + 24)) };
        acc3 = _mm256_fmadd_ps(slice_x, slice_y, acc3);

        i += 32;
    }

    while i + 8 <= len {
        let x0 = unsafe { _mm256_loadu_ps(ptr_x.add(i)) };
        let y0 = unsafe { _mm256_loadu_ps(ptr_y.add(i)) };
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);

        i += 8;
    }

    let acc0 = _mm256_add_ps(acc0, acc1);
    let acc1 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc0, acc1);

    let s1 = _mm256_extractf128_ps(acc, 1);
    let s2 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(s1, s2);

    let iv64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, iv64);
    let shuf64 = _mm_shuffle_ps(sum64, sum64, 0x1);
    let sum = _mm_add_ps(sum64, shuf64);

    let mut sum = _mm_cvtss_f32(sum);

    while i < len {
        sum += unsafe { *ptr_x.add(i) } * unsafe { *ptr_y.add(i) };
        i += 1;
    }

    sum
}
