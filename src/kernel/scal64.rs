use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use std::arch::x86_64::*;

#[inline(always)]
pub fn par_scal(x: &mut [f64], a: f64, chunk_size: usize) {
    x.par_chunks_mut(chunk_size).for_each(|x| scal(x, a));
}

/// # Safety
/// This functon have custom SIMD construction.
/// Not recommended.
#[target_feature(enable = "avx2")]
pub unsafe fn _par_scal_avx2(x: &mut [f64], a: f64, chunk_size: usize) {
    x.par_chunks_mut(chunk_size).for_each(|x| unsafe { _scal_avx2(x, a) });
}

/// # Safety
/// This functon have custom SIMD construction.
/// Not recommended.
#[target_feature(enable = "avx2")]
pub unsafe fn _scal_avx2(x: &mut [f64], a: f64) {
    let len = x.len();
    let ptr_x = x.as_mut_ptr();
    let var_a = _mm256_set1_pd(a);

    let mut i = 0;
    while i + 32 <= len {
        let p0 = unsafe { ptr_x.add(i) };
        let p1 = unsafe { ptr_x.add(i + 4) };
        let p2 = unsafe { ptr_x.add(i + 8) };
        let p3 = unsafe { ptr_x.add(i + 12) };
        let p4 = unsafe { ptr_x.add(i + 16) };
        let p5 = unsafe { ptr_x.add(i + 20) };
        let p6 = unsafe { ptr_x.add(i + 24) };
        let p7 = unsafe { ptr_x.add(i + 28) };

        let v0 = unsafe { _mm256_loadu_pd(p0) };
        let v1 = unsafe { _mm256_loadu_pd(p1) };
        let v2 = unsafe { _mm256_loadu_pd(p2) };
        let v3 = unsafe { _mm256_loadu_pd(p3) };
        let v4 = unsafe { _mm256_loadu_pd(p4) };
        let v5 = unsafe { _mm256_loadu_pd(p5) };
        let v6 = unsafe { _mm256_loadu_pd(p6) };
        let v7 = unsafe { _mm256_loadu_pd(p7) };

        unsafe {
            _mm256_storeu_pd(p0, _mm256_mul_pd(v0, var_a));
            _mm256_storeu_pd(p1, _mm256_mul_pd(v1, var_a));
            _mm256_storeu_pd(p2, _mm256_mul_pd(v2, var_a));
            _mm256_storeu_pd(p3, _mm256_mul_pd(v3, var_a));
            _mm256_storeu_pd(p4, _mm256_mul_pd(v4, var_a));
            _mm256_storeu_pd(p5, _mm256_mul_pd(v5, var_a));
            _mm256_storeu_pd(p6, _mm256_mul_pd(v6, var_a));
            _mm256_storeu_pd(p7, _mm256_mul_pd(v7, var_a));
        }

        i += 32;
    }

    while i + 8 <= len {
        unsafe {
            let p0 = ptr_x.add(i);
            let v0 = _mm256_loadu_pd(p0);
            _mm256_storeu_pd(p0, _mm256_mul_pd(v0, var_a));
        };

        i += 8;
    }

    while i < len {
        x[i] *= a;
        i += 1;
    }
}

#[inline(always)]
pub fn scal(x: &mut [f64], a: f64) {
    for x in x.iter_mut() {
        *x *= a;
    }
}
