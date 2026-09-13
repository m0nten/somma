use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

/// # Safety
/// This functon have custom SIMD construction.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_axpy_avx2(y: &mut [f32], x: &[f32], a: f32, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size).zip(x.par_chunks(chunk_size)).for_each(|(y, x)| unsafe { axpy_avx2(y, x, a) });
}

/// # Safety
/// This functon have custom SIMD construction.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn axpy_avx2(y: &mut [f32], x: &[f32], a: f32) {
    assert_eq!(x.len(), y.len());

    let len = y.len();
    let var_a = _mm256_set1_ps(a);
    let ptr_y = y.as_mut_ptr();
    let ptr_x = x.as_ptr();

    let mut i = 0;
    while i + 32 <= len {
        let y0 = unsafe { _mm256_loadu_ps(ptr_y.add(i)) };
        let x0 = unsafe { _mm256_loadu_ps(ptr_x.add(i)) };

        let y1 = unsafe { _mm256_loadu_ps(ptr_y.add(i + 8)) };
        let x1 = unsafe { _mm256_loadu_ps(ptr_x.add(i + 8)) };

        let y2 = unsafe { _mm256_loadu_ps(ptr_y.add(i + 16)) };
        let x2 = unsafe { _mm256_loadu_ps(ptr_x.add(i + 16)) };

        let y3 = unsafe { _mm256_loadu_ps(ptr_y.add(i + 24)) };
        let x3 = unsafe { _mm256_loadu_ps(ptr_x.add(i + 24)) };

        let v0 = _mm256_fmadd_ps(x0, var_a, y0);
        let v1 = _mm256_fmadd_ps(x1, var_a, y1);
        let v2 = _mm256_fmadd_ps(x2, var_a, y2);
        let v3 = _mm256_fmadd_ps(x3, var_a, y3);

        unsafe {
            _mm256_storeu_ps(ptr_y.add(i), v0);
            _mm256_storeu_ps(ptr_y.add(i + 8), v1);
            _mm256_storeu_ps(ptr_y.add(i + 16), v2);
            _mm256_storeu_ps(ptr_y.add(i + 24), v3);
        };

        i += 32;
    }

    while i + 8 <= len {
        let part_y = unsafe { _mm256_loadu_ps(ptr_y.add(i)) };
        let part_x = unsafe { _mm256_loadu_ps(ptr_x.add(i)) };
        let var = _mm256_fmadd_ps(part_x, var_a, part_y);
        unsafe {
            _mm256_storeu_ps(ptr_y.add(i), var);
        };

        i += 8;
    }

    while i < len {
        y[i] += x[i] * a;
        i += 1;
    }
}
