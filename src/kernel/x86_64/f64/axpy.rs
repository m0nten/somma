use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

/// # Safety
/// This functon have custom SIMD construction.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn par_axpy_avx2(y: &mut [f64], x: &[f64], a: f64, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size).zip(x.par_chunks(chunk_size)).for_each(|(y, x)| unsafe { axpy_avx2(y, x, a) });
}

/// # Safety
/// This functon have custom SIMD construction.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn axpy_avx2(y: &mut [f64], x: &[f64], a: f64) {
    assert_eq!(x.len(), y.len());

    let len = y.len();
    let var_a = _mm256_set1_pd(a);
    let ptr_y = y.as_mut_ptr();
    let ptr_x = x.as_ptr();

    let mut i = 0;
    while i + 32 <= len {
        let y0 = unsafe { _mm256_loadu_pd(ptr_y.add(i)) };
        let x0 = unsafe { _mm256_loadu_pd(ptr_x.add(i)) };

        let y1 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 4)) };
        let x1 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 4)) };

        let y2 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 8)) };
        let x2 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 8)) };

        let y3 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 12)) };
        let x3 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 12)) };

        let y4 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 16)) };
        let x4 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 16)) };

        let y5 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 20)) };
        let x5 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 20)) };

        let y6 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 24)) };
        let x6 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 24)) };

        let y7 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 28)) };
        let x7 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 28)) };

        let v0 = _mm256_fmadd_pd(x0, var_a, y0);
        let v1 = _mm256_fmadd_pd(x1, var_a, y1);
        let v2 = _mm256_fmadd_pd(x2, var_a, y2);
        let v3 = _mm256_fmadd_pd(x3, var_a, y3);
        let v4 = _mm256_fmadd_pd(x4, var_a, y4);
        let v5 = _mm256_fmadd_pd(x5, var_a, y5);
        let v6 = _mm256_fmadd_pd(x6, var_a, y6);
        let v7 = _mm256_fmadd_pd(x7, var_a, y7);

        unsafe {
            _mm256_storeu_pd(ptr_y.add(i), v0);
            _mm256_storeu_pd(ptr_y.add(i + 4), v1);
            _mm256_storeu_pd(ptr_y.add(i + 8), v2);
            _mm256_storeu_pd(ptr_y.add(i + 12), v3);
            _mm256_storeu_pd(ptr_y.add(i + 16), v4);
            _mm256_storeu_pd(ptr_y.add(i + 20), v5);
            _mm256_storeu_pd(ptr_y.add(i + 24), v6);
            _mm256_storeu_pd(ptr_y.add(i + 28), v7);
        };

        i += 32;
    }

    while i + 8 <= len {
        let y0 = unsafe { _mm256_loadu_pd(ptr_y.add(i)) };
        let x0 = unsafe { _mm256_loadu_pd(ptr_x.add(i)) };

        let y1 = unsafe { _mm256_loadu_pd(ptr_y.add(i + 4)) };
        let x1 = unsafe { _mm256_loadu_pd(ptr_x.add(i + 4)) };

        let v0 = _mm256_fmadd_pd(x0, var_a, y0);
        let v1 = _mm256_fmadd_pd(x1, var_a, y1);

        unsafe {
            _mm256_storeu_pd(ptr_y.add(i), v0);
            _mm256_storeu_pd(ptr_y.add(i + 4), v1);
        }

        i += 8;
    }

    while i < len {
        y[i] += x[i] * a;
        i += 1;
    }
}
