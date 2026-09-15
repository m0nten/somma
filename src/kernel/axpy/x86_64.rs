use crate::arch::{SimdArch, avx2::Avx2};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_axpy_f32<S: SimdArch>(y: &mut [f32], x: &[f32], a: f32, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .for_each(|(y, x)| unsafe { axpy_f32::<S>(y, x, a) });
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_axpy_f64<S: SimdArch>(y: &mut [f64], x: &[f64], a: f64, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .for_each(|(y, x)| unsafe { axpy_f64::<S>(y, x, a) });
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_axpy_avx2_f32(y: &mut [f32], x: &[f32], a: f32, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .for_each(|(y, x)| unsafe { axpy_avx2_f32(y, x, a) });
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_axpy_avx2_f64(y: &mut [f64], x: &[f64], a: f64, chunk_size: usize) {
    assert_eq!(x.len(), y.len());
    y.par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .for_each(|(y, x)| unsafe { axpy_avx2_f64(y, x, a) });
}

/// # Safety
/// The function uses SIMD.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn axpy_avx2_f32(y: &mut [f32], x: &[f32], a: f32) {
    unsafe { axpy_f32::<Avx2>(y, x, a) };
}

/// # Safety
/// The function uses SIMD.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn axpy_avx2_f64(y: &mut [f64], x: &[f64], a: f64) {
    unsafe { axpy_f64::<Avx2>(y, x, a) };
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn axpy_f32<S: SimdArch>(y: &mut [f32], x: &[f32], a: f32) {
    unsafe {
        assert_eq!(x.len(), y.len());

        let len = y.len();
        let var_a = S::set1_m256_f32(a);
        let ptr_y = y.as_mut_ptr();
        let ptr_x = x.as_ptr();

        let mut i = 0;
        while i + 32 <= len {
            let y0 = S::loadu_m256_f32(ptr_y.add(i));
            let x0 = S::loadu_m256_f32(ptr_x.add(i));

            let y1 = S::loadu_m256_f32(ptr_y.add(i + 8));
            let x1 = S::loadu_m256_f32(ptr_x.add(i + 8));

            let y2 = S::loadu_m256_f32(ptr_y.add(i + 16));
            let x2 = S::loadu_m256_f32(ptr_x.add(i + 16));

            let y3 = S::loadu_m256_f32(ptr_y.add(i + 24));
            let x3 = S::loadu_m256_f32(ptr_x.add(i + 24));

            let v0 = S::fmadd_f32(x0, var_a, y0);
            let v1 = S::fmadd_f32(x1, var_a, y1);
            let v2 = S::fmadd_f32(x2, var_a, y2);
            let v3 = S::fmadd_f32(x3, var_a, y3);

            S::storeu_m256_f32(ptr_y.add(i), v0);
            S::storeu_m256_f32(ptr_y.add(i + 8), v1);
            S::storeu_m256_f32(ptr_y.add(i + 16), v2);
            S::storeu_m256_f32(ptr_y.add(i + 24), v3);

            i += 32;
        }

        while i + 8 <= len {
            let y0 = S::loadu_m256_f32(ptr_y.add(i));
            let x0 = S::loadu_m256_f32(ptr_x.add(i));
            let v0 = S::fmadd_f32(x0, var_a, y0);
            S::storeu_m256_f32(ptr_y.add(i), v0);

            i += 8;
        }

        while i < len {
            y[i] += x[i] * a;
            i += 1;
        }
    }
}

/// # Safety
/// Only on devices that support AVX2 and FMA.
/// # Safety
/// Требует поддержки AVX2 и FMA на целевом процессоре.
#[inline(always)]
pub unsafe fn axpy_f64<S: SimdArch>(y: &mut [f64], x: &[f64], a: f64) {
    unsafe {
        assert_eq!(x.len(), y.len());

        let len = y.len();
        let var_a = S::set1_m256_f64(a);
        let ptr_y = y.as_mut_ptr();
        let ptr_x = x.as_ptr();

        let mut i = 0;
        // Развертка по 16 элементов (4 вектора по 4 f64) — помещается в 16 YMM-регистров
        while i + 16 <= len {
            let y0 = S::loadu_m256_f64(ptr_y.add(i));
            let x0 = S::loadu_m256_f64(ptr_x.add(i));
            let y1 = S::loadu_m256_f64(ptr_y.add(i + 4));
            let x1 = S::loadu_m256_f64(ptr_x.add(i + 4));
            let y2 = S::loadu_m256_f64(ptr_y.add(i + 8));
            let x2 = S::loadu_m256_f64(ptr_x.add(i + 8));
            let y3 = S::loadu_m256_f64(ptr_y.add(i + 12));
            let x3 = S::loadu_m256_f64(ptr_x.add(i + 12));

            let v0 = S::fmadd_f64(x0, var_a, y0);
            let v1 = S::fmadd_f64(x1, var_a, y1);
            let v2 = S::fmadd_f64(x2, var_a, y2);
            let v3 = S::fmadd_f64(x3, var_a, y3);

            S::storeu_m256_f64(ptr_y.add(i), v0);
            S::storeu_m256_f64(ptr_y.add(i + 4), v1);
            S::storeu_m256_f64(ptr_y.add(i + 8), v2);
            S::storeu_m256_f64(ptr_y.add(i + 12), v3);

            i += 16;
        }

        // Обработка 1 вектора (4 элемента)
        while i + 4 <= len {
            let y0 = S::loadu_m256_f64(ptr_y.add(i));
            let x0 = S::loadu_m256_f64(ptr_x.add(i));
            let v0 = S::fmadd_f64(x0, var_a, y0);
            S::storeu_m256_f64(ptr_y.add(i), v0);

            i += 4;
        }

        // Скалярный хвост (< 4 элементов)
        while i < len {
            y[i] += x[i] * a;
            i += 1;
        }
    }
}
