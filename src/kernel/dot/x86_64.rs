use crate::arch::{SimdArch, avx2::Avx2};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_dot_f32<S: SimdArch>(x: &[f32], y: &[f32], chunk_size: usize) -> f32 {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| unsafe { dot_f32::<S>(x, y) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_dot_avx2_f32(x: &[f32], y: &[f32], chunk_size: usize) -> f32 {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| unsafe { dot_avx2_f32(x, y) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_dot_avx2_f64(x: &[f64], y: &[f64], chunk_size: usize) -> f64 {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| unsafe { dot_avx2_f64(x, y) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_dot_f64<S: SimdArch>(x: &[f64], y: &[f64], chunk_size: usize) -> f64 {
    assert_eq!(x.len(), y.len());

    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| unsafe { dot_f64::<S>(x, y) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2_f64(x: &[f64], y: &[f64]) -> f64 {
    unsafe { dot_f64::<Avx2>(x, y) }
}

/// # Safety
/// The function uses SIMD.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2_f32(x: &[f32], y: &[f32]) -> f32 {
    unsafe { dot_f32::<Avx2>(x, y) }
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn dot_f32<S: SimdArch>(x: &[f32], y: &[f32]) -> f32 {
    unsafe {
        assert_eq!(x.len(), y.len());

        let len = x.len();
        let mut i = 0;

        let ptr_x = x.as_ptr();
        let ptr_y = y.as_ptr();

        let mut acc0 = S::setzero_m256_f32();
        let mut acc1 = S::setzero_m256_f32();
        let mut acc2 = S::setzero_m256_f32();
        let mut acc3 = S::setzero_m256_f32();

        while i + 32 <= len {
            let x0 = S::loadu_m256_f32(ptr_x.add(i));
            let y0 = S::loadu_m256_f32(ptr_y.add(i));

            let x1 = S::loadu_m256_f32(ptr_x.add(i + 8));
            let y1 = S::loadu_m256_f32(ptr_y.add(i + 8));

            let x2 = S::loadu_m256_f32(ptr_x.add(i + 16));
            let y2 = S::loadu_m256_f32(ptr_y.add(i + 16));

            let x3 = S::loadu_m256_f32(ptr_x.add(i + 24));
            let y3 = S::loadu_m256_f32(ptr_y.add(i + 24));

            acc0 = S::fmadd_f32(x0, y0, acc0);
            acc1 = S::fmadd_f32(x1, y1, acc1);
            acc2 = S::fmadd_f32(x2, y2, acc2);
            acc3 = S::fmadd_f32(x3, y3, acc3);

            i += 32;
        }

        while i + 8 <= len {
            let x0 = S::loadu_m256_f32(ptr_x.add(i));
            let y0 = S::loadu_m256_f32(ptr_y.add(i));
            acc0 = S::fmadd_f32(x0, y0, acc0);

            i += 8;
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

        while i < len {
            sum += *ptr_x.add(i) * *ptr_y.add(i);
            i += 1;
        }

        sum
    }
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn dot_f64<S: SimdArch>(x: &[f64], y: &[f64]) -> f64 {
    unsafe {
        assert_eq!(x.len(), y.len());

        let len = x.len();
        let mut i = 0;

        let ptr_x = x.as_ptr();
        let ptr_y = y.as_ptr();

        let mut acc0 = S::setzero_m256_f64();
        let mut acc1 = S::setzero_m256_f64();
        let mut acc2 = S::setzero_m256_f64();
        let mut acc3 = S::setzero_m256_f64();
        let mut acc4 = S::setzero_m256_f64();
        let mut acc5 = S::setzero_m256_f64();
        let mut acc6 = S::setzero_m256_f64();
        let mut acc7 = S::setzero_m256_f64();

        while i + 32 <= len {
            let x0 = S::loadu_m256_f64(ptr_x.add(i));
            let y0 = S::loadu_m256_f64(ptr_y.add(i));

            let x1 = S::loadu_m256_f64(ptr_x.add(i + 4));
            let y1 = S::loadu_m256_f64(ptr_y.add(i + 4));

            let x2 = S::loadu_m256_f64(ptr_x.add(i + 8));
            let y2 = S::loadu_m256_f64(ptr_y.add(i + 8));

            let x3 = S::loadu_m256_f64(ptr_x.add(i + 12));
            let y3 = S::loadu_m256_f64(ptr_y.add(i + 12));

            let x4 = S::loadu_m256_f64(ptr_x.add(i + 16));
            let y4 = S::loadu_m256_f64(ptr_y.add(i + 16));

            let x5 = S::loadu_m256_f64(ptr_x.add(i + 20));
            let y5 = S::loadu_m256_f64(ptr_y.add(i + 20));

            let x6 = S::loadu_m256_f64(ptr_x.add(i + 24));
            let y6 = S::loadu_m256_f64(ptr_y.add(i + 24));

            let x7 = S::loadu_m256_f64(ptr_x.add(i + 28));
            let y7 = S::loadu_m256_f64(ptr_y.add(i + 28));

            acc0 = S::fmadd_f64(x0, y0, acc0);
            acc1 = S::fmadd_f64(x1, y1, acc1);
            acc2 = S::fmadd_f64(x2, y2, acc2);
            acc3 = S::fmadd_f64(x3, y3, acc3);
            acc4 = S::fmadd_f64(x4, y4, acc4);
            acc5 = S::fmadd_f64(x5, y5, acc5);
            acc6 = S::fmadd_f64(x6, y6, acc6);
            acc7 = S::fmadd_f64(x7, y7, acc7);

            i += 32;
        }

        while i + 8 <= len {
            let x0 = S::loadu_m256_f64(ptr_x.add(i));
            let y0 = S::loadu_m256_f64(ptr_y.add(i));

            let x1 = S::loadu_m256_f64(ptr_x.add(i + 4));
            let y1 = S::loadu_m256_f64(ptr_y.add(i + 4));

            acc0 = S::fmadd_f64(x0, y0, acc0);
            acc1 = S::fmadd_f64(x1, y1, acc1);

            i += 8;
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

        while i < len {
            sum += *ptr_x.add(i) * *ptr_y.add(i);
            i += 1;
        }

        sum
    }
}
