use crate::arch::SimdArch;
use crate::arch::avx2::Avx2;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_sum_f32<S: SimdArch>(x: &[f32], chunk_size: usize) -> f32 {
    x.par_chunks(chunk_size).map(|x| unsafe { sum_f32::<S>(x) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_sum_f64<S: SimdArch>(x: &[f64], chunk_size: usize) -> f64 {
    x.par_chunks(chunk_size).map(|x| unsafe { sum_f64::<S>(x) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_sum_avx2_f32(x: &[f32], chunk_size: usize) -> f32 {
    x.par_chunks(chunk_size).map(|x| unsafe { sum_avx2_f32(x) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn par_sum_avx2_f64(x: &[f64], chunk_size: usize) -> f64 {
    x.par_chunks(chunk_size).map(|x| unsafe { sum_avx2_f64(x) }).sum()
}

/// # Safety
/// The function uses SIMD.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sum_avx2_f32(x: &[f32]) -> f32 {
    unsafe { sum_f32::<Avx2>(x) }
}

/// # Safety
/// The function uses SIMD.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sum_avx2_f64(x: &[f64]) -> f64 {
    unsafe { sum_f64::<Avx2>(x) }
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn sum_f32<S: SimdArch>(x: &[f32]) -> f32 {
    unsafe {
        let len = x.len();
        let ptr_x = x.as_ptr();

        let mut acc0 = S::setzero_m256_f32();
        let mut acc1 = S::setzero_m256_f32();
        let mut acc2 = S::setzero_m256_f32();
        let mut acc3 = S::setzero_m256_f32();

        let mut i = 0;
        while i + 32 <= len {
            acc0 = S::add_f32(S::loadu_m256_f32(ptr_x.add(i)), acc0);
            acc1 = S::add_f32(S::loadu_m256_f32(ptr_x.add(i + 8)), acc1);
            acc2 = S::add_f32(S::loadu_m256_f32(ptr_x.add(i + 16)), acc2);
            acc3 = S::add_f32(S::loadu_m256_f32(ptr_x.add(i + 24)), acc3);

            i += 32;
        }

        while i + 8 <= len {
            acc0 = S::add_f32(S::loadu_m256_f32(ptr_x.add(i)), acc0);
            i += 8;
        }

        let acc0 = S::add_f32(acc0, acc1);
        let acc1 = S::add_f32(acc2, acc3);
        let acc = S::add_f32(acc0, acc1);

        let s1 = S::extract_m128_f32(acc);
        let s2 = S::cast_m256f32_m128f32(acc);
        let acc = S::add_m128_f32(s1, s2);

        let shuf = S::movehdup_f32(acc);
        let sums = S::add_m128_f32(acc, shuf);
        let shuf2 = S::movehl_f32(sums, sums);
        let vec = S::add_ss_f32(sums, shuf2);
        let mut sum = S::cvtss_f32(vec);

        while i < len {
            sum += *ptr_x.add(i);
            i += 1;
        }

        sum
    }
}

/// # Safety
/// The function uses SIMD.
#[inline(always)]
pub unsafe fn sum_f64<S: SimdArch>(x: &[f64]) -> f64 {
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
            acc0 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i)), acc0);
            acc1 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 4)), acc1);
            acc2 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 8)), acc2);
            acc3 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 12)), acc3);
            acc4 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 16)), acc4);
            acc5 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 20)), acc5);
            acc6 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 24)), acc6);
            acc7 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 28)), acc7);

            i += 32;
        }

        while i + 8 <= len {
            acc0 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i)), acc0);
            acc1 = S::add_f64(S::loadu_m256_f64(ptr_x.add(i + 4)), acc1);
            i += 8;
        }

        let acc0 = S::add_f64(acc0, acc1);
        let acc1 = S::add_f64(acc2, acc3);
        let acc2 = S::add_f64(acc4, acc5);
        let acc3 = S::add_f64(acc6, acc7);

        let acc0 = S::add_f64(acc0, acc1);
        let acc1 = S::add_f64(acc2, acc3);
        let acc = S::add_f64(acc0, acc1);

        let s1 = S::extract_m128_f64(acc);
        let s2 = S::cast_m256f64_m128f64(acc);
        let acc = S::add_m128_f64(s1, s2);

        let hi = S::npackhi_f64(acc, acc);
        let var = S::add_sd(acc, hi);
        let mut sum = S::cvtsd_f64(var);

        while i < len {
            sum += *ptr_x.add(i);
            i += 1;
        }

        sum
    }
}
