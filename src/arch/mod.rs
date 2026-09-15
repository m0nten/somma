#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

/// # Safety
/// Abstraction over architecture-specific SIMD operations.
///
/// Implementors must ensure that every operation is only executed when the
/// CPU features required by the underlying intrinsic are available.
pub unsafe trait SimdArch {
    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn abs_f32(a: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn abs_f64(a: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn reduce_max_f32(a: M256MF32) -> f32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn reduce_max_f64(a: M256MF64) -> f64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn and_f32(a: M256MF32, b: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn and_f64(a: M256MF64, b: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn and_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn and_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn max_f32(a: M256MF32, b: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn max_f64(a: M256MF64, b: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn max_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn max_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn div_f32(a: M256MF32, b: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn div_f64(a: M256MF64, b: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn div_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn div_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn mul_f32(a: M256MF32, b: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn mul_f64(a: M256MF64, b: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn mul_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn mul_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn setzero_m256_f32() -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn setzero_m256_f64() -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn setzero_m128_f32() -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn setzero_m128_f64() -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn set1_m256_f32(a: f32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn set1_m256_f64(a: f64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn set1_m128_f32(a: f32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn set1_m128_f64(a: f64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for reading 8 consecutive
    /// `f32` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn loadu_m256_f32(mem: *const f32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for reading 4 consecutive
    /// `f64` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn loadu_m256_f64(mem: *const f64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for reading 4 consecutive
    /// `f32` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn loadu_m128_f32(mem: *const f32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for reading 2 consecutive
    /// `f64` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn loadu_m128_f64(mem: *const f64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for writing 4 consecutive
    /// `f32` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn storeu_m128_f32(mem: *mut f32, a: M128MF32);

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for writing 2 consecutive
    /// `f64` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn storeu_m128_f64(mem: *mut f64, a: M128MF64);

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for writing 8 consecutive
    /// `f32` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn storeu_m256_f32(mem: *mut f32, a: M256MF32);

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for writing 4 consecutive
    /// `f64` values.
    ///
    /// The caller must also ensure that the CPU supports the SIMD instruction
    /// set required by the implementation.
    unsafe fn storeu_m256_f64(mem: *mut f64, a: M256MF64);

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn fmadd_f32(a: M256MF32, b: M256MF32, c: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn fmadd_f64(a: M256MF64, b: M256MF64, c: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn fmadd_m128_f32(a: M128MF32, b: M128MF32, c: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn fmadd_m128_f64(a: M128MF64, b: M128MF64, c: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn add_sd(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn add_f32(a: M256MF32, b: M256MF32) -> M256MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn add_f64(a: M256MF64, b: M256MF64) -> M256MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn add_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn add_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn extract_m128_f32(a: M256MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn cast_m256f32_m128f32(a: M256MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn movehl_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn shuffle_f32(a: M128MF32, b: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn cvtss_f32(a: M128MF32) -> f32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn npackhi_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn cvtsd_f64(a: M128MF64) -> f64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn cast_m256f64_m128f64(a: M256MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn extract_m128_f64(a: M256MF64) -> M128MF64;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn movehdup_f32(a: M128MF32) -> M128MF32;

    /// # Safety
    ///
    /// The caller must ensure that the CPU supports the SIMD instruction set
    /// required by the implementation.
    unsafe fn add_ss_f32(a: M128MF32, b: M128MF32) -> M128MF32;
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct M256MF32 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m256,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct M256MF64 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m256d,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct M128MF32 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m128,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct M128MF64 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m128d,
}
