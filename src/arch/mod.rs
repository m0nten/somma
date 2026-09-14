#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub mod avx2fma;
#[cfg(target_arch = "x86_64")]
pub use avx2fma::Avx2Fma;

pub unsafe trait SimdArch {
    unsafe fn setzero_m256_f32() -> M256MF32;
    unsafe fn setzero_m256_f64() -> M256MF64;
    unsafe fn setzero_m128_f32() -> M128MF32;
    unsafe fn setzero_m128_f64() -> M128MF64;

    unsafe fn loadu_m256_f32(mem: *const f32) -> M256MF32;
    unsafe fn loadu_m256_f64(mem: *const f64) -> M256MF64;
    unsafe fn loadu_m128_f32(mem: *const f32) -> M128MF32;
    unsafe fn loadu_m128_f64(mem: *const f64) -> M128MF64;

    unsafe fn storeu_m128_f32(mem: *mut f32, a: M128MF32);
    unsafe fn storeu_m128_f64(mem: *mut f64, a: M128MF64);
    unsafe fn storeu_m256_f32(mem: *mut f32, a: M256MF32);
    unsafe fn storeu_m256_f64(mem: *mut f64, a: M256MF64);

    unsafe fn fmadd_f32(a: M256MF32, b: M256MF32, c: M256MF32) -> M256MF32;
    unsafe fn fmadd_f64(a: M256MF64, b: M256MF64, c: M256MF64) -> M256MF64;
    unsafe fn fmadd_m128_f32(a: M128MF32, b: M128MF32, c: M128MF32) -> M128MF32;
    unsafe fn fmadd_m128_f64(a: M128MF64, b: M128MF64, c: M128MF64) -> M128MF64;

    unsafe fn add_sd(a: M128MF64, b: M128MF64) -> M128MF64;
    unsafe fn add_f32(a: M256MF32, b: M256MF32) -> M256MF32;
    unsafe fn add_f64(a: M256MF64, b: M256MF64) -> M256MF64;
    unsafe fn add_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32;
    unsafe fn add_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64;

    unsafe fn extract_m128_f32(a: M256MF32) -> M128MF32;
    unsafe fn cast_m256f32_m128f32(a: M256MF32) -> M128MF32;
    unsafe fn movehl_f32(a: M128MF32, b: M128MF32) -> M128MF32;
    unsafe fn shuffle_f32(a: M128MF32, b: M128MF32) -> M128MF32;
    unsafe fn cvtss_f32(a: M128MF32) -> f32;
    unsafe fn npackhi_f64(a: M128MF64, b: M128MF64) -> M128MF64;
    unsafe fn cvtsd_f64(a: M128MF64) -> f64;
}

#[repr(transparent)]
pub struct M256MF32 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m256,
}

#[repr(transparent)]
pub struct M256MF64 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m256d,
}

#[repr(transparent)]
pub struct M128MF32 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m128,
}

#[repr(transparent)]
pub struct M128MF64 {
    #[cfg(target_arch = "x86_64")]
    pub mem: __m128d,
}
