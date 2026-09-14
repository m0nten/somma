#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::{M128MF32, M128MF64, M256MF32, M256MF64, SimdArch};

pub struct Avx2Fma;

unsafe impl SimdArch for Avx2Fma {
    #[target_feature(enable = "avx2")]
    unsafe fn setzero_m256_f32() -> M256MF32 {
        M256MF32 { mem: _mm256_setzero_ps() }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn setzero_m256_f64() -> M256MF64 {
        M256MF64 { mem: _mm256_setzero_pd() }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn setzero_m128_f32() -> M128MF32 {
        M128MF32 { mem: _mm_setzero_ps() }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn setzero_m128_f64() -> M128MF64 {
        M128MF64 { mem: _mm_setzero_pd() }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn loadu_m256_f32(mem: *const f32) -> M256MF32 {
        M256MF32 {
            mem: unsafe { _mm256_loadu_ps(mem) },
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn loadu_m256_f64(mem: *const f64) -> M256MF64 {
        M256MF64 {
            mem: unsafe { _mm256_loadu_pd(mem) },
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn loadu_m128_f32(mem: *const f32) -> M128MF32 {
        M128MF32 {
            mem: unsafe { _mm_loadu_ps(mem) },
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn loadu_m128_f64(mem: *const f64) -> M128MF64 {
        M128MF64 {
            mem: unsafe { _mm_loadu_pd(mem) },
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn storeu_m128_f32(mem: *mut f32, a: M128MF32) {
        unsafe { _mm_storeu_ps(mem, a.mem) };
    }

    #[target_feature(enable = "avx2")]
    unsafe fn storeu_m128_f64(mem: *mut f64, a: M128MF64) {
        unsafe { _mm_storeu_pd(mem, a.mem) };
    }

    #[target_feature(enable = "avx2")]
    unsafe fn storeu_m256_f32(mem: *mut f32, a: M256MF32) {
        unsafe { _mm256_storeu_ps(mem, a.mem) };
    }

    #[target_feature(enable = "avx2")]
    unsafe fn storeu_m256_f64(mem: *mut f64, a: M256MF64) {
        unsafe { _mm256_storeu_pd(mem, a.mem) };
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fmadd_f32(a: M256MF32, b: M256MF32, c: M256MF32) -> M256MF32 {
        M256MF32 {
            mem: _mm256_fmadd_ps(a.mem, b.mem, c.mem),
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fmadd_f64(a: M256MF64, b: M256MF64, c: M256MF64) -> M256MF64 {
        M256MF64 {
            mem: _mm256_fmadd_pd(a.mem, b.mem, c.mem),
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fmadd_m128_f32(a: M128MF32, b: M128MF32, c: M128MF32) -> M128MF32 {
        M128MF32 {
            mem: _mm_fmadd_ps(a.mem, b.mem, c.mem),
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fmadd_m128_f64(a: M128MF64, b: M128MF64, c: M128MF64) -> M128MF64 {
        M128MF64 {
            mem: _mm_fmadd_pd(a.mem, b.mem, c.mem),
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_sd(a: M128MF64, b: M128MF64) -> M128MF64 {
        M128MF64 { mem: _mm_add_sd(a.mem, b.mem) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_f32(a: M256MF32, b: M256MF32) -> M256MF32 {
        M256MF32 { mem: _mm256_add_ps(a.mem, b.mem) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_f64(a: M256MF64, b: M256MF64) -> M256MF64 {
        M256MF64 { mem: _mm256_add_pd(a.mem, b.mem) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_m128_f32(a: M128MF32, b: M128MF32) -> M128MF32 {
        M128MF32 { mem: _mm_add_ps(a.mem, b.mem) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_m128_f64(a: M128MF64, b: M128MF64) -> M128MF64 {
        M128MF64 { mem: _mm_add_pd(a.mem, b.mem) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn extract_m128_f32(a: M256MF32) -> M128MF32 {
        M128MF32 {
            mem: _mm256_extractf128_ps(a.mem, 1),
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cast_m256f32_m128f32(a: M256MF32) -> M128MF32 {
        M128MF32 {
            mem: _mm256_castps256_ps128(a.mem),
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn movehl_f32(a: M128MF32, b: M128MF32) -> M128MF32 {
        M128MF32 { mem: _mm_movehl_ps(a.mem, b.mem) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn shuffle_f32(a: M128MF32, b: M128MF32) -> M128MF32 {
        M128MF32 {
            mem: _mm_shuffle_ps::<0x01>(a.mem, b.mem),
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cvtss_f32(a: M128MF32) -> f32 {
        _mm_cvtss_f32(a.mem)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn npackhi_f64(a: M128MF64, b: M128MF64) -> M128MF64 {
        M128MF64 {
            mem: _mm_unpackhi_pd(a.mem, b.mem),
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cvtsd_f64(a: M128MF64) -> f64 {
        _mm_cvtsd_f64(a.mem)
    }
}
