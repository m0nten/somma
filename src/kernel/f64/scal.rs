use crate::kernel::scal64;

pub fn scal(x: &mut [f64], a: f64) {
    scal64::scal(x, a);
}

pub fn par_scal(x: &mut [f64], a: f64, chunk_size: usize) {
    scal64::par_scal(x, a, chunk_size);
}
