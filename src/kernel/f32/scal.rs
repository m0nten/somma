use crate::kernel::scal32;

pub fn scal(x: &mut [f32], a: f32) {
    scal32::scal(x, a);
}

pub fn par_scal(x: &mut [f32], a: f32, chunk_size: usize) {
    scal32::par_scal(x, a, chunk_size);
}
