use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::ops::{AddAssign, Mul};

#[inline(always)]
pub fn par_axpy<T: Copy + Sync + Send + AddAssign + Mul<Output = T>>(y: &mut [T], x: &[T], a: T, chunk_size: usize) {
    y.par_chunks_mut(chunk_size).zip(x.par_chunks(chunk_size)).for_each(|(mut y, x)| axpy(&mut y, x, a));
}

#[inline(always)]
pub fn axpy<T: Copy + AddAssign + Mul<Output = T>>(y: &mut [T], x: &[T], a: T) {
    assert_eq!(y.len(), x.len());
    for (y, x) in y.iter_mut().zip(x) {
        *y += *x * a;
    }
}
