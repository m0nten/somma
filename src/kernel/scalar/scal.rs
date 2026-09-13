use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use std::ops::{Mul, MulAssign};

pub fn par_scal<T: Copy + Sync + Send + Mul + MulAssign>(x: &mut [T], a: T, chunk_size: usize) {
    x.par_chunks_mut(chunk_size).for_each(|x| scal(x, a));
}

#[inline(always)]
pub fn scal<T: Copy + Mul + MulAssign>(x: &mut [T], a: T) {
    for x in x.iter_mut() {
        *x *= a;
    }
}
