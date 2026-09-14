use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul},
};

#[inline(always)]
pub fn par_sum<T: Copy + Sum + Send + Sync + Default + Add<Output = T> + AddAssign + Mul<Output = T>>(x: &[T], chunk_size: usize) -> T {
    x.par_chunks(chunk_size).map(|x| sum(x)).sum()
}

#[inline(always)]
pub fn sum<T: Copy + Default + Add<Output = T> + AddAssign + Mul<Output = T>>(x: &[T]) -> T {
    let mut x = x;

    let zero = T::default();
    let (mut acc0, mut acc1, mut acc2, mut acc3, mut acc4, mut acc5, mut acc6, mut acc7) = (zero, zero, zero, zero, zero, zero, zero, zero);

    while x.len() >= 8 {
        acc0 += x[0];
        acc1 += x[1];
        acc2 += x[2];
        acc3 += x[3];
        acc4 += x[4];
        acc5 += x[5];
        acc6 += x[6];
        acc7 += x[7];

        x = &x[8..];
    }

    let mut sum = (acc0 + acc4) + (acc1 + acc5) + (acc2 + acc6) + (acc3 + acc7);

    for x in x.iter() {
        sum += *x;
    }

    sum
}
