use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul},
};

#[inline(always)]
pub fn par_dot<T: Copy + Sum + Send + Sync + Default + Add<Output = T> + AddAssign + Mul<Output = T>>(x: &[T], y: &[T], chunk_size: usize) -> T {
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| dot(x, y)).sum()
}

#[inline(always)]
pub fn dot<T: Copy + Default + Add<Output = T> + AddAssign + Mul<Output = T>>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len());

    let mut x = x;
    let mut y = y;

    let zero = T::default();
    let (mut acc0, mut acc1, mut acc2, mut acc3, mut acc4, mut acc5, mut acc6, mut acc7) = (zero, zero, zero, zero, zero, zero, zero, zero);

    while x.len() >= 8 {
        acc0 += x[0].mul(y[0]);
        acc1 += x[1].mul(y[1]);
        acc2 += x[2].mul(y[2]);
        acc3 += x[3].mul(y[3]);
        acc4 += x[4].mul(y[4]);
        acc5 += x[5].mul(y[5]);
        acc6 += x[6].mul(y[6]);
        acc7 += x[7].mul(y[7]);

        x = &x[8..];
        y = &y[8..];
    }

    let mut sum = (acc0 + acc4) + (acc1 + acc5) + (acc2 + acc6) + (acc3 + acc7);

    for (x, y) in x.iter().zip(y) {
        sum += x.mul(*y);
    }

    sum
}
