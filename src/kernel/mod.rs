pub(crate) mod dot32;
pub(crate) mod dot64;
pub(crate) mod scal32;
pub(crate) mod scal64;
pub(crate) mod sum32;
pub(crate) mod sum64;

pub mod f32;
pub mod f64;

pub const DEFAULT_DOT_CHUNK: usize = 16 * 1024;
