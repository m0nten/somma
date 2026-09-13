use thiserror::Error;

#[derive(Debug, Error)]
pub enum SommaError {
    #[error("vectors is different")]
    DifferentVectors,
}
