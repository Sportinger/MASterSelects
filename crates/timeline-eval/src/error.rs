//! Error types for timeline evaluation.

use thiserror::Error;

/// Errors that can occur during timeline evaluation.
#[derive(Error, Debug)]
pub enum TimelineEvalError {
    #[error("Nested composition nesting too deep (max {max_depth}): {source_id}")]
    NestingTooDeep { max_depth: usize, source_id: String },

    #[error("Composition not found: {source_id}")]
    CompositionNotFound { source_id: String },

    #[error("Invalid timeline: {reason}")]
    InvalidTimeline { reason: String },
}
