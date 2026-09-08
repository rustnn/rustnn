use crate::{Expression, Vec, expression::Shape};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("dimensions {left} and {right} must be equal")]
    UnequalDimensions { left: i64, right: i64 },
    #[error("dimensions {left:?} and {right:?} are not equal at runtime")]
    RuntimeUnequalDimensions { left: cel::Value, right: cel::Value },
    #[error("cannot broadcast dimensions {left} and {right}")]
    IncompatibleBroadcastDimensions { left: i64, right: i64 },
    #[error("concat needs at least one shape")]
    EmptyConcat,
    #[error("concat shape has rank {actual}, expected {expected}")]
    ConcatRankMismatch { expected: usize, actual: usize },
    #[error("transpose permutation has length {actual}, expected {expected}")]
    TransposePermutationLength { expected: usize, actual: usize },
    #[error("transpose axis {axis} is out of range for rank {rank}")]
    TransposeAxisOutOfRange { axis: usize, rank: usize },
    #[error("transpose axis {axis} appears more than once")]
    DuplicateTransposeAxis { axis: usize },
    #[error("reduction axis {axis} is out of range for rank {rank}")]
    ReductionAxisOutOfRange { axis: usize, rank: usize },
    #[error("reduction axis {axis} appears more than once")]
    DuplicateReductionAxis { axis: usize },
    #[error("unsqueeze axis {axis} is out of range for output rank {rank}")]
    UnsqueezeAxisOutOfRange { axis: usize, rank: usize },
    #[error("unsqueeze axis {axis} appears more than once")]
    DuplicateUnsqueezeAxis { axis: usize },
    #[error("tile has {actual} repetitions for rank {expected}")]
    TileRankMismatch { expected: usize, actual: usize },
    #[error("matmul requires non-scalar inputs")]
    MatmulScalarInput,
    #[error("convolution requires rank-four input and filter, got ranks {input} and {filter}")]
    ConvolutionRank { input: usize, filter: usize },
    #[error("convolution groups must be positive")]
    ZeroConvolutionGroups,
    #[error("squeeze axis {axis} is out of range for rank {rank}")]
    SqueezeAxisOutOfRange { axis: usize, rank: usize },
    #[error("squeeze axis {axis} appears more than once")]
    DuplicateSqueezeAxis { axis: usize },
    #[error("window stride must be positive")]
    ZeroWindowStride,
    #[error("window dilation must be positive")]
    ZeroWindowDilation,
    #[error("split axis {axis} is out of range for rank {rank}")]
    SplitAxisOutOfRange { axis: usize, rank: usize },
    #[error("split requires at least one part")]
    SplitZeroParts,
    #[error("split dimension {dimension} is not divisible by {parts} parts")]
    SplitNotDivisible { dimension: i64, parts: u32 },
    #[error("split requires at least one size")]
    SplitSizesEmpty,
    #[error("split sizes sum {total} exceeds i64")]
    SplitSizesOverflow { total: u64 },
    #[error("slice arguments have ranks {starts}, {sizes}, {strides}; input rank is {input}")]
    SliceRankMismatch {
        input: usize,
        starts: usize,
        sizes: usize,
        strides: usize,
    },
    #[error("slice stride at axis {axis} must be positive")]
    SliceZeroStride { axis: usize },
    #[error("slice start {start} plus size {size} exceeds input dimension {input} at axis {axis}")]
    SliceOutOfBounds {
        axis: usize,
        start: u32,
        size: i64,
        input: i64,
    },
    #[error("flatten axis {axis} is out of range for rank {rank}")]
    FlattenAxisOutOfRange { axis: usize, rank: usize },
    #[error("scale {scale} must be finite and positive")]
    InvalidScale { scale: f32 },
    #[error("missing upper bound for symbolic dimension {name}")]
    MissingUpperBound { name: String },
    #[error("cannot bound division by a possibly zero or negative dimension")]
    UnboundedDivision,
    #[error("unsupported symbolic function {name}")]
    UnsupportedSymbolicFunction { name: String },
    #[error("unsupported symbolic dimension expression")]
    UnsupportedSymbolicExpression,
    #[error("symbolic upper bound {value} is outside u32")]
    UpperBoundOutOfRange { value: i64 },
    #[error("failed to parse symbolic expression: {source}")]
    ParseExpression {
        #[from]
        source: cel::parser::ParseErrors,
    },
    #[error("roundScale requires constant scale bits")]
    NonConstantScaleBits,
    #[error(
        "Number of expressionsfor ExpressionTensor does not match requeste shape {shape:?}. Expressions: {expressions:#?}"
    )]
    InvalidExpressionTensorShape {
        expressions: Box<Vec<Expression>>,
        shape: Box<Vec<u8>>,
    },

    #[error("Invalid concat axis {axis} for rank {rank}")]
    InvalidConcatAxis { rank: usize, axis: usize },

    #[error(
        "Invalid dimension value {value} for a shape. Shape dimensions must be all non-negative"
    )]
    NegativeShapeDimension { value: i64 },

    #[error("Exceeded i64 range with value {value}")]
    ExceededI64Range { value: u64 },

    #[error("Inconsistent shapes for concat: {shapes:#?}")]
    InconsistentShapesConcat { shapes: Box<Vec<Shape>> },

    #[error("Cannot broadcast empty list of shapes")]
    BroadcastEmptyShapeList,

    #[error("Failed to evaluate using rust-cel: {source}")]
    CelExecutionError {
        #[from]
        source: cel::ExecutionError,
    },

    #[cfg(feature = "z3")]
    #[error("Shape constraints are unsatisfiable for the supplied dimensions")]
    UnsatisfiableShapeConstraints,

    #[cfg(feature = "z3")]
    #[error("Z3 could not determine whether shape constraints are satisfiable")]
    Z3Unknown,

    #[cfg(feature = "z3")]
    #[error("Z3 did not provide a model for the shape constraints")]
    Z3ModelUnavailable,

    #[cfg(feature = "z3")]
    #[error("Z3 could not evaluate shape dimension {expression}")]
    Z3DimensionEvaluationFailed { expression: String },

    #[cfg(feature = "z3")]
    #[error("Z3 shape dimension {expression} evaluated to {value}, outside the i64 range")]
    Z3DimensionOutOfRange { expression: String, value: String },

    #[error("Dimension not in u32 range: {i64}")]
    I64DimensionNotInU32Range { i64: i64 },

    #[error("Dimension not in u32 range: {u64}")]
    U64DimensionNotInU32Range { u64: u64 },

    #[error("Dimension value is not a constant: {expr:?}")]
    DimensionNotAConstant { expr: cel::Value },

    #[error("Invalid pad argument. Shape and pad have different ranks: {shape:?} vs {pad:?}")]
    InvalidPadArgument { shape: Box<Shape>, pad: Vec<u32> },

    /// keep in sync with MAX_SHAPE_DIMENSION
    #[error("Can't create a shape with dimension {dim}. Maximum supported dimension is 8")]
    TooManyDimensionsForShape { dim: usize },
}
