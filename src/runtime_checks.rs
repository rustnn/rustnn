use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::{Dimension, OperandDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Input,
    Output,
}

impl TensorKind {
    fn as_str(self) -> &'static str {
        match self {
            TensorKind::Input => "input",
            TensorKind::Output => "output",
        }
    }
}

#[derive(Debug, Clone)]
struct BoundDynamicDim {
    value: usize,
}

#[derive(Debug, Default, Clone)]
pub struct RuntimeShapeState {
    bound_dims: HashMap<String, BoundDynamicDim>,
}

impl RuntimeShapeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate_named_shapes(
        &mut self,
        actual_shapes: &HashMap<String, Vec<usize>>,
        descriptors: &HashMap<String, OperandDescriptor>,
        kind: TensorKind,
    ) -> Result<(), GraphError> {
        for name in descriptors.keys() {
            if !actual_shapes.contains_key(name) {
                return Err(GraphError::RuntimeTensorMissing {
                    kind: kind.as_str().to_string(),
                    name: name.clone(),
                });
            }
        }

        for name in actual_shapes.keys() {
            if !descriptors.contains_key(name) {
                return Err(GraphError::RuntimeTensorUnexpected {
                    kind: kind.as_str().to_string(),
                    name: name.clone(),
                });
            }
        }

        for (name, actual_shape) in actual_shapes {
            let descriptor = descriptors.get(name).expect("checked above");
            self.validate_shape(name, actual_shape, descriptor, kind)?;
        }

        Ok(())
    }

    pub fn validate_shape(
        &mut self,
        name: &str,
        actual_shape: &[usize],
        descriptor: &OperandDescriptor,
        kind: TensorKind,
    ) -> Result<(), GraphError> {
        if actual_shape.len() != descriptor.shape.len() {
            return Err(GraphError::RuntimeTensorRankMismatch {
                kind: kind.as_str().to_string(),
                name: name.to_string(),
                expected_rank: descriptor.shape.len(),
                actual_rank: actual_shape.len(),
            });
        }

        for (axis, (actual, expected_dim)) in actual_shape
            .iter()
            .copied()
            .zip(&descriptor.shape)
            .enumerate()
        {
            match expected_dim {
                Dimension::Static(expected) => {
                    if actual != *expected as usize {
                        return Err(GraphError::RuntimeStaticDimensionMismatch {
                            kind: kind.as_str().to_string(),
                            name: name.to_string(),
                            axis,
                            expected: *expected,
                            actual,
                        });
                    }
                }
                Dimension::Dynamic(dynamic) => {
                    if actual > dynamic.max_size as usize {
                        return Err(GraphError::RuntimeDynamicDimensionExceeded {
                            kind: kind.as_str().to_string(),
                            name: name.to_string(),
                            axis,
                            dim_name: dynamic.name.clone(),
                            max_size: dynamic.max_size,
                            actual,
                        });
                    }

                    if let Some(bound) = self.bound_dims.get(&dynamic.name) {
                        if bound.value != actual {
                            return Err(GraphError::RuntimeDynamicDimensionNameMismatch {
                                dim_name: dynamic.name.clone(),
                                expected: bound.value,
                                actual,
                            });
                        }
                    } else {
                        self.bound_dims
                            .insert(dynamic.name.clone(), BoundDynamicDim { value: actual });
                    }
                }
            }
        }

        Ok(())
    }
}

pub fn validate_shape_data_length(
    name: &str,
    shape: &[usize],
    data_len: usize,
) -> Result<(), GraphError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| GraphError::RuntimeTensorShapeOverflow {
            name: name.to_string(),
            shape: shape.to_vec(),
        })?;

    if data_len != expected {
        return Err(GraphError::RuntimeTensorDataLengthMismatch {
            name: name.to_string(),
            expected,
            actual: data_len,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Dimension, DynamicDimension, OperandDescriptor};

    fn input_desc(shape: Vec<Dimension>) -> OperandDescriptor {
        OperandDescriptor {
            data_type: DataType::Float32,
            shape,
            pending_permutation: vec![],
        }
    }

    #[test]
    fn validates_static_shape() {
        let mut state = RuntimeShapeState::new();
        let desc = input_desc(vec![Dimension::Static(2), Dimension::Static(3)]);
        assert!(
            state
                .validate_shape("x", &[2, 3], &desc, TensorKind::Input)
                .is_ok()
        );
    }

    #[test]
    fn rejects_static_mismatch() {
        let mut state = RuntimeShapeState::new();
        let desc = input_desc(vec![Dimension::Static(2), Dimension::Static(3)]);
        let err = state
            .validate_shape("x", &[2, 4], &desc, TensorKind::Input)
            .unwrap_err();
        assert!(matches!(
            err,
            GraphError::RuntimeStaticDimensionMismatch { axis: 1, .. }
        ));
    }

    #[test]
    fn rejects_dynamic_greater_than_max_size() {
        let mut state = RuntimeShapeState::new();
        let desc = input_desc(vec![Dimension::Dynamic(DynamicDimension {
            name: "batch".to_string(),
            max_size: 8,
        })]);
        let err = state
            .validate_shape("x", &[9], &desc, TensorKind::Input)
            .unwrap_err();
        assert!(matches!(
            err,
            GraphError::RuntimeDynamicDimensionExceeded { .. }
        ));
    }

    #[test]
    fn enforces_same_named_dynamic_dimensions_across_inputs_and_outputs() {
        let mut state = RuntimeShapeState::new();
        let input = input_desc(vec![
            Dimension::Dynamic(DynamicDimension {
                name: "batch".to_string(),
                max_size: 16,
            }),
            Dimension::Static(64),
        ]);
        let output = input_desc(vec![
            Dimension::Dynamic(DynamicDimension {
                name: "batch".to_string(),
                max_size: 16,
            }),
            Dimension::Static(10),
        ]);

        state
            .validate_shape("x", &[4, 64], &input, TensorKind::Input)
            .unwrap();

        let err = state
            .validate_shape("y", &[5, 10], &output, TensorKind::Output)
            .unwrap_err();
        assert!(matches!(
            err,
            GraphError::RuntimeDynamicDimensionNameMismatch { .. }
        ));
    }

    #[test]
    fn validates_named_shape_sets() {
        let mut state = RuntimeShapeState::new();
        let mut descs = HashMap::new();
        descs.insert(
            "x".to_string(),
            input_desc(vec![Dimension::Static(2), Dimension::Static(3)]),
        );
        let mut actual = HashMap::new();
        actual.insert("x".to_string(), vec![2, 3]);

        assert!(
            state
                .validate_named_shapes(&actual, &descs, TensorKind::Input)
                .is_ok()
        );
    }

    #[test]
    fn rejects_missing_named_tensor() {
        let mut state = RuntimeShapeState::new();
        let mut descs = HashMap::new();
        descs.insert("x".to_string(), input_desc(vec![Dimension::Static(1)]));
        let actual = HashMap::new();

        let err = state
            .validate_named_shapes(&actual, &descs, TensorKind::Input)
            .unwrap_err();
        assert!(matches!(err, GraphError::RuntimeTensorMissing { .. }));
    }

    #[test]
    fn rejects_unexpected_named_tensor() {
        let mut state = RuntimeShapeState::new();
        let descs = HashMap::new();
        let mut actual = HashMap::new();
        actual.insert("x".to_string(), vec![1]);

        let err = state
            .validate_named_shapes(&actual, &descs, TensorKind::Input)
            .unwrap_err();
        assert!(matches!(err, GraphError::RuntimeTensorUnexpected { .. }));
    }

    #[test]
    fn validates_shape_data_length() {
        assert!(validate_shape_data_length("x", &[2, 3], 6).is_ok());
        assert!(validate_shape_data_length("x", &[], 1).is_ok());
    }

    #[test]
    fn rejects_shape_data_length_mismatch() {
        let err = validate_shape_data_length("x", &[2, 3], 7).unwrap_err();
        assert!(matches!(
            err,
            GraphError::RuntimeTensorDataLengthMismatch { .. }
        ));
    }
}
