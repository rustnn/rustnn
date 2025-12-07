/// Shape inference and validation for WebNN operations
use crate::error::GraphError;

/// Compute the broadcasted shape for two operands following NumPy broadcasting rules
///
/// Broadcasting rules:
/// 1. If arrays have different ranks, prepend 1s to the smaller rank
/// 2. Two dimensions are compatible if they are equal or one of them is 1
/// 3. Output shape is the maximum of each dimension
pub fn broadcast_shapes(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    let max_rank = shape_a.len().max(shape_b.len());
    let mut result = Vec::with_capacity(max_rank);

    // Iterate from right to left (least significant dimension first)
    for i in 0..max_rank {
        let dim_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };

        let dim_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b || dim_a == 1 || dim_b == 1 {
            result.push(dim_a.max(dim_b));
        } else {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Incompatible shapes for broadcasting: {:?} and {:?} (dimension {} incompatible: {} vs {})",
                    shape_a, shape_b, i, dim_a, dim_b
                ),
            });
        }
    }

    // Reverse to get back to original order
    result.reverse();
    Ok(result)
}

/// Infer output shape for matrix multiplication (matmul)
///
/// For 2D matrices: [M, K] @ [K, N] -> [M, N]
/// For batched matmul: broadcasting is applied to batch dimensions
pub fn infer_matmul_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    if shape_a.len() < 2 || shape_b.len() < 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Matmul requires at least 2D tensors, got shapes {:?} and {:?}",
                shape_a, shape_b
            ),
        });
    }

    let a_rows = shape_a[shape_a.len() - 2];
    let a_cols = shape_a[shape_a.len() - 1];
    let b_rows = shape_b[shape_b.len() - 2];
    let b_cols = shape_b[shape_b.len() - 1];

    if a_cols != b_rows {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Incompatible shapes for matmul: {:?} and {:?} (inner dimensions {} != {})",
                shape_a, shape_b, a_cols, b_rows
            ),
        });
    }

    // For simple 2D case
    if shape_a.len() == 2 && shape_b.len() == 2 {
        return Ok(vec![a_rows, b_cols]);
    }

    // For batched matmul, broadcast batch dimensions
    let batch_a = &shape_a[..shape_a.len() - 2];
    let batch_b = &shape_b[..shape_b.len() - 2];
    let mut batch_dims = broadcast_shapes(batch_a, batch_b)?;
    batch_dims.push(a_rows);
    batch_dims.push(b_cols);

    Ok(batch_dims)
}

/// Validate that a reshape operation is valid
pub fn validate_reshape(input_shape: &[u32], output_shape: &[u32]) -> Result<(), GraphError> {
    let input_size: u32 = input_shape.iter().product();
    let output_size: u32 = output_shape.iter().product();

    if input_size != output_size {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Reshape requires same number of elements: input {:?} ({} elements) != output {:?} ({} elements)",
                input_shape, input_size, output_shape, output_size
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_same_shape() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_with_ones() {
        assert_eq!(broadcast_shapes(&[2, 3], &[1, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_different_ranks() {
        assert_eq!(
            broadcast_shapes(&[2, 3, 4], &[3, 4]).unwrap(),
            vec![2, 3, 4]
        );
        assert_eq!(
            broadcast_shapes(&[3, 4], &[2, 3, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_scalar() {
        assert_eq!(broadcast_shapes(&[2, 3], &[1]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
        assert!(broadcast_shapes(&[2, 3, 4], &[2, 5, 4]).is_err());
    }

    #[test]
    fn test_matmul_2d() {
        assert_eq!(infer_matmul_shape(&[2, 3], &[3, 4]).unwrap(), vec![2, 4]);
    }

    #[test]
    fn test_matmul_batched() {
        assert_eq!(
            infer_matmul_shape(&[5, 2, 3], &[5, 3, 4]).unwrap(),
            vec![5, 2, 4]
        );
    }

    #[test]
    fn test_matmul_incompatible() {
        assert!(infer_matmul_shape(&[2, 3], &[4, 5]).is_err());
        assert!(infer_matmul_shape(&[2], &[3, 4]).is_err());
    }

    #[test]
    fn test_validate_reshape_valid() {
        assert!(validate_reshape(&[2, 3], &[6]).is_ok());
        assert!(validate_reshape(&[2, 3, 4], &[6, 4]).is_ok());
        assert!(validate_reshape(&[6], &[2, 3]).is_ok());
    }

    #[test]
    fn test_validate_reshape_invalid() {
        assert!(validate_reshape(&[2, 3], &[5]).is_err());
        assert!(validate_reshape(&[2, 3, 4], &[5, 5]).is_err());
    }
}
