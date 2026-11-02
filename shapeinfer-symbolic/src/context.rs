// SPDX-FileCopyrightText: 2026 NVIDIA
//
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use cel::FunctionContext;
#[cfg(feature = "z3")]
use z3::ast::{Ast, Bool};

pub use crate::Result;
use crate::expression::Shape;
use crate::{Error, Expression, Vec};

#[derive(Default, Debug, Clone)]
pub struct ContextOptions {}

#[derive(Debug, Clone)]
pub struct Context {
    equalities: std::vec::Vec<(Expression, Expression)>,
    #[cfg(feature = "z3")]
    solver: z3::Solver,
}

pub enum InputDimension<'string> {
    Static(u64),
    Dynamic(&'string str),
}

impl<'string> From<&'string str> for InputDimension<'string> {
    fn from(value: &'string str) -> Self {
        Self::Dynamic(value)
    }
}

impl From<u64> for InputDimension<'_> {
    fn from(value: u64) -> Self {
        Self::Static(value)
    }
}

fn cel_broadcast(
    ftx: &FunctionContext,
    a: i64,
    b: i64,
    msg: Arc<String>,
) -> std::result::Result<i64, cel::ExecutionError> {
    match (a, b) {
        (a, 1) => Ok(a),
        (1, b) => Ok(b),
        (a, b) if a == b => Ok(a),
        _ => Err(ftx.error(msg)),
    }
}

pub type InputShape<'slice, 'string> = &'slice [InputDimension<'string>];

impl Context {
    pub fn equal_dim(&mut self, lhs: &Expression, rhs: &Expression) -> Result<()> {
        if let (Some(a), Some(b)) = (lhs.as_const(), rhs.as_const())
            && a != b
        {
            return Err(Error::UnequalDimensions { left: a, right: b });
        }
        self.equalities.push((lhs.clone(), rhs.clone()));
        #[cfg(feature = "z3")]
        self.solver.assert(lhs.z3.eq(&rhs.z3));
        Ok(())
    }

    pub fn new(_options: &ContextOptions) -> Self {
        #[cfg(feature = "z3")]
        let solver = z3::Solver::new();

        Self {
            equalities: std::vec::Vec::new(),
            #[cfg(feature = "z3")]
            solver,
        }
    }

    pub fn input(&mut self, shape: &[u64]) -> Result<Shape> {
        let mut exprs = Vec::new();
        for s in shape.iter().copied() {
            exprs.push(Expression::new_const(
                s.try_into()
                    .map_err(|_| crate::Error::ExceededI64Range { value: s })?,
            ));
        }
        Ok(Shape::from_vec(exprs))
    }

    pub fn dynamic_input(&mut self, shape: InputShape) -> Result<Shape> {
        let mut exprs = Vec::new();
        for s in shape.iter() {
            exprs.push(match s {
                &InputDimension::Static(s) => Expression::new_const(
                    s.try_into()
                        .map_err(|_| crate::Error::ExceededI64Range { value: s })?,
                ),
                InputDimension::Dynamic(d) => Expression::new_dynamic(d),
            });
        }

        let shape = Shape::from_vec(exprs);
        self.ensure_shape(&shape)?;

        Ok(shape)
    }

    #[cfg(feature = "z3")]
    fn ensure_shape_z3(&mut self, shape: &Shape) -> Result<()> {
        for s in shape.iter() {
            self.solver.assert(s.z3.ge(0));
        }
        Ok(())
    }

    /// Ensure that this model is consistent with having `shape` be a shape of a tensor.
    /// This means that all expressions of shape must be non-negative.
    #[allow(unused_variables)]
    pub fn ensure_shape(&mut self, shape: &Shape) -> Result<()> {
        #[cfg(feature = "z3")]
        self.ensure_shape_z3(shape)?;

        Ok(())
    }

    pub fn broadcast_dim(
        &mut self,
        lhs: Expression,
        rhs: Expression,
        assertion_message: &str,
    ) -> Result<Expression> {
        if lhs.as_const() == Some(1) {
            return Ok(rhs);
        }
        if rhs.as_const() == Some(1) || lhs.to_string() == rhs.to_string() {
            return Ok(lhs);
        }
        #[cfg(feature = "z3")]
        self.solver
            .assert(Bool::or(&[lhs.z3.eq(&rhs.z3), lhs.z3.eq(1), rhs.z3.eq(1)]));
        Ok(Expression {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(cel::common::ast::CallExpr {
                    func_name: "broadcast".to_string(),
                    target: None,
                    args: vec![
                        lhs.cel,
                        rhs.cel,
                        cel::common::ast::IdedExpr {
                            id: 0,
                            expr: cel::common::ast::Expr::Literal(
                                cel::common::ast::LiteralValue::String(
                                    assertion_message.to_string().into(),
                                ),
                            ),
                        },
                    ],
                }),
            },
            #[cfg(feature = "z3")]
            z3: lhs.z3.gt(&rhs.z3).ite(&lhs.z3, &rhs.z3),
        })
    }

    //pub fn broadcast_shapes_no_rank_change(shapes: &[Shape]) -> Result<Shape> {
    //if shapes.is_empty() {
    //return Err(Error::BroadcastEmptyShapeList);
    //}
    //let mut shape = shapes[0];

    //}

    //pub fn concat(&mut self, shapes: &[Shape], axis: usize) -> Result<Shape> {}

    #[cfg(feature = "z3")]
    pub fn compute_shapes_z3(
        &self,
        shapes_to_resolve: &[Shape],
        input_variables: &BTreeMap<&str, u32>,
    ) -> Result<Vec<Vec<cel::Value>>> {
        let assumptions: std::vec::Vec<Bool> = input_variables
            .iter()
            .map(|(&name, &value)| z3::ast::Int::new_const(name).eq(value as i64))
            .collect();

        match self.solver.check_assumptions(&assumptions) {
            z3::SatResult::Unsat => return Err(Error::UnsatisfiableShapeConstraints),
            z3::SatResult::Unknown => {
                return Err(Error::Z3Unknown);
            }
            z3::SatResult::Sat => {}
        }

        let model = self.solver.get_model().ok_or(Error::Z3ModelUnavailable)?;
        shapes_to_resolve
            .iter()
            .map(|shape| {
                shape
                    .iter()
                    .map(|dimension| {
                        let value = model.eval(&dimension.z3, true).ok_or_else(|| {
                            Error::Z3DimensionEvaluationFailed {
                                expression: dimension.to_string(),
                            }
                        })?;
                        let number =
                            value.as_i64().ok_or_else(|| Error::Z3DimensionOutOfRange {
                                expression: dimension.to_string(),
                                value: value.simplify().to_string(),
                            })?;
                        Ok(cel::Value::Int(number))
                    })
                    .collect()
            })
            .collect()
    }

    pub fn compute_shapes_cel(
        &self,
        shapes_to_resolve: &[Shape],
        input_variables: &BTreeMap<&str, u32>,
    ) -> Result<Vec<Vec<cel::Value>>> {
        let mut out = Vec::new();

        let mut context = cel::Context::default();
        context.add_function("broadcast", cel_broadcast);
        context.add_function("max", |a: i64, b: i64| a.max(b));
        context.add_function("min", |a: i64, b: i64| a.min(b));
        context.add_function("roundScale", |dimension: i64, bits: i64| {
            ((dimension as f32) * f32::from_bits(bits as u32))
                .round()
                .max(1.0) as i64
        });

        for (&name, &value) in input_variables.iter() {
            context.add_variable_from_value(name, value as i64);
        }

        for (lhs, rhs) in &self.equalities {
            let left = context.resolve(&lhs.cel)?;
            let right = context.resolve(&rhs.cel)?;
            if left != right {
                return Err(Error::RuntimeUnequalDimensions { left, right });
            }
        }

        for shape in shapes_to_resolve.iter() {
            let mut exprs = Vec::new();
            for s in shape.iter() {
                exprs.push(context.resolve(&s.cel)?);
            }

            out.push(exprs);
        }

        Ok(out)
    }

    /// Compute conservative upper bounds from upper bounds for input dimensions.
    pub fn compute_shape_upper_bounds(
        &self,
        shape: &Shape,
        input_upper_bounds: &BTreeMap<&str, u32>,
    ) -> Result<std::vec::Vec<u32>> {
        shape
            .iter()
            .map(|expression| expression.upper_bound(input_upper_bounds))
            .collect()
    }

    pub fn compute_shapes_cel_comcrete(
        &self,
        shapes_to_resolve: &[Shape],
        input_variables: &BTreeMap<&str, u32>,
    ) -> Result<Vec<Vec<u32>>> {
        let shapes = self.compute_shapes_cel(shapes_to_resolve, input_variables)?;
        shapes
            .iter()
            .map(|shape| {
                shape
                    .iter()
                    .map(|s| match s {
                        cel::Value::Int(i) => (*i)
                            .try_into()
                            .map_err(|_| Error::I64DimensionNotInU32Range { i64: *i }),
                        cel::Value::UInt(u) => (*u)
                            .try_into()
                            .map_err(|_| Error::U64DimensionNotInU32Range { u64: *u }),
                        s => Err(Error::DimensionNotAConstant { expr: s.clone() }),
                    })
                    .collect()
            })
            .collect()
    }

    pub fn binary_with_same_rank_broadcast(
        &mut self,
        a: &Shape,
        b: &Shape,
        label: &str,
    ) -> Result<Shape> {
        let expr = a
            .iter()
            .zip(b.iter())
            .map(|(d1, d2)| {
                self.broadcast_dim(
                    d1.clone(),
                    d2.clone(),
                    &format!("failed to perform broadcast for {label}").clone(),
                )
            })
            .collect::<Result<Vec<Expression>>>()?;

        Ok(Shape::from_vec(expr))
    }
}

pub fn pad(shape: &Shape, pad: &[u32]) -> Result<Shape> {
    if shape.rank_usize() != pad.len() {
        return Err(Error::InvalidPadArgument {
            shape: Box::new(shape.clone()),
            pad: pad.into(),
        });
    }

    let mut expr = Vec::new();
    for (s, p) in shape.iter().zip(pad.iter().copied()) {
        expr.push(s.clone() + Expression::new_const(p as i64));
    }

    Ok(Shape::from_vec(expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "z3")]
    #[test]
    fn z3_out_of_range_error_keeps_expression_and_value() {
        let context = Context::new(&ContextOptions::default());
        let expression = Expression::new_dynamic("n") * Expression::new_const(i64::MAX);
        let shape = Shape::from_vec(vec![expression].into());
        let error = context
            .compute_shapes_z3(&[shape], &[("n", 2)].into_iter().collect())
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Z3DimensionOutOfRange { expression, value }
                if expression.contains("n") && value == "18446744073709551614"
        ));
    }

    #[cfg(feature = "z3")]
    #[test]
    fn eval_z3_with_independent_inputs() {
        let mut context = Context::new(&ContextOptions::default());
        let input = context.dynamic_input(&["a".into(), "b".into()]).unwrap();
        let padded = pad(&input, &[3, 4]).unwrap();
        let shapes = [input, padded];

        let values = context
            .compute_shapes_z3(&shapes, &[("a", 2), ("b", 5)].into_iter().collect())
            .unwrap();
        assert_eq!(&values[0][..], &[2.into(), 5.into()]);
        assert_eq!(&values[1][..], &[5.into(), 9.into()]);

        let values = context
            .compute_shapes_z3(&shapes, &[("a", 7), ("b", 1)].into_iter().collect())
            .unwrap();
        assert_eq!(&values[0][..], &[7.into(), 1.into()]);
        assert_eq!(&values[1][..], &[10.into(), 5.into()]);
    }

    #[cfg(feature = "z3")]
    #[test]
    fn eval_z3_rejects_invalid_broadcast() {
        let mut context = Context::new(&ContextOptions::default());
        let a = context.dynamic_input(&["a".into()]).unwrap();
        let b = context.dynamic_input(&["b".into()]).unwrap();
        let output = context
            .binary_with_same_rank_broadcast(&a, &b, "test")
            .unwrap();

        std::assert_matches!(
            context.compute_shapes_z3(
                std::slice::from_ref(&output),
                &[("a", 2), ("b", 3)].into_iter().collect()
            ),
            Err(Error::UnsatisfiableShapeConstraints)
        );

        let values = context
            .compute_shapes_z3(&[output], &[("a", 2), ("b", 1)].into_iter().collect())
            .unwrap();
        assert_eq!(&values[0][..], &[2.into()]);
    }

    #[test]
    fn eval() {
        let mut context = Context::new(&ContextOptions::default());
        let shape = context.dynamic_input(&["a".into(), "b".into()]).unwrap();
        assert_eq!(
            &(context
                .compute_shapes_cel_comcrete(
                    std::slice::from_ref(&shape),
                    &[("a", 1u32), ("b", 2u32)].iter().copied().collect()
                )
                .unwrap()[0])[..],
            &[1u32, 2u32][..]
        );
        let padded = pad(&shape, &[3, 3]).unwrap();
        assert_eq!(
            &(context
                .compute_shapes_cel_comcrete(
                    std::slice::from_ref(&padded),
                    &[("a", 1u32), ("b", 2u32)].iter().copied().collect()
                )
                .unwrap()[0])[..],
            &[4u32, 5u32][..]
        );

        std::assert_matches!(
            context
                .compute_shapes_cel(std::slice::from_ref(&padded), &[].iter().copied().collect())
                .unwrap_err(),
            Error::CelExecutionError {
                source: cel::ExecutionError::UndeclaredReference(..)
            }
        );

        let other = context.dynamic_input(&["a".into(), 1.into()]).unwrap();
        let output = context
            .binary_with_same_rank_broadcast(&shape, &other, "c")
            .unwrap();
        assert_eq!(
            &(context
                .compute_shapes_cel_comcrete(
                    std::slice::from_ref(&output),
                    &[("a", 5u32), ("b", 7u32)].iter().copied().collect()
                )
                .unwrap()[0])[..],
            &[5u32, 7u32][..]
        );

        let other = context.dynamic_input(&["a".into(), 9.into()]).unwrap();
        let output = context
            .binary_with_same_rank_broadcast(&shape, &other, "c")
            .unwrap();
        std::assert_matches!(
            context.compute_shapes_cel_comcrete(
                std::slice::from_ref(&output),
                &[("a", 5u32), ("b", 7u32)].iter().copied().collect()
            ),
            Err(Error::CelExecutionError { .. })
        );
    }
}
