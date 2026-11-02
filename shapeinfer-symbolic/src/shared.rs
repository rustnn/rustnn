// SPDX-FileCopyrightText: 2026 NVIDIA
//
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{self, Display},
    sync::Arc,
};

use cel::common::ast::{
    CallExpr, Expr, IdedExpr,
    operators::{ADD, DIVIDE, MULTIPLY, SUBSTRACT},
};

use crate::{Error, Result};

#[derive(Clone)]
pub(crate) struct SharedExpression(Arc<Node>);

struct Node {
    template: IdedExpr,
    args: std::vec::Vec<SharedExpression>,
}

// Call-scoped caches use pointer keys while the live roots retain every shared node.
pub(crate) type ValueCache = HashMap<usize, cel::Value>;
pub(crate) type BoundCache = HashMap<usize, (i64, i64)>;

impl SharedExpression {
    pub(crate) fn from_ast(ast: IdedExpr) -> Self {
        if let Expr::Call(call) = &ast.expr
            && call.target.is_none()
            && matches!(
                call.func_name.as_str(),
                ADD | SUBSTRACT | MULTIPLY | DIVIDE | "max" | "min" | "roundScale" | "broadcast"
            )
            && call.args.len() == if call.func_name == "broadcast" { 3 } else { 2 }
        {
            let Expr::Call(call) = ast.expr else {
                unreachable!()
            };
            return Self::call(
                &call.func_name,
                call.args.into_iter().map(Self::from_ast).collect(),
            );
        }
        Self(Arc::new(Node {
            template: ast,
            args: std::vec::Vec::new(),
        }))
    }

    pub(crate) fn call(function: &str, args: std::vec::Vec<Self>) -> Self {
        let template = IdedExpr {
            id: 0,
            expr: Expr::Call(CallExpr {
                func_name: function.to_owned(),
                target: None,
                args: (0..args.len())
                    .map(|i| IdedExpr {
                        id: 0,
                        expr: Expr::Ident(format!("__shape_arg_{i}")),
                    })
                    .collect(),
            }),
        };
        Self(Arc::new(Node { template, args }))
    }

    fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub(crate) fn as_const(&self) -> Option<i64> {
        match &self.0.template.expr {
            Expr::Literal(cel::common::ast::LiteralValue::Int(value)) => Some(*value.inner()),
            _ => None,
        }
    }

    pub(crate) fn as_dynamic(&self) -> Option<&str> {
        if !self.0.args.is_empty() {
            return None;
        }
        match &self.0.template.expr {
            Expr::Ident(name) => Some(name),
            _ => None,
        }
    }

    pub(crate) fn structurally_equal(&self, other: &Self) -> bool {
        let mut visited = HashSet::new();
        let mut pending = vec![(self, other)];
        while let Some((a, b)) = pending.pop() {
            if Arc::ptr_eq(&a.0, &b.0) || !visited.insert((a.id(), b.id())) {
                continue;
            }
            if a.0.template.expr != b.0.template.expr || a.0.args.len() != b.0.args.len() {
                return false;
            }
            pending.extend(a.0.args.iter().zip(&b.0.args));
        }
        true
    }

    pub(crate) fn resolve(
        &self,
        context: &cel::Context<'_>,
        cache: &mut ValueCache,
    ) -> std::result::Result<cel::Value, cel::ExecutionError> {
        if let Some(value) = cache.get(&self.id()) {
            return Ok(value.clone());
        }
        let value = if self.0.args.is_empty() {
            context.resolve(&self.0.template)?
        } else {
            let mut scope = context.new_inner_scope();
            for (i, arg) in self.0.args.iter().enumerate() {
                scope.add_variable_from_value(
                    format!("__shape_arg_{i}"),
                    arg.resolve(context, cache)?,
                );
            }
            scope.resolve(&self.0.template)?
        };
        cache.insert(self.id(), value.clone());
        Ok(value)
    }

    pub(crate) fn interval(
        &self,
        bounds: &BTreeMap<&str, u32>,
        cache: &mut BoundCache,
    ) -> Result<(i64, i64)> {
        if let Some(interval) = cache.get(&self.id()) {
            return Ok(*interval);
        }
        let interval = match &self.0.template.expr {
            Expr::Literal(cel::common::ast::LiteralValue::Int(value)) => {
                let value = *value.inner();
                (value, value)
            }
            Expr::Ident(name) => {
                let max = bounds
                    .get(name.as_str())
                    .ok_or_else(|| Error::MissingUpperBound { name: name.clone() })?;
                (0, *max as i64)
            }
            Expr::Call(call) if self.0.args.len() >= 2 => {
                let (a0, a1) = self.0.args[0].interval(bounds, cache)?;
                let (b0, b1) = self.0.args[1].interval(bounds, cache)?;
                match call.func_name.as_str() {
                    ADD => (a0.saturating_add(b0), a1.saturating_add(b1)),
                    SUBSTRACT => (a0.saturating_sub(b1), a1.saturating_sub(b0)),
                    MULTIPLY => {
                        let products = [
                            a0.saturating_mul(b0),
                            a0.saturating_mul(b1),
                            a1.saturating_mul(b0),
                            a1.saturating_mul(b1),
                        ];
                        (
                            *products.iter().min().unwrap(),
                            *products.iter().max().unwrap(),
                        )
                    }
                    DIVIDE if b0 > 0 => {
                        let quotients = [
                            a0.div_euclid(b0),
                            a0.div_euclid(b1),
                            a1.div_euclid(b0),
                            a1.div_euclid(b1),
                        ];
                        (
                            *quotients.iter().min().unwrap(),
                            *quotients.iter().max().unwrap(),
                        )
                    }
                    DIVIDE => return Err(Error::UnboundedDivision),
                    "max" => (a0.max(b0), a1.max(b1)),
                    "min" => (a0.min(b0), a1.min(b1)),
                    "broadcast" => (a0.min(b0), a1.max(b1)),
                    "roundScale" if b0 == b1 => {
                        let scale = f32::from_bits(b0 as u32);
                        if !scale.is_finite() || scale <= 0.0 {
                            return Err(Error::InvalidScale { scale });
                        }
                        (
                            ((a0.max(1) as f32) * scale).round().max(1.0) as i64,
                            ((a1.max(1) as f32) * scale).round().max(1.0) as i64,
                        )
                    }
                    "roundScale" => return Err(Error::NonConstantScaleBits),
                    name => {
                        return Err(Error::UnsupportedSymbolicFunction {
                            name: name.to_owned(),
                        });
                    }
                }
            }
            Expr::Call(call) if call.args.len() >= 2 => {
                return Err(Error::UnsupportedSymbolicFunction {
                    name: call.func_name.clone(),
                });
            }
            _ => return Err(Error::UnsupportedSymbolicExpression),
        };
        cache.insert(self.id(), interval);
        Ok(interval)
    }
}

impl fmt::Debug for SharedExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedExpression")
            .field("template", &self.0.template)
            .field(
                "children",
                &self
                    .0
                    .args
                    .iter()
                    .map(Self::id)
                    .collect::<std::vec::Vec<_>>(),
            )
            .finish()
    }
}

impl Display for SharedExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.args.is_empty() {
            return crate::expression::MyExpression(&self.0.template).fmt(f);
        }
        let Expr::Call(call) = &self.0.template.expr else {
            unreachable!()
        };
        let operator = match call.func_name.as_str() {
            ADD => Some("+"),
            SUBSTRACT => Some("-"),
            MULTIPLY => Some("*"),
            DIVIDE => Some("/"),
            _ => None,
        };
        if let Some(operator) = operator {
            return write!(f, "({} {operator} {})", self.0.args[0], self.0.args[1]);
        }
        write!(f, "{}(", call.func_name)?;
        for (i, arg) in self.0.args.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }
            Display::fmt(arg, f)?;
        }
        f.write_str(")")
    }
}
