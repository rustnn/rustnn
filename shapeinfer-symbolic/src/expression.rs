use std::{
    collections::BTreeMap,
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

const MAX_SHAPE_DIMENSION: usize = 8;

use cel::common::{
    ast::{
        CallExpr, Expr,
        operators::{ADD, DIVIDE, MULTIPLY, SUBSTRACT},
    },
    types::CelInt,
};

use crate::{Error, Result, Vec};
#[cfg(feature = "z3")]
use z3::{
    DeclKind,
    ast::{Ast, Dynamic},
};

#[cfg(feature = "z3")]
struct Z3Infix(Dynamic);

#[cfg(feature = "z3")]
impl Z3Infix {
    fn write_binary(&self, f: &mut std::fmt::Formatter<'_>, operator: &str) -> std::fmt::Result {
        let children = self.0.children();
        if children.len() < 2 {
            return write!(f, "{}", self.0.simplify());
        }
        write!(f, "(")?;
        for (index, child) in children.into_iter().enumerate() {
            if index != 0 {
                write!(f, " {operator} ")?;
            }
            write!(f, "{}", Z3Infix(child))?;
        }
        write!(f, ")")
    }
}

#[cfg(feature = "z3")]
impl Display for Z3Infix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.0.is_app() {
            return write!(f, "{}", self.0.simplify());
        }
        match self.0.decl().kind() {
            DeclKind::Add => self.write_binary(f, "+"),
            DeclKind::Sub => self.write_binary(f, "-"),
            DeclKind::Mul => self.write_binary(f, "*"),
            DeclKind::Div | DeclKind::Idiv => self.write_binary(f, "/"),
            DeclKind::Mod | DeclKind::Rem => self.write_binary(f, "%"),
            DeclKind::Eq => self.write_binary(f, "=="),
            DeclKind::Distinct if self.0.num_children() == 2 => self.write_binary(f, "!="),
            DeclKind::Lt => self.write_binary(f, "<"),
            DeclKind::Le => self.write_binary(f, "<="),
            DeclKind::Gt => self.write_binary(f, ">"),
            DeclKind::Ge => self.write_binary(f, ">="),
            DeclKind::And => self.write_binary(f, "&&"),
            DeclKind::Or => self.write_binary(f, "||"),
            DeclKind::Uminus | DeclKind::Not => {
                let children = self.0.children();
                if let [child] = children.as_slice() {
                    let operator = if self.0.decl().kind() == DeclKind::Not {
                        "!"
                    } else {
                        "-"
                    };
                    write!(f, "({operator}{})", Z3Infix(Dynamic::from_ast(child)))
                } else {
                    write!(f, "{}", self.0.simplify())
                }
            }
            DeclKind::Ite => {
                let children = self.0.children();
                if let [condition, when_true, when_false] = children.as_slice() {
                    write!(
                        f,
                        "({} ? {} : {})",
                        Z3Infix(Dynamic::from_ast(condition)),
                        Z3Infix(Dynamic::from_ast(when_true)),
                        Z3Infix(Dynamic::from_ast(when_false))
                    )
                } else {
                    write!(f, "{}", self.0.simplify())
                }
            }
            _ => write!(f, "{}", self.0.simplify()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Expression {
    pub(crate) cel: cel::parser::Expression,
    #[cfg(feature = "z3")]
    pub(crate) z3: z3::ast::Int,
}

impl Expression {
    /// Render the Z3 expression for inspection and diagnostics.
    #[cfg(feature = "z3")]
    pub fn z3_string(&self) -> String {
        Z3Infix(Dynamic::from_ast(&self.z3.simplify())).to_string()
    }

    pub fn round_scaled(self, scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::InvalidScale { scale });
        }
        let bits = scale.to_bits();
        #[cfg(feature = "z3")]
        let scaled_z3 = {
            let exponent = ((bits >> 23) & 0xff) as i32;
            let mantissa = if exponent == 0 {
                bits & 0x7f_ffff
            } else {
                (bits & 0x7f_ffff) | 0x80_0000
            };
            let shift = if exponent == 0 { -149 } else { exponent - 150 };
            let mut numerator = z3::ast::Int::from_i64(mantissa as i64);
            let mut denominator = z3::ast::Int::from_i64(1);
            if shift >= 0 {
                for _ in 0..shift {
                    numerator = numerator.mul(z3::ast::Int::from_i64(2));
                }
            } else {
                for _ in 0..-shift {
                    denominator = denominator.mul(z3::ast::Int::from_i64(2));
                }
            }
            let rounded = (self.z3.clone().mul(numerator)
                + denominator.clone().div(z3::ast::Int::from_i64(2)))
            .div(denominator);
            let one = z3::ast::Int::from_i64(1);
            rounded.lt(&one).ite(&one, &rounded)
        };
        Ok(Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: Expr::Call(CallExpr {
                    func_name: "roundScale".to_owned(),
                    target: None,
                    args: vec![self.cel, Expression::new_const(bits as i64).cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: scaled_z3,
        })
    }

    /// Conservative upper bound for a symbolic dimension given bounds for input variables.
    pub fn upper_bound(&self, bounds: &BTreeMap<&str, u32>) -> Result<u32> {
        fn interval(
            expr: &cel::parser::Expression,
            bounds: &BTreeMap<&str, u32>,
        ) -> Result<(i64, i64)> {
            match &expr.expr {
                Expr::Literal(cel::common::ast::LiteralValue::Int(value)) => {
                    let value = *value.inner();
                    Ok((value, value))
                }
                Expr::Ident(name) => bounds
                    .get(name.as_str())
                    .map(|&max| (0, max as i64))
                    .ok_or_else(|| Error::MissingUpperBound { name: name.clone() }),
                Expr::Call(call) if call.args.len() >= 2 => {
                    let (a0, a1) = interval(&call.args[0], bounds)?;
                    let (b0, b1) = interval(&call.args[1], bounds)?;
                    match call.func_name.as_str() {
                        ADD => Ok((a0.saturating_add(b0), a1.saturating_add(b1))),
                        SUBSTRACT => Ok((a0.saturating_sub(b1), a1.saturating_sub(b0))),
                        MULTIPLY => {
                            let products = [
                                a0.saturating_mul(b0),
                                a0.saturating_mul(b1),
                                a1.saturating_mul(b0),
                                a1.saturating_mul(b1),
                            ];
                            Ok((
                                *products.iter().min().unwrap(),
                                *products.iter().max().unwrap(),
                            ))
                        }
                        DIVIDE if b0 > 0 => {
                            let quotients = [
                                a0.div_euclid(b0),
                                a0.div_euclid(b1),
                                a1.div_euclid(b0),
                                a1.div_euclid(b1),
                            ];
                            Ok((
                                *quotients.iter().min().unwrap(),
                                *quotients.iter().max().unwrap(),
                            ))
                        }
                        DIVIDE => Err(Error::UnboundedDivision),
                        "max" => Ok((a0.max(b0), a1.max(b1))),
                        "min" => Ok((a0.min(b0), a1.min(b1))),
                        "broadcast" => Ok((a0.min(b0), a1.max(b1))),
                        "roundScale" if b0 == b1 => {
                            let scale = f32::from_bits(b0 as u32);
                            if !scale.is_finite() || scale <= 0.0 {
                                return Err(Error::InvalidScale { scale });
                            }
                            let low = ((a0.max(1) as f32) * scale).round().max(1.0) as i64;
                            let high = ((a1.max(1) as f32) * scale).round().max(1.0) as i64;
                            Ok((low, high))
                        }
                        "roundScale" => Err(Error::NonConstantScaleBits),
                        name => Err(Error::UnsupportedSymbolicFunction {
                            name: name.to_owned(),
                        }),
                    }
                }
                _ => Err(Error::UnsupportedSymbolicExpression),
            }
        }
        let (_, upper) = interval(&self.cel, bounds)?;
        u32::try_from(upper).map_err(|_| Error::UpperBoundOutOfRange { value: upper })
    }

    pub fn as_const(&self) -> Option<i64> {
        match &self.cel.expr {
            Expr::Literal(cel::common::ast::LiteralValue::Int(value)) => Some(*value.inner()),
            _ => None,
        }
    }

    pub fn as_dynamic(&self) -> Option<&str> {
        match &self.cel.expr {
            Expr::Ident(name) => Some(name),
            _ => None,
        }
    }

    #[cfg(not(feature = "z3"))]
    pub fn parse(source: &str) -> Result<Self> {
        cel::parser::Parser::default()
            .parse(source)
            .map(|cel| Self { cel })
            .map_err(Error::from)
    }

    #[cfg(feature = "z3")]
    pub fn parse(source: &str) -> Result<Self> {
        fn to_z3(expression: &cel::parser::Expression) -> Result<z3::ast::Int> {
            match &expression.expr {
                Expr::Literal(cel::common::ast::LiteralValue::Int(value)) => {
                    Ok(z3::ast::Int::from_i64(*value.inner()))
                }
                Expr::Ident(name) => Ok(z3::ast::Int::new_const(name.as_str())),
                Expr::Call(call) if call.args.len() >= 2 => {
                    let lhs = to_z3(&call.args[0])?;
                    let rhs = to_z3(&call.args[1])?;
                    match call.func_name.as_str() {
                        ADD => Ok(lhs.add(rhs)),
                        SUBSTRACT => Ok(lhs.sub(rhs)),
                        MULTIPLY => Ok(lhs.mul(rhs)),
                        DIVIDE => Ok(lhs.div(rhs)),
                        "max" | "broadcast" => Ok(lhs.gt(&rhs).ite(&lhs, &rhs)),
                        "min" => Ok(lhs.lt(&rhs).ite(&lhs, &rhs)),
                        "roundScale" => {
                            let Expr::Literal(cel::common::ast::LiteralValue::Int(bits)) =
                                &call.args[1].expr
                            else {
                                return Err(Error::NonConstantScaleBits);
                            };
                            let base = Expression {
                                cel: call.args[0].clone(),
                                z3: lhs,
                            };
                            Ok(base.round_scaled(f32::from_bits(*bits.inner() as u32))?.z3)
                        }
                        name => Err(Error::UnsupportedSymbolicFunction {
                            name: name.to_owned(),
                        }),
                    }
                }
                _ => Err(Error::UnsupportedSymbolicExpression),
            }
        }
        let cel = cel::parser::Parser::default()
            .parse(source)
            .map_err(Error::from)?;
        let z3 = to_z3(&cel)?;
        Ok(Self { cel, z3 })
    }

    pub fn new_const(x: i64) -> Self {
        Self {
            cel: cel::common::ast::IdedExpr {
                id: 0,
                expr: cel::common::ast::Expr::Literal(cel::common::ast::LiteralValue::Int(
                    CelInt::from(x),
                )),
            },
            #[cfg(feature = "z3")]
            z3: z3::ast::Int::from_i64(x),
        }
    }

    pub fn new_dynamic(name: &str) -> Self {
        Self {
            cel: cel::common::ast::IdedExpr {
                id: 0,
                expr: Expr::Ident(name.to_owned()),
            },
            #[cfg(feature = "z3")]
            z3: z3::ast::Int::new_const(name),
        }
    }

    pub fn max(self, rhs: Expression) -> Self {
        Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(CallExpr {
                    func_name: "max".to_string(),
                    target: None,
                    args: vec![self.cel, rhs.cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: self.z3.gt(&rhs.z3).ite(&self.z3, &rhs.z3),
        }
    }

    pub fn min(self, rhs: Expression) -> Self {
        Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(CallExpr {
                    func_name: "min".to_string(),
                    target: None,
                    args: vec![self.cel, rhs.cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: self.z3.lt(&rhs.z3).ite(&self.z3, &rhs.z3),
        }
    }
}

impl Add for Expression {
    type Output = Expression;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(CallExpr {
                    func_name: ADD.to_owned(),
                    target: None,
                    args: vec![self.cel, rhs.cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: self.z3.add(rhs.z3),
        }
    }
}

impl Sub for Expression {
    type Output = Expression;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(CallExpr {
                    func_name: SUBSTRACT.to_owned(),
                    target: None,
                    args: vec![self.cel, rhs.cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: self.z3.sub(rhs.z3),
        }
    }
}

impl Mul for Expression {
    type Output = Expression;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(CallExpr {
                    func_name: MULTIPLY.to_owned(),
                    target: None,
                    args: vec![self.cel, rhs.cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: self.z3.mul(rhs.z3),
        }
    }
}

impl Div for Expression {
    type Output = Expression;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            cel: cel::parser::Expression {
                id: 0,
                expr: cel::common::ast::Expr::Call(CallExpr {
                    func_name: DIVIDE.to_owned(),
                    target: None,
                    args: vec![self.cel, rhs.cel],
                }),
            },
            #[cfg(feature = "z3")]
            z3: self.z3.div(rhs.z3),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Shape {
    shape: Vec<Expression>,
}

impl Shape {
    pub fn new(shape: &[Expression]) -> Result<Self> {
        if shape.len() >= MAX_SHAPE_DIMENSION {
            return Err(Error::TooManyDimensionsForShape { dim: shape.len() });
        }
        Ok(Self {
            shape: shape.into(),
        })
    }
    pub fn from_vec(shape: Vec<Expression>) -> Self {
        Self { shape }
    }

    pub fn new_const(shape: &[i64]) -> Result<Self> {
        for &s in shape.iter() {
            if s < 0 {
                return Err(crate::Error::NegativeShapeDimension { value: s });
            }
        }

        Ok(Self {
            shape: shape.iter().copied().map(Expression::new_const).collect(),
        })
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Expression> {
        self.shape.iter()
    }

    pub fn rank(&self) -> u32 {
        self.shape.len() as u32
    }

    pub fn rank_usize(&self) -> usize {
        self.shape.len()
    }
}

//#[derive(Clone, Debug)]
//pub struct ExpressionTensor {
//expressions: Vec<Expression>,
//shape: Vec<u8>,
//}

//impl ExpressionTensor {
//pub fn new(expressions: &[Expression], shape: &[u8]) -> Result<Self> {
//if expressions.len() != shape.iter().map(|i| *i as usize).product::<usize>().min(1) {
//return Err(Error::InvalidExpressionTensorShape {
//expressions: Box::new(expressions.into()),
//shape: Box::new(shape.into()),
//});
//}
//Ok(Self {
//expressions: expressions.into(),
//shape: shape.into(),
//})
//}
//}

struct MyExpression<'a>(&'a cel::parser::Expression);

impl Display for MyExpression<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0.expr {
            Expr::Unspecified => todo!(),
            Expr::Call(call_expr) => match call_expr.func_name.as_str() {
                ADD => {
                    write!(
                        f,
                        "({} + {})",
                        MyExpression(&call_expr.args[0]),
                        MyExpression(&call_expr.args[1])
                    )?;
                }
                SUBSTRACT => {
                    write!(
                        f,
                        "({} - {})",
                        MyExpression(&call_expr.args[0]),
                        MyExpression(&call_expr.args[1])
                    )?;
                }
                MULTIPLY => {
                    write!(
                        f,
                        "({} * {})",
                        MyExpression(&call_expr.args[0]),
                        MyExpression(&call_expr.args[1])
                    )?;
                }
                DIVIDE => {
                    write!(
                        f,
                        "({} / {})",
                        MyExpression(&call_expr.args[0]),
                        MyExpression(&call_expr.args[1])
                    )?;
                }
                call => {
                    write!(f, "{}", call)?;
                    write!(f, "(")?;
                    let mut first = true;
                    for expr in call_expr.args.iter() {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", MyExpression(expr))?;
                    }
                    write!(f, ")")?;
                }
            },
            Expr::Comprehension(_comprehension_expr) => todo!(),
            Expr::Ident(ident) => write!(f, "{}", ident)?,
            Expr::List(list_expr) => {
                write!(f, "[")?;
                let mut first = true;
                for expr in list_expr.elements.iter() {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", MyExpression(expr))?;
                }
                write!(f, "]")?;
            }
            Expr::Literal(literal_value) => match literal_value {
                cel::common::ast::LiteralValue::Boolean(bool1) => {
                    f.write_fmt(format_args!("{}", bool1.inner()))?;
                }
                cel::common::ast::LiteralValue::Bytes(_bytes) => todo!(),
                cel::common::ast::LiteralValue::Double(double) => {
                    write!(f, "{}", double.inner())?;
                }
                cel::common::ast::LiteralValue::Int(int) => {
                    write!(f, "{}", int.inner())?;
                }
                cel::common::ast::LiteralValue::Null => {
                    f.write_str("null")?;
                }
                cel::common::ast::LiteralValue::String(s) => {
                    write!(f, "{:?}", s.inner())?;
                }
                cel::common::ast::LiteralValue::UInt(uint) => {
                    write!(f, "{}", uint.inner())?;
                }
            },
            Expr::Map(_map_expr) => todo!(),
            Expr::Select(_select_expr) => todo!(),
            Expr::Struct(_struct_expr) => todo!(),
        }
        Ok(())
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", MyExpression(&self.cel))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cel::{Context, Value};
    #[cfg(feature = "z3")]
    use z3::Solver;
    #[cfg(feature = "z3")]
    use z3::ast::{Ast, Int};

    use crate::{Error, Expression, Result};

    #[cfg(feature = "z3")]
    #[test]
    fn z3_string_simplifies_before_display() {
        let expression = Expression::new_const(2) * Expression::new_const(1);
        assert_eq!(expression.z3_string(), "2");

        let expression = Expression::new_dynamic("batch") * Expression::new_const(2);
        assert_eq!(expression.z3_string(), "(2 * batch)");

        let expression = Expression::new_dynamic("a").max(Expression::new_dynamic("b"));
        let text = expression.z3_string();
        assert!(text.contains(" ? "), "{text}");
        assert!(!text.contains("(ite"), "{text}");
    }

    #[test]
    fn fallible_expression_methods_use_crate_error() {
        let parse: Result<Expression> = Expression::parse("(");
        assert!(matches!(parse, Err(Error::ParseExpression { .. })));

        let scaled: Result<Expression> = Expression::new_dynamic("n").round_scaled(0.0);
        assert!(matches!(scaled, Err(Error::InvalidScale { .. })));

        let bound: Result<u32> = Expression::new_dynamic("n").upper_bound(&Default::default());
        assert!(matches!(bound, Err(Error::MissingUpperBound { name }) if name == "n"));
    }

    #[test]
    fn check_cel_ast() {
        dbg!(cel::parser::Parser::default().parse("1+2").unwrap());
        dbg!(cel::parser::Parser::default().parse("max(1,2)").unwrap());
    }

    #[test]
    fn check_add() {
        let expr =
            Expression::new_const(1) + Expression::new_const(2) - Expression::new_dynamic("foo");
        let mut context = Context::default();
        context.add_variable("foo", 1).unwrap();
        assert_eq!(Value::resolve(&expr.cel, &context).unwrap(), 2.into());

        #[cfg(feature = "z3")]
        let foo = z3::ast::Int::new_const("foo");
        #[cfg(feature = "z3")]
        let solver = Solver::new();
        #[cfg(feature = "z3")]
        solver.assert(foo.eq(1));
        #[cfg(feature = "z3")]
        assert_eq!(
            solver.check_and_get_model(&[&expr.z3], true).unwrap()[0]
                .as_i64()
                .unwrap(),
            2
        );
        #[cfg(feature = "z3")]
        assert_eq!(format!("{}", expr), "((1 + 2) - foo)");
        #[cfg(feature = "z3")]
        assert_eq!(
            format!(
                "{}",
                expr.z3.substitute(&[(&foo, &Int::from_i64(1))]).simplify()
            ),
            "2"
        );
    }
}
