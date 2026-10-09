//! Materialized MIL boundaries that cannot be removed by CoreML optimization.

use super::*;
use crate::graph::OperandDescriptor;
use crate::protos::coreml::mil_spec::{DataType as MilDataType, value_type};
use crate::protos::coreml::specification::{FeatureDescription, ModelDescription, Pipeline, model};
use std::collections::{BTreeSet, HashSet};

fn tensor(value: &NamedValueType) -> Option<&TensorType> {
    match &value.r#type.as_ref()?.r#type {
        Some(value_type::Type::TensorType(tensor)) => Some(tensor),
        _ => None,
    }
}

fn inputs(operation: &MilOperation) -> HashSet<String> {
    let mut names = HashSet::new();
    for argument in operation.inputs.values() {
        for binding in &argument.arguments {
            if let Some(Binding::Name(name)) = &binding.binding {
                names.insert(name.clone());
            }
        }
    }
    for block in &operation.blocks {
        let local: HashSet<_> = block
            .inputs
            .iter()
            .map(|value| value.name.clone())
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| operation.outputs.iter().map(|value| value.name.clone())),
            )
            .collect();
        for operation in &block.operations {
            names.extend(
                inputs(operation)
                    .into_iter()
                    .filter(|name| !local.contains(name)),
            );
        }
        names.extend(
            block
                .outputs
                .iter()
                .filter(|name| !local.contains(*name))
                .cloned(),
        );
    }
    names
}

fn named_input<'a>(operation: &'a MilOperation, key: &str) -> Option<&'a str> {
    match &operation.inputs.get(key)?.arguments.first()?.binding {
        Some(Binding::Name(name)) => Some(name),
        _ => None,
    }
}

fn integer_argument<'a>(operation: &'a MilOperation, key: &str) -> Option<&'a [i32]> {
    use crate::protos::coreml::mil_spec::{tensor_value, value};
    let Some(Binding::Value(value)) = &operation.inputs.get(key)?.arguments.first()?.binding else {
        return None;
    };
    let Some(value::Value::ImmediateValue(value)) = &value.value else {
        return None;
    };
    let Some(value::immediate_value::Value::Tensor(tensor)) = &value.value else {
        return None;
    };
    match &tensor.value {
        Some(tensor_value::Value::Ints(ints)) => Some(&ints.values),
        _ => None,
    }
}

fn constant(operation: &MilOperation) -> bool {
    operation.r#type == "const" || operation.r#type.starts_with("constexpr_")
}

fn constant_closure(operations: &[MilOperation]) -> HashSet<usize> {
    use crate::protos::coreml::mil_spec::{tensor_value, value};
    let exact_unit_transport = |operation: &MilOperation| {
        if !matches!(operation.r#type.as_str(), "mul" | "real_div")
            || !operation.outputs.iter().all(|output| {
                tensor(output).is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
            })
        {
            return false;
        }
        let Some(Binding::Value(value)) = operation
            .inputs
            .get("y")
            .and_then(|argument| argument.arguments.first())
            .and_then(|binding| binding.binding.as_ref())
        else {
            return false;
        };
        let Some(value::Value::ImmediateValue(value)) = &value.value else {
            return false;
        };
        let Some(value::immediate_value::Value::Tensor(tensor)) = &value.value else {
            return false;
        };
        matches!(&tensor.value, Some(tensor_value::Value::Floats(floats)) if floats.values.as_slice() == [1.0])
    };
    let mut values = HashSet::new();
    let mut represented_half = HashSet::new();
    let mut closure = HashSet::new();
    for (index, operation) in operations.iter().enumerate() {
        let source_is_half =
            named_input(operation, "x").is_some_and(|name| represented_half.contains(name));
        let true_narrowing = operation.r#type == "cast"
            && operation.outputs.iter().any(|output| {
                tensor(output).is_some_and(|value| value.data_type == MilDataType::Float16 as i32)
            })
            && !source_is_half;
        if constant(operation)
            || (operation.blocks.is_empty()
                && (matches!(
                    operation.r#type.as_str(),
                    "cast" | "identity" | "reshape" | "transpose"
                ) || exact_unit_transport(operation))
                && !true_narrowing
                && inputs(operation).iter().all(|name| values.contains(name)))
        {
            closure.insert(index);
            values.extend(operation.outputs.iter().map(|value| value.name.clone()));
            represented_half.extend(
                operation
                    .outputs
                    .iter()
                    .filter(|output| {
                        source_is_half
                            || tensor(output)
                                .is_some_and(|value| value.data_type == MilDataType::Float16 as i32)
                    })
                    .map(|output| output.name.clone()),
            );
        }
    }
    closure
}

#[derive(Default)]
struct SelectPreparation {
    // The condition's existing casts are part of the protected FP32 child.
    prefixes: HashMap<String, String>,
    // Computed conditions and true Half-to-integer casts retain their producer
    // in a separate child, outside a Half-widening child.
    isolated: Vec<(String, String)>,
}

fn prepare_select_conditions(
    function_inputs: &[NamedValueType],
    block: &mut Block,
    promoted: &HashSet<String>,
) -> SelectPreparation {
    let producers: HashMap<_, _> = block
        .operations
        .iter()
        .enumerate()
        .flat_map(|(index, operation)| {
            operation
                .outputs
                .iter()
                .map(move |value| (value.name.clone(), index))
        })
        .collect();
    let types: HashMap<_, _> = function_inputs
        .iter()
        .chain(
            block
                .operations
                .iter()
                .flat_map(|operation| &operation.outputs),
        )
        .map(|value| (value.name.clone(), value.clone()))
        .collect();
    let mut reserved: HashSet<_> = types.keys().cloned().collect();
    let mut candidates = HashSet::new();
    let mut insertions = HashMap::new();
    let mut conditions = HashMap::new();
    let mut result = SelectPreparation::default();
    for (index, operation) in block.operations.iter().enumerate() {
        if operation.r#type != "select"
            || !operation
                .outputs
                .iter()
                .any(|value| promoted.contains(&value.name))
        {
            continue;
        }
        let Some(mut source) = named_input(operation, "cond") else {
            continue;
        };
        let mut prelude = Vec::new();
        while let Some(&producer) = producers.get(source) {
            let cast = &block.operations[producer];
            if cast.r#type != "cast"
                || cast.outputs.len() != 1
                || !tensor(&cast.outputs[0]).is_some_and(|value| {
                    [
                        MilDataType::Bool,
                        MilDataType::Uint8,
                        MilDataType::Int8,
                        MilDataType::Int32,
                    ]
                    .iter()
                    .any(|dtype| value.data_type == *dtype as i32)
                })
            {
                break;
            }
            let Some(input) = named_input(cast, "x") else {
                break;
            };
            prelude.push(producer);
            source = input;
        }
        if let Some(&producer) = producers.get(source)
            && !constant(&block.operations[producer])
            && types.get(source).and_then(tensor).is_some_and(|value| {
                [
                    MilDataType::Bool,
                    MilDataType::Uint8,
                    MilDataType::Int8,
                    MilDataType::Int32,
                ]
                .iter()
                .any(|dtype| value.data_type == *dtype as i32)
            })
        {
            let name = block.operations[producer].outputs[0].name.clone();
            result.isolated.push((name.clone(), name));
        }
        if prelude.is_empty() {
            continue;
        }
        prelude.reverse();
        if types
            .get(source)
            .and_then(tensor)
            .is_some_and(|value| value.data_type == MilDataType::Float16 as i32)
        {
            // A real Half-to-integer conversion retains its original schedule.
            for segment in prelude.chunk_by(|a, b| *b == *a + 1) {
                result.isolated.push((
                    block.operations[segment[0]].outputs[0].name.clone(),
                    block.operations[*segment.last().unwrap()].outputs[0]
                        .name
                        .clone(),
                ));
            }
            continue;
        }
        // Integer/Boolean condition casts are cheap and pure. Recreate their
        // exact chain in each protected kernel instead of exposing shared
        // narrow integers through native Pipeline features. This avoids
        // buggy cast/feature adapters on older CoreML stacks without changing
        // arbitrary integer values to floating-point proxies.
        let mut replacements: HashMap<String, String> = HashMap::new();
        let mut local = Vec::new();
        for &producer in &prelude {
            let mut cast = block.operations[producer].clone();
            for argument in cast.inputs.values_mut() {
                for binding in &mut argument.arguments {
                    if let Some(Binding::Name(name)) = &mut binding.binding
                        && let Some(replacement) = replacements.get(name)
                    {
                        *name = replacement.clone();
                    }
                }
            }
            let original = cast.outputs[0].name.clone();
            let base = format!("{original}_condition_{index}");
            let mut name = base.clone();
            let mut suffix = 0;
            while !reserved.insert(name.clone()) {
                suffix += 1;
                name = format!("{base}_{suffix}");
            }
            cast.outputs[0].name = name.clone();
            replacements.insert(original, name);
            local.push(cast);
            candidates.insert(producer);
        }
        result.prefixes.insert(
            operation.outputs[0].name.clone(),
            local[0].outputs[0].name.clone(),
        );
        conditions.insert(index, local.last().unwrap().outputs[0].name.clone());
        insertions.insert(index, local);
    }
    let original = std::mem::take(&mut block.operations);
    let candidate_names: HashSet<_> = candidates
        .iter()
        .map(|&index| original[index].outputs[0].name.clone())
        .collect();
    for (index, mut operation) in original.into_iter().enumerate() {
        if let Some(prelude) = insertions.remove(&index) {
            block.operations.extend(prelude);
            let argument = operation.inputs.get_mut("cond").unwrap();
            argument.arguments[0].binding = Some(Binding::Name(conditions.remove(&index).unwrap()));
        }
        block.operations.push(operation);
    }
    // Remove original condition casts only after all consumers have been
    // redirected. Shared casts still used elsewhere remain unchanged.
    loop {
        let used: HashSet<_> = block
            .operations
            .iter()
            .flat_map(inputs)
            .chain(block.outputs.iter().cloned())
            .collect();
        let before = block.operations.len();
        block.operations.retain(|operation| {
            operation.outputs.is_empty()
                || !operation.outputs.iter().all(|output| {
                    candidate_names.contains(&output.name) && !used.contains(&output.name)
                })
        });
        if before == block.operations.len() {
            break;
        }
    }
    result
}
fn failure(reason: impl Into<String>) -> GraphError {
    GraphError::ConversionFailed {
        format: "coreml_mlprogram".into(),
        reason: reason.into(),
    }
}

fn raw_shape(value: &NamedValueType) -> Option<Vec<GraphDimension>> {
    tensor(value)?
        .dimensions
        .iter()
        .map(|dimension| match &dimension.dimension {
            Some(dimension::Dimension::Constant(dimension)) => u32::try_from(dimension.size)
                .ok()
                .map(GraphDimension::Static),
            _ => None,
        })
        .collect()
}

fn argument_shape(
    operation: &MilOperation,
    key: &str,
    shapes: &HashMap<String, Vec<GraphDimension>>,
) -> Option<Vec<GraphDimension>> {
    match &operation.inputs.get(key)?.arguments.first()?.binding {
        Some(Binding::Name(name)) => shapes.get(name).cloned(),
        Some(Binding::Value(value)) => raw_shape(&NamedValueType {
            name: String::new(),
            r#type: value.r#type.clone(),
        }),
        _ => None,
    }
}

fn shape_bounds(
    graph: &LoweringGraph<'_>,
    types: &HashMap<String, NamedValueType>,
    operations: &[MilOperation],
) -> HashMap<String, Vec<GraphDimension>> {
    shape_bounds_seeded(graph, types, operations, &HashMap::new())
}

fn shape_bounds_seeded(
    graph: &LoweringGraph<'_>,
    types: &HashMap<String, NamedValueType>,
    operations: &[MilOperation],
    retained: &HashMap<String, Vec<GraphDimension>>,
) -> HashMap<String, Vec<GraphDimension>> {
    let mut shapes: HashMap<_, _> = types
        .iter()
        .filter_map(|(name, value)| raw_shape(value).map(|shape| (name.clone(), shape)))
        .collect();
    for (index, operand) in graph.operands.iter().enumerate() {
        let name = operand_name(graph, index as u32);
        if let Some(value) = types.get(&name) {
            let mut shape = operand.descriptor.shape.clone();
            if shape.is_empty() && tensor(value).is_some_and(|tensor| tensor.rank == 1) {
                shape.push(GraphDimension::Static(1));
            }
            if tensor(value).is_some_and(|tensor| tensor.dimensions.len() == shape.len()) {
                shapes.insert(name, shape);
            }
        }
    }
    // Packing retains exact source-proven shapes before replacing a Cast with
    // runtime-shape restoration. Seed those bounds before propagating derived
    // arithmetic; adding them afterward leaves generated live values unbounded.
    shapes.extend(retained.clone());
    // Promotions have the same shape on both sides of the cast. Propagate
    // bounds backward too: a generated wide result gets the original graph
    // output's constraints from the final narrowing cast.
    loop {
        let mut changed = false;
        for operation in operations {
            if matches!(
                operation.r#type.as_str(),
                "add"
                    | "sub"
                    | "mul"
                    | "real_div"
                    | "pow"
                    | "maximum"
                    | "minimum"
                    | "equal"
                    | "not_equal"
                    | "greater"
                    | "greater_equal"
                    | "less"
                    | "less_equal"
                    | "logical_and"
                    | "logical_or"
                    | "logical_xor"
            ) {
                if let Some(left) = argument_shape(operation, "x", &shapes)
                    && let Some(right) = argument_shape(operation, "y", &shapes)
                    && let Ok(shape) =
                        crate::shape_inference::broadcast_shapes_dimensions(&left, &right)
                {
                    for output in &operation.outputs {
                        if !shapes.contains_key(&output.name)
                            && tensor(output)
                                .is_some_and(|value| value.dimensions.len() == shape.len())
                        {
                            shapes.insert(output.name.clone(), shape.clone());
                            changed = true;
                        }
                    }
                }
                continue;
            }
            if operation.r#type == "transpose" {
                if let Some(input) = named_input(operation, "x")
                    && let Some(permutation) = integer_argument(operation, "perm")
                {
                    for output in &operation.outputs {
                        if !shapes.contains_key(&output.name)
                            && let Some(shape) = shapes.get(input)
                            && permutation.len() == shape.len()
                        {
                            let permuted: Option<Vec<_>> = permutation
                                .iter()
                                .map(|&axis| {
                                    usize::try_from(axis)
                                        .ok()
                                        .and_then(|axis| shape.get(axis))
                                        .cloned()
                                })
                                .collect();
                            if let Some(shape) = permuted {
                                shapes.insert(output.name.clone(), shape);
                                changed = true;
                            }
                        }
                        if !shapes.contains_key(input)
                            && let Some(shape) = shapes.get(&output.name)
                            && permutation.len() == shape.len()
                        {
                            let mut inverse = vec![None; shape.len()];
                            for (&axis, dimension) in permutation.iter().zip(shape) {
                                if let Ok(axis) = usize::try_from(axis)
                                    && let Some(target) = inverse.get_mut(axis)
                                {
                                    *target = Some(dimension.clone());
                                }
                            }
                            if let Some(shape) = inverse.into_iter().collect::<Option<Vec<_>>>() {
                                shapes.insert(input.into(), shape);
                                changed = true;
                            }
                        }
                    }
                }
                continue;
            }
            let input = match operation.r#type.as_str() {
                "cast" | "identity" | "band_part" => named_input(operation, "x"),
                "fill_like" => named_input(operation, "ref_tensor"),
                _ => None,
            };
            let Some(input) = input else {
                continue;
            };
            for output in &operation.outputs {
                if !shapes.contains_key(&output.name)
                    && let Some(shape) = shapes.get(input).cloned()
                {
                    shapes.insert(output.name.clone(), shape);
                    changed = true;
                }
                if !shapes.contains_key(input)
                    && let Some(shape) = shapes.get(&output.name).cloned()
                {
                    shapes.insert(input.into(), shape);
                    changed = true;
                }
            }
        }
        if !changed {
            return shapes;
        }
    }
}

struct Boundary {
    feature: FeatureDescription,
    value: NamedValueType,
    original: NamedValueType,
    scalar_name: String,
}

#[derive(Default)]
struct PackedWidenings {
    outputs: HashSet<String>,
    shapes: HashMap<String, Vec<GraphDimension>>,
    input_views: Vec<(String, NamedValueType)>,
}

// Materializing a high-rank Half value and widening it in the next child can
// read the native feature's padded layout incorrectly. Keep the logical-rank
// widening inside the producer, and expose only a compact Half vector. Shape
// restoration is FP32-only; dynamic dimensions come from the live source, not
// the graph's upper bounds.
fn pack_half_widenings(
    graph: &LoweringGraph<'_>,
    function_inputs: &[NamedValueType],
    block: &mut Block,
) -> Result<PackedWidenings, GraphError> {
    let types: HashMap<_, _> = function_inputs
        .iter()
        .chain(
            block
                .operations
                .iter()
                .flat_map(|operation| &operation.outputs),
        )
        .map(|value| (value.name.clone(), value.clone()))
        .collect();
    let shapes = shape_bounds(graph, &types, &block.operations);
    let constant_operations = constant_closure(&block.operations);
    let constant_values: HashSet<_> = block
        .operations
        .iter()
        .enumerate()
        .filter(|(index, _)| constant_operations.contains(index))
        .flat_map(|(_, operation)| operation.outputs.iter().map(|value| value.name.clone()))
        .collect();
    let mut reserved: HashSet<_> = types.keys().cloned().collect();
    let mut fresh = |base: String| {
        let mut name = base.clone();
        let mut suffix = 0;
        while !reserved.insert(name.clone()) {
            suffix += 1;
            name = format!("{base}_{suffix}");
        }
        name
    };
    let mut packed = PackedWidenings::default();
    let original_inputs: HashSet<_> = function_inputs
        .iter()
        .map(|value| value.name.as_str())
        .collect();
    let mut input_views = HashMap::<String, NamedValueType>::new();
    for mut operation in std::mem::take(&mut block.operations) {
        let Some(source) = named_input(&operation, "x").map(str::to_owned) else {
            block.operations.push(operation);
            continue;
        };
        if operation.r#type != "cast"
            || operation.outputs.len() != 1
            || constant_values.contains(&source)
            || !types.get(&source).and_then(tensor).is_some_and(|value| {
                value.rank > 2 && value.data_type == MilDataType::Float16 as i32
            })
            || !tensor(&operation.outputs[0])
                .is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
        {
            block.operations.push(operation);
            continue;
        }
        let original = operation.outputs[0].clone();
        let shape = shapes.get(&source).ok_or_else(|| {
            failure(format!(
                "Half widening {source} has no bounded shape provenance"
            ))
        })?;
        // The generated restore has exactly this proven logical source shape.
        // A runtime-shape reshape is not otherwise invertible by shape_bounds;
        // retain its bounds if the restored widening becomes a native feature.
        packed.shapes.insert(original.name.clone(), shape.clone());
        if !original_inputs.contains(source.as_str())
            && !shape.last().is_some_and(|dimension| match dimension {
                GraphDimension::Static(size) => *size == 1,
                // Bounded dimensions can reach one at prediction time.
                GraphDimension::Dynamic(_) => true,
            })
        {
            // Real produced features with a non-singleton innermost axis
            // retain the direct widening. Packing every high-rank result
            // adds native children and substantial per-child runtime memory.
            block.operations.push(operation);
            continue;
        }
        let maximum = shape
            .iter()
            .try_fold(1u32, |product, dimension| {
                product.checked_mul(match dimension {
                    GraphDimension::Static(size) => *size,
                    GraphDimension::Dynamic(dimension) => dimension.max_size,
                })
            })
            .ok_or_else(|| {
                failure(format!(
                    "Half widening {source} exceeds the compact feature size limit"
                ))
            })?;
        let dynamic = shape
            .iter()
            .any(|dimension| matches!(dimension, GraphDimension::Dynamic(_)));
        let flat_shape = vec![if dynamic {
            GraphDimension::Dynamic(crate::graph::DynamicDimension {
                name: fresh(format!("{}_compact_elements", original.name)),
                max_size: maximum,
            })
        } else {
            GraphDimension::Static(maximum)
        }];
        let wide = CoremlMlProgramConverter::create_named_value_type(
            fresh(format!("{}_compact_wide", original.name)),
            MilDataType::Float32 as i32,
            &flat_shape,
            true,
        );
        packed.shapes.insert(wide.name.clone(), flat_shape.clone());
        if original_inputs.contains(source.as_str()) {
            // Bind a flat view over the caller's original Half storage. Do not
            // first enter a logical-rank Half kernel: that ingress itself can
            // canonicalize zeros or read a padded layout incorrectly.
            let view = input_views.entry(source.clone()).or_insert_with(|| {
                let view = CoremlMlProgramConverter::create_named_value_type(
                    fresh(format!("{source}_compact_input")),
                    MilDataType::Float16 as i32,
                    &flat_shape,
                    true,
                );
                packed.input_views.push((source.clone(), view.clone()));
                packed.shapes.insert(view.name.clone(), flat_shape.clone());
                view
            });
            block
                .operations
                .push(CoremlMlProgramConverter::create_cast_operation(
                    view.name.clone(),
                    wide.clone(),
                    "fp32",
                ));
            // Keep widening separate from restoration for both static and
            // dynamic source shapes. The original feature remains available
            // for other consumers and for its actual runtime shape.
            packed.outputs.insert(wide.name.clone());
        } else {
            let logical = fresh(format!("{}_packing_wide", original.name));
            operation.outputs[0].name = logical.clone();
            block.operations.push(operation);
            let flat = CoremlMlProgramConverter::create_named_value_type(
                fresh(format!("{}_packing_flat", original.name)),
                MilDataType::Float32 as i32,
                &flat_shape,
                true,
            );
            block
                .operations
                .push(Boundary::reshape(logical, flat.clone(), &[-1]));
            let half = CoremlMlProgramConverter::create_named_value_type(
                fresh(format!("{}_compact_half", original.name)),
                MilDataType::Float16 as i32,
                &flat_shape,
                true,
            );
            block
                .operations
                .push(CoremlMlProgramConverter::create_cast_operation(
                    flat.name,
                    half.clone(),
                    "fp16",
                ));
            packed.outputs.insert(half.name.clone());
            packed.shapes.insert(half.name.clone(), flat_shape);
            block
                .operations
                .push(CoremlMlProgramConverter::create_cast_operation(
                    half.name,
                    wide.clone(),
                    "fp32",
                ));
        }
        if dynamic {
            // Dynamic restoration can fold the widening back into a Half
            // transport kernel. Its FP32 endpoint must be real before the
            // shape-dependent reshape is evaluated by the next child.
            packed.outputs.insert(wide.name.clone());
        }
        if dynamic {
            let runtime_shape = CoremlMlProgramConverter::create_named_value_type(
                fresh(format!("{}_logical_shape", original.name)),
                MilDataType::Int32 as i32,
                &[GraphDimension::Static(shape.len() as u32)],
                true,
            );
            block
                .operations
                .push(CoremlMlProgramConverter::create_mil_operation(
                    "shape",
                    HashMap::from([(
                        "x".into(),
                        CoremlMlProgramConverter::create_name_argument(source),
                    )]),
                    vec![runtime_shape.clone()],
                ));
            block
                .operations
                .push(CoremlMlProgramConverter::create_mil_operation(
                    "reshape",
                    HashMap::from([
                        (
                            "x".into(),
                            CoremlMlProgramConverter::create_name_argument(wide.name),
                        ),
                        (
                            "shape".into(),
                            CoremlMlProgramConverter::create_name_argument(runtime_shape.name),
                        ),
                    ]),
                    vec![original],
                ));
        } else {
            let restore: Result<Vec<_>, _> = shape
                .iter()
                .map(|dimension| {
                    let GraphDimension::Static(size) = dimension else {
                        unreachable!()
                    };
                    i32::try_from(*size)
                        .map_err(|_| failure("Half widening dimension exceeds MIL reshape limits"))
                })
                .collect();
            block
                .operations
                .push(Boundary::reshape(wide.name, original, &restore?));
        }
    }
    Ok(packed)
}

impl Boundary {
    fn new(
        value: &NamedValueType,
        shapes: &HashMap<String, Vec<GraphDimension>>,
        reserved: &mut HashSet<String>,
    ) -> Result<Self, GraphError> {
        let tensor = tensor(value).ok_or_else(|| failure("Non-tensor precision boundary"))?;
        let shape = shapes.get(&value.name).ok_or_else(|| {
            failure(format!(
                "Precision boundary {} has an unknown dimension without bounded shape provenance",
                value.name
            ))
        })?;
        let (data_type, mil_type) = match tensor.data_type {
            value if value == MilDataType::Float16 as i32 => (DataType::Float16, value),
            value if value == MilDataType::Float32 as i32 => (DataType::Float32, value),
            value if value == MilDataType::Int32 as i32 => (DataType::Int32, value),
            value
                if value == MilDataType::Bool as i32
                    || value == MilDataType::Int8 as i32
                    || value == MilDataType::Uint8 as i32 =>
            {
                (DataType::Int32, MilDataType::Int32 as i32)
            }
            dtype => {
                return Err(failure(format!(
                    "Unsupported precision-boundary dtype {dtype}"
                )));
            }
        };
        let adapted = tensor.data_type != mil_type || tensor.rank == 0;
        let mut name = value.name.clone();
        if adapted {
            let base = format!("{name}_precision_io");
            name = base.clone();
            let mut suffix = 0;
            while !reserved.insert(name.clone()) {
                suffix += 1;
                name = format!("{base}_{suffix}");
            }
        }
        let descriptor = OperandDescriptor {
            data_type,
            shape: shape.clone(),
            pending_permutation: vec![],
        };
        let scalar_base = format!("{name}_scalar");
        let mut scalar_name = scalar_base.clone();
        let mut suffix = 0;
        while !reserved.insert(scalar_name.clone()) {
            suffix += 1;
            scalar_name = format!("{scalar_base}_{suffix}");
        }
        Ok(Self {
            feature: FeatureDescription {
                name: name.clone(),
                r#type: Some(CoremlMlProgramConverter::create_feature_type(&descriptor)?),
                ..Default::default()
            },
            value: CoremlMlProgramConverter::create_named_value_type(name, mil_type, shape, true),
            original: value.clone(),
            scalar_name,
        })
    }

    fn cast_dtype(value: &NamedValueType) -> Result<&'static str, GraphError> {
        match tensor(value).unwrap().data_type {
            value if value == MilDataType::Bool as i32 => Ok("bool"),
            value if value == MilDataType::Int8 as i32 => Ok("int8"),
            value if value == MilDataType::Uint8 as i32 => Ok("uint8"),
            value => CoremlMlProgramConverter::cast_dtype_string_for_mil_type(value),
        }
    }

    fn reshape(input: String, output: NamedValueType, shape: &[i32]) -> MilOperation {
        CoremlMlProgramConverter::create_mil_operation(
            "reshape",
            [
                (
                    "x".into(),
                    CoremlMlProgramConverter::create_name_argument(input),
                ),
                (
                    "shape".into(),
                    CoremlMlProgramConverter::create_int_array_argument(shape.to_vec()),
                ),
            ]
            .into_iter()
            .collect(),
            vec![output],
        )
    }

    fn scalar_type(&self) -> NamedValueType {
        CoremlMlProgramConverter::create_named_value_type(
            self.scalar_name.clone(),
            tensor(&self.value).unwrap().data_type,
            &[],
            false,
        )
    }

    fn input_cast(&self) -> Result<Vec<MilOperation>, GraphError> {
        if self.value.name == self.original.name {
            return Ok(vec![]);
        }
        let mut operations = Vec::new();
        let mut input = self.value.name.clone();
        if tensor(&self.original).unwrap().rank == 0 {
            if tensor(&self.original).unwrap().data_type == tensor(&self.value).unwrap().data_type {
                return Ok(vec![Self::reshape(input, self.original.clone(), &[])]);
            }
            operations.push(Self::reshape(input, self.scalar_type(), &[]));
            input = self.scalar_name.clone();
        }
        operations.push(CoremlMlProgramConverter::create_cast_operation(
            input,
            self.original.clone(),
            Self::cast_dtype(&self.original)?,
        ));
        Ok(operations)
    }

    fn output_cast(&self) -> Result<Vec<MilOperation>, GraphError> {
        if self.value.name == self.original.name {
            return Ok(vec![]);
        }
        let mut operations = Vec::new();
        if tensor(&self.original).unwrap().rank == 0 {
            let mut input = self.original.name.clone();
            if tensor(&self.original).unwrap().data_type != tensor(&self.value).unwrap().data_type {
                operations.push(CoremlMlProgramConverter::create_cast_operation(
                    input,
                    self.scalar_type(),
                    Self::cast_dtype(&self.value)?,
                ));
                input = self.scalar_name.clone();
            }
            operations.push(Self::reshape(input, self.value.clone(), &[1]));
        } else {
            operations.push(CoremlMlProgramConverter::create_cast_operation(
                self.original.name.clone(),
                self.value.clone(),
                Self::cast_dtype(&self.value)?,
            ));
        }
        Ok(operations)
    }
}

impl CoremlMlProgramConverter {
    fn materialize_public_half_constants(function_inputs: &[NamedValueType], block: &mut Block) {
        let outputs: HashSet<_> = block.outputs.iter().cloned().collect();
        let mut reserved: HashSet<_> = function_inputs
            .iter()
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| &operation.outputs),
            )
            .map(|value| value.name.clone())
            .collect();
        let mut fresh = |base: String| {
            let mut name = base.clone();
            let mut suffix = 0;
            while !reserved.insert(name.clone()) {
                suffix += 1;
                name = format!("{base}_{suffix}");
            }
            name
        };
        for (index, mut operation) in std::mem::take(&mut block.operations)
            .into_iter()
            .enumerate()
        {
            if operation.r#type != "const"
                || operation.outputs.len() != 1
                || !outputs.contains(&operation.outputs[0].name)
                || !tensor(&operation.outputs[0])
                    .is_some_and(|value| value.data_type == MilDataType::Float16 as i32)
            {
                block.operations.push(operation);
                continue;
            }
            let original = operation.outputs[0].clone();
            let stored = fresh(format!("{}_constant_storage_{index}", original.name));
            operation.outputs[0].name = stored.clone();
            block.operations.push(operation);
            let mut wide = original.clone();
            wide.name = fresh(format!("{}_constant_wide_{index}", original.name));
            let Some(value_type::Type::TensorType(value)) =
                &mut wide.r#type.as_mut().unwrap().r#type
            else {
                unreachable!("tensor")
            };
            value.data_type = MilDataType::Float32 as i32;
            block
                .operations
                .push(Self::create_cast_operation(stored, wide.clone(), "fp32"));
            let mut product = wide.clone();
            product.name = fresh(format!("{}_constant_materialized_{index}", original.name));
            block.operations.push(Self::create_mil_operation(
                "real_div",
                [
                    ("x".into(), Self::create_name_argument(wide.name)),
                    ("y".into(), Self::create_immediate_float(1.0)),
                ]
                .into_iter()
                .collect(),
                vec![product.clone()],
            ));
            block
                .operations
                .push(Self::create_cast_operation(product.name, original, "fp16"));
        }
    }

    pub(super) fn fold_constant_half_casts(
        graph: &LoweringGraph<'_>,
        model: &mut Model,
        weights: &mut super::super::WeightFileBuilder,
    ) -> Result<(), GraphError> {
        let Some(model::Type::MlProgram(program)) = &mut model.r#type else {
            return Ok(());
        };
        let function = program.functions.get_mut("main").unwrap();
        let block = function
            .block_specializations
            .get_mut(&function.opset)
            .unwrap();
        let mut roots: HashMap<_, _> = graph
            .constant_operand_ids_to_handles
            .keys()
            .map(|&id| (id, id))
            .collect();
        let mut rounded = HashMap::<u32, MilOperation>::new();
        for operation in &graph.operations {
            let (input, outputs, narrowing) = match operation {
                Operation::Cast {
                    input,
                    data_type,
                    outputs,
                    ..
                } if matches!(
                    data_type,
                    MLOperandDataType::Float16 | MLOperandDataType::Float32
                ) =>
                {
                    (*input, outputs, *data_type == MLOperandDataType::Float16)
                }
                Operation::Identity { input, outputs, .. }
                | Operation::Reshape { input, outputs, .. } => (*input, outputs, false),
                _ => continue,
            };
            let Some(&root) = roots.get(&input) else {
                continue;
            };
            let source = &graph.operands[root as usize];
            if !matches!(
                source.descriptor.data_type,
                DataType::Float16 | DataType::Float32
            ) {
                continue;
            }
            for &output in outputs {
                roots.insert(output, root);
                if !narrowing
                    || graph.operands[input as usize].descriptor.data_type == DataType::Float16
                {
                    continue;
                }
                let name = operand_name(graph, output);
                let Some(index) = block.operations.iter().position(|operation| {
                    operation.r#type == "cast"
                        && operation.outputs.iter().any(|value| value.name == name)
                }) else {
                    continue;
                };
                let mut folded = if let Some(folded) = rounded.get(&root) {
                    folded.clone()
                } else if source.descriptor.data_type == DataType::Float16 {
                    // Already represented Half constants reuse the original BLOB.
                    let source_name = operand_name(graph, root);
                    block
                        .operations
                        .iter()
                        .find(|operation| {
                            constant(operation)
                                && operation
                                    .outputs
                                    .iter()
                                    .any(|value| value.name == source_name)
                        })
                        .unwrap()
                        .clone()
                } else {
                    let data = &graph.constant_operand_ids_to_handles[&root].data;
                    let constant = crate::graph::ConstantData {
                        data: data
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|&bytes| {
                                half::f16::from_f32(f32::from_le_bytes(bytes))
                                    .to_bits()
                                    .to_le_bytes()
                            })
                            .collect(),
                        label: None,
                    };
                    Self::create_const_operation(
                        graph,
                        output,
                        &graph.operands[output as usize],
                        &constant,
                        weights,
                    )?
                };
                folded.outputs = block.operations[index].outputs.clone();
                if let Some(value) = folded.attributes.get_mut("val") {
                    value.r#type = folded.outputs[0].r#type.clone();
                }
                rounded.entry(root).or_insert_with(|| folded.clone());
                block.operations[index] = folded;
            }
        }
        Self::materialize_public_half_constants(&function.inputs, block);
        Ok(())
    }

    pub(super) fn materialize_precision_boundaries(
        &self,
        graph: &LoweringGraph<'_>,
        mut model: Model,
        promoted: &HashSet<String>,
    ) -> Result<Model, GraphError> {
        let Some(model::Type::MlProgram(program)) = &mut model.r#type else {
            return Ok(model);
        };
        let (preparation, packed, protected_widenings) = {
            let function = program.functions.get_mut("main").unwrap();
            let block = function
                .block_specializations
                .get_mut(&function.opset)
                .unwrap();
            let preparation = prepare_select_conditions(&function.inputs, block, promoted);
            let protected_inputs: HashSet<_> = graph
                .operations
                .iter()
                .filter(|operation| {
                    matches!(
                        operation,
                        Operation::Reciprocal { .. }
                            | Operation::Neg { .. }
                            | Operation::RoundEven { .. }
                            | Operation::Sub { .. }
                            | Operation::Prelu { .. }
                            | Operation::Gemm { .. }
                            | Operation::Triangular { .. }
                            | Operation::IsNaN { .. }
                            | Operation::IsInfinite { .. }
                    )
                })
                .flat_map(|operation| operation.all_input_operands())
                .filter(|&id| {
                    graph
                        .operand(id)
                        .is_some_and(|operand| operand.descriptor.data_type == DataType::Float16)
                })
                .map(|id| operand_name(graph, id))
                .collect();
            let protected_widenings: HashSet<_> = block
                .operations
                .iter()
                .filter(|operation| {
                    operation.r#type == "cast"
                        && named_input(operation, "x")
                            .is_some_and(|name| protected_inputs.contains(name))
                })
                .flat_map(|operation| &operation.outputs)
                .filter(|output| {
                    tensor(output)
                        .is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
                })
                .map(|output| output.name.clone())
                .collect();
            let packed = pack_half_widenings(graph, &function.inputs, block)?;
            function
                .inputs
                .extend(packed.input_views.iter().map(|(_, view)| view.clone()));
            (preparation, packed, protected_widenings)
        };
        if !packed.input_views.is_empty() {
            let description = model.description.as_mut().unwrap();
            let metadata = description.metadata.get_or_insert_with(Default::default);
            metadata.user_defined.insert(
                "rustnn.webnn.compact_input_views".into(),
                serde_json::to_string(
                    &packed
                        .input_views
                        .iter()
                        .map(|(source, view)| serde_json::json!({"source":source,"view":view.name}))
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| {
                    failure(format!("Could not serialize compact input views: {error}"))
                })?,
            );
            for (_, view) in &packed.input_views {
                let shape = &packed.shapes[&view.name];
                description.input.push(FeatureDescription {
                    name: view.name.clone(),
                    r#type: Some(Self::create_feature_type(&OperandDescriptor {
                        data_type: DataType::Float16,
                        shape: shape.clone(),
                        pending_permutation: vec![],
                    })?),
                    ..Default::default()
                });
            }
        }
        let function = &program.functions["main"];
        let block = &function.block_specializations[&function.opset];
        let types: HashMap<_, _> = function
            .inputs
            .iter()
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| &operation.outputs),
            )
            .map(|value| (value.name.clone(), value.clone()))
            .collect();
        let mut last_use = HashMap::new();
        for (index, operation) in block.operations.iter().enumerate() {
            for input in inputs(operation) {
                if types.contains_key(&input) {
                    last_use.insert(input, index);
                }
            }
        }
        let narrowing: HashSet<_> = graph
            .operations
            .iter()
            .filter_map(|operation| match operation {
                Operation::Cast {
                    input,
                    data_type: MLOperandDataType::Float16,
                    outputs,
                    ..
                } if graph
                    .operand(*input)
                    .is_some_and(|operand| operand.descriptor.data_type != DataType::Float16) =>
                {
                    Some(outputs)
                }
                _ => None,
            })
            .flatten()
            .map(|&id| operand_name(graph, id))
            .collect();
        let source_float_casts: HashSet<_> = graph.operations.iter().filter(|operation| {
            matches!(operation, Operation::Cast { input, outputs, .. } if graph.operands[*input as usize].descriptor.data_type == DataType::Float16 || outputs.iter().any(|&output| graph.operands[output as usize].descriptor.data_type == DataType::Float16))
        }).flat_map(|operation| operation.outputs()).map(|&id| operand_name(graph, id)).collect();
        let half_arithmetic: HashSet<_> = graph
            .operations
            .iter()
            .filter(|operation| {
                !matches!(
                    operation.op_type().to_lowercase().as_str(),
                    "cast"
                        | "identity"
                        | "reshape"
                        | "transpose"
                        | "slice"
                        | "reverse"
                        | "concat"
                        | "split"
                        | "gather"
                        | "gatherelements"
                        | "gathernd"
                        | "tile"
                        | "pad"
                        | "where"
                        | "expand"
                        | "squeeze"
                        | "unsqueeze"
                        | "scatterelements"
                        | "scatternd"
                        | "triangular"
                )
            })
            .flat_map(|operation| operation.outputs())
            .filter(|&&id| {
                graph
                    .operand(id)
                    .is_some_and(|operand| operand.descriptor.data_type == DataType::Float16)
            })
            .map(|&id| operand_name(graph, id))
            .collect();
        let explicit_widening: HashSet<_> = graph
            .operations
            .iter()
            .filter_map(|operation| match operation {
                Operation::Cast {
                    input,
                    data_type: MLOperandDataType::Float32,
                    outputs,
                    ..
                } if graph
                    .operand(*input)
                    .is_some_and(|operand| operand.descriptor.data_type == DataType::Float16) =>
                {
                    Some(outputs)
                }
                _ => None,
            })
            .flatten()
            .map(|&id| operand_name(graph, id))
            .collect();
        let widened: HashSet<_> = block
            .operations
            .iter()
            .filter_map(|operation| {
                if operation.r#type != "cast"
                    || !operation.outputs.iter().any(|output| {
                        tensor(output)
                            .is_some_and(|tensor| tensor.data_type == MilDataType::Float32 as i32)
                    })
                {
                    return None;
                }
                let input = named_input(operation, "x")?;
                types
                    .get(input)
                    .filter(|value| {
                        tensor(value)
                            .is_some_and(|tensor| tensor.data_type == MilDataType::Float16 as i32)
                    })
                    .map(|_| input.to_string())
            })
            .collect();
        let producers: HashMap<_, _> = block
            .operations
            .iter()
            .enumerate()
            .flat_map(|(index, operation)| {
                operation
                    .outputs
                    .iter()
                    .map(move |value| (value.name.clone(), index))
            })
            .collect();
        let constant_operations = constant_closure(&block.operations);
        let mut cuts = BTreeSet::new();
        let graph_outputs: HashSet<_> = block.outputs.iter().cloned().collect();
        let float32_affine_layer_norm_inputs: HashSet<_> = graph
            .operations
            .iter()
            .filter_map(|operation| {
                let Operation::LayerNormalization {
                    input,
                    options: Some(options),
                    ..
                } = operation
                else {
                    return None;
                };
                ((options.scale.is_some() || options.bias.is_some())
                    && graph
                        .operand(*input)
                        .is_some_and(|operand| operand.descriptor.data_type == DataType::Float32))
                .then(|| operand_name(graph, *input))
            })
            .collect();
        let protected_half_results: HashSet<_> = graph
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    Operation::RoundEven { .. }
                        | Operation::Sub { .. }
                        | Operation::Neg { .. }
                        | Operation::Prelu { .. }
                        | Operation::Gemm { .. }
                        | Operation::Triangular { .. }
                )
            })
            .flat_map(|operation| operation.outputs())
            .filter(|&&id| {
                graph
                    .operand(id)
                    .is_some_and(|operand| operand.descriptor.data_type == DataType::Float16)
            })
            .map(|&id| operand_name(graph, id))
            .collect();
        for (first, last) in &preparation.isolated {
            cuts.insert(producers[first]);
            cuts.insert(producers[last] + 1);
        }
        for (index, operation) in block.operations.iter().enumerate() {
            // Keep explicitly lowered LayerNorm affine work outside the
            // native normalization plan. A fused native gamma/beta plan can
            // omit or reorder nontrailing-axis affine parameters.
            if operation.r#type == "layer_norm"
                && named_input(operation, "x")
                    .is_some_and(|name| float32_affine_layer_norm_inputs.contains(name))
                && !operation.inputs.contains_key("gamma")
                && !operation.inputs.contains_key("beta")
                && operation.outputs.iter().any(|output| {
                    tensor(output)
                        .is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
                        && block.operations.iter().skip(index + 1).any(|consumer| {
                            matches!(consumer.r#type.as_str(), "mul" | "add")
                                && inputs(consumer).contains(&output.name)
                        })
                })
            {
                cuts.insert(index + 1);
            }
            // A source FP32 result must exist before an explicit Half cast.
            // Otherwise the compiler can fuse the producer into a Half kernel
            // even though the WebNN operation preceding Cast is FP32-typed.
            if operation.r#type == "cast"
                && operation
                    .outputs
                    .iter()
                    .any(|output| narrowing.contains(&output.name))
                && let Some(source) = named_input(operation, "x")
                && types
                    .get(source)
                    .and_then(tensor)
                    .is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
                && producers
                    .get(source)
                    .is_some_and(|producer| !constant_operations.contains(producer))
            {
                cuts.insert(index);
            }
            if operation.r#type == "cast"
                && operation
                    .outputs
                    .iter()
                    .any(|output| protected_half_results.contains(&output.name))
                && named_input(operation, "x")
                    .and_then(|name| types.get(name))
                    .and_then(tensor)
                    .is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
            {
                // Expose the FP32 math result before a pure terminal Half
                // conversion so the typed output cannot lower the math back
                // into a Half kernel with different zero/tie behavior.
                cuts.insert(index);
            }
            if operation.outputs.iter().any(|output| {
                packed.outputs.contains(&output.name) || protected_widenings.contains(&output.name)
            }) {
                cuts.insert(index + 1);
            }
            if operation.outputs.iter().any(|output| {
                graph_outputs.contains(&output.name) && source_float_casts.contains(&output.name)
            }) {
                cuts.insert(index + 1);
            }
            if !constant_operations.contains(&index)
                && operation.outputs.iter().any(|output| {
                    (narrowing.contains(&output.name)
                        || explicit_widening.contains(&output.name)
                        || (half_arithmetic.contains(&output.name)
                            && widened.contains(&output.name)))
                        && last_use.get(&output.name).is_some_and(|&last| last > index)
                })
            {
                cuts.insert(index + 1);
            }
            // An exposed Half output must not share a native child with the
            // next Half-to-FP32 widening. Mixed output fusion can corrupt the
            // widened values despite leaving the exposed Half value exact.
            if operation.r#type == "cast"
                && named_input(operation, "x")
                    .and_then(|name| types.get(name))
                    .and_then(tensor)
                    .is_some_and(|value| value.data_type == MilDataType::Float32 as i32)
                && operation.outputs.iter().any(|output| {
                    graph_outputs.contains(&output.name)
                        && tensor(output)
                            .is_some_and(|value| value.data_type == MilDataType::Float16 as i32)
                })
            {
                cuts.insert(index + 1);
            }
            let protected = operation.outputs.iter().any(|output| {
                promoted.contains(&output.name)
                    && (matches!(operation.r#type.as_str(), "select" | "pad")
                        || (operation.r#type == "identity" && raw_shape(output).is_none()))
            });
            if protected {
                let start = operation
                    .outputs
                    .iter()
                    .find_map(|value| preparation.prefixes.get(&value.name))
                    .map_or(index, |name| producers[name]);
                cuts.insert(start);
                cuts.insert(index + 1);
            }
        }
        cuts.remove(&0);
        cuts.remove(&block.operations.len());
        if cuts.is_empty() {
            return Ok(model);
        }
        let shapes = shape_bounds_seeded(graph, &types, &block.operations, &packed.shapes);
        let mut reserved: HashSet<_> = types.keys().cloned().collect();
        let original_inputs: HashMap<_, _> = function
            .inputs
            .iter()
            .map(|value| (value.name.clone(), value.clone()))
            .collect();
        let input_features: HashMap<_, _> = model
            .description
            .as_ref()
            .unwrap()
            .input
            .iter()
            .map(|feature| (feature.name.clone(), feature.clone()))
            .collect();
        let mut boundaries = HashMap::new();
        let mut models = Vec::new();
        let mut start = 0;
        for end in cuts
            .into_iter()
            .chain(std::iter::once(block.operations.len()))
        {
            let local_outputs: HashSet<_> = block.operations[start..end]
                .iter()
                .flat_map(|operation| operation.outputs.iter().map(|output| output.name.clone()))
                .collect();
            let outputs: BTreeSet<_> = local_outputs
                .iter()
                .filter(|name| {
                    graph_outputs.contains(*name)
                        || (!producers
                            .get(*name)
                            .is_some_and(|&index| constant_operations.contains(&index))
                            && last_use.get(*name).is_some_and(|&last| last >= end))
                })
                .cloned()
                .collect();
            // Constant-only/dead prefixes need no native stage: their live
            // constant closure is added to each actual consuming child.
            if outputs.is_empty() {
                start = end;
                continue;
            }
            let mut required: HashSet<_> = block.operations[start..end]
                .iter()
                .flat_map(inputs)
                .collect();
            required.extend(outputs.iter().cloned());
            let mut constants = BTreeSet::new();
            loop {
                let needed: Vec<_> = required
                    .iter()
                    .filter_map(|name| {
                        producers
                            .get(name)
                            .filter(|&&index| {
                                constant_operations.contains(&index) && !constants.contains(&index)
                            })
                            .copied()
                    })
                    .collect();
                if needed.is_empty() {
                    break;
                }
                for index in needed {
                    constants.insert(index);
                    required.extend(inputs(&block.operations[index]));
                }
            }
            let inputs: BTreeSet<_> = required
                .into_iter()
                .filter(|name| {
                    !local_outputs.contains(name)
                        && !producers
                            .get(name)
                            .is_some_and(|&index| constants.contains(&index))
                        && types.contains_key(name)
                })
                .collect();
            let mut stage_block = Block::default();
            let mut stage_inputs = Vec::new();
            let mut description = ModelDescription::default();
            for name in &inputs {
                if let Some(value) = original_inputs.get(name) {
                    stage_inputs.push(value.clone());
                    description.input.push(input_features[name].clone());
                } else {
                    if !boundaries.contains_key(name) {
                        boundaries.insert(
                            name.clone(),
                            Boundary::new(&types[name], &shapes, &mut reserved)?,
                        );
                    }
                    let boundary = &boundaries[name];
                    stage_inputs.push(boundary.value.clone());
                    description.input.push(boundary.feature.clone());
                    stage_block.operations.extend(boundary.input_cast()?);
                }
            }
            stage_block.operations.extend(
                constants
                    .iter()
                    .map(|&index| block.operations[index].clone()),
            );
            stage_block.operations.extend(
                block.operations[start..end]
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !constant_operations.contains(&(start + index)))
                    .map(|(_, operation)| operation.clone()),
            );
            for name in outputs {
                if !boundaries.contains_key(&name) {
                    boundaries.insert(
                        name.clone(),
                        Boundary::new(&types[&name], &shapes, &mut reserved)?,
                    );
                }
                let boundary = &boundaries[&name];
                stage_block.operations.extend(boundary.output_cast()?);
                stage_block.outputs.push(boundary.value.name.clone());
                description.output.push(boundary.feature.clone());
            }
            // Constants exposed directly by WebNN are emitted only once. A
            // direct input output stays in the Pipeline's initial feature map.
            if start == 0 {
                for name in &graph_outputs {
                    if !producers.contains_key(name) {
                        continue;
                    }
                    let index = producers[name];
                    if !constant(&block.operations[index])
                        || description
                            .output
                            .iter()
                            .any(|feature| feature.name == *name)
                    {
                        continue;
                    }
                    if !constants.contains(&index) {
                        stage_block
                            .operations
                            .insert(0, block.operations[index].clone());
                    }
                    stage_block.outputs.push(name.clone());
                    description.output.push(
                        model
                            .description
                            .as_ref()
                            .unwrap()
                            .output
                            .iter()
                            .find(|feature| feature.name == *name)
                            .unwrap()
                            .clone(),
                    );
                }
            }
            let mut stage_function = Function {
                inputs: stage_inputs,
                opset: function.opset.clone(),
                attributes: function.attributes.clone(),
                ..Default::default()
            };
            // Rank adapters introduced above are movement operations too.
            // Preserve half transport in them just as in the original lowering.
            Self::preserve_float16_transport(&mut stage_function, &mut stage_block);
            stage_function
                .block_specializations
                .insert(function.opset.clone(), stage_block);
            models.push(Model {
                specification_version: model.specification_version,
                description: Some(description),
                r#type: Some(model::Type::MlProgram(Program {
                    version: program.version,
                    functions: [("main".into(), stage_function)].into_iter().collect(),
                    ..Default::default()
                })),
                ..Default::default()
            });
            start = end;
        }
        model.r#type = Some(model::Type::Pipeline(Pipeline {
            names: (0..models.len())
                .map(|index| format!("stage{index}"))
                .collect(),
            models,
        }));
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_boolean_boundary_restores_rank_and_type() {
        let value = CoremlMlProgramConverter::create_named_value_type(
            "scalar_condition".into(),
            MilDataType::Bool as i32,
            &[],
            false,
        );
        let boundary = Boundary::new(
            &value,
            &[(value.name.clone(), vec![])].into_iter().collect(),
            &mut [value.name.clone()].into_iter().collect(),
        )
        .unwrap();
        assert_eq!(tensor(&boundary.value).unwrap().rank, 1);
        assert_eq!(
            tensor(&boundary.value).unwrap().data_type,
            MilDataType::Int32 as i32
        );
        let inputs = boundary.input_cast().unwrap();
        assert_eq!(
            inputs
                .iter()
                .map(|op| op.r#type.as_str())
                .collect::<Vec<_>>(),
            ["reshape", "cast"]
        );
        assert_eq!(tensor(&inputs[1].outputs[0]).unwrap().rank, 0);
        assert_eq!(
            tensor(&inputs[1].outputs[0]).unwrap().data_type,
            MilDataType::Bool as i32
        );
        assert_eq!(
            boundary
                .output_cast()
                .unwrap()
                .iter()
                .map(|op| op.r#type.as_str())
                .collect::<Vec<_>>(),
            ["cast", "reshape"]
        );
    }
}
