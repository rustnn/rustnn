//! A constant dequantization chain lowered to one `constexpr_blockwise_shift_scale` constant
//! returns the bits of the literal `cast`/`sub`/`reshape`/`mul`/`reshape` lowering.
//!
//! Each case builds the chain twice. The first graph is recognized by the CoreML converter;
//! in the second the `cast` has another reader, which keeps the literal lowering. Both run
//! on the CoreML runtime, as a stateless program through the in-memory route and through
//! the URL compiler and as a program that carries state, and both must return
//! `fl((q - zero) * scale)`, rounded once in the chain's float type. Every block holds every
//! byte value, so each scale meets each stored integer, including the one equal to the zero
//! point (a zero whose sign is the scale's).

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use prost::Message;
use rustnn::backend_selection::DeviceType;
use rustnn::converters::CoremlMlProgramConverter;
use rustnn::converters::coreml_state::carried_state_pairs;
use rustnn::executors::coreml::CoremlLoadRoute;
use rustnn::graph::{DataType, GraphInfo};
use rustnn::mlcontext::{
    BackendDevice, LoadDiagnostics, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands,
    MLNamedTensors, MLOperand, MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::mlcontextoptions::{CoremlFloat16Precision, CoremlOptions, RustNNOptions};
use rustnn::operator_options::MLDimension;
use rustnn::protos::coreml::specification::{Model, model};

const ROWS: u32 = 8;
const GROUPS: u32 = 2;
const WIDTH: u32 = 256;
const COLUMNS: u32 = GROUPS * WIDTH;
const BLOCKWISE: &[u8] = b"constexpr_blockwise_shift_scale";

/// One scale per (row, block): both signs, values whose products with the stored integers
/// need rounding in float16 and in float32, magnitudes up to the largest that keeps every
/// product finite in float16, one whose float16 products are subnormal, and one that is
/// subnormal in float32 and zero in float16.
const SCALES: [f32; (ROWS * GROUPS) as usize] = [
    0.1,
    -0.1,
    0.333_333_34,
    -3.0,
    1.000_976_6,
    0.007_812_5,
    123.456,
    -0.000_123,
    1.0e-6,
    255.0,
    0.999,
    -1.5,
    6.25e-2,
    -1.0e-40,
    250.0,
    0.5,
];

#[derive(Clone, Copy, Debug)]
struct Case {
    stored: DataType,
    float: DataType,
    zero: Option<f32>,
}

/// How the program is built and loaded.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Program {
    /// Stateless, loaded as an in-memory asset.
    InMemory,
    /// Stateless, with a `gather` over a rank-0 index, which the backend compiles from a URL.
    Url,
    /// Carries a state, which always compiles from a URL; `matmul` reads the constant with
    /// `transpose_y`.
    Stateful,
}

impl Program {
    fn route(self) -> CoremlLoadRoute {
        match self {
            Program::InMemory => CoremlLoadRoute::InMemoryAsset,
            Program::Url | Program::Stateful => CoremlLoadRoute::CompiledUrl,
        }
    }

    fn options(self, precision: CoremlFloat16Precision) -> CoremlOptions {
        let mut options = RustNNOptions::default().coreml;
        options.float16_precision = precision;
        if self == Program::Stateful {
            options.carried_state_pairs = vec![("state".into(), "next_state".into())];
        }
        options
    }
}

fn dimensions(shape: &[u32]) -> Vec<MLDimension> {
    shape
        .iter()
        .map(|&size| MLDimension::Static(size))
        .collect()
}

/// Every byte value once in every block of `WIDTH` columns, in a different order per row.
fn stored_bytes() -> Vec<u8> {
    (0..ROWS * COLUMNS)
        .map(|i| ((i % COLUMNS) * 37 + (i / COLUMNS) * 11) as u8)
        .collect()
}

fn float_bytes(values: &[f32], data_type: DataType) -> Vec<u8> {
    match data_type {
        DataType::Float16 => values
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_le_bytes())
            .collect(),
        _ => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn element_size(data_type: DataType) -> usize {
    if data_type == DataType::Float16 { 2 } else { 4 }
}

/// The scale as the graph stores it, widened exactly.
fn stored_scale(index: usize, data_type: DataType) -> f64 {
    match data_type {
        DataType::Float16 => half::f16::from_f32(SCALES[index]).to_f64(),
        _ => f64::from(SCALES[index]),
    }
}

/// `fl((q - zero) * scale)` for every element, as little-endian bytes of the float type.
/// The product of an integer of magnitude at most 255 and a float32 is exact in `f64`, so
/// the conversion below is the single rounding.
fn expected(case: Case) -> Vec<u8> {
    stored_bytes()
        .iter()
        .enumerate()
        .flat_map(|(index, &byte)| {
            let (row, column) = (index as u32 / COLUMNS, index as u32 % COLUMNS);
            let stored = match case.stored {
                DataType::Int8 => f64::from(byte as i8),
                _ => f64::from(byte),
            };
            let scale = stored_scale((row * GROUPS + column / WIDTH) as usize, case.float);
            let product = (stored - f64::from(case.zero.unwrap_or(0.0))) * scale;
            match case.float {
                DataType::Float16 => half::f16::from_f64(product).to_le_bytes().to_vec(),
                _ => (product as f32).to_le_bytes().to_vec(),
            }
        })
        .collect()
}

/// The chain, read by a `gather` of its rows and by `matmul(x, transpose(weight))`.
///
/// With `literal`, the `cast` has a second reader and the converter keeps the chain as it
/// is.
fn graph(case: Case, literal: bool, program: Program) -> GraphInfo {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let descriptor = |data_type, shape: &[u32]| {
        MLOperandDescriptor::new(data_type, shape.iter().map(|&d| u64::from(d)).collect())
    };
    let data = builder
        .constant_from_bytes(&descriptor(case.stored, &[ROWS, COLUMNS]), stored_bytes())
        .unwrap();
    let widened = builder.cast(data, case.float).unwrap();
    let centered = match case.zero {
        Some(zero) => {
            let zero = builder
                .constant_from_bytes(
                    &descriptor(case.float, &[]),
                    float_bytes(&[zero], case.float),
                )
                .unwrap();
            builder.sub(widened, zero).unwrap()
        }
        None => widened,
    };
    let blocks = builder
        .reshape(centered, dimensions(&[ROWS, GROUPS, WIDTH]))
        .unwrap();
    let scale = builder
        .constant_from_bytes(
            &descriptor(case.float, &[ROWS, GROUPS, 1]),
            float_bytes(&SCALES, case.float),
        )
        .unwrap();
    let scaled = builder.mul(blocks, scale).unwrap();
    let weight = builder
        .reshape(scaled, dimensions(&[ROWS, COLUMNS]))
        .unwrap();

    let index = builder
        .input("index", &descriptor(DataType::Int32, &[ROWS]))
        .unwrap();
    let rows = builder.gather(weight, index).unwrap();
    let x = builder
        .input("x", &descriptor(case.float, &[COLUMNS, COLUMNS]))
        .unwrap();
    let transposed = builder.transpose(weight).unwrap();
    let product = builder.matmul(x, transposed).unwrap();
    let mut outputs: Vec<(&str, MLOperand)> = vec![("rows", rows), ("product", product)];
    if literal {
        outputs.push(("probe", builder.gather(widened, index).unwrap()));
    }
    match program {
        Program::InMemory => {}
        Program::Url => {
            let first = builder
                .input("first", &descriptor(DataType::Int32, &[]))
                .unwrap();
            outputs.push(("row", builder.gather(weight, first).unwrap()));
        }
        Program::Stateful => {
            let state = builder
                .input("state", &descriptor(case.float, &[1, 4]))
                .unwrap();
            outputs.push(("next_state", builder.add(state, state).unwrap()));
        }
    }
    let mut named = MLNamedOperands::new();
    for (name, operand) in outputs {
        named.insert(name, operand);
    }
    builder.finish_graph_info(&named).unwrap()
}

/// The serialized model of `graph` as the backend converts it.
fn convert(graph: &GraphInfo, program: Program, precision: CoremlFloat16Precision) -> Vec<u8> {
    let pairs = carried_state_pairs(graph, &program.options(precision)).unwrap();
    CoremlMlProgramConverter
        .convert_with_carried_state(graph, precision, &pairs)
        .expect("the graph converts")
        .data
}

fn is_blockwise(model: &[u8]) -> bool {
    model
        .windows(BLOCKWISE.len())
        .any(|window| window == BLOCKWISE)
}

/// Number of `transpose` operations in the main function of a single-program model.
fn transposes(model: &[u8]) -> usize {
    let Some(model::Type::MlProgram(program)) = Model::decode(model).unwrap().r#type else {
        panic!("one ML program")
    };
    program.functions["main"]
        .block_specializations
        .values()
        .flat_map(|block| &block.operations)
        .filter(|operation| operation.r#type == "transpose")
        .count()
}

/// Runs `graph` on the CPU and returns the bytes of `rows` and `product`.
fn run(
    case: Case,
    graph: GraphInfo,
    precision: CoremlFloat16Precision,
    program: Program,
) -> (Vec<u8>, Vec<u8>) {
    let output_names: Vec<String> = graph
        .output_operands
        .iter()
        .map(|&id| graph.operand(id).unwrap().name.clone().unwrap())
        .collect();
    let mut options = RustNNOptions::default();
    options.coreml = program.options(precision);
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_device_hint(BackendDevice::Coreml {
                device_type: DeviceType::Cpu,
            })
            .with_rustnn_options(options),
    )
    .expect("create a CPU CoreML context");
    let mut compiled = context.rustnn_build_graph(graph).expect("the graph builds");
    let Some(LoadDiagnostics::Coreml(diagnostics)) = compiled.rustnn_load_diagnostics() else {
        panic!("CoreML load diagnostics")
    };
    assert_eq!(diagnostics.route, program.route(), "{case:?} {precision:?}");
    assert_eq!(
        diagnostics.carried_state_pairs.len(),
        usize::from(program == Program::Stateful)
    );

    let index = context
        .create_tensor(
            &MLTensorDescriptor::new(DataType::Int32, vec![u64::from(ROWS)]).to_writable(),
        )
        .unwrap();
    context
        .write_tensor(&index, &(0..ROWS as i32).collect::<Vec<_>>())
        .unwrap();
    let first = context
        .create_tensor(&MLTensorDescriptor::new(DataType::Int32, vec![]).to_writable())
        .unwrap();
    context.write_tensor(&first, &[0i32]).unwrap();
    let x = context
        .create_tensor(
            &MLTensorDescriptor::new(case.float, vec![u64::from(COLUMNS); 2]).to_writable(),
        )
        .unwrap();
    // The identity matrix: `product` is then the transposed weight, each element an exact
    // sum of one value and zeros.
    let identity: Vec<f32> = (0..COLUMNS * COLUMNS)
        .map(|i| if i / COLUMNS == i % COLUMNS { 1.0 } else { 0.0 })
        .collect();
    context
        .write_tensor(&x, &float_bytes(&identity, case.float))
        .unwrap();

    let output = |context: &mut MLContext<'_>, shape: &[u32]| {
        context
            .create_tensor(
                &MLTensorDescriptor::new(case.float, shape.iter().map(|&d| u64::from(d)).collect())
                    .to_readable(),
            )
            .unwrap()
    };
    let rows = output(&mut context, &[ROWS, COLUMNS]);
    let product = output(&mut context, &[COLUMNS, ROWS]);
    let probe = output(&mut context, &[ROWS, COLUMNS]);
    let row = output(&mut context, &[COLUMNS]);
    let mut inputs = MLNamedTensors::from([("index", &index), ("x", &x)]);
    let mut outputs = MLNamedTensors::from([("rows", &rows), ("product", &product)]);
    if output_names.iter().any(|name| name == "probe") {
        outputs.insert("probe", &probe);
    }
    if output_names.iter().any(|name| name == "row") {
        inputs.insert("first", &first);
        outputs.insert("row", &row);
    }
    context
        .dispatch(&mut compiled, &inputs, &outputs)
        .expect("dispatch");
    let size = (ROWS * COLUMNS) as usize * element_size(case.float);
    let mut row_bytes = vec![0u8; size];
    context.read_tensor(&rows, &mut row_bytes).unwrap();
    let mut product_bytes = vec![0u8; size];
    context.read_tensor(&product, &mut product_bytes).unwrap();
    (row_bytes, product_bytes)
}

/// Compares element bits and names the stored byte and scale of the elements that differ.
fn assert_same_bits(label: &str, case: Case, actual: &[u8], expected: &[u8]) {
    let element = element_size(case.float);
    let stored = stored_bytes();
    let differences: Vec<String> = actual
        .chunks_exact(element)
        .zip(expected.chunks_exact(element))
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected)
        .map(|(index, (actual, expected))| {
            let block = index as u32 / COLUMNS * GROUPS + index as u32 % COLUMNS / WIDTH;
            format!(
                "element {index}: stored {} scale {} returned {actual:02x?} expected {expected:02x?}",
                stored[index], SCALES[block as usize]
            )
        })
        .collect();
    assert!(
        differences.is_empty() && actual.len() == expected.len(),
        "{label}: {} of {} elements differ\n{}",
        differences.len(),
        expected.len() / element,
        differences
            .iter()
            .take(24)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Float values of little-endian bytes, for comparisons where the sign of zero is free.
fn values(bytes: &[u8], data_type: DataType) -> Vec<f64> {
    match data_type {
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f64())
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|chunk| f64::from(f32::from_le_bytes(chunk.try_into().unwrap())))
            .collect(),
    }
}

/// Runs the recognized and the literal graph and compares both with the reference.
/// `blockwise` is whether the converter is expected to rewrite the first graph.
fn check(case: Case, precision: CoremlFloat16Precision, program: Program, blockwise: bool) {
    let label = format!("{case:?} {precision:?} {program:?}");
    let (recognized, literal) = (graph(case, false, program), graph(case, true, program));
    let model = convert(&recognized, program, precision);
    assert_eq!(is_blockwise(&model), blockwise, "{label}");
    assert!(
        !is_blockwise(&convert(&literal, program, precision)),
        "{label}: literal"
    );
    if blockwise {
        // Only a program that carries state reads the constant with `transpose_y`.
        assert_eq!(
            transposes(&model),
            usize::from(program != Program::Stateful),
            "{label}"
        );
    }

    let (blockwise_rows, blockwise_product) = run(case, recognized, precision, program);
    let (literal_rows, literal_product) = run(case, literal, precision, program);
    let expected = expected(case);
    assert_same_bits(
        &format!("{label}: literal lowering"),
        case,
        &literal_rows,
        &expected,
    );
    assert_same_bits(
        &format!("{label}: blockwise constant"),
        case,
        &blockwise_rows,
        &expected,
    );

    // The transposed reader returns the same matrix. A sum with exact zeros may lose the
    // sign of a zero, so compare values there.
    let weight = values(&expected, case.float);
    let transposed: Vec<f64> = (0..COLUMNS * ROWS)
        .map(|i| weight[((i % ROWS) * COLUMNS + i / ROWS) as usize])
        .collect();
    assert_eq!(
        values(&blockwise_product, case.float),
        transposed,
        "{label}: matmul over the blockwise constant"
    );
    assert_eq!(
        values(&literal_product, case.float),
        transposed,
        "{label}: matmul of the literal transpose"
    );
}

fn cases(float: DataType) -> [Case; 5] {
    [
        (DataType::Uint8, Some(128.0)),
        (DataType::Int8, Some(0.0)),
        (DataType::Int8, None),
        (DataType::Int8, Some(-5.0)),
        (DataType::Uint8, Some(255.0)),
    ]
    .map(|(stored, zero)| Case {
        stored,
        float,
        zero,
    })
}

#[test]
fn float16_chains_return_the_bits_of_the_literal_lowering() {
    for case in cases(DataType::Float16) {
        for program in [Program::InMemory, Program::Url, Program::Stateful] {
            check(case, CoremlFloat16Precision::Native, program, true);
        }
    }
}

#[test]
fn float32_chains_return_the_bits_of_the_literal_lowering() {
    for case in cases(DataType::Float32) {
        for program in [Program::InMemory, Program::Url, Program::Stateful] {
            check(case, CoremlFloat16Precision::Native, program, true);
        }
        // Without a float16 operand the exact lowering is the native one; it compiles
        // from a URL.
        check(case, CoremlFloat16Precision::Exact, Program::Url, true);
    }
}

/// Under the exact float16 lowering a float16 graph becomes a precision pipeline, where the
/// compressed constant returned `+0` for a zero difference times a negative scale and the
/// literal chain returns `-0`. Such a graph keeps the literal chain.
#[test]
fn exact_float16_graphs_keep_the_literal_chain() {
    for case in cases(DataType::Float16) {
        check(case, CoremlFloat16Precision::Exact, Program::Url, false);
    }
}
