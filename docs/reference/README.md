# WebNN Specification Reference

Download or refresh the live W3C WebNN Bikeshed source from the repository root:

```bash
make fetch-webnn-spec
```

The recipe saves the source to `target/webnn-index.bs`. The download is a local
build artifact, excluded from version control. It comes from the
[WebNN main branch](https://github.com/webmachinelearning/webnn/blob/main/index.bs),
which can change independently of rustnn.

The source is human-readable and contains operation definitions, signatures, and
validation algorithms. To find builder signatures:

```bash
rg 'MLOperand \w+\(' target/webnn-index.bs
```

## Generate allowed data type metadata

```bash
make generate-operation-data-types
make check-operation-data-types
python3 -m unittest scripts/test_generate_operation_data_types.py
```

Generated explicit type lists use `MLOperandDataType` variants. Spec types absent
from the enum are omitted and listed in comments beside each affected rule.

`operation_type_support::check_allowed_data_types` checks every present operand
of an `Operation` against its `GraphInfo`, including actual `SameAs` relationships.
It returns `AllowedDataTypesError` with operation, operand, allowed types, and used
type. Builder inference applies it to single- and multi-output operations by
default. `MLContextOptions::with_rustnn_bypass_allowed_data_types_checks(true)`
skips type checks with a warning for every operation; shape and reference checks
remain active.

For packed 4-bit backend extensions, prefer
`MLContextOptions::with_rustnn_allow_int4_uint4(true)`. This allows int4/uint4
where int8/uint8 are permitted, while retaining other type and equality checks.
`check_allowed_data_types_with_options` applies both context policies to a graph
operation; the const lookup continues to describe the unextended spec domains.

For another source or output path, or offline drift checking:

```bash
python3 scripts/generate_operation_data_types.py --input target/webnn-index.bs --output target/operation_data_types.rs
python3 scripts/generate_operation_data_types.py --input target/webnn-index.bs --check
```

## Cross-References

- [Generated backend operator support](../development/backend-operator-support.md)
- [Published WebNN specification](https://www.w3.org/TR/webnn/)
- [Specification repository](https://github.com/webmachinelearning/webnn)
