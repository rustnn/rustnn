# WPT Test Data

This directory contains test data converted from the [W3C Web Platform Tests (WPT) for WebNN](https://github.com/web-platform-tests/wpt/tree/master/webnn).

## Directory Structure

```
wpt_data/
├── README.md           # This file
├── conformance/        # Conformance test data (operation correctness)
│   ├── relu.json
│   ├── reduce_sum.json
│   └── ...
└── validation/         # Validation test data (parameter validation)
    ├── relu.json
    └── ...
```

## Test Data Format

Each JSON file contains test cases for a single operation:

```json
{
  "operation": "reduce_sum",
  "wpt_version": "2025-12-07",
  "wpt_commit": "abc123...",
  "source_file": "conformance_tests/reduce.https.any.js",
  "tests": [
    {
      "name": "reduce_sum float32 2D tensor with axes=[1]",
      "inputs": {
        "input": {
          "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          "shape": [2, 3],
          "dataType": "float32"
        }
      },
      "operators": [
        {
          "name": "reduce_sum",
          "arguments": {
            "input": "input",
            "axes": [1],
            "keepDimensions": false
          },
          "output": "output"
        }
      ],
      "expectedOutputs": {
        "output": {
          "data": [6.0, 15.0],
          "shape": [2],
          "dataType": "float32"
        }
      },
      "tolerance": {
        "type": "ULP",
        "value": 0
      }
    }
  ]
}
```

## Updating Test Data

### Automatic Update (Recommended)

Use the update script to sync with WPT upstream:

```bash
# Update all test data from WPT repository
./scripts/update_wpt_tests.sh

# Update specific operations only
./scripts/update_wpt_tests.sh --operations reduce_sum,relu,add
```

The script will:
1. Clone/update the WPT repository
2. Convert JavaScript test files to JSON
3. Preserve any local modifications
4. Generate a change report

### Manual Conversion

If you need to convert tests manually:

```bash
# Convert a specific operation
python scripts/convert_wpt_tests.py \
  --wpt-repo ~/wpt \
  --operation reduce_sum \
  --output tests/wpt_data/conformance/

# Convert multiple operations
python scripts/convert_wpt_tests.py \
  --wpt-repo ~/wpt \
  --operations reduce_sum,reduce_mean,relu \
  --output tests/wpt_data/conformance/
```

## Test Data Versioning

Each JSON file includes metadata about the WPT version:
- `wpt_version`: Date of WPT snapshot
- `wpt_commit`: Git commit SHA from WPT repository
- `source_file`: Original JavaScript test file path

This allows tracking which WPT version each test came from, making updates easier.

## Running Tests

```bash
# Run all WPT conformance tests
pytest tests/test_wpt_conformance.py -v

# Run tests for specific operation
pytest tests/test_wpt_conformance.py -k "reduce_sum" -v

# Run with coverage report
pytest tests/test_wpt_conformance.py --cov=webnn --cov-report=html
```

## Contributing

When adding new operations:
1. Run the converter script to generate test data
2. Verify tests pass with your implementation
3. Commit both the implementation and test data
4. Document any test failures or skips

## Notes

- Test data is committed to the repository for reproducibility
- Large test files (>1MB) may be stored separately
- Tolerance values are preserved from WPT tests
- Custom tolerance overrides can be specified in `tests/wpt_utils.py`
