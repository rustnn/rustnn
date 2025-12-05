# Examples

Practical examples demonstrating the WebNN Python API.

## Basic Examples

### Simple Addition

```python
import webnn
import numpy as np

# Create context and builder
ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Define computation: z = x + y
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)

# Compile and export
graph = builder.build({"z": z})
context.convert_to_onnx(graph, "add.onnx")
print("✓ Simple addition graph exported")
```

### ReLU Activation

```python
import webnn

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Apply ReLU to input
x = builder.input("x", [10], "float32")
y = builder.relu(x)

graph = builder.build({"y": y})
context.convert_to_onnx(graph, "relu.onnx")
```

## Intermediate Examples

### Linear Layer

A simple fully-connected layer: `output = input @ weights + bias`

```python
import webnn
import numpy as np

def create_linear_layer(builder, input_op, in_features, out_features):
    """Creates a linear layer with random initialization."""

    # Create weight matrix [in_features, out_features]
    weights = np.random.randn(in_features, out_features).astype('float32') * 0.01
    weights_op = builder.constant(weights)

    # Create bias vector [out_features]
    bias = np.zeros(out_features, dtype='float32')
    bias_op = builder.constant(bias)

    # Compute: output = input @ weights + bias
    matmul_result = builder.matmul(input_op, weights_op)
    output = builder.add(matmul_result, bias_op)

    return output

# Use the linear layer
ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Input: batch_size=1, features=784 (e.g., flattened 28x28 image)
input_tensor = builder.input("input", [1, 784], "float32")

# Linear layer: 784 -> 10 (e.g., for digit classification)
output = create_linear_layer(builder, input_tensor, 784, 10)

# Compile and export
graph = builder.build({"output": output})
context.convert_to_onnx(graph, "linear_layer.onnx")

print(f"Linear layer: {graph.operand_count} operands, {graph.operation_count} operations")
```

### Multi-Layer Network

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Input layer
input_tensor = builder.input("input", [1, 784], "float32")

# Hidden layer 1: 784 -> 128
w1 = builder.constant(np.random.randn(784, 128).astype('float32') * 0.01)
b1 = builder.constant(np.zeros(128, dtype='float32'))
hidden1 = builder.matmul(input_tensor, w1)
hidden1 = builder.add(hidden1, b1)
hidden1 = builder.relu(hidden1)

# Hidden layer 2: 128 -> 64
w2 = builder.constant(np.random.randn(128, 64).astype('float32') * 0.01)
b2 = builder.constant(np.zeros(64, dtype='float32'))
hidden2 = builder.matmul(hidden1, w2)
hidden2 = builder.add(hidden2, b2)
hidden2 = builder.relu(hidden2)

# Output layer: 64 -> 10
w3 = builder.constant(np.random.randn(64, 10).astype('float32') * 0.01)
b3 = builder.constant(np.zeros(10, dtype='float32'))
output = builder.matmul(hidden2, w3)
output = builder.add(output, b3)

# Compile
graph = builder.build({"logits": output})
context.convert_to_onnx(graph, "mlp.onnx")

print(f"Multi-layer network compiled:")
print(f"  Operands: {graph.operand_count}")
print(f"  Operations: {graph.operation_count}")
```

## Advanced Examples

### Multiple Outputs

Create a graph with multiple outputs:

```python
import webnn

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Input
x = builder.input("x", [1, 10], "float32")

# Multiple transformations
relu_out = builder.relu(x)
sigmoid_out = builder.sigmoid(x)
tanh_out = builder.tanh(x)

# Build with multiple named outputs
graph = builder.build({
    "relu": relu_out,
    "sigmoid": sigmoid_out,
    "tanh": tanh_out
})

# Check outputs
print("Outputs:", graph.get_output_names())
# Output: ['relu', 'sigmoid', 'tanh']

context.convert_to_onnx(graph, "multi_output.onnx")
```

### Working with Different Data Types

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Float16 for reduced memory
x_fp16 = builder.input("x_fp16", [100, 100], "float16")
y_fp16 = builder.relu(x_fp16)

# Int8 for quantized models
x_int8 = builder.input("x_int8", [100, 100], "int8")
# Note: Quantized operations would need appropriate scaling

# Float32 (default)
x_fp32 = builder.input("x_fp32", [100, 100], "float32")
y_fp32 = builder.relu(x_fp32)

graph = builder.build({
    "out_fp16": y_fp16,
    "out_fp32": y_fp32
})

print(f"Graph with mixed precision: {graph.operand_count} operands")
```

### Reshaping Tensors

```python
import webnn

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Flatten image: [1, 28, 28, 1] -> [1, 784]
image = builder.input("image", [1, 28, 28, 1], "float32")
flattened = builder.reshape(image, [1, 784])

# Unflatten back: [1, 784] -> [1, 28, 28, 1]
unflattened = builder.reshape(flattened, [1, 28, 28, 1])

graph = builder.build({"output": unflattened})
context.convert_to_onnx(graph, "reshape.onnx")
```

### Converting Pre-trained NumPy Weights

```python
import webnn
import numpy as np

def convert_numpy_model_to_webnn(weights_dict):
    """
    Convert a model with NumPy weights to WebNN graph.

    Args:
        weights_dict: Dictionary with keys like 'fc1.weight', 'fc1.bias', etc.
    """
    ml = webnn.ML()
    context = ml.create_context()
    builder = context.create_graph_builder()

    # Input
    x = builder.input("input", [1, 784], "float32")

    # Layer 1
    w1 = builder.constant(weights_dict['fc1.weight'].astype('float32'))
    b1 = builder.constant(weights_dict['fc1.bias'].astype('float32'))
    h1 = builder.matmul(x, w1)
    h1 = builder.add(h1, b1)
    h1 = builder.relu(h1)

    # Layer 2
    w2 = builder.constant(weights_dict['fc2.weight'].astype('float32'))
    b2 = builder.constant(weights_dict['fc2.bias'].astype('float32'))
    output = builder.matmul(h1, w2)
    output = builder.add(output, b2)

    # Build and export
    graph = builder.build({"logits": output})
    context.convert_to_onnx(graph, "converted_model.onnx")

    return graph

# Example usage
weights = {
    'fc1.weight': np.random.randn(784, 128),
    'fc1.bias': np.zeros(128),
    'fc2.weight': np.random.randn(128, 10),
    'fc2.bias': np.zeros(10),
}

graph = convert_numpy_model_to_webnn(weights)
print(f"✓ Converted model: {graph.operation_count} operations")
```

## Error Handling Examples

### Graceful Error Handling

```python
import webnn
import numpy as np

def build_and_export_safely(output_path):
    """Build a graph with proper error handling."""
    try:
        ml = webnn.ML()
        context = ml.create_context()
        builder = context.create_graph_builder()

        x = builder.input("x", [10], "float32")
        y = builder.relu(x)

        graph = builder.build({"y": y})

        # Try ONNX conversion
        try:
            context.convert_to_onnx(graph, output_path)
            print(f"✓ ONNX model saved to {output_path}")
            return True
        except IOError as e:
            print(f"✗ Failed to save ONNX: {e}")
            return False

    except ValueError as e:
        print(f"✗ Graph validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

# Use it
success = build_and_export_safely("model.onnx")
```

### Validating Shapes

```python
import webnn
import numpy as np

def create_safe_matmul(builder, a_shape, b_shape):
    """Create matmul with shape validation."""
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("matmul requires 2D tensors")

    if a_shape[1] != b_shape[0]:
        raise ValueError(
            f"Incompatible shapes for matmul: "
            f"{a_shape} and {b_shape}"
        )

    a = builder.input("a", a_shape, "float32")
    b_data = np.random.randn(*b_shape).astype('float32')
    b = builder.constant(b_data)

    result = builder.matmul(a, b)
    return result

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

try:
    # Valid
    output = create_safe_matmul(builder, [10, 20], [20, 30])
    print("✓ Valid matmul created")

    # Invalid - will raise error
    output = create_safe_matmul(builder, [10, 20], [15, 30])
except ValueError as e:
    print(f"✗ Shape validation failed: {e}")
```

## Complete Application Example

### Image Classification Pipeline

```python
import webnn
import numpy as np

class SimpleClassifier:
    """A simple image classifier using WebNN."""

    def __init__(self, num_classes=10):
        self.ml = webnn.ML()
        self.context = self.ml.create_context()
        self.graph = None
        self.num_classes = num_classes

    def build_model(self):
        """Build the classification model."""
        builder = self.context.create_graph_builder()

        # Input: 28x28 grayscale images
        input_tensor = builder.input("image", [1, 28, 28, 1], "float32")

        # Flatten
        x = builder.reshape(input_tensor, [1, 784])

        # Hidden layer
        w1 = builder.constant(np.random.randn(784, 128).astype('float32') * 0.01)
        b1 = builder.constant(np.zeros(128, dtype='float32'))
        x = builder.matmul(x, w1)
        x = builder.add(x, b1)
        x = builder.relu(x)

        # Output layer
        w2 = builder.constant(np.random.randn(128, self.num_classes).astype('float32') * 0.01)
        b2 = builder.constant(np.zeros(self.num_classes, dtype='float32'))
        logits = builder.matmul(x, w2)
        logits = builder.add(logits, b2)

        # Softmax
        output = builder.softmax(logits)

        # Build
        self.graph = builder.build({"probabilities": output})
        print(f"✓ Model built: {self.graph.operation_count} operations")

    def export(self, path="classifier.onnx"):
        """Export the model to ONNX."""
        if self.graph is None:
            raise RuntimeError("Build model first!")

        self.context.convert_to_onnx(self.graph, path)
        print(f"✓ Model exported to {path}")

    def get_info(self):
        """Get model information."""
        if self.graph is None:
            return "Model not built yet"

        return {
            "operands": self.graph.operand_count,
            "operations": self.graph.operation_count,
            "inputs": self.graph.get_input_names(),
            "outputs": self.graph.get_output_names(),
        }

# Use the classifier
classifier = SimpleClassifier(num_classes=10)
classifier.build_model()
classifier.export("mnist_classifier.onnx")

print("\nModel Info:")
for key, value in classifier.get_info().items():
    print(f"  {key}: {value}")
```

This comprehensive set of examples should help you get started with various use cases!
