> **Archived (2026-09-17).** Historical document kept for reference only. It describes
> plans, investigations or code states that no longer match the repository (removed Python
> bindings, the old executor API, superseded designs, stale status numbers). Do not use it as
> an API or workflow reference; the current documentation starts at `docs/index.md`.

# all-MiniLM-L6-v2 Text Embeddings Demo

This demo shows how to use the **all-MiniLM-L6-v2** BERT-based model for generating text embeddings using PyWebNN.

## Model Information

- **Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/tarekziade/all-MiniLM-L6-v2-webnn)
- **Base**: BERT architecture
- **Size**: 6 layers, 384 hidden dimensions, ~33M parameters
- **Output**: 384-dimensional sentence embeddings
- **Use Case**: Semantic similarity, information retrieval, clustering

## Features

- Tokenization using Hugging Face transformers
- WebNN-accelerated inference 
- Mean pooling over token embeddings
- L2 normalization for cosine similarity
- Batch processing support

## Requirements

```bash
# Build and install PyWebNN
make python-dev

# Install additional dependencies
.venv-webnn/bin/pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

## Model Files

The demo automatically downloads the WebNN model from Hugging Face Hub:
- **Model Repository**: [tarekziade/all-MiniLM-L6-v2-webnn](https://huggingface.co/tarekziade/all-MiniLM-L6-v2-webnn)

Model files (automatically cached):
```
model.webnn          # WebNN graph (text format)
model.weights        # Model weights (127 MB)
manifest.json        # Weight manifest
```

The model is automatically downloaded and cached by `huggingface_hub` on first use. Subsequent runs will use the cached version.

### Choosing the model source

By default the demo downloads `tarekziade/all-MiniLM-L6-v2-webnn` from the Hugging Face Hub. You can override this:

```bash
# Use a different Hub repo
export MINILM_MODEL_ID=my-user/my-minilm-webnn

# Or use a local directory (must contain model.webnn/model.weights/manifest.json)
export MINILM_LOCAL_MODEL_DIR=/path/to/local/model

# Troubleshoot backend selection
python examples/minilm_embeddings.py --debug
```

### Using Local Model Files

You can also use a local model directory:

```python
embedder = WebNNEmbedder(
    model_id="/path/to/local/model/directory",
    device_type="cpu"
)
```

## Usage

### Run the Demo

```bash
# Set library path for ONNX Runtime
export DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib

# Run the demo
.venv-webnn/bin/python examples/minilm_embeddings.py
```

### Expected Output

```
======================================================================
all-MiniLM-L6-v2 Text Embeddings Demo
Comparing Transformers vs WebNN implementations
======================================================================

----------------------------------------------------------------------
Initializing Transformers (Reference) Implementation
----------------------------------------------------------------------
[INFO] Loading transformers model: sentence-transformers/all-MiniLM-L6-v2
[OK] Transformers model loaded

----------------------------------------------------------------------
Encoding with Transformers
----------------------------------------------------------------------
Encoding 5 sentences...
[OK] Generated 5 embeddings
[OK] Shape: (5, 384)
[OK] First embedding (first 10 dims): [ 0.03483906  0.06283448 -0.02740623 ...]
[OK] L2 norm: 1.000000

----------------------------------------------------------------------
Semantic Similarity (Transformers)
----------------------------------------------------------------------
Cosine similarity matrix:
       S1    S2    S3    S4    S5
S1   1.000 -0.024  0.040  0.007  0.253
S2  -0.024  1.000  0.545 -0.036  0.161
S3   0.040  0.545  1.000 -0.011  0.066
S4   0.007 -0.036 -0.011  1.000  0.069
S5   0.253  0.161  0.066  0.069  1.000

Sentences:
  S1. This is a sample sentence to encode
  S2. The cat sits on the mat
  S3. A feline rests on the carpet
  S4. The weather is sunny today
  S5. Python is a programming language

Note: S2 and S3 (both about cats) have high similarity (0.545)

----------------------------------------------------------------------
Initializing WebNN Implementation
----------------------------------------------------------------------
[INFO] Loading tokenizer...
[INFO] Loading WebNN model from tarekziade/all-MiniLM-L6-v2-webnn...
[INFO] Downloading model from Hugging Face Hub: tarekziade/all-MiniLM-L6-v2-webnn
[OK] Model downloaded to ~/.cache/huggingface/hub/models--tarekziade--all-MiniLM-L6-v2-webnn/...
[INFO] Loading graph from text format...
[OK] WebNN model loaded

----------------------------------------------------------------------
Encoding with WebNN
----------------------------------------------------------------------
[WARNING] WebNN inference not yet implemented, returning random embeddings
[OK] Generated 5 embeddings

======================================================================
Comparing Transformers (Reference) vs WebNN
======================================================================

Sentence 1:
  Cosine Similarity: -0.058191
  Euclidean Distance: 1.454779
  Mean Squared Error: 0.00551141

Overall Statistics:
  Average Cosine Similarity: -0.011865 (should be >0.99 when working)

[ERROR] Embeddings are DIFFERENT (cosine < 0.80)

[NOTE] Once WebNN inference is implemented, embeddings should match
[NOTE] with cosine similarity > 0.99
======================================================================
```

## Code Structure

The demo provides two implementations for comparison:

### 1. TransformersEmbedder Class (Reference Implementation)

Uses the original Hugging Face transformers library with PyTorch:

```python
from minilm_embeddings import TransformersEmbedder

# Initialize reference embedder
embedder = TransformersEmbedder()

# Encode text
embeddings = embedder.encode([
    "Hello world",
    "How are you?"
])

# Compute similarity
similarity = embedder.compute_similarity(
    "Hello world",
    "Hi there"
)
```

### 2. WebNNEmbedder Class (WebNN Implementation)

Uses PyWebNN for accelerated inference:

```python
from minilm_embeddings import WebNNEmbedder

# Initialize WebNN embedder (downloads from Hub automatically)
embedder = WebNNEmbedder(
    model_id="tarekziade/all-MiniLM-L6-v2-webnn",
    device_type="cpu"  # or "gpu"
)

# Or use a local model directory
embedder = WebNNEmbedder(
    model_id="/path/to/local/model/directory",
    device_type="cpu"
)

# Encode text
embeddings = embedder.encode([
    "Hello world",
    "How are you?"
])

# Compute similarity
similarity = embedder.compute_similarity(
    "Hello world",
    "Hi there"
)
```

### Comparison Function

```python
from minilm_embeddings import compare_embeddings

# Compare two sets of embeddings
compare_embeddings(
    transformers_embeddings,
    webnn_embeddings,
    "Transformers",
    "WebNN"
)
```

### Key Methods

1. **`encode(texts, normalize=True)`**
   - Tokenizes input texts
   - Runs WebNN inference
   - Applies mean pooling
   - Returns normalized embeddings

2. **`compute_similarity(text1, text2)`**
   - Encodes both texts
   - Computes cosine similarity
   - Returns similarity score

## Model Architecture

The all-MiniLM-L6-v2 model follows this pipeline:

```
Input Text
    |
    v
Tokenization (BERT WordPiece)
    |
    v
Input IDs + Attention Mask + Token Type IDs
    |
    v
Embedding Layer (word + position + token_type)
    |
    v
12 x Transformer Layers (self-attention + FFN)
    |
    v
Last Hidden State [batch, seq_len, 384]
    |
    v
Mean Pooling (over sequence length)
    |
    v
L2 Normalization
    |
    v
Sentence Embedding [batch, 384]
```

## Model Inputs

The WebNN model expects three inputs:

1. **input_ids**: `int64[batch, seq_len]`
   - Token IDs from BERT vocabulary (30522 tokens)
   - Special tokens: [CLS]=101, [SEP]=102, [PAD]=0

2. **attention_mask**: `int64[batch, seq_len]`
   - 1 for real tokens, 0 for padding

3. **token_type_ids**: `int64[batch, seq_len]`
   - Segment IDs (0 for first sentence, 1 for second)
   - Usually all zeros for single sentence tasks

## Model Outputs

The model outputs:

- **last_hidden_state**: `float32[batch, seq_len, 384]`
  - Hidden representations for each token
  - Used for token-level tasks or pooling

After mean pooling and normalization:

- **sentence_embedding**: `float32[batch, 384]`
  - Single vector per sentence
  - Suitable for cosine similarity

## Validation Approach

The demo includes a **reference implementation using transformers** to validate WebNN results:

1. **Reference Embeddings**: The `TransformersEmbedder` generates ground truth embeddings using the original PyTorch model
2. **WebNN Embeddings**: The `WebNNEmbedder` generates embeddings using WebNN (currently random placeholders)
3. **Comparison Metrics**:
   - Cosine Similarity (should be > 0.99 for identical models)
   - Euclidean Distance (should be close to 0)
   - Mean Squared Error (should be < 0.0001)
   - Mean Absolute Error (should be < 0.01)

### Example Comparison Output

```
Comparing Transformers (Reference) vs WebNN
======================================================================

Sentence 1:
  Cosine Similarity: 0.999234  [EXPECTED: >0.99 when working]
  Euclidean Distance: 0.039123  [EXPECTED: ~0 when working]
  Mean Squared Error: 0.00000398  [EXPECTED: <0.0001 when working]
  Mean Absolute Error: 0.00156789  [EXPECTED: <0.01 when working]

Overall Statistics:
  Average Cosine Similarity: 0.999128

[OK] Embeddings are VERY SIMILAR (cosine > 0.99)
```

This validation ensures that when WebNN inference is fully implemented, it produces numerically identical results to the reference implementation.

## Current Limitations

The current demo uses **placeholder random embeddings** for the WebNN implementation. Full WebNN inference is not yet implemented because:

1. WebNN text format loading needs complete implementation
2. Text format to graph conversion pipeline needs to be built
3. Weight loading from manifest needs to be integrated

## Next Steps

To enable full WebNN inference, the following needs to be implemented:

1. **WebNN Text Format Parser**
   - Parse `.webnn` text format into graph structure
   - Extract operation types, shapes, and connections

2. **Weight Loader**
   - Read manifest JSON
   - Load weights from binary file using byte offsets
   - Create constant tensors in WebNN graph

3. **Graph Builder**
   - Convert parsed operations to WebNN API calls
   - Build complete computational graph
   - Set up input/output mappings

4. **Inference Pipeline**
   - Run actual WebNN compute() calls
   - Handle BERT-specific operations (embeddings, attention, etc.)
   - Return real tensor outputs

## Use Cases

Once fully implemented, this model can be used for:

1. **Semantic Search**
   - Index documents as embeddings
   - Find similar documents via cosine similarity

2. **Clustering**
   - Group similar texts together
   - Topic detection

3. **Duplicate Detection**
   - Find near-duplicate text content
   - Paraphrase detection

4. **Question Answering**
   - Match questions to answers
   - Retrieve relevant passages

5. **Recommendation Systems**
   - Content-based recommendations
   - Similar item retrieval

## Performance

Expected performance (once fully implemented):

- **CPU (ONNX Runtime)**: ~50-100 ms/sentence
- **GPU (ONNX Runtime)**: ~10-20 ms/sentence
- **Apple Silicon (CoreML)**: ~15-30 ms/sentence

Batch processing significantly improves throughput for multiple sentences.

## References

- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2 Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [W3C WebNN Specification](https://www.w3.org/TR/webnn/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

## License

The all-MiniLM-L6-v2 model is licensed under Apache 2.0. See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for details.
