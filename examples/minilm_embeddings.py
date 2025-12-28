#!/usr/bin/env python3
"""
all-MiniLM-L6-v2 Text Embeddings Demo using PyWebNN

This demo shows how to use the all-MiniLM-L6-v2 BERT-based model
for generating text embeddings using WebNN acceleration, and compares
the results with the original transformers implementation.

Model: Xenova/all-MiniLM-L6-v2 (sentence-transformers)
- 12 layers, 384 hidden dimensions
- Optimized for semantic similarity tasks
- Outputs 384-dimensional embeddings
"""

import os
import sys
import numpy as np
from pathlib import Path

# Local model cache to avoid network access when running the demo.
LOCAL_MODEL_DIR = Path("/Users/tarekziade/Dev/all-MiniLM-L6-v2-webnn")
MODEL_ID = os.environ.get("MINILM_MODEL_ID", str(LOCAL_MODEL_DIR))
# WebNN graph uses static [1, 128] inputs; keep tokenizer config aligned.
MAX_LEN = 128

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    print("[ERROR] transformers or torch library not found")
    print("Install them with: pip install transformers torch")
    sys.exit(1)

import webnn


class TransformersEmbedder:
    """Reference implementation using Hugging Face transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder using transformers

        Args:
            model_name: Hugging Face model identifier
        """
        local_only = bool(
            os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        )
        print(f"[INFO] Loading transformers model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_only
        )
        self.model.eval()  # Set to evaluation mode
        print("[OK] Transformers model loaded")

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings using attention mask

        Args:
            token_embeddings: Shape [batch, seq_len, hidden_dim]
            attention_mask: Shape [batch, seq_len]

        Returns:
            Sentence embeddings: Shape [batch, hidden_dim]
        """
        # Expand attention mask to match embeddings dimensions
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Sum embeddings weighted by mask
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

        # Sum of mask (number of valid tokens)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Calculate mean
        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit length (L2 norm)"""
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings using transformers

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings to unit length

        Returns:
            Embeddings array of shape [num_texts, 384]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",  # Return as PyTorch tensors
        )

        # Run inference
        with torch.no_grad():
            model_output = self.model(**encoded)

        # Get token embeddings from last hidden state
        token_embeddings = model_output.last_hidden_state

        # Apply mean pooling
        embeddings = self._mean_pooling(token_embeddings, encoded["attention_mask"])

        # Normalize
        if normalize:
            embeddings = self._normalize(embeddings)

        # Convert to numpy
        return embeddings.cpu().numpy()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1
        """
        embeddings = self.encode([text1, text2], normalize=True)
        # Cosine similarity with normalized vectors is just dot product
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)


class WebNNEmbedder:
    """Text embedding generator using all-MiniLM-L6-v2 model via WebNN"""

    def __init__(
        self,
        model_id: str = str(LOCAL_MODEL_DIR),
        device_type: str = "cpu",
    ):
        """
        Initialize the embedder

        Args:
            model_id: Local path to directory containing .webnn and .weights files
                      (defaults to LOCAL_MODEL_DIR), or a Hugging Face model identifier
            device_type: Device to use ('cpu' or 'gpu')
        """
        self.model_id = model_id
        self.device_type = device_type

        # Initialize tokenizer (from Hugging Face)
        print("[INFO] Loading tokenizer...")
        local_only = bool(
            os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=local_only,
        )

        # Load WebNN model
        print(f"[INFO] Loading WebNN model from {model_id}...")
        self.graph = self._load_webnn_model()
        print("[OK] WebNN model loaded")

    def _download_from_hub(self, model_id: str) -> Path:
        """
        Download model files from Hugging Face Hub using Hub class

        Args:
            model_id: Hugging Face model identifier

        Returns:
            Path to downloaded model directory
        """
        from huggingface_hub import snapshot_download

        print(f"[INFO] Downloading model from Hugging Face Hub: {model_id}")

        # Download all files and get the local directory
        model_dir = snapshot_download(
            repo_id=model_id, repo_type="model", local_dir_use_symlinks=False
        )

        print(f"[OK] Model downloaded to {model_dir}")
        return Path(model_dir)

    def _find_model_files(self, model_dir: Path) -> tuple[Path, Path, Path]:
        """
        Find model files in directory, handling different naming conventions

        Returns:
            Tuple of (webnn_file, weights_file, manifest_file)
        """
        # Try different naming conventions (Hub uses "model" naming)
        webnn_candidates = [
            model_dir / "model.webnn",
            model_dir / "all-MiniLM-L6-v2.webnn",
        ]
        weights_candidates = [
            model_dir / "model.weights",
            model_dir / "all-MiniLM-L6-v2.weights",
        ]
        manifest_candidates = [
            model_dir / "manifest.json",
            model_dir / "all-MiniLM-L6-v2.manifest.json",
        ]

        # Find existing files
        webnn_file = next((f for f in webnn_candidates if f.exists()), None)
        weights_file = next((f for f in weights_candidates if f.exists()), None)
        manifest_file = next((f for f in manifest_candidates if f.exists()), None)

        if not webnn_file:
            raise FileNotFoundError(
                f"WebNN file not found. Tried: {[str(f) for f in webnn_candidates]}"
            )
        if not weights_file:
            raise FileNotFoundError(
                f"Weights file not found. Tried: {[str(f) for f in weights_candidates]}"
            )
        if not manifest_file:
            raise FileNotFoundError(
                f"Manifest file not found. Tried: {[str(f) for f in manifest_candidates]}"
            )

        return webnn_file, weights_file, manifest_file

    def _load_webnn_model(self):
        """Load the WebNN graph using Hub and MLGraph.load()"""
        # Create WebNN context
        ml = webnn.ML()

        # Use CPU or GPU based on device_type
        if self.device_type == "cpu":
            context = ml.create_context(power_preference="default", accelerated=False)
        else:
            context = ml.create_context(power_preference="high-performance", accelerated=True)

        print(f"[INFO] Created WebNN context (accelerated={context.accelerated})")

        # Check if model_id is a local path or hub identifier
        model_path = Path(self.model_id)
        if model_path.exists() and model_path.is_dir():
            # Local path - find files manually
            print(f"[INFO] Using local model directory: {model_path}")
            webnn_file, weights_file, manifest_file = self._find_model_files(model_path)
            model_files = {
                'graph': str(webnn_file),
                'weights': str(weights_file),
                'manifest': str(manifest_file)
            }
        else:
            # Download from Hugging Face Hub
            print(f"[INFO] Downloading model from Hugging Face Hub: {self.model_id}")
            hub = webnn.Hub()
            model_files = hub.download_model(self.model_id)
            print(f"[OK] Model downloaded")

        # Load graph from files
        print(f"[INFO] Loading graph...")
        graph = webnn.MLGraph.load(
            model_files['graph'],
            manifest_path=model_files['manifest'],
            weights_path=model_files['weights']
        )
        print(f"[OK] Graph loaded successfully")
        print(f"    - Operand count: {graph.operand_count}")
        print(f"    - Operation count: {graph.operation_count}")
        print(f"    - Inputs: {graph.get_input_names()}")
        print(f"    - Outputs: {graph.get_output_names()}")

        return {
            "context": context,
            "graph": graph,
        }

    def _mean_pooling(
        self, token_embeddings: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply mean pooling to token embeddings using attention mask

        Args:
            token_embeddings: Shape [batch, seq_len, hidden_dim]
            attention_mask: Shape [batch, seq_len]

        Returns:
            Sentence embeddings: Shape [batch, hidden_dim]
        """
        # Expand attention mask to match embeddings dimensions
        mask_expanded = np.expand_dims(attention_mask, axis=-1)

        # Multiply embeddings by mask and sum
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)

        # Sum of mask (number of valid tokens per sentence)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)

        # Calculate mean
        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length (L2 norm)"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings using WebNN

        This method demonstrates the complete WebNN inference pipeline:
        1. Tokenize input texts using transformers tokenizer
        2. Load WebNN graph from Hub or local files (in __init__)
        3. Prepare inputs as NumPy arrays
        4. Execute inference using context.compute(graph, inputs)
        5. Apply mean pooling to token embeddings
        6. Normalize to unit length

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings to unit length

        Returns:
            Embeddings array of shape [num_texts, 384]

        Note:
            Currently returns placeholder embeddings due to ONNX converter issue
            with the expand operation. The WebNN graph loading and API integration
            work correctly. See inline TODO for converter fix needed.
        """
        # The WebNN-exported graph uses static shapes [1, 128] for inputs,
        # so we run each sentence independently with fixed-length padding.
        context = self.graph["context"]
        graph = self.graph["graph"]
        output_names = graph.get_output_names()
        if len(output_names) == 0:
            raise RuntimeError("Graph has no outputs")
        output_name = output_names[0]

        embeddings = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            token_type_ids = encoded.get(
                "token_type_ids", np.zeros_like(input_ids)
            ).astype(np.int64)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            results = context.compute(graph, inputs)
            token_embeddings = results[output_name]

            sent_embedding = self._mean_pooling(
                token_embeddings, attention_mask.astype(np.float32)
            )
            embeddings.append(sent_embedding[0])

        embeddings = np.stack(embeddings, axis=0)

        if normalize:
            embeddings = self._normalize(embeddings)

        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1
        """
        embeddings = self.encode([text1, text2], normalize=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)


def compare_embeddings(
    emb1: np.ndarray, emb2: np.ndarray, label1: str = "Model 1", label2: str = "Model 2"
):
    """
    Compare two sets of embeddings and print statistics

    Args:
        emb1: First embeddings [batch, dim]
        emb2: Second embeddings [batch, dim]
        label1: Name of first model
        label2: Name of second model
    """
    print("\n" + "=" * 70)
    print(f"Comparing {label1} vs {label2}")
    print("=" * 70)

    # Compute metrics for each pair
    for i in range(len(emb1)):
        print(f"\nSentence {i + 1}:")

        # Cosine similarity
        cosine_sim = np.dot(emb1[i], emb2[i])
        print(f"  Cosine Similarity: {cosine_sim:.6f}")

        # Euclidean distance
        euclidean_dist = np.linalg.norm(emb1[i] - emb2[i])
        print(f"  Euclidean Distance: {euclidean_dist:.6f}")

        # Mean squared error
        mse = np.mean((emb1[i] - emb2[i]) ** 2)
        print(f"  Mean Squared Error: {mse:.8f}")

        # Mean absolute error
        mae = np.mean(np.abs(emb1[i] - emb2[i]))
        print(f"  Mean Absolute Error: {mae:.8f}")

    # Overall statistics
    print("\n" + "-" * 70)
    print("Overall Statistics:")
    print("-" * 70)

    all_cosine_sims = [np.dot(emb1[i], emb2[i]) for i in range(len(emb1))]
    print(f"  Average Cosine Similarity: {np.mean(all_cosine_sims):.6f}")
    print(f"  Min Cosine Similarity: {np.min(all_cosine_sims):.6f}")
    print(f"  Max Cosine Similarity: {np.max(all_cosine_sims):.6f}")

    all_euclidean = [np.linalg.norm(emb1[i] - emb2[i]) for i in range(len(emb1))]
    print(f"  Average Euclidean Distance: {np.mean(all_euclidean):.6f}")

    all_mse = [np.mean((emb1[i] - emb2[i]) ** 2) for i in range(len(emb1))]
    print(f"  Average MSE: {np.mean(all_mse):.8f}")

    # Check if embeddings are close
    if np.mean(all_cosine_sims) > 0.99:
        print("\n[OK] Embeddings are VERY SIMILAR (cosine > 0.99)")
    elif np.mean(all_cosine_sims) > 0.95:
        print("\n[OK] Embeddings are SIMILAR (cosine > 0.95)")
    elif np.mean(all_cosine_sims) > 0.80:
        print("\n[WARNING] Embeddings are SOMEWHAT SIMILAR (cosine > 0.80)")
    else:
        print("\n[ERROR] Embeddings are DIFFERENT (cosine < 0.80)")


def main():
    """Demo: Generate embeddings and compare implementations"""
    import argparse

    parser = argparse.ArgumentParser(
        description="all-MiniLM-L6-v2 text embeddings demo"
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=(
            "Local model directory or Hugging Face Hub model id "
            f"(default: {MODEL_ID})"
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("all-MiniLM-L6-v2 Text Embeddings Demo")
    print("Comparing Transformers vs WebNN implementations")
    print("=" * 70)
    print()

    # Test sentences
    test_sentences = [
        "This is a sample sentence to encode",
        "The cat sits on the mat",
        "A feline rests on the carpet",
        "The weather is sunny today",
        "Python is a programming language",
    ]

    # Initialize transformers embedder (reference implementation)
    print("-" * 70)
    print("Initializing Transformers (Reference) Implementation")
    print("-" * 70)
    try:
        transformers_embedder = TransformersEmbedder()
    except Exception as e:
        print(f"[ERROR] Failed to initialize transformers embedder: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Generate embeddings with transformers
    print("\n" + "-" * 70)
    print("Encoding with Transformers")
    print("-" * 70)
    print(f"Encoding {len(test_sentences)} sentences...")

    transformers_embeddings = transformers_embedder.encode(
        test_sentences, normalize=True
    )
    print(f"[OK] Generated {len(transformers_embeddings)} embeddings")
    print(f"[OK] Shape: {transformers_embeddings.shape}")
    print(f"[OK] First embedding (first 10 dims): {transformers_embeddings[0][:10]}")
    print(f"[OK] L2 norm: {np.linalg.norm(transformers_embeddings[0]):.6f}")

    # Show semantic similarity using transformers
    print("\n" + "-" * 70)
    print("Semantic Similarity (Transformers)")
    print("-" * 70)
    print("Cosine similarity matrix:")
    print("     ", end="")
    for i in range(len(test_sentences)):
        print(f"  S{i+1}  ", end="")
    print()

    for i in range(len(test_sentences)):
        print(f"S{i+1}  ", end="")
        for j in range(len(test_sentences)):
            similarity = np.dot(transformers_embeddings[i], transformers_embeddings[j])
            print(f"{similarity:6.3f} ", end="")
        print()

    print("\nSentences:")
    for i, sent in enumerate(test_sentences, 1):
        print(f"  S{i}. {sent}")

    # Initialize WebNN embedder (if available)
    print("\n" + "-" * 70)
    print("Initializing WebNN Implementation")
    print("-" * 70)
    try:
        # Use local compliant model
        webnn_embedder = WebNNEmbedder(model_id=args.model_id, device_type="cpu")

        # Generate embeddings with WebNN
        print("\n" + "-" * 70)
        print("Encoding with WebNN")
        print("-" * 70)
        print(f"Encoding {len(test_sentences)} sentences...")

        webnn_embeddings = webnn_embedder.encode(test_sentences, normalize=True)
        print(f"[OK] Generated {len(webnn_embeddings)} embeddings")
        print(f"[OK] Shape: {webnn_embeddings.shape}")
        print(f"[OK] First embedding (first 10 dims): {webnn_embeddings[0][:10]}")
        print(f"[OK] L2 norm: {np.linalg.norm(webnn_embeddings[0]):.6f}")

        # Compare embeddings
        compare_embeddings(
            transformers_embeddings,
            webnn_embeddings,
            "Transformers (Reference)",
            "WebNN",
        )

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            print(f"[WARNING] Model not found on Hugging Face Hub")
            print(
                "[INFO] The model may not be uploaded yet to: tarekziade/all-MiniLM-L6-v2-webnn"
            )

            # Try local path as fallback
            local_path = LOCAL_MODEL_DIR
            if local_path.exists():
                print(f"[INFO] Trying local model at: {local_path}")
                try:
                    webnn_embedder = WebNNEmbedder(
                        model_id=str(local_path), device_type="cpu"
                    )

                    # Generate embeddings with WebNN
                    print("\n" + "-" * 70)
                    print("Encoding with WebNN")
                    print("-" * 70)
                    print(f"Encoding {len(test_sentences)} sentences...")

                    webnn_embeddings = webnn_embedder.encode(
                        test_sentences, normalize=True
                    )
                    print(f"[OK] Generated {len(webnn_embeddings)} embeddings")
                    print(f"[OK] Shape: {webnn_embeddings.shape}")
                    print(
                        f"[OK] First embedding (first 10 dims): {webnn_embeddings[0][:10]}"
                    )
                    print(f"[OK] L2 norm: {np.linalg.norm(webnn_embeddings[0]):.6f}")

                    # Compare embeddings
                    compare_embeddings(
                        transformers_embeddings,
                        webnn_embeddings,
                        "Transformers (Reference)",
                        "WebNN",
                    )
                except Exception as local_e:
                    print(f"[WARNING] Local model also failed: {local_e}")
                    print("[INFO] Skipping WebNN comparison")
            else:
                print("[INFO] Local model not found either")
                print("[INFO] Skipping WebNN comparison")
        else:
            print(f"[WARNING] WebNN initialization failed: {e}")
            print("[INFO] Skipping WebNN comparison")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("[INFO] Demo completed")
    print("[NOTE] The transformers implementation provides the reference embeddings")
    print("[NOTE] Once WebNN inference is implemented, we can validate it produces")
    print("[NOTE] identical results (cosine similarity > 0.99)")
    print("=" * 70)


if __name__ == "__main__":
    main()
