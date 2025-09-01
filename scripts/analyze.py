#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def convert_tensor_dtype(tensor):
    """Convert tensor to numpy-compatible dtype."""
    if isinstance(tensor, torch.Tensor):
        # Convert PyTorch tensor to numpy
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        tensor = tensor.detach().cpu().numpy()
    
    # Handle numpy arrays
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.dtype('bfloat16') or str(tensor.dtype) == 'bfloat16':
            tensor = tensor.astype(np.float32)
        elif tensor.dtype not in [np.float16, np.float32, np.float64, np.int32, np.int64]:
            tensor = tensor.astype(np.float32)
    
    return tensor


def load_embedding_tensor(model_dir: Path):
    possible_keys = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embed_tokens.weight",
        "model.wte.weight",
        "tok_embeddings.weight",
        "embeddings.word_embeddings.weight",
    ]
    
    # Check index file first
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            
            # Find embedding key and its shard
            for key in possible_keys:
                if key in weight_map:
                    shard_file = weight_map[key]
                    shard_path = model_dir / shard_file
                    if shard_path.exists():
                        print(f"Found embedding tensor '{key}' in shard: {shard_file}")
                        return key, load_tensor_from_shard(shard_path, key)
        except Exception as e:
            print(f"Error reading index: {e}")
    
    # Fallback: search all shards
    shard_files = [f for f in model_dir.glob("*.safetensors") if not f.name.startswith("._")]
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    
    for shard in shard_files:
        try:
            with safe_open(str(shard), framework="pt") as f:
                for key in possible_keys:
                    if key in f.keys():
                        print(f"Found embedding tensor '{key}' in shard: {shard.name}")
                        return key, f.get_tensor(key)
                # Heuristic fallback
                for k in f.keys():
                    lk = k.lower()
                    if "embed" in lk and "weight" in lk:
                        print(f"Found embedding tensor '{k}' in shard: {shard.name}")
                        return k, f.get_tensor(k)
        except Exception as e:
            print(f"Error reading shard {shard}: {e}")
            continue
    
    raise KeyError("Embedding weight tensor not found. Update key list for this model.")


def load_tensor_from_shard(shard_path: Path, tensor_key: str):
    """Load specific tensor from shard using PyTorch framework."""
    with safe_open(str(shard_path), framework="pt") as f:
        tensor = f.get_tensor(tensor_key)
        return convert_tensor_dtype(tensor)


def tensor_stats(x, rank_k: int = 128, zero_epsilon: float = 1e-8) -> dict:
    # Ensure we have a numpy array
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = x.detach().cpu().numpy()
    
    x2d = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    
    # Limit SVD computation for very large matrices
    max_dim = min(1000, min(x2d.shape))
    if x2d.shape[0] > max_dim or x2d.shape[1] > max_dim:
        x2d_sample = x2d[:max_dim, :max_dim]
        print(f"Matrix too large ({x.shape}), using sample ({x2d_sample.shape}) for SVD")
    else:
        x2d_sample = x2d
    
    try:
        _, s, _ = np.linalg.svd(x2d_sample, full_matrices=False)
        r = int((s > 1e-6 * s.max()).sum()) if s.size else 0
    except Exception as e:
        print(f"SVD failed: {e}")
        r = 0
    
    num_total = x.size
    num_near_zero = int((np.abs(x) <= zero_epsilon).sum()) if num_total else 0
    sparsity = float(num_near_zero) / num_total if num_total else 0.0
    
    return {
        "shape": str(tuple(x.shape)),
        "rows": int(x2d.shape[0]),
        "cols": int(x2d.shape[1]),
        "rank": int(min(r, rank_k)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "var": float(x.var()),
        "sparsity": float(sparsity),
        "dtype": str(x.dtype),
    }


def analyze_embedding(model_path: str, out_csv: str, model_id: str | None = None, revision: str | None = None) -> None:
    model_dir = Path(model_path)
    print(f"Analyzing model at: {model_dir}")
    
    key, tensor = load_embedding_tensor(model_dir)
    print(f"Embedding tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    stats = tensor_stats(tensor)
    print(f"Computed stats: rank={stats['rank']}, sparsity={stats['sparsity']:.4f}")

    created_at = datetime.utcnow().isoformat()
    header = [
        "model_path",
        "model_id",
        "revision",
        "param_name",
        "layer_index",
        "block_type",
        "shape",
        "rows",
        "cols",
        "rank",
        "mean",
        "std",
        "var",
        "sparsity",
        "dtype",
        "created_at",
    ]
    row = [
        str(model_dir),
        model_id or "",
        revision or "",
        key,
        "",
        "embedding",
        stats["shape"],
        stats["rows"],
        stats["cols"],
        stats["rank"],
        stats["mean"],
        stats["std"],
        stats["var"],
        stats["sparsity"],
        stats["dtype"],
        created_at,
    ]

    Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    
    print(f"Results saved to: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embedding stats to CSV from a model directory with safetensors")
    parser.add_argument("--model_path", required=True, help="Path to local model directory (snapshot root)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--model_id", default="Qwen/Qwen3-4B", help="Optional model id for metadata")
    parser.add_argument("--revision", default=None, help="Optional revision for metadata")
    args = parser.parse_args()

    analyze_embedding(args.model_path, args.out, args.model_id, args.revision)


if __name__ == "__main__":
    main()
