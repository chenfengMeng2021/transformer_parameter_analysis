#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from safetensors import safe_open


def classify_matrix_type(param_name: str) -> Tuple[str, int, str]:
    """Classify matrix type and extract layer index."""
    name_lower = param_name.lower()
    layer_idx = None
    matrix_type = "other"
    
    # Extract layer index
    if ".layers." in name_lower:
        try:
            layer_idx = int(name_lower.split(".layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            layer_idx = None
    
    # Classify matrix types
    if "embed" in name_lower and "weight" in name_lower:
        matrix_type = "embedding"
    elif ".q_proj" in name_lower and "weight" in name_lower:
        matrix_type = "q_proj"
    elif ".k_proj" in name_lower and "weight" in name_lower:
        matrix_type = "k_proj"
    elif ".v_proj" in name_lower and "weight" in name_lower:
        matrix_type = "v_proj"
    elif ".o_proj" in name_lower and "weight" in name_lower:
        matrix_type = "o_proj"
    elif "moe" in name_lower and "weight" in name_lower:
        if "gate" in name_lower:
            matrix_type = "moe_gate"
        elif "expert" in name_lower:
            matrix_type = "moe_expert"
        else:
            matrix_type = "moe"
    elif "lm_head" in name_lower and "weight" in name_lower:
        matrix_type = "output"
    elif "output" in name_lower and "weight" in name_lower:
        matrix_type = "output"
    elif "norm" in name_lower and "weight" in name_lower:
        matrix_type = "norm"
    
    return matrix_type, layer_idx, param_name


def compute_matrix_stats(tensor: np.ndarray, rank_k: int = 128, zero_epsilon: float = 1e-8) -> Dict[str, Any]:
    """Compute comprehensive matrix statistics including rank analysis."""
    x2d = tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
    
    # Design rank (theoretical minimum rank)
    design_rank = min(x2d.shape)
    
    # Actual rank (numerical rank)
    try:
        # Limit SVD computation for very large matrices
        max_dim = min(1000, min(x2d.shape))
        if x2d.shape[0] > max_dim or x2d.shape[1] > max_dim:
            x2d_sample = x2d[:max_dim, :max_dim]
            print(f"  Matrix too large {tensor.shape}, using sample {x2d_sample.shape} for SVD")
        else:
            x2d_sample = x2d
        
        _, s, _ = np.linalg.svd(x2d_sample, full_matrices=False)
        actual_rank = int((s > 1e-6 * s.max()).sum()) if s.size else 0
        actual_rank = min(actual_rank, rank_k)
    except Exception as e:
        print(f"  SVD failed: {e}")
        actual_rank = 0
    
    # Sparsity
    num_total = tensor.size
    num_near_zero = int((np.abs(tensor) <= zero_epsilon).sum()) if num_total else 0
    sparsity = float(num_near_zero) / num_total if num_total else 0.0
    
    # Basic statistics
    stats = {
        "shape": str(tuple(tensor.shape)),
        "rows": int(x2d.shape[0]),
        "cols": int(x2d.shape[1]),
        "design_rank": int(design_rank),
        "actual_rank": int(actual_rank),
        "rank_ratio": float(actual_rank / design_rank) if design_rank > 0 else 0.0,
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "var": float(tensor.var()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "sparsity": float(sparsity),
        "dtype": str(tensor.dtype),
    }
    
    return stats


def load_model_tensors(model_dir: Path) -> List[Tuple[str, np.ndarray]]:
    """Load all model tensors from safetensors files."""
    tensors = []
    
    # Check index file first
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            
            # Group tensors by shard
            shard_tensors = {}
            for tensor_name, shard_file in weight_map.items():
                if shard_file not in shard_tensors:
                    shard_tensors[shard_file] = []
                shard_tensors[shard_file].append(tensor_name)
            
            # Load each shard
            for shard_file, tensor_names in shard_tensors.items():
                shard_path = model_dir / shard_file
                if shard_path.exists():
                    print(f"Loading shard: {shard_file}")
                    try:
                        with safe_open(str(shard_path), framework="pt") as f:
                            for name in tensor_names:
                                if name in f.keys():
                                    tensor = f.get_tensor(name)
                                    # Convert to numpy immediately to avoid bfloat16 issues
                                    if tensor.dtype == torch.bfloat16:
                                        tensor = tensor.to(torch.float32)
                                    tensor_np = tensor.detach().cpu().numpy()
                                    tensors.append((name, tensor_np))
                    except Exception as e:
                        print(f"Error loading shard {shard_file}: {e}")
        except Exception as e:
            print(f"Error reading index: {e}")
    
    # Fallback: search all shards
    if not tensors:
        shard_files = [f for f in model_dir.glob("*.safetensors") if not f.name.startswith("._")]
        for shard in shard_files:
            print(f"Loading shard: {shard.name}")
            try:
                with safe_open(str(shard), framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Convert to numpy immediately to avoid bfloat16 issues
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float32)
                        tensor_np = tensor.detach().cpu().numpy()
                        tensors.append((key, tensor_np))
            except Exception as e:
                print(f"Error loading shard {shard}: {e}")
    
    return tensors


def analyze_transformer_model(model_path: str, out_csv: str, model_id: str | None = None, 
                           revision: str | None = None, rank_k: int = 128) -> None:
    """Analyze all transformer matrices and output to CSV."""
    model_dir = Path(model_path)
    print(f"Analyzing transformer model at: {model_dir}")
    
    # Load all tensors
    print("Loading model tensors...")
    tensors = load_model_tensors(model_dir)
    print(f"Loaded {len(tensors)} tensors")
    
    # Analyze each tensor
    results = []
    created_at = datetime.utcnow().isoformat()
    
    for param_name, tensor in tensors:
        matrix_type, layer_idx, full_name = classify_matrix_type(param_name)
        
        print(f"Analyzing: {param_name} ({matrix_type}, layer {layer_idx})")
        stats = compute_matrix_stats(tensor, rank_k)
        
        # Create result row
        row = {
            "model_path": str(model_dir),
            "model_id": model_id or "",
            "revision": revision or "",
            "param_name": param_name,
            "layer_index": layer_idx or "",
            "block_type": matrix_type,
            "shape": stats["shape"],
            "rows": stats["rows"],
            "cols": stats["cols"],
            "design_rank": stats["design_rank"],
            "actual_rank": stats["actual_rank"],
            "rank_ratio": stats["rank_ratio"],
            "mean": stats["mean"],
            "std": stats["std"],
            "var": stats["var"],
            "min": stats["min"],
            "max": stats["max"],
            "sparsity": stats["sparsity"],
            "dtype": stats["dtype"],
            "created_at": created_at,
        }
        results.append(row)
    
    # Write to CSV
    if results:
        header = list(results[0].keys())
        Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Analysis complete! Results saved to: {out_csv}")
        print(f"Total matrices analyzed: {len(results)}")
        
        # Summary by matrix type
        type_counts = {}
        for row in results:
            matrix_type = row["block_type"]
            type_counts[matrix_type] = type_counts.get(matrix_type, 0) + 1
        
        print("\nMatrix type summary:")
        for matrix_type, count in sorted(type_counts.items()):
            print(f"  {matrix_type}: {count}")
    else:
        print("No tensors found to analyze")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze all transformer matrices and output comprehensive stats to CSV")
    parser.add_argument("--model_path", required=True, help="Path to local model directory")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--model_id", default="Qwen/Qwen3-4B", help="Model ID for metadata")
    parser.add_argument("--revision", default=None, help="Revision for metadata")
    parser.add_argument("--rank_k", type=int, default=128, help="Maximum rank to compute (default: 128)")
    args = parser.parse_args()

    analyze_transformer_model(args.model_path, args.out, args.model_id, args.revision, args.rank_k)


if __name__ == "__main__":
    main()
