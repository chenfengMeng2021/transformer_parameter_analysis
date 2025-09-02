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

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  CUDA not available, using CPU")


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


def compute_matrix_stats(tensor: np.ndarray, zero_epsilon: float = 1e-8, use_gpu: bool = True) -> Dict[str, Any]:
    """Compute comprehensive matrix statistics including rank analysis and singular value analysis."""
    # Handle 1D tensors (like layer norm weights)
    if tensor.ndim == 1:
        x2d = tensor.reshape(1, -1)
    elif tensor.ndim > 2:
        x2d = tensor.reshape(tensor.shape[0], -1)
    else:
        x2d = tensor
    
    # Design rank (theoretical minimum rank)
    design_rank = min(x2d.shape)
    
    # Actual rank and singular value analysis
    actual_rank = 0
    svd_90_percent_rank = 0
    
    try:
        # Use GPU if available and requested
        if use_gpu and CUDA_AVAILABLE:
            # Convert to torch tensor and move to GPU
            x2d_torch = torch.tensor(x2d, dtype=torch.float32, device='cuda')
            
            # Use torch SVD for GPU acceleration
            try:
                U, S, V = torch.linalg.svd(x2d_torch, full_matrices=False)
                s = S.cpu().numpy()
                print(f"  🚀 GPU SVD completed for matrix {tensor.shape}")
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  GPU OOM, falling back to CPU for matrix {tensor.shape}")
                # Fall back to CPU with sampling for very large matrices
                max_dim = min(2000, min(x2d.shape))
                if x2d.shape[0] > max_dim or x2d.shape[1] > max_dim:
                    x2d_sample = x2d[:max_dim, :max_dim]
                    print(f"    Using CPU sample {x2d_sample.shape} for SVD")
                    _, s, _ = np.linalg.svd(x2d_sample, full_matrices=False)
                else:
                    _, s, _ = np.linalg.svd(x2d, full_matrices=False)
        else:
            # CPU SVD with larger matrix support
            max_dim = min(2000, min(x2d.shape))
            if x2d.shape[0] > max_dim or x2d.shape[1] > max_dim:
                x2d_sample = x2d[:max_dim, :max_dim]
                print(f"  Matrix large {tensor.shape}, using CPU sample {x2d_sample.shape} for SVD")
                _, s, _ = np.linalg.svd(x2d_sample, full_matrices=False)
            else:
                _, s, _ = np.linalg.svd(x2d, full_matrices=False)
        
        # Actual rank (numerical rank)
        actual_rank = int((s > 1e-6 * s.max()).sum()) if s.size else 0
        
        # Calculate how many singular values represent 90% of variance
        if s.size > 0:
            # Calculate total variance (sum of squared singular values)
            total_variance = np.sum(s ** 2)
            cumulative_variance = np.cumsum(s ** 2)
            
            # Find the number of singular values needed for 90% variance
            svd_90_percent_rank = np.searchsorted(cumulative_variance, 0.9 * total_variance) + 1
            svd_90_percent_rank = min(svd_90_percent_rank, s.size)
            
    except Exception as e:
        print(f"  SVD failed: {e}")
        actual_rank = 0
        svd_90_percent_rank = 0
    
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
        "svd_90_percent_rank": int(svd_90_percent_rank),
        "svd_90_percent_ratio": float(svd_90_percent_rank / design_rank) if design_rank > 0 else 0.0,
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


def format_layer_analysis(results: List[Dict[str, Any]]) -> None:
    """Format and display layer-by-layer matrix analysis."""
    print("\n" + "="*80)
    print("LAYER-BY-LAYER MATRIX ANALYSIS")
    print("="*80)
    
    # Group results by layer
    layer_groups = {}
    for row in results:
        layer_idx = row["layer_index"]
        if layer_idx != "":
            if layer_idx not in layer_groups:
                layer_groups[layer_idx] = []
            layer_groups[layer_idx].append(row)
    
    # Sort layers numerically
    sorted_layers = sorted(layer_groups.keys(), key=lambda x: int(x) if x != "" else -1)
    
    for layer_idx in sorted_layers:
        if layer_idx == "":
            continue
            
        layer_matrices = layer_groups[layer_idx]
        print(f"\n🔹 LAYER {layer_idx}")
        print("-" * 50)
        
        # Sort matrices by type for consistent display
        type_order = ["q_proj", "k_proj", "v_proj", "o_proj", "moe_gate", "moe_expert", "norm"]
        sorted_matrices = sorted(layer_matrices, key=lambda x: type_order.index(x["block_type"]) if x["block_type"] in type_order else 999)
        
        for matrix in sorted_matrices:
            matrix_type = matrix["block_type"]
            shape = matrix["shape"]
            rows = matrix["rows"]
            cols = matrix["cols"]
            design_rank = matrix["design_rank"]
            actual_rank = matrix["actual_rank"]
            svd_90_rank = matrix["svd_90_percent_rank"]
            svd_90_ratio = matrix["svd_90_percent_ratio"]
            sparsity = matrix["sparsity"]
            mean = matrix["mean"]
            std = matrix["std"]
            
            print(f"  📊 {matrix_type:12} | Shape: {shape:15} | Rank: {actual_rank:3}/{design_rank:3} ({actual_rank/design_rank*100:5.1f}%)")
            print(f"      SVD 90%: {svd_90_rank:3}/{design_rank:3} ({svd_90_ratio*100:5.1f}%) | Sparsity: {sparsity*100:5.1f}% | μ={mean:8.4f} σ={std:8.4f}")
    
    # Show non-layer matrices (embedding, output, etc.)
    non_layer_matrices = [row for row in results if row["layer_index"] == ""]
    if non_layer_matrices:
        print(f"\n🔹 NON-LAYER MATRICES")
        print("-" * 50)
        
        for matrix in non_layer_matrices:
            matrix_type = matrix["block_type"]
            shape = matrix["shape"]
            rows = matrix["rows"]
            cols = matrix["cols"]
            design_rank = matrix["design_rank"]
            actual_rank = matrix["actual_rank"]
            svd_90_rank = matrix["svd_90_percent_rank"]
            svd_90_ratio = matrix["svd_90_percent_ratio"]
            sparsity = matrix["sparsity"]
            mean = matrix["mean"]
            std = matrix["std"]
            
            print(f"  📊 {matrix_type:12} | Shape: {shape:15} | Rank: {actual_rank:3}/{design_rank:3} ({actual_rank/design_rank*100:5.1f}%)")
            print(f"      SVD 90%: {svd_90_rank:3}/{design_rank:3} ({svd_90_ratio*100:5.1f}%) | Sparsity: {sparsity*100:5.1f}% | μ={mean:8.4f} σ={std:8.4f}")


def analyze_transformer_model(model_path: str, out_csv: str, model_id: str | None = None, 
                           revision: str | None = None, use_gpu: bool = True) -> None:
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
        stats = compute_matrix_stats(tensor, use_gpu=use_gpu)
        
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
            "svd_90_percent_rank": stats["svd_90_percent_rank"],
            "svd_90_percent_ratio": stats["svd_90_percent_ratio"],
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
        # Sort rows by numeric layer index (ascending). Non-layer rows ("") go last.
        def _layer_sort_key(row: Dict[str, Any]):
            layer_val = row["layer_index"]
            is_non_layer = (layer_val == "")
            if is_non_layer:
                return (True, 10**9, row["param_name"])  # Non-layer rows last
            numeric_layer = int(layer_val)
            # Sort layers by numeric order (1, 2, 3, ...)
            return (False, numeric_layer, row["param_name"])

        results = sorted(results, key=_layer_sort_key)

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
        
        # Format and display layer-by-layer analysis
        format_layer_analysis(results)
    else:
        print("No tensors found to analyze")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze all transformer matrices and output comprehensive stats to CSV")
    parser.add_argument("--model_path", required=True, help="Path to local model directory")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--model_id", default="Qwen/Qwen3-4B", help="Model ID for metadata")
    parser.add_argument("--revision", default=None, help="Revision for metadata")
    # rank_k removed: always compute full numerical rank
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    analyze_transformer_model(args.model_path, args.out, args.model_id, args.revision, use_gpu)


if __name__ == "__main__":
    main()
