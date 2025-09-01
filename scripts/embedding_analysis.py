#!/usr/bin/env python3

"""
Embedding Analysis Script for Qwen3-4B
Part 1: Load embedding tensors
Part 2: K-means clustering with elbow method
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def convert_tensor_dtype(tensor):
    """Convert tensor to numpy-compatible dtype."""
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        tensor = tensor.detach().cpu().numpy()
    
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.dtype('bfloat16') or str(tensor.dtype) == 'bfloat16':
            tensor = tensor.astype(np.float32)
        elif tensor.dtype not in [np.float16, np.float32, np.float64, np.int32, np.int64]:
            tensor = tensor.astype(np.float32)
    
    return tensor


def load_embedding_tensor(model_dir: Path) -> Tuple[str, np.ndarray]:
    """Load embedding tensor from model directory."""
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
                for k in f.keys():
                    lk = k.lower()
                    if "embed" in lk and "weight" in lk:
                        print(f"Found embedding tensor '{k}' in shard: {shard.name}")
                        return k, f.get_tensor(k)
        except Exception as e:
            print(f"Error reading shard {shard}: {e}")
            continue
    
    raise KeyError("Embedding weight tensor not found.")


def load_tensor_from_shard(shard_path: Path, tensor_key: str) -> np.ndarray:
    """Load specific tensor from shard using PyTorch framework."""
    with safe_open(str(shard_path), framework="pt") as f:
        tensor = f.get_tensor(tensor_key)
        return convert_tensor_dtype(tensor)


# ============================================================================
# PART 1: LOADING EMBEDDING TENSORS
# ============================================================================

def load_embeddings(model_path: str) -> Tuple[str, np.ndarray]:
    """Load and prepare embedding tensors for analysis."""
    print("=" * 60)
    print("PART 1: LOADING EMBEDDING TENSORS")
    print("=" * 60)
    
    model_dir = Path(model_path)
    print(f"Loading embeddings from: {model_dir}")
    
    key, tensor = load_embedding_tensor(model_dir)
    print(f"Embedding tensor: {key}")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    
    # Ensure tensor is numpy array with compatible dtype
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        tensor = tensor.detach().cpu().numpy()
    
    print(f"Converted to numpy array with dtype: {tensor.dtype}")
    
    # Normalize embeddings for better clustering
    scaler = StandardScaler()
    tensor_normalized = scaler.fit_transform(tensor)
    print(f"Normalized embeddings shape: {tensor_normalized.shape}")
    
    return key, tensor_normalized


# ============================================================================
# PART 2: K-MEANS CLUSTERING WITH ELBOW METHOD
# ============================================================================

def find_optimal_clusters(embeddings: np.ndarray, max_k: int = 2000, random_state: int = 42) -> Tuple[int, list]:
    """Find optimal number of clusters using two-stage elbow method."""
    print("\n" + "=" * 60)
    print("PART 2: FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 60)
    
    # Stage 1: Coarse search with large step size
    print("Stage 1: Coarse search with step size 100...")
    coarse_step = 100
    coarse_k_values = list(range(coarse_step, max_k + 1, coarse_step))
    if 1 not in coarse_k_values:
        coarse_k_values = [1] + coarse_k_values
    
    print(f"Testing K values: {coarse_k_values}")
    
    coarse_sse_values = []
    for k in coarse_k_values:
        print(f"  Testing K={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(embeddings)
        sse = kmeans.inertia_
        coarse_sse_values.append(sse)
        print(f"SSE: {sse:.2e}")
    
    # Find rough elbow point from coarse search
    rough_elbow_k = find_elbow_point(coarse_k_values, coarse_sse_values)
    print(f"\nRough elbow point found at K={rough_elbow_k}")
    
    # Stage 2: Fine search around rough elbow point
    print(f"\nStage 2: Fine search around K={rough_elbow_k}...")
    fine_range = max(50, rough_elbow_k // 4)  # Search range around rough elbow
    fine_start = max(1, rough_elbow_k - fine_range)
    fine_end = min(max_k, rough_elbow_k + fine_range)
    
    fine_k_values = list(range(fine_start, fine_end + 1, 10))  # Step size 10 for fine search
    if fine_start > 1:
        fine_k_values = [1] + fine_k_values
    if fine_end not in fine_k_values:
        fine_k_values.append(fine_end)
    
    print(f"Fine search range: K from {fine_start} to {fine_end}")
    print(f"Fine search K values: {fine_k_values}")
    
    fine_sse_values = []
    for k in fine_k_values:
        print(f"  Testing K={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(embeddings)
        sse = kmeans.inertia_
        fine_sse_values.append(sse)
        print(f"SSE: {sse:.2e}")
    
    # Find optimal K from fine search
    optimal_k = find_elbow_point(fine_k_values, fine_sse_values)
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Combine results for plotting
    all_k_values = coarse_k_values + [k for k in fine_k_values if k not in coarse_k_values]
    all_sse_values = coarse_sse_values + [fine_sse_values[fine_k_values.index(k)] for k in fine_k_values if k not in coarse_k_values]
    
    # Sort by K values
    sorted_indices = np.argsort(all_k_values)
    all_k_values = [all_k_values[i] for i in sorted_indices]
    all_sse_values = [all_sse_values[i] for i in sorted_indices]
    
    return optimal_k, all_sse_values, all_k_values


def find_elbow_point(k_values: list, sse_values: list) -> int:
    """Find elbow point using second derivative method."""
    if len(sse_values) < 3:
        return k_values[0]
    
    # Calculate second derivative
    second_derivatives = []
    for i in range(1, len(sse_values) - 1):
        d2 = sse_values[i+1] - 2*sse_values[i] + sse_values[i-1]
        second_derivatives.append(d2)
    
    # Find the point with maximum second derivative (sharpest bend)
    elbow_idx = np.argmax(second_derivatives) + 1
    return k_values[elbow_idx]


def plot_elbow_curve(k_values: list, sse_values: list, optimal_k: int, output_path: str):
    """Plot SSE vs K curve with elbow point highlighted."""
    plt.figure(figsize=(12, 8))
    
    # Plot SSE curve
    plt.plot(k_values, sse_values, 'bo-', linewidth=2, markersize=6, label='SSE', alpha=0.7)
    
    # Highlight optimal K point
    if optimal_k in k_values:
        optimal_idx = k_values.index(optimal_k)
        plt.plot(optimal_k, sse_values[optimal_idx], 'ro', markersize=12, label=f'Optimal K={optimal_k}')
        
        # Add text annotation for optimal K
        plt.annotate(f'Optimal K = {optimal_k}', 
                    xy=(optimal_k, sse_values[optimal_idx]),
                    xytext=(optimal_k * 1.1, sse_values[optimal_idx] * 1.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red')
    
    # Add annotations
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.title('Two-Stage Elbow Method for Optimal K Selection', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale for better visualization of large K values
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Elbow curve saved to: {output_path}")
    plt.show()


def perform_final_clustering(embeddings: np.ndarray, optimal_k: int, random_state: int = 42):
    """Perform final clustering with optimal K and show results."""
    print(f"\nPerforming final clustering with K={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    print(f"Clustering completed!")
    print(f"Cluster sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} tokens ({count/len(embeddings)*100:.1f}%)")
    
    return cluster_labels, kmeans


def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings with K-means clustering and elbow method")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--output_dir", default="data/outputs", help="Output directory for plots")
    parser.add_argument("--max_k", type=int, default=2000, help="Maximum K to test (coarse search)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # PART 1: Load embeddings
        key, embeddings = load_embeddings(args.model_path)
        
        # PART 2: Find optimal clusters and plot
        optimal_k, sse_values, k_values = find_optimal_clusters(embeddings, args.max_k, args.random_state)
        
        # Plot elbow curve
        plot_path = output_dir / "elbow_curve.png"
        plot_elbow_curve(k_values, sse_values, optimal_k, str(plot_path))
        
        # Perform final clustering
        cluster_labels, kmeans = perform_final_clustering(embeddings, optimal_k, args.random_state)
        
        # Save clustering results
        results = {
            "optimal_k": optimal_k,
            "sse_values": sse_values,
            "k_values": k_values,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_)
        }
        
        results_path = output_dir / "clustering_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Clustering results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
