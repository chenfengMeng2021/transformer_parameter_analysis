#!/usr/bin/env python3

"""
Embedding Analysis Script for Qwen3-4B
Part 1: Load embedding tensors
Part 2: K-means clustering with elbow method (GPU accelerated with cuML)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

# GPU-accelerated libraries
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    GPU_AVAILABLE = True
    print("✓ cuML GPU acceleration available")
except ImportError:
    print("⚠ cuML not available, falling back to scikit-learn")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    GPU_AVAILABLE = False

# Fallback to scikit-learn if cuML fails
if not GPU_AVAILABLE:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        print("✓ scikit-learn fallback available")
    except ImportError:
        raise ImportError("Neither cuML nor scikit-learn is available")


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


def get_gpu_memory_info():
    """Get GPU memory usage information."""
    if GPU_AVAILABLE:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'total': info.total / 1024**3,  # GB
                'used': info.used / 1024**3,    # GB
                'free': info.free / 1024**3     # GB
            }
        except:
            return None
    return None


def print_gpu_memory_info():
    """Print current GPU memory usage."""
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU Memory: {gpu_info['used']:.1f}GB / {gpu_info['total']:.1f}GB (Free: {gpu_info['free']:.1f}GB)")


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
    print("Normalizing embeddings...")
    if GPU_AVAILABLE:
        # Use GPU-accelerated scaling
        scaler = cuStandardScaler()
        tensor_normalized = scaler.fit_transform(tensor)
        print("✓ GPU-accelerated normalization completed")
    else:
        # Fallback to CPU
        scaler = StandardScaler()
        tensor_normalized = scaler.fit_transform(tensor)
        print("✓ CPU normalization completed")
    
    print(f"Normalized embeddings shape: {tensor_normalized.shape}")
    
    if GPU_AVAILABLE:
        print_gpu_memory_info()
    
    return key, tensor_normalized


# ============================================================================
# PART 2: K-MEANS CLUSTERING WITH ELBOW METHOD (GPU ACCELERATED)
# ============================================================================

def find_optimal_clusters(embeddings: np.ndarray, max_k: int = 2000, random_state: int = 42) -> Tuple[int, list, list]:
    """Find optimal number of clusters using two-stage elbow method with GPU acceleration."""
    print("\n" + "=" * 60)
    print("PART 2: FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 60)
    
    if GPU_AVAILABLE:
        print("🚀 Using GPU-accelerated K-means clustering")
        print_gpu_memory_info()
    else:
        print("💻 Using CPU-based K-means clustering")
    
    # Stage 1: Coarse search with large step size
    print("Stage 1: Coarse search with step size 100...")
    coarse_step = 100
    coarse_k_values = list(range(coarse_step, max_k + 1, coarse_step))
    if 1 not in coarse_k_values:
        coarse_k_values = [1] + coarse_k_values
    
    print(f"Testing K values: {coarse_k_values}")
    
    coarse_sse_values = []
    stage1_start = time.time()
    
    for i, k in enumerate(coarse_k_values):
        print(f"  Testing K={k}...", end=" ")
        start_time = time.time()
        
        if GPU_AVAILABLE:
            # GPU-accelerated K-means
            kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(embeddings)
            sse = kmeans.inertia_
        else:
            # CPU fallback
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(embeddings)
            sse = kmeans.inertia_
        
        elapsed = time.time() - start_time
        coarse_sse_values.append(sse)
        print(f"SSE: {sse:.2e} (Time: {elapsed:.2f}s)")
        
        # Progress update
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{len(coarse_k_values)} completed")
            if GPU_AVAILABLE:
                print_gpu_memory_info()
    
    stage1_time = time.time() - stage1_start
    print(f"\nStage 1 completed in {stage1_time:.2f}s")
    
    # Find rough elbow point from coarse search
    rough_elbow_k = find_elbow_point(coarse_k_values, coarse_sse_values)
    print(f"Rough elbow point found at K={rough_elbow_k}")
    
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
    stage2_start = time.time()
    
    for i, k in enumerate(fine_k_values):
        print(f"  Testing K={k}...", end=" ")
        start_time = time.time()
        
        if GPU_AVAILABLE:
            # GPU-accelerated K-means
            kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(embeddings)
            sse = kmeans.inertia_
        else:
            # CPU fallback
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(embeddings)
            sse = kmeans.inertia_
        
        elapsed = time.time() - start_time
        fine_sse_values.append(sse)
        print(f"SSE: {sse:.2e} (Time: {elapsed:.2f}s)")
        
        # Progress update
        if (i + 1) % 3 == 0:
            print(f"    Progress: {i + 1}/{len(fine_k_values)} completed")
            if GPU_AVAILABLE:
                print_gpu_memory_info()
    
    stage2_time = time.time() - stage2_start
    print(f"\nStage 2 completed in {stage2_time:.2f}s")
    print(f"Total clustering time: {stage1_time + stage2_time:.2f}s")
    
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
    title = 'Two-Stage Elbow Method for Optimal K Selection'
    if GPU_AVAILABLE:
        title += ' (GPU Accelerated)'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use linear scale for better visualization
    # plt.xscale('log')  # Commented out to use linear scale
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Elbow curve saved to: {output_path}")
    plt.show()


def find_closest_tokens_to_centers(embeddings: np.ndarray, cluster_centers: np.ndarray, n_tokens: int = 10):
    """Find the n closest tokens to each cluster center."""
    print(f"\nFinding {n_tokens} closest tokens to each cluster center...")
    
    closest_tokens = {}
    
    if GPU_AVAILABLE:
        # Use GPU-accelerated nearest neighbors
        from cuml.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_tokens, algorithm='brute')
        nn.fit(embeddings)
        
        for i, center in enumerate(cluster_centers):
            # Find closest tokens to this center
            distances, indices = nn.kneighbors(center.reshape(1, -1))
            closest_tokens[i] = {
                'indices': indices[0].tolist(),
                'distances': distances[0].tolist()
            }
    else:
        # CPU fallback
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_tokens, algorithm='brute')
        nn.fit(embeddings)
        
        for i, center in enumerate(cluster_centers):
            # Find closest tokens to this center
            distances, indices = nn.kneighbors(center.reshape(1, -1))
            closest_tokens[i] = {
                'indices': indices[0].tolist(),
                'distances': distances[0].tolist()
            }
    
    return closest_tokens


def decode_tokens(tokenizer, token_indices, max_length=50):
    """Decode token indices to text using the tokenizer."""
    try:
        # Try to decode individual tokens
        decoded_tokens = []
        for idx in token_indices:
            try:
                # Decode single token
                token_text = tokenizer.decode([idx], skip_special_tokens=True)
                if token_text.strip():  # Only add non-empty tokens
                    decoded_tokens.append(token_text)
            except:
                # If single token decoding fails, try with context
                try:
                    # Create a sequence with the token
                    sequence = [idx] + [tokenizer.eos_token_id] if tokenizer.eos_token_id else [idx]
                    token_text = tokenizer.decode(sequence, skip_special_tokens=True)
                    if token_text.strip():
                        decoded_tokens.append(token_text)
                except:
                    decoded_tokens.append(f"[TOKEN_{idx}]")
        
        # Join tokens and truncate if too long
        result = " ".join(decoded_tokens)
        if len(result) > max_length:
            result = result[:max_length] + "..."
        
        return result
    except Exception as e:
        return f"[DECODE_ERROR: {e}]"


def analyze_cluster_representatives(embeddings: np.ndarray, cluster_centers: np.ndarray, 
                                  cluster_labels: np.ndarray, tokenizer=None, n_tokens: int = 10):
    """Analyze and display representative tokens for each cluster."""
    print(f"\n" + "=" * 60)
    print("CLUSTER REPRESENTATIVE ANALYSIS")
    print("=" * 60)
    
    # Find closest tokens to each center
    closest_tokens = find_closest_tokens_to_centers(embeddings, cluster_centers, n_tokens)
    
    # Get cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    print(f"\nRepresentative tokens for each cluster (top {n_tokens} closest to center):")
    print("-" * 80)
    
    cluster_analysis = {}
    
    for cluster_id in sorted(closest_tokens.keys()):
        cluster_size = cluster_sizes.get(cluster_id, 0)
        size_percentage = cluster_size / len(embeddings) * 100
        
        print(f"\nCluster {cluster_id} (Size: {cluster_size:,} tokens, {size_percentage:.1f}%):")
        
        # Get representative token indices
        token_indices = closest_tokens[cluster_id]['indices']
        distances = closest_tokens[cluster_id]['distances']
        
        # Decode tokens if tokenizer is available
        if tokenizer:
            representative_text = decode_tokens(tokenizer, token_indices)
            print(f"  Representative tokens: {representative_text}")
        else:
            print(f"  Representative token indices: {token_indices[:5]}...")  # Show first 5
        
        print(f"  Average distance to center: {np.mean(distances):.4f}")
        
        # Store analysis results
        cluster_analysis[cluster_id] = {
            'size': cluster_size,
            'percentage': size_percentage,
            'token_indices': token_indices,
            'distances': distances,
            'representative_text': decode_tokens(tokenizer, token_indices) if tokenizer else None
        }
    
    return cluster_analysis


def perform_final_clustering(embeddings: np.ndarray, optimal_k: int, random_state: int = 42, 
                           tokenizer=None, analyze_representatives: bool = True, n_representatives: int = 10):
    """Perform final clustering with optimal K and show results."""
    print(f"\nPerforming final clustering with K={optimal_k}...")
    
    start_time = time.time()
    
    if GPU_AVAILABLE:
        # GPU-accelerated final clustering
        kmeans = cuKMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        print("✓ GPU-accelerated final clustering completed")
    else:
        # CPU fallback
        kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        print("✓ CPU final clustering completed")
    
    elapsed = time.time() - start_time
    print(f"Final clustering completed in {elapsed:.2f}s!")
    
    print(f"Cluster sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} tokens ({count/len(embeddings)*100:.1f}%)")
    
    # Analyze cluster representatives if requested
    cluster_analysis = None
    if analyze_representatives:
        cluster_analysis = analyze_cluster_representatives(
            embeddings, kmeans.cluster_centers_, cluster_labels, 
            tokenizer, n_representatives
        )
    
    return cluster_labels, kmeans, cluster_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings with K-means clustering and elbow method")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--output_dir", default="data/outputs", help="Output directory for plots")
    parser.add_argument("--max_k", type=int, default=2000, help="Maximum K to test (coarse search)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--n_representatives", type=int, default=10, help="Number of representative tokens per cluster")
    parser.add_argument("--skip_representatives", action="store_true", help="Skip cluster representative analysis")
    args = parser.parse_args()
    
    # Override GPU availability if CPU is forced
    global GPU_AVAILABLE
    if args.use_cpu:
        GPU_AVAILABLE = False
        print("⚠ CPU mode forced by user")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # PART 1: Load embeddings
        key, embeddings = load_embeddings(args.model_path)
        
        # Try to load tokenizer for token decoding
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            print("\nAttempting to load tokenizer for token decoding...")
            # Try to load tokenizer from the same model directory
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load tokenizer: {e}")
            print("  Token analysis will show token indices instead of decoded text")
        
        # PART 2: Find optimal clusters and plot
        optimal_k, sse_values, k_values = find_optimal_clusters(embeddings, args.max_k, args.random_state)
        
        # Plot elbow curve
        plot_path = output_dir / "elbow_curve.png"
        plot_elbow_curve(k_values, sse_values, optimal_k, str(plot_path))
        
        # Perform final clustering with representative analysis
        cluster_labels, kmeans, cluster_analysis = perform_final_clustering(
            embeddings, optimal_k, args.random_state, 
            tokenizer=tokenizer,
            analyze_representatives=not args.skip_representatives,
            n_representatives=args.n_representatives
        )
        
        # Save clustering results
        results = {
            "optimal_k": optimal_k,
            "sse_values": sse_values,
            "k_values": k_values,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_),
            "gpu_accelerated": GPU_AVAILABLE,
            "total_tokens": len(embeddings),
            "embedding_dimension": embeddings.shape[1],
            "cluster_analysis": cluster_analysis
        }
        
        results_path = output_dir / "clustering_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Clustering results saved to: {results_path}")
        
        # Save detailed cluster analysis
        if cluster_analysis:
            analysis_path = output_dir / "cluster_representatives.json"
            with open(analysis_path, 'w') as f:
                json.dump(cluster_analysis, f, indent=2)
            print(f"Cluster representatives saved to: {analysis_path}")
        
        # Final GPU memory status
        if GPU_AVAILABLE:
            print_gpu_memory_info()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
