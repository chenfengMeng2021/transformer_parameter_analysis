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

def calculate_anova_f_ratio(embeddings: np.ndarray, cluster_labels: np.ndarray, cluster_centers: np.ndarray):
    """Calculate ANOVA F-ratio for clustering quality assessment."""
    # Remove clusters with only one member
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    valid_clusters = unique_labels[counts > 1]
    
    if len(valid_clusters) < 2:
        return 0.0, 0, 0  # Return 0 F-ratio if not enough valid clusters
    
    # Calculate total mean
    total_mean = np.mean(embeddings, axis=0)
    
    # Calculate between-group sum of squares (SSB)
    ssb = 0
    total_n = 0
    for cluster_id in valid_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_mean = cluster_centers[cluster_id]
        
        # Sum of squared differences between cluster mean and total mean
        diff = cluster_mean - total_mean
        ssb += cluster_size * np.sum(diff ** 2)
        total_n += cluster_size
    
    # Calculate within-group sum of squares (SSW)
    ssw = 0
    for cluster_id in valid_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_data = embeddings[cluster_mask]
        cluster_center = cluster_centers[cluster_id]
        
        # Sum of squared differences within cluster
        for point in cluster_data:
            diff = point - cluster_center
            ssw += np.sum(diff ** 2)
    
    # Calculate degrees of freedom
    df_between = len(valid_clusters) - 1
    df_within = total_n - len(valid_clusters)
    
    # Calculate mean squares
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 0
    
    # Calculate F-ratio
    f_ratio = msb / msw if msw > 0 else 0
    
    return f_ratio, len(valid_clusters), total_n


def calculate_silhouette_score(embeddings: np.ndarray, cluster_labels: np.ndarray, sample_size: int = 10000):
    """Calculate silhouette score for clustering quality assessment."""
    try:
        from sklearn.metrics import silhouette_score
        
        # For large datasets, sample to speed up calculation
        if len(embeddings) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = cluster_labels[indices]
        else:
            sample_embeddings = embeddings
            sample_labels = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(sample_embeddings, sample_labels)
        return silhouette_avg
        
    except ImportError:
        print("⚠ sklearn.metrics not available for silhouette calculation")
        return None
    except Exception as e:
        print(f"⚠ Error calculating silhouette score: {e}")
        return None


def find_optimal_clusters_anova(embeddings: np.ndarray, max_k: int = 2000, random_state: int = 42) -> Tuple[int, list, list, list]:
    """Find optimal number of clusters using ANOVA F-ratio method with GPU acceleration."""
    print("\n" + "=" * 60)
    print("PART 2: FINDING OPTIMAL NUMBER OF CLUSTERS (ANOVA METHOD)")
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
    
    coarse_f_ratios = []
    stage1_start = time.time()
    
    for i, k in enumerate(coarse_k_values):
        print(f"  Testing K={k}...", end=" ")
        start_time = time.time()
        
        if GPU_AVAILABLE:
            # GPU-accelerated K-means
            kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            # CPU fallback
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate ANOVA F-ratio
        f_ratio, valid_clusters, total_points = calculate_anova_f_ratio(
            embeddings, cluster_labels, kmeans.cluster_centers_
        )
        
        elapsed = time.time() - start_time
        coarse_f_ratios.append(f_ratio)
        print(f"F-ratio: {f_ratio:.2e} (Valid clusters: {valid_clusters}, Time: {elapsed:.2f}s)")
        
        # Progress update
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{len(coarse_k_values)} completed")
            if GPU_AVAILABLE:
                print_gpu_memory_info()
    
    stage1_time = time.time() - stage1_start
    print(f"\nStage 1 completed in {stage1_time:.2f}s")
    
    # Find rough optimal K from coarse search (highest F-ratio)
    rough_optimal_k_idx = np.argmax(coarse_f_ratios)
    rough_optimal_k = coarse_k_values[rough_optimal_k_idx]
    print(f"Rough optimal K found at K={rough_optimal_k} (F-ratio: {coarse_f_ratios[rough_optimal_k_idx]:.2e})")
    
    # Stage 2: Fine search around rough optimal point
    print(f"\nStage 2: Fine search around K={rough_optimal_k}...")
    fine_range = max(50, rough_optimal_k // 4)  # Search range around rough optimal
    fine_start = max(1, rough_optimal_k - fine_range)
    fine_end = min(max_k, rough_optimal_k + fine_range)
    
    fine_k_values = list(range(fine_start, fine_end + 1, 10))  # Step size 10 for fine search
    if fine_start > 1:
        fine_k_values = [1] + fine_k_values
    if fine_end not in fine_k_values:
        fine_k_values.append(fine_end)
    
    print(f"Fine search range: K from {fine_start} to {fine_end}")
    print(f"Fine search K values: {fine_k_values}")
    
    fine_f_ratios = []
    stage2_start = time.time()
    
    for i, k in enumerate(fine_k_values):
        print(f"  Testing K={k}...", end=" ")
        start_time = time.time()
        
        if GPU_AVAILABLE:
            # GPU-accelerated K-means
            kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            # CPU fallback
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate ANOVA F-ratio
        f_ratio, valid_clusters, total_points = calculate_anova_f_ratio(
            embeddings, cluster_labels, kmeans.cluster_centers_
        )
        
        elapsed = time.time() - start_time
        fine_f_ratios.append(f_ratio)
        print(f"F-ratio: {f_ratio:.2e} (Valid clusters: {valid_clusters}, Time: {elapsed:.2f}s)")
        
        # Progress update
        if (i + 1) % 3 == 0:
            print(f"    Progress: {i + 1}/{len(fine_k_values)} completed")
            if GPU_AVAILABLE:
                print_gpu_memory_info()
    
    stage2_time = time.time() - stage2_start
    print(f"\nStage 2 completed in {stage2_time:.2f}s")
    print(f"Total clustering time: {stage1_time + stage2_time:.2f}s")
    
    # Find optimal K from fine search (highest F-ratio)
    optimal_k_idx = np.argmax(fine_f_ratios)
    optimal_k = fine_k_values[optimal_k_idx]
    print(f"\nOptimal number of clusters: {optimal_k} (F-ratio: {fine_f_ratios[optimal_k_idx]:.2e})")
    
    # Combine results for plotting
    all_k_values = coarse_k_values + [k for k in fine_k_values if k not in coarse_k_values]
    all_f_ratios = coarse_f_ratios + [fine_f_ratios[fine_k_values.index(k)] for k in fine_k_values if k not in coarse_k_values]
    
    # Sort by K values
    sorted_indices = np.argsort(all_k_values)
    all_k_values = [all_k_values[i] for i in sorted_indices]
    all_f_ratios = [all_f_ratios[i] for i in sorted_indices]
    
    return optimal_k, all_f_ratios, all_k_values, fine_f_ratios


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


def plot_anova_curve(k_values: list, f_ratios: list, optimal_k: int, output_path: str):
    """
    Purpose: Plot the ANOVA F-ratio across different K values and highlight the optimal K.

    Inputs:
    - k_values (list): Monotonic list of tested cluster counts K used for ANOVA.
    - f_ratios (list): ANOVA F-ratio values corresponding to each K in k_values.
    - optimal_k (int): The selected optimal number of clusters to highlight on the plot.
    - output_path (str): Filesystem path where the generated plot image will be saved.

    Output:
    - None: Saves the plot to output_path and shows the figure.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot F-ratio curve
    plt.plot(k_values, f_ratios, 'go-', linewidth=2, markersize=6, label='F-ratio', alpha=0.7)
    
    # Highlight optimal K point
    if optimal_k in k_values:
        optimal_idx = k_values.index(optimal_k)
        plt.plot(optimal_k, f_ratios[optimal_idx], 'ro', markersize=12, label=f'Optimal K={optimal_k}')
        
        # Add text annotation for optimal K
        plt.annotate(f'Optimal K = {optimal_k}', 
                    xy=(optimal_k, f_ratios[optimal_idx]),
                    xytext=(optimal_k * 1.1, f_ratios[optimal_idx] * 1.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red')
    
    # Add annotations
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('ANOVA F-ratio', fontsize=12)
    title = 'Two-Stage ANOVA F-ratio Method for Optimal K Selection'
    if GPU_AVAILABLE:
        title += ' (GPU Accelerated)'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use linear scale for better visualization to avoid hiding near-zero values
    # plt.xscale('log')  # Keep X linear
    # plt.yscale('log')  # Do not use log scale on Y to preserve pre-optimal curve
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ANOVA curve saved to: {output_path}")
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


def decode_tokens(tokenizer, token_indices, max_length=100):
    """Decode token indices to text using the tokenizer."""
    try:
        # Try to decode as a sequence first
        try:
            # Decode the entire sequence
            sequence_text = tokenizer.decode(token_indices, skip_special_tokens=True)
            if sequence_text.strip():
                # Clean up the text
                cleaned_text = sequence_text.strip()
                # Remove excessive whitespace
                cleaned_text = ' '.join(cleaned_text.split())
                if len(cleaned_text) > max_length:
                    cleaned_text = cleaned_text[:max_length] + "..."
                return cleaned_text
        except:
            pass
        
        # Fallback: decode individual tokens
        decoded_tokens = []
        for idx in token_indices:
            try:
                # Try different decoding strategies
                token_text = None
                
                # Strategy 1: Single token decode
                try:
                    token_text = tokenizer.decode([idx], skip_special_tokens=True)
                except:
                    pass
                
                # Strategy 2: With EOS token
                if not token_text or not token_text.strip():
                    try:
                        sequence = [idx]
                        if tokenizer.eos_token_id:
                            sequence.append(tokenizer.eos_token_id)
                        token_text = tokenizer.decode(sequence, skip_special_tokens=True)
                    except:
                        pass
                
                # Strategy 3: With BOS and EOS tokens
                if not token_text or not token_text.strip():
                    try:
                        sequence = []
                        if tokenizer.bos_token_id:
                            sequence.append(tokenizer.bos_token_id)
                        sequence.append(idx)
                        if tokenizer.eos_token_id:
                            sequence.append(tokenizer.eos_token_id)
                        token_text = tokenizer.decode(sequence, skip_special_tokens=True)
                    except:
                        pass
                
                # Add token if we got something meaningful
                if token_text and token_text.strip():
                    decoded_tokens.append(token_text.strip())
                else:
                    # Try to get token as string
                    try:
                        token_str = tokenizer.convert_ids_to_tokens(idx)
                        if token_str and token_str != tokenizer.unk_token:
                            decoded_tokens.append(token_str)
                        else:
                            decoded_tokens.append(f"[TOKEN_{idx}]")
                    except:
                        decoded_tokens.append(f"[TOKEN_{idx}]")
                        
            except Exception as e:
                decoded_tokens.append(f"[ERROR_{idx}]")
        
        # Join tokens and clean up
        if decoded_tokens:
            result = " ".join(decoded_tokens)
            # Remove excessive whitespace and special characters
            result = ' '.join(result.split())
            if len(result) > max_length:
                result = result[:max_length] + "..."
            return result
        else:
            return f"[NO_TOKENS_{len(token_indices)}]"
            
    except Exception as e:
        return f"[DECODE_ERROR: {e}]"


def extract_meaningful_text(decoded_text):
    """Extract meaningful text from decoded tokens."""
    if not decoded_text or decoded_text.startswith('['):
        return decoded_text
    
    # Try to extract meaningful patterns
    import re
    
    # Remove common token artifacts
    text = decoded_text
    
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', ' ', text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Try to find Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
    if chinese_chars:
        return ' '.join(chinese_chars[:3])  # Return first 3 Chinese words
    
    # Try to find English words
    english_words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    if english_words:
        return ' '.join(english_words[:5])  # Return first 5 English words
    
    # Return cleaned text
    return text[:50] if len(text) > 50 else text


def analyze_cluster_representatives(embeddings: np.ndarray, cluster_centers: np.ndarray, 
                                  cluster_labels: np.ndarray, tokenizer=None, n_tokens: int = 10):
    """Analyze and display representative tokens for each cluster."""
    print(f"\n" + "=" * 60)
    print("CLUSTER REPRESENTATIVE ANALYSIS")
    print("=" * 60)
    
    # Calculate overall silhouette score
    silhouette_avg = calculate_silhouette_score(embeddings, cluster_labels)
    if silhouette_avg is not None:
        print(f"Overall Silhouette Score: {silhouette_avg:.4f}")
        print(f"Clustering Quality: {'Excellent' if silhouette_avg > 0.7 else 'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.25 else 'Poor'}")
    
    # Find closest tokens to each center
    closest_tokens = find_closest_tokens_to_centers(embeddings, cluster_centers, n_tokens)
    
    # Get cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    print(f"\nCluster Analysis Summary:")
    print(f"Total clusters: {len(unique)}")
    print(f"Clusters with >1 token: {sum(counts > 1)}")
    print(f"Clusters with 1 token: {sum(counts == 1)}")
    
    print(f"\nRepresentative tokens for each cluster (top {n_tokens} closest to center):")
    print("-" * 100)
    
    cluster_analysis = {}
    
    for cluster_id in sorted(closest_tokens.keys()):
        cluster_size = cluster_sizes.get(cluster_id, 0)
        size_percentage = cluster_size / len(embeddings) * 100
        
        # Calculate cluster silhouette score (for clusters with >1 member)
        cluster_silhouette = None
        if cluster_size > 1:
            try:
                from sklearn.metrics import silhouette_samples
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                cluster_silhouettes = silhouette_samples(embeddings, cluster_labels)
                cluster_silhouette = np.mean(cluster_silhouettes[cluster_mask])
            except:
                pass
        
        print(f"\nCluster {cluster_id:2d} (Size: {cluster_size:6,} tokens, {size_percentage:5.1f}%):")
        
        # Get representative token indices
        token_indices = closest_tokens[cluster_id]['indices']
        distances = closest_tokens[cluster_id]['distances']
        
        # Decode tokens if tokenizer is available
        if tokenizer:
            representative_text = decode_tokens(tokenizer, token_indices)
            meaningful_text = extract_meaningful_text(representative_text)
            print(f"  Raw tokens: {representative_text}")
            print(f"  Meaningful: {meaningful_text}")
        else:
            print(f"  Token indices: {token_indices[:5]}...")  # Show first 5
        
        print(f"  Avg distance: {np.mean(distances):.4f}")
        if cluster_silhouette is not None:
            print(f"  Silhouette: {cluster_silhouette:.4f}")
        
        # Store analysis results
        cluster_analysis[cluster_id] = {
            'size': cluster_size,
            'percentage': size_percentage,
            'token_indices': token_indices,
            'distances': distances,
            'representative_text': decode_tokens(tokenizer, token_indices) if tokenizer else None,
            'meaningful_text': extract_meaningful_text(decode_tokens(tokenizer, token_indices)) if tokenizer else None,
            'silhouette_score': float(cluster_silhouette) if cluster_silhouette is not None else None
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
        
        # PART 2: Find optimal clusters using ANOVA method
        optimal_k, f_ratios, k_values, fine_f_ratios = find_optimal_clusters_anova(embeddings, args.max_k, args.random_state)
        
        # Plot ANOVA curve
        plot_path = output_dir / "anova_curve.png"
        plot_anova_curve(k_values, f_ratios, optimal_k, str(plot_path))
        
        # Perform final clustering with representative analysis
        cluster_labels, kmeans, cluster_analysis = perform_final_clustering(
            embeddings, optimal_k, args.random_state, 
            tokenizer=tokenizer,
            analyze_representatives=not args.skip_representatives,
            n_representatives=args.n_representatives
        )
        
        # Calculate overall silhouette score
        overall_silhouette = calculate_silhouette_score(embeddings, cluster_labels)
        
        # Save clustering results
        results = {
            "optimal_k": int(optimal_k),
            "f_ratios": [float(f) for f in f_ratios],
            "k_values": [int(k) for k in k_values],
            "fine_f_ratios": [float(f) for f in fine_f_ratios],
            "cluster_labels": [int(l) for l in cluster_labels.tolist()],
            "cluster_centers": [[float(c) for c in center] for center in kmeans.cluster_centers_.tolist()],
            "inertia": float(kmeans.inertia_),
            "silhouette_score": float(overall_silhouette) if overall_silhouette is not None else None,
            "gpu_accelerated": GPU_AVAILABLE,
            "total_tokens": int(len(embeddings)),
            "embedding_dimension": int(embeddings.shape[1])
        }
        
        # Convert cluster_analysis to JSON-serializable format
        if cluster_analysis:
            serializable_analysis = {}
            for cluster_id, analysis in cluster_analysis.items():
                serializable_analysis[int(cluster_id)] = {
                    "size": int(analysis['size']),
                    "percentage": float(analysis['percentage']),
                    "token_indices": [int(idx) for idx in analysis['token_indices']],
                    "distances": [float(d) for d in analysis['distances']],
                    "representative_text": analysis['representative_text'],
                    "meaningful_text": analysis['meaningful_text'],
                    "silhouette_score": analysis['silhouette_score']
                }
            results["cluster_analysis"] = serializable_analysis
        
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
