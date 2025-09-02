#!/usr/bin/env python3

"""
Test script for ANOVA-based clustering optimization
"""

import numpy as np
import time
from pathlib import Path

# Import the ANOVA function
import sys
sys.path.append('.')
from scripts.embedding_analysis import calculate_anova_f_ratio, find_optimal_clusters_anova


def test_anova_calculation():
    """Test ANOVA F-ratio calculation with synthetic data."""
    print("Testing ANOVA F-ratio calculation...")
    
    # Generate synthetic data with known clusters
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Create 3 well-separated clusters
    cluster1 = np.random.randn(n_samples // 3, n_features) + np.array([5, 5, 5] + [0] * (n_features - 3))
    cluster2 = np.random.randn(n_samples // 3, n_features) + np.array([-5, -5, -5] + [0] * (n_features - 3))
    cluster3 = np.random.randn(n_samples - 2 * (n_samples // 3), n_features) + np.array([0, 0, 0] + [0] * (n_features - 3))
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    # Create cluster labels
    labels1 = np.zeros(n_samples // 3)
    labels2 = np.ones(n_samples // 3)
    labels3 = np.full(n_samples - 2 * (n_samples // 3), 2)
    cluster_labels = np.concatenate([labels1, labels2, labels3]).astype(int)
    
    # Calculate cluster centers
    cluster_centers = np.array([
        np.mean(cluster1, axis=0),
        np.mean(cluster2, axis=0),
        np.mean(cluster3, axis=0)
    ])
    
    # Calculate F-ratio
    f_ratio, valid_clusters, total_points = calculate_anova_f_ratio(embeddings, cluster_labels, cluster_centers)
    
    print(f"F-ratio: {f_ratio:.4f}")
    print(f"Valid clusters: {valid_clusters}")
    print(f"Total points: {total_points}")
    
    # Should have a high F-ratio for well-separated clusters
    assert f_ratio > 1.0, f"F-ratio should be > 1.0 for well-separated clusters, got {f_ratio}"
    assert valid_clusters == 3, f"Should have 3 valid clusters, got {valid_clusters}"
    
    print("✓ ANOVA calculation test passed!")
    return True


def test_small_scale_anova():
    """Test ANOVA method on a small scale."""
    print("\nTesting small-scale ANOVA clustering...")
    
    # Generate smaller synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    
    # Create synthetic embeddings
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Test with small K values
    max_k = 20
    print(f"Testing K values from 1 to {max_k}...")
    
    start_time = time.time()
    optimal_k, f_ratios, k_values, fine_f_ratios = find_optimal_clusters_anova(
        embeddings, max_k=max_k, random_state=42
    )
    elapsed = time.time() - start_time
    
    print(f"Optimal K: {optimal_k}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"F-ratios: {[f'{f:.2e}' for f in f_ratios[:5]]}...")  # Show first 5
    
    assert optimal_k > 0, f"Optimal K should be > 0, got {optimal_k}"
    assert len(f_ratios) > 0, "Should have F-ratio values"
    
    print("✓ Small-scale ANOVA test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ANOVA CLUSTERING OPTIMIZATION TESTS")
    print("=" * 60)
    
    try:
        # Test 1: ANOVA calculation
        test_anova_calculation()
        
        # Test 2: Small-scale clustering
        test_small_scale_anova()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("ANOVA method is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()


