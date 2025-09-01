#!/usr/bin/env python3
"""
Test script for transformer_analysis.py
Creates a simple test case to verify the new SVD analysis functionality
"""

import numpy as np
from scripts.transformer_analysis import compute_matrix_stats, format_layer_analysis

def test_svd_analysis():
    """Test the SVD analysis functionality with synthetic matrices."""
    print("Testing SVD analysis functionality...")
    
    # Create a test matrix with known rank properties
    # Matrix with rank 3 but 90% variance captured by first 2 singular values
    np.random.seed(42)
    
    # Create a low-rank matrix (large enough to avoid sampling)
    U = np.random.randn(100, 3)
    V = np.random.randn(3, 100)
    S = np.array([10.0, 8.0, 1.0])  # Singular values: first two dominate
    
    test_matrix = U @ np.diag(S) @ V
    
    print(f"Test matrix shape: {test_matrix.shape}")
    print(f"Expected actual rank: 3")
    print(f"Expected 90% variance rank: 2 (first two singular values dominate)")
    
    # Analyze the matrix
    stats = compute_matrix_stats(test_matrix)
    
    print("\nAnalysis results:")
    print(f"Design rank: {stats['design_rank']}")
    print(f"Actual rank: {stats['actual_rank']}")
    print(f"SVD 90% rank: {stats['svd_90_percent_rank']}")
    print(f"SVD 90% ratio: {stats['svd_90_percent_ratio']:.3f}")
    
    # Verify the results
    assert stats['actual_rank'] == 3, f"Expected actual rank 3, got {stats['actual_rank']}"
    assert stats['svd_90_percent_rank'] == 2, f"Expected 90% rank 2, got {stats['svd_90_percent_rank']}"
    
    print("✅ SVD analysis test passed!")
    
    # Test the formatting function with mock data
    print("\nTesting formatting function...")
    mock_results = [
        {
            "layer_index": 0,
            "block_type": "q_proj",
            "shape": "(512, 512)",
            "rows": 512,
            "cols": 512,
            "design_rank": 512,
            "actual_rank": 128,
            "svd_90_percent_rank": 64,
            "svd_90_percent_ratio": 0.125,
            "sparsity": 0.1,
            "mean": 0.0,
            "std": 0.1
        },
        {
            "layer_index": 0,
            "block_type": "k_proj",
            "shape": "(512, 512)",
            "rows": 512,
            "cols": 512,
            "design_rank": 512,
            "actual_rank": 128,
            "svd_90_percent_rank": 64,
            "svd_90_percent_ratio": 0.125,
            "sparsity": 0.1,
            "mean": 0.0,
            "std": 0.1
        },
        {
            "layer_index": "",
            "block_type": "embedding",
            "shape": "(32000, 512)",
            "rows": 32000,
            "cols": 512,
            "design_rank": 512,
            "actual_rank": 256,
            "svd_90_percent_rank": 128,
            "svd_90_percent_ratio": 0.25,
            "sparsity": 0.05,
            "mean": 0.0,
            "std": 0.1
        }
    ]
    
    format_layer_analysis(mock_results)
    print("✅ Formatting test completed!")

if __name__ == "__main__":
    test_svd_analysis()
