#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced transformer_analysis.py
"""

import numpy as np
from scripts.transformer_analysis import compute_matrix_stats, format_layer_analysis

def create_demo_matrices():
    """Create demo matrices with different properties for demonstration."""
    np.random.seed(42)
    
    matrices = {}
    
    # 1. Low-rank attention matrix (q_proj)
    U = np.random.randn(512, 64)
    V = np.random.randn(64, 512)
    S = np.array([10.0] * 32 + [5.0] * 16 + [1.0] * 16)  # 32 dominant singular values
    matrices["q_proj"] = U @ np.diag(S) @ V
    
    # 2. Sparse embedding matrix
    embedding = np.random.randn(32000, 512) * 0.1
    # Make it sparse
    mask = np.random.rand(32000, 512) > 0.9
    embedding[mask] = 0
    matrices["embedding"] = embedding
    
    # 3. High-rank output projection
    matrices["output"] = np.random.randn(512, 32000) * 0.05
    
    return matrices

def demo_analysis():
    """Demonstrate the enhanced analysis capabilities."""
    print("🚀 TRANSFORMER MATRIX ANALYSIS DEMO")
    print("=" * 60)
    
    # Create demo matrices
    matrices = create_demo_matrices()
    
    # Analyze each matrix
    results = []
    for matrix_name, matrix in matrices.items():
        print(f"\n📊 Analyzing {matrix_name} matrix...")
        print(f"   Shape: {matrix.shape}")
        
        # Analyze the matrix
        stats = compute_matrix_stats(matrix)
        
        # Create result row for formatting
        result_row = {
            "layer_index": 0 if matrix_name != "embedding" else "",
            "block_type": matrix_name,
            "shape": stats["shape"],
            "rows": stats["rows"],
            "cols": stats["cols"],
            "design_rank": stats["design_rank"],
            "actual_rank": stats["actual_rank"],
            "svd_90_percent_rank": stats["svd_90_percent_rank"],
            "svd_90_percent_ratio": stats["svd_90_percent_ratio"],
            "sparsity": stats["sparsity"],
            "mean": stats["mean"],
            "std": stats["std"]
        }
        results.append(result_row)
        
        # Display key insights
        print(f"   Design Rank: {stats['design_rank']}")
        print(f"   Actual Rank: {stats['actual_rank']} ({stats['rank_ratio']*100:.1f}% of design)")
        print(f"   SVD 90% Rank: {stats['svd_90_percent_rank']} ({stats['svd_90_percent_ratio']*100:.1f}% of design)")
        print(f"   Sparsity: {stats['sparsity']*100:.1f}%")
        
        # Interpret the results
        if stats['svd_90_percent_ratio'] < 0.3:
            print(f"   💡 This matrix is highly compressible! Only {stats['svd_90_percent_rank']} singular values needed for 90% variance.")
        elif stats['svd_90_percent_ratio'] < 0.6:
            print(f"   ⚠️  Moderate compressibility. {stats['svd_90_percent_rank']} singular values needed for 90% variance.")
        else:
            print(f"   🔒 Low compressibility. Most singular values are important.")
    
    # Show formatted layer analysis
    print("\n" + "="*60)
    print("FORMATTED LAYER ANALYSIS")
    print("="*60)
    format_layer_analysis(results)
    
    print("\n✅ Demo completed! The enhanced analysis shows:")
    print("   • SVD 90% variance analysis for compression insights")
    print("   • Formatted layer-by-layer display")
    print("   • Rank analysis for model understanding")

if __name__ == "__main__":
    demo_analysis()
