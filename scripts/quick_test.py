#!/usr/bin/env python3

"""
Quick Test Script for GPU-Accelerated Embedding Analysis
Demonstrates the usage of optimized embedding analysis with GPU acceleration
"""

import time
import numpy as np
from pathlib import Path

# GPU-accelerated libraries
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
    print("✓ cuML GPU acceleration available")
except ImportError:
    print("⚠ cuML not available, using CPU fallback")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    GPU_AVAILABLE = False


def quick_gpu_test():
    """Quick test to demonstrate GPU acceleration."""
    print("=" * 60)
    print("QUICK GPU ACCELERATION TEST")
    print("=" * 60)
    
    # Generate test data (similar to transformer embeddings)
    print("Generating test data...")
    n_samples, n_features = 10000, 768
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Test data normalization
    print("\n1. Testing data normalization...")
    start_time = time.time()
    
    if GPU_AVAILABLE:
        scaler = cuStandardScaler()
        data_normalized = scaler.fit_transform(data)
        method = "GPU (cuML)"
    else:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        method = "CPU (scikit-learn)"
    
    norm_time = time.time() - start_time
    print(f"✓ {method} normalization completed in {norm_time:.3f}s")
    
    # Test K-means clustering
    print("\n2. Testing K-means clustering...")
    k_values = [50, 100, 200]
    
    for k in k_values:
        print(f"\n   Testing K={k}:")
        
        start_time = time.time()
        if GPU_AVAILABLE:
            kmeans = cuKMeans(n_clusters=k, random_state=42, n_init=10)
            method = "GPU (cuML)"
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            method = "CPU (scikit-learn)"
        
        kmeans.fit(data_normalized)
        elapsed = time.time() - start_time
        
        print(f"     {method} K-means completed in {elapsed:.3f}s")
        print(f"     Inertia: {kmeans.inertia_:.2e}")
    
    print(f"\n✓ Quick test completed using {method}")
    
    # Performance summary
    if GPU_AVAILABLE:
        print("\n🚀 GPU acceleration is working!")
        print("   You can now run the full embedding analysis with:")
        print("   python scripts/embedding_analysis.py --model_path /path/to/model")
    else:
        print("\n💻 Running in CPU mode")
        print("   Install cuML for GPU acceleration:")
        print("   uv add cuml-cu12 --extra-index-url https://pypi.nvidia.com")


def check_system_info():
    """Check system information and GPU status."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # GPU information
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"GPU devices found: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = memory_info.total / 1024**3
            
            print(f"  GPU {i}: {name}")
            print(f"    Memory: {total_gb:.1f} GB")
            
    except ImportError:
        print("GPU monitoring not available (pynvml not installed)")
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    
    # Library versions
    print("\nLibrary versions:")
    try:
        print(f"  NumPy: {np.__version__}")
    except:
        print("  NumPy: version unknown")
    
    if GPU_AVAILABLE:
        try:
            print(f"  cuML: {cuml.__version__}")
        except:
            print("  cuML: version unknown")
    
    # Memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\nSystem memory: {memory.total / 1024**3:.1f} GB")
        print(f"Available memory: {memory.available / 1024**3:.1f} GB")
    except ImportError:
        print("\nSystem memory: psutil not available")


def main():
    """Main function."""
    print("GPU-Accelerated Embedding Analysis - Quick Test")
    print("=" * 60)
    
    # Check system info
    check_system_info()
    
    # Run quick test
    print("\n")
    quick_gpu_test()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
