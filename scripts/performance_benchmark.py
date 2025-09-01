#!/usr/bin/env python3

"""
Performance Benchmark Script for GPU vs CPU K-means Clustering
Compare the performance of cuML (GPU) vs scikit-learn (CPU) implementations
"""

import time
import numpy as np
import argparse
from pathlib import Path

# GPU-accelerated libraries
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
    print("✓ cuML GPU acceleration available")
except ImportError:
    print("⚠ cuML not available")
    GPU_AVAILABLE = False

# CPU fallback
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    CPU_AVAILABLE = True
    print("✓ scikit-learn CPU fallback available")
except ImportError:
    print("⚠ scikit-learn not available")
    CPU_AVAILABLE = False

if not GPU_AVAILABLE and not CPU_AVAILABLE:
    raise ImportError("Neither cuML nor scikit-learn is available")


def generate_test_data(n_samples=10000, n_features=768, random_state=42):
    """Generate synthetic test data for benchmarking."""
    print(f"Generating test data: {n_samples} samples × {n_features} features")
    np.random.seed(random_state)
    
    # Generate random embeddings (similar to transformer embeddings)
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Normalize data
    if GPU_AVAILABLE:
        scaler = cuStandardScaler()
        data_normalized = scaler.fit_transform(data)
        print("✓ GPU-accelerated normalization completed")
    else:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        print("✓ CPU normalization completed")
    
    return data_normalized


def benchmark_kmeans(data, k_values, n_runs=3, random_state=42):
    """Benchmark K-means clustering performance."""
    results = {
        'gpu': {'times': [], 'memory_usage': []},
        'cpu': {'times': [], 'memory_usage': []}
    }
    
    print(f"\nBenchmarking K-means with {len(k_values)} different K values...")
    print(f"Data shape: {data.shape}")
    print(f"Number of runs per K: {n_runs}")
    
    for k in k_values:
        print(f"\nTesting K={k}:")
        
        # GPU benchmark
        if GPU_AVAILABLE:
            gpu_times = []
            for run in range(n_runs):
                start_time = time.time()
                kmeans = cuKMeans(n_clusters=k, random_state=random_state, n_init=10)
                kmeans.fit(data)
                elapsed = time.time() - start_time
                gpu_times.append(elapsed)
                print(f"  GPU run {run+1}: {elapsed:.3f}s")
            
            avg_gpu_time = np.mean(gpu_times)
            results['gpu']['times'].append(avg_gpu_time)
            print(f"  GPU average: {avg_gpu_time:.3f}s")
        
        # CPU benchmark
        if CPU_AVAILABLE:
            cpu_times = []
            for run in range(n_runs):
                start_time = time.time()
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                kmeans.fit(data)
                elapsed = time.time() - start_time
                cpu_times.append(elapsed)
                print(f"  CPU run {run+1}: {elapsed:.3f}s")
            
            avg_cpu_time = np.mean(cpu_times)
            results['cpu']['times'].append(avg_cpu_time)
            print(f"  CPU average: {avg_cpu_time:.3f}s")
            
            # Calculate speedup
            if GPU_AVAILABLE:
                speedup = avg_cpu_time / avg_gpu_time
                print(f"  GPU speedup: {speedup:.2f}x")
    
    return results


def print_benchmark_summary(k_values, results):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if GPU_AVAILABLE and CPU_AVAILABLE:
        print(f"{'K':<8} {'GPU (s)':<12} {'CPU (s)':<12} {'Speedup':<12}")
        print("-" * 50)
        
        for i, k in enumerate(k_values):
            gpu_time = results['gpu']['times'][i] if results['gpu']['times'] else 0
            cpu_time = results['cpu']['times'][i] if results['cpu']['times'] else 0
            
            if gpu_time > 0 and cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"{k:<8} {gpu_time:<12.3f} {cpu_time:<12.3f} {speedup:<12.2f}x")
            else:
                print(f"{k:<8} {gpu_time:<12.3f} {cpu_time:<12.3f} {'N/A':<12}")
        
        # Overall statistics
        if results['gpu']['times'] and results['cpu']['times']:
            avg_speedup = np.mean([cpu/gpu for cpu, gpu in zip(results['cpu']['times'], results['gpu']['times'])])
            print(f"\nAverage GPU speedup: {avg_speedup:.2f}x")
            
            total_gpu_time = sum(results['gpu']['times'])
            total_cpu_time = sum(results['cpu']['times'])
            total_speedup = total_cpu_time / total_gpu_time
            print(f"Total time - GPU: {total_gpu_time:.2f}s, CPU: {total_cpu_time:.2f}s")
            print(f"Overall speedup: {total_speedup:.2f}x")
    
    elif GPU_AVAILABLE:
        print("Only GPU results available:")
        for i, k in enumerate(k_values):
            gpu_time = results['gpu']['times'][i]
            print(f"K={k}: {gpu_time:.3f}s")
    
    elif CPU_AVAILABLE:
        print("Only CPU results available:")
        for i, k in enumerate(k_values):
            cpu_time = results['cpu']['times'][i]
            print(f"K={k}: {cpu_time:.3f}s")


def save_benchmark_results(k_values, results, output_path):
    """Save benchmark results to JSON file."""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for method in results:
        results_serializable[method] = {
            'times': [float(t) for t in results[method]['times']]
        }
    
    benchmark_data = {
        'k_values': k_values,
        'results': results_serializable,
        'gpu_available': GPU_AVAILABLE,
        'cpu_available': CPU_AVAILABLE,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nBenchmark results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU K-means clustering performance")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--n_features", type=int, default=768, help="Number of features per sample")
    parser.add_argument("--max_k", type=int, default=500, help="Maximum K to test")
    parser.add_argument("--k_step", type=int, default=100, help="Step size for K values")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs per K value")
    parser.add_argument("--output_dir", default="data/benchmarks", help="Output directory for results")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate test data
        data = generate_test_data(args.n_samples, args.n_features, args.random_state)
        
        # Define K values to test
        k_values = list(range(100, args.max_k + 1, args.k_step))
        if 1 not in k_values:
            k_values = [1] + k_values
        
        print(f"Testing K values: {k_values}")
        
        # Run benchmarks
        results = benchmark_kmeans(data, k_values, args.n_runs, args.random_state)
        
        # Print summary
        print_benchmark_summary(k_values, results)
        
        # Save results
        results_path = output_dir / f"benchmark_results_{args.n_samples}_{args.n_features}.json"
        save_benchmark_results(k_values, results, results_path)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()
