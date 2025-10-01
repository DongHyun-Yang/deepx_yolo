#!/usr/bin/env python3
"""
Script to compare raw outputs from simple and ultralytics versions.
Checks if the model outputs are identical between two implementations.
"""

import numpy as np
import os
import argparse
from pathlib import Path

def compare_raw_outputs(file1_path, file2_path, tolerance=1e-6):
    """
    Compare two numpy arrays from saved .npy files.
    
    Args:
        file1_path: Path to first .npy file
        file2_path: Path to second .npy file  
        tolerance: Numerical tolerance for comparison (default: 1e-6)
    
    Returns:
        dict: Comparison results with detailed statistics
    """
    
    # Check if files exist
    if not os.path.exists(file1_path):
        return {"error": f"File not found: {file1_path}"}
    if not os.path.exists(file2_path):
        return {"error": f"File not found: {file2_path}"}
    
    # Load numpy arrays
    try:
        array1 = np.load(file1_path)
        array2 = np.load(file2_path)
    except Exception as e:
        return {"error": f"Error loading files: {str(e)}"}
    
    # Basic shape comparison
    shape_match = array1.shape == array2.shape
    
    results = {
        "file1": file1_path,
        "file2": file2_path,
        "file1_shape": array1.shape,
        "file2_shape": array2.shape,
        "shape_match": shape_match,
        "file1_dtype": str(array1.dtype),
        "file2_dtype": str(array2.dtype),
        "file1_size": array1.size,
        "file2_size": array2.size
    }
    
    if not shape_match:
        results["identical"] = False
        results["reason"] = "Shapes do not match"
        return results
    
    # Statistical comparison
    results.update({
        "file1_min": float(np.min(array1)),
        "file1_max": float(np.max(array1)),
        "file1_mean": float(np.mean(array1)),
        "file1_std": float(np.std(array1)),
        "file2_min": float(np.min(array2)),
        "file2_max": float(np.max(array2)),
        "file2_mean": float(np.mean(array2)),
        "file2_std": float(np.std(array2))
    })
    
    # Exact equality check
    exact_equal = np.array_equal(array1, array2)
    results["exact_equal"] = exact_equal
    
    # Tolerance-based comparison
    close_equal = np.allclose(array1, array2, rtol=tolerance, atol=tolerance)
    results["close_equal"] = close_equal
    results["tolerance"] = tolerance
    
    # Calculate differences
    diff = np.abs(array1 - array2)
    results.update({
        "max_absolute_diff": float(np.max(diff)),
        "mean_absolute_diff": float(np.mean(diff)),
        "num_different_elements": int(np.sum(diff > tolerance)),
        "percent_different": float(np.sum(diff > tolerance) / array1.size * 100)
    })
    
    # Relative differences (avoid division by zero)
    mask = np.abs(array2) > 1e-10
    if np.any(mask):
        rel_diff = np.zeros_like(diff)
        rel_diff[mask] = diff[mask] / np.abs(array2[mask])
        results.update({
            "max_relative_diff": float(np.max(rel_diff)),
            "mean_relative_diff": float(np.mean(rel_diff[mask])) if np.any(mask) else 0.0
        })
    
    # Final determination
    if exact_equal:
        results["identical"] = True
        results["status"] = "IDENTICAL"
    elif close_equal:
        results["identical"] = True
        results["status"] = f"NEARLY_IDENTICAL (within tolerance {tolerance})"
    else:
        results["identical"] = False
        results["status"] = "DIFFERENT"
    
    return results

def print_comparison_results(results):
    """Print formatted comparison results."""
    
    if "error" in results:
        print(f"❌ ERROR: {results['error']}")
        return
    
    print("="*80)
    print("🔍 RAW OUTPUT COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n📁 Files:")
    print(f"   File 1: {Path(results['file1']).name}")
    print(f"   File 2: {Path(results['file2']).name}")
    
    print(f"\n📊 Basic Info:")
    print(f"   Shape 1: {results['file1_shape']}")
    print(f"   Shape 2: {results['file2_shape']}")
    print(f"   Shape Match: {'✅' if results['shape_match'] else '❌'}")
    print(f"   Data Type 1: {results['file1_dtype']}")
    print(f"   Data Type 2: {results['file2_dtype']}")
    print(f"   Total Elements: {results['file1_size']:,}")
    
    if not results['shape_match']:
        print(f"\n❌ RESULT: Files have different shapes - cannot compare values")
        return
    
    print(f"\n📈 Statistics:")
    print(f"   File 1 - Min: {results['file1_min']:.6f}, Max: {results['file1_max']:.6f}")
    print(f"   File 1 - Mean: {results['file1_mean']:.6f}, Std: {results['file1_std']:.6f}")
    print(f"   File 2 - Min: {results['file2_min']:.6f}, Max: {results['file2_max']:.6f}")
    print(f"   File 2 - Mean: {results['file2_mean']:.6f}, Std: {results['file2_std']:.6f}")
    
    print(f"\n🔍 Comparison:")
    print(f"   Exact Equal: {'✅' if results['exact_equal'] else '❌'}")
    print(f"   Close Equal (tolerance {results['tolerance']}): {'✅' if results['close_equal'] else '❌'}")
    print(f"   Max Absolute Difference: {results['max_absolute_diff']:.10f}")
    print(f"   Mean Absolute Difference: {results['mean_absolute_diff']:.10f}")
    
    if 'max_relative_diff' in results:
        print(f"   Max Relative Difference: {results['max_relative_diff']:.10f}")
        print(f"   Mean Relative Difference: {results['mean_relative_diff']:.10f}")
    
    print(f"   Different Elements: {results['num_different_elements']:,} ({results['percent_different']:.4f}%)")
    
    print(f"\n🎯 FINAL RESULT:")
    if results['identical']:
        print(f"   ✅ {results['status']}")
        if results['exact_equal']:
            print("   📝 The arrays are byte-for-byte identical!")
        else:
            print("   📝 The arrays are numerically equivalent within tolerance.")
    else:
        print(f"   ❌ {results['status']}")
        print("   📝 The arrays have significant differences.")
    
    print("="*80)

def main():
    """Main function to run the comparison."""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Compare two numpy arrays saved as .npy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.npy file2.npy
  %(prog)s --file1 output1.npy --file2 output2.npy
  %(prog)s -f1 runs/predict/onnx/simple/raw_output_simple.npy -f2 runs/predict/onnx/ultralytics_wrapper/raw_output_ultralytics.npy
        """
    )
    
    # Define default paths
    default_file1 = os.path.join("runs", "predict", "onnx", "simple", "raw_output_simple.npy")
    default_file2 = os.path.join("runs", "predict", "onnx", "ultralytics_wrapper", "raw_output_ultralytics.npy")
    
    parser.add_argument(
        'file1',
        nargs='?',
        default=default_file1,
        help=f'Path to first .npy file (default: {default_file1})'
    )
    parser.add_argument(
        'file2',
        nargs='?',
        default=default_file2,
        help=f'Path to second .npy file (default: {default_file2})'
    )
    parser.add_argument(
        '--file1', '-f1',
        dest='file1_alt',
        help='Alternative way to specify first file path'
    )
    parser.add_argument(
        '--file2', '-f2',
        dest='file2_alt',
        help='Alternative way to specify second file path'
    )
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=1e-8,
        help='Numerical tolerance for comparison (default: 1e-8)'
    )
    
    args = parser.parse_args()
    
    # Use alternative paths if provided
    file1_path = args.file1_alt if args.file1_alt else args.file1
    file2_path = args.file2_alt if args.file2_alt else args.file2
    
    print("🚀 Starting Raw Output Comparison...")
    print(f"📁 Comparing:")
    print(f"   File 1: {file1_path}")
    print(f"   File 2: {file2_path}")
    
    # Perform comparison
    results = compare_raw_outputs(file1_path, file2_path, tolerance=args.tolerance)
    
    # Print results
    print_comparison_results(results)
    
    # Additional detailed analysis for debugging
    if not results.get('identical', False) and 'error' not in results:
        print("\n🔬 DETAILED ANALYSIS FOR DEBUGGING:")
        print("-" * 50)
        
        # Load arrays again for detailed analysis
        array1 = np.load(file1_path)
        array2 = np.load(file2_path)
        
        # Find locations of largest differences
        diff = np.abs(array1 - array2)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        
        print(f"   Location of max difference: {max_diff_idx}")
        print(f"   Value 1 at max diff: {array1[max_diff_idx]}")
        print(f"   Value 2 at max diff: {array2[max_diff_idx]}")
        print(f"   Difference at max diff: {diff[max_diff_idx]}")
        
        # Sample a few values for inspection
        print(f"\n   Sample comparison (first 5 values):")
        flat1 = array1.flatten()
        flat2 = array2.flatten()
        for i in range(min(5, len(flat1))):
            print(f"   [{i}] {flat1[i]:.10f} vs {flat2[i]:.10f} (diff: {abs(flat1[i] - flat2[i]):.2e})")

if __name__ == "__main__":
    main()