#!/usr/bin/env python3
"""
Analyze comparison between Baseline ESFM and ESFM+OutlierLoss
FILTERED to only analyze the current comparison run
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import glob
from pathlib import Path
import re

def find_results_in_aggregated(base_dir, epoch_filter="num_epochs_10"):
    """
    Find results files in the aggregated directory structure.
    Filters to only specific epoch configuration to avoid mixing old experiments.
    
    Args:
        base_dir: Base directory to search
        epoch_filter: Pattern to match in path (e.g., "num_epochs_10" for debug mode)
    """
    results = []
    
    # Look for all xlsx files in aggregated directory
    aggregated_dir = os.path.join(base_dir, "aggregated")
    
    if not os.path.exists(aggregated_dir):
        aggregated_dir = os.path.join("results", "aggregated")
    
    if not os.path.exists(aggregated_dir):
        print(f"Aggregated directory not found: {aggregated_dir}")
        return []
    
    # Find all xlsx files matching the epoch filter
    # This avoids mixing results from different experiment runs
    search_pattern = os.path.join(aggregated_dir, epoch_filter + "*", "**/*.xlsx")
    xlsx_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Searching in: {search_pattern}")
    print(f"Found {len(xlsx_files)} xlsx files matching epoch filter '{epoch_filter}'")
    
    # Filter out Excel temporary files (start with ~$)
    xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith('~$')]
    print(f"After filtering temp files: {len(xlsx_files)} files")
    
    for filepath in xlsx_files:
        try:
            # Extract information from path
            path_parts = Path(filepath).parts
            
            # Get architecture (esfm_deep or esfm_outliers_deep)
            architecture = None
            for part in path_parts:
                if part == 'esfm_deep':
                    architecture = part
                    break
                elif part == 'esfm_outliers_deep':
                    architecture = part
                    break
            
            if not architecture:
                continue
                
            # Get block configuration (e.g., block_num_block_size_5_2)
            block_config = None
            for part in path_parts:
                if 'block_num_block_size' in part:
                    block_config = part
                    break
            
            if not block_config:
                continue
            
            # Parse blocks and size from config
            # Format: block_num_block_size_5_2
            config_parts = block_config.split('_')
            if len(config_parts) >= 6:
                blocks = int(config_parts[4])
                size = int(config_parts[5])
                layers = blocks * size
            else:
                continue
            
            # Get loss function - only process ESFMLoss and AdaptiveConfidenceWeightedOutliersLoss
            loss_func = None
            for part in path_parts:
                if part == 'ESFMLoss':
                    loss_func = part
                    break
                elif part == 'AdaptiveConfidenceWeightedOutliersLoss':
                    loss_func = part
                    break
            
            if not loss_func:
                continue
            
            # Skip stage 2 results (those with weight methods like mad_alpha, std_alpha, etc.)
            if any('alpha' in part for part in path_parts):
                continue
            
            # Determine method name
            if architecture == 'esfm_deep' and loss_func == 'ESFMLoss':
                method = 'Baseline_ESFM'
            elif architecture == 'esfm_outliers_deep' and loss_func == 'AdaptiveConfidenceWeightedOutliersLoss':
                method = 'ESFM_OutlierLoss'
            else:
                # Skip other combinations (old experiments)
                continue
            
            # Store result info
            results.append({
                'filepath': filepath,
                'method': method,
                'architecture': architecture,
                'loss': loss_func,
                'blocks': blocks,
                'size': size,
                'layers': layers,
                'depth': f"{layers}L"
            })
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    return results

def load_results_data(results_info):
    """Load data from results files"""
    all_data = []
    
    for info in results_info:
        try:
            # Read the Excel file
            df = pd.read_excel(info['filepath'], index_col='Scene')
            
            # Get data for each scene
            for scene_idx in df.index:
                row = df.loc[scene_idx]
                
                # Convert to numeric, handling any string values
                trans = pd.to_numeric(row.get('Trans', np.nan), errors='coerce')
                rot = pd.to_numeric(row.get('Rot', np.nan), errors='coerce')
                nr = pd.to_numeric(row.get('Nr', np.nan), errors='coerce')
                conv_time = pd.to_numeric(row.get('Convergence Time', np.nan), errors='coerce')
                best_epoch = pd.to_numeric(row.get('Best Epoch', np.nan), errors='coerce')
                
                data = {
                    'method': info['method'],
                    'depth': info['depth'],
                    'layers': info['layers'],
                    'blocks': info['blocks'],
                    'size': info['size'],
                    'scene': f"{scene_idx:04d}" if isinstance(scene_idx, int) else str(scene_idx),
                    'trans': trans,
                    'rot': rot,
                    'nr': nr,
                    'conv_time': conv_time,
                    'best_epoch': best_epoch
                }
                
                all_data.append(data)
                
        except Exception as e:
            print(f"Error loading {info['filepath']}: {e}")
            continue
    
    return pd.DataFrame(all_data)

def calculate_comparison(df):
    """Calculate comparison statistics"""
    
    # Get unique depths
    depths = sorted(df['layers'].unique())
    
    comparison_results = []
    
    for depth in depths:
        depth_str = f"{depth}L"
        
        # Filter data for this depth
        baseline_data = df[(df['method'] == 'Baseline_ESFM') & (df['layers'] == depth)]
        proposed_data = df[(df['method'] == 'ESFM_OutlierLoss') & (df['layers'] == depth)]
        
        if len(baseline_data) == 0 or len(proposed_data) == 0:
            print(f"Warning: No data for {depth_str}")
            continue
        
        # Calculate means
        result = {
            'depth': depth_str,
            'layers': depth,
            
            # Baseline stats
            'baseline_trans_mean': baseline_data['trans'].mean(),
            'baseline_trans_std': baseline_data['trans'].std(),
            'baseline_rot_mean': baseline_data['rot'].mean(),
            'baseline_rot_std': baseline_data['rot'].std(),
            'baseline_nr_mean': baseline_data['nr'].mean(),
            'baseline_nr_std': baseline_data['nr'].std(),
            'baseline_conv_time_mean': baseline_data['conv_time'].mean(),
            
            # Proposed stats  
            'proposed_trans_mean': proposed_data['trans'].mean(),
            'proposed_trans_std': proposed_data['trans'].std(),
            'proposed_rot_mean': proposed_data['rot'].mean(),
            'proposed_rot_std': proposed_data['rot'].std(),
            'proposed_nr_mean': proposed_data['nr'].mean(),
            'proposed_nr_std': proposed_data['nr'].std(),
            'proposed_conv_time_mean': proposed_data['conv_time'].mean(),
            
            # Sample sizes
            'baseline_n': len(baseline_data),
            'proposed_n': len(proposed_data)
        }
        
        # Calculate improvements (negative is better for trans/rot, positive for nr)
        if result['baseline_trans_mean'] > 0:
            result['trans_improvement'] = (result['baseline_trans_mean'] - result['proposed_trans_mean']) / result['baseline_trans_mean'] * 100
        
        if result['baseline_rot_mean'] > 0:
            result['rot_improvement'] = (result['baseline_rot_mean'] - result['proposed_rot_mean']) / result['baseline_rot_mean'] * 100
        
        if result['baseline_nr_mean'] > 0:
            result['nr_improvement'] = (result['proposed_nr_mean'] - result['baseline_nr_mean']) / result['baseline_nr_mean'] * 100
        
        if result['baseline_conv_time_mean'] > 0:
            result['conv_time_improvement'] = (result['baseline_conv_time_mean'] - result['proposed_conv_time_mean']) / result['baseline_conv_time_mean'] * 100
        
        # Statistical tests (paired t-test if same scenes)
        baseline_scenes = set(baseline_data['scene'])
        proposed_scenes = set(proposed_data['scene'])
        common_scenes = baseline_scenes & proposed_scenes
        
        if len(common_scenes) >= 3:
            # Get paired data
            baseline_paired = baseline_data[baseline_data['scene'].isin(common_scenes)].sort_values('scene')
            proposed_paired = proposed_data[proposed_data['scene'].isin(common_scenes)].sort_values('scene')
            
            # T-tests
            if len(baseline_paired) == len(proposed_paired):
                _, result['trans_pvalue'] = stats.ttest_rel(baseline_paired['trans'], proposed_paired['trans'])
                _, result['rot_pvalue'] = stats.ttest_rel(baseline_paired['rot'], proposed_paired['rot'])
                _, result['nr_pvalue'] = stats.ttest_rel(baseline_paired['nr'], proposed_paired['nr'])
        
        comparison_results.append(result)
    
    return pd.DataFrame(comparison_results)

def create_detailed_results_with_means(df):
    """Create detailed results CSV with MEAN rows per depth/method"""
    
    detailed_rows = []
    
    # Get unique combinations of method and depth
    for method in sorted(df['method'].unique()):
        for depth in sorted(df['layers'].unique()):
            depth_str = f"{depth}L"
            
            # Filter data
            subset = df[(df['method'] == method) & (df['layers'] == depth)]
            
            if len(subset) == 0:
                continue
            
            # Add individual scene rows
            for _, row in subset.iterrows():
                detailed_rows.append({
                    'method': method,
                    'depth': depth_str,
                    'scene': row['scene'],
                    'trans': row['trans'],
                    'rot': row['rot'],
                    'nr': row['nr'],
                    'conv_time': row['conv_time']
                })
            
            # Add MEAN row - ensure we're calculating means on numeric data only
            detailed_rows.append({
                'method': method,
                'depth': depth_str,
                'scene': 'MEAN',
                'trans': pd.to_numeric(subset['trans'], errors='coerce').mean(),
                'rot': pd.to_numeric(subset['rot'], errors='coerce').mean(),
                'nr': pd.to_numeric(subset['nr'], errors='coerce').mean(),
                'conv_time': pd.to_numeric(subset['conv_time'], errors='coerce').mean()
            })
    
    return pd.DataFrame(detailed_rows)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_comparison.py <results_dir> [epoch_filter]")
        print("Example: python analyze_comparison.py results")
        print("  Optional: python analyze_comparison.py results num_epochs_10")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    # Allow optional epoch filter
    epoch_filter = "num_epochs_10" if len(sys.argv) < 3 else sys.argv[2]
    
    print("=" * 80)
    print("ESFM vs ESFM+OutlierLoss COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {base_dir}")
    print(f"Epoch filter: {epoch_filter}")
    print()
    
    # Find results in aggregated directory
    print("Searching for results in aggregated directory...")
    results_info = find_results_in_aggregated(base_dir, epoch_filter)
    
    if not results_info:
        print("\nNo result files found!")
        print(f"\nTried filter: {epoch_filter}")
        print("\nIf you're analyzing a different run, specify the epoch pattern:")
        print("  python analyze_comparison.py results num_epochs_10000")
        sys.exit(1)
    
    print(f"\nFound {len(results_info)} result files for comparison")
    
    # Show what was found
    print("\nResults found:")
    for info in results_info:
        filename = os.path.basename(info['filepath'])
        print(f"  {info['method']:20s} {info['depth']:5s} - {filename[:80]}...")
    
    print("\nLoading results data...")
    df = load_results_data(results_info)
    
    if len(df) == 0:
        print("No data loaded!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} result entries")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Depths: {sorted(df['depth'].unique())}")
    print(f"Scenes: {sorted(df['scene'].unique())}")
    
    # Calculate comparison
    print("\nCalculating comparison statistics...")
    comparison_df = calculate_comparison(df)
    
    if len(comparison_df) == 0:
        print("No comparison data generated!")
        sys.exit(1)
    
    # Create detailed results
    print("Creating detailed results with means...")
    detailed_df = create_detailed_results_with_means(df)
    
    # Save results
    output_dir = base_dir
    
    summary_file = os.path.join(output_dir, "comparison_summary.csv")
    comparison_df.to_csv(summary_file, index=False)
    print(f"\nComparison summary saved to: {summary_file}")
    
    detailed_file = os.path.join(output_dir, "detailed_results_with_means.csv")
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"\nDepth: {row['depth']} ({row['layers']} layers)")
        print(f"  Translation Error:")
        print(f"    Baseline: {row['baseline_trans_mean']:.4f} ± {row['baseline_trans_std']:.4f}")
        print(f"    Proposed: {row['proposed_trans_mean']:.4f} ± {row['proposed_trans_std']:.4f}")
        print(f"    Improvement: {row['trans_improvement']:.2f}%")
        
        print(f"  Rotation Error:")
        print(f"    Baseline: {row['baseline_rot_mean']:.4f} ± {row['baseline_rot_std']:.4f}")
        print(f"    Proposed: {row['proposed_rot_mean']:.4f} ± {row['proposed_rot_std']:.4f}")
        print(f"    Improvement: {row['rot_improvement']:.2f}%")
        
        print(f"  Number of Cameras:")
        print(f"    Baseline: {row['baseline_nr_mean']:.1f} ± {row['baseline_nr_std']:.1f}")
        print(f"    Proposed: {row['proposed_nr_mean']:.1f} ± {row['proposed_nr_std']:.1f}")
        print(f"    Improvement: {row['nr_improvement']:.2f}%")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
