#!/usr/bin/env python3
"""Load only the latest comparison - uses both tracking file and time filter"""

import pandas as pd
import glob
import time
from pathlib import Path

def load_latest_comparison(results_dir, hours_ago=48):
    """
    Load results from latest comparison run
    - Uses tracking file if available
    - Falls back to time-based filter
    - Excludes old results directory
    - Only includes standard depths
    """
    
    # Try to find tracking file first
    tracking_files = glob.glob(f"{results_dir}/comparison/comparison_tracking_*.csv")
    if tracking_files:
        latest_tracking = sorted(tracking_files)[-1]
        tracking = pd.read_csv(latest_tracking)
        run_timestamp = tracking['timestamp'].iloc[0]
        print(f"Found tracking file for run: {run_timestamp}")
        print(f"Expected {len(tracking)} results")
    else:
        print(f"No tracking file found, using time filter (last {hours_ago} hours)")
        run_timestamp = None
    
    # Find files
    cutoff_time = time.time() - (hours_ago * 3600)
    all_files = glob.glob(f"{results_dir}/aggregated/**/*.xlsx", recursive=True)
    
    # Filter files
    valid_files = []
    for f in all_files:
        # Skip old results and temp files
        if 'old results' in f or Path(f).name.startswith('~$'):
            continue
        
        # Only standard depths
        if not any(f"block_num_block_size_{cfg}" in f for cfg in ['5_2', '6_2', '5_3', '6_3']):
            continue
        
        # Check time
        if Path(f).stat().st_mtime > cutoff_time:
            valid_files.append(f)
    
    print(f"Found {len(valid_files)} valid result files")
    
    # Load results
    results = []
    for filepath in valid_files:
        try:
            df = pd.read_excel(filepath, index_col='Scene')
            
            # Method
            if 'esfm_outliers_deep' in filepath:
                method = 'ESFM_OutlierLoss'
            elif 'esfm_deep' in filepath:
                method = 'Baseline_ESFM'
            else:
                continue
            
            # Depth
            for part in Path(filepath).parts:
                if 'block_num_block_size' in part:
                    blocks, size = part.split('_')[-2:]
                    layers = int(blocks) * int(size)
                    depth = f"{layers}L"
                    break
            else:
                continue
            
            if depth not in ['10L', '12L', '15L', '18L']:
                continue
            
            # Convert and add data
            for col in ['Trans', 'Rot', 'Nr']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            for scene in df.index:
                if pd.notna(df.loc[scene, 'Trans']):
                    results.append({
                        'method': method,
                        'depth': depth,
                        'layers': layers,
                        'scene': str(scene).zfill(4),
                        'trans': float(df.loc[scene, 'Trans']),
                        'rot': float(df.loc[scene, 'Rot']),
                        'nr': float(df.loc[scene, 'Nr'])
                    })
        except Exception as e:
            print(f"Skipping {Path(filepath).name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    print(f"\n✓ Loaded {len(df)} result entries")
    print(f"  Methods: {df['method'].unique()}")
    print(f"  Depths: {sorted(df['depth'].unique())}")
    print(f"  Scenes: {sorted(df['scene'].unique())}")
    
    return df

# Run
df = load_latest_comparison('results', hours_ago=48)

# Analyze
print("\n" + "="*80)
print("LATEST COMPARISON SUMMARY")
print("="*80)

for depth in sorted(df['depth'].unique()):
    print(f"\n{depth}:")
    for method in ['Baseline_ESFM', 'ESFM_OutlierLoss']:
        subset = df[(df['method'] == method) & (df['depth'] == depth)]
        if len(subset) > 0:
            print(f"  {method:20s}: Trans={subset['trans'].mean():.4f}±{subset['trans'].std():.4f}, "
                  f"Rot={subset['rot'].mean():.4f}±{subset['rot'].std():.4f}, "
                  f"Nr={subset['nr'].mean():.1f}±{subset['nr'].std():.1f} ({len(subset)} scenes)")

df.to_csv('results/latest_comparison.csv', index=False)
print(f"\n✓ Saved to: results/latest_comparison.csv")