#!/bin/bash
# Comprehensive ESFM vs ESFM+OutlierLoss Comparison Script
# Runs each experiment (depth × scene × method) as a separate job

# Load CUDA module
module load CUDA/11.8.0

# Store timestamp for this comparison session
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPO_ROOT=$(pwd)

echo "=============================================="
echo "ESFM vs ESFM+OutlierLoss Comparison"
echo "Timestamp: ${TIMESTAMP}"
echo "=============================================="
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Methods to compare
METHODS=(
    "esfm_deep,ESFMLoss"                    # Baseline ESFM
    "esfm_outliers_deep,AdaptiveConfidenceWeightedOutliersLoss"  # ESFM + OutlierLoss
)

METHOD_NAMES=(
    "Baseline_ESFM"
    "ESFM_OutlierLoss"
)

# Fixed optimal weights (for OutlierLoss method)
REPROJ_LOSS_WEIGHT=1.0
CLASSIFICATION_LOSS_WEIGHT=1.0

# Depth configurations
# Format: "block_number,block_size,epochs,eval_intervals"
DEPTH_CONFIGS=(
    "5,2,180000,9000"    # 10 layers (5×2)
    "6,2,200000,10000"   # 12 layers (6×2)
    "5,3,250000,12500"   # 15 layers (5×3)
    "6,3,300000,15000"   # 18 layers (6×3)
)

# Evaluation scenes
SCANS="0007,0015,0024,0063,0104,0147,0327,0411,1001"
IFS=',' read -ra SCENE_ARRAY <<< "$SCANS"

# LSF configuration
QUEUE="risk"  # Base script adds "waic-" prefix, so this becomes "waic-risk"
BA_ONLY_LAST_EVAL="true"
GPU_STRATEGY="manual"  # Set to manual since we have a specific queue

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --scans)
            SCANS="$2"
            IFS=',' read -ra SCENE_ARRAY <<< "$SCANS"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            if [ "$QUEUE" = "multi" ]; then
                GPU_STRATEGY="multi"
            elif [ "$QUEUE" = "auto" ]; then
                GPU_STRATEGY="auto"
            else
                GPU_STRATEGY="manual"
            fi
            shift 2
            ;;
        --gpu-strategy)
            GPU_STRATEGY="$2"
            shift 2
            ;;
        --quick)
            # Quick mode: fewer epochs
            DEPTH_CONFIGS=(
                "5,2,100000,5000"
                "6,2,120000,6000"
                "5,3,150000,7500"
                "6,3,180000,9000"
            )
            echo "QUICK MODE: Using reduced epochs"
            shift
            ;;
        --conservative)
            # Conservative mode: only up to 12 layers
            DEPTH_CONFIGS=(
                "5,2,180000,9000"
                "6,2,200000,10000"
            )
            echo "CONSERVATIVE MODE: Testing only 10-12 layers"
            shift
            ;;
        --debug)
            # Debug mode: 100 epochs, fast queue, minimal scenes
            DEPTH_CONFIGS=(
                "5,2,5000,1000"      # 10 layers: 100 epochs, eval every 10
                "6,2,5000,1000"      # 12 layers: 100 epochs, eval every 10
                "5,3,5000,1000"      # 15 layers: 100 epochs, eval every 10
                "6,3,5000,1000"      # 18 layers: 100 epochs, eval every 10
            )
            QUEUE="short"  # Base script adds "waic-" prefix automatically
            SCANS="0007"  # Just one scene for debug
            IFS=',' read -ra SCENE_ARRAY <<< "$SCANS"
            echo "DEBUG MODE: 100 epochs, waic-short queue, 1 scene"
            echo "  This is for testing the pipeline, not real results!"
            shift
            ;;
        --ba-only-last-eval)
            BA_ONLY_LAST_EVAL="$2"
            echo "Bundle adjustment only on last eval set to: ${BA_ONLY_LAST_EVAL}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Comprehensive comparison between Baseline ESFM and ESFM+OutlierLoss"
            echo "Runs each experiment (method × depth × scene) as a separate job"
            echo ""
            echo "Options:"
            echo "  --scans LIST        Comma-separated scan list"
            echo "                      Default: 0007,0015,0024,0063,0104,0147,0327,0411,1001"
            echo "  --queue QUEUE       LSF queue selection"
            echo "                      Default: waic-risk (your available queue)"
            echo "                      You can specify different queue if available"
            echo "  --quick             Use reduced epochs for faster testing"
            echo "  --conservative      Test only 10-12 layers"
            echo "  --debug             Debug mode: 10 epochs only, waic-short queue, 1 scene"
            echo "                      For testing pipeline, not real results!"
            echo "  --ba-only-last-eval BOOL"
            echo "                      Run bundle adjustment only on last evaluation"
            echo "                      Values: true or false (default: true)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  # Standard run (uses waic-risk queue)"
            echo "  $0"
            echo ""
            echo "  # Debug mode - test pipeline in minutes"
            echo "  $0 --debug"
            echo ""
            echo "  # Debug mode without bundle adjustment (faster, avoids BA errors)"
            echo "  $0 --debug --ba-only-last-eval false"
            echo ""
            echo "  # Quick test (16 jobs)"
            echo "  $0 --quick --scans 0007,0015"
            echo ""
            echo "  # Conservative (up to 12 layers only)"
            echo "  $0 --conservative"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use $0 --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# GPU SELECTION FUNCTIONS
# ============================================================================

detect_best_gpu_queue() {
    # Tries GPU queues in order of preference: A100 > V100 > RTX > generic
    # Returns the first available queue with reasonable load
    
    local gpu_preferences=("gpu_a100" "gpu.120h" "gpu.large" "gpu_v100" "gpu.24h" "gpu_rtx" "gpu.4h" "gpu" "long")
    
    for queue_name in "${gpu_preferences[@]}"; do
        # Check if queue exists
        if bqueues "$queue_name" &>/dev/null; then
            # Check queue status
            local status=$(bqueues "$queue_name" 2>/dev/null | tail -n1 | awk '{print $2}')
            if [ "$status" = "Open" ] || [ "$status" = "open" ]; then
                echo "$queue_name"
                return 0
            fi
        fi
    done
    
    # Fallback to first GPU queue found
    local first_gpu=$(bqueues 2>/dev/null | grep -i gpu | head -1 | awk '{print $1}')
    if [ -n "$first_gpu" ]; then
        echo "$first_gpu"
        return 0
    fi
    
    # Ultimate fallback
    echo "gpu"
    return 1
}

get_available_gpu_queues() {
    # Returns list of available GPU queues in priority order
    local gpu_preferences=("gpu_a100" "gpu.120h" "gpu.large" "gpu_v100" "gpu.24h" "gpu_rtx" "gpu.4h" "gpu")
    local available_queues=()
    
    for queue_name in "${gpu_preferences[@]}"; do
        if bqueues "$queue_name" &>/dev/null; then
            local status=$(bqueues "$queue_name" 2>/dev/null | tail -n1 | awk '{print $2}')
            if [ "$status" = "Open" ] || [ "$status" = "open" ]; then
                available_queues+=("$queue_name")
            fi
        fi
    done
    
    # If nothing found, get all GPU queues
    if [ ${#available_queues[@]} -eq 0 ]; then
        mapfile -t available_queues < <(bqueues 2>/dev/null | grep -i gpu | awk '{print $1}')
    fi
    
    echo "${available_queues[@]}"
}

# ============================================================================
# APPLY GPU SELECTION STRATEGY
# ============================================================================

if [ "$GPU_STRATEGY" = "auto" ]; then
    echo "Auto-detecting best GPU queue..."
    DETECTED_QUEUE=$(detect_best_gpu_queue)
    if [ -n "$DETECTED_QUEUE" ]; then
        QUEUE="$DETECTED_QUEUE"
        echo "Selected queue: $QUEUE"
    else
        echo "Warning: Could not detect GPU queue, using default: gpu"
        QUEUE="gpu"
    fi
    echo ""
elif [ "$GPU_STRATEGY" = "multi" ]; then
    echo "Multi-queue mode enabled: Jobs will be distributed across available GPU types"
    AVAILABLE_QUEUES=($(get_available_gpu_queues))
    if [ ${#AVAILABLE_QUEUES[@]} -eq 0 ]; then
        echo "Warning: No GPU queues detected, falling back to 'gpu'"
        AVAILABLE_QUEUES=("gpu")
    fi
    echo "Available queues: ${AVAILABLE_QUEUES[@]}"
    echo ""
fi

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo ""
echo "=============================================="
echo "COMPARISON CONFIGURATION"
echo "=============================================="
echo "Methods to compare:"
echo "  1. Baseline ESFM (ReprojectionLoss)"
echo "  2. ESFM + OutlierLoss (AdaptiveConfidenceWeightedOutliersLoss)"
echo ""
echo "Depth configurations:"
for config in "${DEPTH_CONFIGS[@]}"; do
    IFS=',' read -r blocks size epochs evals <<< "$config"
    layers=$((blocks * size))
    echo "  ${layers} layers (${blocks}×${size}): ${epochs} epochs"
done
echo ""
echo "Scenes: ${#SCENE_ARRAY[@]} (${SCANS})"
if [ "$GPU_STRATEGY" = "multi" ]; then
    echo "GPU Strategy: Multi-queue distribution"
    echo "Queues: ${AVAILABLE_QUEUES[@]}"
else
    echo "GPU Strategy: ${GPU_STRATEGY}"
    echo "Queue: ${QUEUE}"
fi
echo ""

# Calculate total experiments
total_experiments=$((${#METHODS[@]} * ${#DEPTH_CONFIGS[@]} * ${#SCENE_ARRAY[@]}))
echo "Total experiments: ${total_experiments}"
echo "  = ${#METHODS[@]} methods × ${#DEPTH_CONFIGS[@]} depths × ${#SCENE_ARRAY[@]} scenes"
echo "=============================================="
echo ""

# ============================================================================
# CONFIRMATION
# ============================================================================

read -p "Proceed with comparison experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================================================
# CREATE RESULTS TRACKING FILE
# ============================================================================

RESULTS_DIR="${REPO_ROOT}/results/comparison"
mkdir -p "${RESULTS_DIR}"

TRACKING_FILE="${RESULTS_DIR}/comparison_tracking_${TIMESTAMP}.csv"

# Create CSV header
echo "timestamp,method,method_name,architecture,loss_function,layers,blocks,size,epochs,eval_intervals,scene,queue,job_id,status" > "${TRACKING_FILE}"

echo ""
echo "Results tracking file: ${TRACKING_FILE}"
echo ""

# ============================================================================
# RUN COMPARISON EXPERIMENTS
# ============================================================================

experiment_num=0
jobs_submitted=0
jobs_failed=0

echo "=============================================="
echo "SUBMITTING COMPARISON EXPERIMENTS"
echo "=============================================="
echo ""

for method_idx in "${!METHODS[@]}"; do
    IFS=',' read -r architecture loss_func <<< "${METHODS[$method_idx]}"
    method_name="${METHOD_NAMES[$method_idx]}"
    
    echo "──────────────────────────────────────────────"
    echo "METHOD: ${method_name}"
    echo "  Architecture: ${architecture}"
    echo "  Loss function: ${loss_func}"
    echo "──────────────────────────────────────────────"
    echo ""
    
    for config in "${DEPTH_CONFIGS[@]}"; do
        IFS=',' read -r blocks size epochs evals <<< "$config"
        layers=$((blocks * size))
        
        echo "  DEPTH: ${layers} layers (${blocks}×${size})"
        echo ""
        
        # Calculate scheduler milestones (50%, 70%, 90%)
        milestone1=$((epochs * 50 / 100))
        milestone2=$((epochs * 70 / 100))
        milestone3=$((epochs * 90 / 100))
        scheduler_milestones="${milestone1},${milestone2},${milestone3}"
        
        for scene in "${SCENE_ARRAY[@]}"; do
            ((experiment_num++))
            
            # Select queue for this job (multi-queue distribution)
            JOB_QUEUE="$QUEUE"
            if [ "$GPU_STRATEGY" = "multi" ] && [ ${#AVAILABLE_QUEUES[@]} -gt 0 ]; then
                # Distribute jobs round-robin across available queues
                # Prioritize faster queues by giving them more jobs
                queue_idx=$(( (experiment_num - 1) % ${#AVAILABLE_QUEUES[@]} ))
                JOB_QUEUE="${AVAILABLE_QUEUES[$queue_idx]}"
            fi
            
            echo "    Scene ${scene} (${experiment_num}/${total_experiments}) → Queue: ${JOB_QUEUE}"
            
            # Build command
            CMD="./run_single_scene_optimization.sh \
                --architecture_type \"${architecture}\" \
                --block_number ${blocks} \
                --block_size ${size} \
                --num_epochs ${epochs} \
                --eval_intervals ${evals} \
                --scheduler_milestone \"${scheduler_milestones}\" \
                --loss_function \"${loss_func}\" \
                --scans \"${scene}\" \
                --queue \"${JOB_QUEUE}\" \
                --ba-only_last_eval \"${BA_ONLY_LAST_EVAL}\""
            
            # Add weights if using OutlierLoss
            if [[ "$loss_func" == "AdaptiveConfidenceWeightedOutliersLoss" ]]; then
                CMD="${CMD} \
                --reproj_loss_weight ${REPROJ_LOSS_WEIGHT} \
                --classification_loss_weight ${CLASSIFICATION_LOSS_WEIGHT}"
            fi
            
            # Submit job
            JOB_OUTPUT=$(eval $CMD 2>&1)
            JOB_STATUS=$?
            
            if [ $JOB_STATUS -eq 0 ]; then
                # Extract job ID
                JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Job <\K[0-9]+' | head -1)
                if [ -z "$JOB_ID" ]; then
                    JOB_ID="submitted"
                fi
                
                echo "      ✓ Job submitted: ${JOB_ID}"
                ((jobs_submitted++))
                
                # Log to tracking file
                echo "${TIMESTAMP},${method_idx},${method_name},${architecture},${loss_func},${layers},${blocks},${size},${epochs},${evals},${scene},${JOB_QUEUE},${JOB_ID},submitted" >> "${TRACKING_FILE}"
            else
                echo "      ✗ Job submission failed"
                echo "      Error output:"
                echo "$JOB_OUTPUT" | head -20  # Show first 20 lines of error
                ((jobs_failed++))
                
                # Log failure
                echo "${TIMESTAMP},${method_idx},${method_name},${architecture},${loss_func},${layers},${blocks},${size},${epochs},${evals},${scene},${JOB_QUEUE},failed,failed" >> "${TRACKING_FILE}"
            fi
            
            # Small delay between submissions
            sleep 1
        done
        echo ""
    done
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "SUBMISSION COMPLETE"
echo "=============================================="
echo "Total experiments: ${total_experiments}"
echo "Jobs submitted: ${jobs_submitted}"
echo "Jobs failed: ${jobs_failed}"
echo ""
echo "Breakdown:"
echo "  Methods: ${#METHODS[@]}"
echo "  Depths: ${#DEPTH_CONFIGS[@]}"
echo "  Scenes: ${#SCENE_ARRAY[@]}"
echo ""
echo "Tracking file: ${TRACKING_FILE}"
echo ""
echo "Monitor jobs with:"
echo "  bjobs"
echo "  bjobs -u \$(whoami) | wc -l  # Count running jobs"
echo ""
echo "View tracking file:"
echo "  cat ${TRACKING_FILE}"
echo "  column -t -s, ${TRACKING_FILE}"
echo ""
echo "=============================================="

# ============================================================================
# CREATE ANALYSIS SCRIPT
# ============================================================================

ANALYSIS_SCRIPT="${RESULTS_DIR}/analyze_comparison_${TIMESTAMP}.py"

cat > "${ANALYSIS_SCRIPT}" << 'ANALYSIS_EOF'
#!/usr/bin/env python3
"""
Analyze ESFM vs ESFM+OutlierLoss comparison results
Searches for results in aggregated directory structure
"""

import pandas as pd
import numpy as np
import sys
import os
import glob
from scipy import stats
from pathlib import Path

def find_results_in_aggregated(base_dir):
    """
    Find results files in the aggregated directory structure.
    
    Structure:
    results/aggregated/num_epochs_{N}_eval{M}/stage1/{architecture}/{blocks}/{loss}/.../*.xlsx
    """
    results = []
    
    # Look for all xlsx files in aggregated directory
    aggregated_dir = os.path.join(base_dir, "aggregated")
    
    if not os.path.exists(aggregated_dir):
        # Try searching from results directory
        aggregated_dir = os.path.join("results", "aggregated")
    
    if not os.path.exists(aggregated_dir):
        print(f"Aggregated directory not found: {aggregated_dir}")
        return []
    
    # Find all xlsx files (looking for Aggregated_across_jobs files)
    xlsx_files = glob.glob(
        os.path.join(aggregated_dir, "**/*.xlsx"),
        recursive=True
    )
    
    print(f"Found {len(xlsx_files)} xlsx files in aggregated directory")
    
    for filepath in xlsx_files:
        try:
            # Extract information from path
            path_parts = Path(filepath).parts
            
            # Get architecture (esfm_deep or esfm_outliers_deep)
            architecture = None
            for part in path_parts:
                if 'esfm_deep' in part or 'esfm_outliers_deep' in part:
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
            
            # Get loss function
            loss_func = None
            for part in path_parts:
                if 'ESFMLoss' in part or 'AdaptiveConfidenceWeightedOutliersLoss' in part:
                    loss_func = part
                    break
            
            if not loss_func:
                continue
            
            # Determine method name
            if architecture == 'esfm_deep':
                method = 'Baseline_ESFM'
            elif architecture == 'esfm_outliers_deep':
                method = 'ESFM_OutlierLoss'
            else:
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
                
                data = {
                    'method': info['method'],
                    'depth': info['depth'],
                    'layers': info['layers'],
                    'blocks': info['blocks'],
                    'size': info['size'],
                    'scene': f"{scene_idx:04d}" if isinstance(scene_idx, int) else str(scene_idx),
                    'trans': row.get('Trans', np.nan),
                    'rot': row.get('Rot', np.nan),
                    'nr': row.get('Nr', np.nan),
                    'conv_time': row.get('Convergence Time', np.nan),
                    'best_epoch': row.get('Best Epoch', np.nan)
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
            
            # Add MEAN row
            detailed_rows.append({
                'method': method,
                'depth': depth_str,
                'scene': 'MEAN',
                'trans': subset['trans'].mean(),
                'rot': subset['rot'].mean(),
                'nr': subset['nr'].mean(),
                'conv_time': subset['conv_time'].mean()
            })
    
    return pd.DataFrame(detailed_rows)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_comparison.py <results_dir>")
        print("Example: python analyze_comparison.py results")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    print("=" * 80)
    print("ESFM vs ESFM+OutlierLoss COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {base_dir}")
    print()
    
    # Find results in aggregated directory
    print("Searching for results in aggregated directory...")
    results_info = find_results_in_aggregated(base_dir)
    
    if not results_info:
        print("\nNo result files found!")
        print("\nSearching in alternative locations...")
        
        # Try from current directory
        results_info = find_results_in_aggregated(".")
        
        if not results_info:
            print("\nStill no results found. Check that jobs have completed.")
            print("\nExpected location: results/aggregated/num_epochs_*/stage1/*/...")
            sys.exit(1)
    
    print(f"\nFound {len(results_info)} result files")
    
    # Show what was found
    print("\nResults found:")
    for info in results_info:
        print(f"  {info['method']:20s} {info['depth']:5s} - {info['filepath']}")
    
    print("\nLoading results data...")
    df = load_results_data(results_info)
    
    if len(df) == 0:
        print("No data loaded!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} result entries")
    print(f"Methods: {df['method'].unique()}")
    print(f"Depths: {sorted(df['depth'].unique())}")
    print(f"Scenes: {sorted(df['scene'].unique())}")
    
    # Calculate comparison
    print("\nCalculating comparison statistics...")
    comparison_df = calculate_comparison(df)
    
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
ANALYSIS_EOF

chmod +x "${ANALYSIS_SCRIPT}"

echo ""
echo "Analysis script created: ${ANALYSIS_SCRIPT}"
echo ""
echo "After experiments complete, analyze results with:"
echo "  python3 ${ANALYSIS_SCRIPT} ${RESULTS_DIR}"
echo ""
echo "This will:"
echo "  - Compare both methods across all depths"
echo "  - Calculate improvement percentages"
echo "  - Perform statistical significance tests"
echo "  - Generate comparison plots"
echo "  - Identify best configurations"
echo ""

