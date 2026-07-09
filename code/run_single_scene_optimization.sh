#!/bin/bash
# cript for running single scan optimization experiments

# Load CUDA module
module load CUDA/11.8.0

# Store timestamp early for use later
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPO_ROOT=$(pwd)
# CRITICAL: Save original arguments BEFORE they get consumed by parsing
ORIGINAL_ARGS=("$@")

# Parse command line arguments
OUTLIER_MODE="with"  # DEFAULT CHANGED: Now defaults to "with" instead of "both"
SKIP_EXISTING=false   # Default: run all regardless of existing results
NUM_EPOCHS="200000"          # Default: 200000 epochs for Trans optimization
EVAL_INTERVALS="1000"        # Default: 1000 for frequent evaluation
SCHEDULER_MILESTONE="60000,100000,150000"  # Default: delayed milestones for longer training
EARLY_STOPPING_PATIENCE="20000"  # Default: 20000 epochs patience
MAX_SCENES=""        # Default: process all datasets
SPECIFIC_SCANS=""   # Default: empty (process all)
WEIGHT_METHOD="global"  # Default: global weight method
ALPHA=""                 # Default: empty (will use default in Python)
STAGE=1              # Default: stage 1 (first stage)
GENERATE_CONFIGS=true  # Default: always generate configs
MODEL="SetOfSetOutliersNet"  # Default model
ARCHITECTURE_TYPE="esfm_orig"  # Default architecture type
BLOCK_NUMBER=""      # Default: empty (will run all combinations)
BLOCK_SIZE=""        # Default: empty (will run all combinations)
QUEUE="long"     # Default: long-gpu queue
INTERACTIVE_MODE=false  # Default: submit jobs to LSF (false), or run directly (true)
VALIDATION_METRIC="our_repro"  # Default: our_repro
BA_ONLY_LAST_EVAL=""  # Default: empty (use config file value)
DEEPER_MODE=false    # Default: standard depth (up to 12 layers)
ALL_DEEP_MODE=false  # Default: false (when true, all combinations up to 18)
MAX_PRODUCT_LIMIT=12 # Default: maximum product of block_number * block_size
MAX_PRODUCT_LIMIT=12 # Default: maximum product of block_number * block_size
RUN_ALL_STATISTICAL=false  # Default: don't run all statistical experiments
LOSS_FUNCTION=""        # Default: empty (will use automatic selection based on stage)
REPROJ_LOSS_WEIGHT="1.0"  # Default: alpha for CombinedLoss
CLASSIFICATION_LOSS_WEIGHT="0.3"  # Default: beta for CombinedLoss
PROGRESSIVE_MODE=false   # Default: no progressive learning
LEARNING_RATE="1e-4"  # Default learning rate
COMPUTE_LOSS_NORMALIZATION=false  # Default: normal training mode
GAMMA=""

while [[ $# -gt 0 ]]; do
    case $1 in
    
        --compute_loss_normalization)
            COMPUTE_LOSS_NORMALIZATION=true
            echo "COMPUTE LOSS NORMALIZATION MODE ENABLED"
            shift
            ;;
        --progressive)
            PROGRESSIVE_MODE=true
            shift
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --loss_function)
            LOSS_FUNCTION="$2"
            shift 2
            ;;
        --reproj_loss_weight)
            REPROJ_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --classification_loss_weight)
            CLASSIFICATION_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --deeper)
            DEEPER_MODE=true
            MAX_PRODUCT_LIMIT=18
            echo "DEEPER MODE ENABLED: Exploring architectures from 12 to 18 total layers"
            shift
            ;;
        --all_deep_combinations)
            ALL_DEEP_MODE=true
            MAX_PRODUCT_LIMIT=18
            echo "ALL DEEP COMBINATIONS MODE ENABLED: Exploring ALL architectures up to 18 total layers"
            shift
            ;;
        --with)
            OUTLIER_MODE="with"
            shift
            ;;
        --without)
            OUTLIER_MODE="without"
            shift
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --eval_intervals)
            EVAL_INTERVALS="$2"
            shift 2
            ;;
        --scheduler_milestone)
            SCHEDULER_MILESTONE="$2"
            # Validate format (should be comma-separated numbers)
            if ! [[ "$2" =~ ^[0-9,]+$ ]]; then
                echo "Error: Invalid scheduler milestone format '$2'"
                echo "Expected format: 60000,100000,150000"
                exit 1
            fi
            shift 2
            ;;
        --early_stopping_patience)
            EARLY_STOPPING_PATIENCE="$2"
            # Validate it's a number
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: early_stopping_patience must be a number"
                exit 1
            fi
            shift 2
            ;;
        --max_scans)
            MAX_SCANS="$2"
            shift 2
            ;;
        --scans)
            SPECIFIC_SCANS="$2"
            shift 2
            ;;
        --weight_method)
            WEIGHT_METHOD="$2"
            # Validate weight method
            if [ "$2" != "std" ] && [ "$2" != "mad" ] && [ "$2" != "huber" ]; then
                echo "Error: Invalid weight method '$2'"
                echo "Valid options: std, mad, huber"
                exit 1
            fi
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"  # 1 or 2
            # Validate that it's either 1 or 2
            if [ "$2" != "1" ] && [ "$2" != "2" ]; then
                echo "Error: Invalid stage '$2'"
                echo "Valid options: 1 (first stage) or 2 (second stage)"
                exit 1
            fi
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --architecture_type)
            ARCHITECTURE_TYPE="$2"
            # Set model based on architecture type mapping
            case "$2" in
                "esfm_orig")
                    MODEL="SetOfSetNet"
                    ;;
                "esfm_outliers")
                    MODEL="SetOfSetOutliersNet"
                    ;;
                "esfm_deep")
                    MODEL="DeepSetOfSetNet"
                    ;;
                "esfm_outliers_deep")
                    MODEL="DeepSetOfSetOutliersNet"
                    ;;
                *)
                    echo "Error: Invalid architecture type '$2'"
                    echo "Valid options: esfm_orig, esfm_outliers, esfm_deep, esfm_outliers_deep"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --config_generation)
            GENERATE_CONFIGS=true
            shift
            ;;
        --block_number)
            BLOCK_NUMBER="$2"
            shift 2
            ;;
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE_MODE=true
            shift
            ;;
        --ba-only_last_eval)
            BA_ONLY_LAST_EVAL="$2"
            # Validate that it's true or false
            if [ "$2" != "true" ] && [ "$2" != "false" ]; then
                echo "Error: --ba-only-last-eval must be 'true' or 'false'"
                exit 1
            fi
            shift 2
            ;;
        --validation_metric)
            VALIDATION_METRIC="$2"
            # Validate that it's one of the allowed metrics
            valid_metrics=("our_repro" "triangulated_repro" "ts_mean" "ts_med" "Rs_mean" "Rs_med" "repro_ba_final" "#registered_cams_final" "ts_ba_final_mean" "ts_ba_final_med" "Rs_ba_final_mean" "Rs_ba_final_med")
            
            # Check if it's a list (starts with '[' or contains comma)
            if [[ "$2" == "["* ]] || [[ "$2" == *","* ]]; then
                # It's a list, don't validate individual items
                VALIDATION_METRIC="$2"
            else
                # Single metric - validate it
                is_valid=false
                for valid in "${valid_metrics[@]}"; do
                    if [ "$2" = "$valid" ]; then
                        is_valid=true
                        break
                    fi
                done
                
                if [ "$is_valid" = false ]; then
                    echo "Error: Invalid validation metric '$2'"
                    echo "Valid options: ${valid_metrics[*]}"
                    exit 1
                fi
            fi
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            # Validate it's a valid number
            if ! [[ "$2" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                echo "Error: --gamma must be a valid number (e.g., 0.1, 0.2, 0.5)"
                exit 1
            fi
            shift 2
            ;;
        --run_all_statistical)
            RUN_ALL_STATISTICAL=true
            shift
            ;;
        --run_all_statistical)
            RUN_ALL_STATISTICAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --deeper              EXPLORE DEEPER ARCHITECTURES from 12 to 18 total layers"
            echo "                        (default: up to 12 layers)"
            echo "  --all_deep_combinations  EXPLORE ALL COMBINATIONS up to 18 total layers (1-18)"
            echo "                        (includes everything from 1 to 18 layers)"
            echo "  --with                Run only with outliers (DEFAULT)"
            echo "  --without             Run only without outliers"
            echo "  --skip-existing       Skip scans where results already exist"
            echo "  --num-epochs N        Override number of epochs (default: from config file)"
            echo "  --max-scans N        Limit number of datasets to process (default: all)"
            echo "  --scans LIST         Comma-separated list of specific scans to run"
            echo "  --weight-method METHOD Weight method: global, track_and_global, remove-outliers, statistical-mad, statistical-std (default: global)"
            echo "  --alpha VALUE         Alpha value for statistical methods (e.g., 1.5, 2.0, 2.5, 3.0, 3.5)"
            echo "                        Required when using statistical-mad or statistical-std"
            echo "  --stage STAGE         Training stage: 1 (first) or 2 (second) (default: 1)"
            echo "  --model MODEL         Model name (default: SetOfSetOutliersNet)"
            echo "  --architecture-type TYPE Architecture type: esfm_orig, esfm_deep, or esfm_orig_deep (default: esfm_orig)"
            echo "                        esfm_deep automatically sets model to DeepSetOfSetOutliersNet"
            echo "                        esfm_deep automatically sets model to DeepSetOfSetNet"
            echo "  --block-number N      Number of blocks in the model"
            echo "  --block-size N        Number of layers per block"
            echo "                        If neither --block-number nor --block-size is specified,"
            echo "                        runs all combinations where both are 1-6 and product ≤ 12"
            echo "                        (or ≤ 18 with --deeper option)"
            echo "  --config-generation automatic config generation (use existing configs)"
            echo "  --queue QUEUE         LSF queue: long-gpu (80G) or short-gpu (40G) (default: long-gpu)"
            echo "  --interactive         Run directly in current session (for interactive jobs)"
            echo "  --eval-intervals N    Evaluation intervals (default: 5000)"
            echo "  --ba-only-last-eval BOOL Run bundle adjustment only at last evaluation (true/false)"
            echo "                        If not specified, uses config file default (False)"
            echo "  --validation-metric METRIC Validation metric for model selection (default: our_repro)"
            echo "                        Valid options: our_repro, triangulated_repro, ts_mean, ts_med,"
            echo "                                      Rs_mean, Rs_med, repro_ba_final, #registered_cams_final,"
            echo "                                      ts_ba_final_mean, ts_ba_final_med, Rs_ba_final_mean,"
            echo "                                      Rs_ba_final_med"
            echo "                        Or a list: '[\"our_repro\",\"ts_mean\"]' (use quotes)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "  --run-all-statistical Run all statistical experiments (MAD and STD with multiple alphas)"
            echo ""
            echo "DEEPER MODE:"
            echo "  When --deeper is specified, the script will explore block combinations"
            echo "  where 12 ≤ block_number * block_size ≤ 18 (instead of the default ≤ 12)."
            echo "  This allows testing deeper architectures beyond your current best (12 layers):"
            echo "    - 6 blocks × 3 layers = 18 total layers"
            echo "    - 9 blocks × 2 layers = 18 total layers"
            echo "    - 4 blocks × 4 layers = 16 total layers"
            echo "    - 7 blocks × 2 layers = 14 total layers"
            echo ""
            echo "Note: By default, only 'with_outliers' configurations are generated and run"
            echo "Note: GPU memory is automatically set - long-gpu uses 80G, short-gpu uses 40G"
            echo ""
            echo "Interactive Mode Usage:"
            echo "  1. First, start an interactive job:"
            echo "     bsub -q waic-short -R rusage[mem=50000] -gpu \"num=1:j_exclusive=yes:gmem=40G\" -Is /bin/bash"
            echo "  2. Then run this script with --interactive flag:"
            echo "     $0 --scans 0007 --num-epochs 10 --interactive"
            echo ""
            echo "  --loss-function FUNC  Override loss function (e.g., AdaptiveConfidenceWeightedOutliersLoss)"
            echo "  --reproj-loss-weight VALUE  Reprojection loss weight (alpha) for CombinedLoss (default: 1.0)"
            echo "  --classification-loss-weight VALUE  Classification loss weight (beta) for CombinedLoss (default: 0.3)"
            echo "  --scheduler-milestone LIST   LR decay milestones (default: 60000,100000,150000)"
            echo "                               Format: comma-separated epochs (e.g., 60000,100000,150000)"
            echo "  --early-stopping-patience NUM Early stopping patience (default: 20000)"
            echo "                               Stop if no improvement for this many epochs (0 to disable)"
            echo "  --compute-loss-normalization  Run loss normalization analysis instead of training"
            echo "                                Uses special script and outputs to lsf_output/compute_loss_normalization/"
            echo "  --gamma VALUE         Override gamma (LR decay factor) (e.g., 0.1, 0.2, 0.5)"
            echo "                        If not specified, uses adaptive calculation based on network depth"
            echo ""
            echo "Examples:"
            echo "  # Use adaptive gamma (default)"
            echo "  $0 --num_epochs 400000"
            echo ""
            echo "  # Override with fixed gamma"
            echo "  $0 --num_epochs 400000 --gamma 0.2"
            echo ""
            echo "  # Different gamma for shallow networks"
            echo "  $0 --block_number 1 --block_size 3 --gamma 0.1"
                                    Adjusts the gamma parameter for loss normalization"
            
            echo "Examples:"
            echo "  # Run first stage with 10k epochs (with_outliers only by default)"
            echo "  $0 --stage 1 --num-epochs 10000 --weight-method global"
            echo ""
            echo "  # Run statistical MAD with alpha=2.0"
            echo "  $0 --stage 2 --num-epochs 10000 --weight-method statistical-mad --alpha 2.0"
            echo ""
            echo "  # Run statistical STD with alpha=1.5"
            echo "  $0 --stage 2 --num-epochs 10000 --weight-method statistical-std --alpha 1.5"
            echo ""
            echo "  # DEEPER MODE: Explore architectures from 12 to 18 layers"
            echo "  $0 --deeper --stage 1 --num-epochs 10000 --architecture-type esfm_outliers_deep"
            echo ""
            echo "  # ALL DEEP COMBINATIONS: Explore ALL architectures up to 18 layers (1-18)"
            echo "  $0 --all_deep_combinations --stage 1 --num-epochs 10000 --architecture-type esfm_outliers_deep"
            echo ""
            echo "  # Run specific deep configuration"
            echo "  $0 --block-number 6 --block-size 3 --num-epochs 10000"
            echo ""
            echo "  # Run in interactive mode (after starting interactive job)"
            echo "  $0 --scans 0007 --num-epochs 10000 --interactive"
            echo ""
            echo "  # Use short-gpu queue (automatically sets 40GB GPU memory)"
            echo "  $0 --scans 0007 --num-epochs 10000 --queue short-gpu"
            echo ""
            echo "  # Use long-gpu queue (automatically sets 80GB GPU memory)"
            echo "  $0 --scans 0007 --num-epochs 10000 --queue long-gpu"
            echo ""
            echo "  # Use custom validation metric"
            echo "  $0 --scans 0007 --num-epochs 10000 --validation-metric ts_mean"
            echo ""
            echo "  # Use multiple validation metrics (average)"
            echo "  $0 --scans 0007 --num-epochs 10000 --validation-metric '[\"our_repro\",\"ts_mean\"]'"
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
# RUN ALL STATISTICAL EXPERIMENTS IF REQUESTED
# ============================================================================
if [ "$RUN_ALL_STATISTICAL" = true ]; then
    echo ""
    echo "############################################################"
    echo "## RUNNING ALL STATISTICAL EXPERIMENTS"
    echo "############################################################"
    echo "This will submit 15 separate jobs:"
    echo "  - MAD with alpha: 1.5, 2.0, 2.5, 3.0, 3.5"
    echo "  - STD with alpha: 1.0, 1.5, 2.0, 2.5, 3.0"
    echo "  - Huber with alpha: 0.5, 1.0, 1.5, 2.0, 2.5"
    echo "############################################################"
    echo ""
    
    # Build base arguments from saved original args
    BASE_ARGS=()
    skip_next=false
    
    
    for arg in "${ORIGINAL_ARGS[@]}"; do
        if [ "$skip_next" = true ]; then
            skip_next=false
            continue
        fi
        
        case "$arg" in
            --run_all_statistical)
                continue
                ;;
            --weight_method|--alpha)
                skip_next=true
                continue
                ;;
            *)
                BASE_ARGS+=("$arg")
                ;;
        esac
    done
    
    echo "Base command for all experiments:"
    echo "$0 ${BASE_ARGS[*]}"
    echo ""
    
    # MAD experiments
    echo "Starting MAD experiments..."
    for alpha in 1.5 2.0 2.5 3.0 3.5; do
        echo ""
        echo "================================================"
        echo "Submitting MAD with alpha=$alpha"
        echo "================================================"
        "$0" "${BASE_ARGS[@]}" --weight-method mad --alpha "$alpha"
        echo "Submitted MAD alpha=$alpha, sleeping 2 seconds..."
        sleep 2
    done
    
    # STD experiments
    echo ""
    echo "Starting STD experiments..."
    for alpha in 1.0 1.5 2.0 2.5 3.0; do
        echo ""
        echo "================================================"
        echo "Submitting STD with alpha=$alpha"
        echo "================================================"
        "$0" "${BASE_ARGS[@]}" --weight-method std --alpha "$alpha"
        echo "Submitted STD alpha=$alpha, sleeping 2 seconds..."
        sleep 2
    done
    
    # Huber experiments
    echo ""
    echo "Starting Huber experiments..."
    for alpha in 0.5 1.0 1.5 2.0 2.5; do
        echo ""
        echo "================================================"
        echo "Submitting Huber with alpha=$alpha"
        echo "================================================"
        "$0" "${BASE_ARGS[@]}" --weight-method huber --alpha "$alpha"
        echo "Submitted Huber alpha=$alpha, sleeping 2 seconds..."
        sleep 2
    done
    
    echo ""
    echo "############################################################"
    echo "## ALL STATISTICAL EXPERIMENTS SUBMITTED"
    echo "############################################################"
    echo "Total experiments submitted: 15"
    echo "  - 5 MAD experiments (alpha: 1.5, 2.0, 2.5, 3.0, 3.5)"
    echo "  - 5 STD experiments (alpha: 1.0, 1.5, 2.0, 2.5, 3.0)"
    echo "  - 5 Huber experiments (alpha: 0.5, 1.0, 1.5, 2.0, 2.5)"
    echo ""
    echo "Monitor jobs with:"
    echo "  bjobs | grep statistical"
    echo "############################################################"
    exit 0
fi

# Auto-adjust OUTLIER_MODE for esfm_orig architecture
if [ "$ARCHITECTURE_TYPE" = "esfm_orig" ] && [ "$OUTLIER_MODE" = "with" ]; then
    OUTLIER_MODE="both"
    echo "Note: For esfm_orig architecture, automatically running BOTH with and without outliers"
    echo ""
fi

# Settings
# REPO_ROOT already set above for aggregation

# Generate block combinations based on architecture type
if [ -z "$BLOCK_NUMBER" ] || [ -z "$BLOCK_SIZE" ]; then
    # If either block parameter is not specified
    if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
        # For esfm_deep and esfm_outliers_deep: generate all valid combinations
        BLOCK_COMBINATIONS=()
        
        # Determine max values based on deeper mode
        if [ "$DEEPER_MODE" = true ]; then
            # For deeper mode, we need to check larger values
            # Since we want products up to 18, max single value would be 18
            max_block_val=18
        else
            # Standard mode - products up to 12
            max_block_val=12
        fi
        
        # Generate combinations
        for block_num in $(seq 1 $max_block_val); do
            for blsock_sz in $(seq 1 $max_block_val); do
                # Check if product is within the valid range
                product=$((block_num * block_sz))
                if [ "$ALL_DEEP_MODE" = true ]; then
                    # All deep mode: all combinations up to 18 layers
                    if [ $product -le 18 ]; then
                        BLOCK_COMBINATIONS+=("${block_num},${block_sz}")
                    fi
                elif [ "$DEEPER_MODE" = true ]; then
                    # For deeper mode: only combinations from 12 to 18 layers
                    if [ $product -ge 12 ] && [ $product -le 18 ]; then
                        BLOCK_COMBINATIONS+=("${block_num},${block_sz}")
                    fi
                else
                    # Standard mode: combinations up to 12 layers
                    if [ $product -le $MAX_PRODUCT_LIMIT ]; then
                        BLOCK_COMBINATIONS+=("${block_num},${block_sz}")
                    fi
                fi
            done
        done
        
        echo "=============================================="
        if [ "$ALL_DEEP_MODE" = true ]; then
            echo "ALL DEEP COMBINATIONS MODE: Block parameters not specified for ${ARCHITECTURE_TYPE}"
            echo "Will run ALL combinations where product ≤ 18"
        elif [ "$DEEPER_MODE" = true ]; then
            echo "DEEPER MODE: Block parameters not specified for ${ARCHITECTURE_TYPE}"
            echo "Will run combinations where 12 ≤ product ≤ 18"
        else
            echo "Block parameters not specified for ${ARCHITECTURE_TYPE} - will run all valid combinations"
            echo "Maximum product limit: ${MAX_PRODUCT_LIMIT}"
        fi
        echo "=============================================="
        echo "Valid combinations (block_number,block_size):"
        
        # Count combinations by total layers
        declare -A layer_counts
        for combo in "${BLOCK_COMBINATIONS[@]}"; do
            IFS=',' read -r bn bs <<< "$combo"
            prod=$((bn * bs))
            ((layer_counts[$prod]++))
        done
        
        # Display summary by layer count
        echo ""
        echo "Summary by total layers:"
        for layers in $(echo "${!layer_counts[@]}" | tr ' ' '\n' | sort -n); do
            echo "  ${layers} layers: ${layer_counts[$layers]} combinations"
        done
        echo ""
        
        # Show some example combinations
        echo "Example combinations:"
        if [ "$ALL_DEEP_MODE" = true ]; then
            # Show examples across the full range
            echo "  1-6 layers:"
            for combo in "${BLOCK_COMBINATIONS[@]}"; do
                IFS=',' read -r bn bs <<< "$combo"
                prod=$((bn * bs))
                if [ $prod -le 6 ]; then
                    echo "    Block Number: $bn, Block Size: $bs (product: $prod)"
                fi
            done | head -3
            echo "  12-18 layers:"
            for combo in "${BLOCK_COMBINATIONS[@]}"; do
                IFS=',' read -r bn bs <<< "$combo"
                prod=$((bn * bs))
                if [ $prod -ge 12 ] && [ $prod -le 18 ]; then
                    echo "    Block Number: $bn, Block Size: $bs (product: $prod)"
                fi
            done | head -5
        elif [ "$DEEPER_MODE" = true ]; then
            # Show interesting deeper combinations
            echo "  13-15 layers:"
            for combo in "${BLOCK_COMBINATIONS[@]}"; do
                IFS=',' read -r bn bs <<< "$combo"
                prod=$((bn * bs))
                if [ $prod -ge 13 ] && [ $prod -le 15 ]; then
                    echo "    Block Number: $bn, Block Size: $bs (product: $prod)"
                fi
            done | head -5
            echo "  16-18 layers:"
            for combo in "${BLOCK_COMBINATIONS[@]}"; do
                IFS=',' read -r bn bs <<< "$combo"
                prod=$((bn * bs))
                if [ $prod -ge 16 ] && [ $prod -le 18 ]; then
                    echo "    Block Number: $bn, Block Size: $bs (product: $prod)"
                fi
            done | head -5
        else
            # Show standard combinations
            for combo in "${BLOCK_COMBINATIONS[@]}"; do
                IFS=',' read -r bn bs <<< "$combo"
                echo "  Block Number: $bn, Block Size: $bs (product: $((bn * bs)))"
            done | head -10
            if [ ${#BLOCK_COMBINATIONS[@]} -gt 10 ]; then
                echo "  ... and $((${#BLOCK_COMBINATIONS[@]} - 10)) more combinations"
            fi
        fi
        
        echo ""
        echo "Total combinations to run: ${#BLOCK_COMBINATIONS[@]}"
        echo "=============================================="
        echo ""
        
        RUN_ALL_COMBINATIONS=true
    else
        # For esfm_orig: use default values (1 block, 3 layers)
        BLOCK_NUMBER=1
        BLOCK_SIZE=3
        BLOCK_COMBINATIONS=("${BLOCK_NUMBER},${BLOCK_SIZE}")
        echo "=============================================="
        echo "Using default block configuration for ${ARCHITECTURE_TYPE}"
        echo "Block Number: ${BLOCK_NUMBER}, Block Size: ${BLOCK_SIZE}"
        echo "=============================================="
        echo ""
        RUN_ALL_COMBINATIONS=false
    fi
else
    # Both parameters specified, use single combination
    # Check if the specified combination exceeds the limit
    specified_product=$((BLOCK_NUMBER * BLOCK_SIZE))
    if [ $specified_product -gt $MAX_PRODUCT_LIMIT ]; then
        echo "WARNING: Specified combination (${BLOCK_NUMBER} × ${BLOCK_SIZE} = ${specified_product}) exceeds limit of ${MAX_PRODUCT_LIMIT}"
        if [ "$DEEPER_MODE" = false ] && [ $specified_product -le 18 ]; then
            echo "Consider using --deeper flag to allow architectures up to 18 layers"
        fi
        echo "Proceeding with specified configuration anyway..."
    fi
    BLOCK_COMBINATIONS=("${BLOCK_NUMBER},${BLOCK_SIZE}")
    RUN_ALL_COMBINATIONS=false
fi

DATASET_DIR="${REPO_ROOT}/datasets/megadepth"
phase="OPTIMIZATION"


# Use NUM_EPOCHS (which now has a default value)
num_epochs="${NUM_EPOCHS}"
epochs_text="${NUM_EPOCHS}epochs"
echo "Using number of epochs: $num_epochs"

# Use EVAL_INTERVALS (which now has a default value)
eval_intervals="${EVAL_INTERVALS}"
eval_text="eval${EVAL_INTERVALS}"
echo "Using eval intervals: $eval_intervals"


# Set eval_intervals from argument or use default
if [ -n "$EVAL_INTERVALS" ]; then
    eval_intervals="$EVAL_INTERVALS"
    eval_text="eval${EVAL_INTERVALS}"
    echo "Using custom eval intervals: $eval_intervals"
else
    eval_intervals=5000  # Default
    eval_text="eval${eval_intervals}"
fi

# ============================================================================
# CREATE AGGREGATED FILE - Now that we have epochs and eval_intervals values
# ============================================================================
# Include deeper mode in the aggregated file path if enabled
if [ "$ALL_DEEP_MODE" = true ]; then
    deeper_suffix="_all_deep18"
elif [ "$DEEPER_MODE" = true ]; then
    deeper_suffix="_deeper18"
else
    deeper_suffix=""
fi

# Create weight method suffix that includes alpha for statistical methods
if [[ "$WEIGHT_METHOD" == "mad" ]] || [[ "$WEIGHT_METHOD" == "std" ]] || [[ "$WEIGHT_METHOD" == "huber" ]]; then
    if [ -z "$ALPHA" ]; then
        echo "Error: --alpha parameter is required when using $WEIGHT_METHOD"
        exit 1
    fi
    # Replace dot with underscore in alpha for directory names
    ALPHA_SAFE=$(echo "$ALPHA" | tr '.' '_')
    WEIGHT_METHOD_DIR="${WEIGHT_METHOD}_alpha_${ALPHA_SAFE}"
else
    WEIGHT_METHOD_DIR="${WEIGHT_METHOD}"
fi

# Create loss function directory name for Stage 1
if [ -n "$LOSS_FUNCTION" ]; then
    LOSS_FUNCTION_DIR="${LOSS_FUNCTION}"
else
    # Default loss function based on stage
    if [ "$STAGE" = "1" ]; then
        LOSS_FUNCTION="ESFMLoss"
        LOSS_FUNCTION_DIR="ESFMLoss"
    else
        LOSS_FUNCTION="ESFMLoss_weighted_by_rep_err"
        LOSS_FUNCTION_DIR="ESFMLoss_weighted_by_rep_err"
    fi
fi


# # # Create safe versions of loss weight values (replace dots with underscores)
REPROJ_LOSS_WEIGHT_SAFE=$(echo "$REPROJ_LOSS_WEIGHT" | tr '.' '_')
CLASSIFICATION_LOSS_WEIGHT_SAFE=$(echo "$CLASSIFICATION_LOSS_WEIGHT" | tr '.' '_')
LR_SAFE=$(echo "$LEARNING_RATE" | tr '.-' '__')

block_combo="${BLOCK_COMBINATIONS//,/_}"
if [ "$STAGE" = "1" ]; then
    AGGREGATED_DIR="${REPO_ROOT}/results/aggregated/num_epochs_${num_epochs}_eval${eval_intervals}/stage${STAGE}/${ARCHITECTURE_TYPE}/block_num_block_size_${block_combo}/progressive_${PROGRESSIVE_MODE}/${LOSS_FUNCTION}/with_outliers/lr${LR_SAFE}"
    mkdir -p "${AGGREGATED_DIR}"
    if [ "$LOSS_FUNCTION" = "CombinedLoss" ] || [ "$LOSS_FUNCTION" = "AdaptiveConfidenceWeightedOutliersLoss" ]; then
        AGGREGATED_FILE="${AGGREGATED_DIR}/Aggregated_across_jobs_${num_epochs}epochs_eval${eval_intervals}_${ARCHITECTURE_TYPE}_block_num_block_size_${block_combo}_${LOSS_FUNCTION}_progressive_${PROGRESSIVE_MODE}_reproj${REPROJ_LOSS_WEIGHT_SAFE}_class${CLASSIFICATION_LOSS_WEIGHT_SAFE}_lr${LR_SAFE}_early_stopping_${EARLY_STOPPING_PATIENCE}_${OUTLIER_MODE}_outliers_${TIMESTAMP}.xlsx"
    else
        AGGREGATED_FILE="${AGGREGATED_DIR}/Aggregated_across_jobs_${num_epochs}epochs_eval${eval_intervals}_lr${LR_SAFE}_${ARCHITECTURE_TYPE}_block_num_block_size_${block_combo}_${LOSS_FUNCTION}_progressive_${PROGRESSIVE_MODE}_lr${LR_SAFE}_early_stopping_${EARLY_STOPPING_PATIENCE}_${OUTLIER_MODE}_outliers_${TIMESTAMP}.xlsx"
    fi

elif [ "$STAGE" = "2" ]; then
    AGGREGATED_DIR="${REPO_ROOT}/results/aggregated/num_epochs_${num_epochs}_eval${eval_intervals}/stage${STAGE}/${ARCHITECTURE_TYPE}/block_num_block_size_${block_combo}/${WEIGHT_METHOD_DIR}/progressive_${PROGRESSIVE_MODE}/${LOSS_FUNCTION}/with_outliers/lr${LR_SAFE}"
    mkdir -p "${AGGREGATED_DIR}"
    if [ "$LOSS_FUNCTION" = "CombinedLoss" ] || [ "$LOSS_FUNCTION" = "AdaptiveConfidenceWeightedOutliersLoss" ]; then
        AGGREGATED_FILE="${AGGREGATED_DIR}/Aggregated_across_jobs_${num_epochs}epochs_eval${eval_intervals}_lr${LR_SAFE}_${ARCHITECTURE_TYPE}_block_num_block_size_${block_combo}_${WEIGHT_METHOD_DIR}_${LOSS_FUNCTION}_progressive_${PROGRESSIVE_MODE}_reproj${REPROJ_LOSS_WEIGHT_SAFE}_class${CLASSIFICATION_LOSS_WEIGHT_SAFE}_lr${LR_SAFE}_early_stopping_${EARLY_STOPPING_PATIENCE}_${TIMESTAMP}.xlsx"
    else
        AGGREGATED_FILE="${AGGREGATED_DIR}/Aggregated_across_jobs_${num_epochs}epochs_eval${eval_intervals}_lr${LR_SAFE}_${ARCHITECTURE_TYPE}_block_num_block_size_${block_combo}_${WEIGHT_METHOD_DIR}_${LOSS_FUNCTION}_progressive_${PROGRESSIVE_MODE}_lr${LR_SAFE}_early_stopping_${EARLY_STOPPING_PATIENCE}_${TIMESTAMP}.xlsx"
    fi
fi



echo ""
echo "=============================================="
echo "Creating aggregated results file"
if [ "$ALL_DEEP_MODE" = true ]; then
    echo "MODE: ALL DEEP COMBINATIONS (1-18 layers)"
elif [ "$DEEPER_MODE" = true ]; then
    echo "MODE: DEEPER (12-18 layers)"
fi
echo "=============================================="
echo "Epochs: ${num_epochs}"
echo "Eval intervals: ${eval_intervals}"
echo "Timestamp: ${TIMESTAMP}"
echo "Aggregated file: ${AGGREGATED_FILE}"

# Create DataFrame with Scene as index to prevent KeyError
# Line 1: Create initial DataFrame with RENAMED columns matching Python script
python3 -c "import pandas as pd; df=pd.DataFrame(columns=['Trans','Rot','Nr','Convergence Time','Best Epoch','model_type','block_size','block_number','weight_method','num_epochs','eval_intervals','stage']); df.index.name='Scene'; df.to_excel('${AGGREGATED_FILE}', index=True)"

if [ $? -eq 0 ]; then
    echo "✓ Successfully created aggregated results file"
else
    echo "⚠ Warning: Could not create aggregated Excel file"
    AGGREGATED_FILE=""
fi
echo "=============================================="
echo ""
# ============================================================================

# ============================================================================
# MAIN EXECUTION LOOP FOR BLOCK COMBINATIONS
# ============================================================================

# Initialize overall counters
overall_total_jobs_submitted=0
overall_total_jobs_skipped=0

# Loop through all block combinations
for block_combo in "${BLOCK_COMBINATIONS[@]}"; do
    # Parse block combination
    IFS=',' read -r BLOCK_NUMBER BLOCK_SIZE <<< "$block_combo"
    
    echo ""
    echo "##############################################"
    echo "## PROCESSING BLOCK COMBINATION"
    echo "## Block Number: ${BLOCK_NUMBER}, Block Size: ${BLOCK_SIZE}"
    echo "## Product: $((BLOCK_NUMBER * BLOCK_SIZE))"
    if [ "$DEEPER_MODE" = true ] && [ $((BLOCK_NUMBER * BLOCK_SIZE)) -gt 12 ]; then
        echo "## [DEEPER MODE: Architecture exceeds standard 12-layer limit]"
    fi
    echo "##############################################"
    echo ""
    
    # ============================================================================
    # DIRECTORY CONFIGURATION (per block combination)
    # ============================================================================
    # Base directories
    CONFS_BASE_DIR="${REPO_ROOT}/confs"
    
    # Create BA subdirectory name based on BA_ONLY_LAST_EVAL setting
    if [ "$BA_ONLY_LAST_EVAL" = "true" ]; then
        BA_SUBDIR="ba_only_last_eval_true"
    elif [ "$BA_ONLY_LAST_EVAL" = "false" ]; then
        BA_SUBDIR="ba_only_last_eval_false"
    else
        # Default when not specified - use config default (False)
        BA_SUBDIR="ba_only_last_eval_false"
    fi
    
    
    # Modify LOG_BASE_DIR based on compute_loss_normalization flag
    if [ "$COMPUTE_LOSS_NORMALIZATION" = true ]; then
        # Special directory for loss normalization outputs
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            LOG_BASE_DIR="${REPO_ROOT}/lsf_output/compute_loss_normalization/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}"
        else
            LOG_BASE_DIR="${REPO_ROOT}/lsf_output/compute_loss_normalization/${ARCHITECTURE_TYPE}"
        fi
        echo "Loss normalization mode - LSF outputs will be saved to: ${LOG_BASE_DIR}"
    else
        # Normal mode - existing logic
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            BLOCK_SUBDIR="block_number_${BLOCK_NUMBER}_block_size_${BLOCK_SIZE}"
            LOG_BASE_DIR="${REPO_ROOT}/lsf_output/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}"
        else
            LOG_BASE_DIR="${REPO_ROOT}/lsf_output/${ARCHITECTURE_TYPE}/${BA_SUBDIR}"
        fi
    fi


    # Set config paths based on stage
    if [ "$STAGE" = "1" ]; then
        # Stage 1: Use LOSS_FUNCTION_DIR in path
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            CONFS_WITH_OUTLIERS="${CONFS_BASE_DIR}/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/with_outliers/confs_esfm_${epochs_text}_${eval_text}_with_outliers"
            CONFS_WITHOUT_OUTLIERS="${CONFS_BASE_DIR}/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/without_outliers/confs_esfm_${epochs_text}_${eval_text}_without_outliers"
        else
            CONFS_WITH_OUTLIERS="${CONFS_BASE_DIR}/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/with_outliers/confs_esfm_${epochs_text}_with_outliers"
            CONFS_WITHOUT_OUTLIERS="${CONFS_BASE_DIR}/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/without_outliers/confs_esfm_${epochs_text}_without_outliers"
        fi
    elif [ "$STAGE" = "2" ]; then
        # Stage 2: Use WEIGHT_METHOD_DIR in path
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            CONFS_WITH_OUTLIERS="${CONFS_BASE_DIR}/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${WEIGHT_METHOD}/with_outliers/confs_esfm_${epochs_text}_${eval_text}_with_outliers"
        else
            CONFS_WITH_OUTLIERS="${CONFS_BASE_DIR}/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${WEIGHT_METHOD}/with_outliers/confs_esfm_${epochs_text}_with_outliers"
        fi
    fi

echo "Directory configuration:"
echo "  BA setting: ${BA_SUBDIR}"
if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
    echo "  Config (with outliers): ${CONFS_WITH_OUTLIERS}"
fi
if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
    echo "  Config (without outliers): ${CONFS_WITHOUT_OUTLIERS}"
fi
echo "  Results: ${RESULTS_DIR}"
echo "  LSF output: ${LOG_BASE_DIR}"

# ============================================================================
# EMBEDDED CONFIG GENERATOR FUNCTION
# ============================================================================

generate_config_file() {
    local scan="$1"
    local model="$2"
    local num_epochs="$3"
    local eval_intervals="$4"
    local validation_metric="$5"
    local weight_method="$6"
    local stage="$7"
    local outlier_mode="$8"  # with_outliers or removed_outliers
    local output_file="$9"
    
    # Format BA_ONLY_LAST_EVAL for Python (capitalize first letter)
    local ba_only_last_eval_formatted="False"
    local run_ba_formatted="True"  # Default: run BA
    
    if [ -n "$BA_ONLY_LAST_EVAL" ]; then
        if [ "$BA_ONLY_LAST_EVAL" = "true" ]; then
            ba_only_last_eval_formatted="True"
            run_ba_formatted="True"  # Run BA, but only on last eval
        else
            ba_only_last_eval_formatted="False"
            run_ba_formatted="False"  # Don't run BA at all
        fi
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"
    
    # Determine if outliers should be removed
    local remove_outliers="false"
    if [ "$outlier_mode" = "without_outliers" ]; then
        remove_outliers="true"
    fi
    
    # Set scheduler milestones - same for both stages
    local scheduler_milestone="[$SCHEDULER_MILESTONE]"
 

    # Set loss function - use provided loss function or default based on stage and weight method
    local loss_func
    
    if [ -n "$LOSS_FUNCTION" ]; then
            loss_func="$LOSS_FUNCTION"
        echo "Using user-specified loss function: ${loss_func}"
        
    else
        # Default to ESFMLoss for all stages
        loss_func="ESFMLoss"
        echo "Using default loss function: ESFMLoss"
    fi
    
    # Generate experiment name
    local exp_name="${scan}_${ARCHITECTURE_TYPE}_${stage}_stage_${num_epochs}epochs_${eval_intervals}_${weight_method}_${outlier_mode}"

    # Format validation_metric as a list (always)
    if [[ "${validation_metric}" == "["* ]]; then
        local validation_metric_formatted="${validation_metric}"
    else
        local validation_metric_formatted="[\"${validation_metric}\"]"
    fi
    
    # Calculate adaptive dropout based on network depth
    # Formula: p = 1 - (target_retention)^(1/n)
    # Target: Keep 65% information retention across network
    local total_layers=$((BLOCK_NUMBER * BLOCK_SIZE))
    local dropout_rate

    dropout_rate=0.1
    

   

    # Calculate adaptive gamma (LR decay factor) based on network depth
    local gamma

    # Check if gamma was provided via command line
    if [ -n "$GAMMA" ]; then
        # User provided gamma via --gamma flag
        gamma="$GAMMA"
        echo "Using user-specified gamma = $gamma (overriding adaptive calculation)"
    else
        # Adaptive calculation
        if [ $total_layers -le 3 ]; then
            gamma=0.2
        else
            # Use bc for calculation
            gamma=$(echo "scale=4; 
                x = ($total_layers - 6) / 8; 
                ex = e(x); 
                emx = e(-x); 
                tanh = (ex - emx) / (ex + emx); 
                result = 0.2 + 0.15 * tanh; 
                if (result > 0.5) result = 0.5; 
                if (result < 0.15) result = 0.15; 
                result" | bc -l)
            
            # Validate bc result
            if ! [[ "$gamma" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                echo "Warning: Gamma calculation failed, using default 0.2"
                gamma=0.2
            fi
        fi
        echo "Using adaptive gamma = $gamma for $total_layers layers"
    fi

    # Set configuration-specific results directory
    local outlier_dir
    if [ "$outlier_mode" = "with_outliers" ]; then
        outlier_dir="with_outliers"
    else
        outlier_dir="without_outliers"
    fi
    

    if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
        local config_results_path="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${stage}_stage/${LOSS_FUNCTION}/${WEIGHT_METHOD}/${outlier_dir}/${num_epochs}epochs_eval${eval_intervals}"
    else
        local config_results_path="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${stage}_stage/${LOSS_FUNCTION}/${WEIGHT_METHOD}/${outlier_dir}/${num_epochs}epochs_eval${eval_intervals}"
    fi


# Set stage and output_mode based on loss function
    local config_stage="${stage}"
    local output_mode=1
    # if [ "$LOSS_FUNCTION" = "adaptive_confidence_loss" ] || [ "$LOSS_FUNCTION" = "adaptive_confidence_combined_loss" ]; then
    if [ "$LOSS_FUNCTION" = "CombinedLoss" ] || [ "$LOSS_FUNCTION" = "AdaptiveConfidenceWeightedOutliersLoss" ]; then
        output_mode=3
    fi

    # Start writing the config file in pyhocon format
    cat > "$output_file" << EOF
exp_name = ${exp_name}
results_path = "${config_results_path}/${scan}_ba"
num_iter = 1
random_seed = 20
num_jobs=1
num_gpus=1
general
{
    stage = ${config_stage}
}
dataset
{
    dataset = "megadepth"
    use_gt = False
    calibrated = True
    remove_outliers_gt = ${remove_outliers}
      
    # Single scan (for single_scan_optimization.py)
    scan = "${scan}"
}
model
{
    type = "SetOfSet.${model}"
    num_features = 256
    num_blocks = ${BLOCK_NUMBER}
    block_size = ${BLOCK_SIZE}
    use_skip = False
    multires = 0
    use_progressive = False
    use_layer_norm = True
    use_residual = True
    dropout_rate = ${dropout_rate}
}
EOF

cat >> "$output_file" << EOF
train
{
    lr = ${LEARNING_RATE}
    num_epochs = ${num_epochs}
    scheduler_milestone = ${scheduler_milestone}
    gamma = ${gamma}
    eval_intervals = ${eval_intervals}
    early_stopping_patience = ${EARLY_STOPPING_PATIENCE}
    output_mode = ${output_mode}
    validation_metric = ${validation_metric_formatted}
EOF

    # Add loss weights to train section for CombinedLoss
    if [ "$LOSS_FUNCTION" = "CombinedLoss" ] || [ "$LOSS_FUNCTION" = "AdaptiveConfidenceWeightedOutliersLoss" ]; then
        cat >> "$output_file" << EOF
    reproj_loss_weight = ${REPROJ_LOSS_WEIGHT}
    classification_loss_weight = ${CLASSIFICATION_LOSS_WEIGHT}
EOF
    fi

    cat >> "$output_file" << EOF
}

loss
{
    func = ${loss_func}
    infinity_pts_margin = 1e-4
    normalize_grad = True
    hinge_loss = True
    hinge_loss_weight = 1
EOF

    # Add loss weights for CombinedLoss
    if [ "$LOSS_FUNCTION" = " CombinedLoss" ]; then
        cat >> "$output_file" << EOF
    reproj_loss_weight = ${REPROJ_LOSS_WEIGHT}
    classification_loss_weight = ${CLASSIFICATION_LOSS_WEIGHT}
EOF
    fi

    cat >> "$output_file" << EOF
}
EOF


    # Conditionally append postprocessing (ONLY for stage 2)
    if [ "$stage" = "2" ]; then
        # Check if statistical method and add alpha
        if [[ "${weight_method}" == "mad" ]] || [[ "${weight_method}" == "std" ]] || [[ "${weight_method}" == "huber" ]]; then
            cat >> "$output_file" << EOF
postprocessing
{
    weight_method = "${weight_method}"
    alpha = ${ALPHA}
}
EOF
        else
            cat >> "$output_file" << EOF
postprocessing
{
    weight_method = "${weight_method}"
}
EOF
        fi
    fi

    # Always append ba section
    cat >> "$output_file" << EOF
ba
{
    run_ba = ${run_ba_formatted}
    repeat = True
    triangulation = False
    only_last_eval = ${ba_only_last_eval_formatted}
    filter_outliers = 4.0
}
EOF
    
    echo "Generated config: $output_file"
    echo "  Results will be saved to: ${config_results_path}/${scan}_ba"
}
# ============================================================================
# CONFIG GENERATION (automatic by default)
# ============================================================================
if [ "$GENERATE_CONFIGS" = true ]; then
    echo ""
    echo "=============================================="
    echo "Generating Configuration Files (Automatic)"
    if [ "$ALL_DEEP_MODE" = true ]; then
        echo "ALL DEEP COMBINATIONS MODE: Configurations for ALL architectures up to 18 layers"
    elif [ "$DEEPER_MODE" = true ]; then
        echo "DEEPER MODE ENABLED: Configurations for 12-18 layer architectures"
    fi
    echo "=============================================="
    echo "Stage: ${STAGE}"
    echo "Epochs: ${num_epochs}"
    echo "Evaluation interval: ${eval_intervals}"
    echo "Validation metric: ${VALIDATION_METRIC}"
    # Loss function will be set based on stage in generate_config_file
    echo "Weight method: ${WEIGHT_METHOD}"
    if [[ "$WEIGHT_METHOD" == "mad" ]] || [[ "$WEIGHT_METHOD" == "std" ]] || [[ "$WEIGHT_METHOD" == "huber" ]]; then
        echo "Alpha: ${ALPHA}"
    fi
    echo "Model: ${MODEL}"
    echo "Architecture: ${BLOCK_NUMBER} blocks × ${BLOCK_SIZE} layers = $((BLOCK_NUMBER * BLOCK_SIZE)) total layers"
    # Calculate and display dropout rate
    total_layers=$((BLOCK_NUMBER * BLOCK_SIZE))
    if [ $total_layers -le 3 ]; then
        calc_dropout=0.1
    else
        calc_dropout=$(echo "scale=4; 1 - e(l(0.65)/$total_layers)" | bc -l)
        if (( $(echo "$calc_dropout < 0.02" | bc -l) )); then
            calc_dropout=0.02
        elif (( $(echo "$calc_dropout > 0.1" | bc -l) )); then
            calc_dropout=0.1
        fi
    fi
    effective_retention=$(echo "scale=2; (1 - $calc_dropout)^$total_layers * 100" | bc -l)
    echo "Dropout rate: ${calc_dropout} (effective retention: ${effective_retention}%)"
    echo "Outlier mode: ${OUTLIER_MODE}"
    if [ -n "$BA_ONLY_LAST_EVAL" ]; then
        echo "BA only last eval: ${BA_ONLY_LAST_EVAL}"
    else
        echo "BA only last eval: (using config default: False)"
    fi
    echo ""
    
    # Create config directories if they don't exist - ONLY for requested mode
    echo "Creating config directories if needed..."
    if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
        mkdir -p "${CONFS_WITH_OUTLIERS}"
        echo "  Created/verified: ${CONFS_WITH_OUTLIERS}"
    fi
    if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
        mkdir -p "${CONFS_WITHOUT_OUTLIERS}"
        echo "  Created/verified: ${CONFS_WITHOUT_OUTLIERS}"
    fi
    echo ""
    
    # Check if directory exists
    if [ ! -d "$DATASET_DIR" ]; then
        echo "Error: Directory $DATASET_DIR not found!"
        echo "Please update the DATASET_DIR variable with the correct path"
        exit 1
    fi
    
    # Process dataset selection for config generation
    if [ -n "$SPECIFIC_SCANS" ]; then
        # User specified specific scans
        echo "Generating configs for user-specified scans: ${SPECIFIC_SCANS}"
        
        # Convert comma-separated string to array
        IFS=',' read -ra scans <<< "$SPECIFIC_SCANS"
        
        # Trim whitespace from each element
        for i in "${!scans[@]}"; do
            scans[$i]=$(echo "${scans[$i]}" | xargs)
        done
    else
        # Read scan names from .npz files in the directory
        echo "Reading scans from: $DATASET_DIR"
        scans=()
        
        for file in "$DATASET_DIR"/*.npz; do
            if [ -f "$file" ]; then
                # Extract filename without path and extension
                filename=$(basename "$file" .npz)
                scans+=("$filename")
                echo "Found scan: $filename"
            fi
        done
    fi
    
    # Check if any scans were found
    if [ ${#scans[@]} -eq 0 ]; then
        echo "No scans to process for config generation"
        exit 1
    fi
    
    echo ""
    echo "================================================"
    echo "Configuration Summary:"
    echo "================================================"
    echo "Found ${#scans[@]} scans total"
    echo "Training stage: $STAGE"
    echo "Model: $MODEL"
    echo "Block configuration: ${BLOCK_NUMBER} blocks, ${BLOCK_SIZE} layers per block"
    echo "Total layers: $((BLOCK_NUMBER * BLOCK_SIZE))"
    if [ "$DEEPER_MODE" = true ] && [ $((BLOCK_NUMBER * BLOCK_SIZE)) -gt 12 ]; then
        echo "  [DEEPER: Exceeds standard 12-layer limit]"
    fi
    echo "Number of epochs: $num_epochs"
    echo "Eval intervals: $eval_intervals"
    echo "Validation metric: $VALIDATION_METRIC"
    echo "Weight method: $WEIGHT_METHOD"
    if [[ "$WEIGHT_METHOD" == "mad" ]] || [[ "$WEIGHT_METHOD" == "std" ]] || [[ "$WEIGHT_METHOD" == "huber" ]]; then
        echo "Alpha: $ALPHA"
    fi
    echo "Configuration mode: $OUTLIER_MODE"
    echo "================================================"
    echo ""
    
    # Loop through each scan for config generation
    for scan in "${scans[@]}"; do
        echo "Generating configs for scan: $scan (Stage: $STAGE)"
        
        # Generate config with outliers ONLY if requested
        if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
            config_with="${CONFS_WITH_OUTLIERS}/${scan}_w_outliers.conf"
            generate_config_file "$scan" "$MODEL" "$num_epochs" "$eval_intervals" "$VALIDATION_METRIC" "$WEIGHT_METHOD" "$STAGE" "with_outliers" "$config_with"
        fi
        
        # Generate config without outliers ONLY if requested
        if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
            config_without="${CONFS_WITHOUT_OUTLIERS}/${scan}_without_outliers.conf"
            generate_config_file "$scan" "$MODEL" "$num_epochs" "$eval_intervals" "$VALIDATION_METRIC" "$WEIGHT_METHOD" "$STAGE" "without_outliers" "$config_without"
        fi
    done
    
    echo ""
    echo "================================================"
    echo "All configs generated for stage: $STAGE!"
    echo "================================================"
    echo ""
    echo "Generated configs in:"
    if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
        echo "  - ${CONFS_WITH_OUTLIERS}/"
    fi
    if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
        echo "  - ${CONFS_WITHOUT_OUTLIERS}/"
    fi
    echo ""
    
    # Check and list configs with outliers
    if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
        if [ -d "$CONFS_WITH_OUTLIERS" ]; then
            echo "Configs with outliers:"
            ls -la "$CONFS_WITH_OUTLIERS" | head -10
            num_files=$(ls -1 "$CONFS_WITH_OUTLIERS" | wc -l)
            echo "Total files: $num_files"
            echo ""
        fi
    fi
    
    # Check and list configs without outliers
    if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
        if [ -d "$CONFS_WITHOUT_OUTLIERS" ]; then
            echo "Configs without outliers:"
            ls -la "$CONFS_WITHOUT_OUTLIERS" | head -10
            num_files=$(ls -1 "$CONFS_WITHOUT_OUTLIERS" | wc -l)
            echo "Total files: $num_files"
            echo ""
        fi
    fi
    
    echo "================================================"
    echo "Config generation completed successfully"
    echo "================================================"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "Skipping Config Generation (--config-generation flag used)"
    echo "=============================================="
    echo "Using existing config files from:"
    if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
        echo "  ${CONFS_WITH_OUTLIERS}"
    fi
    if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
        echo "  ${CONFS_WITHOUT_OUTLIERS}"
    fi
    echo ""
    echo "Warning: Script will fail if configs don't exist!"
    echo "=============================================="
    echo ""
fi

# ============================================================================
# JOB SUBMISSION
# ============================================================================

# ============================================================================
# CHANGE 3: Remove weight method from LOG_BASE_DIR creation for Stage 1
# ============================================================================
# Create necessary base directories

# Create necessary base directories
echo "Creating base directories for stage: ${STAGE}"
if [ "$STAGE" = "1" ]; then
    mkdir -p "${LOG_BASE_DIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/num_epochs_${num_epochs}_eval${eval_intervals}"
else
    mkdir -p "${LOG_BASE_DIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/num_epochs_${num_epochs}_eval${eval_intervals}"
fi

# ============================================================================
# CHANGE 4: Remove weight method from LOG_PATH verification for Stage 1
# ============================================================================

# Note: Scene-specific log directories will be created for each scan
if [ "$STAGE" = "1" ]; then
    LOG_PATH="${LOG_BASE_DIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/num_epochs_${num_epochs}_eval${eval_intervals}"
    echo "Base log directory structure: ${LOG_PATH}/scan_*/"
else
    LOG_PATH="${LOG_BASE_DIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/num_epochs_${num_epochs}_eval${eval_intervals}"
    echo "Base log directory structure: ${LOG_PATH}/scan_*/"
fi

echo "Log base directory created/verified:"
echo "  ${LOG_PATH}"
echo "  Scene subdirectories will be created as: scan_<name>/"
echo "  Log files will be: *.out (output) and *.err (error)"

# Clear previous LSF output files for this stage and epochs combination
if [ "$STAGE" = "1" ]; then
    echo "Clearing previous LSF output files for stage '${STAGE}' and epochs '${num_epochs}'..."
else
    echo "Clearing previous LSF output files for stage '${STAGE}', weight method '${WEIGHT_METHOD}' and epochs '${num_epochs}'..."
fi

if [ -d "${LOG_PATH}" ]; then
    echo "  Clearing: ${LOG_PATH}/scan_*/*.err"
    echo "  Clearing: ${LOG_PATH}/scan_*/*.out"
    find "${LOG_PATH}" -type f \( -name "*.err" -o -name "*.out" \) -delete 2>/dev/null || true
fi

cd "${REPO_ROOT}"

# Check if LSF is available
if ! command -v bsub &> /dev/null; then
    echo "ERROR: bsub command not found. LSF might not be available on this system."
    echo "Please ensure you're running this on a system with LSF installed."
    echo "You might need to load an LSF module first:"
    echo "  module load lsf"
    exit 1
fi

echo "LSF is available. Checking queue status..."
bqueues -u $(whoami) 2>/dev/null || echo "Warning: Could not check queue status"

# Process dataset selection for job submission
if [ -n "$SPECIFIC_SCANS" ]; then
    # User specified specific scans to run
    echo "Processing user-specified scans: ${SPECIFIC_SCANS}"
    
    # Convert comma-separated string to array
    IFS=',' read -ra datasets <<< "$SPECIFIC_SCANS"
    
    # Trim whitespace from each element
    for i in "${!datasets[@]}"; do
        datasets[$i]=$(echo "${datasets[$i]}" | xargs)
    done
    
    # Check if any scans were provided
    if [ ${#datasets[@]} -eq 0 ]; then
        echo "Error: No scans provided in the list"
        exit 1
    fi
    
    echo "Will process ${#datasets[@]} specified scan(s):"
    for scan in "${datasets[@]}"; do
        echo "  - ${scan}"
    done
    
else
    # Read all dataset names from .npz files
    echo "Reading datasets from: ${DATASET_DIR}"
    datasets=()
    
    for file in "${DATASET_DIR}"/*.npz; do
        if [ -f "$file" ]; then
            # Extract filename without path and extension
            filename=$(basename "$file" .npz)
            datasets+=("$filename")
        fi
    done
    
    # Check if any datasets were found
    if [ ${#datasets[@]} -eq 0 ]; then
        echo "Error: No .npz files found in ${DATASET_DIR}"
        exit 1
    fi
    
    # Limit datasets if MAX_SCENES is set
    if [ -n "$MAX_SCANS" ]; then
        if [ "$MAX_SCANS" -lt ${#datasets[@]} ]; then
            echo "Limiting to first ${MAX_SCANS} datasets (out of ${#datasets[@]} total)"
            datasets=("${datasets[@]:0:$MAX_SCANS}")
        else
            echo "MAX_SCENES (${MAX_SCANS}) >= total datasets (${#datasets[@]}), using all"
        fi
    fi
fi

echo ""
echo "=============================================="
echo "Running Experiments"
echo "       High Memory GPU (40GB) Request"
echo "=============================================="
echo "Stage: ${STAGE}"
echo "Mode: ${OUTLIER_MODE}"
echo "Epochs: ${num_epochs}"
echo "Weight Method: ${WEIGHT_METHOD}"
if [[ "$WEIGHT_METHOD" == "mad" ]] || [[ "$WEIGHT_METHOD" == "std" ]] || [[ "$WEIGHT_METHOD" == "huber" ]]; then
    echo "Alpha: ${ALPHA}"
fi
echo "Model: ${MODEL}"
if [ "$SKIP_EXISTING" = true ]; then
    echo "Skip mode: Will skip scans with existing results"
else
    echo "Skip mode: Will run all scans (even if results exist)"
fi
if [ "$OUTLIER_MODE" = "with" ]; then
    echo "Configuration: WITH outliers only (DEFAULT)"
elif [ "$OUTLIER_MODE" = "without" ]; then
    echo "Configuration: WITHOUT outliers only (removed)"
else
    echo "Configuration: BOTH with and without outliers"
fi
if [ -n "$SPECIFIC_SCANS" ]; then
    echo "Scene selection: User-specified list"
elif [ -n "$MAX_SCANS" ]; then
    echo "Scene selection: First ${MAX_SCANS} scans"
else
    echo "Scene selection: All available scans"
fi
echo "Datasets to process: ${#datasets[@]}"
echo "Dataset names:"
for dataset in "${datasets[@]}"; do
    echo "  - ${dataset}"
done
echo "Phase: ${phase}"
echo "GPU Memory: 40GB (typically A100)"
echo ""
echo "--- CONFIG DIRECTORIES ---"
if [ "$OUTLIER_MODE" = "with" ] || [ "$OUTLIER_MODE" = "both" ]; then
    echo "With outliers: ${CONFS_WITH_OUTLIERS}"
fi
if [ "$OUTLIER_MODE" = "without" ] || [ "$OUTLIER_MODE" = "both" ]; then
    echo "Without outliers: ${CONFS_WITHOUT_OUTLIERS}"
fi
echo ""
echo "--- FILE LOCATIONS ---"
echo "Results directory: Configuration-specific (with_outliers or without_outliers folders)"
echo "LSF logs directory structure:"
echo "  ${LOG_PATH}/scan_<name>/"
echo "=============================================="
echo ""

# Counter for submitted jobs
total_jobs_submitted=0
total_jobs_skipped=0

echo ""
echo "=============================================="
echo "Starting Single Scene Optimization"
echo "=============================================="
echo "Architecture Type: ${ARCHITECTURE_TYPE}"
echo "Block Configuration: ${BLOCK_NUMBER} blocks, ${BLOCK_SIZE} layers per block"
if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
    echo "  -> LSF outputs will be in: lsf_output/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/"
    echo "  -> Results will be in: results/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/"
else
    echo "  -> LSF outputs will be in: lsf_output/${ARCHITECTURE_TYPE}/"
    echo "  -> Results will be in: results/${ARCHITECTURE_TYPE}/"
fi
echo "=============================================="
echo ""


configs_to_run=()

# Check if loss function is outlier-specific
# OUTLIER_SPECIFIC_LOSSES=("AdaptiveConfidenceWeightedOutliersLoss" "ConfidenceWeightedOutliersLoss" "OutlierSupervisionLoss")
OUTLIER_SPECIFIC_LOSSES=("adaptive_confidence_combined_loss")
IS_OUTLIER_LOSS=false
for loss_name in "${OUTLIER_SPECIFIC_LOSSES[@]}"; do
    if [ "$LOSS_FUNCTION" = "$loss_name" ]; then
        IS_OUTLIER_LOSS=true
        break
    fi
done

# If outlier-specific loss is specified, only run with_outliers
if [ "$IS_OUTLIER_LOSS" = true ]; then
    configs_to_run=("with_outliers")
    echo "Note: Using outlier-specific loss function '${LOSS_FUNCTION}' - running ONLY with_outliers configuration"
elif [ "$STAGE" = "2" ]; then
    # Stage 2: Only run with_outliers
    configs_to_run=("with_outliers")
elif [ "$STAGE" = "1" ]; then
    # Stage 1: By default run both with and without outliers
    configs_to_run=("with_outliers" "without_outliers")
elif [ "$OUTLIER_MODE" = "with" ]; then
    configs_to_run=("with_outliers")
elif [ "$OUTLIER_MODE" = "without" ]; then
    configs_to_run=("without_outliers")
elif [ "$OUTLIER_MODE" = "both" ]; then
    configs_to_run=("with_outliers" "without_outliers")
fi


PYTHON_SCRIPT="${REPO_ROOT}/single_scene_optimization.py"


# Run for selected configurations
for outlier_config in "${configs_to_run[@]}"; do
    
    echo ""
    echo "=============================================="
    

    if [ "$outlier_config" = "with_outliers" ]; then
        echo "PROCESSING: WITH OUTLIERS (Stage: ${STAGE})"
        CONFS_DIR="${CONFS_WITH_OUTLIERS}"
        config_suffix="w_outliers"
        
        # Set configuration-specific results directory
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            if [ "$STAGE" = "1" ]; then
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/with_outliers/${epochs_text}_${eval_text}"
            else
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/with_outliers/${epochs_text}_${eval_text}"
            fi
        else
            if [ "$STAGE" = "1" ]; then
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/with_outliers/${epochs_text}_${eval_text}"
            else
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/with_outliers/${epochs_text}_${eval_text}"
            fi
        fi
    else
        echo "PROCESSING: WITHOUT OUTLIERS (REMOVED) (Stage: ${STAGE})"
        CONFS_DIR="${CONFS_WITHOUT_OUTLIERS}"
        config_suffix="without_outliers"
        
        # Set configuration-specific results directory
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            if [ "$STAGE" = "1" ]; then
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/without_outliers/${epochs_text}_${eval_text}"
            else
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/without_outliers/${epochs_text}_${eval_text}"
            fi
        else
            if [ "$STAGE" = "1" ]; then
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/without_outliers/${epochs_text}_${eval_text}"
            else
                CONFIG_RESULTS_DIR="${REPO_ROOT}/results/${ARCHITECTURE_TYPE}/${BA_SUBDIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/without_outliers/${epochs_text}_${eval_text}"
            fi
        fi
    fi
    echo "Config directory: ${CONFS_DIR}"
    echo "Config suffix: ${config_suffix}"
    echo "Weight method: ${WEIGHT_METHOD}"
    if [[ "$WEIGHT_METHOD" == "mad" ]] || [[ "$WEIGHT_METHOD" == "std" ]] || [[ "$WEIGHT_METHOD" == "huber" ]]; then
        echo "Alpha: ${ALPHA}"
    fi
    echo "Architecture type: ${ARCHITECTURE_TYPE}"
    echo "Python script: ${PYTHON_SCRIPT}"
    echo "Results directory: ${CONFIG_RESULTS_DIR}"
    echo "=============================================="
    echo ""
    
    mkdir -p "${CONFIG_DIR}"
    # Check if this specific config directory exists
    if [ ! -d "${CONFS_DIR}" ]; then
        echo "Warning: Config directory not found: ${CONFS_DIR}"
        echo "Skipping ${outlier_config} configuration..."
        continue
    fi
    
    jobs_submitted=0
    jobs_skipped=0
    
    # Loop through datasets
    for dataset_name in "${datasets[@]}"; do
        SCAN="${dataset_name}"
        echo "Dataset: ${dataset_name} (${outlier_config}, weight=${WEIGHT_METHOD}, stage=${STAGE})"
        echo "----------------------------------------"
        
        # Create safe name (replace spaces with underscores)
        safe_name="${dataset_name// /_}"
        
        # Config file path - simplified naming without epochs in filename
        config_file="${CONFS_DIR}/${safe_name}_${config_suffix}.conf"
        
        # Check if config exists
        if [ ! -f "${config_file}" ]; then
            echo "  Warning: Config not found: ${config_file}"
            echo "  Looking for alternative naming..."
            
            # Try without safe_name conversion (in case dataset has no spaces)
            alt_config_file="${CONFS_DIR}/${SCAN}_${config_suffix}.conf"
            if [ -f "${alt_config_file}" ]; then
                config_file="${alt_config_file}"
                echo "  Found: ${config_file}"
            else
                echo "  No config found. Config should have been generated automatically."
                echo "  Check that config generation completed successfully."
                echo ""
                ((jobs_skipped++))
                ((total_jobs_skipped++))
                continue
            fi
        else
            echo "  Using config: ${config_file}"
        fi
        
        # Debug: Show the results_path from the config
        echo "  Checking results_path in config:"
        grep "results_path" "${config_file}" | head -1
        
        # Note: RESULTS_DIR already includes the full path with architecture_type, BA setting, stage, weight_method, outlier mode, and epochs
        # Set exp_version to include architecture type and BA subdirectory
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            results_exp_version="${ARCHITECTURE_TYPE}/${BLOCK_SUBDIR}/${BA_SUBDIR}"
        else
            results_exp_version="${ARCHITECTURE_TYPE}/${BA_SUBDIR}"
        fi

        # If current_exp_version exists, prepend architecture_type
        if [ -n "${current_exp_version}" ]; then
            results_exp_version="${ARCHITECTURE_TYPE}/${current_exp_version}"
        else
            # If no exp_version, use architecture_type as the base
            results_exp_version="${ARCHITECTURE_TYPE}"
        fi
        echo "Updated exp_version to: ${results_exp_version}"
        
        # Check if we should skip this dataset/configuration
        if [ "$SKIP_EXISTING" = true ]; then
            results_path="${CONFIG_RESULTS_DIR}/${SCAN}_ba"
            
            if [ -d "${results_path}" ]; then
                result_files=$(find "${results_path}" -type f \( -name "*.npz" -o -name "*.txt" -o -name "*.csv" -o -name "*.json" \) 2>/dev/null | head -1)
                
                if [ -n "${result_files}" ]; then
                    echo "  Skipping: Results already exist at ${results_path}"
                    echo "    Found: $(find "${results_path}" -type f | wc -l) files"
                    echo ""
                    ((jobs_skipped++))
                    ((total_jobs_skipped++))
                    continue
                else
                    echo "  -> Results directory exists but is empty, will process"
                    echo "    Directory: ${results_path}"
                fi
            else
                echo "  -> No results directory found, will process"
                echo "    Expected directory: ${results_path}"
            fi
        fi
        
        # Create job name (includes weight method and stage for better identification)
        # Add architecture suffix: d for esfm_deep, o for esfm_orig
        if [ "$ARCHITECTURE_TYPE" = "esfm_deep" ] || [ "$ARCHITECTURE_TYPE" = "esfm_outliers_deep" ]; then
            arch_suffix="d"
        else
            arch_suffix="o"
        fi
        
        if [ "$outlier_config" = "with_outliers" ]; then
            job_name="${SCAN}_${STAGE}_w_${arch_suffix}"
        else
            job_name="${SCAN}_${STAGE}_r_${arch_suffix}"
        fi
        
        # ============================================================================
        # CHANGE 6: Remove weight method from SCENE_LOG_DIR for Stage 1
        # ============================================================================
      
        # Construct paths for LSF output files
        if [ "$STAGE" = "1" ]; then
            SCENE_LOG_DIR="${LOG_BASE_DIR}/${STAGE}_stage/${LOSS_FUNCTION_DIR}/num_epochs_${num_epochs}_${eval_text}/scan_${safe_name}/${outlier_config}"
        else
            SCENE_LOG_DIR="${LOG_BASE_DIR}/${STAGE}_stage/${WEIGHT_METHOD_DIR}/num_epochs_${num_epochs}_${eval_text}/scan_${safe_name}/${outlier_config}"
        fi
        # Create scan-specific log directory with explicit feedback
        echo "  Creating LSF directory structure..."
        echo "    Full path: ${SCENE_LOG_DIR}"
        mkdir -p "${SCENE_LOG_DIR}"
        if [ -d "${SCENE_LOG_DIR}" ]; then
            echo "    ✓ Directory created/verified successfully"
        else
            echo "    ✗ Failed to create directory"
        fi
        
        lsf_output_file="${SCENE_LOG_DIR}/${job_name}.out"
        lsf_error_file="${SCENE_LOG_DIR}/${job_name}.err"
        
        echo "  Config: ${config_file}"
        echo "  Job name: ${job_name}"
        echo "  Weight method: ${WEIGHT_METHOD}"
        echo "  Stage: ${STAGE}"
        echo "  Python script: ${PYTHON_SCRIPT}"
        echo "  --- Output Files ---"
        echo "  Results will be saved to: ${CONFIG_RESULTS_DIR}/${SCAN}_ba/"
        echo "    (includes architecture: ${ARCHITECTURE_TYPE})"
        echo "  LSF output log: ${lsf_output_file}"
        echo "  LSF error log: ${lsf_error_file}"
        echo "  Submitting to LSF queue..."
        
        # Build the command with optional epochs parameter
        if [ -n "$NUM_EPOCHS" ]; then
            epochs_param="--external_params train.num_epochs:${NUM_EPOCHS}"
        else
            epochs_param=""
        fi

        
        # Add compute_loss_normalization parameter if enabled
            if [ "$COMPUTE_LOSS_NORMALIZATION" = true ]; then
                loss_norm_param="--compute_loss_normalization"
            else
                loss_norm_param=""
            fi

        # Check if running in interactive mode
        if [ "$INTERACTIVE_MODE" = true ]; then
            # Interactive mode: Run directly without bsub
            echo "  Running in INTERACTIVE mode..."
            echo "  Executing Python script directly"
            echo "  Script: ${PYTHON_SCRIPT}"
            echo "  Output will be saved to:"
            echo "    stdout: ${lsf_output_file}"
            echo "    stderr: ${lsf_error_file}"
            echo "  (and displayed in terminal)"
            echo ""
            
            cd "${REPO_ROOT}"
            


            # In the Python execution command (both interactive and batch modes):
            { uv run ${PYTHON_SCRIPT} \
                --conf "${config_file}" \
                --scan "${SCAN}" \
                --stage "${STAGE}" \
                --architecture_type "${ARCHITECTURE_TYPE}" \
                --phase OPTIMIZATION \
                --exp_version "${results_exp_version}" \
                --results_aggregation_file "${AGGREGATED_FILE}" \
                ${epochs_param} \
                ${loss_norm_param} \
                --wandb 0 \
                2> >(tee -a "${lsf_error_file}" >&2) \
                | tee -a "${lsf_output_file}"; }
            
        else
            # Batch mode: Submit job to LSF
            # Set GPU memory based on queue selection
            if [ "${QUEUE}" = "long-gpu" ]; then
                GPU_MEMORY="80G"
            elif [ "${QUEUE}" = "short-gpu" ]; then
                GPU_MEMORY="40G"
            else
                # Default to 40G for any other queue
                GPU_MEMORY="40G"
            fi
            
            # Build GPU request string with automatic memory specification
            gpu_request="num=1:j_exclusive=yes:gmem=${GPU_MEMORY}"
            
            # Submit job to LSF
            echo "  Submitting to LSF queue..."
            echo "  Queue: ${QUEUE}"
            echo "  GPU memory: ${GPU_MEMORY}"
            echo "  GPU request: ${gpu_request}"
            
     
            bsub -q "waic-${QUEUE}" \
                -J "${job_name}" \
                -oo "${lsf_output_file}" \
                -eo "${lsf_error_file}" \
                -gpu "${gpu_request}" \
                -R "rusage[mem=50000]" \
                "cd ${REPO_ROOT}; \
                uv run ${PYTHON_SCRIPT} \
                    --conf \"${config_file}\" \
                    --scan \"${SCAN}\" \
                    --stage \"${STAGE}\" \
                    --architecture_type \"${ARCHITECTURE_TYPE}\" \
                    --phase OPTIMIZATION \
                    --exp_version \"${results_exp_version}\" \
                    --results_aggregation_file '${AGGREGATED_FILE}' \
                    ${epochs_param} \
                    ${loss_norm_param} \
                    --wandb 0"


            JOB_STATUS=$?
        fi


        # Check job status
        if [ ${JOB_STATUS} -eq 0 ]; then
            if [ "$INTERACTIVE_MODE" = true ]; then
                echo "  [SUCCESS] Script completed successfully"
            else
                echo "  [SUCCESS] Job submitted successfully"
                echo "  Check job status with: bjobs | grep ${job_name}"
            fi
            ((jobs_submitted++))
            ((total_jobs_submitted++))
        else
            if [ "$INTERACTIVE_MODE" = true ]; then
                echo "  [FAILED] Script failed (exit code: ${JOB_STATUS})"
                echo "  Check output above for errors"
            else
                echo "  [FAILED] Failed to submit job (exit code: ${JOB_STATUS})"
                echo "  Check if LSF is available: which bsub"
                echo "  Check LSF status: bhosts"
            fi
        fi
        
        echo ""
        sleep 1  # Small delay between submissions
    done
    
    echo "Subtotal for ${outlier_config}: ${jobs_submitted} submitted, ${jobs_skipped} skipped"
done

    # Add to overall totals for this block combination
    overall_total_jobs_submitted=$((overall_total_jobs_submitted + total_jobs_submitted))
    overall_total_jobs_skipped=$((overall_total_jobs_skipped + total_jobs_skipped))
    
    echo ""
    echo "=============================================="
    echo "BLOCK COMBINATION SUMMARY"
    echo "=============================================="
    echo "Block Configuration: ${BLOCK_NUMBER} blocks, ${BLOCK_SIZE} layers per block"
    echo "Jobs submitted for this combination: ${total_jobs_submitted}"
    echo "Jobs skipped for this combination: ${total_jobs_skipped}"
    echo "=============================================="

# End of block combination loop
done

echo ""
echo ""
echo "############################################################"
echo "## FINAL OVERALL SUBMISSION SUMMARY"
echo "############################################################"
echo "Stage: ${STAGE}"
echo "Architecture Type: ${ARCHITECTURE_TYPE}"
if [ "$RUN_ALL_COMBINATIONS" = true ]; then
    echo "Block Configurations: ALL COMBINATIONS (${#BLOCK_COMBINATIONS[@]} total)"
    echo "  Combinations run:"
    for combo in "${BLOCK_COMBINATIONS[@]}"; do
        IFS=',' read -r bn bs <<< "$combo"
        echo "    - Block Number: $bn, Block Size: $bs (product: $((bn * bs)))"
    done
else
    echo "Block Configuration: ${BLOCK_NUMBER} blocks, ${BLOCK_SIZE} layers per block"
fi
echo "Mode: ${OUTLIER_MODE}"
echo "Epochs: ${epochs_text} (${num_epochs})"
echo "Weight Method: ${WEIGHT_METHOD}"
if [[ "$WEIGHT_METHOD" == "mad" ]] || [[ "$WEIGHT_METHOD" == "std" ]] || [[ "$WEIGHT_METHOD" == "huber" ]]; then
    echo "Alpha: ${ALPHA}"
fi
echo "Model: ${MODEL}"
if [ "$SKIP_EXISTING" = true ]; then
    echo "Skip existing: YES (skipped scans with existing results)"
else
    echo "Skip existing: NO (ran all scans)"
fi
if [ -n "$SPECIFIC_SCANS" ]; then
    echo "Scene selection: User-specified (${#datasets[@]} scans)"
elif [ -n "$MAX_SCANS" ]; then
    echo "Scene selection: Limited to ${MAX_SCANS} scans"
else
    echo "Scene selection: All available scans"
fi
echo "Total scans processed: ${#datasets[@]} per block combination"
echo "Total block combinations: ${#BLOCK_COMBINATIONS[@]}"
echo "Overall total jobs submitted: ${overall_total_jobs_submitted}"
echo "Overall total jobs skipped: ${overall_total_jobs_skipped}"
echo ""
# ADDED FOR AGGREGATION - Display aggregated file info with epochs and eval intervals
echo "Aggregated results file: ${AGGREGATED_FILE}"
echo "Filename format: Aggregated_across_jobs_${num_epochs}epochs_eval${eval_intervals}_[architecture]_[loss/weight_method]_stage[1/2]_[with/without]_reproj${REPROJ_LOSS_WEIGHT_SAFE}_class${CLASSIFICATION_LOSS_WEIGHT_SAFE}_{}${TIMESTAMP}.xlsx"
echo "View aggregated results with:"
echo "  python3 -c \"import pandas as pd; print(pd.read_excel('${AGGREGATED_FILE}'))\""
echo ""
echo "Monitor all jobs with:"
echo "  bjobs | grep ${epochs_text}_${WEIGHT_METHOD}"
echo "  bjobs | grep ${STAGE}"
echo "############################################################"# #!/bin/bash
