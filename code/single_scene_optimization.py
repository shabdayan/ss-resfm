"""
Single scene optimization script
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Default kept at '1' (historical behavior) but now overridable: the single-scene
# benchmark sets CUDA_LAUNCH_BLOCKING=0 so the U-ESFM arm is timed with the same
# async CUDA launches as the ESFM baseline (approved 2026-07-12).
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
os.environ['PYTHONHASHSEED'] = '0'
# Suppress CUDA graphs warning for incompatible operations
os.environ['TORCH_LOGS'] = '-dynamo'

import cv2  # DO NOT REMOVE
import argparse
import numpy as np
import torch
import pandas as pd 

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from datasets import SceneData, ScenesDataSet
import train
from utils import general_utils, path_utils
from utils.Phases import Phases

from lightning.fabric import Fabric
import sys

from utils.general_utils import worker_init_fn 
import loss_functions 
from datetime import datetime
import fcntl
import time
import torch.nn as nn
import torch.nn.init as init


def initialize_fabric(seed=None, use_progressive=False):
    from lightning.fabric.strategies import DDPStrategy
    
    # Only enable when needed!
    strategy = DDPStrategy(
        find_unused_parameters=use_progressive, 
        static_graph=not use_progressive
    )
    
    fabric = Fabric(
        accelerator="cuda", 
        devices="auto", 
        strategy=strategy,
    )
    
    if seed is not None:
        fabric.seed_everything(seed)
    
    fabric.launch()
    return fabric




def init_weights_kaiming(module):
    """
    Kaiming/He initialization for deep networks with ReLU.
    Best for networks with residual connections and layer normalization.
    """
    if isinstance(module, nn.Linear):
        # Kaiming initialization for linear layers
        init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    
    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for layer norm
        init.ones_(module.weight)
        init.zeros_(module.bias)


def apply_kaiming_if_deep(model, conf):
    """Kaiming init is one of the DEEP-architecture measures (report §2.1);
    original ESFM/RESfM (1 block x 3 layers) trains from PyTorch's default
    init. Gate on total depth so shallow arms stay faithful to the baselines.
    (FINE_TUNE loads a checkpoint right after, so this only affects fresh
    training runs.)"""
    total_layers = (conf.get_int('model.num_blocks', default=1)
                    * conf.get_int('model.block_size', default=3))
    if total_layers > 3:
        model.apply(init_weights_kaiming)
        print(f"✓ Kaiming initialization applied (deep model: {total_layers} layers)")
    else:
        print(f"– Default PyTorch initialization (shallow model: {total_layers} layers; Kaiming is deep-only)")


            
def train_single_model(conf, device, phase, stage=1, architecture_type="esfm_outliers", results_aggregation_file_name=None):
    """
    Train a single model with configurable stage and reprojection error support.
    
    Args:
        conf: configuration object
        device: torch device
        phase: training phase (OPTIMIZATION, FINE_TUNE, etc.)
        stage: 1 or 2 stage of training
        with_proj_err: whether to include reprojection error weights (for 2nd stage)
        architecture_type: architecture type for organizing results (default: esfm_orig)
        results_aggregation_file_name: name of the aggregated results file to append to
    """

    seed = conf.get_int('random_seed', default=None)
    use_progressive = conf.get_bool('model.use_progressive', default=False)
    fabric = initialize_fabric(seed=seed, use_progressive=use_progressive)

    # Paper ground rule: the 27 multi-scene training scenes must never be evaluated.
    if phase is Phases.FINE_TUNE:
        from utils.experiment_guard import assert_not_training_scene
        assert_not_training_scene(conf.get_string('dataset.scan'), context='FINE_TUNE evaluation entry')

    # Restored from upstream RESfM: before fine-tuning with output_mode 3, run a
    # TEST evaluation with the pretrained model so its predicted outliers get saved
    # (epoch_evaluation -> save_outliers); the FINE_TUNE data load below prunes the
    # tracks with that file (Euclidean.get_raw_data).
    if phase is Phases.FINE_TUNE and conf.get_int('train.output_mode', default=3) == 3:
        print("Run test before fine-tuning")
        test_model_class = general_utils.get_class("models." + conf.get_string("model.type"))
        test_model = test_model_class(conf, Phases.TEST).to(device)
        test_scene_data = SceneData.create_scene_data(conf, Phases.TEST, stage=stage)
        test_dataset = ScenesDataSet.ScenesDataSet([test_scene_data], return_all=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=ScenesDataSet.collate_fn)
        _, _, test_errors = train.test(conf, test_model, Phases.TEST, train_data=None, validation_data=None, test_data=test_loader, fabric=fabric, run_ba=False)
        print(test_errors.to_string(), flush=True)
        test_errors = test_errors.drop(['Mean'])
        general_utils.write_results(conf, test_errors, file_name="myTest_fineTuning", phase=phase, append=True)

    scene_data = SceneData.create_scene_data(conf, phase, stage=stage)
    
    # Debug output for stage verification
    if stage == 2:
        print(f'Stage: {stage} - Loading scene data WITH reprojection errors')
    else:
        print(f'Stage: {stage} - Loading scene data WITHOUT reprojection errors')
    
    # Create model
    model_type = conf.get_string("model.type")
    print(f"Creating model with type from config: {model_type}")
    print("models." + model_type)
    
    # Handle different model signatures
    model_class = general_utils.get_class("models." + model_type)
    # Check for models that don't accept phase parameter (with or without module prefix)
    if model_type in ["SetOfSet.SetOfSetNet", "SetOfSet.DeepSetOfSetNet"]:
        # These models don't accept phase parameter
        model = model_class(conf).to(device)
    elif model_type in ["SetOfSet.SetOfSetOutliersNet", "SetOfSet.DeepSetOfSetOutliersNet"]:
        # SetOfSetOutliersNet and DeepSetOfSetOutliersNet accept phase parameter
        model = model_class(conf, phase).to(device)
    else:
        print(f'Unknown model type: {model_type}')
        print(f'Expected one of: SetOfSet.SetOfSetNet, SetOfSet.DeepSetOfSetNet, SetOfSet.SetOfSetOutliersNet, SetOfSet.DeepSetOfSetOutliersNet')  

    # Weight initialization: Kaiming for deep architectures only (report §2.1)
    apply_kaiming_if_deep(model, conf)
    # ============================================================================

    
    # if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
    
    print(f'Number of parameters: {sum([x.numel() for x in model.parameters()])}')
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Load checkpoint if fine-tuning
    if phase is Phases.FINE_TUNE:
        path = path_utils.path_to_model(conf, Phases.TRAINING, epoch=None, best=True)
        checkpoint = torch.load(path)
        # Leakage guard: the fine-tune base must be a clean TRAINING checkpoint —
        # never weights from a prior run on a test scene (e.g. a TTT snapshot).
        from utils.experiment_guard import assert_checkpoint_provenance_clean
        assert_checkpoint_provenance_clean(checkpoint, path, context='FINE_TUNE base checkpoint')

        # Handle _orig_mod prefix mismatch between checkpoint and current model
        state_dict = checkpoint['model_state_dict']
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Check if there's a prefix mismatch
        if checkpoint_keys != model_keys:
            new_state_dict = {}
            prefix_to_remove = "_orig_mod."
            prefix_to_add = "_orig_mod."
            
            # Case 1: Checkpoint has _orig_mod prefix, but model doesn't
            if any(k.startswith(prefix_to_remove) for k in checkpoint_keys) and \
               not any(k.startswith(prefix_to_remove) for k in model_keys):
                print(f"Removing '{prefix_to_remove}' prefix from checkpoint keys...")
                for key, value in state_dict.items():
                    new_key = key.replace(prefix_to_remove, "", 1) if key.startswith(prefix_to_remove) else key
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
            
            # Case 2: Model has _orig_mod prefix, but checkpoint doesn't
            elif not any(k.startswith(prefix_to_add) for k in checkpoint_keys) and \
                 any(k.startswith(prefix_to_add) for k in model_keys):
                print(f"Adding '{prefix_to_add}' prefix to checkpoint keys...")
                for key, value in state_dict.items():
                    new_key = prefix_to_add + key if not key.startswith(prefix_to_add) else key
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print("Fine-tuning a scene - The model is resuming from checkpoint: ", path)

    # Create data with appropriate reprojection error setting
    scene_data = SceneData.create_scene_data(conf, phase, stage=stage)
    scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)

    scene_loader = torch.utils.data.DataLoader(
        scene_dataset, 
        collate_fn=ScenesDataSet.collate_fn,
        num_workers=0,              # Use 0 for max reproducibility
        worker_init_fn=worker_init_fn,  #  Important for reproducibility
        shuffle=False,              # Important for reproducibility
        drop_last=False,
        pin_memory=True  # Can keep for speed
    )

    # check if dataloader has reprojection errors 
    if stage == 2 and hasattr(scene_data, 'reprojection_errors'):
        print(f"\n{'='*60}")
        print(f"✓ Stored reprojection weights on DataLoader")
        print(f"  Shape: {scene_loader.reprojection_weights.shape}")
        print(f"  Device: {scene_loader.reprojection_weights.device}")
        print(f"{'='*60}\n")
    
    # Regular training mode
    print(f"Running {stage} stage optimization")
    
    # Train the model
    train_stat, train_errors, _, _ = train.train(conf, scene_loader, model, phase, fabric=fabric)
    train_errors.drop("Mean", inplace=True)
    print('train_errors.to_string()')
    print(train_errors.to_string(), flush=True)

 
    scene_names = train_errors.index
    # print('scene_names', scene_names)
    train_stat["Scene"] = scene_names
    train_stat.set_index("Scene", inplace=True)
   
    
    train_res = train_errors.join(train_stat)
    # Add stage and architecture suffix to results filename
    results_filename = f"Results_{phase.name}_stage_{stage}_{architecture_type}"
    general_utils.write_results(conf, train_res, file_name=results_filename, append=False)
    print('train_res', train_res.columns)
    
    train_res = train_res.rename(columns={
    'ts_ba_final_mean': 'Trans',
    'Rs_ba_final_mean': 'Rot',
    '#registered_cams_final': 'Nr',
    'Convergence time': 'Convergence Time', 
    'best_epoch': 'Best Epoch'
    })
    
    # BA columns (Trans/Rot/Nr) only exist when ba.run_ba = true; keep whatever is
    # available so no-BA runs (single-scene benchmark) don't crash here.
    agg_cols = [c for c in ['Trans', 'Rot', 'Nr', 'Convergence Time', 'Best Epoch']
                if c in train_res.columns]
    train_res_aggregare_across_jobs = train_res[agg_cols]
    if 'Convergence Time' in train_res_aggregare_across_jobs.columns:
        train_res_aggregare_across_jobs['Convergence Time'] =  train_res_aggregare_across_jobs['Convergence Time'].astype(int)


    # Add configuration columns to the aggregated results
    train_res_aggregare_across_jobs['model_type'] = conf.get_string('model.type')
    train_res_aggregare_across_jobs['block_size'] = conf.get_string('model.block_size')
    train_res_aggregare_across_jobs['block_number'] = conf.get_string('model.num_blocks')
    train_res_aggregare_across_jobs['num_epochs'] = conf.get_string('train.num_epochs')
    train_res_aggregare_across_jobs['eval_intervals'] = conf.get_string('train.eval_intervals')
    # Add new training parameters to results
    train_res_aggregare_across_jobs['scheduler_milestone'] = conf.get_string('train.scheduler_milestone', default='[60000,100000,150000]')
    train_res_aggregare_across_jobs['early_stopping_patience'] = conf.get_string('train.early_stopping_patience', default='20000')
    train_res_aggregare_across_jobs['stage'] = conf.get_string('general.stage')
    if stage == 2:
        train_res_aggregare_across_jobs['weight_method'] = conf.get_string('postprocessing.weight_method')

 
    # Optional: Add more configuration parameters if needed
   
    # Print to verify the columns
    print('Aggregated results columns:', train_res_aggregare_across_jobs.columns.tolist())
    print('Aggregated results shape:', train_res_aggregare_across_jobs.shape)
    
    # Check if file exists and has data

    if os.path.exists(results_aggregation_file_name):
        try:
            # Read existing data
            existing_df = pd.read_excel(results_aggregation_file_name, index_col='Scene')
            print(existing_df)
            train_res_aggregare_across_jobs.index = train_res_aggregare_across_jobs.index.astype(int)
            print(train_res_aggregare_across_jobs)
            combined_df = pd.concat([existing_df, train_res_aggregare_across_jobs], ignore_index=False).sort_index()
            combined_df.to_excel(results_aggregation_file_name, index=True)
   
        except Exception as e:
            print(f"Error appending to aggregated file: {e}")
    else:
        # File doesn't exist, create it
        train_res_aggregare_across_jobs.to_excel(results_aggregation_file_name, index=True)
       
    # Additional functionality for first stage: generate and save reprojection errors
    # train.extract_reproj_errors (default true = original behavior) lets the
    # single-scene benchmark skip this extra test+BA pass, which is only needed as
    # input for a stage-2 run and would otherwise pollute wall-clock timing.
    if stage == 1 and conf.get_bool('train.extract_reproj_errors', default=True):
        train.test(conf, model, Phases.OPTIMIZATION,
                                train_data=None, validation_data=None,
                                test_data=scene_loader, fabric=fabric, run_ba=True)
        print("reprojection error extraction completed for first stage")
    
    
    return model, train_res



def save_losses_to_csv(scene, esfm_sum_loss, class_sum_loss, num_samples, results_path):
    """
    Save loss statistics to CSV - one file per scene for parallel safety.
    """
    
    # Define output directory
    output_dir = os.path.join(
        os.path.dirname(results_path),
        "loss_normalization_per_scene"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scene index for sorting (convert '0007' to 7)
    try:
        scene_index = int(scene)
    except:
        scene_index = scene
    
    # Calculate metrics
    esfm_mean = esfm_sum_loss / num_samples
    class_mean = class_sum_loss / num_samples
    scale_factor = esfm_mean / class_mean if class_sum_loss != 0 else 0
    
    # Create row for this scene
    scene_data = pd.DataFrame({
        'scene': [scene_index],
        'num_samples': [num_samples],
        'esfm_sum_loss': [esfm_sum_loss],
        'esfm_mean_loss': [esfm_mean],
        'class_sum_loss': [class_sum_loss],
        'class_mean_loss': [class_mean],
        'scale_factor': [scale_factor]
    })
    
    # Save to individual scene file
    scene_file = os.path.join(output_dir, f"scene_{scene_index:04d}.csv")
    scene_data.to_csv(scene_file, index=False)
    print(f'Scene {scene} results saved to: {scene_file}')
    
    # Display current scene results
    print(f"\nScene {scene} results:")
    print(f"  ESFM sum: {esfm_sum_loss:.4f}, mean: {esfm_mean:.6f}")
    print(f"  Class sum: {class_sum_loss:.4f}, mean: {class_mean:.6f}")
    print(f"  Scale factor: {scale_factor:.6f}")


def merge_loss_normalization_files(results_path):
    """
    Merge all individual scene files into a single CSV.
    Call this AFTER all parallel jobs complete.
    """
    import glob
    
    output_dir = os.path.join(
        os.path.dirname(results_path),
        "loss_normalization_per_scene"
    )
    
    # Find all scene files
    scene_files = sorted(glob.glob(os.path.join(output_dir, "scene_*.csv")))
    
    if not scene_files:
        print("No scene files found to merge!")
        return
    
    # Read and concatenate all files
    all_scenes = []
    for scene_file in scene_files:
        df = pd.read_csv(scene_file)
        all_scenes.append(df)
    
    # Combine and sort
    combined_df = pd.concat(all_scenes, ignore_index=True)
    combined_df = combined_df.sort_values('scene').set_index('scene')
    
    # Save merged file
    merged_file = os.path.join(
        os.path.dirname(results_path),
        "loss_normalization_stage.csv"
    )
    combined_df.to_csv(merged_file, index=True)
    
    print(f"\n{'='*60}")
    print(f"Merged {len(scene_files)} scene files into: {merged_file}")
    print(f"{'='*60}")
    
    # Show cumulative statistics
    total_samples = combined_df['num_samples'].sum()
    overall_esfm_mean = combined_df['esfm_sum_loss'].sum() / total_samples
    overall_class_mean = combined_df['class_sum_loss'].sum() / total_samples
    
    print(f"\nCumulative ({len(combined_df)} scenes):")
    print(f"  Total samples: {total_samples}")
    print(f"  Overall ESFM mean: {overall_esfm_mean:.6f}")
    print(f"  Overall Class mean: {overall_class_mean:.6f}")
    print(f"  Scale factor: {overall_esfm_mean/overall_class_mean:.2f}")
    
    print("\nPer-scene results:")
    print(combined_df[['esfm_mean_loss', 'class_mean_loss', 'scale_factor']])
    
    return merged_file

        
def loss_normalization(conf, device, phase, num_samples=100):
    stage = 1
    scene = conf.get_string('dataset.scan')
    results_path = conf.get_string('results_path')
    # print('results_path:', results_path)
    # exit()
    all_esfm_losses = []
    all_class_losses = []
    
    # Create loss functions
    compute_esfm_loss = loss_functions.ESFMLoss_weighted(conf)
    compute_class_loss = loss_functions.AdaptiveConfidenceWeightedOutliersLoss(conf)
    
    # Create model
    model_type = conf.get_string("model.type")
    print(f"Creating model with type from config: {model_type}")
    
    model_class = general_utils.get_class("models." + model_type)
    if model_type in ["SetOfSet.SetOfSetNet", "SetOfSet.DeepSetOfSetNet"]:
        model = model_class(conf).to(device)
    elif model_type in ["SetOfSet.SetOfSetOutliersNet", "SetOfSet.DeepSetOfSetOutliersNet"]:
        model = model_class(conf, phase).to(device)
    else:
        print(f'Unknown model type: {model_type}')
        return None, None
    
    # Apply initialization (Kaiming for deep architectures only)
    apply_kaiming_if_deep(model, conf)
    model.eval()
    
    # Load the full scene data once
    scene_data = SceneData.create_scene_data(conf, phase, stage=1)
    
    # Sample 100 times from this scene
    # with torch.no_grad():
    for i in range(num_samples):  # ← THIS LOOP IS CRITICAL!
        # Each iteration samples a different subset of cameras
        num_cams_to_sample = min(30, len(scene_data.y))  # Sample 30 cameras (or less if scene is smaller)
        sampled_data = SceneData.sample_data(scene_data, num_cams_to_sample, adjacent=False)
        sampled_data = sampled_data.to(device)
        
        # Forward pass
        pred_cam, pred_outliers = model(sampled_data)
        
        # Compute losses
        esfm_loss = compute_esfm_loss(pred_cam, pred_outliers, sampled_data)
        class_loss = compute_class_loss(pred_cam, pred_outliers, sampled_data)
        
        # Store each sample's loss
        all_esfm_losses.append(esfm_loss.detach().item() if torch.is_tensor(esfm_loss) else esfm_loss)
        all_class_losses.append(class_loss.detach().item() if torch.is_tensor(class_loss) else class_loss)

    # Return the SUM for this scene (to be aggregated later across scenes)
    esfm_sum = sum(all_esfm_losses)
    class_sum = sum(all_class_losses)
    
    print(f"Scene {conf.get_string('dataset.scan')}:")
    print(f"  Processed {len(all_esfm_losses)} samples")
    print(f"  ESFM sum: {esfm_sum:.4f}, mean: {esfm_sum/num_samples:.6f}")
    print(f"  Class sum: {class_sum:.4f}, mean: {class_sum/num_samples:.6f}")

    save_losses_to_csv(
        scene,
        esfm_sum,
        class_sum,
        num_samples,
        results_path
    )
    
    return esfm_sum, class_sum


def main():
    """Main function with argument parsing for stage selection."""
    
    # Store the original sys.argv
    original_argv = sys.argv.copy()
    
    # Create parser for our custom arguments
    parser = argparse.ArgumentParser(description='Single Scene Optimization', add_help=False)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='Training stage: 1 (first) or 2 (second)')
    parser.add_argument('--architecture_type', type=str, default='esfm_outliers',
                       help='Architecture type')
    parser.add_argument('--results_aggregation_file', type=str, default=None,
                       help='Aggregated results file')
    parser.add_argument('--compute_loss_normalization', action='store_true',
                        help='Compute loss normalization')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples')
    
    # Parse known args
    args, unknown = parser.parse_known_args()
    
    # MORE ROBUST CLEANING: Remove our custom arguments from sys.argv
    clean_argv = []
    skip_next = False
    
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
            
        # Check if this is one of our custom arguments
        if arg in ['--stage', '--architecture_type', '--results_aggregation_file', '--num_samples']:
            skip_next = True  # Skip the next item (the value)
            continue
        elif arg in ['--compute_loss_normalization']:
            continue  # Just skip this flag (no value to skip)
        elif arg.startswith('--results_aggregation_file='):
            # Handle case where value is attached with =
            continue
        else:
            clean_argv.append(arg)
    
    # Debug: Print what we're keeping and removing
    print(f"Original argv ({len(original_argv)} items): {original_argv[:5]}...")
    print(f"Cleaned argv ({len(clean_argv)} items): {clean_argv}")
    
    # Temporarily replace sys.argv
    sys.argv = clean_argv
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    # Initialize experiment
    print('Initializing experiment...')
    conf, device, phase = general_utils.init_exp(Phases.OPTIMIZATION.name)
   
    if args.compute_loss_normalization:
        print("Running loss normalization analysis...")
    
        esfm_sum, class_sum = loss_normalization(conf, device, phase, num_samples=args.num_samples)
        if esfm_sum is not None:
            print(f"Loss normalization completed: ESFM sum={esfm_sum}, Class sum={class_sum}")
        else:
            print("Loss normalization failed - no trained model found")
    else:
    
        # Run normal training
        model, results = train_single_model(conf, device, phase, 
                                           stage=args.stage, 
                                           architecture_type=args.architecture_type,
                                           results_aggregation_file_name=args.results_aggregation_file)
       
        print(f"Optimization completed for stagse: {args.stage}")
        print(f"Architecture type: {args.architecture_type}")
        print(f"Results saved with suffix: stage_{args.stage}")
        
        if args.results_aggregation_file:
            print(f"Aggregated results appended to: {args.results_aggregation_file}")

if __name__ == "__main__":
    main()





