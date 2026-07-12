import sys
# import time
import torch
import math
# import time
import utils.general_utils as general_utils
import loss_functions
import evaluation
import copy

from utils.general_utils import save_metrics_excel
from utils import path_utils, dataset_utils, plot_utils
from utils.metrics_utils import OutliersMetrics, CalcMeanBatchMetrics, nanMetrics
from time import time
import pandas as pd
from utils.Phases import Phases
from utils.path_utils import path_to_exp
# from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from torch.distributed import all_gather_object
from torch.nn import functional as F

from lightning.fabric import Fabric
import numpy as np
# from reprojection_error_matrix import modified_epoch_evaluation

os.environ["WANDB__SERVICE_WAIT"] = "300"
OPTIMIZER_TYPES = {
    'Adam': torch.optim.Adam,
}

OUTPUT_MODES_TYPES = {
    1: "ESFM",
    2: "Outliers Removal",
    3: "RESFM",
    4: "3mlps_ablation",

}

def checkpoint_provenance(conf, phase, epoch):
    # Traceability (SPEC_cvpr_experiments R2.4): never let a provenance failure
    # break a checkpoint save.
    try:
        from utils.experiment_guard import provenance
        return provenance(conf, phase=phase, extra={'epoch': epoch})
    except Exception as e:
        print(f"Warning: could not record checkpoint provenance: {e}")
        return None


def conf_snapshot(conf):
    # Checkpoints must carry the conf they were trained with (needed by the
    # test-time fine-tune stage). conf may hold non-serializable values (e.g. a
    # Phases enum under 'phase'), so never let the snapshot break a save.
    try:
        from pyhocon import HOCONConverter
        return HOCONConverter.convert(conf, 'hocon')
    except Exception as e:
        print(f"Warning: could not snapshot conf into checkpoint: {e}")
        return None

def epoch_evaluation(data_loader, model, conf, epoch, phase, save_predictions=False, bundle_adjustment=True, plot=False):
    metrics_list = []
    errors = None
    model.eval()
    with torch.no_grad():
        i = 0 
        for batch_data in data_loader:
            i +=1
            n = 0 
            # A batch of scenes
            for curr_data in batch_data:
                n +=1
                # Get predictions
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()  # R16: per-scene peak memory
                begin_time = time()
                pred_cam, pred_outliers = model(curr_data)
                inference_seconds = time() - begin_time

                # Eval results
                metrics = {}
                metrics['Scene'] = curr_data.scan_name

                if pred_cam is not None:
                    # print('line 66 in train.py pred cam is not none')
                    outputs = evaluation.prepare_predictions(curr_data, pred_cam, conf, bundle_adjustment, phase, curr_epoch=epoch)
                    
                    errors, errors_per_cam, outputs = evaluation.compute_errors(outputs, conf, bundle_adjustment, compute_rep_errs=True)
                    metrics.update(errors)

                else:
                    outputs = {}

                # R16 uniform timing/resources — numeric-only columns so
                # organize_errors' Mean row stays valid; they flow into every
                # results file (Results_*.xlsx, ttt_snapshots.csv) unchanged.
                metrics['inference_seconds'] = round(inference_seconds, 4)
                metrics['ba_seconds'] = outputs.get('ba_seconds', float('nan'))
                metrics['peak_gpu_mem_gb'] = round(
                    torch.cuda.max_memory_allocated() / 2 ** 30, 3) if torch.cuda.is_available() else float('nan')
                metrics['param_count'] = sum(p.numel() for p in model.parameters())

                # Restored from upstream RESfM: package predicted outliers so the
                # test evaluation can save them — the FINE_TUNE stage prunes tracks
                # with this file (Euclidean.get_raw_data, output_mode == 3).
                if pred_outliers is not None:
                    outliersOutputs = evaluation.prepare_outliers_predictions(curr_data, pred_outliers, conf)

                metrics_list.append(metrics)

                if save_predictions:
                    # Restored from upstream ESFM/RESfM: persist the FINAL predicted
                    # cameras (epoch is None = best model) so the shared single-scene
                    # evaluator (evaluate_single_scene.py) can recompute harmonized
                    # metrics from raw outputs. Intermediate epochs are not saved.
                    if pred_cam is not None and epoch is None:
                        dataset_utils.save_cameras(outputs, conf, curr_epoch=epoch, phase=phase)
                    if conf.get_string('test.outlier_source', default='') == 'mad' \
                            and pred_cam is not None and phase is Phases.TEST:
                        # U-ESFM: outliers come from per-scene reprojection-error
                        # statistics (MAD), not a learned classifier. Save a binary
                        # mask in the same file/format the FINE_TUNE pruning step
                        # consumes (Euclidean.get_raw_data), so the rest of the
                        # RESfM evaluation protocol runs unchanged.
                        from datasets.Euclidean import detect_outliers_statistical
                        from utils import geo_utils
                        rep = geo_utils.reprojection_error_with_points(
                            outputs['Ps'], outputs['pts3D_pred'].T, outputs['xs'])
                        valid = ~np.isnan(rep)
                        flags = detect_outliers_statistical(
                            torch.from_numpy(rep[valid]).float(), weight_method='mad',
                            alpha=conf.get_float('test.mad_alpha', default=2.0))
                        mask = np.zeros(rep.shape, dtype=np.float32)
                        mask[valid] = flags.numpy()
                        dataset_utils.save_outliers(
                            {'scan_name': curr_data.scan_name, 'outliers_pred': mask},
                            conf, curr_epoch=epoch, phase=phase)
                    elif pred_outliers is not None:
                        dataset_utils.save_outliers(outliersOutputs, conf, curr_epoch=epoch, phase=phase)
                    if errors is not None:
                        errors.update(errors_per_cam)
                        if phase != Phases.TEST and plot:
                            plot_utils.plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan=curr_data.scan_name, epoch=epoch, bundle_adjustment=bundle_adjustment)

    df_metrics = evaluation.organize_errors(metrics_list)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return df_metrics, metrics_list


def epoch_train(conf, train_data, model, loss_func, optimizer, scheduler, epoch, phase, fabric):
    model.train()
    train_losses = []
    train_metrics = []
    
    # DEBUG: Check first batch
    stage = conf.get_int('general.stage')
    if epoch == 1 and stage == 2:
        print("\n=== DEBUG: Checking if weights are in data ===")
        for batch in train_data:
            for data in batch:
                print(f"Has reprojection_weights: {hasattr(data, 'reprojection_weights')}")
                if hasattr(data, 'reprojection_weights'):
                    print(f"  Shape: {data.reprojection_weights.shape}")
                break
            break
        print("="*50 + "\n")
    
    # ... rest of the function


    # k = 0 
    for train_batch in train_data:
        # k += 1
        # print('num train_batch', k)
        
        batch_loss = torch.tensor([0.0], device=train_data.device)
        optimizer.zero_grad()

        for i, curr_data in enumerate(train_batch):
            # print('curr_data', type(curr_data))
            # print('num sample in train batch', i)
            if not dataset_utils.is_valid_sample(curr_data, phase=phase, min_pts_per_cam=0):
                print(f"{fabric.global_rank}: {epoch} {curr_data.scan_name} has a camera with not enough points")
                continue

            # # === ATTACH WEIGHTS FROM DATALOADER IF AVAILABLE ===
            # if hasattr(train_data, 'reprojection_weights'):
            #     curr_data.reprojection_weights = train_data.reprojection_weights

            pred_cam, pred_outliers = model(curr_data)

            if pred_outliers is not None and pred_cam is None:
                pred_weights_M = torch.zeros(curr_data.x.shape[0], curr_data.x.shape[1], device=curr_data.x.device)
                pred_weights_M[curr_data.x.indices[0, :], curr_data.x.indices[1, :]] = pred_outliers.squeeze(dim=-1)
                loss = loss_func(pred_outliers, pred_weights_M, curr_data, epoch)



            elif pred_cam is not None and pred_outliers is None:
                # For both stage 1 (no weights) and stage 2 (weights in data.proj_err_weight)
                # print('using loss function for pred_cam is not None and pred_outliers is Non in train.py line 142')
              
                # if loss_func.__class__ == loss_func.ESFMLoss_weighted_by_rep_err or loss_func.__class__ == loss_func.ESFMLoss:
                #     loss = loss_func(pred_cam, curr_data, epoch)
                # elif loss_func.__class__ == loss_func.AdaptiveConfidenceWeightedOutliersLoss:
                #     loss = loss_func(pred_cam, pred_outliers, curr_data, epoch)
                # else:
                #     raise ValueError(f"Unknown loss function: {loss_func.__class__}")
                loss = loss_func(pred_cam, curr_data, epoch)
            else:
                # in resfm 
                # pred_weights_M = torch.zeros(curr_data.x.shape[0], curr_data.x.shape[1], device=curr_data.x.device)
                # pred_weights_M[curr_data.x.indices[0, :], curr_data.x.indices[1, :]] = pred_outliers.squeeze(dim=-1)
                # loss = loss_func(pred_cam, pred_outliers, pred_weights_M, curr_data, epoch)
                
                # combined loss 
                loss = loss_func(pred_cam, pred_outliers, curr_data, epoch)
                if hasattr(loss_func, 'finalize_and_save'):
                    loss_func.finalize_and_save()
                


            metrics = OutliersMetrics(pred_outliers, curr_data) if pred_outliers is not None else {}

            batch_loss += loss
            train_losses.append(loss.item())
            train_metrics.append(metrics)

        if batch_loss.item() > 0:
            fabric.backward(batch_loss)
            optimizer.step()

        else:
            print("Negative or zero batch loss:", batch_loss.item())
            metrics = nanMetrics()
            train_metrics = [metrics]
            train_losses = [torch.tensor([0.0], device=train_data.device).item()]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



    scheduler.step()

    mean_loss = torch.tensor(train_losses).mean()
    train_metrics = CalcMeanBatchMetrics(train_metrics, None)

    return mean_loss, train_losses, [train_metrics]


def train(conf, train_data, model, phase, validation_data=None, test_data=None, fabric=None):

    num_epochs = conf.get_int('train.num_epochs')
    eval_intervals = conf.get_int('train.eval_intervals', default=500)
    validation_metric = conf.get_list('train.validation_metric') if phase is not Phases.FINE_TUNE else conf.get_list('train.validation_metric_fine_tuning')
    early_stopping_patience = conf.get_int('train.early_stopping_patience', default=0)  # 0 means no early stopping

    # === Loss Function ===
    if phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION]:
        conf["resuming_epoch"] = -1
        try:
            loss_name = conf.get_string('loss.func_tuning')
        except KeyError:
            print("[Warning] Missing 'loss.func_tuning' in config — falling back to 'loss.func'")
            loss_name = conf.get_string('loss.func')
    else:
        loss_name = conf.get_string('loss.func')
    # Get loss function from the loss_functions module
    loss_func = getattr(loss_functions, loss_name)(conf)
    # print('loss_func', loss_func)

    # === Optimizer & Scheduler ===
    lr = conf.get_float('train.lr')
    scheduler_milestone = conf.get_list('train.scheduler_milestone')
    gamma = conf.get_float('train.gamma', default=0.1)
    optim_type = conf.get_string('train.optim_type', default='Adam')
    optimizer = OPTIMIZER_TYPES[optim_type](model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestone, gamma=gamma)

    # print(f'Training with loss of type {loss_func}')
    # print(f'Training with optimizer of type {type(optimizer)}')


    # === Resume Training (Optional) ===
    resumed_best_metric = None
    resumed_best_epoch = None
    try:
        if conf['resume']:
            if phase is Phases.OPTIMIZATION:
                path, epoch = path_utils.path_to_model_resume_optimizing(conf, scan=conf.dataset.scan)
            elif phase is Phases.TRAINING:
                path, epoch = path_utils.path_to_model_resume_learning(conf)
            checkpoint = torch.load(path)
            if phase is Phases.TRAINING:
                # Leakage guard: multi-scene training must never resume from
                # weights produced by a run on a test scene (e.g. a TTT snapshot).
                from utils.experiment_guard import assert_checkpoint_provenance_clean
                assert_checkpoint_provenance_clean(checkpoint, path, context='multi-scene training resume')
            conf["resuming_epoch"] = checkpoint['epoch']
            if checkpoint['epoch'] >= num_epochs:
                sys.exit()
            
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Newer checkpoints also carry scheduler + best-metric state so a
            # preempted run resumes with the right LR schedule and does not let a
            # worse post-resume model shadow the earlier best.
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            resumed_best_metric = checkpoint.get('best_validation_metric', None)
            resumed_best_epoch = checkpoint.get('best_epoch', None)
            print("The model is resuming from checkpoint:", path)
    except Exception as e:
        from utils.experiment_guard import LeakageError
        if isinstance(e, LeakageError):
            raise  # leakage must fail loudly, never degrade to a fresh start
        print(f"Resume requested but starting fresh ({e})")


    # === Setup Fabric and Dataloaders ===
    model, optimizer = fabric.setup(model, optimizer)
    train_data = fabric.setup_dataloaders(train_data, use_distributed_sampler=True)
    if test_data is not None:
        validation_data = fabric.setup_dataloaders(validation_data)
        test_data = fabric.setup_dataloaders(test_data)

    # === WandB Logging ===
    if fabric.global_rank == 0:
        mode = 'online' if conf['wandb'] else 'disabled'
        name = conf.get_string('exp_name')
        if phase is Phases.OPTIMIZATION:
            name += '_' + conf['dataset.scan']
        logger = wandb.init(
            mode=mode,
            name=name,
            project="RESfM",
            dir=path_utils.path_to_wandb_logs(conf, phase),
        )
        print(f'Starting a logger at {path_utils.path_to_wandb_logs(conf, phase, scan=None)}')

    # === Tracking Best Metrics ===
    # FINE_TUNE/OPTIMIZATION always minimize; TRAINING historically maximized (RESfM's
    # "Accuracy"), so geometric validation metrics (e.g. our_repro) need
    # train.validation_metric_higher_is_better = False in the conf.
    minimize_metric = phase in [Phases.FINE_TUNE, Phases.OPTIMIZATION] or \
        not conf.get_bool('train.validation_metric_higher_is_better', default=True)
    best_validation_metric = math.inf if minimize_metric else -math.inf
    best_epoch = 0
    if resumed_best_metric is not None:
        best_validation_metric = resumed_best_metric
        best_epoch = resumed_best_epoch if resumed_best_epoch is not None else 0
    converge_time = -1
    best_model = copy.deepcopy(model) 
    begin_time = time()
    # === Early Stopping Variables ===
    epochs_without_improvement = 0  # Counter for early stopping
    early_stop_triggered = False  # Flag to indicate if early stopping was triggered

    # R16: record hardware once per run, next to the results (measured values in
    # the CSVs must be attributable to the machine that produced them).
    if fabric.global_rank == 0:
        try:
            import json
            with open(os.path.join(conf.get_string('results_path'), 'hardware.json'), 'w') as f:
                json.dump({'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                           'torch': torch.__version__,
                           'param_count': sum(p.numel() for p in model.parameters())}, f, indent=1)
        except Exception as e:
            print(f"Warning: could not record hardware info: {e}")

    # === TTT step-count snapshots (SPEC_cvpr_experiments R2) ===
    # Only active when train.snapshot_epochs is set; snapshot 0 = frozen inference.
    # Pre-BA metrics at every snapshot; full robust-BA eval only at the first and
    # last snapshot (mode recorded per row).
    snapshot_epochs = [int(e) for e in conf.get_list('train.snapshot_epochs', default=[])]
    snapshot_rows = []
    ttt_step_times = []

    def take_snapshot(step):
        import time as _time
        with_ba = conf.get_bool('ba.run_ba', default=True) and step in (min(snapshot_epochs), max(snapshot_epochs))
        t0 = _time.time()
        # A snapshot is instrumentation: it must never kill the TTT run itself
        # (seen: BA dropping one 3D point -> reshape crash in compute_errors on
        # scene 0099). Record the failure as its own row and keep training.
        try:
            snap_df, _ = epoch_evaluation(train_data, model, conf, step if step > 0 else None, phase,
                                          save_predictions=False, bundle_adjustment=with_ba)
            snap_error = None
        except Exception as e:
            snap_df = None
            snap_error = f'{type(e).__name__}: {e}'
            print(f"[snapshot] step {step} evaluation FAILED ({snap_error}); continuing TTT", flush=True)
        row = {'step': step, 'ba_mode': ('post_ba' if with_ba else 'pre_ba') if snap_error is None else 'eval_failed',
               'eval_seconds': round(_time.time() - t0, 2),
               'mean_ttt_step_seconds': round(sum(ttt_step_times) / max(len(ttt_step_times), 1), 3),
               'total_ttt_seconds': round(sum(ttt_step_times), 1)}
        if snap_error is not None:
            row['error'] = snap_error[:300]
        try:
            row.update({k: float(v) for k, v in snap_df.loc['Mean'].items()})
        except Exception:
            pass
        stats_src = getattr(loss_func, 'outliers_loss', loss_func)
        row.update(getattr(stats_src, 'last_stats', {}) or {})
        prov = checkpoint_provenance(conf, phase, step) or {}
        row.update({'loss_variant': conf.get_string('loss.func_tuning', default=conf.get_string('loss.func', default='')),
                    'scan': conf.get_string('dataset.scan', default=''), 'git': prov.get('git', '')})
        snapshot_rows.append(row)
        import pandas as _pd
        _pd.DataFrame(snapshot_rows).to_csv(
            os.path.join(conf.get_string('results_path'), 'ttt_snapshots.csv'), index=False)
        # R7.4: snapshot checkpoints carry provenance (base checkpoint id, scene,
        # step count, loss variant) so a TTT-adapted weight can never silently
        # seed another run (experiment_guard rejects it on load).
        snap_dir = os.path.join(conf.get_string('results_path'), 'models')
        os.makedirs(snap_dir, exist_ok=True)
        torch.save({'step': step, 'model_state_dict': model.state_dict(),
                    'conf': conf_snapshot(conf),
                    'provenance': {**prov, 'ttt_step': step, 'loss_variant': row['loss_variant']}},
                   os.path.join(snap_dir, f'ttt_snapshot_step{step}.pt'))
        print(f"[snapshot] step {step}: {row.get('our_repro', 'n/a')} px ({row['ba_mode']})", flush=True)

    if snapshot_epochs and 0 in snapshot_epochs and fabric.global_rank == 0:
        take_snapshot(0)

    # === Training Loop ===
    counter = 0
    
    for epoch in range(conf["resuming_epoch"] + 1, num_epochs):
        ba_during_training = not conf.get_bool('ba.only_last_eval') and conf.get_bool('ba.run_ba', default=True)

        import time as _time
        _t_step = _time.time()
        mean_train_loss, train_losses, train_metrics = epoch_train(conf, train_data, model, loss_func, optimizer, scheduler, epoch, phase=phase, fabric=fabric)
        if snapshot_epochs:
            ttt_step_times.append(_time.time() - _t_step)
            # epoch 0 is excluded: step 0 = FROZEN inference, taken once before
            # the loop; the in-loop epoch-0 model has already had one optimizer
            # step and must not overwrite the frozen row.
            if epoch in snapshot_epochs and epoch > 0 and fabric.global_rank == 0:
                take_snapshot(epoch)
        mean_train_loss = fabric.all_reduce(mean_train_loss, reduce_op="mean")
        train_metrics = fabric.all_gather(train_metrics)

        if fabric.global_rank == 0:
            wandb.log({"training loss": mean_train_loss}, step=epoch)
            wandb.log({"LR": scheduler.get_last_lr()[-1]}, step=epoch)
            # R2 collapse check: per-epoch adaptive-loss diagnostics (mean predicted
            # outlier fraction + current percentile thresholds; last scene of the
            # epoch). The fraction should track scene contamination, never drift to 1.
            adaptive_stats = getattr(getattr(loss_func, 'outliers_loss', None), 'last_stats', None)
            if adaptive_stats:
                wandb.log(adaptive_stats, step=epoch)
            if conf.get_int('train.output_mode', default=3) != 1:
                train_metrics_means = CalcMeanBatchMetrics(train_metrics, phase)
                wandb.log(train_metrics_means, step=epoch)
            if epoch % 100 == 0:
                print(f'{fabric.global_rank}:{epoch} Train Loss: {mean_train_loss}')


        # === Evaluation ===
        
        # print('eval_intervals', eval_intervals)
        if epoch % eval_intervals == 0 or epoch == num_epochs - 1:
            # counter +=1
            # print('epoch', epoch)
            if phase is Phases.TRAINING:
                validation_metrics, metrics_list = epoch_evaluation(validation_data, model, conf, epoch, Phases.VALIDATION, save_predictions=True, bundle_adjustment=ba_during_training)
                # validation_metrics, metrics_list = modified_epoch_evaluation(validation_data, model, conf, epoch, Phases.VALIDATION, save_predictions=True, bundle_adjustment=ba_during_training)
            else:
                # print('epoch_evaluation for non Phases.TRAINING line 311 train.py')
                validation_metrics, metrics_list = epoch_evaluation(train_data, model, conf, epoch, phase, save_predictions=True, bundle_adjustment=ba_during_training)
                # validation_metrics, metrics_list = modified_epoch_evaluation(train_data, model, conf, epoch, phase, save_predictions=True, bundle_adjustment=ba_during_training)

            metrics_list_all = metrics_list
            if fabric.world_size > 1:
                metrics_list_all2 = [None for _ in range(fabric.world_size)]
                all_gather_object(metrics_list_all2, metrics_list)
                metrics_list_all = []
                [metrics_list_all.extend(e) for e in metrics_list_all2]

            if fabric.global_rank == 0:
                validation_metrics = evaluation.organize_errors(metrics_list_all)
                validation_metrics2 = validation_metrics.copy()
                validation_metrics2.insert(loc=0, column='epoch', value=[epoch] * len(validation_metrics2))

                validation_metrics['epoch'] = epoch 
                if phase is Phases.OPTIMIZATION:
                    validation_metrics = validation_metrics.drop(['Mean'])
                    # save_metrics_excel(conf, phase=Phases.OPTIMIZATION, df=validation_metrics, epoch=epoch, append=False)
                    general_utils.write_results(conf, validation_metrics, file_name="OPTIMIZATION_over_epochs", append=True)
                if phase is Phases.TRAINING:
                    print(validation_metrics.to_string(), flush=True)
                    save_metrics_excel(conf, phase=Phases.VALIDATION, df=validation_metrics, epoch=epoch, append=False)
                    general_utils.write_results(conf, validation_metrics, file_name="Validation_over_epochs", append=True)
                elif phase is Phases.FINE_TUNE:
                    validation_metrics2 = validation_metrics2.drop(['Mean'])
                    ## commented out also in original resfm code:  save_metrics_excel(conf, phase=Phases.FINE_TUNE, df=validation_metrics2, epoch=None, append=True)
                    general_utils.write_results(conf, validation_metrics2, file_name="Validation_over_fine_tuning", phase=phase, append=True)               
                    
                # metric = validation_metrics.loc[["Mean"], validation_metric].sum(axis=1).values.item()
                
                
                # # #  removed MEAN column becaue for single scene no need for mean 
                
                if phase is Phases.OPTIMIZATION:
                    metric = validation_metrics[validation_metric].mean(axis=0).values.tolist()
                else:
                    metric = validation_metrics.loc[["Mean"], validation_metric].mean(axis=0).values.tolist()

            
                for scene in validation_metrics.index:
                
                    for metric_name in validation_metrics.columns:
                        wandb.log({"Validation_" + metric_name: validation_metrics.loc[scene, metric_name]}, step=epoch)

                path = path_utils.path_to_model(conf, phase, epoch=epoch, best=False)
                current_model = copy.deepcopy(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': current_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_validation_metric': best_validation_metric,
                    'best_epoch': best_epoch,
                    'conf': conf_snapshot(conf),
                    'provenance': checkpoint_provenance(conf, phase, epoch),
                }, path)

                
        
                if isinstance(validation_metric, list) and len(validation_metric) > 1:
                    raise ValueError(f"validation_metric has more than one item in it: {validation_metric} perhaps in future will support this")
                elif isinstance(validation_metric, list) and len(validation_metric) == 1:
                    is_better = (metric[0] < best_validation_metric) if minimize_metric else (metric[0] > best_validation_metric)
                else:
                    is_better = (metric < best_validation_metric) if minimize_metric else (metric > best_validation_metric)
                

                if is_better:
                    converge_time = time() - begin_time
                    if isinstance(metric, list) and len(metric) > 1:
                        raise ValueError(f"metric has more than one item in it: {metric} perhaps in future will support this")
                    elif isinstance(metric, list) and len(metric) == 1:
                        best_validation_metric = metric[0]                   
                    else:
                        best_validation_metric = metric
                  
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                     # saving best model   
                    path = path_utils.path_to_model(conf, phase, epoch=epoch, best=True)
                    # print('path to best model saved', path)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_validation_metric': best_validation_metric,
                        'best_epoch': best_epoch,
                        'conf': conf_snapshot(conf),
                        'provenance': checkpoint_provenance(conf, phase, epoch),
                    }, path)
                    print(f'Updated best validation metric: {best_validation_metric} time so far: {converge_time}')
                    # Reset early stopping counter since we found improvement
                    epochs_without_improvement = 0
                else:
                    # No improvement found
                    epochs_without_improvement += 1
                    
                    # Check if early stopping patience is exceeded
                    if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                        early_stop_triggered = True
                        print(f'\n=== EARLY STOPPING TRIGGERED ===')
                        print(f'No improvement for {epochs_without_improvement} epochs')
                        print(f'Best epoch: {best_epoch} with {validation_metric}: {best_validation_metric}')
                        print(f'Stopping training at epoch {epoch}')
                        print(f'================================\n')
                        break  # Exit the training loop


    # print('best model saved', path)
    # Print early stopping status
    if early_stopping_patience > 0:
        if early_stop_triggered:
            print(f'Training stopped early at epoch {epoch} (patience: {early_stopping_patience} epochs)')
        else:
            print(f'Training completed all {num_epochs} epochs (early stopping was not triggered)')
    
    # === Final Evaluation ===
    train_stat = {}
    run_ba = conf.get_bool('ba.run_ba', default=True)

    best_model_path = path_utils.path_to_model(conf, phase, epoch=None, best=True)
    # print('best model uploaded', best_model_path)
    best_checkpoint = torch.load(best_model_path)
    
    # Handle _orig_mod prefix mismatch between checkpoint and current model
    state_dict = best_checkpoint['model_state_dict']
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
    best_model = model

    print(f"Evaluating using the best model: {best_model_path}")

    if phase is not Phases.TRAINING:
        print("Evaluate training set")
        train_errors, _ = epoch_evaluation(train_data, best_model, conf, None, phase, save_predictions=True, bundle_adjustment=run_ba, plot=True)

    else:
        train_errors = None

    if phase is Phases.TRAINING:
        print("Evaluate validation set")
        validation_errors, _ = epoch_evaluation(validation_data, best_model, conf, None, Phases.VALIDATION, save_predictions=True, bundle_adjustment=run_ba, plot=True)

    else:
        validation_errors = None

    if phase is Phases.TRAINING:
        print("Evaluate test set")
        test_errors, _ = epoch_evaluation(test_data, best_model, conf, None, Phases.TEST, save_predictions=True, bundle_adjustment=run_ba, plot=True)

    else:
        test_errors = None

    # === Return Statistics ===
    train_stat['Convergence time'] = converge_time
    train_stat['best_epoch'] = best_epoch
    train_stat['best_validation_metric'] = best_validation_metric
    train_stat = pd.DataFrame([train_stat])

   
    return train_stat, train_errors, validation_errors, test_errors


def test(conf, model, phase, train_data=None, validation_data=None, test_data=None, fabric=None, run_ba=False):
    # global fabric
    # fabric = fabri
    # Load best model checkpoint
    best_model_path = path_utils.path_to_model(conf, phase, epoch=None, best=True)
    print('best_model_path in dep test', best_model_path)
    best_checkpoint = torch.load(best_model_path)
    
    # Handle _orig_mod prefix mismatch between checkpoint and current model
    state_dict = best_checkpoint['model_state_dict']
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
    best_model = model
    print(f"Evaluating using the best model: {best_model_path}")

    # Evaluate on train data
    if train_data is not None:
        print("Evaluate training set")
        train_data = fabric.setup_dataloaders(train_data)
        train_errors, _ = epoch_evaluation(
            train_data, best_model, conf, epoch=None,
            phase=phase, save_predictions=True, bundle_adjustment=run_ba
        )
      
    else:
        train_errors = None

    # Evaluate on validation data
    if validation_data is not None:
        print("Evaluate validation set")
        validation_data = fabric.setup_dataloaders(validation_data)
        validation_errors, _ = epoch_evaluation(
            validation_data, best_model, conf, epoch=None,
            phase=Phases.VALIDATION, save_predictions=True, bundle_adjustment=run_ba
        )
       
    else:
        validation_errors = None

    # Evaluate on test data
    if test_data is not None:
        print("Evaluate test set")
        test_data = fabric.setup_dataloaders(test_data)
        test_errors, _ = epoch_evaluation(
            test_data, best_model, conf, epoch=None,
            phase=Phases.TEST, save_predictions=True, bundle_adjustment=run_ba,
        )
        
    else:
        test_errors = None

    return train_errors, validation_errors, test_errors


