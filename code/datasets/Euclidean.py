import csv
import sys
# sys.path.append('/home/projects/ronen/fadi/myProjects/outliersRemoval3D/code/datasets')
# sys.path.append('/home/projects/ronen/fadi/myProjects/outliersRemoval3D/code/utils')
# sys.path.append('/home/projects/ronen/fadi/myProjects/outliersRemoval3D/code')
# print(sys.path)
import cv2  # Do not remove
import torch
import os
import sys
import warnings 
from utils.dataset_utils import get_M_valid_points
from utils.Phases import Phases
from utils.path_utils import path_to_outliers
from utils import geo_utils, general_utils, dataset_utils, path_utils, plot_utils
import scipy.io as sio
import numpy as np
import os.path
import networkx as nx
from tqdm import tqdm
import copy

import torch
import numpy as np
import re


# from utils.path_utils import path_to_outliers, write_results
from utils.general_utils import write_results



def detect_outliers_statistical(reprojection_errors, weight_method='mad', alpha=1.0):
    """
    Detect outliers based on reprojection error statistics
    
    Args:
        reprojection_errors: Per-point reprojection errors
        method: 'mad' (Median Absolute Deviation) or 'std' (Standard Deviation)
        alpha: Threshold multiplier
    
    Returns:
        outlier_mask: Binary mask indicating outliers
    """
    
    if weight_method == 'mad':
        median = torch.median(reprojection_errors)
        mad = torch.median(torch.abs(reprojection_errors - median))
        threshold = median + alpha * 1.4826 * mad  # 1.4826 converts MAD to std
        
    elif weight_method == 'std':
        mean = torch.mean(reprojection_errors)
        std = torch.std(reprojection_errors)
        threshold = mean + alpha * std

    elif weight_method == 'huber':
        # Huber M-estimate of scale via a few IRLS iterations (robust, sits
        # between MAD and STD in sensitivity), thresholded like the others:
        # median + alpha * scale.
        r = reprojection_errors
        med = torch.median(r)
        scale = 1.4826 * torch.median(torch.abs(r - med)) + 1e-9  # MAD initialisation
        c = 1.345  # standard Huber tuning constant (95% efficiency)
        for _ in range(5):
            z = torch.abs(r - med) / scale
            w = torch.clamp(c / z.clamp(min=1e-9), max=1.0)  # Huber weights
            scale = torch.sqrt((w * (r - med) ** 2).sum() / w.sum().clamp(min=1.0))
        threshold = med + alpha * scale

    # outlier_mask = reprojection_errors > threshold
    outlier_mask = (reprojection_errors > threshold).to(torch.float32)

    return outlier_mask

def compute_huber_weights(reprojection_errors, threshold=1.0):
    """
    Huber-style robust weighting for probabilistic self-supervision
    
    Args:
        reprojection_errors: Per-point reprojection errors
        threshold: Huber threshold parameter (typically 1.0 pixel)
    
    Returns:
        weights: Huber weights for each point
    """
    weights = torch.ones_like(reprojection_errors)
    mask = reprojection_errors > threshold
    weights[mask] = threshold / reprojection_errors[mask]
    return weights

def get_raw_data(conf, scan, phase, stage=1):
    """
    Load raw data for SfM training or evaluation.

    Returns:
        M (torch.Tensor): 2D points matrix [2m, n]
        Ns (torch.Tensor): Inverse calibration matrices [m, 3, 3]
        Ps_gt (torch.Tensor): Ground-truth projection matrices [m, 3, 4]
        outliers (torch.Tensor): Ground-truth outlier mask [m, n]
        dict_info (dict): Metadata (e.g., outliers percent)
        names_list (list): List of image names
        M_original (torch.Tensor): Original points matrix (before filtering)
    """

    # === Setup paths and parameters ===
    dataset_name = conf.get_string('dataset.dataset', default="megadepth")
    dataset_path = os.path.join(path_utils.path_to_datasets(dataset_name), f'{scan}.npz')
    
    output_mode = conf.get_int('train.output_mode', default=-1)
    use_gt = conf.get_bool('dataset.use_gt')
    remove_outliers_gt = conf.get_bool('dataset.remove_outliers_gt', default=False)
    remove_outliers_pred = False
    outliers_threshold = conf.get_float("test.outliers_threshold", default=0.6)
    if scan is None:
        scan = conf.get_string('dataset.scan')

    print(f"Used Dataset: {dataset_name}")
    print(f"Loading from: {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True)
 
   

    # === Extract raw data ===
    M_np = dataset['M']
    Ps_gt_np = dataset['Ps_gt']
    Ns_np = dataset['Ns']
    names_list = dataset['namesList']
    # Some datasets (Olsson/Euclidean) store namesList as shape (1, N); flatten to
    # (N,) so per-camera indexing (names_list[valid_cam_indices]) works uniformly.
    names_list = np.asarray(names_list).reshape(-1)
    outliers_np = dataset.get('outliers2', np.zeros((M_np.shape[0] // 2, M_np.shape[1])))

    # === Initialize info dictionary ===
    dict_info = {
        'pointsNum': M_np.shape[1],
        'camsNum': M_np.shape[0] // 2,
        'outliersPercent': float("%.4f" % dataset.get('outlier_pct', 0.0)),
        'outliers_pred': torch.zeros_like(torch.from_numpy(outliers_np).float())
    }

    # === Convert to torch tensors ===
    M = torch.from_numpy(M_np).float()
    M_original = M.clone()
    Ps_gt = torch.from_numpy(Ps_gt_np).float()
    Ns = torch.from_numpy(Ns_np).float()
    outliers = torch.from_numpy(outliers_np).float()
    outliers_mask = outliers.clone()
    

    # weight_not_remove: soft-weight points by the frozen head score instead of
    # hard-removing them (U-ESFM soft-weighting variant). Scores are used as a
    # per-point weight (proj_err_weight) by ESFMLoss_weighted_by_rep_err.
    frozen_weights = None
    weight_not_remove = conf.get_bool('test.weight_not_remove', default=False)
    # hybrid_remove_weight: 3-band adaptive scheme (mirrors the adaptive loss's
    # percentile bands). Per-scene head-score thresholds: score > high -> REMOVE,
    # score < low -> full weight, in-between -> soft-weight by (1-score).
    hybrid_remove_weight = conf.get_bool('test.hybrid_remove_weight', default=False)
    # mad_remove_head_weight: MAD (reprojection-error statistics) removes the
    # confident outliers; the head score soft-weights the survivors by (1-score).
    # Requires the TEST pass to have saved both 'outliers_pred' (head) and
    # 'outliers_mad' (MAD mask) in the npz (train.py test.mad_remove_head_weight).
    mad_remove_head_weight = conf.get_bool('test.mad_remove_head_weight', default=False)
    # std_/huber_remove_head_weight: ablations of madweight that keep the head
    # soft-weight identical but swap the robust statistic used for removal
    # (STD = non-robust; Huber = robust IRLS scale). Require the TEST pass to have
    # saved 'outliers_std'/'outliers_huber' respectively.
    std_remove_head_weight = conf.get_bool('test.std_remove_head_weight', default=False)
    huber_remove_head_weight = conf.get_bool('test.huber_remove_head_weight', default=False)
    # === Fine-tuning: Load predicted outliers ===
    if phase is Phases.FINE_TUNE and output_mode == 3:
        print(f"Fine-tuning phase: loading predicted outliers for scan {scan}")
        print("Loading outliers from:", path_to_outliers(conf, Phases.TEST, epoch=None, scan=scan))
        _npz = np.load(path_to_outliers(conf, Phases.TEST, epoch=None, scan=scan) + ".npz")
        outliers_mask_np = _npz['outliers_pred']
        if mad_remove_head_weight:
            # MAD removes confident outliers; head soft-weights the survivors.
            madmask = np.asarray(_npz['outliers_mad'])
            outliers_mask = torch.from_numpy(madmask > 0.5)
            remove_outliers_pred = True
            frozen_weights = torch.from_numpy(outliers_mask_np).float()  # head scores
            print(f"[mad+headweight] mad_remove_frac={float((madmask > 0.5).mean()):.3f}")
        elif std_remove_head_weight:
            # STD removes confident outliers (non-robust threshold); head soft-weights survivors.
            stdmask = np.asarray(_npz['outliers_std'])
            outliers_mask = torch.from_numpy(stdmask > 0.5)
            remove_outliers_pred = True
            frozen_weights = torch.from_numpy(outliers_mask_np).float()  # head scores
            print(f"[std+headweight] std_remove_frac={float((stdmask > 0.5).mean()):.3f}")
        elif huber_remove_head_weight:
            # Huber (robust IRLS scale) removes confident outliers; head soft-weights survivors.
            hubmask = np.asarray(_npz['outliers_huber'])
            outliers_mask = torch.from_numpy(hubmask > 0.5)
            remove_outliers_pred = True
            frozen_weights = torch.from_numpy(outliers_mask_np).float()  # head scores
            print(f"[huber+headweight] huber_remove_frac={float((hubmask > 0.5).mean()):.3f}")
        elif hybrid_remove_weight:
            # Adaptive 3-band: remove confident outliers, soft-weight the ambiguous,
            # keep confident inliers at full weight. Thresholds are per-scene
            # percentiles of the head-score distribution (default 20/80).
            s = torch.from_numpy(outliers_mask_np).float()
            lo_pct = conf.get_float('test.hybrid_low_pct', default=20.0)
            hi_pct = conf.get_float('test.hybrid_high_pct', default=80.0)
            # np.quantile, not torch.quantile: torch.quantile raises "input tensor
            # is too large" past ~16M elements, which crashes on big scenes (large
            # 1DSfM / MegaDepth); numpy has no such limit.
            flat_np = np.asarray(outliers_mask_np).reshape(-1)
            low_thr = float(np.quantile(flat_np, lo_pct / 100.0))
            high_thr = float(np.quantile(flat_np, hi_pct / 100.0))
            if high_thr - low_thr < 1e-6:
                # Degenerate head-score distribution: the head scores ~0 almost
                # everywhere (a clean scene it's confident about), so the 20/80
                # percentiles collapse to the same value and the 3-band scheme
                # produced NaN loss. Fall back to a no-op: keep all points at full
                # weight, no removal (== plain reprojection for this scene).
                frozen_weights = torch.zeros_like(s)
                print(f"[hybrid] degenerate score dist (low={low_thr:.3f} high={high_thr:.3f}) -> no-op")
            else:
                outliers_mask = s > high_thr            # remove confident outliers
                remove_outliers_pred = True
                fw = s.clone()                           # weight survivors by (1-score)
                fw[s < low_thr] = 0.0                     # confident inliers -> full weight
                frozen_weights = fw
                print(f"[hybrid] low_thr={low_thr:.3f} high_thr={high_thr:.3f} "
                      f"remove_frac={float((s > high_thr).float().mean()):.3f}")
        elif weight_not_remove:
            # Keep continuous frozen scores as per-point weights; do NOT remove.
            frozen_weights = torch.from_numpy(outliers_mask_np).float()
        else:
            outliers_mask = torch.from_numpy(outliers_mask_np > outliers_threshold)
            remove_outliers_pred = True


    # === Remove outliers ===
    if remove_outliers_gt or remove_outliers_pred:
        outliers_mask = outliers_mask > 0  # ensure boolean mask

        # Convert shape [2m, n] → [n, m, 2] → [m, n, 2]
        M = M.transpose(0, 1).reshape(-1, M.shape[0] // 2, 2).transpose(0, 1)
        M[outliers_mask] = 0
        # Back to [2m, n]
        M = M.transpose(0, 1).reshape(-1, M.shape[0] * 2).transpose(0, 1)

    # === Keep only largest connected component if fine-tuning with outputMode == 3 ===
    if phase is Phases.FINE_TUNE and output_mode == 3:
        _, valid_cam_indices = dataset_utils.check_if_M_connected(M, thr=1, return_largest_component=True)
        double_cam_indices = [j for i in [[idx * 2, idx * 2 + 1] for idx in valid_cam_indices] for j in i]

        Ns = Ns[valid_cam_indices]
        Ps_gt = Ps_gt[valid_cam_indices]
        outliers = outliers[valid_cam_indices]
        names_list = names_list[valid_cam_indices]
        M = M[double_cam_indices]
        M_original = M_original[double_cam_indices]
        if frozen_weights is not None:
            frozen_weights = frozen_weights[valid_cam_indices]

    if stage == 2:
        # Try to get the path from config, but make it optional
        # For stage 2, weights are loaded manually in the training script
        
        try:
            weight_method = conf.get_string('postprocessing.weight_method')
            print(f"Got weight_method from postprocessing: {weight_method}")
        except Exception as e:
            weight_method = 'global'  # Default fallback
            print(f"Failed to get weight_method: {e}")
        
        
        alpha = conf.get_float('postprocessing.alpha')
        
        print(conf.get_string('results_path'))
        
        
        # proj_err_path = os.path.join(
        # re.sub(r'/2_stage/(global|track_and_global|(?:mad|std|huber)_alpha_\d+_\d+)/', '/1_stage/', 
        #         conf.get_string('results_path')),
        #     'reprojection_errors.npy'
        # )

        proj_err_path = os.path.join(
        re.sub(r'/2_stage/ESFMLoss_weighted_by_rep_err/(global|track_and_global|(?:mad|std|huber)(?:_alpha_\d+_\d+)?)/', 
                '/1_stage/ESFMLoss/', 
                conf.get_string('results_path')
            ),
            'reprojection_errors.npy'
        )

        print('proj_err_path', proj_err_path)
        try:
         
            reprojection_errors = np.load(proj_err_path, allow_pickle=True)
            reprojection_errors = torch.from_numpy(reprojection_errors).float()
        except:
            # Config key not found - will be loaded manually later (e.g., in stage 2)
            print(f"proj_err_path not found: {proj_err_path}")

   
        print('weight_method', weight_method)  
        print('alpha', alpha)  
        if weight_method == 'huber':
            rep_error_weights = compute_huber_weights(reprojection_errors, threshold=alpha)
        elif weight_method == 'mad' or weight_method == 'std':
            rep_error_weights = detect_outliers_statistical(reprojection_errors, weight_method=weight_method, alpha=alpha)
        
        try:
            proj_err_weight_path = conf.get_string('results_path')
            
            proj_err_dir = os.path.dirname(proj_err_weight_path)
            if not os.path.exists(proj_err_dir):
                os.makedirs(proj_err_dir)
                print(f"Created directory: {proj_err_dir}")
                
            # Save the reprojection error weights
            write_results(conf, rep_error_weights, file_name="reprojection_errror_weights", append=False)

            print(f"Successfully saved projection error weights to: {proj_err_weight_path}")
        
        except Exception as e:
            print(f"ERROR: Failed to save reprojection errors: {e}")
            import traceback
            traceback.print_exc()
        
        return M, Ns, Ps_gt, outliers, dict_info, names_list, M_original, rep_error_weights
    else:
        return M, Ns, Ps_gt, outliers, dict_info, names_list, M_original, frozen_weights 




# def get_raw_data(conf, scan, phase, with_proj_err=False):
#     """
#     Load raw data for SfM training or evaluation.

#     Returns:
#         M (torch.Tensor): 2D points matrix [2m, n]
#         Ns (torch.Tensor): Inverse calibration matrices [m, 3, 3]
#         Ps_gt (torch.Tensor): Ground-truth projection matrices [m, 3, 4]
#         outliers (torch.Tensor): Ground-truth outlier mask [m, n]
#         dict_info (dict): Metadata (e.g., outliers percent)
#         names_list (list): List of image names
#         M_original (torch.Tensor): Original points matrix (before filtering)
#     """

#     # === Setup paths and parameters ===
#     dataset_name = conf.get_string('dataset.dataset', default="megadepth")
#     dataset_path = os.path.join(path_utils.path_to_datasets(dataset_name), f'{scan}.npz')
    
#     output_mode = conf.get_int('train.output_mode', default=-1)
#     use_gt = conf.get_bool('dataset.use_gt')
#     remove_outliers_gt = conf.get_bool('dataset.remove_outliers_gt', default=False)
#     remove_outliers_pred = False
#     outliers_threshold = conf.get_float("test.outliers_threshold", default=0.6)
#     if scan is None:
#         scan = conf.get_string('dataset.scan')

#     print(f"Used Dataset: {dataset_name}")
#     print(f"Loading from: {dataset_path}")
#     dataset = np.load(dataset_path, allow_pickle=True)
 
   

#     # === Extract raw data ===
#     M_np = dataset['M']
#     Ps_gt_np = dataset['Ps_gt']
#     Ns_np = dataset['Ns']
#     names_list = dataset['namesList']
#     outliers_np = dataset.get('outliers2', np.zeros((M_np.shape[0] // 2, M_np.shape[1])))

#     # === Initialize info dictionary ===
#     dict_info = {
#         'pointsNum': M_np.shape[1],
#         'camsNum': M_np.shape[0] // 2,
#         'outliersPercent': float("%.4f" % dataset.get('outlier_pct', 0.0)),
#         'outliers_pred': torch.zeros_like(torch.from_numpy(outliers_np).float())
#     }

#     # === Convert to torch tensors ===
#     M = torch.from_numpy(M_np).float()
#     M_original = M.clone()
#     Ps_gt = torch.from_numpy(Ps_gt_np).float()
#     Ns = torch.from_numpy(Ns_np).float()
#     outliers = torch.from_numpy(outliers_np).float()
#     outliers_mask = outliers.clone()
    

#     # === Fine-tuning: Load predicted outliers ===
#     if phase is Phases.FINE_TUNE and output_mode == 3:
#         print(f"Fine-tuning phase: loading predicted outliers for scan {scan}")
#         print("Loading outliers from:", path_to_outliers(conf, Phases.TEST, epoch=None, scan=scan))
#         outliers_mask_np = np.load(path_to_outliers(conf, Phases.TEST, epoch=None, scan=scan) + ".npz")['outliers_pred']
#         outliers_mask = torch.from_numpy(outliers_mask_np > outliers_threshold)
#         remove_outliers_pred = True


#     # === Remove outliers ===
#     if remove_outliers_gt or remove_outliers_pred:
#         outliers_mask = outliers_mask > 0  # ensure boolean mask

#         # Convert shape [2m, n] → [n, m, 2] → [m, n, 2]
#         M = M.transpose(0, 1).reshape(-1, M.shape[0] // 2, 2).transpose(0, 1)
#         M[outliers_mask] = 0
#         # Back to [2m, n]
#         M = M.transpose(0, 1).reshape(-1, M.shape[0] * 2).transpose(0, 1)

#     # === Keep only largest connected component if fine-tuning with outputMode == 3 ===
#     if phase is Phases.FINE_TUNE and output_mode == 3:
#         _, valid_cam_indices = dataset_utils.check_if_M_connected(M, thr=1, return_largest_component=True)
#         double_cam_indices = [j for i in [[idx * 2, idx * 2 + 1] for idx in valid_cam_indices] for j in i]

#         Ns = Ns[valid_cam_indices]
#         Ps_gt = Ps_gt[valid_cam_indices]
#         outliers = outliers[valid_cam_indices]
#         names_list = names_list[valid_cam_indices]
#         M = M[double_cam_indices]
#         M_original = M_original[double_cam_indices]

#     if with_proj_err: 
#         # Try to get the path from config, but make it optional
#         # For stage 2, weights are loaded manually in the training script
#         try:
#             proj_err_path = os.path.join(conf.get_string('results_path'), 'reprojection_errors.npy').replace('2_', '1_')
#             #if proj_err_path and os.path.exists(proj_err_path):
#             reprojection_errors = np.load(proj_err_path, allow_pickle=True)
#             reprojection_errors = torch.from_numpy(reprojection_erros).float()
#             # else:
#             #     # Path not found or empty - will be loaded manually later
#             #     proj_err_weight = None
#         except:
#             # Config key not found - will be loaded manually later (e.g., in stage 2)
#             # proj_err_weight = None
#             print(f"proj_err_path not found: {os.path.join(conf.get_string('results_path'), 'reprojection_errors.npy').replace('2_', '1_')}")

         
#         # Get weight_method from postprocessing section
#         try:
#             weight_method = conf.get_string('postprocessing.weight_method')
#             print(f"Got weight_method from postprocessing: {weight_method}")
#         except Exception as e:
#             weight_method = 'global'  # Default fallback
#             print(f"Failed to get weight_method: {e}, using default: {weight_method}")
#         print('weight_method', weight_method)    
#         if weight_method == 'global':
#             # log_errors = np.log(rep_errors_like_M + 1)
#             # mean, std = np.nanmean(log_errors), np.nanstd(log_errors)
#             # normalized = (log_errors - mean) / std
#             # rep_error_weights = 1 / (1 + np.exp(-normalized))
#             # Usage:
#             rep_error_weights = compute_log_sigmoid_weights(reprojection_errors)
#         elif weight_method == 'track_and_global': 

#             rep_error_weights, track_factors = track_normalized_weighting(reprojection_errors) 
#         # elif weight_method == 'remove-outliers': 
#         #     rep_errors_like_M[rep_errors_like_M > 1] = np.nan
#         #     rep_error_weights = rep_errors_like_M
#         try:
#             proj_err_weight_path = conf.get_string('results_path')
            
#             proj_err_dir = os.path.dirname(proj_err_weight_path)
#             if not os.path.exists(proj_err_dir):
#                 os.makedirs(proj_err_dir)
#                 print(f"Created directory: {proj_err_dir}")
                
#             # Save the reprojection error weights
#             write_results(conf, rep_error_weights, file_name="reprojection_errror_weights", append=False)

#             print(f"Successfully saved projection error weights to: {proj_err_weight_path}")
        
#         except Exception as e:
#             print(f"ERROR: Failed to save reprojection errors: {e}")
#             import traceback
#             traceback.print_exc()
        
#         return M, Ns, Ps_gt, outliers, dict_info, names_list, M_original, rep_error_weights
#     else:
#         return M, Ns, Ps_gt, outliers, dict_info, names_list, M_original




def test_Ps_M(Ps, M, Ns):
    global_rep_err = geo_utils.calc_global_reprojection_error(Ps.numpy(), M.numpy(), Ns.numpy())
    print("Reprojection Error: Mean = {}, Max = {}".format(np.nanmean(global_rep_err), np.nanmax(global_rep_err)))
    return np.nanmean(global_rep_err), np.nanmax(global_rep_err)




def test_euclidean_dataset(scan):
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Euclidean', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scan))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']

    M_gt = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    print("Test Ps and M")
    test_Ps_M(Ps_gt, M, Ns)

    print("Test Ps and M_gt")
    test_Ps_M(Ps_gt, M_gt, Ns)




pass