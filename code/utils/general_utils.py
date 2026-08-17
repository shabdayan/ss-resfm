#!/usr/bin/env python3
"""
ResFM Single Scene Optimization - Reproducible Version
"""

# ⭐ CRITICAL: Set these BEFORE any PyTorch imports
import os

# Make CUDA operations deterministic
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For cuBLAS determinism
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'           # Synchronize CUDA calls
os.environ['PYTHONHASHSEED'] = '0'                 # For Python hash consistency

# Now import the rest
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# ... rest of your imports ...

import torch
from datetime import datetime
from scipy.io import savemat
import shutil
from pyhocon import HOCONConverter,ConfigTree
import sys
import json
from pyhocon import ConfigFactory
import argparse
import os
import numpy as np
import pandas as pd
import portalocker
from utils.Phases import Phases
from utils.path_utils import path_to_exp, path_to_cameras, path_to_code_logs, path_to_conf, path_to_outliers, \
    path_to_metrics, path_to_phase
import random




def log_code(conf):
    code_path = path_to_code_logs(conf)

    files_to_log = ["train.py", "single_scene_optimization.py", "multiple_scenes_learning.py", "loss_functions.py"]
    for file_name in files_to_log:
        shutil.copyfile('{}'.format(file_name), os.path.join(code_path, file_name))

    dirs_to_log = ["datasets", "models", "utils"]
    # datasets/ holds the npz DATA folders (megadepth/, 1dsfm/, ...) in this repo,
    # not just loader code — copying them archived ~80GB per training run. The
    # snapshot's purpose is CODE provenance: keep source files, skip data/caches.
    keep = ('.py', '.conf', '.md', '.txt')
    ignore_data = lambda d, names: [n for n in names
                                    if not n.endswith(keep)
                                    and not os.path.isdir(os.path.join(d, n))] \
                                   + [n for n in names
                                      if os.path.isdir(os.path.join(d, n))
                                      and n in ('__pycache__', 'megadepth', '1dsfm', 'strecha',
                                                'blendedmvs', 'Euclidean', 'raw')]
    for dir_name in dirs_to_log:
        shutil.copytree('{}'.format(dir_name), os.path.join(code_path, dir_name),
                        dirs_exist_ok=True, ignore=ignore_data)

    # Print conf
    with open(os.path.join(code_path, 'exp.conf'), 'w') as conf_log_file:
        conf_log_file.write(HOCONConverter.convert(conf, 'hocon'))


def save_camera_mat(conf, save_cam_dict, scan, phase, epoch=None):
    path_cameras = path_to_cameras(conf, phase, epoch=epoch, scan=scan)
    # Move any CUDA/torch tensors to CPU numpy first — np.savez can't serialize a
    # device tensor (hit during sequential-optimization subset saves).
    save_cam_dict = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                     for k, v in save_cam_dict.items()}
    np.savez(path_cameras, **save_cam_dict)  # npz consumed by evaluate_single_scene.py
    try:
        savemat(path_cameras + ".mat", save_cam_dict)  # for matlab file
    except Exception as e:
        # some outputs entries (object arrays, None) are not matlab-serializable;
        # the npz above is the authoritative artifact
        print(f"savemat skipped for {scan}: {e}")

def save_outliers_mat(conf, save_cam_dict, scan, phase, epoch=None):
    path_outliers = path_to_outliers(conf, phase, epoch=epoch, scan=scan)
    np.savez(path_outliers, **save_cam_dict)
    # savemat(path_cameras, save_cam_dict)

def save_metrics_excel(conf, phase, df, epoch=None, append=False):
    results_file_path = path_to_metrics(conf, phase, epoch=epoch, scan=None)

    if append:
        locker_file = results_file_path.replace('xlsx', 'lock')
        lock = portalocker.Lock(locker_file, timeout=1000)
        with lock:
            if os.path.exists(results_file_path):
                prev_df = pd.read_excel(results_file_path).set_index("Scene")
                merged_err_df = pd.concat([prev_df, df])
            else:
                merged_err_df = df

            merged_err_df.to_excel(results_file_path)
    else:
        if os.path.exists(results_file_path):
            os.remove(results_file_path)
        df.to_excel(results_file_path)

# def write_results(conf, data, file_name="Results",  phase=None, append=False):

#     # if phase is Phases.FINE_TUNE:
#     #     exp_path = path_to_phase(conf, phase)
#     # else:
#     #     exp_path = path_to_exp(conf)

#     path_results =conf.get_string('results_path')
#     print('results_path', path_results)

#     if data is pd.DataFrame: 
#         print('data is pd.DataFrame')
#         results_file_path = os.path.join(path_results, '{}.xlsx'.format(file_name))

#         if append:
#             locker_file = os.path.join(path_results, '{}.lock'.format(file_name))
#             lock = portalocker.Lock(locker_file, timeout=1000)
#             with lock:
#                 if os.path.exists(results_file_path):
#                     prev_df = pd.read_excel(results_file_path).set_index("Scene")
#                     merged_err_df = pd.concat([prev_df, df])
#                 else:
#                     merged_err_df = df

#                 merged_err_df.to_excel(results_file_path)
#         else:
#             df.to_excel(results_file_path)
    
#     elif data is np.array: 
#         print('data is np.array')
#         results_file_path = os.path.join(path_results, '{}.npy'.format(file_name))
#         np.save(proj_err_weight_path, data, allow_pickle=True) 

def write_results(conf, data, file_name="Results", phase=None, append=False, aggregated=False):

    # if phase is Phases.FINE_TUNE:
    #     exp_path = path_to_phase(conf, phase)
    # else:
    #     exp_path = path_to_exp(conf)

    if aggregated: 
            path_results = './results/aggregated/'
    else:
        path_results = conf.get_string('results_path')
    # print('results_path', path_results)
    
    # Create results directory if it doesn't exist
    os.makedirs(path_results, exist_ok=True)

    if isinstance(data, pd.DataFrame): 
        # print('data is pd.DataFrame')
        
        results_file_path = os.path.join(path_results, '{}.xlsx'.format(file_name))

        if append:
            locker_file = os.path.join(path_results, '{}.lock'.format(file_name))
            lock = portalocker.Lock(locker_file, timeout=1000)
            with lock:
                if os.path.exists(results_file_path):
                    prev_df = pd.read_excel(results_file_path).set_index("Scene")
                    merged_err_df = pd.concat([prev_df, data])
                else:
                    merged_err_df = data

                merged_err_df.to_excel(results_file_path)
                # print('results_file_path', results_file_path)
        else:
            data.to_excel(results_file_path)
    
    elif isinstance(data, np.ndarray): 
        # print('data is np.ndarray')
        results_file_path = os.path.join(path_results, '{}.npy'.format(file_name))
        np.save(results_file_path, data, allow_pickle=True)


def init_exp_version(phase=None):

    if phase is Phases.OPTIMIZATION:
        return ''
    else:
        return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())



def init_exp_version(phase=None):

    if phase is Phases.OPTIMIZATION:
        return ''
    else:
        return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_error(err_string):
    print(err_string, file=sys.stderr)


def config_tree_to_string(config):
    config_dict={}
    for it in config.keys():
        if isinstance(config[it],ConfigTree):
            it_dict = {key:val for key,val in config[it].items()}
            config_dict[it]=it_dict
        else:
            config_dict[it] = config[it]
    return json.dumps(config_dict)


def bmvm(bmats, bvecs):
    return torch.bmm(bmats, bvecs.unsqueeze(-1)).squeeze()


def get_full_conf_vals(conf):
    # return a conf file as a dictionary as follow:
    # "key.key.key...key": value
    # Useful for the conf.put() command
    full_vals = {}
    for key, val in conf.items():
        if isinstance(val, dict):
            part_vals = get_full_conf_vals(val)
            for part_key, part_val in part_vals.items():
                full_vals[key + "." +part_key] = part_val
        else:
            full_vals[key] = val

    return full_vals


def parse_external_params(ext_params_str, conf):
    for param in ext_params_str.split(','):
        key_val = param.split(':')
        if len(key_val) == 3:
            conf[key_val[0]][key_val[1]] = key_val[2]
        elif len(key_val) == 2:
            conf[key_val[0]] = key_val[1]
    return conf

# def set_seed(seed):

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.
#     torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True



def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
        
    Note:
        Setting deterministic=True may reduce performance by 10-20%.
        Some operations cannot be made fully deterministic.
    """
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # cuDNN reproducibility settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Force deterministic algorithms (with warnings instead of errors)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"Warning: Could not enable deterministic algorithms: {e}")
    
    # Set hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Random seed set to {seed} for reproducibility")
    print("⚠️  Deterministic mode may reduce training speed by 10-30%")


def worker_init_fn(worker_id):
    """
    Initialize DataLoader worker with proper seeding for reproducibility.
    
    Args:
        worker_id (int): ID of the worker process
        
    This ensures each DataLoader worker has a different but deterministic seed.
    Call this in DataLoader as: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    import numpy as np
    import random
    import torch
    
    # Get base seed from torch's generator
    worker_seed = torch.initial_seed() % 2**32
    
    # Seed each worker differently but deterministically
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# def init_exp(default_phase):
#     # Parse Arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--conf', default=None, type=str)
#     parser.add_argument('--scan', type=str, default=None)
#     parser.add_argument('--exp_version', type=str, default=None)
#     parser.add_argument('--external_params', type=str, default=None)
#     parser.add_argument('--phase', type=str, default=default_phase)
#     parser.add_argument('--pretrainedPath', type=str, default=None)
#     parser.add_argument('--resume', type=str, default=None)
#     parser.add_argument('--wandb', type=int, default=1)
#     parser.add_argument('--weight-method', type=str, default=None)
#     opt = parser.parse_args()
#     print(opt.external_params)

#     # weight_method = opt.weight_method

#     # Init Device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Init Conf
#     if opt.conf is not None: 
#         conf_file_path = opt.conf
#     else: 
#         # Default config file path if none provided
#         conf_file_path = '/MVG/final-project/resfm-main-reprojection-error-outlier-weight/code/confs/{opt.weight-method}/confs_esfm_{opt.num_epochs}epochs_with_outliers.conf'
    
#     conf = ConfigFactory.parse_file(conf_file_path)
#     conf["original_file_name"] = opt.conf
#     test_scans_list = conf.get_list('dataset.test_set')
    
#     # if weight_method is None:
#     #     weight_method = conf.get_string('loss.weight_method')

#     # print("The weight method is set to:", weight_method)
#     # exit()
#     # Init Phase
#     phase = Phases[opt.phase]

#     # Init external params
#     if opt.external_params is not None:
#         conf = parse_external_params(opt.external_params, conf)

#     # Init Version
#     if opt.exp_version is None:
#         exp_version = init_exp_version(phase)
#     else:
#         exp_version = opt.exp_version
#     conf['exp_version'] = exp_version

#     # Init scan
#     if opt.scan is not None:
#         conf['dataset']['scan'] = opt.scan
#     elif 'scan' not in conf['dataset'].keys():
#         conf['dataset']['scan'] = 'Multiple_Scenes'

#     # Init Seed
#     seed = conf.get_int('random_seed', default=None)
#     print("The seed is set to:", seed)
#     if seed is not None:
#         set_seed(seed)


#     torch.set_float32_matmul_precision('high')

#     # init wandb or not
#     conf['wandb'] = opt.wandb
    
#     # # Store weight-method in conf
#     # conf['weight_method'] = opt.weight_method

#     # Load checkpoint model for finetuning
#     conf['pretrainedPath'] = opt.pretrainedPath

#     conf['resume'] = opt.resume
#     conf["resuming_epoch"] = 0

#     if phase is Phases.FINE_TUNE:
#         optimization_num_of_epochs = conf.get_int("train.optimization_num_of_epochs")
#         optimization_eval_intervals = conf.get_int('train.optimization_eval_intervals')
#         optimization_lr = conf.get_float('train.optimization_lr')
#         test_scans_list = conf.get_list('dataset.test_set')

#         conf['dataset']['scans_list'] = test_scans_list
#         conf['train']['num_of_epochs'] = optimization_num_of_epochs
#         conf['train']['eval_intervals'] = optimization_eval_intervals
#         conf['train']['lr'] = optimization_lr
    
    
#     return conf, device, phase



def init_exp(default_phase):
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default=None, type=str)
    parser.add_argument('--scan', type=str, default=None)
    parser.add_argument('--exp_version', type=str, default=None)
    parser.add_argument('--external_params', type=str, default=None)
    parser.add_argument('--phase', type=str, default=default_phase)
    parser.add_argument('--pretrainedPath', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--weight_method', type=str, default=None)
    parser.add_argument('--architecture_type', type=str, default=None)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Training stage: 1 (first) or 2 (second)')
    parser.add_argument('--with_proj_err', action='store_true', help='Include projection error weights')
    parser.add_argument('--extract_weights_only', action='store_true', help='Only extract weights from existing model')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples')
    parser.add_argument('--results_aggregation_file', type=str, default=None,
                        help='Aggregated results file')
    parser.add_argument('--compute_loss_normalization', action='store_true',
                        help='Compute loss normalization')
    opt = parser.parse_args()
    # print(opt.external_params)

    # weight_method = opt.weight_method

    # Init Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init Conf
    # if opt.conf is not None: 
    conf_file_path = opt.conf
    # print('conf_file_path', conf_file_path)
    # else: 
    #     # Default config file path if none provided
    #     conf_file_path = f'/home/projects/bagon/ortalda/MVG/final-project/resfm-main-reprojection-error-outlier-weight/code/confs/{opt.weight_method}/confs_esfm_{opt.num_epochs}epochs_with_outliers.conf'
    #     print('conf_file_path2', conf_file_path)
    # exit()
    conf = ConfigFactory.parse_file(conf_file_path)
    # conf["original_file_name"] = opt.conf
    # test_scans_list = conf.get_list('dataset.test_set')
    
    # if weight_method is None:
    #     weight_method = conf.get_string('loss.weight_method')

    # print("The weight method is set to:", weight_method)
    # exit()
    # Init Phase
    phase = Phases[opt.phase]

    # # Init external params
    # if opt.external_params is not None:
    #     conf = parse_external_params(opt.external_params, conf)

    # Init Version
    if opt.exp_version is None:
        exp_version = init_exp_version(phase)
    else:
        exp_version = opt.exp_version
    conf['exp_version'] = exp_version

    # Init scan
    if 'dataset' not in conf:
        conf['dataset'] = {}
    
    if opt.scan is not None:
        print(f"Setting scan from command line: {opt.scan}")
        conf['dataset']['scan'] = opt.scan
    elif 'scan' not in conf['dataset'].keys():
        print("No scan provided, using default: Multiple_Scenes")
        conf['dataset']['scan'] = 'Multiple_Scenes'
    else:
        print(f"Using scan from config file: {conf['dataset']['scan']}")

    # Init Seed
    seed = conf.get_int('random_seed', default=None)
    print("The seed is set to:", seed)
    if seed is not None:
        set_seed(seed)


    torch.set_float32_matmul_precision('high')

    # # init wandb or not
    conf['wandb'] = opt.wandb
    
    # # Store weight-method in conf
    # conf['weight_method'] = opt.weight_method

    # # Load checkpoint model for finetuning
    # conf['pretrainedPath'] = opt.pretrainedPath

    # Resume: CLI --resume overrides; else the conf file's `resume` key; else off.
    # With resume on and no checkpoint yet, training starts fresh (train.py falls
    # through), so `resume = true` in a conf is safe for preemptible-queue runs.
    if opt.resume is not None:
        conf['resume'] = opt.resume
    elif 'resume' not in conf:
        conf['resume'] = False
    conf["resuming_epoch"] = 0

    # if phase is Phases.FINE_TUNE:
    #     optimization_num_of_epochs = conf.get_int("train.optimization_num_of_epochs")
    #     optimization_eval_intervals = conf.get_int('train.optimization_eval_intervals')
    #     optimization_lr = conf.get_float('train.optimization_lr')
    #     test_scans_list = conf.get_list('dataset.test_set')

    #     conf['dataset']['scans_list'] = test_scans_list
    #     conf['train']['num_of_epochs'] = optimization_num_of_epochs
    #     conf['train']['eval_intervals'] = optimization_eval_intervals
    #     conf['train']['lr'] = optimization_lr
    
    
    return conf, device, phase