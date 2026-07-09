"""
Multi-scene learning entry point.

Ported from RESfM's multiple_scenes_learning.py (see
../../resfm-main-orig/code/multiple_scenes_learning.py) and adapted to this repo's
API drift:
  - train.train(..., fabric=fabric) instead of fabri=fabric
  - conf key train.num_epochs instead of train.num_of_epochs
  - model construction matches single_scene_optimization.py (phase argument for
    *OutliersNet classes, Kaiming init). torch.compile is intentionally NOT used
    here: scenes have heterogeneous shapes, so 'reduce-overhead' compilation would
    recompile on nearly every batch.

The trailing per-test-scene fine-tune fan-out from the original is kept but gated
behind conf key train.fine_tune_after_training (default False) — it is the vehicle
for the later TTT experiment and must not run by default.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

import cv2  # DO NOT REMOVE
import torch

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from utils import general_utils
from utils.Phases import Phases
from datasets.ScenesDataSet import ScenesDataSet, collate_fn
from datasets import SceneData
from lightning.fabric import Fabric
from single_scene_optimization import init_weights_kaiming, train_single_model
import train
import copy


def main():
    # Init Experiment
    conf, device, phase = general_utils.init_exp(Phases.TRAINING.name)
    general_utils.log_code(conf)  # Log code to the experiment folder

    seed = conf.get_int('random_seed', default=None)
    # Unlike the single-scene fabric (static_graph=True), multi-scene training must
    # tolerate a variable graph: the adaptive unsupervised loss skips its
    # classification term when a scene has too few confident samples, so parameter
    # participation changes between iterations (torch>=2.0 DDP errors on this with
    # static_graph). Upstream RESfM used plain "ddp" here too.
    from lightning.fabric.strategies import DDPStrategy
    fabric = Fabric(accelerator="cuda", devices="auto",
                    strategy=DDPStrategy(find_unused_parameters=True, static_graph=False))
    if seed is not None:
        fabric.seed_everything(seed)
    fabric.launch()

    # Get configuration
    min_sample_size = conf.get_float('dataset.min_sample_size')
    max_sample_size = conf.get_float('dataset.max_sample_size')
    batch_size = conf.get_int('dataset.batch_size')

    if phase is not Phases.FINE_TUNE:
        # Create model (same conf-driven pattern as single_scene_optimization.py)
        model_type = conf.get_string("model.type")
        print(f"Creating model with type from config: {model_type}")
        model_class = general_utils.get_class("models." + model_type)
        if model_type in ["SetOfSet.SetOfSetOutliersNet", "SetOfSet.DeepSetOfSetOutliersNet"]:
            model = model_class(conf, phase).to(device)
        elif model_type in ["SetOfSet.SetOfSetNet", "SetOfSet.DeepSetOfSetNet"]:
            model = model_class(conf).to(device)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        model.apply(init_weights_kaiming)

        print(f'Number of parameters: {sum([x.numel() for x in model.parameters()])}')
        print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        # Create train, test and validation sets
        test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
        validation_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.validation_set'), conf)
        train_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.train_set'), conf)

        train_set = ScenesDataSet(train_scenes, return_all=False, min_sample_size=min_sample_size, max_sample_size=max_sample_size, phase=Phases.TRAINING)
        validation_set = ScenesDataSet(validation_scenes, return_all=True)
        test_set = ScenesDataSet(test_scenes, return_all=True)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

        fabric.barrier()
        train_stat, train_errors, validation_errors, test_errors = train.train(conf, train_loader, model, phase, validation_loader, test_loader, fabric=fabric)
        if fabric.global_rank == 0:
            # Write results
            general_utils.write_results(conf, train_stat, file_name="Train_Stats")
            general_utils.write_results(conf, validation_errors, file_name="Validation")
            general_utils.write_results(conf, test_errors, file_name="Test")

        test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
        test_set = ScenesDataSet(test_scenes, return_all=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
        train_errors, validation_errors, test_errors = train.test(conf, model, phase, train_data=None, validation_data=None, test_data=test_loader, fabric=fabric, run_ba=False)
        general_utils.write_results(conf, test_errors, file_name="myTest")

    # Per-test-scene fine-tune stage (the later TTT experiment runs through here).
    # Off by default: the original launched it unconditionally, but it is a separate
    # phase of the evaluation protocol and this repo's train_single_model requires a
    # results_aggregation_file_name, so it is left to the dedicated TTT task.
    if fabric.global_rank == 0 and conf.get_bool('train.fine_tune_after_training', default=False):
        test_scans_list = conf.get_list('dataset.test_set')

        conf_test = copy.deepcopy(conf)
        conf_test['dataset']['scans_list'] = test_scans_list
        conf_test['train']['num_epochs'] = conf.get_int("train.optimization_num_of_epochs")
        conf_test['train']['eval_intervals'] = conf.get_int('train.optimization_eval_intervals')
        conf_test['train']['lr'] = conf.get_float('train.optimization_lr')

        optimization_all_sets(conf_test, device, Phases.FINE_TUNE)


def optimization_all_sets(conf, device, phase):
    scans_list = conf.get_list('dataset.scans_list')
    for i, scan in enumerate(scans_list):
        conf["dataset"]["scan"] = scan
        train_single_model(conf, device, phase)


if __name__ == "__main__":
    main()
