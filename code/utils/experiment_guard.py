"""
Ground rules for paper experiments (SPEC_cvpr_experiments.md):
- the 27 multi-scene training scenes must never be evaluated as test scenes;
- every result must be traceable to one run (git hash, conf, seed, provenance).
"""
import os
import subprocess

# RESfM's 27 MegaDepth training scenes (RESFM_Learning.conf; " new" suffix stripped)
TRAIN_SCENES_27 = [
    "0156", "0860", "0412", "0217", "0181", "0285", "0214", "0041", "0275",
    "0186", "0474", "0476", "0768", "0204", "0733", "0047", "0229", "0175",
    "0349", "0107", "0303", "0058", "0286", "0062", "0478", "0271", "0360",
]

VALIDATION_SCENES = ["5015", "0176", "0299", "0290"]

# RESfM's MegaDepth test split (resfm.pdf Table 1; Group 1 + Group 2), as listed
# in confs/uesfm_eval.conf.template. Canonical list for provenance checks.
TEST_SCENES_MEGADEPTH = [
    "0238", "0060", "0197", "0094", "0265", "0083", "0076", "0185", "0048",
    "0024", "0223", "5016", "0046", "0099", "1001", "0231", "0411", "0377",
    "0102", "0147", "0148", "0446", "0022", "0327", "0015", "0455", "0496",
    "1589", "0012", "0104", "0019", "0063", "0130", "0080", "0240", "0007",
]


class LeakageError(RuntimeError):
    """Test-set leakage. Must always fail loudly — never catch-and-continue."""


def assert_not_training_scene(scan, context=""):
    """Fail loudly if a training scene reaches an evaluation entry point."""
    if str(scan).strip() in TRAIN_SCENES_27:
        raise LeakageError(
            f"LEAKAGE GUARD: scene '{scan}' is one of the 27 multi-scene TRAINING "
            f"scenes and must never be evaluated as a test scene ({context}). "
            f"If this is intentional research on a training scene, do not route it "
            f"through the evaluation entry points.")


def assert_checkpoint_provenance_clean(checkpoint, path="", context=""):
    """Reject a checkpoint whose provenance shows it was produced by a run on a
    TEST scene (a prior single-scene experiment or a TTT/fine-tune snapshot) —
    such weights must never initialize training or serve as a fine-tune base.
    Checkpoints without a provenance field (the pre-provenance multi-scene
    training chain) pass; provenance is recorded on every save going forward."""
    prov = checkpoint.get("provenance") if isinstance(checkpoint, dict) else None
    if not prov:
        return
    seen = [prov.get("scan")] + list(prov.get("scenes_seen") or [])
    tainted = sorted({str(s).strip() for s in seen if s} & set(TEST_SCENES_MEGADEPTH))
    if tainted:
        raise LeakageError(
            f"LEAKAGE GUARD: checkpoint '{path or '<in-memory>'}' carries provenance "
            f"from TEST scene(s) {tainted} (provenance: {prov}) and must not seed "
            f"training or evaluation ({context}).")


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def provenance(conf, phase=None, base_checkpoint=None, extra=None):
    """Provenance dict stored inside checkpoints and result files (R2.4)."""
    info = {
        "exp_name": conf.get_string("exp_name", default="unknown"),
        "scan": conf.get_string("dataset.scan", default=None),
        "phase": getattr(phase, "name", str(phase)),
        "loss": conf.get_string("loss.func", default=None),
        "loss_tuning": conf.get_string("loss.func_tuning", default=None),
        "seed": conf.get_int("random_seed", default=None),
        "git": git_hash(),
        "base_checkpoint": base_checkpoint
            or conf.get_string("pretrainedPath", default=None),
    }
    if extra:
        info.update(extra)
    return info
