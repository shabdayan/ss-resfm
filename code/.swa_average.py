"""Design 3: stochastic weight averaging over archived checkpoints.
Averages the model weights of the last k archived epochs of a run and writes
them as models/Model_EpSWA.pt, so the standard eval path can load it."""
import torch, glob, os, re, sys
run = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
paths = sorted(glob.glob(f"{run}/models_all/Model_Ep*.pt"),
               key=lambda p: int(re.search(r"Ep(\d+)", p).group(1)))
paths = paths[-k:]
print(f"averaging {len(paths)} checkpoints: {[os.path.basename(p) for p in paths]}", flush=True)
acc, ref = None, None
for p in paths:
    ck = torch.load(p, map_location="cpu")
    sd = ck["model_state_dict"]
    if acc is None:
        acc = {kk: v.clone().float() for kk, v in sd.items()}; ref = ck
    else:
        for kk in acc: acc[kk] += sd[kk].float()
for kk in acc: acc[kk] /= len(paths)
ref["model_state_dict"] = acc
out = f"{run}/models/Model_EpSWA.pt"
torch.save(ref, out)
print("WROTE", out, flush=True)
