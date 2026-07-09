# Debug Mode Guide

## Overview

Debug mode runs the entire comparison pipeline with minimal resources for **fast testing**:
- ✅ Only 10 epochs (vs 180,000-300,000)
- ✅ Uses waic-short queue (fast turnaround)
- ✅ Only 1 scene (vs 9 scenes)
- ✅ **Completes in minutes, not weeks!**

**Use for:** Testing pipeline, checking for errors, verifying setup  
**NOT for:** Real results or paper data

---

## Quick Start

### Debug Mode (8 jobs, ~10 minutes):
```bash
./run_comparison_esfm_vs_outliers.sh --debug
```

**What it does:**
- Runs 8 jobs total (2 methods × 4 depths × 1 scene)
- 10 epochs each (~1-2 minutes per job)
- waic-short queue
- Scene 0007 only

**Timeline:** ~10-15 minutes total

---

## Debug Mode vs Other Modes

| Mode | Jobs | Epochs/Job | Queue | Scenes | Time | Purpose |
|------|------|------------|-------|--------|------|---------|
| **Debug** | **8** | **10** | **waic-short** | **1** | **~15 min** | **Pipeline test** |
| Quick | 16 | 100k-180k | waic-risk | 2 | 3-5 days | Fast test |
| Full | 72 | 180k-300k | waic-risk | 9 | 2-6 weeks | Production |

---

## What Debug Mode Tests

✅ **Script execution** - All loops and logic work  
✅ **Job submission** - bsub commands correct  
✅ **Queue access** - waic-short available  
✅ **File creation** - Results files generated  
✅ **Analysis pipeline** - CSV generation works  

❌ **NOT tested:** Actual training convergence  
❌ **NOT tested:** Real performance metrics  
❌ **NOT valid:** Results for paper/thesis  

---

## Step-by-Step Debug Workflow

### 1. Run Debug Mode
```bash
./run_comparison_esfm_vs_outliers.sh --debug
```

**Output:**
```
DEBUG MODE: 10 epochs only, waic-short queue, 1 scene
  This is for testing the pipeline, not real results!
==========================================
COMPARISON CONFIGURATION
==========================================
Methods: Baseline ESFM, ESFM+OutlierLoss
Depths: 10L, 12L, 15L, 18L
Scenes: 1 (0007)
GPU Strategy: manual
Queue: waic-short

Total experiments: 8
  = 2 methods × 4 depths × 1 scene
==========================================

Submitting jobs...
```

### 2. Monitor Jobs
```bash
bjobs -q waic-short
```

**Expected:**
```
JOBID   USER    STAT  QUEUE       JOB_NAME
1234567 user    RUN   waic-short  ESFM_0007_10L
1234568 user    RUN   waic-short  ESFM_0007_12L
...
```

### 3. Check Completion (5-15 minutes)
```bash
# Count completed
ls results/comparison/experiment_*/Results_*.xlsx | wc -l

# Should show: 8 (when all done)
```

### 4. Verify Results Created
```bash
# List all result files
ls -lh results/comparison/experiment_*/Results_*.xlsx

# Check tracking file
cat results/comparison/comparison_tracking_*.csv
```

### 5. Test Analysis Script
```bash
python3 results/comparison/analyze_comparison_*.py results/comparison
```

**Should generate:**
- `comparison_summary.csv`
- `detailed_results_with_means.csv`

---

## Debug Mode Variations

### Ultra-Minimal (2 jobs, one depth):
```bash
./run_comparison_esfm_vs_outliers.sh --debug --conservative --scans 0007
```
**Runs:** 2 methods × 1 depth (10L) × 1 scene = **2 jobs**  
**Time:** ~5 minutes

### Debug + Multiple Scenes (32 jobs):
```bash
./run_comparison_esfm_vs_outliers.sh --debug --scans 0007,0015,0024,0063
```
**Runs:** 2 methods × 4 depths × 4 scenes = **32 jobs**  
**Time:** ~30-45 minutes

---

## What Each Job Does in Debug Mode

```
Single debug job execution:

1. Load data (scene 0007)           ~5 seconds
2. Initialize network               ~2 seconds
3. Train for 10 epochs              ~30-60 seconds
4. Evaluate                         ~5 seconds
5. Save results                     ~2 seconds

Total: ~1-2 minutes per job
```

**With 8 jobs running in parallel on waic-short:**
- All complete in ~10-15 minutes

---

## Checking Debug Results

### Results ARE Created:
```bash
$ ls results/comparison/experiment_*/Results_*.xlsx
results/comparison/experiment_Baseline_ESFM_10L_0007/Results_0007.xlsx
results/comparison/experiment_Baseline_ESFM_12L_0007/Results_0007.xlsx
results/comparison/experiment_Baseline_ESFM_15L_0007/Results_0007.xlsx
results/comparison/experiment_Baseline_ESFM_18L_0007/Results_0007.xlsx
results/comparison/experiment_ESFM_OutlierLoss_10L_0007/Results_0007.xlsx
results/comparison/experiment_ESFM_OutlierLoss_12L_0007/Results_0007.xlsx
results/comparison/experiment_ESFM_OutlierLoss_15L_0007/Results_0007.xlsx
results/comparison/experiment_ESFM_OutlierLoss_18L_0007/Results_0007.xlsx
```

### Results NOT Meaningful:
```
⚠️ Warning: 10 epochs is NOT enough for convergence!

Typical convergence: 50,000-150,000 epochs
Debug mode: 10 epochs

These results are for pipeline testing only!
```

---

## Debug Mode Checklist

Use debug mode to verify:

- [ ] Script runs without errors
- [ ] Jobs submit to waic-short
- [ ] All 8 jobs complete
- [ ] Results files created (.xlsx)
- [ ] Tracking CSV populated
- [ ] Analysis script runs
- [ ] Summary CSVs generated
- [ ] No path/permission errors
- [ ] Module loading works
- [ ] Python environment correct

Once all ✅, proceed with real run:
```bash
./run_comparison_esfm_vs_outliers.sh
```

---

## Troubleshooting in Debug Mode

### Issue: Jobs fail immediately

**Check:**
```bash
bjobs -l JOBID
cat lsf_output.JOBID
```

**Common causes:**
- waic-short queue not available → Use `--queue waic-risk`
- Module not loaded → Check CUDA module
- Path issues → Run from correct directory

### Issue: waic-short doesn't exist

**Try waic-long instead:**
```bash
./run_comparison_esfm_vs_outliers.sh --debug --queue waic-long
```

**Or check available queues:**
```bash
bqueues | grep waic
```

### Issue: Jobs pending forever

**Check queue:**
```bash
bqueues waic-short
```

If busy, use different queue:
```bash
./run_comparison_esfm_vs_outliers.sh --debug --queue waic-risk
```

---

## Debug Mode + Other Flags

### Combine with other options:

**Debug + Conservative (2 jobs):**
```bash
./run_comparison_esfm_vs_outliers.sh --debug --conservative
# Tests only 10-12 layer depths
```

**Debug + Custom Queue:**
```bash
./run_comparison_esfm_vs_outliers.sh --debug --queue waic-long
# Uses waic-long instead of waic-short
```

**Debug + Multiple Scenes:**
```bash
./run_comparison_esfm_vs_outliers.sh --debug --scans 0007,0015
# 2 scenes = 16 jobs
```

---

## Example Debug Session

```bash
# Terminal 1: Run debug
$ ./run_comparison_esfm_vs_outliers.sh --debug
DEBUG MODE: 10 epochs only, waic-short queue, 1 scene
Submitting jobs...
[8 jobs submitted]

# Terminal 2: Monitor
$ watch -n 5 bjobs
Every 5s: bjobs

JOBID   STAT  JOB_NAME
123456  RUN   ESFM_0007_10L
123457  RUN   ESFM_0007_12L
...

# After 10-15 minutes: All DONE
$ bjobs
No unfinished job found

# Check results
$ ls results/comparison/experiment_*/Results_*.xlsx | wc -l
8

# Test analysis
$ python3 results/comparison/analyze_comparison_*.py results/comparison
Loaded 8 results
Comparison summary saved to: results/comparison/comparison_summary.csv

# ✅ Pipeline works! Ready for full run
$ ./run_comparison_esfm_vs_outliers.sh
```

---

## When to Use Each Mode

### Use Debug Mode When:
✅ First time running script  
✅ Testing after changes  
✅ Verifying cluster access  
✅ Checking for errors  
✅ Want results in 15 minutes  

### Use Quick Mode When:
✅ Testing with realistic training  
✅ Want preliminary results  
✅ Need faster than full run  
✅ Have 3-5 days available  

### Use Full Mode When:
✅ Generating thesis/paper results  
✅ Need all 9 scenes  
✅ Want best possible accuracy  
✅ Have 2-6 weeks available  

---

## Key Points

### Debug Mode is:
✅ **Fast** - Completes in 10-15 minutes  
✅ **Cheap** - Uses waic-short (fast queue)  
✅ **Small** - Only 8 jobs  
✅ **Safe** - Tests without long commitment  

### Debug Mode is NOT:
❌ **Convergent** - 10 epochs too few  
❌ **Accurate** - Results meaningless  
❌ **Complete** - Only 1 scene  
❌ **Publication-ready** - For testing only  

---

## Summary Commands

### Debug mode (8 jobs, 15 min):
```bash
./run_comparison_esfm_vs_outliers.sh --debug
```

### Ultra-minimal (2 jobs, 5 min):
```bash
./run_comparison_esfm_vs_outliers.sh --debug --conservative
```

### Debug with custom queue:
```bash
./run_comparison_esfm_vs_outliers.sh --debug --queue waic-long
```

### After debug succeeds, run full:
```bash
./run_comparison_esfm_vs_outliers.sh
```

---

## Recommended First-Time Workflow

```bash
# Step 1: Debug mode (verify pipeline)
./run_comparison_esfm_vs_outliers.sh --debug
# Wait 15 minutes

# Step 2: If successful, quick mode (verify convergence)
./run_comparison_esfm_vs_outliers.sh --quick --scans 0007,0015
# Wait 3-5 days

# Step 3: If results look good, full run
./run_comparison_esfm_vs_outliers.sh
# Wait 2-6 weeks
```

**Debug mode gives you confidence before committing to long runs!** 🚀
