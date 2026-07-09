# Debug Mode - Quick Reference

## What is Debug Mode?

**Super-fast pipeline testing:** 10 epochs, waic-short queue, 1 scene  
**Purpose:** Verify setup before committing to long runs  
**Time:** ~10-15 minutes  
**NOT for real results!**

---

## Run Debug Mode

```bash
./run_comparison_esfm_vs_outliers.sh --debug
```

**Runs:**
- 8 jobs (2 methods × 4 depths × 1 scene)
- 10 epochs per job (~1-2 minutes each)
- waic-short queue
- Scene 0007 only

**Timeline:** 10-15 minutes total

---

## What Debug Mode Tests

✅ Script runs without errors  
✅ Jobs submit correctly  
✅ Queue access works  
✅ Results files created  
✅ Analysis pipeline works  

❌ NOT real training  
❌ NOT convergent results  
❌ NOT for papers/thesis  

---

## Monitor Debug Jobs

```bash
# Check jobs
bjobs -q waic-short

# Count completed
ls results/comparison/experiment_*/Results_*.xlsx | wc -l
# Should show: 8 when done
```

---

## Debug Mode Variations

### Ultra-minimal (2 jobs, 5 min):
```bash
./run_comparison_esfm_vs_outliers.sh --debug --conservative
```

### Debug with different queue:
```bash
./run_comparison_esfm_vs_outliers.sh --debug --queue waic-long
```

### Debug with more scenes (32 jobs, 30 min):
```bash
./run_comparison_esfm_vs_outliers.sh --debug --scans 0007,0015,0024,0063
```

---

## Mode Comparison

| Mode | Jobs | Time | Purpose |
|------|------|------|---------|
| **Debug** | **8** | **15 min** | **Test pipeline** |
| Quick | 16 | 3-5 days | Preliminary results |
| Full | 72 | 2-6 weeks | Production |

---

## Recommended Workflow

```bash
# 1. Debug (verify setup)
./run_comparison_esfm_vs_outliers.sh --debug
# Wait 15 minutes

# 2. Quick (verify training)
./run_comparison_esfm_vs_outliers.sh --quick --scans 0007,0015
# Wait 3-5 days

# 3. Full (real results)
./run_comparison_esfm_vs_outliers.sh
# Wait 2-6 weeks
```

---

## Troubleshooting

**Jobs fail?**
```bash
bjobs -l JOBID
cat lsf_output.JOBID
```

**waic-short doesn't exist?**
```bash
./run_comparison_esfm_vs_outliers.sh --debug --queue waic-risk
```

**Check available queues:**
```bash
bqueues | grep waic
```

---

## After Debug Succeeds

✅ Pipeline verified  
✅ No errors  
✅ Ready for real run  

```bash
./run_comparison_esfm_vs_outliers.sh
```

**Debug mode = confidence before long runs!** 🚀
