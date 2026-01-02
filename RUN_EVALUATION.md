# Baseline 7B Evaluation Guide

## Quick Run (Recommended)

```bash
# On islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull origin main

# Activate environment
conda activate baseline_final

# Install smatch if needed
pip install smatch

# Run evaluation
python evaluate_baseline.py \
    --predictions public_test_result_baseline_7b.txt \
    --sentences data/public_test.txt \
    --gold data/public_test_ground_truth.txt \
    --formatted-output predictions_formatted.txt \
    --results evaluation_results.txt
```

## What This Does

1. **Reads files:**
   - Test sentences: `data/public_test.txt` (150 sentences)
   - Raw predictions: `public_test_result_baseline_7b.txt` (150 AMRs without #::snt)
   - Ground truth: `data/public_test_ground_truth.txt` (150 AMRs with #::snt)

2. **Formats predictions:**
   - Adds `#::snt` prefix to match ground truth format
   - Saves to: `predictions_formatted.txt`

3. **Calculates SMATCH:**
   - Uses formula: F1 = 2×P×R/(P+R)
   - Where: P=M/T, R=M/G
   - M = matching triples, T = prediction triples, G = gold triples
   - Saves results to: `evaluation_results.txt`

## Expected Output

```
======================================================================
BASELINE MODEL EVALUATION
======================================================================
Predictions: public_test_result_baseline_7b.txt
Sentences: data/public_test.txt
Gold: data/public_test_ground_truth.txt

[1/3] Reading files...
  ✓ 150 sentences
  ✓ 150 predictions

[2/3] Formatting predictions...
  Sentences: 150
  Predictions: 150
  ✓ Formatted predictions saved to: predictions_formatted.txt

[3/3] Calculating SMATCH score...
======================================================================
CALCULATING SMATCH SCORE
======================================================================

Reading files...
  Predictions: 150 AMRs
  Gold: 150 AMRs

  Processing AMR pairs...
    Processed 10/150...
    Processed 20/150...
    ...

======================================================================
RESULTS
======================================================================
Precision: X.XXXX (XX.XX%)
Recall:    X.XXXX (XX.XX%)
F1 Score:  X.XXXX (XX.XX%)
======================================================================

✓ Results saved to: evaluation_results.txt
✓ Formatted predictions: predictions_formatted.txt

Evaluation complete!
```

## Alternative: Direct SMATCH Calculation

If you already have formatted predictions:

```bash
python calculate_smatch.py \
    --predictions predictions_formatted.txt \
    --gold data/public_test_ground_truth.txt \
    --output smatch_results.txt
```

## Files Generated

- `predictions_formatted.txt`: Predictions with #::snt format
- `evaluation_results.txt`: SMATCH scores summary
- `smatch_results.txt`: (optional) Direct SMATCH calculation results

## Next Steps After Evaluation

1. **Check results:**
   ```bash
   cat evaluation_results.txt
   ```

2. **Compare with MTUP:**
   - MTUP 7B F1: [your MTUP score]
   - Baseline 7B F1: [will be calculated]

3. **Upload to HuggingFace:**
   ```bash
   chmod +x UPLOAD_TO_HF.sh
   ./UPLOAD_TO_HF.sh
   ```

## Troubleshooting

### If smatch not installed:
```bash
pip install smatch
```

### If evaluation fails:
```bash
# Check file exists
ls -lh public_test_result_baseline_7b.txt
wc -l public_test_result_baseline_7b.txt  # Should be ~1600 lines (150 AMRs × ~10 lines each)

# Check first few lines
head -30 public_test_result_baseline_7b.txt
```

### If you need to regenerate predictions:
```bash
python predict_baseline.py \
    --checkpoint outputs/baseline_20260102_125130/checkpoint-1545 \
    --test-file data/public_test.txt \
    --output public_test_result_baseline_7b.txt \
    --ground-truth data/public_test_ground_truth.txt
```
