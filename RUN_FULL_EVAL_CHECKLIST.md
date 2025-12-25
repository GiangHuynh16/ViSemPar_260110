# ✅ Full Evaluation Checklist

## Pre-Flight Check

### On Server

- [ ] SSH into server
  ```bash
  ssh your_server
  ```

- [ ] Navigate to project
  ```bash
  cd ~/ViSemPar_new1
  ```

- [ ] Pull latest changes
  ```bash
  git pull origin main
  ```

- [ ] Check checkpoint exists
  ```bash
  ls -lh outputs/checkpoints_mtup/mtup_*_final
  ```
  Expected: `mtup_full_training_final` (~457 MB)

- [ ] Check test file exists
  ```bash
  wc -l data/public_test_ground_truth.txt
  ```
  Should show line count

- [ ] Check GPU available
  ```bash
  nvidia-smi
  ```
  Should show available GPU

- [ ] Activate conda environment
  ```bash
  conda activate lora_py310
  ```

---

## Run Evaluation

### Option A: Tmux (Recommended)

- [ ] Start evaluation in tmux
  ```bash
  bash RUN_FULL_EVALUATION_TMUX.sh
  ```

- [ ] Wait for confirmation
  ```
  ✅ Evaluation started in tmux session: mtup_eval
  ```

- [ ] Detach or close SSH
  - Can safely disconnect, evaluation continues

### Option B: Direct (If staying connected)

- [ ] Run directly
  ```bash
  bash RUN_FULL_EVALUATION.sh
  ```

- [ ] Keep SSH connected
  - Do NOT close terminal!

---

## Monitor Progress

### Quick Status Check

- [ ] Run status script
  ```bash
  bash CHECK_EVALUATION_STATUS.sh
  ```

- [ ] Review output:
  - [ ] Process running?
  - [ ] Latest log shown?
  - [ ] Progress indicators?

### Attach to Tmux

- [ ] Attach to live session
  ```bash
  tmux attach -t mtup_eval
  ```

- [ ] Watch progress bars
  - `Generating: XX%`
  - `Evaluating: XX%`

- [ ] Detach when done
  - Press `Ctrl+B`, then `D`

### Watch Log

- [ ] Find latest log
  ```bash
  ls -t outputs/evaluation_full_*.log | head -1
  ```

- [ ] Tail log in real-time
  ```bash
  tail -f outputs/evaluation_full_*.log
  ```

- [ ] Stop watching
  - Press `Ctrl+C`

---

## Wait for Completion

### Time Estimate

- [ ] Check sample count
  ```bash
  grep -c "^# ::snt" data/public_test_ground_truth.txt
  ```

- [ ] Calculate time
  ```
  samples × 20 seconds ÷ 60 = minutes

  Examples:
  - 100 samples: ~33 minutes
  - 200 samples: ~67 minutes
  - 500 samples: ~2.8 hours
  ```

### What to Do While Waiting

- [ ] Get coffee ☕
- [ ] Read documentation
- [ ] Check status periodically (every 15-30 min)

---

## Results Check

### Completion Signal

- [ ] Check for completion message
  ```bash
  tail -20 outputs/evaluation_full_*.log
  ```

- [ ] Look for:
  ```
  ✅ EVALUATION COMPLETE
  SMATCH SCORES
    Precision: 0.XXXX
    Recall:    0.XXXX
    F1:        0.XXXX
  ```

### View Results

- [ ] Find results file
  ```bash
  ls -t outputs/evaluation_results_full_*.json | head -1
  ```

- [ ] View formatted
  ```bash
  cat outputs/evaluation_results_full_*.json | python3 -m json.tool
  ```

- [ ] Note the scores:
  - Precision: ______
  - Recall: ______
  - F1: ______
  - Valid samples: ____/____

---

## Interpret Results

### F1 Score Assessment

- [ ] Check F1 range:
  - [ ] **> 0.60**: 🟢 Excellent! Ready for deployment
  - [ ] **0.50-0.60**: 🟡 Good, minor improvements possible
  - [ ] **0.40-0.50**: 🟠 Acceptable, consider enhancements
  - [ ] **< 0.40**: 🔴 Needs retraining

### Compare with Quick Test

Quick test (10 samples): F1 = 0.4933

- [ ] Full test F1: ______
- [ ] Difference: ______ (expect ±0.02)

### Error Analysis

- [ ] Total samples: ______
- [ ] Valid: ______
- [ ] Errors: ______
- [ ] Success rate: ______%

---

## Share Results

### Save Results

- [ ] Copy results file
  ```bash
  cp outputs/evaluation_results_full_*.json \
     outputs/evaluation_results_FINAL.json
  ```

- [ ] Copy log file
  ```bash
  cp outputs/evaluation_full_*.log \
     outputs/evaluation_FINAL.log
  ```

### Document Findings

- [ ] Create summary document
  ```
  F1 Score: ______
  Precision: ______
  Recall: ______
  Success Rate: ______%
  Errors: ______

  Assessment: [Excellent/Good/Acceptable/Needs work]

  Next steps:
  - [ ] ...
  - [ ] ...
  ```

---

## Cleanup (Optional)

- [ ] Remove temporary logs
  ```bash
  ls outputs/evaluation_full_*.log
  # Keep only latest
  ```

- [ ] Kill tmux session
  ```bash
  tmux kill-session -t mtup_eval
  ```

---

## Next Steps

### If F1 > 0.55 (Good/Excellent)

- [ ] Celebrate! 🎉
- [ ] Document success
- [ ] Consider deployment
- [ ] Try test on new data

### If F1 = 0.45-0.55 (Acceptable)

- [ ] Implement duplicate node fix
- [ ] Analyze error patterns
- [ ] Consider training 1-2 more epochs
- [ ] Try better template (v5_cot)

### If F1 < 0.45 (Needs Work)

- [ ] Review training logs
- [ ] Check data quality
- [ ] Increase training epochs (3-5)
- [ ] Consider larger model (7B)
- [ ] Try different hyperparameters

---

## Troubleshooting

### Evaluation Stuck?

- [ ] Check process still running
  ```bash
  ps aux | grep evaluate_mtup_model.py
  ```

- [ ] Check GPU usage
  ```bash
  nvidia-smi
  ```

- [ ] Check last log update
  ```bash
  ls -lth outputs/evaluation_full_*.log | head -1
  ```

- [ ] If stuck > 30 min with no progress:
  - [ ] Kill and restart
    ```bash
    tmux kill-session -t mtup_eval
    bash RUN_FULL_EVALUATION_TMUX.sh
    ```

### No Results File?

- [ ] Check if evaluation completed
  ```bash
  grep "EVALUATION COMPLETE" outputs/evaluation_full_*.log
  ```

- [ ] Check for errors in log
  ```bash
  grep -i "error" outputs/evaluation_full_*.log
  ```

### Unexpected F1?

- [ ] Verify using correct checkpoint
- [ ] Check test file is correct
- [ ] Review log for parsing errors
- [ ] Compare with quick test (should be similar)

---

## Final Checklist

- [ ] Evaluation completed successfully
- [ ] Results saved and documented
- [ ] F1 score recorded: ______
- [ ] Next steps identified
- [ ] Tmux session cleaned up (if desired)
- [ ] Results shared/documented

---

## Quick Reference

```bash
# Start
bash RUN_FULL_EVALUATION_TMUX.sh

# Check
bash CHECK_EVALUATION_STATUS.sh

# Monitor
tmux attach -t mtup_eval
tail -f outputs/evaluation_full_*.log

# View results
cat outputs/evaluation_results_full_*.json | python3 -m json.tool
```

---

## Expected Output Format

```json
{
  "precision": 0.XXXX,
  "recall": 0.XXXX,
  "f1": 0.XXXX,
  "valid": XXX,
  "total": XXX,
  "errors": XX
}
```

---

## Time Tracking

- Start time: __:__ (date: ____)
- Expected completion: __:__
- Actual completion: __:__
- Duration: ____ hours

---

_Use this checklist to ensure smooth full evaluation_
_Check off items as you complete them_
_Good luck! 🚀_
