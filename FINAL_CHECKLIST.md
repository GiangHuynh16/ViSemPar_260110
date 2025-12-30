# Final Checklist - Baseline 7B Training

**Date**: 2025-12-30
**Status**: Ready to deploy on server

---

## ✅ Completed Tasks

### 1. Configuration
- [x] Model updated to 7B (`config/config.py` line 21)
- [x] LoRA rank set to 128 (line 29)
- [x] Training config matched to MTUP (lines 40-57)
- [x] Vietnamese prompt template (lines 117-129)
- [x] Minimal preprocessing config (lines 99-107)
- [x] HuggingFace push disabled by default (line 81)

### 2. Scripts
- [x] Training script: `train_baseline.py`
- [x] Evaluation script: `evaluate_baseline_model.py`
- [x] Launcher script: `START_BASELINE_7B_TRAINING.sh`
- [x] All scripts executable (`chmod +x`)

### 3. Postprocessing
- [x] Minimal postprocessing in `evaluate_baseline_model.py` (lines 22-40)
- [x] Vietnamese markers in extraction logic (lines 110-121)
- [x] No heavy processing (no balancing, no fixing)

### 4. Documentation
- [x] Training guide: `BASELINE_7B_TRAINING_GUIDE.md`
- [x] Improvements doc: `BASELINE_IMPROVEMENTS.md`
- [x] Ready to train: `READY_TO_TRAIN.md`
- [x] Training summary: `TRAINING_SUMMARY.md`
- [x] Environment setup: `ENVIRONMENT_SETUP.md`
- [x] This checklist: `FINAL_CHECKLIST.md`

### 5. Environment
- [x] Conda environment file: `environment.yml`
- [x] Requirements updated: `requirements.txt` (bitsandbytes removed)
- [x] Environment setup guide created

---

## 📋 Pre-Training Checklist (Server)

Run these commands on server before training:

### 1. Code Update
```bash
cd ViSemPar_new1
git pull origin main
```

### 2. Verify Prompt Template
```bash
grep "Bạn là chuyên gia" config/config.py
```
**Expected**: Should see Vietnamese prompt starting with "Bạn là chuyên gia..."

✅ **Pass**: Vietnamese template
❌ **Fail**: English template (need to pull again)

### 3. Verify Postprocessing
```bash
grep -A 5 "Minimal post-processing" evaluate_baseline_model.py
```
**Expected**: Should see simple extraction logic, no balancing

### 4. Verify Model Config
```bash
grep "MODEL_NAME\|LORA_CONFIG\|TRAINING_CONFIG" config/config.py | head -20
```
**Expected**:
- MODEL_NAME: Qwen/Qwen2.5-7B-Instruct
- LoRA rank: 128
- Epochs: 15
- Optimizer: adamw_torch

### 5. Check VRAM
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```
**Expected**: >= 18000 (18GB+)

### 6. Check Data Files
```bash
ls -lh data/train_amr_*.txt
```
**Expected**: train_amr_1.txt and train_amr_2.txt exist

### 7. Check Environment
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('✓ Transformers OK')"
python -c "from peft import LoraConfig; print('✓ PEFT OK')"
```
**Expected**: All imports successful, CUDA available

---

## 🚀 Training Commands

### Start Training
```bash
# Create tmux session
tmux new -s baseline_7b

# Inside tmux:
bash START_BASELINE_7B_TRAINING.sh

# Detach: Ctrl+B, then D
```

### Monitor Training
```bash
# Reattach to tmux
tmux attach -t baseline_7b

# Check logs
tail -f logs/training.log

# GPU usage
watch -n 1 nvidia-smi
```

---

## 📊 Expected Training Details

### Configuration Summary
```
Model: Qwen/Qwen2.5-7B-Instruct
LoRA rank: 128
LoRA alpha: 256
Training epochs: 15
Batch size: 2
Gradient accumulation: 8
Effective batch size: 16
Learning rate: 2e-4
Optimizer: adamw_torch
Scheduler: cosine
```

### Resource Usage
```
Training time: ~12-15 hours
Peak VRAM: ~20-22 GB
Training examples: ~1842
Total steps: ~1545
Checkpoint size: ~1-2 GB
```

### Output Locations
```
Checkpoints: outputs/checkpoints/baseline_7b_final/
Logs: logs/training.log
TensorBoard: outputs/logs/baseline_*/
```

---

## 🔍 Post-Training Checklist

### 1. Verify Training Completed
```bash
# Check if final checkpoint exists
ls -lh outputs/checkpoints/baseline_7b_final/

# Expected files:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer_config.json
# - special_tokens_map.json
```

### 2. Run Evaluation
```bash
python evaluate_baseline_model.py \
  --checkpoint outputs/checkpoints/baseline_7b_final \
  --test-file data/public_test_ground_truth.txt \
  --output results/baseline_7b_evaluation.json \
  --base-model Qwen/Qwen2.5-7B-Instruct
```

### 3. Check Results
```bash
cat results/baseline_7b_evaluation.json
```
**Expected format**:
```json
{
  "precision": 0.XX,
  "recall": 0.XX,
  "f1": 0.XX,
  "valid": XXX,
  "total": 150,
  "errors": X
}
```

### 4. Compare with MTUP
```bash
echo "=== Baseline 7B ==="
cat results/baseline_7b_evaluation.json

echo ""
echo "=== MTUP 7B ==="
cat results/mtup_7b_evaluation.json
```

---

## 📈 Expected Results

### Performance Targets

| Model | F1 Target | Rationale |
|-------|-----------|-----------|
| MTUP 7B | 0.51-0.52 | 2-stage decomposition |
| Baseline 7B | 0.47-0.50 | 1-stage direct mapping |

### Success Criteria

**Baseline training is successful if**:
- ✅ Training completes without errors
- ✅ Final checkpoint exists and is valid
- ✅ F1 score >= 0.45
- ✅ Success rate >= 85% (128/150)
- ✅ No major format errors

**Baseline is competitive if**:
- ✅ F1 score >= 0.48
- ✅ Gap with MTUP < 0.05
- ✅ Success rate >= 90%

---

## 🎯 Next Steps After Evaluation

### If F1 >= 0.48 (Good Results)

1. **Push to HuggingFace**:
```bash
huggingface-cli whoami
hf upload YOUR-USERNAME/vietnamese-amr-baseline-7b \
  outputs/checkpoints/baseline_7b_final
```

2. **Update Thesis**:
- Add baseline 7B results to Table 4.2
- Add comparison section (MTUP vs Baseline)
- Discuss multi-task decomposition benefits

3. **Write Conclusion**:
- MTUP improvement over baseline
- Statistical significance
- Limitations and future work

### If F1 < 0.48 (Needs Investigation)

1. **Error Analysis**:
```bash
# Check error patterns
grep "Error" logs/training.log | tail -20

# Analyze failed examples
python analyze_errors.py \
  --predictions results/baseline_7b_evaluation.json \
  --test-file data/public_test_ground_truth.txt
```

2. **Compare with MTUP Errors**:
- Are baseline errors different from MTUP?
- Does MTUP fix specific error types?
- Where does baseline struggle?

3. **Potential Improvements**:
- Try different prompt variations
- Add few-shot examples
- Adjust temperature/sampling
- Increase training epochs

---

## 📝 Git Commands

### Commit and Push Changes
```bash
# Add files
git add config/config.py
git add evaluate_baseline_model.py
git add train_baseline.py
git add START_BASELINE_7B_TRAINING.sh
git add environment.yml
git add requirements.txt
git add *.md

# Commit
git commit -m "Add baseline 7B training with Vietnamese prompt template

- Update config to 7B model with LoRA 128
- Add Vietnamese prompt template with explicit AMR rules
- Implement minimal postprocessing (trust LLM output)
- Remove bitsandbytes dependency
- Add comprehensive documentation and setup guides
"

# Push
git push origin main
```

---

## 🔗 File References

### Configuration
- Main config: [config/config.py](config/config.py)
- MTUP config: [config/config_mtup.py](config/config_mtup.py)

### Scripts
- Baseline training: [train_baseline.py](train_baseline.py)
- Baseline evaluation: [evaluate_baseline_model.py](evaluate_baseline_model.py)
- MTUP training: [train_mtup.py](train_mtup.py)
- MTUP evaluation: [evaluate_mtup_model.py](evaluate_mtup_model.py)

### Launchers
- Baseline: [START_BASELINE_7B_TRAINING.sh](START_BASELINE_7B_TRAINING.sh)
- MTUP: [START_MTUP_7B_TRAINING.sh](START_MTUP_7B_TRAINING.sh)

### Documentation
- Training guide: [BASELINE_7B_TRAINING_GUIDE.md](BASELINE_7B_TRAINING_GUIDE.md)
- Improvements: [BASELINE_IMPROVEMENTS.md](BASELINE_IMPROVEMENTS.md)
- Ready to train: [READY_TO_TRAIN.md](READY_TO_TRAIN.md)
- Environment setup: [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- Training summary: [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)

---

## ✅ Final Status

**All tasks completed!** Ready to:

1. ✅ Commit and push changes
2. ✅ Pull on server
3. ✅ Start training
4. ✅ Monitor and evaluate
5. ✅ Compare with MTUP
6. ✅ Update thesis

**Command to start on server**:
```bash
cd ViSemPar_new1
git pull origin main
tmux new -s baseline_7b
bash START_BASELINE_7B_TRAINING.sh
```

**Good luck with training! 🚀**
