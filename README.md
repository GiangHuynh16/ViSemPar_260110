# Vietnamese AMR Parser - MTUP Approach

Vietnamese Abstract Meaning Representation (AMR) Parser using Multi-Task Unified Prompt (MTUP) with Qwen 2.5 3B + LoRA.

## 🎯 Quick Start

### Run Full Evaluation

```bash
cd ~/ViSemPar_new1
git pull origin main
bash RUN_FULL_EVALUATION_TMUX.sh
```

### Check Status

```bash
bash CHECK_EVALUATION_STATUS.sh
```

## 📊 Current Results

**Quick Test (10 samples)**:
- F1 Score: **0.4933** (~49%)
- Precision: 0.4978
- Recall: 0.5002
- Success Rate: 7/10 (70%)

## 📁 Project Structure

```
ViSemPar_new1/
├── config/
│   └── prompt_templates.py          # MTUP Vietnamese templates
├── src/
│   ├── data_loader.py               # AMR data loading
│   ├── preprocessor_mtup.py         # MTUP preprocessing
│   └── train_mtup.py                # Training script
├── data/
│   ├── train_amr_*.txt              # Training data
│   └── public_test_ground_truth.txt # Test data
├── outputs/
│   ├── checkpoints_mtup/            # Model checkpoints
│   └── evaluation_results_*.json   # Evaluation results
│
├── evaluate_mtup_model.py           # Evaluation script
├── RUN_FULL_EVALUATION_TMUX.sh      # Run evaluation in tmux
├── CHECK_EVALUATION_STATUS.sh       # Monitor progress
│
└── Documentation:
    ├── EVALUATION_SUMMARY.md        # Complete summary
    ├── EVALUATION_QUICK_REFERENCE.md # Quick commands
    ├── HOW_TO_RUN_FULL_EVALUATION.md # Detailed guide
    └── EVALUATION_FIX.md            # Technical analysis
```

## 🚀 Features

- **MTUP Approach**: Two-task learning (structure → variables)
- **LoRA Training**: Efficient fine-tuning (7.08M trainable params)
- **Vietnamese Prompts**: Native language instruction templates
- **Full Pipeline**: Data loading → Training → Evaluation
- **Tmux Support**: Run long evaluations safely

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md) | Overall status and results |
| [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md) | Quick commands |
| [HOW_TO_RUN_FULL_EVALUATION.md](HOW_TO_RUN_FULL_EVALUATION.md) | Step-by-step guide |
| [EVALUATION_FIX.md](EVALUATION_FIX.md) | Root cause analysis |

## 🎓 MTUP Format

### Training Template (v2_natural)

```
### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
{amr_without_variables}

## Bước 2 - Gán biến cho các khái niệm:
AMR hoàn chỉnh:
{amr_with_variables}
```

### Example

**Input**: "Tôi ăn cơm"

**Step 1** (structure): `(ăn :agent (tôi) :patient (cơm))`

**Step 2** (with vars): `(a / ăn :agent (t / tôi) :patient (c / cơm))`

## 🔧 Model Details

- **Base Model**: Qwen/Qwen2.5-3B-Instruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Trainable Params**: 7,077,888 (0.25% of total)
- **Template**: v2_natural (Vietnamese)
- **Checkpoint**: `outputs/checkpoints_mtup/mtup_full_training_final`

## 📈 Performance

### Expected F1 Scores

| Range | Assessment |
|-------|------------|
| > 0.60 | Excellent |
| 0.50-0.60 | Good |
| 0.40-0.50 | Acceptable |
| < 0.40 | Needs improvement |

**Current**: 0.49 (Acceptable, based on 10-sample test)

### Comparison

- English SOTA: 0.80-0.85
- Vietnamese (limited data): 0.40-0.60 expected
- **Our model**: 0.49 ✅

## 🛠️ Usage

### Training

```bash
bash RUN_FULL_TRAINING.sh
```

### Evaluation

```bash
# Quick test (10 samples)
python3 evaluate_mtup_model.py \
  --checkpoint outputs/checkpoints_mtup/mtup_full_training_final \
  --test-file data/public_test_ground_truth.txt \
  --max-samples 10

# Full evaluation (all samples, in tmux)
bash RUN_FULL_EVALUATION_TMUX.sh
```

### Monitoring

```bash
# Check evaluation status
bash CHECK_EVALUATION_STATUS.sh

# Attach to tmux session
tmux attach -t mtup_eval

# View log
tail -f outputs/evaluation_full_*.log
```

## 🔍 Known Issues

1. **Duplicate node names** (2/10 samples)
   - Model sometimes reuses variable names
   - Can be fixed with post-processing

2. **Unmatched parentheses** (1/10 samples)
   - Occasional generation cutoff
   - Rare occurrence

## 💡 Future Improvements

1. **Post-processing**: Fix duplicate nodes automatically
2. **More training**: Increase epochs for better F1
3. **Better templates**: Try v5_cot (Chain-of-Thought)
4. **Hyperparameter tuning**: Optimize learning rate, batch size

## 📝 Citation

If you use this code, please cite:

```
Vietnamese AMR Parser with MTUP
Model: Qwen 2.5 3B + LoRA
Training: Multi-Task Unified Prompt approach
```

## 📄 License

See LICENSE file.

## 🤝 Contributing

Pull requests welcome! Please:
1. Test your changes
2. Update documentation
3. Follow existing code style

## 📧 Contact

For questions or issues, open a GitHub issue.

---

**Last Updated**: 2025-12-25
**Status**: ✅ Ready for full evaluation
**F1 Score**: 0.49 (quick test)
