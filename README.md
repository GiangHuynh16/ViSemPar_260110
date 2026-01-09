# Vietnamese AMR Parser - MTUP v2

Multi-Task Unified Prompting approach for Vietnamese Abstract Meaning Representation (AMR) parsing.

## 🎯 Goal

Beat baseline F1 score (0.47) using a unified multi-task learning approach.

## 📚 Quick Links

- **START HERE:** [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md) - Quick start guide
- **FULL GUIDE:** [mtup_v2/docs/TRAINING_GUIDE.md](mtup_v2/docs/TRAINING_GUIDE.md) - Complete training guide
- **CONCEPT:** [mtup_v2/docs/MTUP_CONCEPT.md](mtup_v2/docs/MTUP_CONCEPT.md) - Understanding MTUP
- **EXAMPLES:** [mtup_v2/docs/COREFERENCE_EXAMPLES.md](mtup_v2/docs/COREFERENCE_EXAMPLES.md) - Co-reference handling
- **SUMMARY:** [MTUP_V2_SUMMARY.md](MTUP_V2_SUMMARY.md) - Implementation summary
- **SERVER SETUP:** [COPY_TO_SERVER.md](COPY_TO_SERVER.md) - Copy files to server

## 🚀 Quick Start

### 1. Preprocessing (Local)
```bash
python3 mtup_v2/preprocessing/create_mtup_data.py
```

### 2. Training (Server)
```bash
python3 mtup_v2/scripts/train_mtup_unified.py \
    --data_path data/train_mtup_unified.txt \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/mtup_v2 \
    --epochs 5
```

### 3. Prediction (Server)
```bash
python3 mtup_v2/scripts/predict_mtup_unified.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path outputs/mtup_v2/final_adapter \
    --input_file data/public_test.txt \
    --output_file outputs/predictions.txt
```

### 4. Evaluation
```bash
python3 mtup_v2/scripts/evaluate.py \
    --predictions outputs/predictions.txt \
    --ground_truth data/public_test_ground_truth.txt
```

## 📂 Project Structure

```
ViSemPar_new1/
├── mtup_v2/                          # NEW: MTUP v2 implementation
│   ├── scripts/
│   │   ├── train_mtup_unified.py     # Training script (369 lines)
│   │   ├── predict_mtup_unified.py   # Prediction script (340 lines)
│   │   └── evaluate.py               # Evaluation script (314 lines)
│   ├── preprocessing/
│   │   └── create_mtup_data.py       # Data preprocessing (241 lines)
│   └── docs/
│       ├── README.md                 # Overview
│       ├── MTUP_CONCEPT.md           # Concept explanation
│       ├── TRAINING_GUIDE.md         # Detailed guide
│       └── COREFERENCE_EXAMPLES.md   # Co-reference examples
│
├── archive/
│   └── mtup_v1/                      # OLD: Archived previous attempts
│       ├── train_mtup*.py            # Old training scripts
│       └── *.md                      # Old documentation (92 files)
│
├── data/
│   ├── train_mtup_unified.txt        # ✅ Generated: 1,262 samples
│   ├── public_test.txt               # Test input
│   └── public_test_ground_truth.txt  # Ground truth
│
├── outputs/                          # Training outputs (created during training)
│
├── MTUP_V2_QUICKSTART.md            # ⭐ START HERE
├── MTUP_V2_SUMMARY.md               # Implementation summary
├── COPY_TO_SERVER.md                # Server setup guide
└── README.md                        # This file
```

## 🔑 Key Concepts

### What is MTUP?

**MTUP** = **Multi-Task Unified Prompting**

- **1 MODEL** (not 2!)
- **1 PROMPT** (unified for both tasks)
- **2 TASKS** learned simultaneously:
  - Task 1: Vietnamese → AMR Skeleton (no variables)
  - Task 2: Vietnamese → Full AMR (with variables, PENMAN format)

### Why MTUP?

| Aspect | Pipeline (❌ Wrong) | MTUP (✅ Correct) |
|--------|-------------------|------------------|
| Models | 2 separate models | 1 unified model |
| Training | 2 training runs | 1 training run |
| Knowledge | Isolated | Shared learning |
| Efficiency | Lower | Higher |
| F1 Score | Baseline | Target: Better |

## 📊 Data Statistics

- **Training samples:** 1,262 (validated)
- **Data size:** 1.5 MB
- **Format:** Unified prompt with both tasks
- **Quality:** All samples validated for bracket balance

## ⚙️ Technical Details

### Model Configuration
- **Base Model:** Qwen/Qwen2.5-7B-Instruct
- **Method:** 4-bit QLoRA
- **LoRA Rank:** 64
- **Batch Size:** 2 (effective: 32 with gradient accumulation)
- **Learning Rate:** 1e-4
- **Epochs:** 5 (default)

### Hardware Requirements
- **GPU:** ≥24GB VRAM (RTX 3090/4090, A100)
- **RAM:** ≥32GB
- **Disk:** ≥50GB free
- **Training Time:** ~2-3 hours (5 epochs on RTX 4090)

## 🎓 Co-reference Handling

Critical for high F1 score!

### Rules:
1. **Define once:** `(t / tôi)` - First occurrence
2. **Reuse:** `t` - Subsequent occurrences (NOT `(t / tôi)` again!)
3. **Pronouns:** Must reference correct entity

### Example:
```
Câu: Tôi là bác sĩ. Tôi làm ở bệnh viện.

✅ CORRECT:
(a / and
    :op1(b / bác_sĩ :domain(t / tôi))
    :op2(l / làm :ARG0 t :location(b2 / bệnh_viện)))
                    ↑
                    Reuse variable 't'

❌ WRONG:
:op2(l / làm :ARG0(t / tôi) ...)  ← Duplicate definition!
```

See [COREFERENCE_EXAMPLES.md](mtup_v2/docs/COREFERENCE_EXAMPLES.md) for more details.

## 📈 Expected Results

### Success Criteria:
- ✅ F1 > 0.47 (beat baseline)
- ✅ Valid PENMAN format
- ✅ Correct co-reference handling
- ✅ Balanced brackets

### Target Performance:
| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| F1 Score | 0.47 | >0.47 | >0.50 |
| Improvement | - | +2% | +6% |

## 🛠️ Troubleshooting

### Common Issues:

**1. Out of Memory (OOM)**
```python
# Edit train_mtup_unified.py
per_device_train_batch_size=1  # Reduce from 2
gradient_accumulation_steps=32  # Increase from 16
```

**2. No variables in predictions**
```bash
# Model needs more training
--epochs 10  # Increase from 5
```

**3. Duplicate node errors**
```bash
# Model hasn't learned co-reference well
# Check training data has co-reference examples
grep -A 5 "multi-sentence" data/train_mtup_unified.txt
```

See [TRAINING_GUIDE.md](mtup_v2/docs/TRAINING_GUIDE.md) for more solutions.

## 📝 Files to Copy to Server

Essential files:
```bash
# Option 1: Copy directory
scp -r mtup_v2/ user@server:/path/to/ViSemPar_new1/
scp data/train_mtup_unified.txt user@server:/path/to/ViSemPar_new1/data/

# Option 2: Create tarball
tar -czf mtup_v2.tar.gz mtup_v2/ data/train_mtup_unified.txt
scp mtup_v2.tar.gz user@server:/path/to/ViSemPar_new1/
```

See [COPY_TO_SERVER.md](COPY_TO_SERVER.md) for detailed instructions.

## 🔬 Evaluation

The model generates predictions in PENMAN format, which are evaluated against ground truth using SMATCH metric:

- **Precision:** Correct triples / Predicted triples
- **Recall:** Correct triples / Gold triples
- **F1:** Harmonic mean of precision and recall

## 📚 Documentation

### Essential Reading (in order):
1. [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md) - Start here!
2. [mtup_v2/docs/MTUP_CONCEPT.md](mtup_v2/docs/MTUP_CONCEPT.md) - Understand the approach
3. [mtup_v2/docs/TRAINING_GUIDE.md](mtup_v2/docs/TRAINING_GUIDE.md) - Step-by-step guide
4. [mtup_v2/docs/COREFERENCE_EXAMPLES.md](mtup_v2/docs/COREFERENCE_EXAMPLES.md) - Critical for quality

### Reference:
- [MTUP_V2_SUMMARY.md](MTUP_V2_SUMMARY.md) - Implementation details
- [COPY_TO_SERVER.md](COPY_TO_SERVER.md) - Server setup

## 🎯 Next Steps

1. ✅ Read [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md)
2. ✅ Run preprocessing locally
3. ✅ Copy files to server (see [COPY_TO_SERVER.md](COPY_TO_SERVER.md))
4. ✅ Train on server
5. ✅ Evaluate results
6. 🎉 Beat baseline F1!

## 📊 Version History

- **v2.0** (2026-01-10): Complete rewrite with unified MTUP approach
- **v1.x** (archived): Two-stage pipeline approach (incorrect MTUP)

## 🏆 Goal

**Beat Baseline F1: 0.47 → Target: >0.47 → Stretch: >0.50**

---

**Status:** ✅ Ready for Training
**Last Updated:** 2026-01-10
**Author:** Vietnamese AMR Parsing Team
**Competition:** VLSP 2025 - AMR Parsing
