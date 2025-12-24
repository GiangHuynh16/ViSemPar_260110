# MTUP Training Guide

**Multi-Task Unified Prompt (MTUP) Training Strategy for Vietnamese AMR Parser**

---

## 🎯 **What is MTUP?**

MTUP is a **one-prompt multi-task** training strategy that provides **explicit supervision** for easier subtasks:

### **Key Principles:**
1. **Unified Prompt**: Single prompt contains all tasks consecutively
2. **Explicit Supervision**: Each subtask has its own supervision signal
3. **Task Decomposition**:
   - **Task 1**: Vietnamese → AMR structure (no variables)
   - **Task 2**: Add variable binding to structure
4. **Self-Correction**: Model learns to refine from Task 1 → Task 2
5. **Extensible**: Can add more tasks (concept extraction, relation extraction, etc.)
6. **Knowledge Integration**: Easy to add linguistic constraints and knowledge

### **Benefits:**
✅ **2-3x faster training** with smaller models (3-4B params)
✅ **Better learning** through explicit subtask decomposition
✅ **Vietnamese character handling** (đ, ô, ê, etc.)
✅ **Variable collision learning** (n, n1, n2 for different concepts)
✅ **Context preservation** across tasks in unified prompt

---

## 🚀 **Quick Start**

### **1. Quick Test (100 samples, 1 epoch)**
```bash
python3 train_mtup.py --use-case quick_test
```

**What it does:**
- Uses 100 training samples
- Trains for 1 epoch
- Shows sample MTUP format
- Fast validation (~5-10 minutes)

**Expected output:**
```
✓ Loaded 100 examples
✓ Task 1: No variables (đ / đó) → (đó)
✓ Task 2: Has variables (đ / đó)
✅ Training completed
```

---

### **2. Fast Iteration (500 samples, 2 epochs)**
```bash
python3 train_mtup.py --use-case fast_iteration
```

**What it does:**
- Uses 500 training samples
- Trains for 2 epochs
- Good for testing hyperparameters
- Training time: ~30-60 minutes (3B model, GPU)

---

### **3. Full Training (all data, 3 epochs)**
```bash
python3 train_mtup.py --use-case full_training
```

**What it does:**
- Uses all training data (~2500 examples)
- Trains for 3 epochs
- Production-quality model
- Training time: ~2-3 hours (3B model, GPU)

---

## ⚙️ **Advanced Usage**

### **Custom Configuration**

```bash
python3 train_mtup.py \
  --use-case fast_iteration \
  --model qwen2.5-3b \
  --max-samples 1000 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --val-split 0.1 \
  --show-sample
```

### **Available Models**

```bash
--model qwen2.5-3b      # DEFAULT - Qwen 2.5 3B (recommended for MTUP)
--model qwen2.5-7b      # Qwen 2.5 7B (better quality, slower)
--model qwen3-4b        # Qwen 3 4B (newer architecture)
--model gemma-2-2b      # Gemma 2 2B (smaller, faster)
--model phi-3.5-mini    # Phi 3.5 Mini 3.8B (alternative)
```

### **Training Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-case` | `fast_iteration` | Preset: `quick_test`, `fast_iteration`, `full_training` |
| `--train-files` | `train_amr_1.txt train_amr_2.txt` | Training data files |
| `--max-samples` | None (all) | Limit training samples |
| `--val-split` | 0.1 | Validation split ratio |
| `--model` | `qwen2.5-3b` | Model to use |
| `--max-length` | 2048 | Max sequence length (MTUP needs longer) |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 4 | Batch size per device |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--log-steps` | 10 | Logging frequency |
| `--save-steps` | 100 | Checkpoint save frequency |
| `--eval-steps` | 100 | Evaluation frequency |
| `--show-sample` | False | Show MTUP format example |

---

## 📊 **Understanding MTUP Format**

### **Example MTUP Training Instance:**

```
### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
Đó là bi kịch

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
(bi_kịch :domain(chỗ :mod(đó)))

## Bước 2 - Gán biến cho các khái niệm:
Hướng dẫn:
• Mỗi khái niệm được gán một biến riêng (ví dụ: n, n2, p, c...)
• Khái niệm xuất hiện nhiều lần → dùng chung một biến (đồng tham chiếu)
• Format: (biến / khái_niệm :quan_hệ...)

AMR hoàn chỉnh:
(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
```

### **What Model Learns:**

1. **Task 1 Supervision**: Generate AMR structure without variables
   - Focus: semantic structure, relations
   - Easier subtask: no variable naming needed

2. **Task 2 Supervision**: Add variable binding
   - Focus: variable assignment, co-reference
   - Learns: first-letter pattern, collision resolution (n, n1, n2)

3. **Self-Correction**: Model sees both outputs in sequence
   - Can learn to fix Task 1 mistakes in Task 2
   - Context preserved through unified prompt

---

## 🔍 **Monitoring Training**

### **TensorBoard**

```bash
# In another terminal
tensorboard --logdir outputs/logs
```

Open: http://localhost:6006

### **Watch Training Logs**

```bash
tail -f outputs/logs/mtup_*/training.log
```

### **Check GPU Usage**

```bash
watch -n 1 nvidia-smi
```

---

## 📈 **Expected Results**

### **Quick Test (100 samples, 1 epoch)**
- **Purpose**: Verify pipeline works
- **Expected**: Overfitting on small data
- **Time**: ~5-10 minutes
- **Use**: Debugging, testing changes

### **Fast Iteration (500 samples, 2 epochs)**
- **Purpose**: Hyperparameter tuning
- **Expected**: Reasonable AMR structure
- **Time**: ~30-60 minutes
- **Use**: Finding optimal settings

### **Full Training (all data, 3 epochs)**
- **Purpose**: Production model
- **Expected SMATCH**: 70-80% F1 (target)
- **Time**: ~2-3 hours (3B model)
- **Use**: Final evaluation, submission

---

## 🧪 **After Training - Evaluation**

### **1. Run Test Scripts**

```bash
# Test SMATCH evaluation
python3 test_smatch.py

# Evaluate on real data
python3 evaluate_test_data.py
```

### **2. Inference on New Sentences**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "outputs/checkpoints/mtup_full_training_final")
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints/mtup_full_training_final")

# Generate
sentence = "Tôi nhớ lời bạn"
# Create MTUP prompt with sentence (Task 1 and 2 will be generated)
```

### **3. Analyze Task-Specific Performance**

```bash
# Check Task 1 quality (structure without variables)
# Check Task 2 quality (variable binding)
# Compare SMATCH scores
```

---

## ⚠️ **Troubleshooting**

### **Issue 1: Out of Memory (OOM)**

**Solution 1**: Reduce batch size
```bash
python3 train_mtup.py --batch-size 2 --grad-accum 8
```

**Solution 2**: Use smaller model
```bash
python3 train_mtup.py --model gemma-2-2b
```

**Solution 3**: Reduce sequence length
```bash
python3 train_mtup.py --max-length 1536
```

---

### **Issue 2: Training Too Slow**

**Solution 1**: Use smaller model (recommended for MTUP)
```bash
python3 train_mtup.py --model qwen2.5-3b  # 2-3x faster than 7B
```

**Solution 2**: Increase batch size (if GPU memory allows)
```bash
python3 train_mtup.py --batch-size 8
```

**Solution 3**: Use mixed precision (automatic with fp16=True)

---

### **Issue 3: Poor Variable Binding (Task 2)**

**Symptoms**:
- Task 1 output looks good (structure correct)
- Task 2 has wrong variables or missing bindings

**Solutions**:

1. **Check preprocessing**:
```bash
python3 test_mtup_simple.py
# Verify: Task 1 has no variables, Task 2 has variables
```

2. **Verify Vietnamese character handling**:
```bash
# Should see: (đ / đó) not (đ / đ / đó)
```

3. **Increase training data or epochs**:
```bash
python3 train_mtup.py --epochs 5
```

---

### **Issue 4: Model Not Loading**

**Error**: `ModuleNotFoundError: No module named 'peft'`

**Solution**:
```bash
pip install -r requirements.txt
# Or specifically:
pip install peft transformers accelerate bitsandbytes
```

---

## 📝 **Best Practices**

### **1. Start Small, Scale Up**
```bash
# Step 1: Quick test
python3 train_mtup.py --use-case quick_test

# Step 2: Fast iteration
python3 train_mtup.py --use-case fast_iteration

# Step 3: Full training
python3 train_mtup.py --use-case full_training
```

### **2. Monitor Both Tasks**
- Check Task 1 quality (structure)
- Check Task 2 quality (variables)
- Don't optimize only for final SMATCH score

### **3. Use 3B Models for MTUP**
- MTUP makes smaller models effective
- 3B model is 2-3x faster than 7B
- Quality is comparable due to better task decomposition

### **4. Tune Hyperparameters on Fast Iteration**
```bash
# Test different learning rates
python3 train_mtup.py --use-case fast_iteration --lr 1e-4
python3 train_mtup.py --use-case fast_iteration --lr 2e-4
python3 train_mtup.py --use-case fast_iteration --lr 5e-4
```

---

## 🎓 **Understanding the MTUP Advantage**

### **Traditional Single-Task Approach:**
```
Input: Đó là bi kịch
Output: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
```
- ❌ Model must learn structure AND variable binding simultaneously
- ❌ Harder task, needs larger models
- ❌ Slower convergence

### **MTUP Multi-Task Approach:**
```
Input: Đó là bi kịch

Task 1 Output: (bi_kịch :domain(chỗ :mod(đó)))
              ↓ (easier - just structure)

Task 2 Output: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
              ↑ (add variables with explicit guidance)
```
- ✅ Decomposed into easier subtasks
- ✅ Explicit supervision for each task
- ✅ Faster learning, smaller models work
- ✅ Model can self-correct across tasks

---

## 🔗 **Related Documentation**

- [MTUP_IMPLEMENTATION.md](MTUP_IMPLEMENTATION.md) - Implementation details
- [CRITICAL_ANALYSIS.md](CRITICAL_ANALYSIS.md) - Variable collision analysis
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Server deployment
- [QUICK_COMMANDS.md](QUICK_COMMANDS.md) - Copy-paste commands

---

## 📞 **Support**

**Questions about:**
- MTUP strategy → See [MTUP_IMPLEMENTATION.md](MTUP_IMPLEMENTATION.md)
- Variable handling → See [CRITICAL_ANALYSIS.md](CRITICAL_ANALYSIS.md)
- Server setup → See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Quick commands → See [QUICK_COMMANDS.md](QUICK_COMMANDS.md)

---

**Ready to train? Start with:**
```bash
python3 train_mtup.py --use-case quick_test --show-sample
```

Good luck! 🚀
