# 🌐 HuggingFace Deployment Guide

## 🎯 Tại Sao Dùng HuggingFace?

Vì bạn muốn **build API ở local**, không phải server:

### ✅ HuggingFace (Recommended)
- Download về local trong **1-2 phút**
- Dùng được **ở bất kỳ đâu** (local, cloud, colab)
- **Automatic versioning**
- Dễ share với team/reviewer
- Professional (như các model SOTA khác)

### ❌ Lưu Trên Server
- Phải scp download (~2-5 phút)
- Chỉ access được khi có SSH
- Manual versioning
- Khó share
- Không professional

## 🚀 Workflow Hoàn Chỉnh

```
[SERVER]                    [HUGGINGFACE]              [LOCAL]

1. Train model    →    2. Push to HF Hub    →    3. Download & use
   (4-6h)                    (2-3 min)                 (1-2 min)

outputs/models/          your-username/           ~/.cache/huggingface/
mtup_two_task_7b/   →   vietnamese-amr-mtup  →   models/...
                                                    ↓
                                                 API Server
```

## 📋 Complete Steps

### Step 1: Train on Server (Hôm Nay)

```bash
# On server
cd ~/ViSemPar_new1
git pull origin main

# Train MTUP
tmux new -s mtup_training
python3 train_mtup.py \
  --use-case best_accuracy \
  --epochs 10 \
  --output-dir outputs/models/mtup_two_task_7b
```

### Step 2: Setup HuggingFace (One-time)

```bash
# On server
pip install huggingface_hub

# Login to HF
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

### Step 3: Push to HuggingFace (Sau Khi Train Xong)

```bash
# After training completes (~4-6h)
python3 push_to_huggingface.py \
  --model-path outputs/models/mtup_two_task_7b \
  --repo-name vietnamese-amr-mtup-7b \
  --model-type mtup \
  --private  # Use --private for private repo, omit for public
```

**Output**:
```
🚀 Pushing MTUP model to HuggingFace Hub...
✅ Logged in as: your-username
📦 Creating repository...
✅ Repository created: your-username/vietnamese-amr-mtup-7b
📤 Uploading files...
✅ SUCCESS!

🔗 https://huggingface.co/your-username/vietnamese-amr-mtup-7b
```

### Step 4: Use on Local Machine (Your Laptop)

```bash
# On your local machine
pip install transformers peft torch

# No need to download manually - it auto-downloads!
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load from HuggingFace (downloads automatically to cache)
print("Loading model from HuggingFace...")

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# This downloads your model from HF
model = PeftModel.from_pretrained(
    base_model,
    "your-username/vietnamese-amr-mtup-7b"  # ← Your HF repo
)

tokenizer = AutoTokenizer.from_pretrained(
    "your-username/vietnamese-amr-mtup-7b"
)

print("✅ Model ready for API!")
```

## 🌐 Build API on Local

### Option A: Simple Python API

```python
# api_local.py
from fastapi import FastAPI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model from HuggingFace
print("Loading model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base, "your-username/vietnamese-amr-mtup-7b")
tokenizer = AutoTokenizer.from_pretrained("your-username/vietnamese-amr-mtup-7b")
print("✅ Model loaded!")

@app.post("/parse")
async def parse(sentence: str):
    prompt = f"""### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{sentence}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR (chưa có biến)
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AMR
    if "AMR hoàn chỉnh:" in result:
        amr = result.split("AMR hoàn chỉnh:")[-1].strip()
    else:
        amr = result

    return {"sentence": sentence, "amr": amr}

# Run: uvicorn api_local:app --reload
```

### Option B: Gradio Web UI

```python
# gradio_ui.py
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base, "your-username/vietnamese-amr-mtup-7b")
tokenizer = AutoTokenizer.from_pretrained("your-username/vietnamese-amr-mtup-7b")

def parse_amr(sentence):
    prompt = f"""### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{sentence}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR (chưa có biến)
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Create UI
demo = gr.Interface(
    fn=parse_amr,
    inputs=gr.Textbox(label="Vietnamese Sentence", lines=2),
    outputs=gr.Textbox(label="AMR Output", lines=10),
    title="Vietnamese AMR Parser",
    description="Parse Vietnamese sentences to Abstract Meaning Representation"
)

demo.launch()
```

## 📊 Timeline & Storage

### Timeline
1. **Training** (server): 4-6 hours
2. **Push to HF** (server): 2-3 minutes
3. **Download to local**: 1-2 minutes (first time only)
4. **API ready**: Immediate (cached)

### Storage

**On Server** (temporary):
```
outputs/models/mtup_two_task_7b/  (~500MB)
```
→ Can delete after pushing to HF!

**On HuggingFace** (permanent):
```
your-username/vietnamese-amr-mtup-7b  (~500MB)
```

**On Local** (cached):
```
~/.cache/huggingface/hub/models--your-username--vietnamese-amr-mtup-7b/
```
→ Downloaded once, reused forever!

## 🔐 Private vs Public

### Private Repo (Recommended for thesis)
```bash
python3 push_to_huggingface.py \
  --model-path outputs/models/mtup_two_task_7b \
  --repo-name vietnamese-amr-mtup-7b \
  --private  # ← Add this flag
```

- ✅ Only you can access
- ✅ Can share with specific people (add collaborators on HF)
- ✅ Good for thesis work before publication

### Public Repo (After thesis)
```bash
python3 push_to_huggingface.py \
  --model-path outputs/models/mtup_two_task_7b \
  --repo-name vietnamese-amr-mtup-7b
  # No --private flag = public
```

- ✅ Anyone can use
- ✅ Good for citations
- ✅ Contributes to community

## 🎯 Complete Workflow Example

### Day 1: Train MTUP
```bash
# On server
cd ~/ViSemPar_new1
git pull origin main

tmux new -s mtup
python3 train_mtup.py --use-case best_accuracy --output-dir outputs/models/mtup_two_task_7b
# Wait 4-6 hours...
```

### Day 1 (Evening): Push to HF
```bash
# After training completes
huggingface-cli login  # One-time setup

python3 push_to_huggingface.py \
  --model-path outputs/models/mtup_two_task_7b \
  --repo-name vietnamese-amr-mtup-7b \
  --model-type mtup \
  --private

# ✅ Done! Model on HF: https://huggingface.co/your-username/vietnamese-amr-mtup-7b
```

### Day 2: Train Baseline
```bash
# Same process for baseline
python3 train_baseline.py --output-dir outputs/models/baseline_single_task_7b

# Push to HF
python3 push_to_huggingface.py \
  --model-path outputs/models/baseline_single_task_7b \
  --repo-name vietnamese-amr-baseline-7b \
  --model-type baseline \
  --private
```

### Day 3: Build API on Local
```bash
# On your laptop
pip install transformers peft torch gradio

# Run Gradio UI
python gradio_ui.py
# Opens browser at http://localhost:7860
```

## 🔍 Verify Model on HuggingFace

After pushing, check:

1. **Go to**: `https://huggingface.co/your-username/vietnamese-amr-mtup-7b`

2. **Should see**:
   - ✅ Model card (README)
   - ✅ Files: `adapter_model.bin`, `adapter_config.json`, etc.
   - ✅ Model size: ~400-600MB

3. **Test download**:
```python
from peft import PeftModel

# This should work immediately
model = PeftModel.from_pretrained(
    base_model,
    "your-username/vietnamese-amr-mtup-7b"
)
```

## 🎯 Summary

**Question**: Lưu model ở server hay push HF?
**Answer**: **Push lên HuggingFace** ✅

**Why**:
- ✅ Dùng dễ dàng trên local (1 dòng code)
- ✅ Professional & portable
- ✅ Automatic versioning
- ✅ Can share with reviewer/team
- ✅ No need SSH to server

**How**:
1. Train trên server
2. Push lên HF (`push_to_huggingface.py`)
3. Use trên local (`from_pretrained("your-repo")`)

**Time**:
- Push: 2-3 phút
- Download (first time): 1-2 phút
- After that: Instant (cached)

---

**Ready to deploy!** 🚀
