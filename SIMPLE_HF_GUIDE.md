# 🚀 Hướng Dẫn Push Model Lên HuggingFace - CỰC ĐƠN GIẢN

## 🎯 Tại Sao Dùng HF?

Vì bạn muốn **build API ở local**, không phải server:
- ✅ Download về local trong 1-2 phút
- ✅ Dùng được ở bất kỳ đâu
- ✅ Không cần SSH vào server mỗi lần
- ✅ Professional như các model SOTA

## 📋 3 Bước Cực Đơn Giản

### Bước 1: Lấy HuggingFace Token (1 lần duy nhất)

1. Vào https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Đặt tên: `model-upload`
4. Chọn **"Write"** permission
5. Click **"Generate"**
6. **Copy token** (dạng `hf_...`)

### Bước 2: Setup .env File (1 lần duy nhất)

```bash
# On server
cd ~/ViSemPar_new1

# Copy example to .env
cp .env.example .env

# Edit .env file
nano .env
# Hoặc: vim .env
```

**Trong file .env**, thay thế:
```bash
HF_TOKEN=hf_your_token_here      # ← Paste token từ Bước 1
HF_USERNAME=your_username        # ← Thay bằng username HF của bạn

# Optional: Tùy chỉnh tên repo
HF_REPO_MTUP=vietnamese-amr-mtup-7b
HF_REPO_BASELINE=vietnamese-amr-baseline-7b
MAKE_PRIVATE=true  # true = private, false = public
```

Save file (Ctrl+X, Y, Enter nếu dùng nano)

### Bước 3: Push Model (Sau khi train xong)

```bash
# Install python-dotenv (nếu chưa có)
pip install python-dotenv

# Push MTUP model
python3 push_to_hf_simple.py --model-type mtup

# Hoặc push Baseline model
python3 push_to_hf_simple.py --model-type baseline
```

**Chỉ cần vậy thôi!** 🎉

## 📊 Output Mẫu

```
================================================================================
🚀 PUSHING MTUP MODEL TO HUGGINGFACE HUB
================================================================================

📁 Local path: outputs/models/mtup_two_task_7b
👤 Username:   your-username
📦 Repo name:  vietnamese-amr-mtup-7b
🔐 Private:    True

🔐 Logging in to HuggingFace...
✅ Logged in successfully!

📦 Creating repository: your-username/vietnamese-amr-mtup-7b...
✅ Repository ready!

📝 Creating model card...
✅ Model card created

📤 Uploading files to HuggingFace Hub...
   This may take 2-3 minutes...

================================================================================
✅ SUCCESS! MODEL PUSHED TO HUGGINGFACE HUB
================================================================================

🔗 Model URL: https://huggingface.co/your-username/vietnamese-amr-mtup-7b

📥 To use on your local machine:

from peft import PeftModel
from transformers import AutoModelForCausalLM

model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
    "your-username/vietnamese-amr-mtup-7b"
)

✅ You can now delete the model from server to save space!
   rm -rf outputs/models/mtup_two_task_7b
```

## 🌐 Dùng Model Trên Local Machine

### Cách 1: Python Script

```python
# On your laptop
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load from HuggingFace (auto-download)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base,
    "your-username/vietnamese-amr-mtup-7b"  # ← Your HF repo
)

tokenizer = AutoTokenizer.from_pretrained(
    "your-username/vietnamese-amr-mtup-7b"
)

# Parse sentence
sentence = "Tôi yêu Việt Nam"
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

print(result)
```

### Cách 2: Gradio Web UI (Dễ Hơn)

```bash
# Install
pip install gradio transformers peft torch

# Create gradio_app.py
cat > gradio_app.py << 'EOF'
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model
print("Loading model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base, "your-username/vietnamese-amr-mtup-7b")
tokenizer = AutoTokenizer.from_pretrained("your-username/vietnamese-amr-mtup-7b")
print("✅ Model loaded!")

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

    # Extract AMR
    if "AMR hoàn chỉnh:" in result:
        return result.split("AMR hoàn chỉnh:")[-1].strip()
    return result

demo = gr.Interface(
    fn=parse_amr,
    inputs=gr.Textbox(label="Vietnamese Sentence", lines=2),
    outputs=gr.Textbox(label="AMR Output", lines=10),
    title="Vietnamese AMR Parser",
    examples=[
        "Tôi yêu Việt Nam",
        "Cô giáo đang dạy học sinh",
        "Anh ấy muốn mua một chiếc xe mới"
    ]
)

demo.launch()
EOF

# Run
python3 gradio_app.py
# Opens at http://localhost:7860
```

## 🔍 Troubleshooting

### Lỗi: "HF_TOKEN not set"
```bash
# Check .env file exists
ls -la .env

# Check content
cat .env

# Fix: Make sure .env has your token
nano .env
```

### Lỗi: "Model not found"
```bash
# Train model first!
python3 train_mtup.py --use-case best_accuracy

# Check model exists
ls outputs/models/mtup_two_task_7b/
```

### Lỗi: "Permission denied"
```bash
# Token needs write permission
# Go to HF settings → Tokens → Regenerate with "write" permission
```

## 📊 Complete Workflow

```
DAY 1 - SERVER
├── Train MTUP (4-6h)
│   python3 train_mtup.py --use-case best_accuracy
│
├── Setup .env (2 min)
│   cp .env.example .env
│   nano .env  # Add HF_TOKEN
│
└── Push to HF (2-3 min)
    python3 push_to_hf_simple.py --model-type mtup

DAY 2 - SERVER
├── Train Baseline (4-6h)
│   python3 train_baseline.py
│
└── Push to HF (2-3 min)
    python3 push_to_hf_simple.py --model-type baseline

DAY 3 - LOCAL
├── Install deps
│   pip install transformers peft torch gradio
│
├── Create gradio_app.py
│   # See example above
│
└── Run API
    python3 gradio_app.py
    # Open http://localhost:7860 ✅
```

## 🎯 Summary

**Question**: Làm sao push model lên HF dễ nhất?

**Answer**: 3 bước:
1. ✅ Lấy HF token (1 lần)
2. ✅ Edit .env file (1 lần)
3. ✅ Run `python3 push_to_hf_simple.py --model-type mtup`

**Time**: 2-3 phút để push, 1-2 phút để download về local lần đầu

**Result**: Model dùng được ở local bằng 1 dòng code:
```python
model = PeftModel.from_pretrained(base, "your-username/vietnamese-amr-mtup-7b")
```

---

**Cực kỳ đơn giản!** 🚀
