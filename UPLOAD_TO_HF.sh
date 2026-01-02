#!/bin/bash
# Upload baseline 7B model to Hugging Face

set -e

echo "==========================================="
echo "UPLOAD BASELINE 7B TO HUGGING FACE"
echo "==========================================="
echo ""

# Configuration
CHECKPOINT_DIR="outputs/baseline_20260102_125130/checkpoint-1545"
REPO_NAME="vietnamese-amr-baseline-7b"
HF_USERNAME="${HF_USERNAME:-}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT_DIR"
echo ""

# Get Hugging Face username
if [ -z "$HF_USERNAME" ]; then
    echo "Enter your Hugging Face username:"
    read HF_USERNAME
fi

echo "Hugging Face username: $HF_USERNAME"
echo "Repository: $HF_USERNAME/$REPO_NAME"
echo ""

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating baseline_final environment..."
    eval "$(conda shell.bash hook)"
    conda activate baseline_final
fi

# Install huggingface_hub if needed
echo "Checking huggingface_hub..."
pip show huggingface_hub > /dev/null 2>&1 || pip install huggingface_hub
echo ""

# Check login status
echo "Step 1: Checking Hugging Face authentication..."
if huggingface-cli whoami &>/dev/null; then
    CURRENT_USER=$(huggingface-cli whoami | head -1)
    echo "  ✓ Logged in as: $CURRENT_USER"
else
    echo "  ✗ Not logged in"
    echo ""
    echo "Please login to Hugging Face:"
    echo "  Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    huggingface-cli login
fi
echo ""

# Create model card
echo "Step 2: Creating model card..."
cat > "$CHECKPOINT_DIR/README.md" << 'MDEOF'
---
language: vi
license: apache-2.0
tags:
- vietnamese
- amr
- semantic-parsing
- qwen2.5
- vlsp2024
datasets:
- vlsp2024-amr
library_name: peft
base_model: Qwen/Qwen2.5-7B-Instruct
---

# Vietnamese AMR Baseline 7B

LoRA adapter for Vietnamese Abstract Meaning Representation (AMR) parsing, trained on VLSP 2024 dataset.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training Approach**: Single-task baseline with LoRA
- **Language**: Vietnamese
- **Task**: AMR Semantic Parsing
- **Dataset**: VLSP 2024 Vietnamese AMR

## Training Configuration

```yaml
Model: Qwen 2.5 7B Instruct
LoRA Rank: 64
LoRA Alpha: 128
Max Sequence Length: 256
Batch Size: 1 (effective: 16 with gradient accumulation)
Epochs: 15
Learning Rate: 2e-4
Optimizer: AdamW
Precision: BF16
Gradient Checkpointing: Enabled
```

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "YOUR_USERNAME/vietnamese-amr-baseline-7b"
)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Prepare prompt
sentence = "Chủ tịch nước đã phát biểu tại hội nghị."
prompt = f"""Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation).

Quy tắc quan trọng:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới (ví dụ: chủ_tịch, môi_trường)
- Gán biến cho mỗi khái niệm (ví dụ: c / chủ_tịch)
- Sử dụng quan hệ chuẩn AMR (:ARG0, :ARG1, :time, :location, etc.)
- Giữ nguyên cấu trúc cây với dấu ngoặc đơn cân bằng
- Đảm bảo tất cả biến được định nghĩa trước khi sử dụng

Câu tiếng Việt: {sentence}

AMR:
"""

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True
    )

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
amr = result.split("AMR:")[-1].strip()
print(amr)
```

## Training Details

- **Training Time**: ~1.5 hours on NVIDIA RTX A6000 (48GB)
- **Final Training Loss**: ~0.037
- **Validation Loss**: 0.419

## Files

- `adapter_config.json`: LoRA configuration
- `adapter_model.safetensors`: LoRA weights (~200MB)
- `README.md`: This file

## Citation

```bibtex
@misc{vietnamese-amr-baseline-7b,
  title={Vietnamese AMR Baseline 7B},
  author={VLSP 2024 Participant},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/YOUR_USERNAME/vietnamese-amr-baseline-7b}
}
```

## License

Apache 2.0

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
MDEOF

echo "  ✓ Model card created"
echo ""

# Upload to Hugging Face
echo "Step 3: Uploading to Hugging Face..."
echo "  Repository: $HF_USERNAME/$REPO_NAME"
echo "  This will upload ~200MB (LoRA adapter)"
echo ""

python << EOF
from huggingface_hub import HfApi, create_repo
import sys

api = HfApi()
repo_id = "$HF_USERNAME/$REPO_NAME"
checkpoint_dir = "$CHECKPOINT_DIR"

print("Creating repository...")
try:
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"  ✓ Repository ready")
except Exception as e:
    print(f"  ⚠️  {e}")

print("\nUploading files...")
print("  This may take 2-5 minutes...")
try:
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Vietnamese AMR Baseline 7B LoRA adapter"
    )
    print(f"\n  ✓ Upload complete!")
    print(f"\n🎉 Model available at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"\n  ✗ Upload failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "==========================================="
echo "UPLOAD COMPLETE!"
echo "==========================================="
echo ""
echo "Model URL: https://huggingface.co/$HF_USERNAME/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. Visit the model page and verify it loaded correctly"
echo "  2. Test the model with the usage example in README"
echo "  3. Create API endpoint for inference"
echo ""
