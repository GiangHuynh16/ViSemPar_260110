#!/usr/bin/env python3
"""
Push trained models to HuggingFace Hub for easy access from local
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os

def push_model_to_hf(
    model_path: str,
    repo_name: str,
    model_type: str = "mtup",
    private: bool = True
):
    """
    Push model to HuggingFace Hub

    Args:
        model_path: Local path to model (e.g., outputs/models/mtup_two_task_7b)
        repo_name: HF repo name (e.g., "vietnamese-amr-mtup-7b")
        model_type: "mtup" or "baseline"
        private: If True, repo will be private
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Check required files
    required_files = ["adapter_model.bin", "adapter_config.json"]
    for file in required_files:
        if not (model_path / file).exists():
            raise ValueError(f"Missing required file: {file}")

    print(f"🚀 Pushing {model_type.upper()} model to HuggingFace Hub...")
    print(f"   Local path: {model_path}")
    print(f"   Repo name: {repo_name}")
    print(f"   Private: {private}")
    print()

    # Initialize API
    api = HfApi()

    # Get username
    user = api.whoami()
    username = user['name']
    full_repo_name = f"{username}/{repo_name}"

    print(f"✅ Logged in as: {username}")
    print(f"   Full repo: {full_repo_name}")
    print()

    # Create repo (if doesn't exist)
    try:
        print("📦 Creating repository...")
        create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository created/verified: {full_repo_name}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")

    print()

    # Create model card
    model_card = f"""---
language:
- vi
license: apache-2.0
tags:
- amr
- semantic-parsing
- vietnamese
- qwen2.5
- lora
library_name: peft
base_model: Qwen/Qwen2.5-7B-Instruct
---

# Vietnamese AMR Parser - {model_type.upper()}

This is a LoRA adapter for Vietnamese Abstract Meaning Representation (AMR) parsing.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Approach**: {"Two-Task Decomposition (MTUP)" if model_type == "mtup" else "Single-Task Direct Generation"}
- **LoRA Rank**: {64 if model_type == "mtup" else 128}
- **Training Data**: Vietnamese AMR corpus
- **Framework**: PEFT (Parameter-Efficient Fine-Tuning)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{full_repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{full_repo_name}")

# Generate AMR
sentence = "Tôi yêu Việt Nam"
{"# MTUP uses 2-task prompt" if model_type == "mtup" else "# Baseline uses simple prompt"}
prompt = f\"\"\"### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{{sentence}}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR (chưa có biến)
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Performance

- **F1 Score**: {" ~0.49-0.53 (expected)" if model_type == "mtup" else "~0.42-0.46 (expected)"}
- **Evaluation Metric**: SMATCH

## Citation

If you use this model, please cite:

```bibtex
@misc{{vietnamese-amr-{model_type}-2025,
  author = {{Your Name}},
  title = {{Vietnamese AMR Parser ({model_type.upper()})}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{full_repo_name}}}
}}
```

## License

Apache 2.0
"""

    # Save model card
    readme_path = model_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    print("✅ Model card created")

    # Upload all files
    print()
    print("📤 Uploading files to HuggingFace Hub...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_name,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "*.log", "runs/"]
    )

    print()
    print("=" * 80)
    print("✅ SUCCESS! Model pushed to HuggingFace Hub")
    print("=" * 80)
    print()
    print(f"🔗 Model URL: https://huggingface.co/{full_repo_name}")
    print()
    print("📥 To use on local machine:")
    print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
    "{full_repo_name}"
)
""")
    print()


def main():
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (e.g., outputs/models/mtup_two_task_7b)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="HuggingFace repo name (e.g., vietnamese-amr-mtup-7b)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mtup", "baseline"],
        default="mtup",
        help="Model type: mtup or baseline"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private (default: public)"
    )

    args = parser.parse_args()

    push_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        model_type=args.model_type,
        private=args.private
    )


if __name__ == "__main__":
    main()
