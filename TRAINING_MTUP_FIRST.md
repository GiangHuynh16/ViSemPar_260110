# 🚀 Training MTUP First - Clear Organization

## 🎯 Plan

**Hôm nay**: Train MTUP (Qwen 2.5 7B, 2-task decomposition)
**Ngày mai**: Train Baseline (Qwen 2.5 7B, single-task)

## 📁 Folder Structure (Rõ Ràng)

```
ViSemPar_new1/
├── outputs/
│   ├── models/                              # ← Main models folder
│   │   ├── baseline_single_task_7b/        # ← Baseline (tomorrow)
│   │   │   ├── adapter_model.bin
│   │   │   ├── adapter_config.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── training_args.bin
│   │   │
│   │   └── mtup_two_task_7b/               # ← MTUP (today)
│   │       ├── adapter_model.bin
│   │       ├── adapter_config.json
│   │       ├── tokenizer_config.json
│   │       └── training_args.bin
│   │
│   ├── checkpoints_mtup/                    # Training checkpoints (temp)
│   │   ├── checkpoint-250/
│   │   ├── checkpoint-500/
│   │   └── ...
│   │
│   └── evaluation/                          # Evaluation results
│       ├── mtup_results.json
│       └── baseline_results.json
│
├── logs/
│   ├── training_mtup.log
│   └── training_baseline.log
│
└── api/                                      # ← For API deployment (future)
    ├── load_model.py
    ├── api_server.py
    └── README.md
```

## 🔧 Step 1: Update Config for Clear Folder Names

Tôi sẽ update config để lưu vào folder rõ ràng hơn:

### config_mtup.py
```python
# Change line 14:
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_mtup"

# To:
CHECKPOINT_DIR = OUTPUT_DIR / "models/mtup_two_task_7b"
```

### config.py (for baseline - tomorrow)
```python
# Change line 14:
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# To:
CHECKPOINT_DIR = OUTPUT_DIR / "models/baseline_single_task_7b"
```

## 🚀 Step 2: Train MTUP Today

### Create folders
```bash
cd ~/ViSemPar_new1

# Create clean folder structure
mkdir -p outputs/models/mtup_two_task_7b
mkdir -p outputs/models/baseline_single_task_7b
mkdir -p outputs/evaluation
mkdir -p logs
mkdir -p api
```

### Pull latest code
```bash
git stash
git pull origin main
git stash drop
```

### Train MTUP
```bash
# Start training in tmux
tmux new -s mtup_training

# Train with clear output path
python3 train_mtup.py \
  --use-case best_accuracy \
  --epochs 10 \
  --output-dir outputs/models/mtup_two_task_7b

# Detach: Ctrl+B, then D
```

### Monitor training
```bash
# Watch log
tail -f logs/training_mtup.log

# Check GPU
watch -n 1 nvidia-smi

# Re-attach
tmux attach -t mtup_training
```

## 📊 Step 3: After Training - Save Final Model

Training sẽ tự động save vào:
```
outputs/models/mtup_two_task_7b/
├── adapter_model.bin          # LoRA weights (~400MB)
├── adapter_config.json        # LoRA config
├── tokenizer_config.json      # Tokenizer
├── special_tokens_map.json    # Special tokens
├── training_args.bin          # Training args
└── trainer_state.json         # Training state
```

## 🔍 Step 4: Verify Model After Training

```bash
# Check model exists
ls -lh outputs/models/mtup_two_task_7b/

# Check size
du -sh outputs/models/mtup_two_task_7b/
# Should be ~400-600MB (LoRA adapters only)

# Test load
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    device_map='auto',
    torch_dtype=torch.float16
)

print('Loading MTUP adapter...')
model = PeftModel.from_pretrained(
    base,
    'outputs/models/mtup_two_task_7b'
)

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    'outputs/models/mtup_two_task_7b'
)

print('✅ MTUP model loaded successfully!')
print(f'Model device: {model.device}')
"
```

## 📊 Step 5: Evaluate MTUP

```bash
# Evaluate on test set
python3 evaluate_mtup_model.py \
  --checkpoint outputs/models/mtup_two_task_7b \
  --test-file data/public_test_ground_truth.txt \
  --output outputs/evaluation/mtup_results.json

# Results will show F1, Precision, Recall
```

## 🌐 Step 6: Prepare for API (Tomorrow)

### Create API loader script
```bash
cat > api/load_model.py << 'EOF'
"""
Load trained Vietnamese AMR models for API deployment
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

class AMRModelLoader:
    """Load and manage AMR models"""

    def __init__(self, model_type='mtup'):
        """
        Args:
            model_type: 'mtup' or 'baseline'
        """
        self.model_type = model_type
        self.base_model_name = "Qwen/Qwen2.5-7B-Instruct"

        # Model paths
        if model_type == 'mtup':
            self.adapter_path = "outputs/models/mtup_two_task_7b"
        else:
            self.adapter_path = "outputs/models/baseline_single_task_7b"

    def load(self, device='auto'):
        """Load model and tokenizer"""
        print(f"Loading {self.model_type} model...")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)

        print(f"✅ {self.model_type.upper()} model loaded")
        return model, tokenizer

    def generate_amr(self, model, tokenizer, sentence, max_length=512):
        """Generate AMR for a sentence"""

        if self.model_type == 'mtup':
            # MTUP: 2-task prompt
            prompt = f"""### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{sentence}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR (chưa có biến)
"""
        else:
            # Baseline: Simple prompt
            prompt = f"""Convert the following Vietnamese sentence to AMR format.

Sentence: {sentence}

AMR:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract AMR
        if self.model_type == 'mtup':
            # Extract from "BƯỚC 2" section
            if "## BƯỚC 2" in result:
                parts = result.split("## BƯỚC 2")[1]
                if "AMR hoàn chỉnh:" in parts:
                    amr = parts.split("AMR hoàn chỉnh:")[-1].strip()
                else:
                    amr = parts.strip()
            else:
                amr = result.strip()
        else:
            # Extract after prompt
            amr = result.replace(prompt, "").strip()

        # Clean: Extract AMR structure only
        if '(' in amr:
            first_paren = amr.index('(')
            amr = amr[first_paren:].strip()

        return amr


# Usage example
if __name__ == "__main__":
    # Load MTUP model
    loader = AMRModelLoader('mtup')
    model, tokenizer = loader.load()

    # Test
    sentence = "Tôi yêu Việt Nam"
    amr = loader.generate_amr(model, tokenizer, sentence)

    print(f"Sentence: {sentence}")
    print(f"AMR: {amr}")
EOF
```

### Create simple API server (for future)
```bash
cat > api/api_server.py << 'EOF'
"""
Simple FastAPI server for Vietnamese AMR parsing
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from load_model import AMRModelLoader
import uvicorn

# Initialize
app = FastAPI(title="Vietnamese AMR Parser API")

# Load model on startup
print("Loading MTUP model...")
loader = AMRModelLoader('mtup')
model, tokenizer = loader.load()
print("✅ Model ready")

class AMRRequest(BaseModel):
    sentence: str
    max_length: int = 512

class AMRResponse(BaseModel):
    sentence: str
    amr: str
    model_type: str

@app.post("/parse", response_model=AMRResponse)
async def parse_sentence(request: AMRRequest):
    """Parse Vietnamese sentence to AMR"""
    try:
        amr = loader.generate_amr(
            model,
            tokenizer,
            request.sentence,
            max_length=request.max_length
        )

        return AMRResponse(
            sentence=request.sentence,
            amr=amr,
            model_type="mtup_two_task_7b"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model": "mtup_two_task_7b"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

### Create API README
```bash
cat > api/README.md << 'EOF'
# Vietnamese AMR Parser API

## Models Available

1. **MTUP (Two-Task)**: `mtup_two_task_7b`
   - Location: `outputs/models/mtup_two_task_7b/`
   - Approach: 2-task decomposition
   - Expected F1: ~0.49-0.53

2. **Baseline (Single-Task)**: `baseline_single_task_7b`
   - Location: `outputs/models/baseline_single_task_7b/`
   - Approach: Direct generation
   - Expected F1: ~0.42-0.46

## Quick Start

### Load Model Programmatically

```python
from api.load_model import AMRModelLoader

# Load MTUP
loader = AMRModelLoader('mtup')
model, tokenizer = loader.load()

# Parse sentence
amr = loader.generate_amr(model, tokenizer, "Tôi yêu Việt Nam")
print(amr)
```

### Run API Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Start server
cd api
python api_server.py

# Server runs on http://localhost:8000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Parse sentence
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Tôi yêu Việt Nam"}'
```

## Model Paths

Both models are LoRA adapters (~400-600MB each):

```
outputs/models/
├── mtup_two_task_7b/          # MTUP model
│   └── adapter_model.bin
└── baseline_single_task_7b/   # Baseline model
    └── adapter_model.bin
```

Base model (Qwen 2.5 7B) will be downloaded automatically from HuggingFace.
EOF
```

## 📋 Summary

### Today's Tasks (MTUP Training)

1. ✅ Create clear folder structure
2. ✅ Pull latest code
3. 🔄 Train MTUP → `outputs/models/mtup_two_task_7b/`
4. ✅ Verify model loads correctly
5. ✅ Evaluate on test set
6. ✅ Create API loader for future use

### Tomorrow's Tasks (Baseline Training)

1. Train Baseline → `outputs/models/baseline_single_task_7b/`
2. Evaluate baseline
3. Compare MTUP vs Baseline
4. Document results for thesis

### For API Deployment (Future)

Models are ready to use at:
- **MTUP**: `outputs/models/mtup_two_task_7b/`
- **Baseline**: `outputs/models/baseline_single_task_7b/`

Just use `api/load_model.py` to load them! 🚀

## 🚀 Quick Start Command

```bash
# On server - Today
cd ~/ViSemPar_new1
git stash && git pull origin main && git stash drop
mkdir -p outputs/models/{mtup_two_task_7b,baseline_single_task_7b} outputs/evaluation logs api

# Train MTUP
tmux new -s mtup_training
python3 train_mtup.py \
  --use-case best_accuracy \
  --epochs 10 \
  --output-dir outputs/models/mtup_two_task_7b
```

Expected time: **4-6 hours** ⏰

---

**Status**: Ready to train MTUP! 🎯
**Folder**: Clear and organized for API use ✅
**Next**: Baseline training tomorrow 📅
