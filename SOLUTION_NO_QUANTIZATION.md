# Solution: Train 7B Without Quantization

## Problem
Server has CUDA 12.6 but PyTorch was compiled with CUDA 11.8, causing bitsandbytes to fail:
- Missing `libcusparse.so.11` (only has `.so.12`)
- Cannot use 4-bit quantization

## Solution
Train 7B model WITHOUT quantization using memory optimization techniques:

### Configuration Changes

1. **Disabled 4-bit quantization**
   ```python
   USE_4BIT_QUANTIZATION = False
   ```

2. **Reduced sequence length** (512 → 256)
   - AMR graphs are typically shorter than 256 tokens
   - Saves ~50% activation memory

3. **Reduced LoRA rank** (128 → 64)
   - Still sufficient capacity for AMR task
   - Saves LoRA parameter memory

4. **Enabled gradient checkpointing CORRECTLY**
   - Applied AFTER LoRA (critical!)
   - Use `enable_input_require_grads()` for PEFT compatibility
   - Saves ~60% activation memory

### Memory Calculation

Without quantization but WITH gradient checkpointing:
- **Model (BF16)**: ~14 GB
- **LoRA params (rank=64)**: ~0.3 GB
- **Activations (batch=1, seq=256, checkpointing)**: ~3 GB
- **Optimizer states**: ~1 GB
- **TOTAL**: ~18 GB << 48 GB ✓

This should fit comfortably in A6000's 48GB VRAM.

### Code Changes

1. `config/config.py`:
   - `USE_4BIT_QUANTIZATION = False`
   - `MAX_SEQ_LENGTH = 256`
   - `LORA_CONFIG['r'] = 64`

2. `train_baseline.py`:
   - Enable gradient checkpointing AFTER `get_peft_model()`
   - Use `model.enable_input_require_grads()`
   - Use `model.base_model.model.gradient_checkpointing_enable()`

### Next Steps

1. Pull latest code:
   ```bash
   cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
   git pull origin main
   ```

2. Start training:
   ```bash
   conda activate baseline_final
   bash START_TRAINING_NOW.sh
   ```

3. Monitor memory:
   ```bash
   watch -n 5 nvidia-smi
   ```

### Why This Works

**Gradient Checkpointing** trades compute for memory:
- Normally: Store all activations during forward pass
- With checkpointing: Only store subset, recompute others during backward
- Memory reduction: ~60% for transformer models
- Training time increase: ~20% (acceptable tradeoff)

**Reduced Sequence Length**:
- Attention complexity: O(n²) where n = sequence length
- 256 vs 512 = 4x less memory for attention
- Vietnamese AMR graphs rarely exceed 256 tokens

**Smaller LoRA Rank**:
- Rank 64 vs 128 = half the parameters
- Still provides sufficient capacity for fine-tuning
- Baseline MTUP likely uses similar or smaller rank

### Alternative: Install CUDA 11.8

If this doesn't work, we can install CUDA 11.8 toolkit:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
```

But the current approach should work without needing this.
