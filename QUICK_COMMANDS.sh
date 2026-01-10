#!/bin/bash
# MTUP v2 Training - Quick Command Reference
# Copy-paste từng block để chạy

echo "=================================================================="
echo "MTUP v2 TRAINING - QUICK COMMANDS"
echo "=================================================================="
echo ""

# ============================================================
# BƯỚC 1: PUSH CODE (trên Mac)
# ============================================================
echo "📌 BƯỚC 1: Push code lên git"
echo ""
echo "cd /Users/hagiang/ViSemPar_260110"
echo "git add mtup_v2/preprocessing/create_mtup_from_amr12.py"
echo "git add mtup_v2/scripts/train_mtup_higher_capacity.py"
echo "git add mtup_v2/scripts/diagnose_model.py"
echo "git add GUARANTEED_TRAINING_GUIDE.md"
echo "git add QUICK_COMMANDS.sh"
echo "git commit -m 'Fix: Unicode regex + verified training pipeline'"
echo "git push"
echo ""

# ============================================================
# BƯỚC 2: REGENERATE DATA (trên Server)
# ============================================================
echo "📌 BƯỚC 2: Pull và regenerate data trên server"
echo ""
echo "# ⚠️ THAY /path/to/ViSemPar_260110 BẰNG PATH THẬT"
echo "cd /path/to/ViSemPar_260110"
echo "git pull"
echo "mv data/train_mtup_unified.txt data/train_mtup_unified.txt.backup"
echo "python3 mtup_v2/preprocessing/create_mtup_from_amr12.py"
echo ""

# ============================================================
# BƯỚC 3: VERIFY DATA
# ============================================================
echo "📌 BƯỚC 3: Verify data không bị Mojibake"
echo ""
echo "# Check for proper Vietnamese (should see: Bạn là)"
echo "head -30 data/train_mtup_unified.txt | grep 'Bạn là'"
echo ""
echo "# Run full diagnosis"
echo "python3 mtup_v2/scripts/diagnose_model.py --data_path data/train_mtup_unified.txt --adapter_path dummy"
echo ""

# ============================================================
# BƯỚC 4: TRAIN
# ============================================================
echo "📌 BƯỚC 4: Train model với higher capacity"
echo ""
echo "# Clean old models"
echo "rm -rf outputs/mtup_260110/mtup_v2"
echo "rm -rf outputs/mtup_260110/mtup_v2_rank64"
echo ""
echo "# Start training (background)"
echo "nohup python3 mtup_v2/scripts/train_mtup_higher_capacity.py \\"
echo "    --data_path data/train_mtup_unified.txt \\"
echo "    --model_name Qwen/Qwen2.5-7B-Instruct \\"
echo "    --output_dir outputs/mtup_260110/mtup_v2_rank64 \\"
echo "    --epochs 20 > train_rank64.log 2>&1 &"
echo ""
echo "# Monitor training"
echo "tail -f train_rank64.log"
echo ""

# ============================================================
# BƯỚC 5: CHECK TRAINING PROGRESS
# ============================================================
echo "📌 BƯỚC 5: Check training progress"
echo ""
echo "# Check if still running"
echo "ps aux | grep train_mtup_higher_capacity"
echo ""
echo "# Check recent log"
echo "tail -50 train_rank64.log"
echo ""
echo "# Check epochs completed"
echo "grep 'Epoch' train_rank64.log | tail -5"
echo ""

# ============================================================
# BƯỚC 6: TEST MODEL
# ============================================================
echo "📌 BƯỚC 6: Test model sau khi train xong"
echo ""
echo "# Verify training completed"
echo "tail -20 train_rank64.log"
echo ""
echo "# Check files exist"
echo "ls -lh outputs/mtup_260110/mtup_v2_rank64/final_adapter/"
echo ""
echo "# Test prediction"
echo "python3 mtup_v2/scripts/debug_prediction.py \\"
echo "    --adapter_path outputs/mtup_260110/mtup_v2_rank64/final_adapter \\"
echo "    --test_sentence 'bi kịch là ở chỗ đó !'"
echo ""

# ============================================================
# BƯỚC 7: FULL PREDICTION
# ============================================================
echo "📌 BƯỚC 7: Run full prediction"
echo ""
echo "python3 mtup_v2/scripts/predict_mtup_unified.py \\"
echo "    --base_model Qwen/Qwen2.5-7B-Instruct \\"
echo "    --adapter_path outputs/mtup_260110/mtup_v2_rank64/final_adapter \\"
echo "    --input_file data/public_test.txt \\"
echo "    --output_file outputs/predictions_mtup_v2_rank64.txt"
echo ""
echo "# Verify output"
echo "head -5 outputs/predictions_mtup_v2_rank64.txt"
echo "wc -l outputs/predictions_mtup_v2_rank64.txt"
echo ""

# ============================================================
# TROUBLESHOOTING
# ============================================================
echo "=================================================================="
echo "🚨 TROUBLESHOOTING"
echo "=================================================================="
echo ""
echo "If data still has Mojibake after regenerate:"
echo "  # On Mac:"
echo "  scp data/train_amr_12.txt user@server:/path/to/ViSemPar_260110/data/"
echo ""
echo "  # On server:"
echo "  python3 mtup_v2/preprocessing/create_mtup_from_amr12.py"
echo ""
echo "=================================================================="
echo ""
echo "✅ Copy từng block commands ở trên để chạy!"
echo "✅ Read GUARANTEED_TRAINING_GUIDE.md để biết chi tiết!"
echo ""
