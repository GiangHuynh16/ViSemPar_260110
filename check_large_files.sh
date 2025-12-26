#!/bin/bash
# Script to check large files and disk usage in ViSemPar folder

echo "========================================================================"
echo "DISK USAGE CHECK FOR /mnt/nghiepth/giang/ViSemPar"
echo "========================================================================"
echo ""

# Check overall disk usage
echo "1. Overall disk space on /mnt/nghiepth:"
df -h /mnt/nghiepth | tail -1
echo ""

# Check folder sizes
echo "2. Top 20 largest folders (sorted by size):"
du -h /mnt/nghiepth/giang/ViSemPar/* 2>/dev/null | sort -hr | head -20
echo ""

# Check files larger than 500MB
echo "3. Files larger than 500MB:"
find /mnt/nghiepth/giang/ViSemPar -type f -size +500M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
echo ""

# Check files larger than 100MB
echo "4. Files larger than 100MB:"
find /mnt/nghiepth/giang/ViSemPar -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
echo ""

# Check outputs folder specifically
echo "5. Size of outputs/ folder:"
du -sh /mnt/nghiepth/giang/ViSemPar/outputs 2>/dev/null
echo ""

# Check cache folders
echo "6. Cache and temporary folders:"
du -sh /mnt/nghiepth/giang/ViSemPar/.cache 2>/dev/null
du -sh /mnt/nghiepth/giang/ViSemPar/__pycache__ 2>/dev/null
find /mnt/nghiepth/giang/ViSemPar -type d -name "__pycache__" -exec du -sh {} \; 2>/dev/null
echo ""

# Check HuggingFace cache
echo "7. HuggingFace cache (if exists):"
du -sh ~/.cache/huggingface 2>/dev/null
echo ""

# Summary
echo "========================================================================"
echo "RECOMMENDATIONS FOR CLEANUP:"
echo "========================================================================"
echo ""
echo "To delete old model checkpoints (keeps only latest):"
echo "  find outputs/checkpoints_* -name 'checkpoint-*' -type d | sort | head -n -1 | xargs rm -rf"
echo ""
echo "To delete Python cache:"
echo "  find /mnt/nghiepth/giang/ViSemPar -type d -name '__pycache__' -exec rm -rf {} +"
echo ""
echo "To delete HuggingFace cache (WARNING: will re-download models):"
echo "  rm -rf ~/.cache/huggingface/hub"
echo ""
echo "To check what's using disk space interactively:"
echo "  ncdu /mnt/nghiepth/giang/ViSemPar"
echo ""
