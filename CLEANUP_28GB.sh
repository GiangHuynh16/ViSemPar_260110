#!/bin/bash
################################################################################
# CLEANUP 28GB DIRECTORY
# Delete outputs/vlsp_amr_qwen_improved_v2/ to free 28GB
################################################################################

echo "========================================================================"
echo "🧹 CLEANUP 28GB DIRECTORY"
echo "========================================================================"
echo ""

TARGET_DIR="outputs/vlsp_amr_qwen_improved_v2"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Directory not found: $TARGET_DIR"
    echo ""
    echo "Nothing to delete."
    exit 1
fi

# Show current size
echo "Target directory:"
echo "  Path: $TARGET_DIR"
size=$(du -sh "$TARGET_DIR" 2>/dev/null | cut -f1)
echo "  Size: $size"
echo ""

# Show what's inside (first level only)
echo "Contents (top level):"
ls -lh "$TARGET_DIR" 2>/dev/null | head -20
echo ""

if [ $(ls "$TARGET_DIR" 2>/dev/null | wc -l) -gt 20 ]; then
    echo "... (showing first 20 items)"
    echo ""
fi

# Count files
file_count=$(find "$TARGET_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Total files: $file_count"
echo ""

echo "========================================================================"
echo "⚠️  WARNING"
echo "========================================================================"
echo ""
echo "This will PERMANENTLY DELETE:"
echo "  $TARGET_DIR/"
echo "  Size: $size"
echo "  Files: $file_count"
echo ""
echo "This directory appears to be an old training output."
echo "Safe to delete if you're not using it anymore."
echo ""

# Confirmation
read -p "Type 'DELETE' (all caps) to confirm: " confirmation

if [ "$confirmation" = "DELETE" ]; then
    echo ""
    echo "Creating backup log before deletion..."

    # Create log directory if needed
    mkdir -p outputs/deletion_logs

    # Save directory listing
    log_file="outputs/deletion_logs/deleted_vlsp_v2_$(date +%Y%m%d_%H%M%S).log"

    echo "Deletion Log - $(date)" > "$log_file"
    echo "Directory: $TARGET_DIR" >> "$log_file"
    echo "Size: $size" >> "$log_file"
    echo "Files: $file_count" >> "$log_file"
    echo "" >> "$log_file"
    echo "Contents:" >> "$log_file"
    ls -lRh "$TARGET_DIR" >> "$log_file" 2>&1

    echo "✓ Log saved to: $log_file"
    echo ""

    # Delete
    echo "Deleting directory..."
    echo "(This may take a minute for 28GB...)"

    rm -rf "$TARGET_DIR"

    if [ ! -d "$TARGET_DIR" ]; then
        echo ""
        echo "✅ SUCCESS!"
        echo ""
        echo "Deleted: $TARGET_DIR"
        echo "Space freed: ~$size"
        echo "Log: $log_file"
    else
        echo ""
        echo "❌ FAILED to delete directory"
        echo "Check permissions and try again"
        exit 1
    fi

else
    echo ""
    echo "❌ Deletion cancelled"
    echo "(You must type 'DELETE' in all caps to confirm)"
    exit 0
fi

echo ""
echo "========================================================================"
echo "DISK USAGE AFTER CLEANUP"
echo "========================================================================"
echo ""

# Show remaining outputs directories
echo "Remaining in outputs/:"
du -sh outputs/*/ 2>/dev/null | sort -hr | head -10

echo ""

# Show total
total_size=$(du -sh outputs 2>/dev/null | cut -f1)
echo "Total outputs/: $total_size"

echo ""
echo "========================================================================"
echo "✅ CLEANUP COMPLETE"
echo "========================================================================"
echo ""

echo "Space freed: ~28 GB"
echo ""
echo "Kept:"
echo "  ✅ outputs/checkpoints/ (9.4G) - 7B model"
echo "  ✅ outputs/checkpoints_mtup/ (945M) - 3B MTUP (primary)"
echo "  ✅ Training logs and results"
echo ""

echo "========================================================================"
