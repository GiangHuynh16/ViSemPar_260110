#!/bin/bash
################################################################################
# CHECK EVALUATION STATUS
# Kiểm tra tiến độ evaluation đang chạy
################################################################################

echo "========================================================================"
echo "📊 EVALUATION STATUS CHECK"
echo "========================================================================"
echo ""

# Check if tmux session exists
if tmux has-session -t mtup_eval 2>/dev/null; then
    echo "✅ Evaluation is running in tmux session: mtup_eval"
    echo ""
else
    echo "❌ No tmux session 'mtup_eval' found"
    echo ""
    echo "Check if evaluation process is running:"
    if pgrep -f "evaluate_mtup_model.py" > /dev/null; then
        echo "  ✅ Found evaluate_mtup_model.py process"
    else
        echo "  ❌ No evaluation process found"
    fi
    echo ""
fi

# Find latest log file
latest_log=$(ls -t outputs/evaluation_full_*.log 2>/dev/null | head -1)

if [ -n "$latest_log" ]; then
    echo "========================================================================"
    echo "📋 LATEST LOG: $latest_log"
    echo "========================================================================"
    echo ""

    # Show last 30 lines
    echo "Last 30 lines of log:"
    echo "------------------------------------------------------------------------"
    tail -30 "$latest_log"
    echo "------------------------------------------------------------------------"
    echo ""

    # Try to extract progress if available
    if grep -q "Generating:" "$latest_log"; then
        echo "Progress indicators found in log:"
        grep "Generating:" "$latest_log" | tail -1
    fi

    if grep -q "Evaluating:" "$latest_log"; then
        grep "Evaluating:" "$latest_log" | tail -1
    fi
    echo ""

    # Check if completed
    if grep -q "EVALUATION COMPLETE" "$latest_log"; then
        echo "✅ Evaluation has COMPLETED!"
        echo ""

        # Show results
        if grep -q "SMATCH SCORES" "$latest_log"; then
            echo "========================================================================"
            echo "📊 RESULTS:"
            echo "========================================================================"
            grep -A 10 "SMATCH SCORES" "$latest_log"
            echo ""
        fi

        # Show results file
        latest_results=$(ls -t outputs/evaluation_results_full_*.json 2>/dev/null | head -1)
        if [ -n "$latest_results" ]; then
            echo "Results file: $latest_results"
            echo ""
            echo "View full results:"
            echo "  cat $latest_results | python3 -m json.tool"
            echo ""
        fi
    else
        echo "⏳ Evaluation still in progress..."
        echo ""
    fi
else
    echo "❌ No evaluation log files found"
    echo ""
    echo "Expected location: outputs/evaluation_full_*.log"
    echo ""
fi

echo "========================================================================"
echo "MONITORING COMMANDS"
echo "========================================================================"
echo ""
echo "Attach to running evaluation:"
echo "  tmux attach -t mtup_eval"
echo ""
echo "Watch log in real-time:"
echo "  tail -f outputs/evaluation_full_*.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Kill evaluation (if needed):"
echo "  tmux kill-session -t mtup_eval"
echo "  # OR"
echo "  pkill -f evaluate_mtup_model.py"
echo ""
echo "========================================================================"
