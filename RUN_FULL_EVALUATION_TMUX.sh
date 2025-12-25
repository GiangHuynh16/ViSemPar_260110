#!/bin/bash
################################################################################
# RUN FULL EVALUATION IN TMUX
# Chạy evaluation trong tmux session để tránh disconnect
################################################################################

session_name="mtup_eval"

echo "========================================================================"
echo "🚀 STARTING FULL EVALUATION IN TMUX"
echo "========================================================================"
echo ""
echo "Session name: $session_name"
echo ""

# Check if session already exists
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "⚠️  Tmux session '$session_name' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $session_name"
    echo "  2. Kill and restart: tmux kill-session -t $session_name && $0"
    echo ""
    exit 1
fi

# Create new tmux session and run evaluation
tmux new-session -d -s "$session_name" "bash RUN_FULL_EVALUATION.sh"

echo "✅ Evaluation started in tmux session: $session_name"
echo ""
echo "========================================================================"
echo "HOW TO MONITOR:"
echo "========================================================================"
echo ""
echo "1. Attach to session to see live progress:"
echo "   tmux attach -t $session_name"
echo ""
echo "2. Detach from session (keep it running):"
echo "   Press Ctrl+B, then D"
echo ""
echo "3. Check if still running:"
echo "   tmux ls | grep $session_name"
echo ""
echo "4. View current log output:"
echo "   tail -f outputs/evaluation_full_*.log"
echo ""
echo "========================================================================"
echo ""
