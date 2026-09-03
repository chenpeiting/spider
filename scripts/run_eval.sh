#!/usr/bin/env bash
# run_eval.sh — Multi-terrain evaluation wrapper
#
# Usage:
#   bash scripts/run_eval.sh <TASK> <CHECKPOINT> [MODE] [OUT_DIR] [NUM_EPISODES]
#
# Example:
#   bash scripts/run_eval.sh \
#       Isaac-Spider-Hybrid-v0 \
#       logs/skrl/spider_hybrid/2026-.../checkpoints/best_agent.pt \
#       hybrid \
#       logs/eval_hybrid \
#       100

TASK="${1:?Usage: run_eval.sh TASK CHECKPOINT [MODE] [OUT_DIR] [NUM_EP]}"
CKPT="${2:?Missing checkpoint path}"
MODE="${3:-pure_rl}"
OUT="${4:-logs/eval_${MODE}}"
N="${5:-100}"

SCRIPT="scripts/skrl/eval_terrain.py"

echo "======================================================"
echo "  Multi-terrain eval: $TASK  [$MODE]"
echo "  checkpoint : $CKPT"
echo "  out_dir    : $OUT"
echo "  episodes   : $N per terrain"
echo "======================================================"

for TERRAIN in flat noise stairs; do
    echo ""
    echo ">>> Terrain: $TERRAIN"
    python "$SCRIPT" \
        --task "$TASK" \
        --checkpoint "$CKPT" \
        --terrain "$TERRAIN" \
        --mode "$MODE" \
        --num_episodes "$N" \
        --out_dir "$OUT" \
        --headless
    if [ $? -ne 0 ]; then
        echo "[ERROR] $TERRAIN evaluation failed. Aborting."
        exit 1
    fi
done

echo ""
echo ">>> Generating comparison plots..."
python "$SCRIPT" \
    --plot_only \
    --out_dir "$OUT" \
    --mode "$MODE" \
    --task "$TASK"

echo ""
echo "======================================================"
echo "  Done! Results in: $OUT"
echo "======================================================"
