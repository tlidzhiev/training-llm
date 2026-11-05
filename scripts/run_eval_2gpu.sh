#!/usr/bin/env bash
# Second benchmark and the manual comparison on two GPUs:
#   1. MMLU (5-shot) for the base model and the fine-tuned runs, two lanes in parallel;
#   2. answers of the same models on 8 random MedQA test questions.
# Results: results/mmlu/<name>.json, results/manual_eval.md; logs: logs/eval/.
#
# Usage (inside tmux):  scripts/run_eval_2gpu.sh
set -uo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/mmlu logs/eval
CK=logs/training-llm

# name|model
MODELS=(
  "base|base"
  "full_ft|$CK/full_ft/checkpoints"
  "full_ft_lr5e-6|$CK/full_ft_lr5e-6/checkpoints"
  "lora_base|$CK/lora_base/checkpoints"
  "lora_tm_qv|$CK/lora_tm_qv/checkpoints"
)

lane() {
  local gpu=$1
  for i in "${!MODELS[@]}"; do
    (( i % 2 == gpu )) || continue
    local name=${MODELS[$i]%%|*}
    local model=${MODELS[$i]#*|}
    if [[ -f "results/mmlu/$name.json" ]]; then
      echo "[GPU $gpu] $name: already done, skipping"
      continue
    fi
    echo "[GPU $gpu] $(date +%T) start MMLU $name"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 uv run python -m lab.eval_mmlu \
      --model "$model" --output "results/mmlu/$name.json" > "logs/eval/mmlu_$name.log" 2>&1 \
      && echo "[GPU $gpu] $(date +%T) done  MMLU $name" \
      || echo "[GPU $gpu] $(date +%T) FAILED MMLU $name (see logs/eval/mmlu_$name.log)"
  done
}

lane 0 &
lane 1 &
wait

echo "$(date +%T) start manual evaluation"
manual_args=()
for entry in "${MODELS[@]}"; do
  manual_args+=("${entry%%|*}=${entry#*|}")
done
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 uv run python -m lab.manual_eval \
  --models "${manual_args[@]}" --num_examples 8 --output results/manual_eval.md \
  > logs/eval/manual_eval.log 2>&1 \
  && echo "$(date +%T) done  manual evaluation -> results/manual_eval.md" \
  || echo "$(date +%T) FAILED manual evaluation (see logs/eval/manual_eval.log)"

uv run python - <<'EOF'
import json, glob
print(f"{'model':16} {'MMLU':>7} {'medical':>8} {'other':>7} {'n':>6}")
for p in sorted(glob.glob('results/mmlu/*.json')):
    r = json.load(open(p))
    print(f"{p.split('/')[-1][:-5]:16} {r['accuracy']:7.4f} {r['accuracy_medical']:8.4f} "
          f"{r['accuracy_non_medical']:7.4f} {r['num_examples']:6}")
EOF
echo "=== EVALUATION FINISHED ==="
