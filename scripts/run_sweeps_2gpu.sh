#!/usr/bin/env bash
# Extra Full FT learning rates and the one-at-a-time LoRA sweeps (r, alpha, dropout,
# target_modules) on two GPUs: one run per GPU at a time, two lanes in parallel.
# The settings match the baseline runs (full_ft, lora_base), so the numbers are comparable.
#
# Usage (inside tmux):  scripts/run_sweeps_2gpu.sh
# Dry run:              DRY_RUN=1 scripts/run_sweeps_2gpu.sh
set -uo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/sweeps

COMMON="writer.mode=online training_args.gradient_checkpointing=false \
training_args.per_device_eval_batch_size=32 training_args.dataloader_num_workers=4"
LORA="lora=default model.dtype=bfloat16 training_args.learning_rate=2e-4 $COMMON"
FULL="$COMMON"

# name|arguments
RUNS=(
  "full_ft_lr5e-6|$FULL training_args.learning_rate=5e-6"
  "full_ft_lr2e-5|$FULL training_args.learning_rate=2e-5"
  "lora_r4|$LORA lora.r=4"
  "lora_r8|$LORA lora.r=8"
  "lora_r32|$LORA lora.r=32"
  "lora_r64|$LORA lora.r=64"
  "lora_alpha8|$LORA lora.lora_alpha=8"
  "lora_alpha16|$LORA lora.lora_alpha=16"
  "lora_alpha64|$LORA lora.lora_alpha=64"
  "lora_dropout0.0|$LORA lora.lora_dropout=0.0"
  "lora_dropout0.1|$LORA lora.lora_dropout=0.1"
  "lora_dropout0.2|$LORA lora.lora_dropout=0.2"
  "lora_tm_qv|$LORA lora.target_modules=[q_proj,v_proj]"
  "lora_tm_attn|$LORA lora.target_modules=[q_proj,k_proj,v_proj,o_proj]"
  "lora_tm_mlp|$LORA lora.target_modules=[gate_proj,up_proj,down_proj]"
  "lora_r64_alpha128|$LORA lora.r=64 lora.lora_alpha=128"
)

lane() {
  local gpu=$1
  for i in "${!RUNS[@]}"; do
    (( i % 2 == gpu )) || continue
    local name=${RUNS[$i]%%|*}
    local args=${RUNS[$i]#*|}
    if [[ -n "${DRY_RUN:-}" ]]; then
      echo "[GPU $gpu] $name: $args"
      continue
    fi
    if [[ -f "logs/training-llm/$name/checkpoints/all_results.json" ]]; then
      echo "[GPU $gpu] $name: already finished, skipping"
      continue
    fi
    echo "[GPU $gpu] $(date +%T) start $name"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 uv run python -m lab.finetune \
      writer.run_name="$name" override=true $args > "logs/sweeps/$name.log" 2>&1 \
      && echo "[GPU $gpu] $(date +%T) done  $name" \
      || echo "[GPU $gpu] $(date +%T) FAILED $name (see logs/sweeps/$name.log)"
  done
}

lane 0 &
lane 1 &
wait
echo "=== ALL SWEEPS FINISHED ==="
