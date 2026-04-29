#!/usr/bin/env bash
set -euo pipefail

# V4 experiment script – anti-forgetting BN stats protection.
#
# Usage:
#   bash run_task7_experiment_v4.sh hybrid_train           # Train D2+D3 from scratch
#   bash run_task7_experiment_v4.sh hybrid_eval            # Eval from-scratch checkpoints
#   bash run_task7_experiment_v4.sh hybrid_train_from_v2   # Best: reuse V2 D2, retrain D3 with BN protection
#   bash run_task7_experiment_v4.sh hybrid_eval_from_v2    # Eval best V4 checkpoints (Avg=58.2%)
#   bash run_task7_experiment_v4.sh clean_ctrl_train
#   bash run_task7_experiment_v4.sh clean_ctrl_eval

MODE="${1:-hybrid_eval}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ----------------------------
# Paths
# ----------------------------
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-baseline/baseline_DIL_task7_v4.py}"
CLEAN_SCRIPT="${CLEAN_SCRIPT:-baseline/baseline_DIL_task7_clean_trainopt.py}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$WORKDIR/checkpoints/BN_research_v4}"
HYBRID_CHECKPOINT_DIR="${HYBRID_CHECKPOINT_DIR:-$CHECKPOINT_ROOT}"
CLEAN_CTRL_CHECKPOINT_DIR="${CLEAN_CTRL_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/clean_ctrl}"

# V4 best result: reuse V2's D2 checkpoint, only retrain D3 with BN protection
V2_CHECKPOINT_DIR="${V2_CHECKPOINT_DIR:-$WORKDIR/checkpoints/BN_research_v2}"
V4_FROM_V2_DIR="${V4_FROM_V2_DIR:-$WORKDIR/checkpoints/BN_research_v4_from_v2}"

# ----------------------------
# Common train/eval args
# ----------------------------
AUGMENTATION="${AUGMENTATION:-none}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-120}"
USE_CUDA="${USE_CUDA:-1}"

if [[ "$USE_CUDA" == "1" ]]; then
  CUDA_FLAG="--cuda"
else
  CUDA_FLAG=""
fi

COMMON_ARGS=(
  train
  --augmentation "$AUGMENTATION"
  --learning_rate "$LEARNING_RATE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epoch "$EPOCHS"
  --resume
)

HYBRID_ARGS=(
  --routing_mode hybrid
  --routing_temp 1.2
  --routing_topk 2
  --hybrid_entropy_weight 1.0
  --hybrid_conf_weight 0.35
  --hybrid_proto_weight 0.65
  --routing_loss_weight 0.03
  --consistency_loss_weight 0.03
  --pseudo_kd_weight 0.02
  --pseudo_kd_source all_prev
  --bn_clone_init
  --bn_clone_source prev
  --prototype_compact_weight 0.01
  --prototype_separation_weight 0.005
  --prototype_margin 8.0
  --tta_shifts 0.0,0.25,0.5
)

CLEAN_CTRL_ARGS=(
  --routing_mode soft
  --routing_temp 1.2
  --label_smoothing 0.05
  --rdrop_weight 0.0
  --rdrop_temperature 1.0
)

EXTRA_ARGS="${EXTRA_ARGS:-}"

run_cmd() {
  local script="$1"
  shift

  local cmd=("$PYTHON_BIN" "$script" "${COMMON_ARGS[@]}" "$@")
  if [[ -n "$CUDA_FLAG" ]]; then
    cmd+=("$CUDA_FLAG")
  fi

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extras=( $EXTRA_ARGS )
    cmd+=("${extras[@]}")
  fi

  echo "Working directory: $WORKDIR"
  echo "Mode: $MODE"
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  echo

  cd "$WORKDIR"
  "${cmd[@]}"
}

case "$MODE" in
  hybrid_train)
    run_cmd "$BASELINE_SCRIPT" \
      --resume_mode d1_only \
      --save \
      --checkpoint_dir "$HYBRID_CHECKPOINT_DIR" \
      --experiment_name train_d2d3_hybrid_proto \
      "${HYBRID_ARGS[@]}"
    ;;

  hybrid_eval)
    run_cmd "$BASELINE_SCRIPT" \
      --resume_mode all \
      --resume_checkpoint_dir "$HYBRID_CHECKPOINT_DIR" \
      "${HYBRID_ARGS[@]}"
    ;;

  # ---- Best result path: reuse V2 D2 checkpoint + V4 BN-protected D3 ----
  hybrid_train_from_v2)
    # Step 1: copy V2 D2 checkpoint
    mkdir -p "$V4_FROM_V2_DIR"
    if [[ ! -f "$V4_FROM_V2_DIR/checkpoint_D2.pth" ]]; then
      cp "$V2_CHECKPOINT_DIR/checkpoint_D2.pth" "$V4_FROM_V2_DIR/"
      echo "Copied V2 D2 checkpoint to $V4_FROM_V2_DIR"
    fi
    if [[ -f "$V2_CHECKPOINT_DIR/prototype_bank.pt" && ! -f "$V4_FROM_V2_DIR/prototype_bank.pt" ]]; then
      cp "$V2_CHECKPOINT_DIR/prototype_bank.pt" "$V4_FROM_V2_DIR/"
    fi
    # Step 2: train D3 only with V4 BN protection
    run_cmd "$BASELINE_SCRIPT" \
      --resume_mode d1_d2 \
      --resume_checkpoint_dir "$V2_CHECKPOINT_DIR" \
      --save \
      --checkpoint_dir "$V4_FROM_V2_DIR" \
      --experiment_name train_d3_v4_from_v2 \
      "${HYBRID_ARGS[@]}"
    ;;

  hybrid_eval_from_v2)
    run_cmd "$BASELINE_SCRIPT" \
      --resume_mode all \
      --resume_checkpoint_dir "$V4_FROM_V2_DIR" \
      "${HYBRID_ARGS[@]}"
    ;;

  clean_ctrl_train)
    run_cmd "$CLEAN_SCRIPT" \
      --resume_mode d1_only \
      --save \
      --checkpoint_dir "$CLEAN_CTRL_CHECKPOINT_DIR" \
      "${CLEAN_CTRL_ARGS[@]}"
    ;;

  clean_ctrl_eval)
    run_cmd "$CLEAN_SCRIPT" \
      --resume_mode all \
      --resume_checkpoint_dir "$CLEAN_CTRL_CHECKPOINT_DIR" \
      "${CLEAN_CTRL_ARGS[@]}"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    echo "Valid modes: hybrid_train, hybrid_eval, hybrid_train_from_v2, hybrid_eval_from_v2, clean_ctrl_train, clean_ctrl_eval" >&2
    exit 1
    ;;
esac
