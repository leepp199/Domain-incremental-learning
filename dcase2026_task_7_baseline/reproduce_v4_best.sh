#!/usr/bin/env bash
set -euo pipefail
###############################################################################
#  reproduce_v4_best.sh — 复现 V4 最佳结果 (Avg ≈ 57.9–58.2%)
#
#  V4 核心改进: 训练期间 BN running stats 保护，防止辅助损失前向传播
#              污染历史任务的 BatchNorm 统计量。
#
#  前提条件:
#    1. V2 的 D2 checkpoint 已存在于 checkpoints/BN_research_v2/
#    2. D1 checkpoint 存在于默认路径 (config_task7.py 中 save_resume_path)
#    3. 数据集已存在 (config_task7.py 中 audio_folder_DIL)
#
#  历史最佳结果 (v4_from_v2_train.log):
#    D2 after D2: 67.61%
#    D2 after D3: 53.36%  (V2=51.96, +1.40 改善)
#    D3:          63.03%  (V2=62.28, +0.75 改善)
#    Final Avg:   58.195% (V2=57.12, +1.075 改善)
#
#  注：由于训练存在随机性，每次重跑结果会在 ±1% 范围内波动。
#  典型结果范围: Avg 56.5–58.2%
#
#  用法:
#    bash reproduce_v4_best.sh train    # 步骤1: 训练 D3 (约45分钟, 需GPU)
#    bash reproduce_v4_best.sh eval     # 步骤2: 评估
#    bash reproduce_v4_best.sh all      # 训练+评估一步完成
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"

# ======================== 路径配置 ========================
V2_CKPT_DIR="${V2_CKPT_DIR:-$SCRIPT_DIR/checkpoints/BN_research_v2}"
V4_CKPT_DIR="${V4_CKPT_DIR:-$SCRIPT_DIR/checkpoints/BN_research_v4_best}"

# ======================== 超参数 (最佳配置) ========================
# 这些参数与 V2 相同，V4 的改善来自代码层面 (BN stats 保护)
PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE_SCRIPT="baseline/baseline_DIL_task7_v4.py"

COMMON_ARGS=(
  train
  --augmentation none
  --learning_rate 1e-4
  --batch_size 32
  --num_workers 8
  --epoch 120
  --resume
  --cuda
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

# ======================== 函数 ========================

do_train() {
  echo "======================================================"
  echo "  V4 训练: 从 V2 D2 checkpoint 出发, 仅训练 D3"
  echo "  V2 checkpoint: $V2_CKPT_DIR"
  echo "  V4 输出目录:   $V4_CKPT_DIR"
  echo "======================================================"

  # 准备 checkpoint 目录
  mkdir -p "$V4_CKPT_DIR"

  # 复制 V2 的 D2 checkpoint (V4 不修改 D2 参数)
  if [[ ! -f "$V4_CKPT_DIR/checkpoint_D2.pth" ]]; then
    echo "[准备] 复制 V2 D2 checkpoint..."
    cp "$V2_CKPT_DIR/checkpoint_D2.pth" "$V4_CKPT_DIR/"
  fi
  if [[ -f "$V2_CKPT_DIR/prototype_bank.pt" ]]; then
    cp "$V2_CKPT_DIR/prototype_bank.pt" "$V4_CKPT_DIR/"
  fi

  echo "[训练] 开始 D3 训练 (BN stats 保护已启用)..."
  echo ""

  "$PYTHON_BIN" "$BASELINE_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --resume_mode d1_d2 \
    --resume_checkpoint_dir "$V2_CKPT_DIR" \
    --save \
    --checkpoint_dir "$V4_CKPT_DIR" \
    --experiment_name train_d3_v4_best \
    "${HYBRID_ARGS[@]}"

  echo ""
  echo "[训练完成] checkpoint 保存在: $V4_CKPT_DIR"
}

do_eval() {
  echo "======================================================"
  echo "  V4 评估"
  echo "  Checkpoint: $V4_CKPT_DIR"
  echo "======================================================"

  if [[ ! -f "$V4_CKPT_DIR/checkpoint_D3.pth" ]]; then
    echo "错误: 未找到 D3 checkpoint, 请先运行训练"
    echo "  bash $0 train"
    exit 1
  fi

  "$PYTHON_BIN" "$BASELINE_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --resume_mode all \
    --resume_checkpoint_dir "$V4_CKPT_DIR" \
    "${HYBRID_ARGS[@]}"
}

# ======================== 主入口 ========================
case "$MODE" in
  train)
    do_train
    ;;
  eval)
    do_eval
    ;;
  all)
    do_train
    echo ""
    echo "============== 开始评估 =============="
    do_eval
    ;;
  *)
    echo "用法: bash $0 {train|eval|all}"
    exit 1
    ;;
esac
