# DCASE2026 Task7 V4 Baseline: Anti-Forgetting Incremental Learning

## 简介
本仓库为DCASE 2026 Task7（音频分类领域增量学习）V4基线系统，核心目标是：
- 显著缓解D2在D3训练后遗忘问题
- 保持或提升平均准确率
- 代码结构清晰，便于复现和扩展

V4在V2基础上，**新增了BN统计量保护机制**，有效防止辅助损失前向传播时污染历史任务的BN running_mean/var，大幅提升D2-after-D3准确率。

---

## 目录结构
```
dcase2026_task7_baseline/
├── README_v4.md                # 本文件
├── requirements.txt            # 依赖包
├── run_task7_experiment_v4.sh  # V4主训练/评估脚本
├── reproduce_v4_best.sh        # 一键复现V4最佳结果脚本（可选）
├── baseline/
│   ├── baseline_DIL_task7_v4.py  # V4主训练/推理逻辑
│   └── domain_net_v4.py          # V4模型结构（含BN保护、SE、Adapter等）
└── utils/
    ├── config_task7.py           # 数据、路径、类别等配置
    ├── datasetfactory_task7.py   # 数据加载
    ├── utilities.py              # 工具函数
    └── chunking.py               # 音频分块
```

---

## 主要特性
- **BN stats保护**：训练时仅当前任务BN为train，其余全部eval，防止辅助损失污染历史BN统计量。
- **BN快照/恢复**：每个任务训练后可保存/恢复BN running stats，进一步防止意外污染。
- **原型刷新Eval**：刷新原型时强制eval，避免BN统计量被推理流动改变。
- **辅助损失**：支持prototype compact/separation、KD、routing consistency等，均不会破坏历史BN。
- **高可复现性**：严格分离V2/V4代码，参数、数据、权重路径均可配置。

---

## 快速开始
### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据准备
- 下载官方数据集，目录结构如下：
```
task7_data/
├── audio/
└── evaluation_setup/
    ├── development_train.txt
    └── development_test.txt
```
- 修改utils/config_task7.py中的audio_folder_DIL、output_folder为你的实际路径。

### 3. 训练与评估
#### 训练D2+D3（推荐）
```bash
bash run_task7_experiment_v4.sh hybrid_train
```
#### 仅评估（需已有V4权重）
```bash
bash run_task7_experiment_v4.sh hybrid_eval
```
#### 复现最佳结果（D2用V2权重，D3用V4训练）
```bash
bash reproduce_v4_best.sh all
```

---

## checkpoint说明
- 默认权重路径：checkpoints/BN_research_v4/
    - checkpoint_D2.pth
    - checkpoint_D3.pth
    - prototype_bank.pt
- 可通过环境变量或config_task7.py自定义权重路径

---

## 主要改进点（与V2对比）
1. BN stats保护，极大缓解遗忘
2. BN快照/恢复机制
3. 原型刷新Eval
4. 路由/辅助损失更鲁棒

---

## 参考命令
```bash
# 训练+评估一体
bash run_task7_experiment_v4.sh hybrid_train
bash run_task7_experiment_v4.sh hybrid_eval

# 复现最佳（D2用V2权重，D3用V4训练）
bash reproduce_v4_best.sh all
```

---



---

如有问题请联系维护者或在GitHub提issue。
