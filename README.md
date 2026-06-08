# Qwen3-VL + Regression Head VLA 模型

> 基于 Qwen3-VL-Embedding-8B 的 VLA（Vision-Language-Action）模型，输出连续 7 维机械臂动作。  
> 数据集：RoboMIND 2.0 Franka（国产） | **欧氏距离误差：均值 12.6cm / 中位数 5.9cm**

---

## 1. 架构总览

```
多帧图像 (4帧) + 任务指令
        │
        ▼
┌─────────────────────────────┐
│  Qwen3-VL-Embedding-8B     │
│  ┌───────────────────────┐  │
│  │ Visual Encoder (冻结) │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │ LLM (LoRA rank=16)   │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │ last_hidden_state    │  │
│  │ → 4096-dim embedding │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  MLP Regression Head       │
│  4096 → 2048 → 1024 → 7   │
│  (LayerNorm + GELU)        │
└─────────────────────────────┘
        │
        ▼
  连续动作 [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

### 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 基座模型 | Qwen3-VL-Embedding-8B | 去掉了 LM Head，省 ~1.3GB 显存，架构更匹配回归任务 |
| 动作输出 | 连续回归（MSE） | 避免离散 token 化的精度损失和模式坍缩 |
| 时序建模 | 4 帧输入 | 从上次项目的单帧改进，模型能看到运动趋势 |
| 微调方式 | LoRA rank=16 + 4-bit | 4060 Ti 16GB 能跑 8B 模型 |
| 视觉编码器 | 冻结 | 已经能看懂图，不需要微调 |
| 回归头 | 全量训练 | 从零学习动作空间映射 |

### 与上次（失败）项目的对比

| 维度 | 上次（离散 token） | 本次（回归头） |
|------|----------|----------|
| 输出维度 | 2662 类离散 token | **连续 7 维浮点数** |
| Loss | Cross-entropy | **MSE** |
| 评估指标 | 完全匹配率（最高 10%） | **欧氏距离误差（中位数 5.9cm）** |
| 模式坍缩 | 严重 | ✅ **未出现** |

---

## 2. 数据集

**RoboMIND 2.0 Franka**（北京人形机器人创新中心 + 北京大学，ModelScope 托管）

| 属性 | 值 |
|------|------|
| 机器人 | Franka Emika Panda（7-DoF 单臂） |
| 动作维度 | 7（6D 末端姿态 + 1D 夹爪） |
| 数据格式 | HDF5（JPEG + 动作） |
| 国内源 | ✅ ModelScope 直链下载 |
| 总轨迹数 | 1,777 条（三任务合计） |
| 使用量 | 100 条/任务 × 3 任务 = 300 条 |
| 训练/验证 | 8266 / 919 样本 |

**三组任务：**
1. **抓取放置**（move_apple_from_plate_to_bowl）：502 条
2. **抽屉开关**（close_drawer_with_both_arms_simultaneously）：300 条
3. **推拨**（move_tape_to_another_basket）：975 条

数据集选型历程：
```
BridgeData V2（HF/代理不可用）→ LET-Base-Dataset（格式不匹配）
→ RoboMIND 2.0 Franka ✅
```

---

## 3. 训练结果

### 3.1 Loss 曲线

![Loss 曲线](training_curve.png)

### 3.2 Loss 数据

| Epoch | 训练 Avg Loss | 最佳验证 Loss |
|-------|---------------|---------------|
| 1 | 0.2686 | 0.1600 |
| 2 | 0.1130 | 0.1001 |
| 3 | 0.0643 | 0.0588 |
| 4 | 0.0380 | **0.0450** |

### 3.3 评估结果

**919 个验证样本，模型从 checkpoint-best 加载：**

| 指标 | 值 | 含义 |
|------|------|------|
| 均值欧氏距离误差 | **0.1255** | ~12.6cm |
| 中位数欧氏距离误差 | **0.0586** | ~5.9cm |
| dx MAE | **0.0049m** | ~5mm |
| dy MAE | 0.0134m | ~1.3cm |
| dz MAE | 0.0125m | ~1.2cm |
| droll MAE | 0.1109rad | ~6.4° |
| dpitch MAE | **0.0146rad** | ~0.8° |
| dyaw MAE | **0.0140rad** | ~0.8° |
| 夹爪准确率 | **99.89%** | 近乎完美 |

### 3.4 训练配置

```yaml
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
num_epochs: 4（早停）
warmup_ratio: 0.1
max_grad_norm: 1.0
mixed_precision: bf16
gradient_checkpointing: true
```

---

## 4. 代码结构

```
VLM_Robotics/
├── src/
│   ├── model/
│   │   ├── qwen3vl_robot.py      # VLA 模型封装
│   │   └── regression_head.py    # MLP 回归头
│   ├── data/
│   │   ├── dataset.py            # 多帧时序数据集
│   │   └── processing.py         # 动作归一化
│   └── train/
│       └── trainer.py            # Accelerate 训练循环
├── scripts/
│   ├── download_robomind.py      # 下载 RoboMIND 数据
│   ├── convert_robomind.py       # HDF5 → metadata
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # 评估入口
│   ├── plot_loss.py              # 画 Loss 曲线
│   └── run_train.sh              # 启动训练
├── configs/config.yaml           # 训练配置
└── model-embedding/              # 8B 权重
```

---

## 5. 运行指南

```bash
# 1. 环境
bash scripts/server_setup.sh

# 2. 数据下载
python scripts/download_robomind.py --max-episodes 100 --workers 3

# 3. 数据转换
python scripts/convert_robomind.py --data-dir data/raw/robomind --fps 5 --output data/processed/metadata.json

# 4. 训练
python scripts/train.py --config configs/config.yaml --metadata data/processed/metadata.json

# 5. 评估
python scripts/eval.py --checkpoint checkpoints/checkpoint-best --metadata data/processed/metadata.json --normalizer checkpoints/normalizer.json
```

---

## 6. 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.11 |
| 深度学习 | PyTorch 2.6, HuggingFace Transformers |
| 模型微调 | PEFT/LoRA, bitsandbytes 4-bit |
| 训练框架 | Accelerate（自定义循环） |
| 模型 | Qwen3-VL-Embedding-8B |
| 数据集 | RoboMIND 2.0 Franka（ModelScope） |
| 数据格式 | HDF5 |
| 硬件 | NVIDIA RTX 4060 Ti 16GB |

---

## 7. 后续改进

- **仿真部署**：将本策略部署到 MuJoCo / Isaac Sim 仿真环境，
  接入 ROS 接口进行闭环验证，对比离线评估与在线执行的效果差异
- **扩大数据量**：当前仅用 300 条/1,777 条可用轨迹
- **消融实验**：对比不同历史帧数的影响
