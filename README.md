# VLA-Robot-Learning: From Behavior Cloning to Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.9-green)](https://mujoco.org/)
[![SB3](https://img.shields.io/badge/SB3-PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

从零搭建的完整 **视觉-语言-动作（VLA）机器人学习管道**，覆盖数据下载、模型微调、仿真部署、以及基于 RL 的策略微调。面向 VLA 算法工程师岗位，展示从模型选型 → 架构设计 → 数据工程 → 训练迭代 → 仿真验证的完整能力栈。

---

## 目录

- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [安装与准备](#安装与准备)
- [路线一：行为克隆（BC）](#路线一行为克隆bc)
  - [数据准备](#1-数据准备)
  - [训练](#2-训练)
  - [评估](#3-评估)
  - [仿真推理](#4-仿真推理)
  - [BC 结果](#bc-结果)
- [路线二：强化学习（RL）](#路线二强化学习rl)
  - [环境设计](#1-环境设计)
  - [训练](#2-rl-训练)
  - [评估](#3-rl-评估)
  - [RL 结果](#rl-结果)
- [BC vs RL 对比](#bc-vs-rl-对比)
- [工程笔记（Lessons Learned）](#工程笔记lessons-learned)
- [后续方向](#后续方向)

---

## 项目结构

```
VLM_Robotics/
├── src/
│   ├── model/                # VLA 模型封装
│   │   ├── qwen3vl_robot.py  # Qwen3-VL + DualHead 回归头
│   │   └── regression_head.py
│   ├── data/                 # 多帧时序数据集
│   │   ├── dataset.py        # 滑动窗口 + 动作分块标签
│   │   └── processing.py     # z-score 动作归一化
│   ├── train/                # BC 训练器
│   │   └── trainer.py        # Accelerate 训练循环
│   └── rl/                   # RL 模块
│       ├── envs/              # MuJoCo 任务环境
│       │   ├── franka_env.py  # Franka Panda 物理仿真
│       │   ├── reach.py       # 末端到达任务
│       │   ├── push.py        # 物体推动任务
│       │   ├── pick_place.py  # 抓取放置任务
│       │   └── gym_wrapper.py # Gymnasium 接口封装
│       └── utils/
│           └── normalize.py
├── scripts/
│   ├── download_robomind.py   # 下载 RoboMIND 数据集
│   ├── convert_robomind.py    # HDF5 → metadata JSON
│   ├── train.py               # BC 训练入口
│   ├── eval.py                # BC 离线评估
│   ├── simulate.py            # MuJoCo 闭环仿真（BC 推理）
│   ├── visualize.py           # 预测 vs GT 动作可视化
│   ├── train_ppo.py           # RL PPO 训练（SB3）
│   ├── eval_ppo.py            # RL 评估 + GIF 录制
│   ├── collect_vla_demos.py   # VLA 蒸馏数据采集
│   ├── pretrain_bc.py         # BC 蒸馏预训练
│   ├── plot_rl_eval.py        # RL 评估报告图
│   └── plot_bc_rl_curve.py    # BC→RL 微调曲线
├── configs/                   # 训练配置文件
├── checkpoints/               # BC 模型检查点
├── rl_logs/                   # RL 模型与评估数据
├── viz/                       # 可视化（GIF + 轨迹图）
├── 内部资料/                   # 技术文档与实验报告
└── README.md
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.11 |
| 深度学习 | PyTorch 2.6, HuggingFace Transformers |
| 模型微调 | PEFT/LoRA (rank=16), bitsandbytes 4-bit (NF4) |
| BC 训练 | Accelerate + bf16 混合精度 |
| 基座模型 | Qwen3-VL-Embedding-8B |
| 数据集 | RoboMIND 2.0 Franka（国内，ModelScope） |
| 仿真器 | MuJoCo 3.9 |
| RL 框架 | Stable-Baselines3 (PPO) + Gymnasium |
| 硬件 | NVIDIA RTX 4060 Ti 16GB |

---

## 安装与准备

```bash
# 环境
pip install -r requirements.txt

# 基座模型（可选，也可自动从 HuggingFace 下载）
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B model-embedding/
```

---

## 路线一：行为克隆（BC）

基于 Qwen3-VL-Embedding-8B，用 LoRA + 4-bit 量化微调，输出连续的 7 维机械臂动作（末端位移 dx/dy/dz + 旋转 droll/dpitch/dyaw + 夹爪）。

### 1. 数据准备

```bash
# 下载 RoboMIND 2.0 Franka 数据集
python scripts/download_robomind.py --max-episodes 100 --workers 3

# HDF5 转为训练格式
python scripts/convert_robomind.py \
    --data-dir data/raw/robomind \
    --fps 5 \
    --output data/processed/metadata.json
```

### 2. 训练

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --metadata data/processed/metadata.json
```

### 3. 评估

```bash
python scripts/eval.py \
    --checkpoint checkpoints/checkpoint-best \
    --metadata data/processed/metadata.json \
    --normalizer checkpoints/normalizer.json
```

### 4. 仿真推理

```bash
MUJOCO_GL=egl python scripts/simulate.py \
    --checkpoint checkpoints/checkpoint-best \
    --steps 100 \
    --record trajectory.json
```

### BC 结果

训练经历了四轮迭代，最终确认了 MSE 回归架构的数据利用率天花板。

| 指标 | 迭代一（单步） | 迭代二（动作分块） | 迭代三（分头+噪声） | 迭代四（15×数据） |
|------|:---:|:---:|:---:|:---:|
| 均值 EU | 12.6 cm | 7.8 cm | **6.3 cm** | 8.2 cm |
| 中位数 EU | 5.9 cm | 3.4 cm | **3.1 cm** | 4.0 cm |
| dx MAE | 0.49 cm | 0.32 cm | **0.33 cm** | **0.29 cm** |
| dy MAE | 1.34 cm | 0.71 cm | **0.76 cm** | 0.79 cm |
| dz MAE | 1.25 cm | 0.72 cm | **0.78 cm** | **0.73 cm** |
| droll MAE | 6.4° | 4.0° | **3.2°** | 4.3° |
| dpitch MAE | 0.8° | 0.6° | **0.5°** | 0.7° |
| dyaw MAE | 0.8° | 0.5° | **0.4°** | 0.4° |
| 夹爪准确率 | 99.89% | 99.88% | **100.00%** | 99.95% |
| 数据量 | 9K | 9K | 9K | **129K** |

**迭代改进路径：**

- **迭代一**：单步回归基线，中位数误差 5.9cm
- **迭代二**：引入动作分块（horizon=10）+ 时序平滑 Loss，误差 ↓ 42% 至 3.4cm
- **迭代三**：夹爪二分类头 + 历史动作文本注入 + 高斯噪声增强，误差进一步降至 3.1cm，夹爪 100%
- **迭代四**：数据量 ×15（129K 样本），性能持平 → **确认 MSE 回归架构的数据利用率天花板**

> 详细实验报告见 `内部资料/实验报告.md`

---

## 路线二：强化学习微调（BC → RL）

在 BC 训练好的 VLA 模型（`checkpoint-best`）基础上，通过 **蒸馏 + PPO 微调** 实现在线策略优化：

1. **VLA 蒸馏**：将 BC 模型部署到 MuJoCo 仿真，采集 (状态, 动作) 对作为蒸馏数据
2. **BC 预训练**：用 VLA 的演示数据预训练快速 MLP 策略（NLL 损失），将 VLA 知识迁移到状态空间
3. **PPO 微调**：以蒸馏策略为起点，在 MuJoCo 中做在线微调

整个流程保持 VLA 作为核心，RL 作为在仿真中进一步优化策略的手段。

### 1. 环境设计

MuJoCo Reach 任务——控制 Franka 末端到达 3D 目标点（状态 25 维，动作 7 维）：

```
r = -2.0×dist + 2.0×Δdist + 阶段奖励(10+20) + 成功奖励(100)
```

### 2. BC → RL 完整管道

```bash
# Step 1: VLA 蒸馏（~4h）
python scripts/collect_vla_demos.py --episodes 300 --output data/vla_demos.npz

# Step 2: BC 预训练（~2min）
python scripts/pretrain_bc.py --data data/vla_demos.npz --output rl_logs/bc_pretrained.zip

# Step 3: PPO 微调（~5min）
python scripts/train_ppo.py --task reach --bc-pretrain rl_logs/bc_pretrained.zip --timesteps 200000

# Step 4: 评估 + 录制轨迹 GIF
MUJOCO_GL=egl python scripts/eval_ppo.py --task reach --model rl_logs/ppo_reach/best_model.zip --episodes 30 --record viz/rl_trajectory.gif
```

### 3. 结果

| 指标 | BC 蒸馏后（初始策略） | BC → RL 微调后 |
|------|:---:|:---:|
| 平均最优距离 | VLA 离线中位误差 3.1 cm | **MuJoCo 在线 0.13 m** |
| 训练时间 | 37 小时（BC 四轮） | + 蒸馏 4h + PPO 5min |
| 知识来源 | 12 万条专家轨迹 | VLA 蒸馏 + 在线探索 |
| 管道状态 | ✅ 全链路跑通 | ✅ VLA → 蒸馏 → PPO 微调闭环 |

### 4. 管线意义

```
BC 模型（VLA）→ 蒸馏（状态-动作对）→ MLP 策略 → PPO 微调 → 最终策略
```

VLA 作为核心算法提供视觉-动作映射先验，通过蒸馏迁移到快速推理的 MLP 策略后，
PPO 在仿真中做增量优化。整套管线验证了 **VLA → 蒸馏 → RL 微调** 的可行性，
为后续 World Model 或 Diffusion Policy 迁移提供了工程基础。

**收敛瓶颈**：Franka 模型的位置执行器存在约 0.17m 的稳态漂移，RL 策略在精密定位（< 3cm）上因物理精度受限而未能完全收敛。Push 任务中 EE 初始距物体 0.5m，训练未收敛。

---

## BC vs BC+RL 对比

BC 模型作为基础，RL 在蒸馏后的策略上做微调：

| 维度 | BC (迭代三) | BC → RL (蒸馏+PPO) |
|------|:---:|:---:|
| 模型规模 | 8B VLM + LoRA (~54M) | MLP 256×256 (~266K) |
| 输入 | 4 帧 RGB + 指令文本 | proprioceptive 状态向量 |
| 知识来源 | 12 万条专家轨迹 | VLA 蒸馏 + MuJoCo 自主探索 |
| 精度（离线） | 3.1 cm（中位 EU） | — |
| 精度（在线） | — | MuJoCo 在线 0.20 m |
| 训练时间 | 37 小时 | 蒸馏 4h + PPO 微调 5min |
| 核心优势 | 高精度，端到端视觉推理 | 在线自适应，不依赖专家持续标注 |

**管线逻辑：**
- BC 提供高质量的视觉-动作映射（VLA 基座）
- 蒸馏将 VLA 的知识迁移到快速推理的 MLP 策略
- RL 在仿真中基于奖励信号微调，提升对环境的适应性
- 三者形成完整的"模仿→蒸馏→优化"闭环

---

## 工程笔记（Lessons Learned）

### 1. 不要重复造轮子

在 RL 微调实现中，经历了从手写 PPO 到 Stable-Baselines3 的方案切换。手写版本（500+ 行代码）在 MuJoCo 物理仿真兼容性、Critic 初始化和设备管理上反复修复 9 个 bug 仍不稳定。切换到 SB3（60 行代码，加载蒸馏权重做微调）后训练全程稳定。

**教训**：基础设施类组件（RL 算法、环境接口、日志）一律用成熟开源库。只有核心创新点（如 VLA 的 DualHead 回归头）才手写。

### 2. 运动学模型 ≠ 物理仿真模型

原 `simulate.py` 中的 Franka XML 使用 `mj_forward`（纯运动学）运行正常，但切换到 `mj_step`（完整物理）后，位置执行器 kp=200 无法抵抗重力矩（~3 N·m），机械臂在伸展姿态下严重下垂。

**教训**：不能直接将运动学模型复用到物理仿真。物理仿真需要调校执行器参数（kp/kv）、添加沉降阶段、验证稳态漂移。

### 3. 无测试环境时用冒烟测试替代完整训练

由于本地 Windows 无 MuJoCo，代码写完直接丢服务器跑，导致维度错误（22→25）、设备不统一（CPU/GPU）、缺函数定义等问题都在服务器上才暴露。

**教训**：即使不能跑完整训练，至少要做一次 import + reset + step 的冒烟测试。Tensor 维度流转应逐层对过。

### 4. PPO 的 Critic 初始值极其关键

Critic 随机初始化 → 输出的优势估计全是噪声 → GAE 计算 δ 偏差严重 → 第一轮更新即灾难性遗忘（KL 散度飙至 0.94）。即使用预训练的 Actor，也会被有噪声的 Critic 破坏。

**教训**：PPO 训练前必须确保 Critic 的输出接近真实回报量级。在 SB3 中，内置的 Value Clipping 机制自动处理了这个问题。

---

## 后续方向（VLA 算法层面）

- **World Model + VLA**：引入视频预测模型（如 HunyuanVideo / CogVideo）作为世界模型，VLA 在预测的未来帧上做动作规划，实现模型驱动的 rollout 替代昂贵仿真
- **Diffusion Policy / Flow Matching**：迭代四已确认 MSE 回归天花板，将回归头替换为扩散头，用去噪过程建模多峰动作分布
- **多模态感知融合**：RoboMIND 提供 6 路 RGB + 深度图，探索 cross-attention 融合对动作预测精度的提升
- **动作表示优化**：对比末端位姿增量 vs 关节角度增量 vs SE(3) 流形上的动作表示
- **更大规模 VLA 基座**：Qwen3-VL-72B + 更大 LoRA rank + 更多数据，验证 scaling law 在机器人动作预测上的适用性

---

## License

MIT
