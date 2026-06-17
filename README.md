# VLA-Qwen3VL-Embedding-Regression

基于 Qwen3-VL-Embedding-8B 和 RoboMIND 2.0 数据集的 VLA（Vision-Language-Action）机器人操控模型。输入多帧图像与任务指令，通过 DualHead MLP 回归头直接输出 10 步连续动作序列，在 Franka Emika Panda 机械臂上实现厘米级预测精度。

**四轮迭代从基线 12.6cm 降至 3.1cm（↓75%），并实证了 MSE 回归在 129K 样本规模下的数据利用率上限。**

---

## 架构

```
4 帧连续图像 + 任务指令
        │
        ▼
┌─────────────────────────────────┐
│  Qwen3-VL-Embedding-8B          │
│  ├─ Visual Encoder (冻结)        │
│  └─ LLM (LoRA rank=16, 4-bit)  │
│  last_hidden_state → 4096        │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  DualHead Regression Head        │
│  ├─ Position: 4096→2048→1024→60 │  10步 × 6维 (dx,dy,dz,dr,dp,dy)
│  └─ Gripper:  4096→2048→1024→10 │  10步 × 1维 (Sigmoid + BCE)
└─────────────────────────────────┘
        │
    Receding Horizon: 执行第1步，下一帧重新预测
```

## 四轮迭代

| 迭代 | 核心改进 | 中位数误差 | 关键结论 |
|------|---------|:--------:|----------|
| 一 | 单步回归，替代离散 token 方案 | 5.9 cm | 回归头 + MSE 根治模式坍缩 |
| 二 | 动作分块 (10步) + 时序平滑 Loss | 3.4 cm (↓42%) | Receding Horizon 消除误差累积 |
| 三 | 夹爪二分类 + 历史动作注入 + 高斯噪声 | 3.1 cm (↓9%) | 夹爪 100% 准确，姿态误差降 20% |
| 四 | 数据量 15× 扩展 (129K samples) | 持平 | MSE 回归触及数据天花板 → 下一步 Diffusion Policy |

## 最终性能

| 指标 | 数值 |
|------|:----:|
| 位置 MAE (dx/dy/dz) | 0.29 / 0.79 / 0.73 cm |
| 姿态 MAE (roll/pitch/yaw) | 4.3° / 0.7° / 0.4° |
| 夹爪准确率 | 99.95% |
| 训练样本 | 129,083 (599 条 Franka 轨迹) |
| 参数量 | LoRA 21M 可训练 / 8B 总量化 |

![Loss Curve](training_curve.png)

## 工程亮点

**模型设计** — 从零设计的 DualHead 回归架构，位置/姿态回归 + 夹爪二分类分离，每层 LayerNorm + GELU + Dropout。Embedding 版 Qwen3-VL 去 LM Head 省 1.3GB 显存，适配回归任务。

**训练管线** — 自定义 Accelerate 训练循环替代 LLaMA-Factory 黑盒，完全控制 forward / loss。支持断点续训（optimizer / scheduler / global_step 完整存档），eval 裁剪至 100 batch 加速验证。

**数据工程** — RoboMIND 2.0 Franka 国产数据集，HDF5 解析 + 多线程并行下载。Z-score 动作归一化 + 反归一化评估，支持 6 路 RGB 相机切换。

**仿真闭环** — MuJoCo + DLS IK 求解器，VLA 推理 → 末端位姿 → IK → 关节角度 → 执行 → 渲染，完整的仿真验证链路。

**显存优化** — NF4 4-bit 量化 + LoRA rank=16，推理 ~5.5GB，训练峰值 ~12GB，单卡 RTX 4060 Ti 16GB 可完成 8B 模型全流程训练。

## 快速开始

```bash
# 环境
pip install -r requirements.txt

# 下载基座模型（不上传 GitHub，需自行获取后放入 model-embedding/）
# Qwen/Qwen3-VL-Embedding-8B

# 下载数据
python scripts/download_robomind.py --max-episodes 200 --workers 3

# 转换格式
python scripts/convert_robomind.py --data-dir data/raw/robomind --fps 5 \
    --max-files 600 --output data/processed/metadata.json

# 训练
python scripts/train.py --config configs/config.yaml \
    --metadata data/processed/metadata.json

# 评估（需已训练权重）
python scripts/eval.py --checkpoint checkpoints/checkpoint-best \
    --metadata data/processed/metadata.json --normalizer normalizer.json

# 仿真
MUJOCO_GL=egl python scripts/simulate.py \
    --checkpoint checkpoints/checkpoint-best --steps 100
```

## 文件结构

```
├── src/model/          VLA 模型定义 (Qwen3-VL + DualHead)
├── src/data/           数据集加载、归一化、collate
├── src/train/          训练器 (Accelerate、断点续训、CSV 日志)
├── scripts/            训练 / 评估 / 数据下载 / 仿真 / 可视化
├── configs/            训练超参
├── 实验报告.md          四轮迭代完整数据与分析
├── normalizer.json     Z-score 归一化统计量
└── loss_log.csv        迭代四训练日志
```

模型权重与原始数据不上传 GitHub（`.gitignore` 排除 `checkpoints/`、`data/`、`model-embedding/`）。

## 技术栈

Python 3.11 · PyTorch 2.6 · Transformers · PEFT/LoRA · bitsandbytes (NF4) · Accelerate · Qwen3-VL-Embedding-8B · RoboMIND 2.0 · MuJoCo · ModelScope · HDF5

## License

MIT
