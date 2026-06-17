# Qwen3-VL + Regression Head VLA 模型

基于 **Qwen3-VL-Embedding-8B** 的 VLA（Vision-Language-Action）模型，输入 4 帧图像 + 任务指令，通过 DualHead MLP 回归头输出 10 步连续动作（动作分块 + Receding Horizon）。

数据集：**RoboMIND 2.0 Franka**（北京人形机器人创新中心 + 北京大学，ModelScope 直链），国产全链路。

---

## 四迭代演进

| 迭代 | 改进 | 中位数 EU | 关键发现 |
|------|------|-----------|----------|
| 一 | 单步回归（Baseline） | 5.9cm | 回归头 + MSE 解决了离散 token 的模式坍缩 |
| 二 | 动作分块 + 时序平滑 | 3.4cm (↓42%) | 一次预测 10 步消除了误差累积 |
| 三 | 分头输出 + 历史动作 + 高斯噪声 | 3.1cm (↓9%) | 夹爪二分类达 100%，姿态误差降 20% |
| 四 | 15× 数据规模扩展（129K 样本） | 持平 | **确认 MSE 回归数据天花板，架构转移至 Diffusion Policy** |

## 最终评估结果

| 指标 | 迭代一 | 迭代二 | 迭代三 | 迭代四 | 备注 |
|------|--------|--------|--------|--------|------|
| 中位数 EU | 5.9cm | 3.4cm | **3.1cm** | 4.0 | 归一化统计量因数据集不同而变化 |
| dx MAE | 0.49cm | 0.32cm | 0.33cm | **0.29cm** | 持续改进 |
| dz MAE | 1.25cm | 0.72cm | 0.78cm | **0.73cm** | 持续改进 |
| droll MAE | 6.4° | 4.0° | **3.2°** | 4.3° | 迭代四因数据分布变化回升 |
| 夹爪准确率 | 99.89% | 99.88% | **100.00%** | 99.95% | BCE 分头根本性解决 |
| 训练样本 | 9K | 9K | 9K | **129K** | 15 倍 |

详细数据和分析见 [实验报告.md](实验报告.md)，面试问答见 [面试QA.txt](面试QA.txt)。

## 架构

```
4 帧图像 (consecutive timesteps) + 任务指令
        │
        ▼
┌─────────────────────────────┐
│  Qwen3-VL-Embedding-8B     │
│  Visual Encoder (冻结)      │
│  LLM (LoRA rank=16, 4-bit) │
│  last_hidden_state → 4096   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  DualHead Regression        │
│  Pos: 4096→2048→1024→60    │ ← 10步 × 6维 (dx,dy,dz,dr,dp,dy)
│  Grip: 4096→2048→1024→10   │ ← 10步 × 1维 (BCE)
└─────────────────────────────┘
        │
    第 1 步执行 → 下一帧重新预测（Receding Horizon）
```

## 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 基座 | Qwen3-VL-**Embedding**（非 Instruct） | 无 LM Head，省 1.3GB，天然适配回归 |
| 输出 | 连续回归 MSE + BCE | 避免离散 token 化的模式坍缩 |
| 时序 | 4 帧 + 10 步动作分块 | VLM 感知运动 + Receding Horizon 消除累积误差 |
| 微调 | LoRA rank=16 + NF4 4-bit | 推理 ~5.5GB，训练 ~12GB（4060 Ti 16GB） |
| 数据集 | RoboMIND 2.0 Franka | 国产 + 7-DoF 匹配 + ModelScope 直链 |
| 训练 | 自定义 Accelerate 循环 | 完全控制 forward + loss，不依赖 LLaMA-Factory |

## 快速启动

```bash
# 1. 环境（Python 3.11, PyTorch 2.6, CUDA 12.x）
pip install -r requirements.txt

# 2. 下载 Qwen3-VL-Embedding-8B（需自行下载，不上传 GitHub）
#    放到 model-embedding/ 目录下

# 3. 数据下载（RoboMIND 2.0 Franka，三组任务各 200 条）
python scripts/download_robomind.py --max-episodes 200 --workers 3

# 4. HDF5 → metadata
python scripts/convert_robomind.py --data-dir data/raw/robomind --fps 5 \
    --max-files 600 --output data/processed/metadata.json

# 5. 训练
python scripts/train.py --config configs/config.yaml \
    --metadata data/processed/metadata.json

# 6. 评估（需要已训练的 checkpoint，不上传 GitHub）
python scripts/eval.py --config configs/config.yaml \
    --checkpoint checkpoints/checkpoint-best \
    --metadata data/processed/metadata.json \
    --normalizer normalizer.json

# 7. 仿真
MUJOCO_GL=egl python scripts/simulate.py \
    --checkpoint checkpoints/checkpoint-best --steps 100 --record trajectory.json
```

## 文档

| 文件 | 内容 |
|------|------|
| [实验报告.md](实验报告.md) | 四轮迭代完整数据、训练配置、对比分析 |
| [项目总结.txt](项目总结.txt) | 技术回顾 + 简历包装建议 |
| [面试QA.txt](面试QA.txt) | 30+ 面试问题及回答要点 |
| [动作分块设计.md](动作分块设计.md) | Action Chunking 设计思想 |
| [仿真部署方案.md](仿真部署方案.md) | MuJoCo 闭环仿真架构 |
| `normalizer.json` | Z-score 归一化统计量（eval/simulate 需要） |
| `training_curve.png` | 迭代四 Loss 曲线 |

## 技术栈

Python 3.11 · PyTorch 2.6 · HuggingFace Transformers · PEFT/LoRA · bitsandbytes 4-bit · Accelerate · Qwen3-VL-Embedding-8B · RoboMIND 2.0 · MuJoCo · ModelScope · HDF5

## License

MIT
