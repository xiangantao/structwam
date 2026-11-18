# StructWAM

**把视频生成模型里蒸馏出的“结构”，作为第三路专家接进世界-动作模型（World-Action Model），提升机器人策略在各种扰动下的鲁棒性。**

一句话：几何结构管大动作，视频想象管精细的难点，动作分支按接触阶段决定听谁的。全程只在潜空间里工作，不生成像素。

---

## 1. 项目在做什么

### 出发点

[OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) 这类世界-动作模型，让一个视频 DiT 和一个动作 DiT 在每一层做联合注意力：模型一边“想象”未来画面，一边预测动作。它在分布内表现很好，但换背景、换相机视角、改机器人初始位姿、加传感器噪声时，成功率会明显下降（见 LIBERO-plus 基准）。

我们的假设：如果策略能拿到一份**对外观扰动不敏感的场景结构**，比如哪些区域是物体、关键点往哪里运动、指令里的词落在画面哪里，动作预测就不必完全依赖像素外观，鲁棒性会更好。

### 结构从哪里来

结构来自我们之前的工作（`heft/`）：从视频生成模型 **Wan2.1-VACE** 蒸馏出一个**结构学生**，主干是 V-JEPA2。它对每个相机输出：

- 区域图（K 个分段，形成场景的无监督分割）；
- 关键点轨迹（现在到 +16 帧）；
- V-JEPA2 主干特征；
- 指令内容词的注意力热图（“选词模式”：按热图集中度保留最多 5 个词，由模型自己判断哪个是目标、哪个是目的地）。

学生在 StructWAM 里**完全冻结**，通过 `heft/latent_policy/src/latent_dp` 里的 `OnlineMaps` 在线计算。

### StructWAM 的结构：三路专家

| 分支 | 实现 | 内容 |
|---|---|---|
| 视频 | Cosmos-Predict2.5-2B（机器人后训练权重 `robot/action-cond`） | 只在潜空间里训练；推理时只去噪 1 步，从不解码 |
| 动作 | OpenWAM 的 ActionDiT | 32 步动作块，统一 80 维动作空间 |
| 结构 | 冻结的 VACE 学生 + 可训练的结构专家 | 每个相机：64 个图 token、32 个关键点轨迹 token、16 个主干 token、最多 5 个词 token |

三路 token `[视频 | 动作 | 结构]` 在视频 DiT 的每一层做一次联合注意力。骨架沿用 OpenWAM 的三系统 MoT，只是把 VLM 理解专家换成了结构专家。

### 结构如何发挥作用

1. **阶段门控（结构 → 动作）**：门控预测“这个动作块离接触有多近”，监督信号来自示范里的夹爪指令。它给动作 token 的注意力加偏置：离接触远时多看结构，接近接触时多看视频，尤其是腕部画面所在的区域。推理时视频分支只去噪 1 步，之后的动作步复用缓存的视频 / 结构 K/V（见第 2 节）。
2. **训练约束（结构 → 视频）**：
   - 视频损失按显著区域和样本难度（接近接触、轨迹没把握）加权；
   - 轨迹一致性：预测出的下一潜帧在学生轨迹指向的位置上，要和当前帧外观一致；
   - REPA 式对齐：视频 DiT 中间层特征对齐学生的 V-JEPA2 主干。
3. **结构读视频（视频 → 结构）**：结构 token 可以读视频 token，用世界模型的知识修正远期轨迹。
4. **视频损失也训练结构专家**：默认 `video_detach: []`。消融时可以设成 `[structure]`，切断这条梯度。

**当前最佳配置 b2**（`StructWAM/configs/train_struct_b2.yaml`）启用的机制：

| 机制 | 配置 | b2 |
|---|---|---|
| 阶段门控 + 注意力偏置 | `routing.enabled: true`，`lambda_phase: 0.1`，`gate_teacher_forcing: 0.5` | 开 |
| 视频损失按显著区域加权 | `video_saliency: 2.0` | 开 |
| 视频损失按样本难度加权 | `difficulty: [1.0, 0.5, 0.0]` | 接近接触、轨迹没把握两项开 |
| 轨迹一致性 | `lambda_track` | **关**（0） |
| REPA 式对齐 | `lambda_align: 0.5` | 开 |
| 结构读视频 | `structure_reads_video: true` | 开 |
| 视频损失训练结构专家 | `video_detach` | **关**（`[structure]`） |
| 视频 dropout | `action_video_dropout` | **0.3**：训练时 30% 的样本，动作和结构 token 看不到视频 token |
| 一致图像增强 | `dataloader.structure_augment.enabled` | **开**：头部旋转 ±4°、放大 1–1.2 倍、20% 概率轻微模糊；两个相机亮度 / 对比度 / 饱和度抖动；视频画布、首帧条件和学生的 33 帧历史用同一套变换 |
| 推理：视频一步去噪 + K/V 缓存 | `inference.video_steps_far/near: 1`，`inference.video_kv_cache: true` | **开**（模型配置的默认值） |

完整的设计和改动清单见 [`StructWAM/STRUCTWAM.md`](StructWAM/STRUCTWAM.md)。

### 昇腾 NPU 移植

全部训练都在 8×Ascend 910B3 上完成：

- **transformer-engine**：用纯 PyTorch 的 shim 替代（`StructWAM/npu_shims`），在 NPU 上自动调用 torch_npu 融合算子。
- **修正了 OpenWAM 的一个缺陷**：Cosmos-Predict2.5 漏了上游的 `timestep_scale=0.001`，网络看到的时间步大了 1000 倍，预训练的去噪能力用不上。已补上。
- **float32 策略**：用“float32 计算岛”实现，不依赖 NPU 不支持的 float32 autocast。
- **联合注意力按行分组**：视频行和结构行走融合注意力，只有动作行带门控偏置。结果和整段浮点掩码一致，单层快 3.7 倍。

---

## 2. 实验结果

### 评测协议（LIBERO-plus）

- 用 OpenWAM 的 LIBERO-plus 评测代码（`StructWAM/benchmarks/libero-plus`），它在启动时固定并检查协议：每个任务 1 次，seed 10000，`rng_mode: official_global`，settle 30 步，每个 episode 最多 600 个环境步（libero_10 为 700）。OpenWAM 论文里 OpenWAM-α 的数字也是这套协议。
- 使用干净指令：LIBERO-plus 的指令由 bddl 文件名生成，非语言类任务会带上扰动编号（如 “… table 18”），评测时去掉这些编号；语言扰动任务的指令保持原样（`recipes/eval/clean_instructions.py`）。
- 动作去噪 10 步，每次规划执行 10 个动作，DiT 缓存关闭。
- 测评是确定性的：同一检查点的两次测评里，同一个任务的成败完全一致（全量和 10% 抽样重叠的 1004 个任务全部一致）。

### 全量 LIBERO-plus（10030 个任务，b2 40k，视频 1 步 + K/V 缓存）

**总成功率 85.5%**（8573 / 10030）。逐任务结果：`results/libero_plus_full_b2_v1kv/<suite>/summary.csv`。

| 扰动类别 | libero_spatial | libero_object | libero_goal | libero_10 | 合计 |
|---|---|---|---|---|---|
| 相机视角 | 68.1 | 72.5 | 65.4 | 52.5 | 64.4 |
| 机器人初始状态 | 92.0 | 92.7 | 86.6 | 85.2 | 89.0 |
| 语言指令 | 96.4 | 92.1 | 76.1 | 95.0 | 89.7 |
| 光照 | 97.9 | 98.7 | 100.0 | 96.0 | 98.2 |
| 背景纹理 | 89.5 | 92.7 | 78.3 | 65.1 | 80.8 |
| 传感器噪声 | 94.6 | 92.2 | 91.0 | 88.6 | 91.4 |
| 物体布局 | 96.9 | 89.8 | 71.3 | 93.9 | 87.3 |
| **合计** | **90.6** | **89.6** | **80.3** | **81.8** | **85.5** |

### 标准 LIBERO（4 个 suite × 10 个任务 × 50 次，同一个 b2 检查点）

**总成功率 98.9%**（1978 / 2000）。汇总：`results/libero_b2_v1kv/summary.json`。

| suite | 成功 / 次数 | 成功率 | 没到 100% 的任务 |
|---|---|---|---|
| libero_spatial | 494 / 500 | 98.8 | task 4：49/50，task 5：45/50 |
| libero_object | 500 / 500 | 100.0 | — |
| libero_goal | 496 / 500 | 99.2 | task 1：48/50，task 2：48/50 |
| libero_10 | 488 / 500 | 97.6 | task 1：49/50，task 6：47/50，task 8：45/50，task 9：47/50 |
| **合计** | **1978 / 2000** | **98.9** | |

- 协议：OpenWAM 的 LIBERO 评测代码（`StructWAM/benchmarks/libero`）：每个任务 50 次，seed 42，`rng_mode: environment`，settle 30 步，每个 episode 最多 600 个环境步（libero_10 为 700），动作去噪 10 步、每次规划执行 10 个动作。
- 为了加速，每个任务的 50 次被拆成 10 个 5 次的块，放进一个共享队列，由 32 个服务（8 卡 × 每卡 4 个，快速模式）并行跑。拆块前先用 OpenWAM 的 `benchmarks/libero/eval_manifest.py` 给每个任务生成一份密封的 manifest，固定每一次 trial 开始前的随机状态，所以分块跑的结果和 50 次连续跑完全相同。2000 个 episode 用时约 3 小时。
- 参考：LIBERO-plus 论文表 1 的“Original”列（未加扰动的 LIBERO）给出 OpenVLA-OFT 97.1、RIPT-VLA 97.5、UniVLA 95.2、π0 94.2、π0-FAST 85.5。这些是各自论文的设置，步数上限等细节不一定和我们一致。

### 和其他方法比较

OpenWAM 论文的 LIBERO-plus 表（`StructWAM/benchmarks/libero-plus/README.md` 第 4 节），列为相机 / 机器人 / 语言 / 光照 / 背景 / 噪声 / 布局 / 平均：

| 方法 | 类型 | 相机 | 机器人 | 语言 | 光照 | 背景 | 噪声 | 布局 | 平均 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-RobotManip | VLA | 87.2 | 75.5 | 85.6 | 96.6 | 97.7 | 97.7 | 87.3 | 89.0 |
| ACoT-VLA | VLA | 72.6 | 82.6 | 87.5 | 97.7 | 96.5 | 87.8 | 88.1 | 86.6 |
| **StructWAM b2（本仓库）** | WAM | 64.4 | 89.0 | 89.7 | 98.2 | 80.8 | 91.4 | 87.3 | **85.5** |
| π0.5 | VLA | 78.4 | 73.6 | 80.8 | 96.2 | 94.1 | 89.0 | 84.5 | 84.4 |
| ABot-M0.5 | WAM | 70.5 | 87.4 | 88.6 | 94.0 | 89.7 | 75.5 | 85.2 | 83.4 |
| ImageWAM | WAM | 80.8 | 50.3 | 91.4 | 98.1 | 85.5 | 93.8 | 80.5 | 83.1 |
| Cosmos-Policy | WAM | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | 82.2 |
| Being-H0.7 | WAM | 82.0 | 59.0 | 82.8 | 97.8 | 90.0 | 93.5 | 88.5 | 82.1 |
| OpenVLA-OFT | VLA | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| OpenWAM-α | WAM | 33.8 | 76.1 | 88.0 | 97.0 | 87.1 | 39.8 | 77.5 | 69.2 |
| π0 | VLA | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |

- 和 OpenWAM-α 是同一套评测代码和协议，高 16.3 个百分点。
- 表里其他方法的数字取自各自论文，评测细节（步数上限、指令处理）不一定和我们一致，比较时要注意。LIBERO-plus 官方论文和 README 都没写每个 episode 的步数上限；要上排行榜，先核对 LIBERO-plus 自己的评测代码。
- 强项是机器人初始状态（89.0，表里最高）和语言；短板是相机视角（64.4）和背景纹理（80.8），libero_10 和 goal 上尤其明显。

### 配方是怎么选出来的（LIBERO-plus 10% 抽样，1005 个任务）

早期消融都在 10% 抽样上比较（task-sample-seed 42），各 40k 步；“10/10”是视频和动作都去噪 10 步的旧推理设置。

| 变体 | 视频损失训练结构专家 | 轨迹一致性 | 视频 dropout | 图像增强 | 抽样成功率 |
|---|---|---|---|---|---|
| OpenWAM-Cosmos 40k（不带结构） | — | — | — | — | 71.2 |
| StructWAM 原版 | 是 | 开 | — | — | 75.2 |
| a2 | 否 | 关 | 0.3 | — | 76.3 |
| b0 | 是 | 开 | 0.3 | 默认 | 78.7 |
| c0（左右并排画布 + π0.5 式增强） | 是 | 开 | 0.3 | π0.5 式 | 79.2 |
| **b2**（10/10 推理） | 否 | 关 | 0.3 | 默认 | 80.5 |
| **b2 + 视频 1 步 + K/V 缓存** | 否 | 关 | 0.3 | 默认 | **84.6**（对 b2 10/10 配对 74:33，p < 0.001） |

视频只去噪 1 步不但快 1.7 倍，还更好：增益主要来自传感器噪声里的高斯模糊（92 vs 57）。一种解释是多步视频去噪会从被扰动的头部画面“想象”出错误的未来，把动作带偏；一步时动作更依赖当前帧、结构和腕部画面。

细到“某个 suite 的某一类”，抽样每格只有 20–50 个任务，误差可达 ±10–18 个百分点（例如 object 相机：抽样 86.5，全量 72.5），只看总分和全量结果。

### 全量 LIBERO-plus 消融（a1、v0，各 40k，视频 1 步 + K/V 缓存）

和 b2 用同一套全量协议、同样的推理设置（视频 1 步 + K/V 缓存，动作去噪 10 步），每个变体只在 b2 的基础上改动下表所列的地方。逐任务结果：`results/libero_plus_full_{a1,v0}_v1kv/<suite>/summary.csv`。

| 变体 | 相对 b2 的改动 | 成功率 |
|---|---|---|
| **b2** | — | **85.5**（8573 / 10030） |
| v0 | 结构学生换成官方 V-JEPA2 ViT-L：只取主干特征，不融合指令，没有 map / 关键点 / 词槽（`structure_expert.groups=[trunk]`，`video_saliency=0`） | 81.0（8119 / 10028） |
| a1 | 去掉视频 dropout（0.3 → 0）并关掉图像增强 | 75.7（7595 / 10030） |

按扰动类别（4 个 suite 合计）：

| 扰动类别 | b2 | v0 | a1 |
|---|---|---|---|
| 相机视角 | 64.4 | 61.7 | 38.6 |
| 机器人初始状态 | 89.0 | 76.5 | 90.2 |
| 语言指令 | 89.7 | 88.6 | 92.4 |
| 光照 | 98.2 | 97.1 | 95.8 |
| 背景纹理 | 80.8 | 72.9 | 67.6 |
| 传感器噪声 | 91.4 | 87.5 | 64.9 |
| 物体布局 | 87.3 | 84.7 | 85.2 |
| **合计** | **85.5** | **81.0** | **75.7** |

按 suite（spatial / object / goal / libero_10）：b2 90.6 / 89.6 / 80.3 / 81.8，v0 88.0 / 85.1 / 77.8 / 73.4，a1 80.2 / 76.6 / 74.2 / 72.2。

- a1 比 b2 低 9.8 个点，差距集中在传感器噪声（−26.5）、相机视角（−25.8）和背景纹理（−13.2）；机器人初始状态和语言反而略高。a1 同时去掉了 dropout 和增强，这组结果分不开两者各自的贡献。
- v0 比 b2 低 4.5 个点，差距最大的是机器人初始状态（−12.5）和背景纹理（−7.9），libero_10 上差距最大（−8.4）。
- v0 有 2 个 libero_object 任务没有跑出结果（trials = 0：task 194 和 task 1459），总数按 10028 计。

### 和其他工作比较时要注意

- **Robust-WAM**：它的 LIBERO-plus 评测按策略调用次数计步，每次执行 8 个动作，实际允许的环境步数远多于我们的设置，数字不能直接比。
- **OA-WAM**：用了仿真器的真值物体位姿。
- **Cosmos-Policy**：训练样本量约是我们的 30 倍。
- **OpenVLA-OFT+（LIBERO-plus 官方表 79.6）**：用 LIBERO-plus 扰动数据训练，属于见过测试分布的设定，要单独列。我们不使用任何扰动类训练数据。

---

## 3. 快速上手

所有入口都在 `recipes/`。下面以昇腾 NPU（aarch64，CANN 8.3–8.5）为例，所有结果都在 8×Ascend 910B3 上得到。

### 3.1 拉代码、配路径

```bash
git clone https://github.com/xiangantao/structwam.git && cd structwam
git submodule update --init StructWAM/third_party/cosmos-predict2.5
cp recipes/paths.example.sh recipes/paths.local.sh   # 至少填 STRUCTWAM_ASSETS
source recipes/paths.sh
```

仓库外的东西（模型、数据、检查点、评测输出）都放在 `$STRUCTWAM_ASSETS` 下，目录约定写在 `recipes/paths.sh` 开头。配置文件里的路径通过 `${oc.env:STRUCTWAM_ASSETS}` / `${oc.env:STRUCTWAM_REPO}` 读取，不用改。

### 3.2 环境

```bash
bash recipes/env/setup_openwam_npu.sh    # 策略 / 训练环境：Python 3.11，torch 2.7.1 + torch_npu 2.7.1
bash recipes/env/setup_libero_plus.sh    # LIBERO-plus 客户端环境：Python 3.10，MuJoCo 3.3.2，含资源下载
```

- 两个环境的完整依赖版本在 `recipes/env/*.requirements.txt`（就是产出上面结果的环境的 pip freeze，去掉了 CANN 自带的包）。`torch_npu` 要和本机 CANN 匹配。
- LIBERO-plus 客户端需要 ImageMagick（python wand）和 Mesa EGL。镜像里没有的话，把 `LIBERO_SYSDEPS_ENV` 指向一个补库的 env.sh（ModelArts 训练池镜像用的是 `sysdeps/hce2_aarch64/env.sh`，库文件本身不在仓库里）。

### 3.3 下载模型、数据和检查点

```bash
bash recipes/download_assets.sh          # 全部，约 120 GB
bash recipes/download_assets.sh eval     # 只评测已发布的 b2 检查点（不含训练数据）
```

| 内容 | 来源 |
|---|---|
| Cosmos-Predict2.5-2B（`robot/action-cond` + `tokenizer.pth`）、Cosmos-Reason1-7B | `nvidia/Cosmos-Predict2.5-2B`、`nvidia/Cosmos-Reason1-7B` |
| UMT5 文本编码器（只在预计算学生的指令缓存时用） | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 `text_encoder/`、`tokenizer/` |
| 结构学生 `step_0002000.pt`、`text.pt`、`appearance_stats.npz` | [`xiangantao/structwam`](https://huggingface.co/xiangantao/structwam) 的 `student/` |
| StructWAM b2 40k | [`xiangantao/structwam`](https://huggingface.co/xiangantao/structwam) 的 `b2/` |
| 训练数据：OpenWAM 的 native-action LIBERO（LeRobot v3） | `OpenWAM/LIBERO`（数据集） |

### 3.4 评测（全量 LIBERO-plus）

```bash
# 1. 训练检查点 -> 评测检查点：推理改成视频 1 步 + K/V 缓存，路径改成本机（只建软链接）
$OPENWAM_PY recipes/prepare_eval_ckpt.py $STRUCTWAM_ASSETS/checkpoints/b2 $STRUCTWAM_ASSETS/eval_ckpt/b2_v1kv

# 2. 预计算快速模式需要的两份缓存（1 张卡，约 5 分钟）
bash recipes/eval/precompute_caches.sh $STRUCTWAM_ASSETS/eval_ckpt/b2_v1kv

# 3. 每个 suite 一个 8 卡任务，可以分到 4 台机器上同时跑
for suite in spatial object goal long; do
    bash recipes/eval/eval_libero_plus.sh $suite $STRUCTWAM_ASSETS/eval_ckpt/b2_v1kv
done

# 看进度和结果（任意 python3）：
python3 recipes/eval/progress.py $STRUCTWAM_ASSETS/eval/b2_v1kv --diff
```

- 重新运行同一条命令会接着跑，已完成的任务保留。`SAMPLE_RATIO=0.1` 只跑固定种子抽出的 10%，用来快速检查。
- **快速模式（默认开）**：评测时环境渲染（MuJoCo 在 CPU 上软件渲染，每步约 0.65 秒）占了大部分时间，NPU 大约 80% 的时间在等。快速模式把 Cosmos-Reason1 文本编码器放到 CPU、所有指令的编码提前算好，每个服务的显存从约 28 GB 降到约 11.6 GB，每张卡跑 4 个服务、每个服务限用 24% 显存。实测吞吐约为每卡 2 个服务的 3.4 倍，一个 suite 8 卡约 4–6 小时（libero_10 最慢）。
  - 每个服务加客户端约占 30 GB 内存；内存不够时脚本会自动退到每卡 3 个，或者退回原模式（文本编码器在 NPU，每卡 2 个）。
  - 改动都是部署端的开关，不影响训练：`OPENWAM_TEXT_ENCODER_DEVICE`（`openwam/model/video_backbone/cosmos_predict25_backbone.py`）、`OPENWAM_PROMPT_CACHE_FILE` 和 `OPENWAM_NPU_MEM_FRACTION`（`openwam/deploy/engine.py`）、`OPENWAM_STUDENT_TEXT_EXTRA`（`heft/latent_policy/src/latent_dp/online.py`）。

### 3.5 训练 b2

```bash
bash recipes/train_b2.sh --dry-run $STRUCTWAM_ASSETS/runs/b2      # 只打印解析后的配置
bash recipes/train_b2.sh $STRUCTWAM_ASSETS/runs/b2                # 8 卡，40k 步；同一 RUN_DIR 重跑会续训
```

- 40k micro-step，每卡 batch 8、梯度累积 2（有效 batch 128），余弦学习率，每 2000 步存一次，保留最近 2 个。8×910B3 上约 8.5 秒一步，全程约 4 天。
- 训练出的检查点用 3.4 的 `prepare_eval_ckpt.py` 转成评测检查点。

---

## 4. 仓库结构

| 路径 | 内容 |
|---|---|
| `StructWAM/` | 模型代码。保留了 OpenWAM-Official/OpenWAM 到上游 `4d7f642` 的历史（本仓库里是 `7c3cd8d`：树和作者都没变，只去掉了 co-author 尾注），后面接 StructWAM 的提交。b2 配方：`configs/train_struct_b2.yaml` |
| `heft/latent_policy`、`heft/latent_video` | 结构学生及其蒸馏代码（来自 xiangantao/good_state 的 `2fd6a9d`，加上当时未提交的改动） |
| `recipes/` | 环境安装、下载、训练、评测检查点准备、LIBERO-plus 评测（快速模式）、进度查看 |
| `results/libero_plus_full_b2_v1kv/` | 全量 LIBERO-plus 逐任务结果（每个 suite 一个 `summary.csv`，含类别和难度） |
| `results/libero_b2_v1kv/` | 标准 LIBERO 结果（每个 suite 每个任务的成功次数） |
| `sysdeps/hce2_aarch64/env.sh` | LIBERO-plus 客户端在原生训练池镜像上需要的 ImageMagick / Mesa-DRI 库的环境脚本（库文件本身不在仓库里） |

V-JEPA2 用的是 facebookresearch/vjepa2 的 `204698b`，没有改动。
