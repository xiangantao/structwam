# StructWAM

在 OpenWAM 上的改动分支（`structwam`）：把我们从 Wan2.1-VACE 蒸馏出的结构学生，作为世界-动作模型的第三路专家接进去。

一句话：**几何结构管大动作，视频想象管精细的难点，动作分支按接触阶段决定听谁的。** 全程不生成像素。

## 三路专家

| 分支 | 实现 | 内容 |
| --- | --- | --- |
| 视频 | Cosmos-Predict2.5-2B，机器人后训练权重 `robot/action-cond` | 只在潜空间里训练；推理时只去噪 1–2 步，从不解码 |
| 动作 | OpenWAM 的 ActionDiT | 32 步动作块，统一 80 维动作空间 |
| 结构 | VACE 学生（冻结，选词模式）加结构专家（可训练） | 每个相机：64 个图 token、32 个关键点轨迹 token、16 个 V-JEPA2 主干 token（17 个时间步做注意力池化）、最多 5 个词 token |

三路 token 在视频 DiT 的每一层做一次联合注意力。这个骨架沿用 OpenWAM 的三系统架构，只是把 VLM 理解专家换成了结构专家。

## 学生的指令用法：选词模式（不用语法规则）

- 不用 `roles.json`：指令按分词器拆成内容词（只去掉通用停用词），每个窗口按注意力集中度（现在时刻热图的峰值）排序，超过阈值的最多保留 5 个。阈值第三人称 0.64、腕部 0.48，是用留出窗口定的，去掉的是集中度最低、几乎不落在物体上的那 1/5 的词。
- 每个保留词：热图（现在和 +16 帧）进图通道；它的 UMT5 向量和集中度作为一个词 token 给结构专家。由模型自己判断哪个是目标、哪个是目的地。
- 每个词热图得分最高的 K 分段，合起来代替原来的“目标段 / 目的地段”，用于保留区域和关键点。
- 学生文本缓存里没有的新指令，现场用 UMT5 编码。
- 留出窗口上的检查（仿真真值只用于打分）：多数窗口选出 3–5 个词；词分段覆盖目标 64%（第三人称）/ 91%（腕部），+16 分段覆盖目的地 50% / 56%。
- 旧的角色模式（`roles` 非空）原样保留，供已有的 DP 实验复现，结果逐位一致。

## 结构分支和另外两路的作用

1. **结构 → 动作：阶段门控**
   - 门控预测“这个动作块离接触有多近”，监督信号来自示范自身的夹爪指令，不用仿真真值。
   - 它在动作 token 的注意力上加偏置：离接触远时多看结构，接近接触时多看视频 token，尤其是腕部画面所在的区域。
   - 训练时有一部分样本直接用标签来路由（teacher forcing）。
2. **结构 → 动作：草图热启动**
   - 草图头只根据结构（加本体状态）预测整个动作块。
   - 推理时，动作去噪从“草图加噪到 t=600”开始，而不是从纯噪声开始。
3. **结构 → 视频：训练约束**
   - 视频损失按两种权重加权：
     - 像素权重：手、目标、目的地、运动区域的显著图；
     - 样本难度：接近接触、轨迹没把握、只用结构的草图误差大。
   - 轨迹一致性：预测出的下一个潜帧，在学生轨迹指向的位置上，要和当前帧保持外观一致。
   - REPA 式对齐：视频 DiT 中间层的特征，对齐学生的 V-JEPA2 主干。
4. **视频 → 结构：** 结构 token 可以读视频 token（原三系统里理解分支是只读自身的），用世界模型的知识修正远期轨迹。
5. **视频损失参与训练结构专家：** 默认 `video_detach: []`，视频查询读取结构 token 的 K/V 时保留梯度，因此视频损失可以直接更新可训练的结构专家；VACE 学生仍然冻结。若做消融，可显式设为 `[structure]` 来切断直接路径，或 `[structure, action]` 同时切断经动作 token 的间接路径（后者也阻止视频损失训练动作分支）。

## 改动清单

**新增**
- `openwam/structure/`
  - `student.py`：学生封装，调用 latent_policy 里的 `OnlineMaps`，代码不复制
  - `history.py`：部署时的 33 帧历史
  - `layout.py`：关键点 token 布局和画布几何
  - `tokens.py`：结构 token
  - `phase.py`：阶段标签和门控
  - `routing.py`：注意力偏置
  - `sketch.py`：草图头、热启动、去噪调度整形
  - `losses.py`：加权视频损失、轨迹一致性、表示对齐
- `openwam/model/architectures/tri_system/`
  - `struct_expert.py`：结构专家，复用理解专家的块结构
  - `struct_driver.py`：在三系统 MoT 驱动上加：结构读视频、门控偏置、对齐层隐状态捕获
  - `struct_system.py`：新架构，注册名 `tri_system_struct`（framework `tri_system` / variant `struct_expert`）
- `openwam/dataloader/libero_struct.py`：LIBERO 读取器，额外给出每个相机 33 帧历史和阶段标签
- 配置
  - `configs/model/tri_system_struct.yaml`
  - `configs/model/video_backbone/cosmos_predict25_2b_robot.yaml`
  - `configs/dataloader/libero_struct.yaml`
  - `configs/train_struct.yaml`
- 脚本：`scripts/structwam/`，包括 NPU 启动和静态检查

**修改原文件（都很小）**
- `base.py`：`generate()` 增加可选的 `initial_action_latents` 参数
- `engine.py`：把 `structure_frames` 转发给架构
- `obs_preprocess.py`：保留每个相机的原始画面
- `policy.py`：每个控制步都记录历史帧
- 数据集注册表和 tri_system 包导出：各加一行

## 怎么跑（还没跑过）

```bash
bash scripts/structwam/train_npu.sh dataloader.dataset_dir=/path/to/openwam_libero
```

只读检查：

```bash
python scripts/structwam/check_static.py
```

## 跑之前必须补的东西

1. **Cosmos 权重：** Cosmos-Predict2.5-2B（含 `robot/action-cond` 变体）和 Cosmos-Reason1-7B，都放到 `workspace/models/`。
2. **Cosmos 依赖：** `git submodule update --init third_party/cosmos-predict2.5`，再运行 `scripts/install_cosmos_predict25.sh`。注意：这个脚本会编译 transformer-engine，在昇腾上做不到，需要改走 PyTorch 注意力。
3. **OpenWAM 格式的 LIBERO 数据：** 读取器要求 `meta/conversion.json`，由 `download_benchmark_data.py` 生成。需要核对它的画面朝向是否和学生训练时一致；不一致就设 `structure_flip`。
4. **NPU 移植（进展）：**
   - 环境 `openwam-npu` 已装好：OpenWAM、cosmos-oss、cosmos-predict2，torch / torch_npu 未被替换。
   - transformer-engine 用纯 PyTorch 替代模块 `npu_shims/transformer_engine`（RMSNorm、RoPE），在 NPU 上自动调用 torch_npu 融合算子（`TE_SHIM_FUSED=0` 可退回纯 PyTorch）。替代模块不能放进 PYTHONPATH（megatron-core 会把它误当成真的 TE），由 `cosmos_predict25/npu_compat.py` 在 megatron 加载之后再挂上。Cosmos DiT 的注意力后端设为 `torch`。
   - **修正了 OpenWAM 的一个缺陷：** Cosmos-Predict2.5 的网络参数漏了上游的 `timestep_scale=0.001`，网络看到的时间步大了 1000 倍，预训练的去噪能力用不上。已补上。
   - 上游的 float32 策略（时间嵌入、AdaLN 调制、输出层用 float32）用“float32 计算岛”实现，不依赖 NPU 不支持的 float32 autocast（`fp32_strategy: true`）。
   - 联合注意力按行分组：视频行、结构行用布尔掩码（走融合注意力），只有动作行带门控偏置。和整段浮点掩码结果一致，单层快 3.7 倍。
   - 检查结果（`scripts/structwam/check_cosmos_npu.py`，`runs/cosmos_npu_check_20261020/`）：4 段真实 LIBERO 片段加噪还原，相对误差 σ=0.3/0.6/0.9 为 0.076/0.280/0.818（带噪输入为 0.212/0.848/1.909）；错误 RoPE 0.102/0.341、无 RoPE 0.186/0.572，确认 RoPE 正确。DiT 批 4 前向加反向 0.52 s，显存峰值 28 GiB。
   - deepspeed 在昇腾上能否正常工作，还没验证。
5. **新指令：** 不在学生文本缓存里的指令会现场用 UMT5 编码（第一次会加载约 11 GB 的编码器）。
6. **位置关系：** “旁边”“上面”这类定位词目前不专门处理，以后再看。

## 推理和部署的约束

- `cfg_scale` 必须为 1：结构特征没有接入 CFG 的批次扩展。
- `decode_video` 恒为关闭。
- 部署机器上必须能访问学生的检查点、`roles.json` 和外观 PCA 统计文件。
