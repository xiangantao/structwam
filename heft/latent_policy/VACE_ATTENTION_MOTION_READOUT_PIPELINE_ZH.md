# VACE 注意力运动结构读出：观测、Prompt/K 选区、QK 轨迹与 GT 加噪短未来诊断

状态：2026-10-10 起，主线改为 1 步去噪。下文 2026-10-08 的固定案例与产物保留为历史对照。

这份中文版本供检查，英文版
[`VACE_ATTENTION_MOTION_READOUT_PIPELINE.md`](VACE_ATTENTION_MOTION_READOUT_PIPELINE.md)
是规范版本。

当前管线的核心不是“多锚点”本身，而是把 VACE 视频生成器当作一个注意力表征读取器：

```text
观测视频 + 任务 prompt
        -> prompt 定位 + K 空间组织
        -> QK 对应关系解码
        -> 当前观测中的物体/夹爪轨迹

观测条件 + GT 未来 latent 加 sigma=0.8 噪声
        -> 短步去噪诊断
        -> 未来运动结构读出
```

它要分别回答三个问题：

1. VACE 的内部注意力是否暴露了当前观测中的有用空间结构？
2. prompt 能否选出任务相关区域，同时由 K 保持区域内部的空间组织？
3. 当未来 latent 被控制性地破坏、再进行去噪时，中间表征是否暴露运动结构？

## 当前主线：1 步到原第 13 步的终点

观测协议仍是 33 帧真实过去视频前重复 15 帧第 0 帧，让 prompt/K 预热。
学生只输入真实过去 33 帧与指令。未来 teacher 仍使用完整 33+16 帧 GT 的加噪主干 latent；
控制塔只看过去 33 帧，并输入完整任务 prompt。目标是恢复运动结构，不以纹理质量为判据。

```text
GT VAE latent 加噪，sigma=0.8019521832466125，timestep=801
    -> 一次 CFG UniPC 更新，跨过原参考计划的 [0,13] 区间
    -> 实际带噪 latent，sigma=0.4996665418148041，timestep=499
    -> 再做一次条件前向，捕获全部未来特征
    -> prompt 热图 / K 分区 / 种子 / QK / 多锚点 beam 轨迹
```

`teacher.denoise_updates=1` 表示实际执行 1 次更新；
`teacher.denoise_reference_updates=13` 表示终点固定为原 40 步 UniPC 计划中，从约 0.8 开始更新 13 次的位置。
flow_shift=3、CFG=5。它既不是只执行原来的第一步，也不是直接把 GT 加噪到约 0.50。
未来的热图、K、种子和轨迹整条链都从这个终点重新计算，不混用旧的 13 步读出。
每次提取重置调度器历史，保留未使用的前后调度节点；协议保存实际起点/终点噪声、时间步和采样节点。
既有 head、每 8 帧锚点、beam 约束与时间映射沿用。

13 步对照配置保留在 `configs/vace_structured_step13_reference.json`。
1 步是对长去噪过程的近似；终点噪声相同不代表 latent 或表征相同。
episode39 的 agentview 上，用同一组可见且发生位移的 GT 点评测，排除锚点与静止物体：

| 实际更新数 | 移动物体 EPE，源图 256px | 夹爪 EPE，源图 256px |
| ---: | ---: | ---: |
| 1 | 20.43px | 23.70px |
| 13 | 22.06px | 24.50px |

这只是这个片段的结果，不是全场景排名。记录在 `../runs/vace_sparse_future_lowsteps_ep39_20261010/summary.json`。
对比视频的 RGB 背景共用旧的完全去噪视频；每个方案独立重算未来注意力、选区、种子与轨迹。
不能把这个背景说成 1 步直接解码出的画面。

此前单卡 A800、teacher batch2、并行 CFG batch4、模型已驻留的同步实测：
完整 teacher 从 72.59 秒降到 17.99 秒，约 4.04 倍；未来去噪加捕获从 60.00 秒降到 5.13 秒。
包括五个头与 CPU beam，不含模型初始化和学生训练；共享卡单次测量不代表八卡训练吞吐。
记录在 `../runs/vace_sparse_teacher_speed_20261010/summary.json`。

主线启动命令不需要额外参数：

```bash
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_policy/scripts/launch_vace_online_structured.sh
```

新实验输出为 `../runs/vace_structured_online_step1/<task>/`。已经运行的进程保持其加载的旧配置。
13 步 checkpoint 与 1 步配置属于不同 teacher 协议，严格续训检查会拒绝直接混用。
历史视频、矩阵及协议不改名为 1 步产物。

## 这条实验能说明什么，不能说明什么

观测分支是表征和对应关系诊断：读取注意力空间场并画出轨迹。它不是闭环策略，不是使用 GT mask 的物体检测器，也不能证明每条轨迹都对应正确物体。

未来分支是有特权信息的 teacher 诊断：控制条件只包含观测前缀，但主干初始 latent 包含未来 GT 的加噪版本。因此它不是因果未来预测基准。之后如果做因果预测或蒸馏，学生必须只收到过去观测。

## 固定案例和产物

| 项目 | 当前值 |
| --- | --- |
| 数据集 | LIBERO |
| suite / episode | `libero_spatial` / episode `39` |
| 相机 | `agentview` |
| 原视频 | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_tracking_all_suites_20261004/videos/episode_00039/agentview.mp4` |
| 实际观测窗口 | 原始帧 `48..80`，共 33 帧 |
| Python 切片 | `48:81` |
| 任务 prompt | `pick up the black bowl on the wooden cabinet and place it on the plate` |
| 定位短语 | `black bowl` |
| VACE 模型 | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/models/Wan2.1-VACE-1.3B` |
| 文本模型 | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/models/Wan2.1-T2V-1.3B-Diffusers` |
| 最终目录 | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_qk_multi_anchor8_20261008_ep39` |

最终 H.264 视频都是 512x512、10 FPS：

```text
observed_multi_anchor8_h264.mp4              33 帧
future_gt_sigma080_multi_anchor8_h264.mp4    16 帧
observed_then_future_multi_anchor8_h264.mp4  49 帧
```

10 FPS 是记录/渲染视频的帧率，不能直接当成仿真的 20 Hz 控制频率。

## 观测输入到底是多少帧

这里必须区分三个数字：

| 含义 | 数量 | 内容 |
| --- | ---: | --- |
| 实际观测 RGB 帧 | 33 | 原始帧 `48..80`，局部编号 `0..32` |
| 观测读出送入 VACE 的 RGB 槽位 | 48 | 前置 15 个重复帧 + 33 个实际帧 |
| 输出展示的观测时间线 | 33 | 只展示实际观测帧 |

观测分支的输入严格是：

```text
[实际第 0 帧] 重复 15 次
然后接
[实际第 0 帧, 实际第 1 帧, ..., 实际第 32 帧]
```

所以模型看到 48 个时间槽，但真正的观测只有 33 帧。前面的 15 个槽位是第 0 帧的复制，只用于让 prompt/K 原型预热稳定下来；它们不是新增数据，不显示，也不能算作额外的 15 帧观测。

VAE 时间压缩约为 4 倍，48 个输入槽位得到 12 个 latent 时间位置。实际第 0 帧的原型落在 latent index 4。最终保存的观测锚点为：

```text
原始观测帧：  0,  8, 16, 24
latent 索引： 4,  6,  8, 10
```

原始观测第 32 帧是观测时间线的最后一帧，也是接入未来诊断的边界；它没有作为额外观测锚点存进最终 NPZ。可视化时把 latent 坐标线性插值到 33 个原始帧，不能把一个 latent 坐标直接保持四个 RGB 帧。

## 为什么未来要用 GT 加 sigma=0.8，而不是直接纯噪声

这里的目标不是立即回答“VACE 能不能只靠过去预测未来”。前面纯噪声续帧把多个问题混在了一起：

- 模型要推断未来轨迹；
- 要生成合理画面；
- 要维持手和物体的时间身份；
- 还要判断中间特征里是否存在运动信息。

如果纯噪声结果失败，我们无法区分“表征里没有运动信息”和“生成器本身没有把未来画出来”。反过来，如果把干净未来直接给主干，结果又接近简单修复，不足以检验有用的去噪中间状态。

因此采用受控的 GT 加噪：

```text
latent_init = (1 - sigma) * VAE(完整49帧 GT窗口)
              + sigma * 高斯噪声
```

控制塔仍然只收到观测前缀，而主干未来部分被噪声破坏。这样可以单独看：在未来画面不完全干净、但运动结构仍有机会被恢复的情况下，注意力读出能不能找到运动。

为什么选 `sigma=0.8`：

- 噪声较低时，未来内容保留太多，测到的主要是恢复/修补；
- 纯噪声或更高噪声时，生成失败会压过表征读出，难以判断到底是预测错还是特征不可读；
- `0.8` 对这次 16 帧短诊断提供了一个折中：未来 latent 已经被明显破坏，但经过中间去噪步骤后，动作结构先于纹理完全精修而出现。

这不是理论上通用的最优 sigma，也不是因果预测设置。实际调度器 sigma 是 `0.8019521832466125`，不是简单四舍五入的 0.8。

## 未来输入和因果边界

未来诊断使用 49 帧完整窗口：

```text
实际观测前缀：33 帧，源帧 48..80
诊断未来：   16 帧，源帧 81..96
完整 GT 窗口：49 帧
```

控制条件只含观测前缀：

```text
观测控制像素：真实 RGB
未来控制像素：置零
观测 mask：0
未来 mask：1
```

但主干收到的是完整 49 帧 GT latent 的 sigma=0.8 加噪版本。因此准确说法是“观测条件 + 含未来信息的加噪主干 latent”，不是“观测条件 -> 因果预测”。最终未来视频只展示 16 帧未来，QK 读出保留的是完整 49 帧窗口对应的 13 个 latent 时间位置。

## Prompt、K、QK 各自做什么

当前读出把三者分开：

| 读出 | 路径 | 作用 |
| --- | --- | --- |
| Prompt 定位 | 主干 `attn2`，`L22H4` | 找到与 `black bowl` 短语相关的位置 |
| K 空间组织 | `L22H9`、`L15H7` 两组 K 场 | 让 prompt 种子留在空间连贯区域内 |
| 轨迹对应 | 主干 self-attention `attn1`，`L15H4` | 查询一个网格点下一时刻可能在哪里 |

前向计算受到 VACE 控制输入条件；但最终保存的 Q/K 是上面指定的模型 block 注意力读出，不是 GT 分割 mask，也不是另训出来的物体编码器。

## Prompt 到 K 区域

在 `L22H4` 上，把 `black bowl` 对应文本 token 的 `attn2` 注意力概率相加，得到 32x32 prompt 热图。`L22H9` 和 `L15H7` 的每个网格向量先做 L2 归一化，再用向量相似度扩展/筛选 prompt 种子。

每个锚点最多保留 64 个网格格子。最终版本不使用仿真物体 mask、手工聚类中心、光流或固定物体数量。静止物体的候选点也不会因为“没有位移”被预先删掉；只要被 prompt/K 选中，就可以进入跟踪。

prompt 颜色使用所有观测 latent 时间共享的一组全局范围，约为 p1-p99；不对每帧独立拉伸。逐帧拉伸会把早期真实方差很小的特征扰动放大成假运动。

## QK 对应关系解码

跟踪头是 `L15H4`，冻结方向是：

```text
q_t -> k_(t+1)
```

对每个选中的锚点格子，在下一时刻 32x32 网格上计算 QK 相似度，并联合局部描述子连续性、随时间变化的 K 区域门、速度和加速度。每一步保留多个候选路径再选最佳路径，避免逐帧独立 argmax 的孤立大跳。

| 参数 | 值 |
| --- | ---: |
| beam size | 4 |
| 候选 top-k | 32 |
| 单步最大候选距离 | 4.0 个网格格子 |
| motion penalty | 0.15 |
| local descriptor | 0.25 |
| region gate | 0.08 |
| acceleration penalty | 0.05 |
| descriptor EMA | 0.1 |
| 后处理最大步长 | 3.0 个网格格子 |
| 后处理最大加速度 | 1.5 个网格格子 |
| 后处理轮数 | 2 |
| 光流 | 不使用 |

后处理只去掉孤立大跳和过大的加速度，不会把 QK 解码替换成光流，也不使用 GT 坐标。

## 多锚点为什么存在

多锚点只是这个读出管线中的稳定性措施，不是管线的全部含义。单个锚点沿 VAE 压缩的长序列滚动容易漂移，所以观测分支从原始帧 `0、8、16、24` 分别开始，每个锚点约向后滚 8 个原始帧，再把重叠短轨迹一起渲染。这样轨迹更密，也减少长期误差累积。

未来分支从局部未来第 `0、8` 帧开始；在拼接后的 49 帧时间线上是全局位置 `33、41`，完整生成视频 latent 中保存的未来锚点是 `[8, 10]`。

## 时间渲染

Q/K 只存在于 latent 时间位置，不是每个原始 RGB 帧一份。显示时将相邻 latent 坐标和轨迹点线性插值到原始 10 FPS 时间线。这只是时间显示对齐，不增加信息，也不使用光流。

观测和未来视频分别渲染，最后才拼接。合成预览中的紫色、绿色是显示约定，不是模型学习到的两个通道。

## 最终保存内容

`multi_anchor_readout.npz`：

```text
observed_heat             [12, 32, 32]
observed_region_scores    [12, 1024]
observed_anchor_latents   [4] = [4, 6, 8, 10]
observed_seeds            [4, 64]，保存选中的网格格子
future_heat               [13, 32, 32]
future_region_scores      [13, 1024]
future_anchor_latents     [2] = [8, 10]
future_seeds              [2, 64]，保存选中的网格格子
```

`heat` 是 prompt 产生的语义场，`region_scores` 是跟踪使用的 K 空间门；都不是 GT 分割 mask。seed 是 32x32 token 网格格子，不是像素级表面点标签。

最终目录还包括：

```text
observed_multi_anchor8_h264.mp4
future_gt_sigma080_multi_anchor8_h264.mp4
observed_then_future_multi_anchor8_h264.mp4
multi_anchor_readout.npz
protocol.json
```

## 只读检查

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765
cat runs/vace_qk_multi_anchor8_20261008_ep39/protocol.json
python - <<'PY'
import numpy as np
p = "runs/vace_qk_multi_anchor8_20261008_ep39/multi_anchor_readout.npz"
z = np.load(p, allow_pickle=True)
for name in z.files:
    print(name, z[name].shape, z[name].dtype)
PY
```

现有短时 GT 去噪脚本可以单独重新生成 GT-assisted 诊断；但最终多锚点合成使用的一次性 driver 没有提交进仓库，所以当前目录及其 `protocol.json` 是 2026-10-08 结果的事实来源。不能声称当前仓库已经提供一条命令完成 bitwise reproduction。

## 明确排除的旧版本

以下不是当前主线：

- 把纯噪声未来续帧当作最终未来结果；
- 从纯噪声未来结果读取 QK；
- Lucas-Kanade 或其他光流未来合成；
- 只用第一个锚点；
- 不做 15 帧重复预热、直接从真实第 0 帧取原型；
- 每帧独立拉伸 prompt 热图；
- 使用 GT mask、手工聚类中心或固定物体数量假设。

## 解释边界

观测分支检验的是注意力空间场能否暴露连贯的当前结构/时间对应关系，不等于闭环控制或因果物体动力学。未来分支的主干初始化含有未来 GT；任何蒸馏、动作条件预测或策略实验都必须另行定义只使用过去观测的协议。
