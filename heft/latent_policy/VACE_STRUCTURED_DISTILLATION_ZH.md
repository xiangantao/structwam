# VACE 结构化蒸馏：过去视频与指令到观测、未来矩阵

对应英文规范：`VACE_STRUCTURED_DISTILLATION.md`。

2026-10-10 主线更新：未来 teacher 从 GT 加噪约 0.8，用 **1 次 UniPC 更新**到原第 13 步的约 0.50，
再条件前向读取。13 步保留为明确对照。当前默认在线提取；早先的 TOS 离线方案已停止，
历史保存契约见 [离线提取中文说明](VACE_TEACHER_OFFLINE_ZH.md)。

## 学生输入与结构

学生只接受过去33帧 RGB、冻结 UMT5-XXL 的逐词 embedding 和文本 mask。
没有未来 RGB、未来 latent、teacher 特征或仿真状态输入参数。

```text
过去33帧 -> 原版 V-JEPA2 ViT-L/16，可训练
             最后4层之后接文本交叉注意力
指令 -> 冻结 UMT5-XXL -> 可训练4096到1024投影
                         |
                保留完整过去时空tokens
                         |
           观测读出器 / 未来时间、空间查询解码器
                         |
              五个独立的时空Transformer输出头
```

V-JEPA2 tubelet宽2，为保留第33帧，将最后一帧复制一次补到34帧。
默认输入256平方，原生输出17个时间位置、16乘16空间网格。
32乘32是对齐teacher的读出网格，初始化时插值，不代表原生分辨率变高。

观测、未来解码器各两层；每层包含8乘8窗口空间注意力、时间注意力、
对过去视觉和文本的交叉注意力。输出头再用两层时空Transformer，
接`1024 -> 2048 -> 输出维度`残差MLP。语义头用视觉query、文本key出关联矩阵。

默认`shared`模式：一套V-JEPA2、一套文本融合和一对观测／未来解码器，供五个头共同使用。
每次学生前向只编码一次过去视频，五个头的梯度一起更新这套共享表征。
五个头各自有独立Transformer与投影参数，同一头在观测和未来分支共用权重。
单头模式保留作对照；共享版用一个checkpoint保存整套骨干与五个头。

## 各头输出

观测有12个native位置，未来分支保留完整49帧teacher对应的13个native位置。
每个位置1024个空间格子，所有量均有观测和未来。

| task | 输出 |
| --- | --- |
| `seg_l22h9` | L22H9对应的归一化K，`[B,T,1024,128]` |
| `seg_l15h7` | L15H7对应的归一化K，`[B,T,1024,128]` |
| `tracking` | 独立的Q、K，各`[B,T,1024,128]`；乘出来监督关系 |
| `semantic` | 每个位置对文本token的关联，`[B,T,1024,512]` |
| `depth` | 固定尺度相对深度场，`[B,T,1024]`，不是厘米距离 |

语义矩阵保留逐词响应；不能把全部词的softmax响应相加作为热图，那会恒等于1。
分区不预设物体数或聚类中心。轨迹由QK、选区、锚点和beam计算，不是直接回归坐标。

## 策略侧 latent 组织（设计记录，尚未作为训练契约）

当前先把两件事分开：prompt 热图回答“每一帧哪里是第 n 个目标”，track 束回答“这个目标内部的 C 个点如何移动”。prompt 文本只编码一次，但在每个时间位置读取一张 prompt 热图；在稀疏 anchor 处，用这张热图重新框住同一个目标并校正轨迹，不能重新给目标编号。

```text
prompt_heat:     [B,T,N,32,32]
track_heat:      [B,T,N,C,32,32]
K_frame:         [B,T,32,32,Dk]
relative_depth:  [B,T,32,32,1]
anchor_flag:     [B,T]
```

`N` 是 prompt 点出的目标/名词数量，按 batch 最大值 padding 并用 `bundle_mask`；`C` 是每个目标固定的跟踪点数。`track_heat[b,t,n,c]` 是第 `n` 个目标第 `c` 个点在第 `t` 个时间位置的 32x32 概率热图。`anchor_flag` 只是重置标记，不是策略特征。K 和相对深度仍逐帧提供给策略，但不负责 track 的 anchor 重置；这一版不把这些字段直接拼成一个向量。

## Teacher 时间与未来

观测：33帧真实视频前加15个第零帧复制，送控制塔；主干纯噪声、timestep1000。
观测预热native0到3不算直接监督。锚点native4/6/8/10，对应现有RGB0/8/16/24约定。

未来：完整49帧GT的VAE latent以sigma约0.8加噪；控制塔只收过去33帧，
未来16帧像素置零；mask前0后1。使用40步UniPC调度，
**执行1次更新到原第13步终点**，再在timestep499做条件前向捕获矩阵。
`denoise_updates=1`是实际次数，`denoise_reference_updates=13`是原计划的参考终点；
节点为`[0,13]`，sigma从0.8019521832466125到0.4996665418148041。
不是只执行原来的第一步，也不是直接把GT加噪到终点，更不拿干净x0估计充当实际状态。
每批新建调度器历史，保留未使用的前后节点；协议保存实际起点/终点sigma、时间步和节点。
五组未来矩阵与teacher轨迹都取自同一终点条件前向。

`configs/vace_structured_step13_reference.json`保留原来的13次更新。
1步是近似，不保证特征与13步相同。旧checkpoint/cache是不同监督协议，
不能说成1步标签；严格续训会拒绝跨teacher协议直接恢复，可选缓存读取也会检查实际更新数与参考终点。
观测协议与时间映射沿用。

未来锚点native8/10，沿用RGB33/41约定。native8含观测边界，保留它监督进入未来的关系。
native位置有时间感受野，不是精确每RGB帧一份；映射和有效性mask随输出保存。
两条不同提取协议的Q/K不会直接交叉相乘当作统一teacher坐标。

**未来深度接口已接入，但仍需验证。** 观测发现的L16H9通道52，不自动证明
在GT加噪、1次更新后的未来仍单调对应深度；协议里明确标记`unvalidated_protocol_transfer`。

## Loss 与轨迹

分区：teacher单位K的余弦对齐；无固定中心数。
跟踪：全网格抽样query的相邻/跨两步QK分布KL，加多锚点短轨迹软网格监督。
轨迹主监督不要求逐维复制Q/K；teacher的离散beam无需反传，学生loss经过自己的QK。
语义：逐词关联分布KL。深度：固定尺度读数的SmoothL1。

辅助约束是轻微颜色/亮度增强一致性、teacher互相匹配可靠位置上的弱循环约束、
K相近邻域的弱深度正则。整段颜色扰动一致，不加时间EMA。
这些约束尚未证明提高闭环控制，可以置零做纯蒸馏对照。

全任务训练的轨迹种子用完整指令的内容token最高响应，再由两组K扩展筛选。
这是通用的新读出，不等于旧案例里人工指定`black bowl`短语。
每锚点最多64格，不是每物体64点。没有GT mask、光流或物体数量假设。
路径有效mask只表示时间支持范围，不是经过标定的可见性/置信度。

共享版输出字段为`seg_k_l22h9`、`seg_k_l15h7`、`track_q`、`track_k`、`text_attention`、`depth`。
Q与K是一组跟踪头的两个输出，所以六个矩阵对应五个头。此处不拼成ACT特征、不做压缩。

`train.head_weights`给五个头分别设置权重，默认轨迹头为2，两个分区头、语义头和深度头各为1；总loss是加权各头观测和未来loss之和，
再加增强一致性。各头原始loss与加权值分别记录；同权重不代表数值尺度或梯度强度相同。
Teacher在同一次前向挂齐所有读取hook，未来1步去噪每个样本只做一次，不重复五遍。

## 数据、训练与保存

完整本地LIBERO训练集：四suite、40任务、1693段、273465帧，两个视角。
存储FPS10按元数据保留，不当成20Hz控制动作。全部训练轨迹参与，评测需另采episode。

每epoch每段每相机随机4个49帧窗口，窗口与噪声种子可复现，epoch变化会重新采样。
**默认没有视觉特征缓存，没有固定窗口列表。** 只缓存41条文本（40指令加negative）。
teacher在线按microbatch8提取；DataLoader的`teacher_rgb`完整49帧只交给teacher，
学生拿到的`rgb`仅33帧过去。

当前配置：学生每卡batch32、teacher microbatch8、累积1、八卡global256，
基座LR5e-5、头LR5e-4，warmup100，10000次optimizer更新，轨迹头权重2，其余各1。
这些是配置起点，不是显存上限或收敛保证。此前共享A800上、驻留teacher batch2同步单次测量，
完整teacher由13步的72.59秒降到1步的17.99秒，约4.04倍；含VAE和CPU beam，不含学生训练。
记录在`../runs/vace_sparse_teacher_speed_20261010/summary.json`，不代表八卡训练吞吐。
新输出目录为`../runs/vace_structured_online_step1/<task>/`。
支持学生FSDP2；冻结teacher每rank一份，UMT5缓存后卸载。可选DDP。

每100次**optimizer更新**保存，最后一次也保存；保存学生、optimizer、每rank随机状态、
epoch、数据游标和协议哈希，不保存teacher模型。续训检查任务、结构、teacher、数据、loss和world size。
日志分开记录蒸馏主项`distillation_total`和真正优化的`total`，后者包含加权增强一致性。
首次更新前会输出teacher与学生加载阶段。

当前在线启动脚本：`scripts/launch_vace_online_structured.sh`，默认`shared`，也可指定上述单头task。
单卡可设`NPROC_PER_NODE=1`；八卡默认8。完整准备、smoke、训练和续训指令见英文版。
已运行的进程保持其加载的旧配置；此次不自动启动或重启长训练。

## 当前1步主线的验证（2026-10-10）

- 28项CPU检查通过，覆盖调度、学生输入边界、loss/checkpoint和离线序列化。
  1/2/4步共用原第13步终点；13步调度的solver输出与原实现逐值完全相同。
  并行/串行CFG都在实际更新之后捕获。
- 缓存读取会在解码RGB前拒绝把旧13步监督用于1步配置；加入该检查后，6项离线测试重新通过。
- 单张真实A800完成五头teacher提取、观测/未来loss与V-JEPA2反传。
  记录确认节点`[0,13]`、实际1次更新、终点sigma=0.4996665418148041、timestep499。
  骨干/文本投影/未来query梯度范数分别16.98/29.99/0.424，均有限且非零。
  记录：`../runs/vace_structured_step1_smoke_20261010/shared.json`。
  这是batch1的执行验证，没有optimizer更新，不是batch32/八卡容量或表征质量验收。
  验证进程已退出。

## 历史初版单头的检查（13步）

- 10项CPU检查通过：所有头前向与反传、指令影响未来输出、拒绝未来RGB输入、保存恢复输出一致。
- 真实LIBERO窗口的teacher提取与ViT-L反传通过；未来实际sigma为0.801952，执行13次更新后在timestep499捕获。
  基座、文本投影和未来query均有有限且非零梯度。
- 两卡小模型FSDP2反传、集体保存和恢复通过，恢复输出误差为0。这不是完整版八卡显存验收。
- 单卡真实训练入口完成一次optimizer更新：teacher与学生同时驻留，包含增强反传，完整checkpoint写盘成功。
  验收batch为1，更新用时62.95秒，不含初始化和保存；还没有测batch2的容量或收敛。

验收记录在工作区`runs/vace_structured_smoke`。单独学生反传测到的4.03GiB峰值不包含optimizer与teacher同时驻留，
不能当成完整训练的总显存。运行通过也不代表表征质量或控制成功率已经验证。

2026-10-11 清理时删除了这个目录的单步测试 checkpoint、FSDP 虚拟 checkpoint 和 NPZ 测试导出，原路径的 JSON 协议、指标和验证报告保留。其他旧短步试跑的配置与日志集中到 `runs/_test_records/`，它们不是 checkpoint 备份。需要测试权重时应重新运行 smoke 配置生成；最新 1 步主线的反传报告仍保留在 `runs/vace_structured_step1_smoke_20261010/`。

## 纯学生导出

`scripts/infer_vace_structured.py`加载学生checkpoint，只解码`--start`起的过去33帧，
使用缓存文本或冻结UMT5编码新指令。输出NPZ包含观测、未来矩阵及时间、锚点、有效位置。
未来输出由过去视频与指令产生；推理时不调用teacher、不读取未来帧。

共享checkpoint直接导出全部五个头；单头checkpoint只导出对应头。两种结构不能直接互相续训。
英文版提供完整命令。`configs/vace_structured_shared_one_update.json`只做一次optimizer更新，
用于检查在线teacher、增强反传、optimizer和checkpoint写盘，不是长训练配置。

默认八卡共享版启动：

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft
bash latent_policy/scripts/launch_vace_online_structured.sh
```

## 历史共享五头的检查（13步）

- 13项CPU检查通过，覆盖共享骨干只编码一次、五个头均有非零梯度、loss权重和五头checkpoint恢复。
- 两卡小模型FSDP2反传、optimizer更新、集体保存与恢复通过，全部头的恢复输出最大误差为0。
  记录：`runs/vace_structured_smoke/fsdp_shared/fsdp_check.json`。
- 单卡真实共享训练入口完成一次更新，包括在线teacher、五个头的观测与未来loss、增强反传和checkpoint保存。
  batch1的PyTorch峰值显存为13.76GiB，更新49.56秒，不含初始化和保存。
  记录：`runs/vace_structured_smoke/shared_one_update/shared/metrics.jsonl`。
  这还不是完整版八卡或batch2容量验证。
- 重新加载共享checkpoint的纯学生推理通过：仅使用142到174这33帧及指令，导出了两分支全部矩阵及锚点时间信息。
  结果：`runs/vace_structured_smoke/student_shared.npz`及同名JSON。推理没有实例化teacher。
  只更新一步的输出用于验证管线执行，不是表征效果评测。
