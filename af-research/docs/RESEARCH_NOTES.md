# S5 自动对焦提取与再训练调查

这是研究过程、静态证据与未解问题的记录。安装和交接从 [项目入口](../README.md) 开始；本文所有命令均在仓库的 `af-research/` 目录执行，不在 `docs/` 执行。`artifacts/` 是各自电脑重新生成的本地产物，历史报告、照片、模型和虚拟环境不随本源码目录分发。

目标：同时改善追焦/主体识别与静态/低光对焦。当前已恢复两份网络的整数权重、缩放系数与偏置，候选计算图已能从准备好的输入张量执行到分类分数/框回归，并通过梯度检查。权重排列、部分算子语义和预处理仍有歧义，因此尚未证明原厂识别网络数值等价，也未产出训练后模型或相机回写版本。

## 当前交付：电脑端准备完成，正式训练未开始

截至 2026-11-27 的这次准备工作，没有可连接的 S5、CEVA/CDNN 开发包或原执行输出，交付按“先完成电脑端准备”验收。提取、解析、整图实验、图片入口、候选框实验、数据/评估工具和未来原执行对照接口均已落地。正式训练条件仍未满足，不能把电脑端工具就绪写成原模型已完全还原。

一键检查（不会执行任何 optimizer step）：

```sh
# 无固件：检查工具、人工样例与保守的训练条件报告
.venv/bin/python prepare_offline.py --tools-only
# 有完整原始模型：把路径替换为自己的外部输入目录
.venv/bin/python prepare_offline.py --firmware-root /path/to/original-s5-v29
# 可选：同时用一张本地照片检查两份模型、三种排列的推理流程
.venv/bin/python prepare_offline.py --firmware-root /path/to/original-s5-v29 --photo /path/to/photo.jpg
```

macOS 安装 `.venv` 后可双击 `检查训练前准备.command`，默认运行无需固件的 `--tools-only` 检查。总结果在 `artifacts/computer-preparation.json`，每步日志与 SHA 在 `artifacts/preparation-logs/`。源代码摘要绑定本次运行；运行后修改代码须重新检查。工具模式完成标记为 `tools_only_checks_complete`；带原模型的流程完成标记为 `computer_side_preparation_complete`。两者仍输出 `ready_for_formal_training: false`。运行库组件缺失时会列在 `skipped_steps`；被跳过的研究不能算通过。`pretraining-readiness.json` 列出真实数据、原执行一致性、预处理、候选框和物理合焦基线的缺项。

该流程特意不运行旧 `probe_first_conv.py` 的人工优化演示；本次只做前向、梯度计算和校验，不更新模型。

## 图片、候选框和数据入口

```sh
.venv/bin/python photo_probe.py /path/to/photo.jpg --firmware-root /path/to/original-s5-v29 --model 44_hm_d_nw_sng --compare-layouts --output artifacts/my-photo
.venv/bin/python proposal_hypothesis.py --self-check
.venv/bin/python pretraining_readiness.py --firmware-root /path/to/original-s5-v29
```

图片入口的候选处理为 EXIF 朝向修正、Pillow 灰度转换、双线性拉伸到384×256、减117，缩放默认1；可显式设置 `--mean` / `--scale`。这些参数有格式线索支持，但尚未与原机输入核对。输出包含准备后的张量、原始 RPN 数值、实验候选框及完整假设记录，没有训练过程。

`proposal_hypothesis.py` 采用常规 Faster R-CNN anchor、bbox delta、排序和 NMS 解释，参考 [原实现的 proposal 层](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py)。它保留两种坐标约定；S5 实际约定、类别映射、过滤顺序和输出 ABI 尚未确定。公开实现的 anchor 注释例子比其零基坐标代码高1像素，自检按实际公式手算，不能照搬注释作为原机真值。

先前已用 scikit-image 官方仓库的 NASA astronaut 照片完成两模型×三排列的图片流程测试。该次本地运行的图片来源、摘要和用途记录在 `artifacts/public-test/provenance.json`；这个历史文件不会由源码克隆或任意一次 `--photo` 命令自动生成。它是公开测试图，不是训练/验证数据集。某候选排列给出高分不能证明框或类别正确，也没有据此自动选择训练模型。[图片来源说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

数据集、预测结果与原执行捕获的清单样例在 [templates/](../templates/)，详细字段和命令见 [pretraining_data_schema.md](../pretraining_data_schema.md)。工具提供文件摘要/尺寸/框边界校验、按序列防止划分泄漏、IoU0.5 AP及匹配轨迹ID切换统计、原执行卷积张量对照。它们的人工样例测试不计作 S5 性能验证；无原执行证据时不会输出训练就绪。

## 新增运行库证据

ARM 加载指令与字面量确定了识别运行库代码区 `41_hm_c_ddr.bin` 的基址为 `0x84268000`、数据区 `42_hm_d_ddr.bin` 的基址为 `0x84568000`。已核对10个类、86个虚表方法指针，包括 Input/Conv/Pool/Proposal。模型加载链检查 CDNN 标记、6/2类别并加载完整4MiB组件。定位结果和下一次需采集的输入内核地址见 [runtime_investigation.md](../runtime_investigation.md)，逐条 ARM 证据可在提供运行库组件与 Capstone 后重新生成到 `artifacts/runtime-arm-consumer.txt`。

这缩小了以后原执行核对的范围，未解出 XM4 指令语义或证明数值等价。官方工具资料显示 XM4 的解码/模拟需要对应 CEVA 工具；当前本地没有可用工具和原执行输出。

新增静态证据：全部8个分类头解包后，前25/后25行权重逐对余弦相似度低于−0.98，支持背景/目标的两组输出解释；输入辅助块16个uint32之和恰为384×256，支持校准直方图候选。3×3 的 `o4_i4_hw` 平滑性统计优于其余候选，但这只是优先调查顺序，不能升级成已证实排列。旧目录中的 `.cdnn` 各截掉了末尾16个有效参数区零字节，所有流程继续使用摘要固定的完整4MiB原件。

## 已完成的研究检查

以下包括先前使用完整本地输入得到的检查结果；新电脑实际完成了哪些检查，以自己的运行报告为准。缺少可选包或 ZIP 时不会复现相应交叉校验，也不会将它们标成通过。

- 核对原始 V2.9 更新包摘要、现有解密校验清单、离线模型 ZIP 和解密模型组件的一致性。
- 不依赖现有 TSV，直接扫描二进制记录，恢复层名、算子编号、输入连接、外部输入标记及模型输出端口；再与 TSV 逐项交叉校验。
- 完整保留原件，拆分头部、网络描述、参数候选区和 FF 填充；四段按顺序重组须逐字节等于原件。
- 导出机器可读图结构 `graph.json`、Graphviz 图 `graph.dot`、摘要与运行库/DFD 字符串位置索引。
- 恢复 66 / 62 个卷积类算子记录（含名为 `upsample` 的特殊记录）的数值张量，覆盖两份参数区全部字节。保留 DSP 打包形状，不冒充已恢复标准 OIHW 排列。
- 导出 `numeric_tensors.npz`：原始 int16 权重、原始 int32 偏置对，以及按已恢复缩放系数计算的 float32 候选值。
- 两份第一层候选实现加载原厂恢复权重，以 PyTorch 执行前向、反向和一次人工损失下降检查；另用独立整数累加核对前向算术。保存的 `first_conv_candidate.pt` 保持原始恢复值，不保存人工更新结果。
- 两模型完整候选图分别执行 111 / 95 条非 proposal 记录，保留全部原始提取值并显式选择 3×3 排列假设；三种假设均完成前向/反向和分支尺寸检查，输出差异也保存供后续排除假设。

| 组件 | 层数 | 类别标签 | 候选框节点 | 输出端口 |
| --- | ---: | --- | ---: | ---: |
| 43_hm_d_nw | 117 | background、person、person_face、animal、animal_face、bird、bird_face | 6 | 12 |
| 44_hm_d_nw_sng | 97 | background、person、person_face | 2 | 4 |

类别名和候选框节点不能证明实际识别性能、眼部定位能力、跟踪算法或模型切换条件。

## 运行

先在 `af-research/` 创建 Python 3.12 或更新版本的虚拟环境。这里列出的 macOS/Linux 命令使用 `.venv/bin/python`；Windows 创建环境可用 `py -3.12 -m venv .venv`，并将后续解释器路径换成 `.venv\Scripts\python.exe`。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_offline.py --tools-only
```

提取脚本本身只使用 Python 标准库；完整研究流程另需 requirements 中的 NumPy、PyTorch、Pillow 和 Capstone。需要研究原模型时，把外部目录设为 `--firmware-root` 或环境变量 `S5_AF_FIRMWARE_ROOT`。该目录可以直接放两份完整模型，也可以包含 `unpacked-v29/` 子目录。`--source` 可直接指定组件目录，优先级最高。未配置时保留旧项目的 `../s5-v29-offline/unpacked-v29` 回退路径，但这不是新电脑必须采用的目录结构。

```sh
.venv/bin/python extract.py --source /path/to/components --no-archive
# 有可选的旧交接 ZIP/TSV 时，额外做交叉检查
.venv/bin/python extract.py --source /path/to/components --archive /path/to/S5-V29-AF-models.zip
# 后续命令可复用环境变量，或各自传 --firmware-root
export S5_AF_FIRMWARE_ROOT=/path/to/original-s5-v29
.venv/bin/python prepare_offline.py
```

最小输入是摘要固定的 `43_hm_d_nw.bin` 和 `44_hm_d_nw_sng.bin`，各 4 MiB；裁切的 `.cdnn` 或实验修改固件不能代替它们。`verification.json`、原始 `S5___V29.bin` 和 `S5-V29-AF-models.zip` 均为可选交叉证据：manifest 位于组件目录；固件包和自动发现的 ZIP 可位于组件目录或其父目录。发现的证据不一致会报错，缺失则记为 `not_available`/`null`，不会算作通过。ZIP 可由 `--no-archive` 显式关闭；两份模型的固定 SHA-256 检查始终保留。

运行库研究另需 `03_program.bin`、`39_hm_c_prog.bin`、`40_hm_d_prog.bin`、`41_hm_c_ddr.bin`、`42_hm_d_ddr.bin`；缺少时完整流程会报告跳过该步。生成物在 `artifacts/`，默认不提交 Git。脚本不修改源固件、ZIP 或模拟器。

如要逐步执行当前流程，在同一 shell 设置好环境变量后运行：

```sh
.venv/bin/python extract.py
.venv/bin/python decode_parameters.py
.venv/bin/python verify_parameters.py --skip-first-conv-checkpoint
.venv/bin/python ops_investigation.py
.venv/bin/python packing_investigation.py
.venv/bin/python candidate_network.py --compare-layouts
.venv/bin/python verify_graph_hypotheses.py
```

`verify_parameters.py --skip-first-conv-checkpoint` 检查保存的 NPZ 与重新解析结果一致、所有原始数值可重新拼回原参数区，并验证损坏的权重形状、偏置副本、缩放系数和 proposal 块会被拒绝。旧 `probe_first_conv.py` 含一次人工优化演示，不在当前训练前准备命令中执行；历史 `first_conv_candidate.pt` 也不作为新电脑的必需输入。只有显式省略 `--skip-first-conv-checkpoint` 时，验证脚本才额外要求并检查该旧 checkpoint。

`candidate_network.py` 输入是程序生成的 `[1,1,256,384]` 人工浮点张量，不读取照片、不假设已恢复相机预处理。图中采用显式候选 ReLU/max-pool 和上采样通道映射；输出保存为 `artifacts/<模型>/graph-hypotheses/<排列>/candidate_rpn_outputs.npz`，每个排列都有对应 `candidate-network-probe.json`。它执行反向但不更新参数。`--spatial-layout` 可指定单个候选，默认 `hw_o4_i4` 只是实验配置。

`verify_graph_hypotheses.py` 检查原件身份、全部分支形状和排列往返，并以独立标量索引核对分类头的分块乘法；错误 slice、reshape、stride、激活副本与非零补齐通道必须被拒绝。

## 格式证据与尚未确定的内容

两份模型都有 `CDNN`、`XM4` 和 `PVANET-lite` 字面标记。`0x12c` 的小端整数分别是 117 和 97，与扫描到的记录数一致；`0x148` 为输出端口数，随后是 `(层编号, 端口编号)` 的小端整数对。

以 `0x110` 为起点，利用 `0x114`、`0x118` 的两个长度字段拆分得到：

| 组件 | 网络描述区，左闭右开 | 参数候选区，左闭右开 | 尾部 FF 字节数 |
| --- | --- | --- | ---: |
| 43 | 0x110–0x88c0 | 0x88c0–0x33bf60 | 802976 |
| 44 | 0x110–0x78f0 | 0x78f0–0x30c810 | 997360 |

这套边界在两份文件上都恰好落到纯 FF 尾部，且可无损重组。逐层记录的未知字段均按原值保留，不给它们编造语义。输入连接存在算子特例：编号 5 的三处 pooling 记录在输入对前多一个字节，已通过全部现有 TSV 输入关系校验。

## 数值参数恢复：2026-11-27

在每个卷积记录的输入表结束后，以该位置为 `q`，观察到权重描述的长度字段位于 `q+115`，偏置描述的长度字段位于 `q+183`。两组描述均包含字节数、四个存储维度、两个未知数字标签和数值区。当前只解释数值区的第一个 float32 缩放系数，其余原始字节保留；部分未知槽存在非有限 float 位型，不能一律当作有效缩放值。

- **权重**：全部满足 `字节数 = 存储维度乘积 × 2`，按小端 int16 恢复。数字标签是 `(24,8)`，它们的语义仍未知，不能把标签中的 8 直接解释成文件每元素占 8 位。
- **偏置**：逻辑形状为 `[输出通道,1,1,2]`，每个通道的两份小端 int32 完全一致；按 32 字节对齐。分类分支逻辑 50 通道但物理补至 52 通道，导出 float 偏置时裁去补齐部分。
- **缩放关系**：128 个记录全部满足 `bias_scale ≈ input_scale × weight_scale`，两模型最大相对误差分别为 `5.672e-8`、`4.815e-8`。这支持对称线性反量化候选解释；原设备零点、舍入、饱和及激活尚未逐项核验。
- **输入候选区**：首个记录的维度字段为 `[1,1,256,386]`，有效宽高字段为 `384×256`。对应 `197632` 字节的 int16 数组，其每行前 384 元素均为 14976；乘以相邻的 `1/128` 缩放字段得到 117。每行末两元素保留原值，含义未确认。不能据此认定真实传感器图像预处理已恢复。
- **辅助与 proposal**：输入候选区后有 64 字节辅助块，再顺序存放所有卷积权重和偏置。末尾恰好剩 6 / 2 个相同的 576 字节 proposal 参数块，开头两组 float 数列分别约为 `[0.5,0.667,1,1.5,2]` 和 `[3,6,9,16,32]`；比例/尺度语义仍属候选，余下查表内容未完整解释。

两份参数区分别 `3356320`、`3165984` 字节，已全部按上述记录定位且无余字节。int16 权重与 int32 偏置可逐字节重组，证据在 `artifacts/parameter-validation.json`。这不等于所有字段和运行语义已恢复。

第一层候选实现假定存储权重可直接重排为 `[32,1,4,4]`，stride=2、padding=1，暂不加入原厂激活。两份模型以非均匀人工输入运行，与独立整数参考的最大绝对误差分别约 `7.64e-7`、`8.45e-7`；反向梯度有限且非零，人工平方损失经一步 SGD 下降。该检查验证数值导出和候选层的计算/梯度管线，**没有与原 DSP 对照，也没有证明识别或对焦效果改善**。

`upsample` 的非零数值反量化后分别接近 `1/16`、`3/16`、`9/16`；出现次数为 `1024`、`2048`、`1024`，恰好对应 256 个 4×4 双线性核的数值直方图，另有 12288 个零。两模型都满足，最大系数差小于 `1e-4`。这支持固定双线性上采样的解释，但尚未证明物理通道映射和边界处理。相关统计记录在每份 `parameters.json` 的 `upsample_hypothesis` 中。

## 整图与排列调查：2026-11-27 续

**修正 dilation/padding 字段。** 旧解析把 `q+71..86` 四个整数统称 padding；1×1 层的值为 `[1,1,0,0]`，这样解释会错误扩大输出。现将前两个记作 dilation 候选、后两个记作对称 padding 候选，全部 24 个 concat 分支尺寸均吻合。旧字段 `padding_four_fields_candidate` 已删除。详细偏移见 [ops_investigation.md](../ops_investigation.md)。

**图结构约束。** 16 个显式 reshape 保持元素数，126 个普通卷积的图输入通道与存储参数量吻合。两份模型都沿 `256×384 → 128×192 → 64×96 → 32×48 → 16×24 → 8×12 → 16×24` 传播。池化需要 ceil-mode 才能合并分支；max/average 仍需运行证据。激活的两个位置在 128 个卷积类记录中一致，主干为1、输出头及 upsample为0，支持融合 ReLU 候选但未确认精确运算。

**1×1 分块证据。** 八个分类头逻辑50通道、物理52通道。按 `[O/4,I/4,O4,I4]` 解释时，两个补齐输出通道的896个数全部为零；直接 OI 重排却把446–448个非零数放在补齐位置。因而可以排除该直接 OI 解释，但块内输入顺序等细节仍需原执行结果验证。

**3×3 尚未唯一确定。** `packing_investigation.py` 保留 `o4_i4_hw`、`hw_o4_i4`、`o4_hw_i4` 三种排列，均可无损往返。整图三种候选都能执行、产生有限输出，并为全部参数计算有限且非零的梯度；这无法决定哪种排列正确。相同人工输入下，两模型框回归的最大候选间差异分别约0.1594和0.2593，记录在 `graph-hypotheses/layout-comparison.json`。不能凭“能跑”选择一个开始正式训练。

**upsample 的新增限制。** 它的16,384个int16若分为256个64元素块，前128块相同且每块32个非零，后128块全部为零。它可能是重复系数表加零区；仅凭直方图无法恢复通道映射。整图实验用恢复出的三个系数构造独立通道的双线性候选核，明确属于假设，不是已解码的原厂矩阵。

**候选图范围。** 候选图执行到六/两份 `[1,50,16,24]` 分类张量及共享 `[1,100,16,24]` 框回归张量。每格25个 anchor、总9600个候选位置与记录相容。后续已增加常规 proposal 实验解码器，但 S5 的实际坐标/过滤/NMS和原厂输出 ABI 尚未恢复，因此输出不是已验证的识别框。真实照片预处理、定点舍入/饱和、原DSP对照仍是正式微调前的门槛。

`42_hm_d_ddr.bin` 有 CDNN/XM4 运行库和 Proposal 相关字符串；`03_program.bin` 有 `dfd_zm`、`dfd_me_zm` 图像处理输入/输出配置字符串。这些目前只是定位线索，尚无可验证的 DFD 测距、合焦判断或镜头控制调用链。

CEVA 官方说明 CDNN 将离线训练模型转换成面向嵌入式推理的网络，并进行量化、融合等优化。因此，找到部署文件不等于拿到可直接继续训练的 checkpoint；当前 S5 文件的具体编译版本与配置也未确认。参考：[CEVA CDNN 工作流](https://www.ceva-ip.com/wp-content/uploads/E-Book-03-AI-Computer-Vision.pdf)。

## 下一阶段的验收点

1. **恢复可运行原厂识别网络**：优先用运行库消费代码或原DSP逐层输出排除3×3排列和upsample映射歧义，然后核对预处理、激活/重标定、softmax 与 proposal。当前整图是可运行的假设检验平台，不能当成已验证原网络。预处理和框解码必须验证，不能用随机权重替代原厂权重宣称完成。
2. **建立训练和评估**：主体识别需要带框/类别标签的照片与视频，追踪评估还需连续身份标签；按拍摄序列拆分训练/验证，避免相邻帧泄漏。低光、遮挡、运动、逆光单独统计，先建立原网络基线，再微调。
3. **拆 DFD 与镜头闭环**：识别框只是输入之一。需要定位散焦统计、镜头/光学参数、运动预测、驱动指令和到位反馈。静态精度与低光合焦需有焦点位置及镜头状态的实测数据，单靠主体框标注无法验证。
4. **部署校验**：训练结果需重新转换成设备兼容格式，核对尺寸、数值误差、算子支持、延迟和内存。编译器或等效封装方案尚未具备；不能保证新模型能回填原厂固件。

目前 QEMU 菜单运行不构成传感器、识别 DSP、DFD 和镜头闭环的验证环境。因此现阶段也不能用菜单模拟器证明新对焦算法有效。
