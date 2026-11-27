# S5 对焦研究与数据采集项目

面向第一代 **Panasonic LUMIX DC-S5**，同时准备主体识别/追踪和静物/弱光对焦研究。项目包含模型解析、候选网络实验、数据评估接口、机主采集指南和文件交接工具。

**目前停在正式训练之前。** 两份网络的整数参数已提取，候选计算图和图片输入流程能运行；原厂 DSP 数值一致性、预处理、候选框解释和物理对焦控制仍未验证。没有训练后模型，也没有可以刷回相机的对焦改进固件。

## 发给 S5 机主

直接转发 [这段消息](docs/给S5机主的转发消息.md)，附上 [完整采集指南](docs/S5机主采集指南.md) 或 [下载机主采集包 ZIP](https://github.com/LumiMod/s5-emulator/raw/refs/heads/main/af-research/dist/S5-owner-collection-kit.zip)。ZIP 内含说明、空目录、记录表和可选校验工具；机主无需安装开发环境。

第一轮只需约 15 分钟试采：

| 要交的内容 | 用途 |
|---|---|
| 机身/镜头版本及设置照片 | 保留测试条件 |
| 正常光、暗光各 6 次静物 RAW+JPEG | 检查重新合焦与失败情况 |
| 走近退远、暗光走动、横移遮挡各 1 段原视频 | 观察跟焦、遮挡恢复与拉风箱 |
| 同时用手机拍相机 LCD | 记录可见对焦框、提示和操作时序 |
| 正常光、暗光各约 7 张手动对焦参考 | 比较同构图下可达到的清晰度 |

使用相机正常功能即可。保留失败样本；不知道的参数留空。普通机主不需要采集内部 DSP 张量或镜头编码器。收到试采并核对后再扩大数量。

**这是公开代码仓库。相机原片、人像、设置照片和个人信息通过双方约定的私密渠道交接。** 项目不自动上传文件。

## 整理采集文件

Python 3.10+，只用标准库；不需要固件、QEMU、NumPy 或 PyTorch。在本目录执行：

```sh
python3 collection.py init sessions/pilot01 --session-id pilot01
# 按编号复制相机/手机原文件，填写实际设置后：
python3 collection.py inventory sessions/pilot01
python3 collection.py check sessions/pilot01
python3 collection.py pack sessions/pilot01 --output private-data/pilot01.zip
```

详细步骤见 [采集工具说明](docs/COLLECTION_TOOLS.md)。检查范围是目录、文件摘要和事件引用；不会解码全部媒体、自动标注或证明真实合焦。研究者再按 [数据契约](pretraining_data_schema.md) 整理标注、预测和原执行对照。

## 研究代码：安装和复现

研究环境需要 **Python 3.12+**。以下命令在 `af-research/` 内运行；Windows 激活虚拟环境后把 `.venv/bin/python` 换成 `python`。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_collection.py
# 无需固件：运行人工样例测试、候选框自检及保守的训练就绪检查
.venv/bin/python prepare_offline.py --tools-only
```

完整模型实验需要在本地准备原始 V2.9 中的两份完整组件，代码会验证固定摘要。原固件和模型权重不包含在这个子项目的上传内容中；不会从网络自动下载，也不把仓库中的实验补丁固件当作原件。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `43_hm_d_nw.bin` | 4,194,304 字节 | `d74e1f38fa7ce7a40ed49e6ffc3fd65bae70ae77eb9d37add2f896abc52897d6` |
| `44_hm_d_nw_sng.bin` | 4,194,304 字节 | `b510989f1e12fd1e6a2efbc5306b125289cb796c48e5ffefe7e4d4ce26390a50` |

```sh
# 输入目录直接包含上述 .bin，或其下包含 unpacked-v29/
.venv/bin/python prepare_offline.py --firmware-root /path/to/original-components
# 同时用一张本地图片检查两模型 × 三种排列的图片流程
.venv/bin/python prepare_offline.py --firmware-root /path/to/original-components --photo /path/to/photo.jpg
# 或统一配置所有研究脚本的输入位置
export S5_AF_FIRMWARE_ROOT=/path/to/original-components
.venv/bin/python photo_probe.py /path/to/photo.jpg --model 44_hm_d_nw_sng --compare-layouts
```

也支持 `--source` 直接指定组件目录。ZIP/TSV、原更新包和解密清单是可选交叉证据；缺失明确记录为未核对，存在却不匹配会拒绝。若同时提供 `03_program.bin`、`39_hm_c_prog.bin`、`40_hm_d_prog.bin`、`41_hm_c_ddr.bin`、`42_hm_d_ddr.bin`，还会运行加载链/运行库调查；否则明确记录跳过该步骤。

运行产物保存在 `artifacts/`，默认不提交。`computer-preparation.json` 保存各步骤结果、日志摘要和源文件摘要。整个 `prepare_offline.py` 流程只做解析、前向、梯度检查和校验，**不执行优化器更新**。旧 `probe_first_conv.py` 单独运行会做一次人工损失演示，故不在准备流程内。

## 证据、边界与下一步

- [本次复现记录](evidence/README.md)：成功运行的范围、摘要与缺项。
- [研究笔记](docs/RESEARCH_NOTES.md)：参数恢复、候选图、三种权重排列与现有证据。
- [算子调查](ops_investigation.md)、[运行库调查](runtime_investigation.md)：已确认字段和仍待原执行验证的解释。
- [数据和评估契约](pretraining_data_schema.md)：标注、AP50、轨迹、物理焦点数据与原运行张量的边界。

接下来先检查机主试采，建立真实图片/视频和可见对焦表现基线，再整理标注和独立测试集。原网络的数值还原仍需要可靠的原执行输入/输出或对应开发工具；普通相机照片不能补齐这项证据。没有这些条件时，脚本会继续报告 `ready_for_formal_training: false`。

## 开发检查与重新生成机主包

```sh
python3 verify_collection.py
python3 build_owner_kit.py
```

采集工具测试只用人工文件验证完整性流程，不将人工样例算作相机性能数据。机主 ZIP 可由已提交说明和脚本重新生成，内含空白模板，不含采集素材。

仓库内另有 [GitHub Actions 配置示例](ci/af-collection.yml.example)。当前上传令牌不含 `workflow` 权限，故未启用远端 CI；有相应权限的维护者可将它复制为 `.github/workflows/af-collection.yml`。本次复现证据来自实际本地运行。
