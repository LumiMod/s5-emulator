# 采集文件怎么整理、校验和交接

机主可以直接使用下载包内的空目录与表格，不必安装 Python。下面命令给协助整理文件的研究者使用；只需 Python 3.10 以上的标准库，不需要 PyTorch、QEMU 或固件。

## 生成采集目录

在仓库的 `af-research` 目录运行：

```sh
python3 collection.py init sessions/pilot01 --session-id pilot01
```

已有目录不会被覆盖。拍摄后把相机卡中的文件原样复制到对应 `raw/<场景>/camera/`，手机拍屏原文件放到 `raw/<场景>/phone/`，菜单/机身镜头版本照片放 `settings/`。不要先经过聊天软件压缩，不要把 RAW 转 JPEG 后丢弃原件。

```text
pilot01/
  session.json              机身、镜头、版本、日期等
  sequences.json            每段实际使用的模式与曝光设置
  events.csv                事件时间、相机文件与拍屏文件对应关系
  settings/                 菜单和版本照片
  raw/
    S01_static_normal/camera/
    S01_static_normal/phone/
    ...
```

`sequences.json` 中的 AFS/AFC/MF 是试采计划；拍完须按实际设置修改，不能将计划当成观测结果。未知数据保留 `null`。不懂 JSON 的机主可只附文字/语音说明，由研究者回填；保留说明原件和回填者记录。

## 对齐事件

`events.csv` 的时间都是从该文件开头计算的秒数，不是手机或相机墙上时间。两台设备的时钟不必相同。用拍摄开始时双方都可见的遮挡/手势作为同步点，记录两个文件各自对应时间；对焦时间的精度受录制帧率、画面延迟和按键可见性限制。

建议 event 值：`sync`、`focus_press`、`focus_confirm_visible`、`capture`、`occlusion_start`、`occlusion_end`、`manual_reference`。`focus_confirm_visible` 只代表屏幕上的合焦提示，不等于图像实际清晰。看不到按键起点时不要猜时间，留空并写明。

phone_file/camera_file 填各自 camera/phone 目录内的相对文件名。例如：

```csv
sequence_id,trial_id,phone_file,phone_time_s,camera_file,camera_time_s,event,observation,notes
S03_approach_normal,t01,screen_01.mp4,2.40,P1000012.MP4,0.80,sync,两画面出现手掌,时间需逐帧核对
```

以上是格式示例，不是真实采集数据。每个场景的事件可多行；没有可靠时间也可先提交试采，后续由研究者整理。

## 清点与校验

```sh
python3 collection.py inventory sessions/pilot01
python3 collection.py check sessions/pilot01
```

inventory 生成包含每个文件大小和 SHA-256 的清单；check 检查漏文件、改动、未登记新增文件、路径和事件引用。修改元数据或加入文件后，重新执行 inventory。它不会解码全部 RAW/视频，也不会自动证明标注、拍摄来源或对焦质量。`ready_for_research_review` 仅表示有素材可交给研究者检查，`ready_for_training` 始终为 false。

## 私下打包

```sh
python3 collection.py pack sessions/pilot01 --output private-data/pilot01.zip
```

ZIP 使用 ZIP64、流式读取和存储模式，适合已压缩视频；打包后逐文件核对 ZIP 内容，并生成 `pilot01.zip.sha256`。需要额外约一份原素材大小的磁盘空间。现有 ZIP 不会覆盖。把 ZIP 和校验文件通过双方约定的私有网盘或传输方式交给研究者，勿上传公开 GitHub 的 Issue/Release。

`sessions/`、`private-data/` 与常见 RAW/视频扩展名已忽略，但 Git 忽略规则不保证任意位置的照片都不会被提交。采集素材统一放在上述目录。

## 研究者收到后

1. 核对 ZIP 摘要，解压后执行 `collection.py check`；再用照片/视频软件打开原件，确认可解码、拍屏可读、序列齐全。
2. 先检查试采是否满足目的，再安排正式采集，不要求机主先拍几千张。
3. 保留采集包不变；在另一个工作目录提取帧、标注类别框与身份，按完整拍摄序列划分训练/验证/测试。
4. 根据 [数据契约](../pretraining_data_schema.md) 生成研究用 dataset.json。采集包不自动成为训练集：空标注不能被当成负样本，普通视频也不能冒充内部 DSP 输出。
5. 镜头位置和照度未知时不填写伪数值；普通机主素材可建立可见合焦表现基线，真实编码器误差仍须另有可验证硬件记录。
