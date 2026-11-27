#!/usr/bin/env python3
"""Build a deterministic, firmware-free owner handoff kit from tracked sources."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import zipfile

from collection import initialize

BASE = Path(__file__).resolve().parent
NAME = 'S5-owner-collection-kit'
SOURCES = [
    ('docs/给S5机主的转发消息.md', '给S5机主的转发消息.md'),
    ('docs/S5机主采集指南.md', 'S5机主采集指南.md'),
    ('docs/采集记录表.md', '采集记录表.md'),
    ('docs/COLLECTION_TOOLS.md', 'docs/COLLECTION_TOOLS.md'),
    ('pretraining_data_schema.md', 'pretraining_data_schema.md'),
    ('collection.py', 'collection.py'),
]
START = '''S5 机主采集包（第一代 DC-S5）

1. 先读《给S5机主的转发消息.md》和《S5机主采集指南.md》。
2. sessions/pilot01 内已建好六组空目录。相机原文件放 camera，手机拍屏放 phone，设置照片放 settings。
3. 填《采集记录表.md》即可；不懂 JSON 不需要改，让研究者协助整理。未知值留空。
4. S06 的正常光、暗光分别写清文件名。失败、超时也保留并说明。
5. 私下发送采集目录及记录表，不往公开 GitHub 上传原片。

无需安装软件即可按目录放文件。collection.py 是给会用 Python 3.10+ 的协作者使用的可选清点工具，不连接或修改相机。
可选命令（在本包目录执行，放入素材后）：
  python3 collection.py inventory sessions/pilot01
  python3 collection.py check sessions/pilot01
  python3 collection.py pack sessions/pilot01 --output private-data/pilot01.zip
详细命令见 docs/COLLECTION_TOOLS.md。整理时请将记录表也复制到 sessions/pilot01/settings/。

公开项目：https://github.com/LumiMod/s5-emulator/tree/main/af-research
'''


def build():
    entries = {'先读我.txt': START.encode('utf-8')}
    entries.update({dest: (BASE / source).read_bytes() for source, dest in SOURCES})
    # Fresh template contains no captured images, private metadata or timestamps.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / 'pilot01'
        with contextlib.redirect_stdout(io.StringIO()):
            initialize(root, 'pilot01')
        for path in sorted(root.rglob('*')):
            name = 'sessions/pilot01/' + path.relative_to(root).as_posix()
            entries[name + '/' if path.is_dir() else name] = b'' if path.is_dir() else path.read_bytes()
    output = BASE / 'dist' / (NAME + '.zip')
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(NAME + '/' + name, date_time=(2026, 11, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o40755 if name.endswith('/') else 0o100644) << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {NAME + '/' + name for name in entries}
        for name, data in entries.items():
            assert archive.read(NAME + '/' + name) == data
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(digest + '  ' + output.name + '\n', encoding='ascii')
    print(output.name, output.stat().st_size, 'bytes; SHA256', digest)
    return output


if __name__ == '__main__':
    build()
