#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' '未找到本项目的 .venv。请先在 af-research 目录执行：' \
        'python3.12 -m venv .venv' \
        '.venv/bin/python -m pip install -r requirements.txt' >&2
    exit 1
fi
# Finder double-click checks tools without requiring private firmware inputs.
# Terminal callers can supply --firmware-root /path/to/original-s5-v29 instead.
if [[ $# -eq 0 ]]; then
    set -- --tools-only
fi
exec .venv/bin/python prepare_offline.py "$@"
