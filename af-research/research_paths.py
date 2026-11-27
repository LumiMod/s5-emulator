"""Portable input discovery shared by the offline research commands.

Inputs stay outside this repository. A firmware root can contain unpacked-v29/
or be the directory containing the complete extracted component .bin files.
"""
import os
from pathlib import Path

ENVIRONMENT_VARIABLE = 'S5_AF_FIRMWARE_ROOT'
BASE = Path(__file__).resolve().parent


def firmware_root(explicit=None):
    value = explicit or os.environ.get(ENVIRONMENT_VARIABLE)
    return Path(value).expanduser().resolve() if value else BASE.parent / 's5-v29-offline'


def component_source(root=None, source=None):
    if source is not None:
        return Path(source).expanduser().resolve()
    root = firmware_root(root)
    if (root / '43_hm_d_nw.bin').is_file() or root.name == 'unpacked-v29':
        return root
    return root / 'unpacked-v29'


def add_firmware_arguments(parser):
    parser.add_argument('--firmware-root', type=Path,
                        help='External original V2.9 inputs; overrides S5_AF_FIRMWARE_ROOT')
    parser.add_argument('--source', type=Path,
                        help='Directory of complete extracted components; overrides --firmware-root')


def resolve_source(args):
    return component_source(args.firmware_root, args.source)
