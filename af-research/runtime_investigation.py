#!/usr/bin/env python3
"""Locate original S5 CDNN runtime consumers; does not emulate CEVA instructions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from research_paths import add_firmware_arguments, resolve_source
import struct

from extract import require, write_json


ARM_BASE = 0x06280000
CODE_BASE = 0x84268000
DATA_BASE = 0x84568000
CLASSES = ['10InputLayer', '9ConvLayer', '14XM4InputKernel', '13XM4PoolKernel',
           '13XM4ConvKernel', '17XM4ProposalKernel', '14XM4SliceKernel',
           '15XM4ConcatKernel', '16XM4ReshapeKernel', '17XM4UpsampleKernel']


def u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(source):
    files = {name: (source / name).read_bytes() for name in
             ['03_program.bin', '39_hm_c_prog.bin', '40_hm_d_prog.bin', '41_hm_c_ddr.bin', '42_hm_d_ddr.bin']}
    program, code, data = files['03_program.bin'], files['41_hm_c_ddr.bin'], files['42_hm_d_ddr.bin']
    require(u32(program, 0x560858) == CODE_BASE, 'Unexpected code base literal')
    require(u32(program, 0x56085c) == DATA_BASE, 'Unexpected data base literal')
    require(CODE_BASE + len(code) == DATA_BASE, 'Code segment size does not reach data base')
    classes = []
    for name in CLASSES:
        offset = data.index(name.encode() + b'\0')
        rtti = (offset + len(name) + 1 + 3) & ~3
        require(u32(data, rtti + 4) == DATA_BASE + offset, 'RTTI name pointer/base mismatch')
        require(u32(data, rtti + 12) == 0, 'Unexpected vtable offset-to-top')
        require(u32(data, rtti + 16) == DATA_BASE + rtti, 'Vtable RTTI pointer mismatch')
        methods = []
        cursor = rtti + 20
        while cursor + 4 <= len(data):
            va = u32(data, cursor)
            if not CODE_BASE <= va < DATA_BASE:
                break
            file_offset = va - CODE_BASE
            snippet = code[file_offset:file_offset + 64]
            require(snippet and snippet != bytes(len(snippet)) and set(snippet) != {255},
                    'Vtable points into empty code')
            methods.append({'vtable_slot_file_offset': cursor, 'code_va': va,
                            'code_file': '41_hm_c_ddr.bin', 'code_file_offset': file_offset,
                            'first_64_bytes_hex': snippet.hex(), 'first_64_bytes_sha256': sha(snippet)})
            cursor += 4
        require(len(methods) >= 5, 'Too few method pointers')
        classes.append({'class_name': name, 'name_file_offset': offset, 'rtti_file_offset': rtti,
                        'vtable_methods_start_file_offset': rtti + 20, 'method_candidates': methods,
                        'method_semantics': 'Vtable order only; individual method names and instructions undecoded.'})
    return {'component_digests': {name: sha(blob) for name, blob in files.items()},
            'runtime_code_base': CODE_BASE, 'runtime_code_bytes': len(code), 'runtime_data_base': DATA_BASE,
            'base_evidence': {'code_literal_file_offset': 0x560858, 'data_literal_file_offset': 0x56085c,
                              'code_base_load_va': 0x067e05c8, 'code_pointer_store_va': 0x067e05ce,
                              'data_base_load_va': 0x067e05d2, 'data_pointer_store_va': 0x067e05d6,
                              'code_copy_call_va': 0x067e073c, 'code_copy_bytes': 0x300000,
                              'data_copy_call_va': 0x067e0752, 'data_copy_bytes': 0x20000},
            'model_load_checks': {'loader_va': 0x067e0702, 'magic_compare_va': 0x067e0770,
                                  'multi_class_count_compare_va': 0x067e0788, 'multi_class_count': 6,
                                  'human_class_count_compare_va': 0x067e07b4, 'human_class_count': 2,
                                  'model_load_bytes': 0x400000,
                                  'registration_caller_va': 0x0679764c,
                                  'registration_call_va': 0x067976ae,
                                  'registration_target_va': 0x067d1770},
            'classes': classes,
            'limitations': ['Code base is established through ARM consumer literals/stores and segment extent.',
                            'CEVA XM4 target supported by CDNN header and concrete XM4 kernel RTTI/vtables.',
                            'No XM4 instruction decoder or simulator used; exact ISA encoding/revision unverified.',
                            'No original input kernel execution or numerical equivalence against reconstructed network.',
                            'DataMean string is a lead, not proof of complete preprocessing formula.']}


def inspect_secondary_model_files(source, folder):
    result = []
    for filename, original in [('s5_af_multi_6cls.cdnn', '43_hm_d_nw.bin'),
                               ('s5_af_human_2cls.cdnn', '44_hm_d_nw_sng.bin')]:
        path = folder / filename
        if not path.exists():
            continue
        trimmed, full = path.read_bytes(), (source / original).read_bytes()
        declared_end = 0x110 + u32(full, 0x114) + u32(full, 0x118)
        result.append({'file': str(path), 'sha256': sha(trimmed), 'bytes': len(trimmed),
                       'matches_original_prefix': trimmed == full[:len(trimmed)],
                       'header_declared_payload_end': declared_end,
                       'missing_declared_payload_bytes': max(0, declared_end - len(trimmed)),
                       'missing_bytes_hex': full[len(trimmed):declared_end].hex()})
    return result


def dump_arm_evidence(program, output):
    try:
        from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB
    except ImportError:
        return {'available': False}
    disassembler = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    blocks = [(0x5605bc, 0x56061e), (0x560702, 0x5607c4),
              (0x51764c, 0x5176be), (0x560822, 0x560854)]
    lines = []
    for start, end in blocks:
        lines.append('BLOCK file+0x%x VA 0x%x' % (start, ARM_BASE + start))
        for inst in disassembler.disasm(program[start:end], ARM_BASE + start):
            lines.append('0x%08x %-12s %-9s %s' % (inst.address, inst.bytes.hex(), inst.mnemonic, inst.op_str))
        lines.append('')
    for offset in range(0x560854, 0x560888, 4):
        lines.append('literal file+0x%x = 0x%08x' % (offset, u32(program, offset)))
    (output / 'runtime-arm-consumer.txt').write_text('\n'.join(lines) + '\n')
    return {'available': True, 'output': 'runtime-arm-consumer.txt'}


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output', type=Path, default=base / 'artifacts')
    parser.add_argument('--secondary-models', type=Path, help='Optional prior trimmed artifacts for comparison; omitted by default')
    args = parser.parse_args()
    args.source = resolve_source(args)
    report = analyze(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    report['secondary_model_files'] = inspect_secondary_model_files(args.source, args.secondary_models) if args.secondary_models else []
    report['arm_disassembly'] = dump_arm_evidence((args.source / '03_program.bin').read_bytes(), args.output)
    write_json(args.output / 'runtime-investigation.json', report)
    print('Located', len(report['classes']), 'runtime class vtables;',
          sum(len(c['method_candidates']) for c in report['classes']), 'method-pointer entries')
    print('code base', hex(CODE_BASE), 'data base', hex(DATA_BASE), 'ARM evidence', report['arm_disassembly'])


if __name__ == '__main__':
    main()
