#!/usr/bin/env python3
"""Evidence-preserving S5 V2.9 CDNN extraction; NOT a trainable model converter."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile

from research_paths import add_firmware_arguments, resolve_source

MODELS = {
    '43_hm_d_nw': 'd74e1f38fa7ce7a40ed49e6ffc3fd65bae70ae77eb9d37add2f896abc52897d6',
    '44_hm_d_nw_sng': 'b510989f1e12fd1e6a2efbc5306b125289cb796c48e5ffefe7e4d4ce26390a50',
}
FIRMWARE_SHA256 = 'b2ec4aeec209693b874df8dfdf11fc85b38fa4675b44ec554b7a167f430827e1'
MODEL_BASE = 0x110
EXTERNAL = 0xffffffff


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def decode(data):
    require(len(data) == 4 * 1024 * 1024, 'Unexpected component size')
    require(data[:4] == b'CDNN', 'Not CDNN')
    require(data[0x104:0x108] == b'XM4\0', 'Unexpected target marker')
    require(data[0x120:0x12c] == b'PVANET-lite\0', 'Unexpected network marker')
    graph_end = MODEL_BASE + u32(data, 0x114)
    parameters_end = graph_end + u32(data, 0x118)
    require(MODEL_BASE < graph_end < parameters_end <= len(data), 'Invalid region boundaries')
    require(set(data[parameters_end:]) == {255}, 'Unexpected non-FF tail')
    nodes = []
    # Independently locate length-prefixed layer names, without using the TSV.
    for match in re.finditer(b'\x1c\x00\x00\x00', data[MODEL_BASE:graph_end]):
        offset = MODEL_BASE + match.start()
        size = u32(data, offset + 4)
        if not 1 < size < 100:
            continue
        name = data[offset + 8:offset + 8 + size]
        if not re.fullmatch(rb'[A-Za-z0-9_/.-]+\x00', name):
            continue
        name_offset = offset + 8
        cursor = name_offset + size
        operator_id = u32(data, name_offset - 12)
        input_count = data[cursor + 6]
        require(input_count <= 16, 'Implausible input count')
        # Pooling (observed ID 5) has one additional byte before input pairs.
        input_offset = cursor + 7 + (1 if operator_id == 5 else 0)
        inputs = []
        for i in range(input_count):
            layer, port = struct.unpack_from('<II', data, input_offset + i * 8)
            require((layer == EXTERNAL) == (port == EXTERNAL), 'Partial external-input sentinel')
            inputs.append({'layer': None if layer == EXTERNAL else layer,
                           'port': None if port == EXTERNAL else port})
        nodes.append({
            'index': len(nodes), 'name': name[:-1].decode('ascii'),
            'name_offset': name_offset, 'record_start': name_offset - 20,
            'operator_id': operator_id,
            'raw_prefix_words': list(struct.unpack_from('<5I', data, name_offset - 20)),
            'raw_flags': list(data[cursor:cursor + 2]),
            'raw_field_after_flags': u32(data, cursor + 2),
            'inputs': inputs,
        })
    require(len(nodes) == u32(data, 0x12c), 'Layer count mismatch')
    require(len({node['name'] for node in nodes}) == len(nodes), 'Duplicate layer name')
    for index, node in enumerate(nodes):
        end = nodes[index + 1]['record_start'] if index + 1 < len(nodes) else graph_end
        require(end > node['name_offset'], 'Invalid record extent')
        node['record_end'] = end
        for inp in node['inputs']:
            require(inp['layer'] is None or 0 <= inp['layer'] < index, 'Non-topological input')
    count = u32(data, 0x148)
    require(count <= 32, 'Implausible output count')
    outputs = []
    for i in range(count):
        layer, port = struct.unpack_from('<II', data, 0x14c + i * 8)
        require(layer < len(nodes) and port < 16, 'Invalid output')
        outputs.append({'layer': layer, 'port': port, 'name': nodes[layer]['name']})
    classes = []
    for offset in range(0x20, 0x100, 32):
        label = data[offset:offset + 32].split(b'\0', 1)[0]
        if label:
            classes.append(label.decode('ascii'))
    return {'signature': 'CDNN', 'target_marker': 'XM4', 'network_marker': 'PVANET-lite',
            'classes': classes, 'layers': nodes, 'outputs': outputs,
            'regions': {'header': [0, MODEL_BASE], 'network_description': [MODEL_BASE, graph_end],
                        'parameter_candidate': [graph_end, parameters_end],
                        'ff_padding': [parameters_end, len(data)]},
            'limitations': ['Parameter region semantics inferred from header sizes and FF boundary.',
                            'Operator semantics, tensor layout, quantization and preprocessing unverified.',
                            'Raw record extents are candidate boundaries based on adjacent headers.',
                            'No runnable inference, training, lens control or repacking claimed.']}


def check_tsv(parsed, text):
    rows = list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')), delimiter='\t'))
    require(len(rows) == len(parsed['layers']), 'TSV count mismatch')
    for row, node in zip(rows, parsed['layers']):
        require(int(row['层编号']) == node['index'] and row['层名'] == node['name'], 'TSV layer mismatch')
        require(int(row['层名偏移'], 16) == node['name_offset'], 'TSV offset mismatch')
        require(int(row['算子编号']) == node['operator_id'], 'TSV operator mismatch')
        expected = [(int(a), int(b)) for a, b in re.findall(r'(\d+)\.(\d+) ', row['输入〈层.端口〉'])]
        actual = [(p['layer'], p['port']) for p in node['inputs'] if p['layer'] is not None]
        require(actual == expected, 'TSV input mismatch')
        require(int(row['外部输入标记']) == sum(p['layer'] is None for p in node['inputs']),
                'TSV external-input mismatch')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def extract(source, archive, output):
    """Require pinned complete model components; cross-check other evidence if present.

    A missing package/manifest/ZIP is recorded as unverified, never passed.
    If supplied evidence exists but disagrees, extraction fails closed.
    """
    manifest = source / 'verification.json'
    verification = json.loads(manifest.read_text()) if manifest.is_file() else None
    trusted = {row['file']: row for row in verification['components']} if verification else {}
    if verification:
        require(verification['source_sha256'] == FIRMWARE_SHA256, 'Manifest names an unexpected firmware package')
    firmware = source.parent / 'S5___V29.bin'
    if not firmware.is_file() and (source / 'S5___V29.bin').is_file():
        firmware = source / 'S5___V29.bin'
    firmware_checked = firmware.is_file()
    if firmware_checked:
        require(digest(firmware.read_bytes()) == FIRMWARE_SHA256, 'Firmware package digest mismatch')
    summary = {'firmware_sha256_verified': FIRMWARE_SHA256 if firmware_checked else None,
               'firmware_package_check': 'passed' if firmware_checked else 'not_available',
               'manifest_check': 'passed_for_models' if verification else 'not_available',
               'zip_sha256': digest(archive.read_bytes()) if archive is not None else None,
               'zip_check': 'passed' if archive is not None else 'not_available',
               'identity_scope': 'Each complete model component is checked against the repository-pinned original SHA-256; package provenance is a separate optional check.',
               'models': {}}
    # Validate both files completely before creating extraction results.
    pending = []
    z = zipfile.ZipFile(archive) if archive is not None else None
    try:
        if z is not None:
            require(z.testzip() is None, 'ZIP CRC failure')
        for stem, expected in MODELS.items():
            filename = stem + '.bin'
            data = (source / filename).read_bytes()
            require(digest(data) == expected, 'Unexpected source digest: ' + filename)
            if verification:
                require(filename in trusted and trusted[filename]['sha256'] == expected and trusted[filename]['matches_component_table'],
                        'Firmware verification manifest mismatch: ' + filename)
            if z is not None:
                require(z.read(filename) == data, 'ZIP/source mismatch: ' + filename)
            parsed = decode(data)
            if z is not None:
                check_tsv(parsed, z.read(stem + '.layers.tsv').decode('utf-8-sig'))
            pending.append((stem, data, parsed))
    finally:
        if z is not None:
            z.close()
    output.mkdir(parents=True, exist_ok=True)
    for stem, data, parsed in pending:
        folder = output / stem
        folder.mkdir(exist_ok=True)
        (folder / 'original.bin').write_bytes(data)
        parsed['source_sha256'] = digest(data)
        regions = []
        rebuilt = bytearray()
        for name, (start, end) in parsed['regions'].items():
            blob = data[start:end]
            (folder / (name + '.bin')).write_bytes(blob)
            rebuilt.extend(blob)
            regions.append({'name': name, 'offset': start, 'size': len(blob), 'sha256': digest(blob)})
        require(bytes(rebuilt) == data, 'Region reconstruction failed')
        write_json(folder / 'graph.json', parsed)
        lines = ['digraph CDNN {', 'rankdir=LR;']
        for node in parsed['layers']:
            lines.append('n%d [label=%s];' % (node['index'], json.dumps(node['name'])))
            for inp in node['inputs']:
                if inp['layer'] is not None:
                    lines.append('n%d -> n%d [label="port %d"];' % (inp['layer'], node['index'], inp['port']))
        (folder / 'graph.dot').write_text('\n'.join(lines) + '\n}\n')
        summary['models'][stem] = {'sha256': digest(data), 'layers': len(parsed['layers']),
                                   'classes': parsed['classes'], 'output_ports': len(parsed['outputs']),
                                   'regions': regions, 'pinned_model_sha256_verified': True,
                                   'manifest_matches_component': True if verification else None,
                                   'zip_matches_component': True if archive is not None else None,
                                   'independent_graph_matches_existing_tsv': True if archive is not None else None,
                                   'region_roundtrip_identical': True}
    runtime_index = {}
    for filename in ['42_hm_d_ddr.bin', '03_program.bin']:
        path = source / filename
        if not path.is_file():
            runtime_index[filename] = {'status': 'not_available'}
            continue
        data = path.read_bytes()
        if verification:
            require(filename in trusted and digest(data) == trusted[filename]['sha256'], 'Runtime manifest mismatch: ' + filename)
        entries = []
        for match in re.finditer(rb'[\x20-\x7e]{6,}', data):
            if re.search(rb'CDNN|XM4|Proposal|hudet|dfd_zm|dfd_me_zm', match.group()):
                entries.append({'file_offset': match.start(), 'text': match.group().decode('ascii')})
        runtime_index[filename] = {'sha256': digest(data), 'strings': entries,
                                   'manifest_digest_verified': True if verification else None,
                                   'interpretation': 'String locations only; execution and call relationships unverified.'}
    write_json(output / 'runtime-string-index.json', runtime_index)
    write_json(output / 'verification.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--archive', type=Path, help='Optional ZIP/TSV cross-check; auto-detected beside inputs if available')
    parser.add_argument('--no-archive', action='store_true', help='Use complete pinned components without a ZIP cross-check')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    args = parser.parse_args()
    if args.archive and args.no_archive:
        parser.error('--archive and --no-archive are mutually exclusive')
    source = resolve_source(args)
    archive = args.archive
    if archive is None and not args.no_archive:
        candidates = [source.parent / 'S5-V29-AF-models.zip', source / 'S5-V29-AF-models.zip']
        archive = next((p for p in candidates if p.is_file()), None)
    result = extract(source, archive, args.output)
    for name, info in result['models'].items():
        print(name, info['layers'], 'layers;', info['output_ports'], 'output ports; pinned model identity passed')
    print('Optional checks:', {key: result[key] for key in ('firmware_package_check', 'manifest_check', 'zip_check')})


if __name__ == '__main__':
    main()
