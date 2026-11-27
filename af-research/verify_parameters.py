#!/usr/bin/env python3
"""Cross-check numeric exports against bytes and reject damaged structures."""
from research_paths import add_firmware_arguments, resolve_source
from pathlib import Path
import argparse
import struct

import numpy as np
import torch

from decode_parameters import recover
from extract import MODELS, digest, require, write_json
from probe_first_conv import load_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-first-conv-checkpoint', action='store_true',
                        help='Validate numeric extraction without requiring the older toy first-layer checkpoint')
    add_firmware_arguments(parser)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    source = resolve_source(args)
    report = {}
    for stem, expected in MODELS.items():
        data = (source / (stem + '.bin')).read_bytes()
        require(digest(data) == expected, 'Source digest mismatch')
        manifest, tensors = recover(data)
        folder = base / 'artifacts' / stem
        graph_start = manifest['input_candidate']['offset']
        rebuilt = bytearray(tensors['embedded_input_i16'].tobytes())
        aux = manifest['input_auxiliary']
        rebuilt.extend(data[aux['offset']:aux['offset'] + aux['bytes']])
        with np.load(folder / 'numeric_tensors.npz', allow_pickle=False) as saved:
            require(set(saved.files) == set(tensors), 'Export key mismatch')
            for key, tensor in tensors.items():
                require(np.array_equal(saved[key], tensor), 'Export value mismatch: ' + key)
        for record in manifest['convolutions']:
            key = record['tensor_key']
            for suffix in ['_weight_i16_packed', '_bias_i32_pairs']:
                rebuilt.extend(tensors[key + suffix].tobytes())
        for block in manifest['proposal_blocks']:
            rebuilt.extend(data[block['offset']:block['offset'] + block['bytes']])
        require(bytes(rebuilt) == data[graph_start:graph_start + len(rebuilt)], 'Tensor reassembly differs')
        require(len(rebuilt) == manifest['checks']['parameter_bytes'], 'Tensor reassembly incomplete')
        first = manifest['convolutions'][0]
        damaged = []
        for name in ['bias_pair', 'weight_shape', 'weight_scale', 'proposal_block']:
            altered = bytearray(data)
            if name == 'bias_pair':
                altered[first['bias']['data_offset']] ^= 1
            elif name == 'weight_shape':
                struct.pack_into('<I', altered, first['weight']['descriptor_offset'] + 4, 0)
            elif name == 'weight_scale':
                struct.pack_into('<f', altered, first['weight']['descriptor_offset'] + 28,
                                 first['weight_scale'] * 2)
            else:
                altered[manifest['proposal_blocks'][0]['offset']] ^= 1
            try:
                recover(bytes(altered))
            except ValueError as exc:
                damaged.append({'case': name, 'rejection': str(exc)})
            else:
                raise AssertionError('Accepted corrupted ' + name)
        checkpoint_checked = False
        if not args.skip_first_conv_checkpoint:
            candidate, _, _, _ = load_candidate(folder)
            checkpoint = torch.load(folder / 'first_conv_candidate.pt', map_location='cpu', weights_only=True)
            require(all(torch.equal(value, checkpoint[key]) for key, value in candidate.state_dict().items()),
                    'Checkpoint retained synthetic optimizer updates')
            checkpoint_checked = True
        report[stem] = {'raw_parameter_reassembly_identical': True,
                        'npz_matches_fresh_decode': True,
                        'checkpoint_matches_original_recovered_tensors': True if checkpoint_checked else None,
                        'corruption_rejections': damaged}
        print(stem, 'byte reassembly, NPZ and four corruption cases passed; checkpoint checked:',checkpoint_checked)
    write_json(base / 'artifacts/parameter-validation.json', report)


if __name__ == '__main__':
    main()
