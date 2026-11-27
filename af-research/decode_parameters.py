#!/usr/bin/env python3
"""Recover S5 CDNN numeric tensors, preserving unverified DSP packing."""
import argparse
import math
from pathlib import Path

from research_paths import add_firmware_arguments, resolve_source
import struct

import numpy as np

from extract import MODELS, decode, digest, require, u32, write_json


def f32(data, offset):
    return struct.unpack_from('<f', data, offset)[0]


def tail_offset(node):
    return (node['name_offset'] + len(node['name']) + 1 + 7
            + 8 * len(node['inputs']) + (node['operator_id'] == 5))


def output_scale(data, node):
    # Offsets are specific to these two pinned CDNN artifacts.
    relative = {10: 19, 8: 24, 24: 11}.get(node['operator_id'], 7)
    offset = tail_offset(node) + relative
    require(data[offset - 2:offset] == b'\x01\x10', 'Unexpected output-scale header')
    scale = f32(data, offset)
    require(math.isfinite(scale) and scale > 0, 'Invalid output scale')
    return scale


def descriptor(data, offset):
    size, *fields = struct.unpack_from('<7I', data, offset)
    shape, tags = fields[:4], fields[4:]
    scale = f32(data, offset + 28)
    require(math.isfinite(scale), 'Nonfinite descriptor scale')
    return {'descriptor_offset': offset, 'bytes': size, 'stored_shape': shape,
            'raw_numeric_tags': tags, 'scale': scale,
            'raw_scale_area_hex': data[offset + 28:offset + 68].hex()}


def recover(data):
    graph = decode(data)
    start, end = graph['regions']['parameter_candidate']
    inp = graph['layers'][0]
    q = tail_offset(inp)
    require(inp['operator_id'] == 6, 'Unexpected input operator')
    valid_width, valid_height = struct.unpack_from('<II', data, q + 58)
    input_shape = list(struct.unpack_from('<4I', data, q + 83))
    require(input_shape == [1, 1, valid_height, valid_width + 2], 'Unexpected embedded input shape')
    input_bytes = math.prod(input_shape) * 2
    input_blob = data[start:start + input_bytes]
    input_array = np.frombuffer(input_blob, dtype='<i2').reshape(input_shape).copy()
    input_units_scale = f32(data, q + 107)
    input_aux_bytes = u32(data, q + 148)
    require(input_aux_bytes == 64, 'Unexpected input auxiliary block')
    cursor = start + input_bytes + input_aux_bytes
    tensors = {'embedded_input_i16': input_array}
    manifest = {'source_sha256': digest(data),
                'input_candidate': {'offset': start, 'bytes': input_bytes, 'stored_shape': input_shape,
                                    'valid_width': valid_width, 'valid_height': valid_height,
                                    'scale_candidate': input_units_scale,
                                    'valid_area_all_same': bool(np.all(input_array[..., :valid_width] == input_array.flat[0])),
                                    'valid_area_raw_value': int(input_array.flat[0]),
                                    'valid_area_scaled_value': float(input_array.flat[0]) * input_units_scale,
                                    'row_padding_interpretation': 'Two stored i16 elements per row; meaning unverified.'},
                'input_auxiliary': {'offset': start + input_bytes, 'bytes': input_aux_bytes,
                                    'sha256': digest(data[start + input_bytes:cursor])},
                'convolutions': [], 'proposal_blocks': [],
                'limitations': [
                    'Stored dimensions and scales recovered; DSP kernel/input/output packing unverified.',
                    'Dequantization uses symmetric scale multiplication supported by bias-scale identities.',
                    'Zero points, rounding, saturation, activations and complete preprocessing unverified.',
                    'No numerical match against original DSP execution; not a deployable or fully trainable network.'
                ]}
    scales = {node['index']: output_scale(data, node) for node in graph['layers']}
    max_error = 0
    for node in graph['layers']:
        if node['operator_id'] != 1:
            continue
        q = tail_offset(node)
        weight = descriptor(data, q + 115)
        bias = descriptor(data, q + 183)
        # Each descriptor spans 68 bytes. Only its first float is interpreted;
        # unused/raw slots may contain nonfinite float bit patterns.
        require(weight['raw_numeric_tags'] == [24, 8], 'Unexpected weight tags')
        require(bias['raw_numeric_tags'] == [25, 32], 'Unexpected bias tags')
        require(weight['bytes'] == math.prod(weight['stored_shape']) * 2, 'Weight shape/byte mismatch')
        logical_bias_bytes = math.prod(bias['stored_shape']) * 4
        require(bias['bytes'] == (logical_bias_bytes + 31) // 32 * 32, 'Bias alignment mismatch')
        channels = u32(data, q + 57)
        require(bias['stored_shape'] == [channels, 1, 1, 2], 'Unexpected logical bias shape')
        require(cursor + weight['bytes'] + bias['bytes'] <= end, 'Parameter overrun')
        wblob = data[cursor:cursor + weight['bytes']]
        bstart = cursor + weight['bytes']
        bblob = data[bstart:bstart + bias['bytes']]
        wraw = np.frombuffer(wblob, dtype='<i2').reshape(weight['stored_shape']).copy()
        braw = np.frombuffer(bblob, dtype='<i4').reshape(-1, 2).copy()
        require(np.array_equal(braw[:, 0], braw[:, 1]), 'Bias pairs differ; stream alignment unproven')
        predecessor = node['inputs'][0]['layer']
        inscale = scales[predecessor]
        wscale, bscale = weight['scale'], bias['scale']
        require(wscale > 0 and bscale > 0, 'Nonpositive tensor scale')
        relative_error = abs(wscale * inscale - bscale) / bscale
        require(relative_error < 1e-6, 'Bias scale != input scale * weight scale')
        max_error = max(max_error, relative_error)
        key = 'layer_%03d' % node['index']
        tensors[key + '_weight_i16_packed'] = wraw
        tensors[key + '_weight_f32_packed'] = wraw.astype(np.float32) * np.float32(wscale)
        tensors[key + '_bias_i32_pairs'] = braw
        tensors[key + '_bias_f32'] = braw[:channels, 0].astype(np.float32) * np.float32(bscale)
        kh, kw = data[q + 61:q + 63]
        stride = list(struct.unpack_from('<2I', data, q + 63))
        dilation = list(struct.unpack_from('<2I', data, q + 71))
        padding = list(struct.unpack_from('<2I', data, q + 79))
        record = {'index': node['index'], 'name': node['name'], 'tensor_key': key,
                  'input_layer': predecessor, 'output_channels_candidate': channels,
                  'kernel_hw_candidate': [kh, kw], 'stride_hw_candidate': stride,
                  'dilation_hw_candidate': dilation, 'padding_hw_candidate': padding,
                  'activation_flag_candidate': data[q + 101],
                  'activation_tail_value': u32(data, node['record_end'] - 25),
                  'input_scale': inscale, 'output_scale': scales[node['index']],
                  'weight_scale': wscale, 'bias_scale': bscale,
                  'scale_identity_relative_error': relative_error, 'bias_pairs_identical': True,
                  'weight': dict(weight, data_offset=cursor, sha256=digest(wblob)),
                  'bias': dict(bias, data_offset=bstart, sha256=digest(bblob))}
        if node['name'] == 'upsample':
            unique, counts = np.unique(wraw, return_counts=True)
            scaled = unique.astype(np.float64) * wscale
            expected_coefficients = np.array([0, 1, 3, 9], dtype=np.float64) / 16
            record['upsample_hypothesis'] = {
                'raw_values': unique.tolist(), 'raw_counts': counts.tolist(),
                'dequantized_values': scaled.tolist(),
                'matches_256_bilinear_kernel_histograms_plus_padding': bool(
                    np.array_equal(counts, [12288, 1024, 2048, 1024]) and
                    scaled.shape == (4,) and np.max(np.abs(scaled - expected_coefficients)) < 1e-4),
                'interpretation': 'Coefficients match 4x4 separable bilinear kernels within quantization error; '
                                  'physical channel mapping and original execution remain unverified.'}
        manifest['convolutions'].append(record)
        cursor = bstart + bias['bytes']
    proposals = [n for n in graph['layers'] if n['operator_id'] == 24]
    require(end - cursor == len(proposals) * 576, 'Unexpected remainder after convolution tensors')
    first_block = data[cursor:cursor + 576]
    for node in proposals:
        block = data[cursor:cursor + 576]
        require(block == first_block, 'Proposal parameter blocks differ')
        manifest['proposal_blocks'].append({'layer': node['index'], 'offset': cursor,
                                            'bytes': 576, 'sha256': digest(block),
                                            'ratio_candidates': list(struct.unpack_from('<5f', block)),
                                            'scale_candidates': list(struct.unpack_from('<5f', block, 32))})
        cursor += 576
    require(cursor == end, 'Incomplete parameter coverage')
    manifest['checks'] = {'all_parameter_bytes_accounted_for': True,
                          'all_bias_pairs_identical': True,
                          'all_bias_scale_identities_pass': True,
                          'maximum_scale_identity_relative_error': max_error,
                          'convolution_layers': len(manifest['convolutions']),
                          'parameter_bytes': end - start}
    return manifest, tensors


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output', type=Path, default=base / 'artifacts')
    args = parser.parse_args()
    args.source = resolve_source(args)
    for stem, expected in MODELS.items():
        data = (args.source / (stem + '.bin')).read_bytes()
        require(digest(data) == expected, 'Source digest mismatch')
        manifest, tensors = recover(data)
        folder = args.output / stem
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'numeric_tensors.npz', **tensors)
        write_json(folder / 'parameters.json', manifest)
        print(stem, manifest['checks'])


if __name__ == '__main__':
    main()
