#!/usr/bin/env python3
"""Static graph shape constraints and explicit candidate operator metadata.

No DSP numerical equivalence is claimed. Semantic labels remain hypotheses.
"""
import argparse
import math
from pathlib import Path

from research_paths import add_firmware_arguments, resolve_source
import struct

from decode_parameters import tail_offset
from extract import MODELS, decode, digest, require, u32, write_json


def operator_metadata(data, node):
    q = tail_offset(node)
    op = node['operator_id']
    result = {'index': node['index'], 'name': node['name'], 'operator_id': op,
              'tail_offset': q, 'inputs': node['inputs']}
    if op == 6:
        width, height = struct.unpack_from('<II', data, q + 58)
        result.update(kind='input_candidate', shape=[1, 1, height, width])
    elif op == 1:
        # 1x1 entries have q+71..86 = [1,1,0,0], proving these cannot
        # all be padding values. Dilation then symmetric padding fits every node.
        sh, sw, dh, dw, ph, pw = struct.unpack_from('<6I', data, q + 63)
        transposed = bool(data[q + 100])
        relu = bool(data[q + 101])
        require(u32(data, q + 335) == int(relu), 'Activation flag copies disagree')
        result.update(kind='transposed_convolution_candidate' if transposed else 'convolution_candidate',
                      output_channels=u32(data, q + 57), kernel=list(data[q + 61:q + 63]),
                      stride=[sh, sw], dilation=[dh, dw], padding=[ph, pw],
                      relu_candidate=relu, transposed_candidate=transposed,
                      weight_stored_shape=list(struct.unpack_from('<4I', data, q + 119)))
    elif op == 5:
        kh, kw, sh, sw, ph, pw = struct.unpack_from('<6I', data, q + 61)
        result.update(kind='max_pool_candidate', kernel=[kh, kw], stride=[sh, sw],
                      padding=[ph, pw], ceil_mode_candidate=True,
                      pool_extra_flag=data[q - 9], raw_q85=data[q + 85], raw_q86=u32(data, q + 86))
    elif op == 9:
        result.update(kind='concat_candidate', axis=data[q + 57])
    elif op == 10:
        require(data[q + 69] == 1 and u32(data, q + 70) == 1, 'Unexpected slice metadata')
        result.update(kind='slice_candidate', axis=data[q + 69], cuts=[u32(data, q + 74)])
    elif op == 8:
        result.update(kind='reshape_candidate', shape=list(struct.unpack_from('<4I', data, q + 6)))
    elif op == 29:
        result.update(kind='softmax_candidate', axis=data[q + 62],
                      raw_q57=data[q + 57], raw_q58=u32(data, q + 58))
    elif op == 24:
        result.update(kind='proposal_candidate', feature_stride_candidate=list(struct.unpack_from('<2I', data, q + 61)),
                      anchor_base_size_candidate=u32(data, q + 69),
                      ratio_count_candidate=u32(data, q + 73), scale_count_candidate=u32(data, q + 77),
                      pre_nms_top_n_candidate=u32(data, q + 81), post_nms_top_n_candidate=u32(data, q + 85),
                      nms_threshold_candidate=struct.unpack_from('<f', data, q + 89)[0],
                      raw_q93=u32(data, q + 93), raw_q97=u32(data, q + 97))
    else:
        raise ValueError('Unknown operator %d' % op)
    return result


def analyze(data):
    graph = decode(data)
    layers = [operator_metadata(data, n) for n in graph['layers']]
    shapes = {}
    concat_checks = []
    reshape_checks = []
    tensor_channel_checks = []
    for layer in layers:
        i, kind = layer['index'], layer['kind']
        inputs = [shapes[(entry['layer'], entry['port'])] for entry in layer['inputs']
                  if entry['layer'] is not None]
        output = []
        if kind == 'input_candidate':
            output = [layer['shape']]
        elif kind in ('convolution_candidate', 'transposed_convolution_candidate'):
            n, c, h, w = inputs[0]
            kh, kw = layer['kernel']; sh, sw = layer['stride']
            dh, dw = layer['dilation']; ph, pw = layer['padding']
            if layer['transposed_candidate']:
                oh = (h - 1) * sh - 2 * ph + dh * (kh - 1) + 1
                ow = (w - 1) * sw - 2 * pw + dw * (kw - 1) + 1
            else:
                oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
                ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
                stored = layer['weight_stored_shape']
                inferred_input_channels = math.prod(stored[1:]) // (kh * kw)
                require(inferred_input_channels == c, 'Packed input channels conflict at ' + layer['name'])
                tensor_channel_checks.append({'layer': i, 'input_channels': c,
                                              'from_weight_record': inferred_input_channels})
            output = [[n, layer['output_channels'], oh, ow]]
        elif kind == 'max_pool_candidate':
            n, c, h, w = inputs[0]; kh, kw = layer['kernel']; sh, sw = layer['stride']; ph, pw = layer['padding']
            output = [[n, c, math.ceil((h + 2 * ph - kh) / sh) + 1,
                       math.ceil((w + 2 * pw - kw) / sw) + 1]]
        elif kind == 'slice_candidate':
            n, c, h, w = inputs[0]; cut = layer['cuts'][0]
            require(layer['axis'] == 1 and 0 < cut < c, 'Invalid channel split')
            output = [[n, cut, h, w], [n, c - cut, h, w]]
        elif kind == 'concat_candidate':
            require(layer['axis'] == 1, 'Unexpected concat axis')
            require(all([s[0], s[2], s[3]] == [inputs[0][0], inputs[0][2], inputs[0][3]] for s in inputs),
                    'Spatial branch mismatch at ' + layer['name'])
            output = [[inputs[0][0], sum(s[1] for s in inputs), inputs[0][2], inputs[0][3]]]
            concat_checks.append({'layer': i, 'inputs': inputs, 'output': output[0]})
        elif kind == 'reshape_candidate':
            require(math.prod(inputs[0]) == math.prod(layer['shape']), 'Encoded reshape element-count mismatch')
            output = [layer['shape']]
            reshape_checks.append({'layer': i, 'input': inputs[0], 'encoded_output': output[0]})
        elif kind == 'softmax_candidate':
            require(layer['axis'] == 1 and inputs[0][1] == 2, 'Unexpected binary-class softmax dimensions')
            output = [inputs[0]]
        elif kind == 'proposal_candidate':
            anchors = layer['ratio_count_candidate'] * layer['scale_count_candidate']
            require(inputs[0][1] == 2 * anchors and inputs[1][1] == 4 * anchors,
                    'Proposal anchor count conflicts with classifier/regressor channels')
            require(inputs[0][2:] == inputs[1][2:], 'Proposal input grids differ')
            layer['anchor_count_from_grid'] = anchors * inputs[0][2] * inputs[0][3]
            # Actual output shape/field ordering remains unresolved.
        layer['input_shapes_candidate'] = inputs
        layer['output_shapes_candidate'] = output
        for port, shape in enumerate(output):
            require(all(x > 0 for x in shape), 'Nonpositive inferred shape')
            shapes[(i, port)] = shape
    return {'source_sha256': digest(data), 'layers': layers,
            'checks': {'concat_branches_agree': len(concat_checks),
                       'explicit_reshape_counts_agree': len(reshape_checks),
                       'packed_input_channel_counts_agree': len(tensor_channel_checks)},
            'concat_checks': concat_checks, 'reshape_checks': reshape_checks,
            'tensor_channel_checks': tensor_channel_checks,
            'limitations': ['Shape-consistent candidate semantics only; no original DSP execution compared.',
                            'Pool max-vs-average not confirmed; ceil behavior required by concat dimensions.',
                            'Activation 0/1 flags correlate with heads/backbone; ReLU interpretation unverified.',
                            'Proposal names for raw parameters are conventional interpretations, not confirmed ABI.']}


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
        report = analyze(data)
        write_json(args.output / stem / 'operator-investigation.json', report)
        print(stem, report['checks'])


if __name__ == '__main__':
    main()
