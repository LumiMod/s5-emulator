#!/usr/bin/env python3
"""Invertible candidate DSP weight layouts; not proof of DSP semantics.

1x1 output-lane blocking is supported by zero-padded classification heads.
3x3 spatial/channel order and conv1 direct order remain hypotheses.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from extract import MODELS, require, write_json

SPATIAL_LAYOUTS = ('o4_i4_hw', 'hw_o4_i4', 'o4_hw_i4')


def dimensions(record):
    require(record['name'] != 'upsample', 'upsample channel layout is not recovered')
    kh, kw = record['kernel_hw_candidate']
    physical_output = record['weight']['stored_shape'][0]
    logical_output = record['output_channels_candidate']
    count = int(np.prod(record['weight']['stored_shape']))
    require(count % (physical_output * kh * kw) == 0, 'Nonintegral input channels')
    inputs = count // (physical_output * kh * kw)
    return logical_output, physical_output, inputs, kh, kw


def unpack(raw, record, spatial_layout='o4_i4_hw'):
    """Return logical OIHW candidate; reject nonzero output padding.

    The default 3x3 layout is a convenient explicit hypothesis, not a result
    established by byte-level roundtrips. Keep spatial_layout in provenance.
    """
    require(spatial_layout in SPATIAL_LAYOUTS, 'Unknown spatial-layout hypothesis')
    logical, outputs, inputs, kh, kw = dimensions(record)
    raw = np.asarray(raw)
    require(raw.size == outputs * inputs * kh * kw, 'Wrong element count')
    if record['name'] == 'conv1':
        result = raw.reshape(outputs, inputs, kh, kw)
    else:
        require(outputs % 4 == 0 and inputs % 4 == 0, 'Blocking requires multiples of four')
        if (kh, kw) == (1, 1) or spatial_layout == 'o4_i4_hw':
            result = raw.reshape(outputs // 4, inputs // 4, 4, 4, kh, kw).transpose(0, 2, 1, 3, 4, 5)
        elif spatial_layout == 'hw_o4_i4':
            result = raw.reshape(outputs // 4, inputs // 4, kh, kw, 4, 4).transpose(0, 4, 1, 5, 2, 3)
        else:
            result = raw.reshape(outputs // 4, inputs // 4, 4, kh, kw, 4).transpose(0, 2, 1, 5, 3, 4)
        result = result.reshape(outputs, inputs, kh, kw)
    require(np.all(result[logical:] == 0), 'Output padding is not zero under this hypothesis')
    return result[:logical].copy()


def pack(tensor, record, spatial_layout='o4_i4_hw'):
    """Inverse of unpack, including zero output-channel alignment."""
    require(spatial_layout in SPATIAL_LAYOUTS, 'Unknown spatial-layout hypothesis')
    logical, outputs, inputs, kh, kw = dimensions(record)
    tensor = np.asarray(tensor)
    require(tensor.shape == (logical, inputs, kh, kw), 'Wrong logical OIHW shape')
    padded = np.zeros((outputs, inputs, kh, kw), dtype=tensor.dtype)
    padded[:logical] = tensor
    if record['name'] == 'conv1':
        result = padded
    else:
        base = padded.reshape(outputs // 4, 4, inputs // 4, 4, kh, kw)
        if (kh, kw) == (1, 1) or spatial_layout == 'o4_i4_hw':
            result = base.transpose(0, 2, 1, 3, 4, 5)
        elif spatial_layout == 'hw_o4_i4':
            result = base.transpose(0, 2, 4, 5, 1, 3)
        else:
            result = base.transpose(0, 2, 1, 4, 5, 3)
    return result.reshape(record['weight']['stored_shape']).copy()


def inspect_model(folder):
    manifest = json.loads((folder / 'parameters.json').read_text())
    original = (folder / 'original.bin').read_bytes()
    report = {'source_sha256': manifest['source_sha256'], 'classification_padding': [],
              'roundtrip_layers': {}, 'upsample_pattern': {},
              'spatial_hypothesis_statistics': {},
              'limitations': ['Roundtrips prove permutation invertibility, not original DSP layout.',
                             '1x1 within-input-block order and block ordering need runtime corroboration.',
                             '3x3 spatial ordering remains underdetermined; three hypotheses retained.',
                             'Conv1 direct OIHW is retained as a hypothesis, not validated by these tests.']}
    with np.load(folder / 'numeric_tensors.npz', allow_pickle=False) as tensors:
        aux_offset = manifest['input_auxiliary']['offset']
        bins = list(struct.unpack_from('<16I', original, aux_offset))
        expected_pixels = manifest['input_candidate']['valid_width'] * manifest['input_candidate']['valid_height']
        require(sum(bins) == expected_pixels, 'Input histogram candidate count mismatch')
        report['input_auxiliary_histogram_candidate'] = {
            'file_offset': aux_offset, 'uint32_counts': bins,
            'sum_counts': sum(bins), 'image_pixels': expected_pixels,
            'interpretation': 'Counts sum exactly to valid image pixel count. Histogram candidate; '
                              'bin boundaries and calibration/runtime purpose are unverified.'}
        for layout in SPATIAL_LAYOUTS:
            tested = 0
            spatial_statistics = []
            for record in manifest['convolutions']:
                if record['name'] == 'upsample':
                    continue
                raw = tensors[record['tensor_key'] + '_weight_i16_packed']
                candidate = unpack(raw, record, layout)
                rebuilt = pack(candidate, record, layout)
                require(np.array_equal(raw, rebuilt), 'Candidate permutation roundtrip failed')
                tested += 1
                if record['kernel_hw_candidate'] == [3, 3]:
                    floating = candidate.astype(np.float64)
                    energy = float(np.mean(floating * floating))
                    variation = float((np.mean(np.diff(floating, axis=-1) ** 2)
                                       + np.mean(np.diff(floating, axis=-2) ** 2)) / (2 * energy))
                    spatial_statistics.append({'name': record['name'],
                                               'normalized_neighbor_difference_energy': variation})
            report['roundtrip_layers'][layout] = tested
            report['spatial_hypothesis_statistics'][layout] = {
                'mean_normalized_neighbor_difference_energy': float(np.mean([
                    row['normalized_neighbor_difference_energy'] for row in spatial_statistics])),
                'layers': spatial_statistics,
                'interpretation': 'Smoothness heuristic only. Learned kernels need not be smooth; '
                                  'this cannot select or validate original DSP layout.'}
        for record in manifest['convolutions']:
            raw = tensors[record['tensor_key'] + '_weight_i16_packed']
            if record['name'].startswith('rpn_cls_score'):
                logical, outputs, inputs, kh, kw = dimensions(record)
                require((logical, outputs, kh, kw) == (50, 52, 1, 1), 'Unexpected classification head')
                tiled = raw.reshape(outputs // 4, inputs // 4, 4, 4)
                zero_tile_count = int(np.count_nonzero(tiled[-1, :, 2:, :]))
                direct = raw.reshape(outputs, inputs)
                direct_nonzero = int(np.count_nonzero(direct[logical:]))
                require(zero_tile_count == 0 and direct_nonzero > 0, 'Padding evidence changed')
                logical_weights = unpack(raw, record).reshape(logical, inputs).astype(np.float64)
                first, second = logical_weights[:25], logical_weights[25:]
                paired_cosines = np.sum(first * second, axis=1) / np.maximum(
                    np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1), 1e-15)
                require(np.all(paired_cosines < -0.98), 'Foreground/background pairing invariant changed')
                report['classification_padding'].append({
                    'name': record['name'], 'physical_output_channels': outputs,
                    'logical_output_channels': logical, 'input_channels': inputs,
                    'last_two_tiled_output_lanes_nonzero_count': zero_tile_count,
                    'direct_OI_last_two_rows_nonzero_count': direct_nonzero,
                    'zero_elements_explained': (outputs - logical) * inputs,
                    'first25_last25_pair_cosines': paired_cosines.tolist(),
                    'mean_foreground_background_pair_cosine': float(np.mean(paired_cosines)),
                    'pairing_interpretation': 'All 25 row pairs are nearly negatives, independently supporting '
                                              'output ordering as two classes times 25 anchors. '
                                              'This does not establish within-input-channel ordering.'})
            elif record['name'] == 'upsample':
                blocks = raw.reshape(-1, 64)
                nonzeros = np.count_nonzero(blocks, axis=1)
                report['upsample_pattern'] = {
                    'block_elements': 64, 'unique_blocks': int(len(np.unique(blocks, axis=0))),
                    'first_128_blocks_identical': bool(np.all(blocks[:128] == blocks[0])),
                    'first_128_nonzeros_per_block': np.unique(nonzeros[:128]).tolist(),
                    'last_128_blocks_all_zero': bool(np.all(blocks[128:] == 0)),
                    'first_block_values': blocks[0].tolist(),
                    'interpretation': 'Repeated coefficient table plus zero area; histogram does not identify channel mapping.'}
    return report


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-manifest', type=Path, help='Optional prior trimmed-artifact JSON comparison')
    args = parser.parse_args()
    report = {stem: inspect_model(base / 'artifacts' / stem) for stem in MODELS}
    prior = args.prior_manifest
    if prior is not None:
        for previous in json.loads(prior.read_text()).values():
            stem = Path(previous['source_partition']).stem
            if stem not in report:
                continue
            trimmed = prior.parent / previous['extracted_file']
            original = (base / 'artifacts' / stem / 'original.bin').read_bytes()
            old_bytes = trimmed.read_bytes()
            parameter_manifest = json.loads((base / 'artifacts' / stem / 'parameters.json').read_text())
            last_block = parameter_manifest['proposal_blocks'][-1]
            true_end = last_block['offset'] + last_block['bytes']
            report[stem]['prior_artifact_comparison'] = {
                'old_file': str(trimmed), 'old_size': len(old_bytes),
                'matches_canonical_prefix': old_bytes == original[:len(old_bytes)],
                'omitted_parameter_tail_bytes': true_end - len(old_bytes),
                'omitted_bytes_hex': original[len(old_bytes):true_end].hex(),
                'interpretation': 'Prior trim omits 16 parameter-area zero bytes; retain pinned canonical originals.'}
    write_json(base / 'artifacts/packing-investigation.json', report)
    for stem, result in report.items():
        print(stem, 'checked', result['roundtrip_layers'],
              'classification padding heads', len(result['classification_padding']))


if __name__ == '__main__':
    main()
