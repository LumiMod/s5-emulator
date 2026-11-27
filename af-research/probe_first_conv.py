#!/usr/bin/env python3
"""Probe a candidate first-convolution implementation; no AF accuracy claim."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from extract import MODELS, require, write_json


def load_candidate(folder):
    params = json.loads((folder / 'parameters.json').read_text())
    record = params['convolutions'][0]
    require(record['name'] == 'conv1', 'Unexpected first convolution')
    require(record['kernel_hw_candidate'] == [4, 4], 'Unexpected kernel')
    require(record['weight']['stored_shape'] == [32, 1, 1, 16], 'Unexpected stored shape')
    require(record['stride_hw_candidate'] == [2, 2], 'Unexpected stride')
    require(record['padding_hw_candidate'] == [1, 1], 'Unexpected padding')
    require(record['dilation_hw_candidate'] == [1, 1], 'Unexpected dilation')
    key = record['tensor_key']
    with np.load(folder / 'numeric_tensors.npz', allow_pickle=False) as tensors:
        weight = tensors[key + '_weight_f32_packed'].reshape(32, 1, 4, 4).copy()
        bias = tensors[key + '_bias_f32'].copy()
        raw_weight = tensors[key + '_weight_i16_packed'].reshape(32, 4, 4).copy()
        raw_bias = tensors[key + '_bias_i32_pairs'][:, 0].copy()
    model = nn.Conv2d(1, 32, 4, stride=2, padding=1)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(weight))
        model.bias.copy_(torch.from_numpy(bias))
    return model, record, raw_weight, raw_bias


def integer_reference(qinput, qweight, qbias, record):
    # Independent integer accumulation for the same candidate convolution.
    padded = np.pad(qinput.astype(np.int64), ((1, 1), (1, 1)))
    height, width = qinput.shape
    result = np.zeros((32, height // 2, width // 2), dtype=np.float64)
    combined_scale = record['input_scale'] * record['weight_scale']
    for channel in range(32):
        for y in range(height // 2):
            for x in range(width // 2):
                patch = padded[y * 2:y * 2 + 4, x * 2:x * 2 + 4]
                acc = int(np.sum(patch * qweight[channel].astype(np.int64)))
                result[channel, y, x] = acc * combined_scale + int(qbias[channel]) * record['bias_scale']
    return result


def run(folder):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model, record, qweight, qbias = load_candidate(folder)
    # Nonuniform synthetic input exercises sign, borders, orientation and bias.
    qinput = ((np.arange(16 * 20).reshape(16, 20) * 17) % 75 - 37).astype(np.int16)
    x = torch.from_numpy(qinput.astype(np.float32) * np.float32(record['input_scale']))[None, None]
    reference = integer_reference(qinput, qweight, qbias, record)
    with torch.no_grad():
        actual = model(x).numpy()[0].astype(np.float64)
    max_error = float(np.max(np.abs(reference - actual)))
    require(np.allclose(reference, actual, rtol=2e-5, atol=2e-7), 'Integer/float candidate references differ')
    # Verify a finite, nonzero gradient and a loss-decreasing optimizer step.
    # This is an artificial loss, not AF training or a performance measurement.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before = model(x).square().mean()
    before.backward()
    grad_norm = float(torch.linalg.vector_norm(model.weight.grad).item())
    require(math.isfinite(grad_norm) and grad_norm > 0, 'Invalid weight gradient')
    optimizer.step()
    with torch.no_grad():
        after = float(model(x).square().mean().item())
    require(after < float(before.detach().item()), 'Synthetic optimizer loss did not decrease')
    # Save original recovered values, never the synthetic optimizer result.
    model.load_state_dict(initial_state)
    torch.save(model.state_dict(), folder / 'first_conv_candidate.pt')
    report = {'status': 'candidate implementation smoke test passed',
              'weight_blob_sha256': record['weight']['sha256'],
              'input_shape': list(x.shape), 'output_shape': [1] + list(actual.shape),
              'reference_max_absolute_error': max_error,
              'weight_gradient_norm': grad_norm,
              'synthetic_loss_before': float(before.detach().item()),
              'synthetic_loss_after_one_step': after,
              'saved_checkpoint_contains_original_recovered_values': True,
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'limitations': [
                  'Assumes stored conv1 weights map directly to OIHW [32,1,4,4].',
                  'Reference comparison validates our arithmetic implementation, not the original DSP.',
                  'No original preprocessing, activation, rounding or saturation emulated.',
                  'Synthetic input and loss only; no retrained autofocus or accuracy result.'
              ]}
    write_json(folder / 'first_conv_probe.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    args = parser.parse_args()
    for stem in MODELS:
        report = run(args.artifacts / stem)
        print(stem, 'max error', report['reference_max_absolute_error'],
              'gradient', report['weight_gradient_norm'], 'synthetic loss step passed')


if __name__ == '__main__':
    main()
