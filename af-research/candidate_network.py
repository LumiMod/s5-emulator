#!/usr/bin/env python3
"""Executable S5 graph hypothesis through RPN tensors, NOT verified DSP inference.

This deliberately stops before proposal decoding. The input is a prepared tensor,
not a photograph: camera normalization and DSP arithmetic are still unresolved.
"""
import argparse
from pathlib import Path

from research_paths import add_firmware_arguments, resolve_source
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from decode_parameters import recover
from extract import MODELS, digest, require, write_json
from ops_investigation import analyze
from packing_investigation import SPATIAL_LAYOUTS, unpack


def candidate_weight(raw, record, metadata, spatial_layout):
    """Explicit, replaceable physical layout hypotheses; original integers retained."""
    channels = metadata['input_shapes_candidate'][0][1]
    outputs = metadata['output_channels']
    kh, kw = metadata['kernel']
    if metadata['transposed_candidate']:
        # Histogram-constrained kernel; channel mapping remains a hypothesis.
        values, counts = np.unique(raw, return_counts=True)
        require(len(values) == 4 and np.array_equal(counts, [12288, 1024, 2048, 1024]),
                'Unexpected upsample coefficient histogram')
        a, b, c = values[1:]
        kernel = np.array([[a,b,b,a], [b,c,c,b], [b,c,c,b], [a,b,b,a]], dtype=np.float32)
        return np.tile(kernel[None,None], (channels,1,1,1)) * record['weight_scale'], 'bilinear_histogram_candidate'
    weights = unpack(raw, record, spatial_layout)
    require(weights.shape == (outputs,channels,kh,kw), 'Graph and unpacked dimensions differ')
    hypothesis = ('direct_first_layer_candidate' if metadata['name']=='conv1'
                  else 'O4_I4_candidate' if kh==kw==1 else spatial_layout)
    return weights.astype(np.float32) * record['weight_scale'], hypothesis


class CandidateNetwork(nn.Module):
    def __init__(self, data, spatial_layout='hw_o4_i4'):
        super().__init__()
        require(digest(data) in MODELS.values(), 'Unknown source identity')
        self.source_sha256 = digest(data)
        self.analysis = analyze(data)
        params, tensors = recover(data)
        records = {item['index']: item for item in params['convolutions']}
        self.layers = self.analysis['layers']
        self.convs = nn.ModuleDict()
        self.layout_hypotheses = []
        for node in self.layers:
            if node['operator_id'] != 1:
                continue
            record = records[node['index']]
            raw = tensors[record['tensor_key'] + '_weight_i16_packed']
            weights, hypothesis = candidate_weight(raw, record, node, spatial_layout)
            channels = node['input_shapes_candidate'][0][1]
            common = dict(kernel_size=node['kernel'], stride=node['stride'],
                          padding=node['padding'], dilation=node['dilation'])
            if node['transposed_candidate']:
                module = nn.ConvTranspose2d(channels,node['output_channels'],groups=channels,**common)
            else:
                module = nn.Conv2d(channels,node['output_channels'],**common)
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(weights.copy()))
                module.bias.copy_(torch.from_numpy(tensors[record['tensor_key'] + '_bias_f32']))
            self.convs[str(node['index'])] = module
            self.layout_hypotheses.append({'layer': node['index'], 'name': node['name'], 'layout': hypothesis})

    def forward(self, prepared_tensor, capture_trace=False):
        require(list(prepared_tensor.shape) == self.layers[0]['shape'], 'Expected prepared [1,1,256,384] tensor')
        buffers, heads, trace = {}, {}, []
        for node in self.layers:
            index, op = node['index'], node['operator_id']
            inputs = [buffers[(entry['layer'],entry['port'])] for entry in node['inputs'] if entry['layer'] is not None]
            if op == 24:
                # Preserve both inputs to each proposal; do not invent boxes.
                heads[node['name'] + '_probabilities'] = inputs[0]
                heads['bbox_deltas'] = inputs[1]
                continue
            if op == 6:
                outputs = [prepared_tensor]
            elif op == 1:
                value = self.convs[str(index)](inputs[0])
                outputs = [F.relu(value) if node['relu_candidate'] else value]
            elif op == 5:
                outputs = [F.max_pool2d(inputs[0],node['kernel'],node['stride'],node['padding'],ceil_mode=True)]
            elif op == 9:
                outputs = [torch.cat(inputs,dim=node['axis'])]
            elif op == 10:
                cut = node['cuts'][0]
                outputs = list(torch.split(inputs[0],[cut,inputs[0].shape[1]-cut],dim=1))
            elif op == 8:
                outputs = [inputs[0].reshape(node['shape'])]
            elif op == 29:
                outputs = [F.softmax(inputs[0],dim=node['axis'])]
            else:
                raise ValueError('Unsupported operator')
            require([list(x.shape) for x in outputs] == node['output_shapes_candidate'],
                    'Runtime shape differs from metadata: ' + node['name'])
            for port, value in enumerate(outputs):
                require(bool(torch.isfinite(value).all()), 'Nonfinite tensor: ' + node['name'])
                buffers[(index,port)] = value
                if capture_trace:
                    detached = value.detach()
                    trace.append({'layer':index,'port':port,'name':node['name'],'shape':list(value.shape),
                                  'minimum':float(detached.min()),'maximum':float(detached.max()),
                                  'mean':float(detached.mean())})
        return heads, trace


def run_probe(data, folder, spatial_layout='hw_o4_i4'):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = CandidateNetwork(data, spatial_layout)
    # Explicit synthetic preprocessed tensor, no claims about camera normalization.
    grid = torch.arange(256*384,dtype=torch.float32).reshape(1,1,256,384)
    prepared = torch.sin(grid * 0.017) * 0.1
    start = time.monotonic()
    heads, trace = model(prepared, capture_trace=True)
    probability_checks = {}
    for name, value in heads.items():
        if name.endswith('_probabilities'):
            binary = value.reshape(1,2,25*16,24)
            error = float((binary.sum(dim=1)-1).abs().max().detach())
            require(error < 2e-6, 'Invalid candidate softmax normalization')
            probability_checks[name] = error
    # Exercise the entire backbone and all heads without optimizing any weights.
    loss = heads['bbox_deltas'].square().mean()
    for name, value in heads.items():
        if name.endswith('_probabilities'):
            loss = loss + value[:,:25].square().mean()
    loss.backward()
    gradient_stats = []
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), 'Missing/invalid gradient: '+name)
        gradient_stats.append({'parameter':name, 'norm':float(parameter.grad.norm())})
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'candidate_rpn_outputs.npz',**{name:value.detach().numpy() for name,value in heads.items()})
    report = {'source_sha256':model.source_sha256, 'status':'unverified full-graph hypothesis through RPN',
              'spatial_layout_hypothesis':spatial_layout,
              'executed_records':len([n for n in model.layers if n['operator_id']!=24]),
              'output_shapes':{name:list(value.shape) for name,value in heads.items()},
              'layout_hypotheses':model.layout_hypotheses, 'trace':trace,
              'probability_sum_max_errors':probability_checks, 'gradient_stats':gradient_stats,
              'elapsed_seconds':time.monotonic()-start,
              'limitations':['Full forward/backward checks only, with synthetic prepared input.',
                             'Weight packing, ReLU/max-pool choices and channel mappings remain hypotheses.',
                             'No original DSP numerical comparison; no detection/AF accuracy measurement.',
                             'Proposal decoding, camera preprocessing and fixed-point arithmetic omitted.',
                             'No optimizer step or retrained checkpoint produced.']}
    write_json(folder/'candidate-network-probe.json',report)
    return report


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--artifacts',type=Path,default=base/'artifacts')
    parser.add_argument('--spatial-layout',choices=SPATIAL_LAYOUTS,default='hw_o4_i4')
    parser.add_argument('--compare-layouts',action='store_true',help='Run all three unresolved spatial orders and compare outputs')
    args = parser.parse_args()
    args.source = resolve_source(args)
    for stem in MODELS:
        data = (args.source/(stem+'.bin')).read_bytes()
        layouts = SPATIAL_LAYOUTS if args.compare_layouts else [args.spatial_layout]
        outputs = {}
        for layout in layouts:
            folder = args.artifacts/stem/'graph-hypotheses'/layout
            report = run_probe(data,folder,layout)
            with np.load(folder/'candidate_rpn_outputs.npz',allow_pickle=False) as saved:
                outputs[layout] = {key:saved[key].copy() for key in saved.files}
            print(stem,layout,report['executed_records'],'candidate records, forward/backward passed;',round(report['elapsed_seconds'],3),'seconds')
        if args.compare_layouts:
            comparisons = []
            for index, left in enumerate(layouts):
                for right in layouts[index+1:]:
                    comparisons.append({'left':left,'right':right,'output_max_absolute_differences':{
                        key:float(np.max(np.abs(value-outputs[right][key]))) for key,value in outputs[left].items()}})
            write_json(args.artifacts/stem/'graph-hypotheses/layout-comparison.json',
                       {'source_sha256':digest(data),'comparisons':comparisons,
                        'conclusion':'Shape and gradient checks do not resolve packing; these hypotheses produce different numeric predictions.'})


if __name__ == '__main__':
    main()
