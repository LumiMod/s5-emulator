#!/usr/bin/env python3
"""Validate graph constraints and candidate packing, without claiming DSP parity."""
import argparse
from research_paths import add_firmware_arguments, resolve_source
from pathlib import Path
import struct

import numpy as np

from decode_parameters import recover, tail_offset
from extract import MODELS, decode, digest, require, write_json
from ops_investigation import analyze
from packing_investigation import inspect_model, unpack


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    source = resolve_source(parser.parse_args())
    report = {}
    for stem, expected in MODELS.items():
        data = (source/(stem+'.bin')).read_bytes()
        require(digest(data)==expected,'Source identity differs')
        nodes = decode(data)['layers']
        by_name = {n['name']:n for n in nodes}
        graph = analyze(data)
        params, tensors = recover(data)
        packing = inspect_model(base/'artifacts'/stem)
        cases = [
            ('slice_cut',next(n for n in nodes if n['operator_id']==10),74,999),
            ('reshape_count',next(n for n in nodes if n['operator_id']==8),18,99),
            ('branch_stride',by_name['inc3b/conv1'],63,2),
            ('activation_duplicate',by_name['conv1'],335,0),
        ]
        rejected = []
        for label,node,relative,value in cases:
            damaged = bytearray(data)
            struct.pack_into('<I',damaged,tail_offset(node)+relative,value)
            try:
                analyze(bytes(damaged))
            except ValueError as exc:
                rejected.append({'case':label,'reason':str(exc)})
            else:
                raise AssertionError('Graph accepted '+label)
        # An independent scalar offset formula checks the tile interpretation,
        # including the last logical channel next to the two padding channels.
        direct_checks = []
        for record in params['convolutions']:
            if not record['name'].startswith('rpn_cls_score'):
                continue
            raw = tensors[record['tensor_key']+'_weight_i16_packed']
            candidate = unpack(raw,record).reshape(50,448).astype(np.int64)
            x = (np.arange(448,dtype=np.int64)*13)%31-15
            result = candidate@x
            flat = raw.ravel()
            for output in [0,1,3,4,47,48,49]:
                expected_value = 0
                for input_channel in range(448):
                    offset = (((output//4)*112+input_channel//4)*4+output%4)*4+input_channel%4
                    expected_value += int(flat[offset])*int(x[input_channel])
                require(int(result[output])==expected_value,'Blocked scalar reference mismatch')
            damaged = raw.copy().reshape(-1)
            # Output channel50, input0 in the last physical output block.
            damaged[(12*112*4+2)*4] = 1
            try:
                unpack(damaged.reshape(raw.shape),record)
            except ValueError:
                pass
            else:
                raise AssertionError('Nonzero output padding accepted')
            direct_checks.append(record['name'])
        report[stem] = {'source_sha256':expected,'graph_checks':graph['checks'],
                        'packing_roundtrips':packing['roundtrip_layers'],
                        'graph_corruption_rejections':rejected,
                        'classification_scalar_index_checks':direct_checks,
                        'classification_nonzero_padding_rejected':True,
                        'limitations':'These checks validate the recorded hypotheses and parsers, not original DSP equivalence.'}
        print(stem,'static graph, corruption rejection and packed scalar references passed')
    write_json(base/'artifacts/graph-validation.json',report)


if __name__=='__main__':
    main()
