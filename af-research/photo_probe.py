#!/usr/bin/env python3
"""Run an explicitly unverified S5 model hypothesis on a photograph. No training."""
import argparse
import json
from pathlib import Path

from research_paths import add_firmware_arguments, resolve_source

import numpy as np
from PIL import Image, ImageOps
import torch

from candidate_network import CandidateNetwork
from decode_parameters import recover
from extract import MODELS, digest, require, write_json
from packing_investigation import SPATIAL_LAYOUTS
from proposal_hypothesis import proposals


def prepare_image(path, mean=117.0, scale=1.0):
    """Candidate preprocessing only: EXIF orientation, grayscale, resize, center."""
    require(np.isfinite(mean) and np.isfinite(scale) and scale>0,'Invalid preprocessing values')
    with Image.open(path) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB')
        original_size=image.size
        gray=image.convert('L').resize((384,256),Image.Resampling.BILINEAR)
        array=(np.asarray(gray,dtype=np.float32)-np.float32(mean))*np.float32(scale)
    return torch.from_numpy(array.copy())[None,None],{
        'input_file_sha256':digest(Path(path).read_bytes()),'oriented_image_wh':list(original_size),
        'method':'EXIF transpose; Pillow RGB->L; bilinear stretch to384x256; (pixel-mean)*scale',
        'mean_hypothesis':mean,'scale_hypothesis':scale,'verified_against_camera':False}


def run(image_path,data,output,layouts,mean,scale):
    torch.set_num_threads(2)
    tensor,preprocessing=prepare_image(image_path,mean,scale)
    params,_=recover(data)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'prepared_input.npz',prepared_input=tensor.numpy())
    runs=[]
    for layout in layouts:
        torch.manual_seed(0)
        model=CandidateNetwork(data,layout).eval()
        with torch.inference_mode():heads,trace=model(tensor,True)
        head_arrays={name:value.numpy() for name,value in heads.items()}
        proposals_result={}
        proposal_nodes=[n for n in model.layers if n['operator_id']==24]
        for node,block in zip(proposal_nodes,params['proposal_blocks']):
            proposals_result[node['name']]=proposals(head_arrays[node['name']+'_probabilities'],
                head_arrays['bbox_deltas'],node,block,coordinate_mode='faster_rcnn')
        folder=output/layout;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'rpn_outputs.npz',**head_arrays)
        report={'status':'photo inference hypothesis; not validated recognition',
                'model_sha256':digest(data),'spatial_layout':layout,'preprocessing':preprocessing,
                'proposal_results':proposals_result,'trace':trace,
                'class_mapping_unverified':True,'formal_training_started':False,
                'limitations':['Upstream channel layout/activation/upsample remain hypotheses.',
                               'Class number and predicted box must not be treated as confirmed camera output.',
                               'A plausible box or high score on one image does not establish fidelity.']}
        write_json(folder/'photo_probe.json',report)
        runs.append({'layout':layout,'maximum_scores_by_proposal':{
            name:max(item['scores'],default=0) for name,item in proposals_result.items()},
            'finite_outputs':True})
        print(layout,runs[-1]['maximum_scores_by_proposal'])
    write_json(output/'summary.json',{'model_sha256':digest(data),'preprocessing':preprocessing,
                                    'runs':runs,'status':'hypothesis smoke tests only','training_started':False})


def main():
    base=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',choices=list(MODELS),default='44_hm_d_nw_sng')
    add_firmware_arguments(parser)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layout',choices=SPATIAL_LAYOUTS,default='o4_i4_hw')
    parser.add_argument('--compare-layouts',action='store_true')
    parser.add_argument('--mean',type=float,default=117)
    parser.add_argument('--scale',type=float,default=1)
    args=parser.parse_args()
    args.source = resolve_source(args)
    data=(args.source/(args.model+'.bin')).read_bytes()
    require(digest(data)==MODELS[args.model],'Model identity differs')
    run(args.image,data,args.output,SPATIAL_LAYOUTS if args.compare_layouts else [args.layout],args.mean,args.scale)


if __name__=='__main__':
    main()
