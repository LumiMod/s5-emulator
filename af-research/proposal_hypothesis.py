#!/usr/bin/env python3
"""Experimental proposal decoding using conventional Faster R-CNN semantics.

Semantic reference (not evidence of Panasonic ABI):
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py
No claim of original CDNN output equivalence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from extract import require, write_json


def anchors(base, ratios, scales, height, width, stride):
    ratios = np.asarray(ratios,dtype=np.float64)
    scales = np.asarray(scales,dtype=np.float64)
    require(base>0 and np.all(ratios>0) and np.all(scales>0),'Invalid anchors')
    widths = np.rint(np.sqrt(base*base/ratios))
    heights = np.rint(widths*ratios)
    sizes = np.stack((widths[:,None]*scales,heights[:,None]*scales),axis=-1).reshape(-1,2)
    center = (base-1)/2
    base_boxes = np.concatenate((center-(sizes-1)/2,center+(sizes-1)/2),axis=-1)
    sy,sx = stride
    yy,xx = np.mgrid[:height,:width]
    shifts = np.stack((xx*sx,yy*sy,xx*sx,yy*sy),axis=-1).reshape(-1,4)
    return (shifts[:,None,:]+base_boxes[None,:,:]).reshape(-1,4)


def decode_boxes(reference, deltas, image_hw, coordinate_mode='faster_rcnn'):
    require(reference.shape==deltas.shape and reference.shape[1]==4,'Invalid delta shape')
    require(np.isfinite(reference).all() and np.isfinite(deltas).all(),'Nonfinite boxes')
    sizes = reference[:,2:]-reference[:,:2]+1
    centers = reference[:,:2]+sizes/2
    predicted_centers = centers+deltas[:,:2]*sizes
    # Reject extreme exponential values instead of silently introducing a new clamp.
    require(np.max(np.abs(deltas[:,2:]),initial=0)<80,'Delta exponent outside experiment range')
    predicted_sizes = np.exp(deltas[:,2:])*sizes
    far_adjust = 1 if coordinate_mode=='inclusive_identity' else 0
    require(coordinate_mode in ['faster_rcnn','inclusive_identity'],'Unknown coordinate mode')
    boxes = np.concatenate((predicted_centers-predicted_sizes/2,
                            predicted_centers+predicted_sizes/2-far_adjust),axis=1)
    h,w=image_hw
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w-1)
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h-1)
    return boxes


def nms(boxes,scores,threshold,limit):
    require(0<=threshold<=1 and limit>0,'Invalid NMS configuration')
    order=np.argsort(-scores,kind='stable')
    areas=np.prod(np.maximum(boxes[:,2:]-boxes[:,:2]+1,0),axis=1)
    keep=[]
    while len(order) and len(keep)<limit:
        index=int(order[0]);keep.append(index);rest=order[1:]
        intersection=np.prod(np.maximum(np.minimum(boxes[index,2:],boxes[rest,2:])-
                                        np.maximum(boxes[index,:2],boxes[rest,:2])+1,0),axis=1)
        union=areas[index]+areas[rest]-intersection
        overlap=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
        order=rest[overlap<=threshold]
    return np.asarray(keep,dtype=np.int64)


def proposals(probabilities,deltas,metadata,parameters,image_hw=(256,384),
              coordinate_mode='faster_rcnn',minimum_size=0):
    require(probabilities.ndim==4 and deltas.ndim==4 and probabilities.shape[0]==deltas.shape[0]==1,
            'Expected single-batch NCHW')
    _,channels,h,w=probabilities.shape
    count=metadata['ratio_count_candidate']*metadata['scale_count_candidate']
    require(channels==count*2 and deltas.shape==(1,count*4,h,w),'Anchor head dimension mismatch')
    require(np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1)),
            'Invalid probabilities')
    require(np.allclose(probabilities[:,:count]+probabilities[:,count:],1,atol=2e-6),
            'Paired class probabilities do not normalize')
    ref=anchors(metadata['anchor_base_size_candidate'],parameters['ratio_candidates'],
                parameters['scale_candidates'],h,w,metadata['feature_stride_candidate'])
    scores=probabilities[0,count:].transpose(1,2,0).reshape(-1).astype(np.float64)
    delta=deltas[0].transpose(1,2,0).reshape(-1,4).astype(np.float64)
    boxes=decode_boxes(ref,delta,image_hw,coordinate_mode)
    sizes=boxes[:,2:]-boxes[:,:2]+1
    valid=np.flatnonzero(np.all(sizes>=minimum_size,axis=1))
    selected=valid[np.argsort(-scores[valid],kind='stable')[:metadata['pre_nms_top_n_candidate']]]
    kept=nms(boxes[selected],scores[selected],metadata['nms_threshold_candidate'],metadata['post_nms_top_n_candidate'])
    selected=selected[kept]
    return {'boxes_xyxy':boxes[selected].tolist(),'scores':scores[selected].tolist(),
            'anchor_indices':selected.tolist(),'input_anchor_count':len(ref),
            'coordinate_mode_hypothesis':coordinate_mode,'minimum_size_hypothesis':minimum_size,
            'unverified':['anchor/channel ordering','bbox coordinate convention','minimum size and filtering',
                          'original CDNN NMS and output ABI','upstream model/preprocessing']}


def self_check():
    # Hand-computed zero-based anchors: center7.5, widths23/16/11,
    # heights12/16/22, multiplied by8/16/32. The upstream source's
    # comment example is one pixel higher than its actual zero-based formula.
    expected=np.array([[-84,-40,99,55],[-176,-88,191,103],[-360,-184,375,199],
                       [-56,-56,71,71],[-120,-120,135,135],[-248,-248,263,263],
                       [-36,-80,51,95],[-80,-168,95,183],[-168,-344,183,359]])
    require(np.array_equal(anchors(16,[.5,1,2],[8,16,32],1,1,[16,16]),expected),'Reference anchor check failed')
    b=np.array([[10.,20.,29.,39.],[10.,20.,29.,39.],[70.,70.,79.,79.]])
    require(np.array_equal(decode_boxes(b,np.zeros_like(b),(100,100),'inclusive_identity'),b),'Identity decode failed')
    require(np.array_equal(nms(b,np.array([.9,.8,.7]),.3,30),[0,2]),'Duplicate NMS check failed')
    require(np.array_equal(nms(b,np.array([.8,.9,.7]),.3,1),[1]),'Ranking/top-N check failed')
    print('Reference anchors, identity convention, duplicate suppression and top-N checks passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--outputs',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--coordinate-mode',choices=['faster_rcnn','inclusive_identity'],default='faster_rcnn')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        self_check();return
    require(all([args.model_dir,args.outputs,args.output]),'Specify --model-dir, --outputs and --output')
    graph=json.loads((args.model_dir/'operator-investigation.json').read_text())
    params=json.loads((args.model_dir/'parameters.json').read_text())
    nodes=[n for n in graph['layers'] if n['operator_id']==24]
    results={}
    with np.load(args.outputs,allow_pickle=False) as outputs:
        for node,p in zip(nodes,params['proposal_blocks']):
            results[node['name']]=proposals(outputs[node['name']+'_probabilities'],outputs['bbox_deltas'],node,p,
                                           coordinate_mode=args.coordinate_mode)
    write_json(args.output,{'status':'unverified proposal experiment','source_sha256':params['source_sha256'],
                           'results':results,'class_mapping':'Proposal number to class label still requires confirmation.'})


if __name__=='__main__':
    main()
