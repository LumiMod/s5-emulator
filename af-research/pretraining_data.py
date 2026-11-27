#!/usr/bin/env python3
"""Hash-bound AF dataset checks and explicit AP50/tracking/focus metrics.

No training, image normalization or model inference is performed here.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

CLASSES = ('person', 'person_face', 'animal', 'animal_face', 'bird', 'bird_face')
SPLITS = ('train', 'validation', 'test')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(base, item, key='path', hash_key='sha256'):
    relative = Path(item[key])
    need(not relative.is_absolute(), 'Dataset files must use relative paths')
    path = (base / relative).resolve()
    need(path.is_relative_to(base.resolve()), 'Path escapes dataset root')
    need(path.is_file(), 'Missing file: ' + str(relative))
    need(sha256(path) == item[hash_key], 'Hash mismatch: ' + str(relative))
    return path


def image_dimensions(path):
    """Read PNG/JPEG dimension metadata without assuming an installed decoder."""
    data = path.read_bytes()
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        need(data[12:16] == b'IHDR' and len(data) >= 33, 'Malformed PNG header')
        return struct.unpack_from('>II', data, 16)
    need(data[:2] == b'\xff\xd8', 'Only PNG and JPEG frames supported')
    position = 2
    while position + 4 <= len(data):
        need(data[position] == 255, 'Malformed JPEG marker')
        while data[position] == 255:
            position += 1
        marker = data[position]
        position += 1
        if marker in (0xd8, 0xd9) or 0xd0 <= marker <= 0xd7:
            continue
        size = struct.unpack_from('>H', data, position)[0]
        need(size >= 2 and position + size <= len(data), 'Invalid JPEG segment')
        if marker in (0xc0, 0xc1, 0xc2):
            height, width = struct.unpack_from('>HH', data, position + 3)
            return width, height
        need(marker != 0xda, 'JPEG has no supported SOF dimensions')
        position += size
    raise ValueError('Image dimensions not found')


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_box(box, width, height):
    need(isinstance(box, list) and len(box) == 4 and all(finite_number(v) for v in box), 'Invalid box')
    x1, y1, x2, y2 = box
    need(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, 'Box outside frame or empty')


def validate_dataset(manifest_path, verify_files=True):
    data = json.loads(manifest_path.read_text())
    need(data.get('schema_version') == 1, 'Unsupported dataset schema')
    classes = data.get('classes')
    need(isinstance(classes, list) and classes and len(set(classes)) == len(classes), 'Invalid classes')
    need(set(classes) <= set(CLASSES), 'Class outside extracted S5 vocabulary')
    frames = data.get('frames')
    need(isinstance(frames, list) and frames, 'Empty dataset')
    ids, frame_keys, sequences, hashes = set(), set(), {}, {}
    counts, boxes_by_split, conditions, track_frames = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    sequence_times = defaultdict(list)
    focus_count = Counter()
    for frame in frames:
        fid = frame['id']; split = frame['split']; sequence = frame['sequence_id']
        need(isinstance(fid, str) and fid and fid not in ids, 'Duplicate/invalid frame ID')
        need(isinstance(sequence, str) and sequence, 'Invalid sequence ID')
        need(split in SPLITS, 'Unknown split')
        need(frame.get('source_kind') == 'camera_capture', 'Training frames require camera-capture provenance')
        need(isinstance(frame.get('capture_id'), str) and frame['capture_id'], 'Missing capture provenance ID')
        need(sequence not in sequences or sequences[sequence] == split, 'Sequence leakage across splits')
        need(frame['sha256'] not in hashes or hashes[frame['sha256']] == split, 'Identical image leakage across splits')
        need(isinstance(frame['frame_index'], int) and frame['frame_index'] >= 0, 'Invalid frame index')
        need(finite_number(frame['timestamp_s']) and frame['timestamp_s'] >= 0, 'Invalid timestamp')
        key = (sequence, frame['frame_index'])
        need(key not in frame_keys, 'Repeated sequence/frame index')
        width, height = frame['width'], frame['height']
        need(isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0, 'Invalid dimensions')
        if verify_files:
            path = checked_file(manifest_path.parent, frame)
            need(image_dimensions(path) == (width, height), 'Image header dimensions mismatch')
        light = frame['conditions']['light']
        need(light in ('low', 'normal', 'backlit'), 'Invalid lighting stratum')
        need(frame['conditions']['motion'] in ('static', 'moving'), 'Invalid motion stratum')
        ids.add(fid); frame_keys.add(key); sequences[sequence] = split; hashes[frame['sha256']] = split
        sequence_times[sequence].append((frame['frame_index'], frame['timestamp_s']))
        counts[split] += 1; conditions[split][light] += 1
        present_tracks = set()
        need(isinstance(frame['boxes'], list), 'Boxes must be a list, including empty negative frames')
        for box in frame['boxes']:
            need(box['class'] in classes, 'Unknown annotation class')
            validate_box(box['bbox_xyxy'], width, height)
            boxes_by_split[split][box['class']] += 1
            track = box.get('track_id')
            if track is not None:
                need(isinstance(track, str) and track and track not in present_tracks, 'Duplicate/invalid track in frame')
                present_tracks.add(track)
                track_frames[(split, sequence, track)].add(frame['frame_index'])
        focus = frame.get('focus')
        if focus is not None:
            for field in ('lens_id', 'position_unit', 'ground_truth_method'):
                need(isinstance(focus[field], str) and focus[field], 'Missing focus field ' + field)
            for field in ('position', 'target_position', 'focal_length_mm', 'aperture', 'iso', 'illuminance_lux'):
                need(finite_number(focus[field]), 'Invalid focus value ' + field)
            need(all(focus[field] > 0 for field in ('focal_length_mm', 'aperture', 'iso')), 'Invalid lens/exposure values')
            need(focus['illuminance_lux'] >= 0, 'Invalid illuminance')
            if verify_files:
                checked_file(manifest_path.parent, focus, 'trace_path', 'trace_sha256')
            focus_count[split] += 1
    for sequence, times in sequence_times.items():
        ordered = sorted(times)
        need(all(right[1] > left[1] for left, right in zip(ordered, ordered[1:])),
             'Timestamps not strictly increasing in ' + sequence)
    gaps = []
    for split in ('train', 'validation'):
        if not counts[split]: gaps.append('missing_' + split + '_frames')
        for cls in classes:
            if not boxes_by_split[split][cls]: gaps.append('missing_' + split + '_class_' + cls)
        if not conditions[split]['low']: gaps.append('missing_' + split + '_low_light')
        if not any(k[0] == split and len(v) >= 2 for k, v in track_frames.items()):
            gaps.append('missing_' + split + '_multi_frame_track')
        if not focus_count[split]: gaps.append('missing_' + split + '_physical_focus_trace')
    return {'manifest_sha256': sha256(manifest_path), 'frames': len(frames), 'sequences': len(sequences),
            'split_counts': dict(counts), 'class_counts': {k: dict(v) for k, v in boxes_by_split.items()},
            'lighting_counts': {k: dict(v) for k, v in conditions.items()},
            'focus_trace_counts': dict(focus_count), 'data_gaps': gaps,
            'structural_validation_passed': True,
            'provenance_limit': 'File hashes and dimensions verified; capture authenticity and labels need independent review.'}, data


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (left[2]-left[0])*(left[3]-left[1]) + (right[2]-right[0])*(right[3]-right[1]) - intersection
    return intersection / union if union else 0.0


def evaluate_detection(dataset, predictions, split='validation', threshold=0.5, score_threshold=0.5):
    """All-points interpolated AP at one IoU, plus matched track-ID switches.

    This is explicitly not the COCO mAP metric. Empty frames are evaluated.
    """
    need(predictions.get('dataset_manifest_sha256'), 'Prediction manifest must bind a dataset digest')
    frames = {f['id']: f for f in dataset['frames'] if f['split'] == split}
    need(frames, 'No requested evaluation split')
    rows = predictions.get('frames')
    need(isinstance(rows, list), 'Missing prediction rows')
    pred_by_id = {}
    for row in rows:
        fid = row['id']
        need(fid in frames and fid not in pred_by_id, 'Unknown/duplicate prediction frame')
        pred_by_id[fid] = row['detections']
        for detection in row['detections']:
            need(detection['class'] in dataset['classes'], 'Unknown predicted class')
            need(finite_number(detection['score']) and 0 <= detection['score'] <= 1, 'Invalid confidence')
            validate_box(detection['bbox_xyxy'], frames[fid]['width'], frames[fid]['height'])
    need(set(pred_by_id) == set(frames), 'Predictions must cover every held-out frame, including no detections')
    results, track_events = {}, []
    for cls in dataset['classes']:
        ground_truth = {fid: [b for b in f['boxes'] if b['class'] == cls] for fid, f in frames.items()}
        positive_count = sum(len(v) for v in ground_truth.values())
        ordered = sorted([(d['score'], fid, index, d) for fid, rows in pred_by_id.items()
                          for index, d in enumerate(rows) if d['class'] == cls], key=lambda row: (-row[0], row[1], row[2]))
        matched = defaultdict(set); precisions, recalls = [], []; tp = fp = 0; threshold_tp = threshold_fp = 0
        for score, fid, _, det in ordered:
            available = [(iou(det['bbox_xyxy'], box['bbox_xyxy']), index) for index, box in enumerate(ground_truth[fid])
                         if index not in matched[fid]]
            overlap, index = max(available, default=(0.0, -1))
            if index >= 0 and overlap >= threshold:
                tp += 1; matched[fid].add(index)
                if score >= score_threshold: threshold_tp += 1
                truth = ground_truth[fid][index]
                if score >= score_threshold and truth.get('track_id') and det.get('track_id'):
                    frame = frames[fid]
                    track_events.append((frame['sequence_id'], frame['frame_index'], truth['track_id'], det['track_id']))
            else:
                fp += 1
                if score >= score_threshold: threshold_fp += 1
            precisions.append(tp/(tp+fp)); recalls.append(tp/positive_count if positive_count else 0)
        envelope = precisions[:]
        for index in range(len(envelope)-2, -1, -1): envelope[index] = max(envelope[index], envelope[index+1])
        previous, ap = 0.0, 0.0
        for recall, precision in zip(recalls, envelope):
            ap += (recall-previous)*precision; previous = recall
        results[cls] = {'ground_truth_boxes': positive_count, 'predictions': len(ordered),
                        'ap_at_iou': ap if positive_count else None,
                        'tp_at_score': threshold_tp, 'fp_at_score': threshold_fp,
                        'fn_at_score': positive_count-threshold_tp}
    previous_ids, switches = {}, 0
    for sequence, _, truth_id, pred_id in sorted(track_events):
        key = (sequence, truth_id)
        if key in previous_ids and previous_ids[key] != pred_id: switches += 1
        previous_ids[key] = pred_id
    valid_ap = [r['ap_at_iou'] for r in results.values() if r['ap_at_iou'] is not None]
    return {'metric': 'all-points interpolated AP at one IoU; not COCO mAP', 'split': split,
            'iou_threshold': threshold, 'score_threshold': score_threshold, 'frame_count': len(frames),
            'classes': results, 'mean_ap': sum(valid_ap)/len(valid_ap) if valid_ap else None,
            'matched_track_id_switches': switches, 'matched_track_observations': len(track_events),
            'tracking_limit': 'Simple identity changes among matched detections, not MOTA/HOTA; misses reported as detection FN.'}


def evaluate_focus_baseline(dataset, split='validation'):
    groups = defaultdict(list)
    for frame in dataset['frames']:
        focus = frame.get('focus')
        if frame['split'] != split or focus is None:
            continue
        # Different lens encoders, units and reference methods must not be
        # pooled into an apparently meaningful universal focus score.
        key = (focus['lens_id'],focus['position_unit'],focus['ground_truth_method'],frame['conditions']['light'])
        groups[key].append(abs(focus['position']-focus['target_position']))
    return {'metric':'captured absolute lens-position error against supplied reference',
            'split':split,'groups':[{'lens_id':key[0],'unit':key[1],'ground_truth_method':key[2],
                                    'light':key[3],'observations':len(values),
                                    'mean_absolute_error':statistics.mean(values),
                                    'median_absolute_error':statistics.median(values),
                                    'maximum_absolute_error':max(values)} for key,values in sorted(groups.items())],
            'limitations':'Depends on physical target-position truth; does not establish image sharpness, lock time or controller quality.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--split', choices=SPLITS, default='validation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, dataset = validate_dataset(args.dataset)
    report['captured_focus_baseline'] = evaluate_focus_baseline(dataset,args.split)
    if args.predictions:
        predictions = json.loads(args.predictions.read_text())
        need(predictions['dataset_manifest_sha256'] == report['manifest_sha256'], 'Prediction dataset digest differs')
        report['evaluation'] = evaluate_detection(dataset, predictions, args.split)
        report['predictions_sha256'] = sha256(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
