#!/usr/bin/env python3
"""Synthetic utility tests only; never counted as original AF evidence."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from pretraining_data import validate_dataset, evaluate_detection, evaluate_focus_baseline, sha256
from pretraining_readiness import compare_captures


def png(path, pixel):
    def chunk(kind, content):
        return struct.pack('>I',len(content))+kind+content+struct.pack('>I',zlib.crc32(kind+content)&0xffffffff)
    raw = b''.join(b'\0'+bytes([pixel])*16 for _ in range(16))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',16,16,8,0,0,0,0))+
                     chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='s5-af-data-')
        self.base = Path(self.temp.name)
        self.manifest = self.base/'dataset.json'
        frames = []
        for index,(split,sequence) in enumerate([('train','a'),('train','a'),('validation','b'),('validation','b'),('validation','b')]):
            path = self.base/('%d.png'%index); png(path,index)
            frames.append({'id':str(index),'path':path.name,'sha256':sha256(path),'width':16,'height':16,
                           'split':split,'sequence_id':sequence,'frame_index':index,'timestamp_s':float(index),
                           'source_kind':'camera_capture','capture_id':'fixture-only-not-real-data',
                           'conditions':{'light':'low','motion':'moving'},
                           'boxes':[] if index==4 else [{'class':'person','bbox_xyxy':[1,1,8,8],'track_id':'p'}]})
        self.data = {'schema_version':1,'classes':['person'],'frames':frames}
        self.save(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def save(self,data):
        self.manifest.write_text(json.dumps(data))

    def test_valid_structure_keeps_missing_focus_gap(self):
        report,_ = validate_dataset(self.manifest)
        self.assertTrue(report['structural_validation_passed'])
        self.assertIn('missing_validation_physical_focus_trace',report['data_gaps'])

    def test_sequence_leak(self):
        self.data['frames'][2]['sequence_id']='a';self.save(self.data)
        with self.assertRaisesRegex(ValueError,'Sequence leakage'):validate_dataset(self.manifest)

    def test_content_leak(self):
        self.data['frames'][2]['sha256']=self.data['frames'][0]['sha256'];self.save(self.data)
        with self.assertRaisesRegex(ValueError,'Identical image leakage'):validate_dataset(self.manifest)

    def test_corrupt_file_hash(self):
        path=self.base/'2.png';path.write_bytes(path.read_bytes()+b'damaged')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):validate_dataset(self.manifest)

    def test_wrong_box(self):
        self.data['frames'][0]['boxes'][0]['bbox_xyxy']=[0,0,99,99];self.save(self.data)
        with self.assertRaisesRegex(ValueError,'Box outside'):validate_dataset(self.manifest)

    def test_known_ap_and_track_switch(self):
        def detection(score,track):return {'class':'person','bbox_xyxy':[1,1,8,8],'score':score,'track_id':track}
        prediction={'dataset_manifest_sha256':sha256(self.manifest),'frames':[
            {'id':'2','detections':[detection(.9,'x'),detection(.8,'duplicate')]},
            {'id':'3','detections':[detection(.7,'y')]},{'id':'4','detections':[]}]}
        result=evaluate_detection(self.data,prediction)
        self.assertAlmostEqual(result['mean_ap'],5/6)
        self.assertEqual(result['matched_track_id_switches'],1)
        self.assertEqual(result['classes']['person']['fp_at_score'],1)
        prediction['frames'].pop()
        with self.assertRaisesRegex(ValueError,'cover every'):evaluate_detection(self.data,prediction)

    def test_synthetic_capture_cannot_be_original(self):
        path=self.base/'capture.json';path.write_text(json.dumps({'schema_version':1,'captures':[
            {'id':'fake','model':'43_hm_d_nw','capture_kind':'synthetic_candidate'}]}))
        with self.assertRaisesRegex(ValueError,'not original evidence'):compare_captures(path,self.base)

    def test_focus_units_not_pooled(self):
        for index,position,unit in [(2,10,'ticks'),(3,14,'ticks'),(4,0.5,'mm')]:
            self.data['frames'][index]['focus']={'lens_id':'fixture','position_unit':unit,
                'ground_truth_method':'fixture','position':position,'target_position':12 if unit=='ticks' else 0.2}
        result=evaluate_focus_baseline(self.data)
        self.assertEqual(len(result['groups']),2)
        ticks=next(group for group in result['groups'] if group['unit']=='ticks')
        self.assertEqual(ticks['median_absolute_error'],2)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ToolsTest)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    base=Path(__file__).resolve().parent
    evidence={'fixture_only':True,'tests_run':result.testsRun,'failures':len(result.failures),
              'errors':len(result.errors),'successful':result.wasSuccessful(),
              'source_hashes':{name:sha256(base/name) for name in
                  ['pretraining_data.py','pretraining_readiness.py','verify_pretraining_tools.py']}}
    (base/'artifacts').mkdir(exist_ok=True)
    (base/'artifacts/pretraining-tools-validation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
