import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from collection import EVENT_FIELDS,initialize,inventory,validate,package


class CollectionChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'session'
        initialize(self.root,'pilot01')
        self.frame=self.root/'raw/S01_static_normal/camera/P1000001.JPG'
        self.frame.write_bytes(b'fixture-only-not-a-real-photo')
        inventory(self.root)
    def tearDown(self):self.temp.cleanup()
    def test_unknown_telemetry_allowed_but_not_training_ready(self):
        p=self.root/'sequences.json';data=json.loads(p.read_text())
        data['sequences'][0].update(light=None,motion=None,illuminance_lux=None)
        p.write_text(json.dumps(data));inventory(self.root)
        r=validate(self.root);self.assertEqual(r['media_files'],1);self.assertFalse(r['ready_for_training'])
        self.assertFalse(r['raw_media_decoding_verified'])
    def test_manual_reference_does_not_mislabel_mixed_lighting(self):
        data=json.loads((self.root/'sequences.json').read_text())
        reference=next(row for row in data['sequences'] if row['sequence_id']=='S06_manual_reference')
        self.assertEqual(reference['light'],'unknown')
        self.assertIn('normal/low',reference['notes'])
        self.assertIn('文件名',reference['notes'])
        self.assertIn('拆成独立序列',reference['notes'])
    def test_changed_file(self):
        self.frame.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'文件改变'):validate(self.root)
    def test_added_file(self):
        (self.frame.parent/'new.JPG').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'新增文件'):validate(self.root)
    def test_external_path_rejected(self):
        p=self.root/'collection-manifest.json';j=json.loads(p.read_text());j['files'][0]['path']='../outside'
        p.write_text(json.dumps(j))
        with self.assertRaisesRegex(ValueError,'路径必须'):validate(self.root)
    def test_symlink_rejected(self):
        (self.frame.parent/'link.JPG').symlink_to(self.frame)
        with self.assertRaisesRegex(ValueError,'符号链接'):inventory(self.root)
    def test_package_and_checksum(self):
        out=self.base/'handoff.zip';package(self.root,out)
        with zipfile.ZipFile(out) as z:
            self.assertEqual(z.read('session/raw/S01_static_normal/camera/P1000001.JPG'),self.frame.read_bytes())
        self.assertTrue(out.with_name('handoff.zip.sha256').is_file())
        with self.assertRaisesRegex(ValueError,'已存在'):package(self.root,out)
    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,'已存在'):initialize(self.root,'pilot01')
    def rewrite_manifest(self,change):
        p=self.root/'collection-manifest.json';data=json.loads(p.read_text())
        change(data);p.write_text(json.dumps(data))
    def write_event(self,**values):
        with (self.root/'events.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=EVENT_FIELDS);writer.writeheader()
            writer.writerow(dict(sequence_id='S01_static_normal',trial_id='near_1',**values))
        inventory(self.root)
    def test_manifest_output_symlink_rejected(self):
        manifest=self.root/'collection-manifest.json';manifest.unlink()
        outside=self.base/'outside.json';outside.write_text('preserve')
        manifest.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'符号链接'):inventory(self.root)
        self.assertEqual(outside.read_text(),'preserve')
    def test_symlink_session_rejected(self):
        alias=self.base/'alias';alias.symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'采集目录无效'):validate(alias)
    def test_noncanonical_manifest_path_rejected(self):
        self.rewrite_manifest(lambda d:d['files'][0].update(path='./session.json'))
        with self.assertRaisesRegex(ValueError,'规范形式'):validate(self.root)
    def test_windows_path_escape_rejected(self):
        self.rewrite_manifest(lambda d:d['files'][0].update(path='..\\outside'))
        with self.assertRaisesRegex(ValueError,'相对 POSIX'):validate(self.root)
    def test_manifest_scope_rejected(self):
        private=self.root/'private-note.txt';private.write_bytes(b'unrelated')
        def add(data):
            data['files'].append({'path':private.name,'bytes':private.stat().st_size,
                                  'sha256':hashlib.sha256(private.read_bytes()).hexdigest()})
            data['total_bytes']+=private.stat().st_size
        self.rewrite_manifest(add)
        with self.assertRaisesRegex(ValueError,'范围外'):validate(self.root)
    def test_manifest_total_rejected(self):
        self.rewrite_manifest(lambda d:d.update(total_bytes=0))
        with self.assertRaisesRegex(ValueError,'总字节数'):validate(self.root)
    def test_empty_media_rejected(self):
        self.frame.write_bytes(b'');inventory(self.root)
        with self.assertRaisesRegex(ValueError,'素材文件为空'):validate(self.root)
    def test_missing_settings_directory_rejected(self):
        (self.root/'settings').rmdir()
        with self.assertRaisesRegex(ValueError,'缺少目录'):validate(self.root)
    def test_optional_event_timestamps_allowed(self):
        self.write_event(camera_file=self.frame.name,event='acquired')
        self.assertEqual(validate(self.root)['events'],1)
    def test_timestamp_without_file_rejected(self):
        self.write_event(phone_time_s='1.2')
        with self.assertRaisesRegex(ValueError,'必须对应一个文件'):validate(self.root)
    def test_nonfinite_event_timestamp_rejected(self):
        self.write_event(camera_file=self.frame.name,camera_time_s='nan')
        with self.assertRaisesRegex(ValueError,'非负秒数'):validate(self.root)
    def test_csv_short_row_rejected(self):
        with (self.root/'events.csv').open('a') as f:f.write('S01_static_normal,trial1\n')
        inventory(self.root)
        with self.assertRaisesRegex(ValueError,'字段数量'):validate(self.root)
    def test_csv_extra_field_rejected(self):
        with (self.root/'events.csv').open('a') as f:
            csv.writer(f).writerow(['S01_static_normal']+['']*len(EVENT_FIELDS))
        inventory(self.root)
        with self.assertRaisesRegex(ValueError,'字段数量'):validate(self.root)
    def test_existing_checksum_preserved(self):
        out=self.base/'handoff.zip';checksum=out.with_name(out.name+'.sha256');checksum.write_text('keep')
        with self.assertRaisesRegex(ValueError,'校验文件已存在'):package(self.root,out)
        self.assertFalse(out.exists());self.assertEqual(checksum.read_text(),'keep')
    def test_changed_during_pack_rejected_and_cleaned(self):
        out=self.base/'handoff.zip';original_write=zipfile.ZipFile.write
        def changing_write(archive,filename,*args,**kwargs):
            if Path(filename)==self.frame:self.frame.write_bytes(b'changed-during-copy')
            return original_write(archive,filename,*args,**kwargs)
        with mock.patch.object(zipfile.ZipFile,'write',changing_write):
            with self.assertRaisesRegex(ValueError,'ZIP 内容校验失败'):package(self.root,out)
        self.assertFalse(out.exists());self.assertFalse(out.with_name(out.name+'.sha256').exists())
    def test_pack_embeds_validated_manifest_snapshot(self):
        out=self.base/'handoff.zip';manifest=self.root/'collection-manifest.json'
        expected=manifest.read_bytes();original_write=zipfile.ZipFile.write
        def changing_write(archive,filename,*args,**kwargs):
            manifest.write_text('{"changed_after_validation":true}')
            return original_write(archive,filename,*args,**kwargs)
        with mock.patch.object(zipfile.ZipFile,'write',changing_write):package(self.root,out)
        with zipfile.ZipFile(out) as z:
            self.assertEqual(z.read('session/collection-manifest.json'),expected)
    def test_empty_template_cannot_be_packaged(self):
        self.frame.unlink();inventory(self.root)
        with self.assertRaisesRegex(ValueError,'没有素材'):package(self.root,self.base/'empty.zip')
    def test_root_parent_not_emitted_as_zip_traversal(self):
        child=self.root/'child';child.mkdir();out=self.base/'handoff.zip'
        package(child/'..',out)
        with zipfile.ZipFile(out) as z:
            self.assertTrue(all(name.startswith('session/') and '..' not in Path(name).parts for name in z.namelist()))


if __name__=='__main__':unittest.main(verbosity=2)
