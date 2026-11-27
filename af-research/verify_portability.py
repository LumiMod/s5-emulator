#!/usr/bin/env python3
"""Firmware-free regression tests for input discovery and optional evidence.

All model bytes/graphs here are synthetic fixtures. These tests do not validate
firmware parsing, original inference, real camera data or training readiness.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import extract
import research_paths


class PortableInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='s5-af-portability-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / 'unpacked-v29'
        self.source.mkdir()
        self.output = self.base / 'output'
        self.stem = 'synthetic_model'
        self.blob = b'synthetic fixture, not a CDNN model'
        (self.source / (self.stem + '.bin')).write_bytes(self.blob)
        self.addCleanup(patch.stopall)
        patch.object(extract, 'MODELS', {self.stem: extract.digest(self.blob)}).start()
        patch.object(extract, 'decode', return_value={
            'layers': [], 'classes': [], 'outputs': [],
            'regions': {'synthetic': [0, len(self.blob)]}}).start()

    def test_explicit_root_overrides_environment(self):
        with patch.dict(os.environ, {research_paths.ENVIRONMENT_VARIABLE: str(self.base / 'ignored')}):
            self.assertEqual(research_paths.component_source(self.base), self.source)

    def test_environment_root_and_direct_component_directory(self):
        with patch.dict(os.environ, {research_paths.ENVIRONMENT_VARIABLE: str(self.base)}):
            self.assertEqual(research_paths.component_source(), self.source)
        direct = self.base / 'direct'
        direct.mkdir()
        (direct / '43_hm_d_nw.bin').write_bytes(b'path detection fixture')
        self.assertEqual(research_paths.component_source(direct), direct)

    def test_explicit_source_overrides_root(self):
        self.assertEqual(research_paths.component_source(self.base / 'ignored', self.source), self.source)

    def test_optional_missing_evidence_is_not_reported_passed(self):
        result = extract.extract(self.source, None, self.output)
        model = result['models'][self.stem]
        self.assertTrue(model['pinned_model_sha256_verified'])
        self.assertIsNone(model['zip_matches_component'])
        self.assertIsNone(model['manifest_matches_component'])
        self.assertIsNone(result['firmware_sha256_verified'])
        self.assertEqual(result['zip_check'], 'not_available')
        self.assertEqual((self.output / self.stem / 'original.bin').read_bytes(), self.blob)

    def test_changed_component_rejected_before_outputs(self):
        (self.source / (self.stem + '.bin')).write_bytes(b'corrupted fixture')
        with self.assertRaisesRegex(ValueError, 'Unexpected source digest'):
            extract.extract(self.source, None, self.output)
        self.assertFalse(self.output.exists())

    def test_supplied_wrong_package_is_rejected(self):
        (self.base / 'S5___V29.bin').write_bytes(b'wrong package fixture')
        with self.assertRaisesRegex(ValueError, 'Firmware package digest mismatch'):
            extract.extract(self.source, None, self.output)

    def test_supplied_wrong_manifest_is_rejected(self):
        (self.source / 'verification.json').write_text(json.dumps({
            'source_sha256': extract.FIRMWARE_SHA256,
            'components': [{'file': self.stem + '.bin', 'sha256': '0' * 64, 'matches_component_table': True}]}))
        with self.assertRaisesRegex(ValueError, 'Firmware verification manifest mismatch'):
            extract.extract(self.source, None, self.output)

    def test_optional_archive_must_match(self):
        archive = self.base / 'models.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(self.stem + '.bin', b'wrong archive fixture')
            z.writestr(self.stem + '.layers.tsv', '')
        with self.assertRaisesRegex(ValueError, 'ZIP/source mismatch'):
            extract.extract(self.source, archive, self.output)

    def test_matching_archive_reported_separately_from_missing_package(self):
        archive = self.base / 'models.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(self.stem + '.bin', self.blob)
            z.writestr(self.stem + '.layers.tsv', '')
        result = extract.extract(self.source, archive, self.output)
        self.assertEqual(result['zip_check'], 'passed')
        self.assertEqual(result['firmware_package_check'], 'not_available')


if __name__ == '__main__':
    unittest.main(verbosity=2)
