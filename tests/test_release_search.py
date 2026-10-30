"""A concurrent release must not split a search generation/ranker pair."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus import search_api


class ReleaseSearchTests(unittest.TestCase):
    def test_search_retains_one_release_snapshot_during_activation(self):
        class Cursor:
            def __init__(self, rows):
                self.rows = rows
            def fetchall(self):
                return self.rows
            def fetchone(self):
                return self.rows[0] if self.rows else None

        class Connection:
            def execute(self, sql, parameters=None):
                if "SELECT key,value FROM app_state" in sql:
                    # Publication commits immediately after this snapshot.
                    return Cursor([('active_generation', 'old'), ('ranking_version', 'v1')])
                if "WHERE key='active_generation'" in sql:
                    return Cursor([('old',)])
                if "WHERE key='ranking_version'" in sql:
                    return Cursor([('v2',)])
                if 'SELECT model_revision,manifest_sha256' in sql:
                    return Cursor([('revision', hashlib.sha256(b'{}').hexdigest())])
                return Cursor([])
            def close(self):
                pass

        class Model:
            def encode(self, *args, **kwargs):
                return [[0] * 384]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'data/generations/old/manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_bytes(b'{}')
            with patch.object(search_api, 'ROOT', root), patch.object(search_api, '_connect', return_value=Connection()), \
                 patch.object(search_api, '_model', return_value=Model()), patch.object(search_api, 'model_revision', return_value='revision'), \
                 patch.object(search_api, '_documents', return_value={}), patch.object(search_api, '_published_chunks', return_value={}):
                result = search_api.search('Patents Act')
            self.assertEqual(result['generation_id'], 'old')
            self.assertEqual(result['ranking_version'], 'v1')
            self.assertEqual(result['hits'], [])


if __name__ == '__main__':
    unittest.main()
