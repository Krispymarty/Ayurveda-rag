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
                if "FROM app_state active" in sql:
                    # A later publish does not change the captured binding.
                    return Cursor([("old", "v1", 1, 1, "revision", hashlib.sha256(b"{}").hexdigest())])
                if sql == search_api.retrieval_legacy.SEARCH_SQL:
                    assert parameters["generation_id"] == "old"
                elif sql == search_api.retrieval.SEARCH_SQL:
                    raise AssertionError("ranker changed after binding snapshot")
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
