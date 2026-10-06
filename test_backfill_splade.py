"""Offline migration checks. Run: python -m unittest test_backfill_splade"""

import copy
from types import SimpleNamespace
import unittest

import numpy as np
from qdrant_client import models

from backfill_splade import backfill


class FakeClient:
    def __init__(self):
        self.sparse = {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}
        self.points = [models.Record(
            id=i, payload={"document": f"Document {i}", "metadata": {"source": "original"}},
            vector={"bm25": models.SparseVector(indices=[1], values=[0.5]),
                    "voyage-3": [0.1, 0.2]},
        ) for i in range(5)]
        self.writes = []
        self.http = SimpleNamespace(client=SimpleNamespace(request=self.create_vector))

    def get_collection(self, collection):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors={"voyage-3": models.VectorParams(size=2, distance=models.Distance.COSINE)},
            sparse_vectors=self.sparse)))

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        if with_vectors and "splade" not in self.sparse:
            raise ValueError("Not existing vector name: splade")
        start = offset or 0
        page = copy.deepcopy(self.points[start:start + limit])
        for p in page:
            p.vector = {k: v for k, v in p.vector.items() if with_vectors and k in with_vectors}
        end = start + len(page)
        return page, end if end < len(self.points) else None

    def create_vector(self, type_, method, url, path_params, params, json):
        assert type_ is dict
        assert params == {"wait": "true"}
        assert method == "PUT"
        assert url == "/collections/{collection_name}/vectors/{vector_name}"
        assert path_params["vector_name"] == "splade"
        assert json == {"sparse": {}}
        self.sparse["splade"] = models.SparseVectorParams()
        self.writes.append("schema")

    def update_vectors(self, collection_name, points, wait):
        assert wait
        self.writes.append("vectors")
        for update in points:
            self.points[update.id].vector.update(update.vector)


class FakeModel:
    def __init__(self, model_name):
        assert model_name == "prithivida/Splade_PP_en_v1"

    def embed(self, texts, batch_size):
        for text in texts:
            yield SimpleNamespace(indices=np.array([2, 4]), values=np.array([0.7, 0.3]))


class BackfillTests(unittest.TestCase):
    def test_pagination_preservation_and_resume(self):
        client = FakeClient()
        originals = copy.deepcopy(client.points)
        self.assertEqual(backfill(client, batch_size=2, apply=True, model_factory=FakeModel), 5)
        for original, point in zip(originals, client.points):
            self.assertEqual(point.payload, original.payload)
            self.assertEqual(point.vector["bm25"], original.vector["bm25"])
            self.assertEqual(point.vector["voyage-3"], original.vector["voyage-3"])
            self.assertIn("splade", point.vector)
        writes = list(client.writes)
        self.assertEqual(backfill(client, batch_size=2, apply=True, model_factory=FakeModel), 0)
        self.assertEqual(client.writes, writes)
        self.assertEqual(backfill(client, batch_size=2, apply=True, overwrite=True,
                                  model_factory=FakeModel), 5)

    def test_inspection_does_not_write_or_load_model(self):
        client = FakeClient()
        def forbidden(**kwargs):
            self.fail("Inspection must not load the embedding model")
        self.assertEqual(backfill(client, batch_size=2, model_factory=forbidden), 0)
        self.assertEqual(client.writes, [])

    def test_missing_text_fails_before_any_writes(self):
        client = FakeClient()
        client.points[-1].payload = {"document": " "}
        with self.assertRaisesRegex(ValueError, "Point 4"):
            backfill(client, batch_size=2, apply=True, model_factory=FakeModel)
        self.assertEqual(client.writes, [])

    def test_text_fallback_and_custom_field(self):
        for field in ["text", "content"]:
            client = FakeClient()
            for p in client.points:
                p.payload = {field: "Source text"}
            self.assertEqual(backfill(client, apply=True, model_factory=FakeModel,
                                      text_field=field if field == "content" else None), 5)

    def test_idf_is_rejected(self):
        client = FakeClient()
        client.sparse["splade"] = models.SparseVectorParams(modifier=models.Modifier.IDF)
        with self.assertRaisesRegex(ValueError, "IDF"):
            backfill(client, apply=True, model_factory=FakeModel)
        self.assertEqual(client.writes, [])


if __name__ == "__main__":
    unittest.main()
