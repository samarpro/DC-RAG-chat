"""Add the app's SPLADE named vector to existing Qdrant points.

Inspect only (default): python backfill_splade.py
Write missing vectors: python backfill_splade.py --apply
Recompute existing SPLADE vectors: python backfill_splade.py --apply --overwrite

Uses .env beside this script. Existing vectors and payloads are preserved.
The first apply run may download the FastEmbed SPLADE model.
Creating a named vector requires Qdrant server 1.18 or later.
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

COLLECTION = "hybrid-search-splade"
VECTOR_NAME = "splade"
MODEL_NAME = "prithivida/Splade_PP_en_v1"
LOGGER = logging.getLogger(__name__)


def batches(client, collection, batch_size, include_splade=True):
    """Scroll all points without downloading unrelated dense vectors."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=[VECTOR_NAME] if include_splade else False,
        )
        if points:
            yield points
        if offset is None:
            break


def document_text(point, text_field):
    payload = point.payload or {}
    # Live collection uses "document"; main.py expects "text".
    fields = [text_field] if text_field else ["document", "text"]
    for field in fields:
        text = payload.get(field)
        if isinstance(text, str) and text.strip():
            return text
    raise ValueError(
        f"Point {point.id!r} has no nonempty string in payload fields {fields}. "
        "Use --text-field to select the source text."
    )


def has_splade(point):
    return isinstance(point.vector, dict) and VECTOR_NAME in point.vector


def backfill(client, collection=COLLECTION, batch_size=16, text_field=None,
             apply=False, overwrite=False, model_factory=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    info = client.get_collection(collection)
    dense = info.config.params.vectors
    sparse = info.config.params.sparse_vectors or {}
    if isinstance(dense, dict) and VECTOR_NAME in dense:
        raise ValueError("The existing 'splade' vector is dense, not sparse.")
    if VECTOR_NAME in sparse and sparse[VECTOR_NAME].modifier == models.Modifier.IDF:
        raise ValueError("The existing 'splade' vector has an IDF modifier, unsuitable for SPLADE.")

    # Validate every candidate before changing the collection or any points.
    scanned = existing = pending = 0
    for points in batches(client, collection, batch_size, include_splade=VECTOR_NAME in sparse):
        for point in points:
            scanned += 1
            if has_splade(point) and not overwrite:
                existing += 1
                continue
            document_text(point, text_field)
            pending += 1
    LOGGER.info("Collection %s: %d points, %d already populated, %d to embed",
                collection, scanned, existing, pending)
    if not apply:
        LOGGER.info("Inspection only. Run with --apply to write SPLADE vectors.")
        return 0
    if not pending:
        return 0

    if model_factory is None:
        from fastembed import SparseTextEmbedding
        model_factory = SparseTextEmbedding
    model = model_factory(model_name=MODEL_NAME)
    if VECTOR_NAME not in sparse:
        # The pinned client predates create_vector_name. Use its authenticated
        # HTTP transport for the Qdrant 1.18+ vector schema creation endpoint.
        client.http.client.request(
            type_=dict,
            method="PUT",
            url="/collections/{collection_name}/vectors/{vector_name}",
            path_params={"collection_name": collection, "vector_name": VECTOR_NAME},
            params={"wait": "true"},
            json={"sparse": {}},
        )
        updated_config = client.get_collection(collection).config.params.sparse_vectors or {}
        if VECTOR_NAME not in updated_config:
            raise RuntimeError("Qdrant did not add the 'splade' sparse vector configuration.")

    written = 0
    for points in batches(client, collection, batch_size):
        candidates = [p for p in points if overwrite or not has_splade(p)]
        if not candidates:
            continue
        texts = [document_text(p, text_field) for p in candidates]
        embeddings = list(model.embed(texts, batch_size=batch_size))
        if len(embeddings) != len(candidates):
            raise RuntimeError("Embedding model returned the wrong number of vectors.")
        updates = []
        for point, embedding in zip(candidates, embeddings):
            if len(embedding.indices) == 0:
                raise ValueError(f"Point {point.id!r} produced an empty SPLADE embedding.")
            updates.append(models.PointVectors(
                id=point.id,
                vector={VECTOR_NAME: models.SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist(),
                )},
            ))
        # Updating only the named vector preserves BM25, dense vectors and payload.
        client.update_vectors(collection_name=collection, points=updates, wait=True)
        written += len(updates)
        LOGGER.info("Wrote %d SPLADE vectors so far", written)

    missing = []
    for points in batches(client, collection, batch_size):
        missing.extend(p.id for p in points if not has_splade(p))
    if missing:
        raise RuntimeError(f"Verification failed: {len(missing)} points lack SPLADE; first IDs: {missing[:10]}")
    LOGGER.info("Verified all points have SPLADE. Updated %d points.", written)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-field", help="Payload text field; default: document, then text")
    parser.add_argument("--apply", action="store_true", help="Write vectors; default is inspection only")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing SPLADE vectors")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    host = os.getenv("QDRANT_HOST")
    if not host:
        parser.error("QDRANT_HOST must be set in the environment or .env")
    client = QdrantClient(url=host, api_key=os.getenv("QDRANT_API_KEY"),
                          timeout=100, check_compatibility=False)
    try:
        backfill(client, collection=args.collection, batch_size=args.batch_size,
                 text_field=args.text_field, apply=args.apply, overwrite=args.overwrite)
    finally:
        client.close()


if __name__ == "__main__":
    main()
