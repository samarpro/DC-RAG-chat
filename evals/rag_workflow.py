"""Streamlit-free, inspectable version of the RAG workflow from ``main.py``.

This module exposes the processing stages and a structured result for future
evaluation code. It does not load a dataset, score results, or run benchmarks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import models
from voyageai.client import Client as VoyageClient


@dataclass(frozen=True)
class RAGResult:
    """Outputs from one workflow run, with intermediate stages preserved."""

    query: str
    expanded_query: str
    retrieved_documents: list[str]
    link_metadata: dict[str, str]
    answer: str


class RAGWorkflow:
    """Expand a query, retrieve hybrid-search context, then generate an answer."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        voyage_client: Any | None = None,
        qdrant_client: Any | None = None,
        sparse_embedding_model: Any | None = None,
    ) -> None:
        load_dotenv()
        self.llm = llm if llm is not None else genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.voyage_client = (
            voyage_client
            if voyage_client is not None
            else VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))
        )
        self.qdrant_client = (
            qdrant_client
            if qdrant_client is not None
            else QdrantClient(
                url=os.getenv("QDRANT_HOST"),
                api_key=os.getenv("QDRANT_API_KEY"),
                https=True,
                timeout=100,
                check_compatibility=False,
            )
        )
        self.sparse_embedding_model = (
            sparse_embedding_model
            if sparse_embedding_model is not None
            else SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        )

    @staticmethod
    def convert_links_to_markdown(text: str, link_dict: dict[str, str]) -> str:
        def replace_link(match: re.Match[str]) -> str:
            link_text = " ".join(match.group(1).split())
            return f"[{link_text}]({link_dict.get(link_text, '#')})"

        return re.sub(r"<link>\s*(.*?)\s*</link>", replace_link, text)

    def expand_query(self, query: str) -> str:
        prompt = f"""
SYSTEM:
Rewrite the student's query to make it explicit, content-rich, and search-optimized for academic information retrieval.

INSTRUCTIONS:
- Expand vague references (“this,” “it,” “that”) into concrete subjects.
- Include specific academic or institutional terms implied by the query.
- Use precise, information-dense wording that would match both keywords and concepts.
- Keep it natural and concise—one clear standalone question or statement.
- Avoid conversational fillers or polite phrases.
- Output only the rewritten query.

QUERY:
{query}
"""
        response = self.llm.models.generate_content(
            model="gemini-flash-latest", contents=[prompt]
        )
        return response.text or ""

    def retrieve_documents(
        self, query: str, *, limit: int = 10
    ) -> tuple[list[str], dict[str, str]]:
        sparse = list(self.sparse_embedding_model.embed(query))[0]
        dense = self.voyage_client.embed(
            model="voyage-3", texts=[query], input_type="query"
        )
        prefetch = [
            models.Prefetch(
                query=[float(value) for value in dense.embeddings[0]],
                limit=25,
                using="voyage3",
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse.indices.tolist(), values=sparse.values.tolist()
                ),
                using="splade",
                limit=10,
            ),
        ]
        result = self.qdrant_client.query_points(
            "hybrid-search-splade",
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )

        documents: list[str] = []
        link_metadata: dict[str, str] = {}
        for point in result.points:
            payload = point.payload or {}
            document = payload.get("text")
            if document:
                documents.append(document)
            link_metadata.update(payload.get("url_dict") or {})
        return documents, link_metadata

    def generate_response(self, query: str, documents: Sequence[str]) -> str:
        context = "\n\n".join(documents)
        prompt = f"""
SYSTEM:
You are an academic support assistant for international undergraduate students.
Answer ONLY using the given CONTEXT. Never guess or add outside info. Preserve all <link></link> tags.

GUIDELINES:
1. Read the context and use only relevant information.
2. Greeting → Respond politely and say you're ready to help with academic questions.
3. Academic query → Use context info only.
4. If information is missing → Say “Insufficient context” and mention what's unclear.
5. Use bullets or numbered steps for clarity when useful.
6. Reference relevant <link></link> resources when possible and suggest a relevant support area.

CONTEXT:
{context}

QUERY:
{query}
"""
        response = self.llm.models.generate_content(
            model="gemini-flash-latest", contents=[prompt]
        )
        return response.text or ""

    def answer(self, query: str) -> RAGResult:
        """Run the complete RAG workflow without UI, cache, or persistence effects."""
        expanded_query = self.expand_query(query)
        documents, link_metadata = self.retrieve_documents(expanded_query)
        raw_answer = self.generate_response(query, documents)
        return RAGResult(
            query=query,
            expanded_query=expanded_query,
            retrieved_documents=documents,
            link_metadata=link_metadata,
            answer=self.convert_links_to_markdown(raw_answer, link_metadata),
        )


def create_instance() -> RAGWorkflow:
    """Create a configured workflow instance using environment credentials."""
    return RAGWorkflow()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question to send through the RAG workflow")
    args = parser.parse_args()
    print(RAGWorkflow().answer(args.query).answer)
