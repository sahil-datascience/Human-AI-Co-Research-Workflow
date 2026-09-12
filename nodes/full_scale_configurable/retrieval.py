from __future__ import annotations

from typing import Any

from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.re_ranker import rerank_openrouter

from .config import get_reranker_model, get_reranker_top_k, get_retrieval_k


def retrieve_and_rerank(
    state: dict[str, Any],
    node_name: str,
    vector_query_path: str,
    *config_parts: str,
    default_reranker_top_k: int = 8,
) -> list:
    vector_config = load_vector_query(vector_query_path)
    retrieval_k = get_retrieval_k(state, node_name, vector_config["k"], *config_parts, "retrieval")

    retriever_tool = PaperRetriever(
        collection_name=state["collection_name"],
        persist_directory=state["persist_directory"],
        embedding_model=state["embedding_model"],
    ).for_paper(state["paper_id"], k=retrieval_k)

    raw_docs = retriever_tool.invoke(vector_config["query"])
    if not raw_docs:
        return []

    reranker_top_k = get_reranker_top_k(
        state,
        node_name,
        default_reranker_top_k,
        *config_parts,
        "reranker",
    )
    reranker_model = get_reranker_model(
        state,
        node_name,
        "cohere/rerank-4-fast",
        *config_parts,
        "reranker",
    )

    return rerank_openrouter(
        vector_config["query"],
        raw_docs,
        top_k=reranker_top_k,
        model=reranker_model,
    )


def format_docs_context(docs: list) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

