from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.parse_llm_json import parse_llm_json


def build_context_text(state, vector_query_path: str) -> str:
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]

    vector_config = load_vector_query(vector_query_path)

    retriever_tool = PaperRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
    ).for_paper(
        state["paper_id"],
        k=vector_config["k"],
    )

    #docs = retriever_tool.invoke(vector_config["query"])

    ## ----------- Start of Re-ranking ---------------
    
    from utils.re_ranker import rerank_openrouter
    def _resolve_reranker_top_k(state: DSRPState, default: int = 10) -> int:
        value = state.get("reranker_top_k", default)
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            return default

        return top_k if top_k > 0 else default


    def _resolve_reranker_model(state: DSRPState, default: str = "cohere/rerank-4-pro") -> str:
        """Extract reranker model from state with validation and fallback to default."""
        value = state.get("reranker_model", default)
        if not isinstance(value, str) or not value.strip():
            return default
        return value.strip()

    raw_docs = retriever_tool.invoke(vector_config["query"])
    print(f'Raw Docs: {len(raw_docs)} retrieved')
    reranker_top_k = _resolve_reranker_top_k(state)
    reranker_model = _resolve_reranker_model(state)
    docs = rerank_openrouter(vector_config["query"], raw_docs, top_k=reranker_top_k, model=reranker_model)
    print(f'Re-ranked Docs: {len(docs)} retrieved')
    #---------- End of re-ranking ----------


    return "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )


def invoke_prompt_with_text(llm, prompt_path: str, context_text: str):
    prompt = load_yaml_prompt(prompt_path)
    response = llm.invoke(prompt.format_messages(input=context_text))
    return parse_llm_json(response.content)


def invoke_prompt_with_json(llm, prompt_path: str, payload) -> dict:
    prompt = load_yaml_prompt(prompt_path)
    response = llm.invoke(prompt.format_messages(input=json.dumps(payload)))
    return parse_llm_json(response.content)


def run_prompt_pair(llm, retriever_path: str, classifier_path: str, context_text: str):
    evidence_json = invoke_prompt_with_text(llm, retriever_path, context_text)
    classification_json = invoke_prompt_with_json(llm, classifier_path, evidence_json)
    return evidence_json, classification_json


def run_classifier_only(llm, classifier_path: str, context_text: str):
    return invoke_prompt_with_text(llm, classifier_path, context_text)


def run_prompt_pairs_parallel(llm, jobs, context_text: str, max_workers: int):
    evidence_by_key = {}
    classification_by_key = {}

    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_key = {
            executor.submit(run_prompt_pair, llm, retriever_path, classifier_path, context_text): key
            for key, retriever_path, classifier_path in jobs
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                evidence_json, classification_json = future.result()
            except Exception as exc:
                raise RuntimeError(f"Parallel prompt pair failed for '{key}': {exc}") from exc
            evidence_by_key[key] = evidence_json
            classification_by_key[key] = classification_json

    return evidence_by_key, classification_by_key


def run_classifiers_parallel(llm, jobs, context_text: str, max_workers: int):
    classification_by_key = {}

    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_key = {
            executor.submit(run_classifier_only, llm, classifier_path, context_text): key
            for key, classifier_path in jobs
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                classification_json = future.result()
            except Exception as exc:
                raise RuntimeError(f"Parallel classifier failed for '{key}': {exc}") from exc
            classification_by_key[key] = classification_json

    return classification_by_key
