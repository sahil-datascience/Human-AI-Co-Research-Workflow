from typing import List, Tuple, Optional
import os
import requests


def _build_doc_text(d) -> str:
    """Render a LangChain-like Document object to a plain text string including useful metadata."""
    heading = d.metadata.get("section_heading") if hasattr(d, "metadata") else None
    page = d.metadata.get("page_no") if hasattr(d, "metadata") else None
    heading = heading or ""
    page = page or ""
    return f"[Page {page} | {heading}]\n{d.page_content}"


def rerank_openrouter(
    query: str,
    docs: List,
    top_k: int = 8,
    api_key: Optional[str] = None,
    model: str = "cohere/rerank-4-pro",
    referer: Optional[str] = None,
    site_title: Optional[str] = None,
    timeout: int = 30,
) -> List:
    """Re-rank `docs` using OpenRouter's rerank endpoint (Cohere reranker).

    - `query`: the textual query to rank documents against.
    - `docs`: a sequence of objects with `page_content` and `metadata` attributes
      (e.g., LangChain Document instances).
    - returns: ordered list of the top_k docs (subset of the input `docs`).

    Requires `OPENROUTER_API_KEY` env var or explicit `api_key` argument.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (pass api_key or set env var)")

    endpoint = "https://openrouter.ai/api/v1/rerank"

    documents = [_build_doc_text(d) for d in docs]

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if site_title:
        headers["X-OpenRouter-Title"] = site_title

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter rerank failed: {resp.status_code} {resp.text}")

    data = resp.json()
    results = data.get("results") or []

    # The API usually returns objects with an `index` pointing to the original document
    ordered_indices = []
    for r in results:
        idx = r.get("index") if isinstance(r, dict) else None
        if idx is None:
            # try alternate keys sometimes used by wrappers
            idx = r.get("document_index") if isinstance(r, dict) else None
        if idx is None:
            continue
        ordered_indices.append(int(idx))

    if not ordered_indices:
        # fallback: just return the first top_k docs
        return docs[: payload["top_n"]]

    ordered = [docs[i] for i in ordered_indices if 0 <= i < len(docs)]
    return ordered[: payload["top_n"]]


def rerank_openrouter_with_scores(
    query: str,
    docs: List,
    top_k: int = 8,
    api_key: Optional[str] = None,
    model: str = "cohere/rerank-4-pro",
    referer: Optional[str] = None,
    site_title: Optional[str] = None,
    timeout: int = 30,
) -> List[Tuple[object, float]]:
    """Like `rerank_openrouter` but returns (doc, score) pairs ordered by score."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (pass api_key or set env var)")

    endpoint = "https://openrouter.ai/api/v1/rerank"
    documents = [_build_doc_text(d) for d in docs]

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if site_title:
        headers["X-OpenRouter-Title"] = site_title

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter rerank failed: {resp.status_code} {resp.text}")

    data = resp.json()
    results = data.get("results") or []

    output: List[Tuple[object, float]] = []
    for r in results:
        idx = r.get("index") if isinstance(r, dict) else None
        if idx is None:
            idx = r.get("document_index") if isinstance(r, dict) else None
        if idx is None:
            continue
        score = r.get("relevance_score") or r.get("score") or 0.0
        try:
            doc = docs[int(idx)]
        except Exception:
            continue
        output.append((doc, float(score)))

    return output
