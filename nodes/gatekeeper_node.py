from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
import json

from utils.dsrp_state import DSRPState

from utils.re_ranker import rerank_openrouter
from utils.llm_output_schemas.gatekeeper_node_schemas import (
    GatekeeperAuditorSchema,
    GatekeeperClassifierSchema,
    GatekeeperEvidenceSchema,
)


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


def _ensure_evidence_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    evidence = payload.get("candidate_evidence")
    if not isinstance(evidence, list):
        evidence = []
    payload["candidate_evidence"] = evidence
    return payload


def _ensure_classifier_payload(payload, paper_id=""):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("paper_id", paper_id)
    payload.setdefault("classification", "Borderline")
    payload.setdefault("inclusion_basis", "")
    payload.setdefault("identified_modelling_paradigm", "")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    payload.setdefault("borderline_explanation_if_applicable", "")
    return payload


def _ensure_audit_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("final_classification", "Borderline")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("validated_reasoning", "")
    payload.setdefault("validated_bibliography", [])
    payload.setdefault("audit_commentary", "")
    return payload


def _canonical_gatekeeper_label(value: str | None) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text == "":
        return None

    if "include" in text or text in {"yes", "eligible", "in-scope", "inscope"}:
        return "Include"
    if "exclude" in text or text in {"no", "out-of-scope", "out of scope"}:
        return "Exclude"
    if "border" in text or text in {"uncertain", "mixed"}:
        return "Borderline"

    return None


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

def gatekeeper_node(state: DSRPState) -> dict:
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")
    llm = set_llm(model=llm_model, reasoning=llm_reasoning)
    
    # 1️⃣ Load vector query
    vector_config = load_vector_query(
        "prompts/ds_gatekeeper/vector_query.yaml"
    )

    retriever_tool = PaperRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    ).for_paper(
        state["paper_id"],
        k=vector_config["k"]
    )

    #docs = retriever_tool.invoke(vector_config["query"])
    #print(f'Docs: {len(docs)} retrieved')
    
    ## -- Re-ranking --
    raw_docs = retriever_tool.invoke(vector_config["query"])
    #print(f'Raw Docs: {len(raw_docs)} retrieved')
    reranker_top_k = _resolve_reranker_top_k(state)
    reranker_model = _resolve_reranker_model(state)
    docs = rerank_openrouter(vector_config["query"], raw_docs, top_k=reranker_top_k, model=reranker_model)
    #print(f'Re-ranked Docs: {len(docs)} retrieved')

    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # 2️⃣ Evidence extraction
    retriever_prompt = load_yaml_prompt(
        "prompts/ds_gatekeeper/retriever.yaml"
    )

    evidence_llm = llm.with_structured_output(GatekeeperEvidenceSchema)
    evidence_response = evidence_llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = _ensure_evidence_payload(_structured_output_to_dict(evidence_response))

    # 3️⃣ Classification
    classifier_prompt = load_yaml_prompt(
        "prompts/ds_gatekeeper/classifier.yaml"
    )

    classifier_llm = llm.with_structured_output(GatekeeperClassifierSchema)
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = _ensure_classifier_payload(
        _structured_output_to_dict(classification_response),
        paper_id=state["paper_id"],
    )

    # 4️⃣ Audit
    auditor_prompt = load_yaml_prompt(
        "prompts/ds_gatekeeper/auditor.yaml"
    )

    auditor_llm = llm.with_structured_output(GatekeeperAuditorSchema)
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = _ensure_audit_payload(_structured_output_to_dict(audit_response))

    # Enforce a strict final label domain so graph routing remains reliable.
    final_label = _canonical_gatekeeper_label(audit_json.get("final_classification"))
    if final_label is None:
        # Fall back to the pre-audit classifier label if auditor output is malformed.
        final_label = _canonical_gatekeeper_label(classification_json.get("classification"))

    if final_label is None:
        # Safe default: uncertain outputs should not automatically fan-out downstream.
        final_label = "Borderline"

    audit_json["final_classification"] = final_label

    return {
        "gatekeeper": audit_json
    }
