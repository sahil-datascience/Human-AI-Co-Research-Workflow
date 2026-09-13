

from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
import json
from utils.dsrp_state import DSRPState
from llm_output_schemas.research_question_node_schemas import (
    ResearchQuestionAuditorSchema,
    ResearchQuestionClassifierSchema,
    ResearchQuestionEvidenceSchema,
    normalize_research_question_label,
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


def _ensure_classifier_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("classification", "Exploratory")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    return payload


def _ensure_audit_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("final_classification", "Exploratory")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("validated_reasoning", "")
    payload.setdefault("validated_bibliography", [])
    payload.setdefault("audit_commentary", "")
    return payload

def research_question_node(state: DSRPState):

    dimension_name = "research_question"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")

    # Load vector query
    vector_config = load_vector_query(
        f"prompts/dsrp/{dimension_name}/vector_query.yaml"
    )

    retriever_tool = PaperRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    ).for_paper(
        state["paper_id"],
        k=state.get("reranker_top_k", vector_config["k"])
    )

    docs = retriever_tool.invoke(vector_config["query"])

    # Format context
    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # Evidence extraction
    retriever_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/retriever.yaml"
    )

    # Set up LLM
    llm = set_llm(model=llm_model, reasoning=llm_reasoning)

    evidence_llm = llm.with_structured_output(ResearchQuestionEvidenceSchema)
    evidence_response = evidence_llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = _ensure_evidence_payload(_structured_output_to_dict(evidence_response))

    # Classification
    classifier_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/classifier.yaml"
    )

    classifier_llm = llm.with_structured_output(ResearchQuestionClassifierSchema)
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = _ensure_classifier_payload(
        _structured_output_to_dict(classification_response)
    )

    # Audit
    auditor_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/auditor.yaml"
    )

    auditor_llm = llm.with_structured_output(ResearchQuestionAuditorSchema)
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = _ensure_audit_payload(_structured_output_to_dict(audit_response))

    final_label = normalize_research_question_label(audit_json.get("final_classification"))
    if final_label is None:
        final_label = normalize_research_question_label(classification_json.get("classification"))
    audit_json["final_classification"] = final_label or "Exploratory"

    # Store result
    state["dsrp_outputs"][dimension_name] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}
