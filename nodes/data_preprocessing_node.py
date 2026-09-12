

from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
import json
from utils.dsrp_state import DSRPState
from utils.llm_output_schemas.data_preprocessing_node_schemas import (
    CandidateEvidenceSchema,
    DataPreprocessingAuditorSchema,
    DataPreprocessingClassifierSchema,
)


def _structured_output_to_dict(output):
    """Convert Pydantic structured output to a plain dict across Pydantic versions."""
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output


def _ensure_audit_defaults(audit_json):
    """Keep downstream notebook extraction stable when the provider omits optional fields."""
    if not isinstance(audit_json, dict):
        return audit_json

    for key in (
        "validated_data_cleaning",
        "validated_data_reduction",
        "validated_data_transformation",
    ):
        value = audit_json.get(key)
        if not isinstance(value, dict):
            value = {}
        value.setdefault("status", "Not Reported")
        value.setdefault("justification", "")
        audit_json[key] = value

    audit_json.setdefault("confidence", 0.0)
    audit_json.setdefault("validated_reasoning", "")
    audit_json.setdefault("validated_bibliography", [])
    audit_json.setdefault("audit_commentary", "")
    return audit_json

def data_preprocessing_node(state: DSRPState):

    dimension_name = "data_preprocessing"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")

    # Load vector query
    vector_config = load_vector_query(
        f"prompts/dsrp/{dimension_name}/vector_query.yaml"
    )
    retrieval_k = state.get("reranker_top_k", vector_config["k"])

    retriever_tool = PaperRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    ).for_paper(
        state["paper_id"],
        k=retrieval_k
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
    base_llm = set_llm(model=llm_model, reasoning=llm_reasoning)

    evidence_llm = base_llm.with_structured_output(CandidateEvidenceSchema)
    evidence_response = evidence_llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = _structured_output_to_dict(evidence_response)

    # Classification
    classifier_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/classifier.yaml"
    )

    classifier_llm = base_llm.with_structured_output(DataPreprocessingClassifierSchema)
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = _structured_output_to_dict(classification_response)

    # Audit
    auditor_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/auditor.yaml"
    )

    auditor_llm = base_llm.with_structured_output(DataPreprocessingAuditorSchema)
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = _ensure_audit_defaults(_structured_output_to_dict(audit_response))

    # Store result
    state["dsrp_outputs"][dimension_name] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}
