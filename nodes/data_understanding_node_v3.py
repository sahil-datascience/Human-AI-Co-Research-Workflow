
from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
from utils.parse_llm_json import parse_llm_json
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.dsrp_state import DSRPState

from llm_output_schemas.data_understanding_node_schemas import (
    Primary_labels_Schema,
    AuditorSchema,
)


def _run_binary_classifier(llm, prompt_path: str, label_name: str, context_text: str, reranker=None):
    """
    Run a single binary classifier for a specific label.
    
    Args:
        llm: Language model instance
        prompt_path: Path to the binary classifier prompt YAML
        label_name: Name of the label being classified
        context_text: Context from initial vector search
        reranker: Optional Cohere re-ranker for evidence ranking
    
    Returns:
        Tuple of (label_name, classification_result_json)
    """
    classifier_prompt = load_yaml_prompt(prompt_path)
    
    # Format the prompt with context
    # Note: If re-ranker is available, it can be integrated here to rank evidence
    # For now, passing full context; re-ranker logic should be in prompt or as preprocessing
    classification_response = llm.invoke(
        classifier_prompt.format_messages(input=context_text)
    )
    #classification_json = parse_llm_json(classification_response.content)
    classification_json = classification_response
    
    return label_name, classification_json


def _run_binary_classifiers_parallel(llm, jobs, context_text: str, max_workers: int, reranker=None):
    """
    Run all binary classifiers in parallel for maximum efficiency.
    
    Args:
        llm: Language model instance
        jobs: List of (label_name, prompt_path) tuples
        context_text: Context from initial vector search
        max_workers: Maximum number of parallel workers
        reranker: Optional Cohere re-ranker
    
    Returns:
        Dictionary mapping label_name -> classification_result_json
    """
    results_by_label = {}
    
    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_label = {
            executor.submit(_run_binary_classifier, llm, prompt_path, label_name, context_text, reranker): label_name
            for label_name, prompt_path in jobs
        }
        
        for future in as_completed(future_to_label):
            label_name = future_to_label[future]
            try:
                _, classification_json = future.result()
            except Exception as exc:
                raise RuntimeError(f"Binary classifier failed for '{label_name}': {exc}") from exc
            results_by_label[label_name] = classification_json
    
    return results_by_label



#-------------------------------------- Main Node Logic --------------------------------------#

def data_understanding_node_v3(state: DSRPState):
    """
    Binary classification node for DSRP Data Understanding dimension.
    
    Architecture:
    1. Initial vector search (k=25)
    2. Run all binary classifiers in parallel for each label
    3. Each classifier performs combined retrieval + classification with re-ranker
    4. Aggregate results and validate with auditor
    5. Return unified output in same format as v2
    """
    
    dimension_name = "data_understanding"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    parallel_workers = max(1, int(os.getenv("DSRP_PARALLEL_WORKERS", "4")))
    
    # STEP 1: Load vector query and perform initial retrieval
    vector_config = load_vector_query(
        f"prompts/dsrp/{dimension_name}/v3_binary/vector_query.yaml"
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

    ## --------------- Re-ranking Logic ---------------
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
    #print(f'Raw Docs: {len(raw_docs)} retrieved')
    reranker_top_k = _resolve_reranker_top_k(state)
    reranker_model = _resolve_reranker_model(state)
    docs = rerank_openrouter(vector_config["query"], raw_docs, top_k=reranker_top_k, model=reranker_model)
    #print(f'Re-ranked Docs: {len(docs)} retrieved')
    
    # Format context from initial retrieval
    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )
    
    # Set up LLM
    base_llm = set_llm(model=llm_model)
    
    # Enforce the structured output schema for all binary classifiers
    llm = base_llm.with_structured_output(Primary_labels_Schema)
    #llm = base_llm

    
    # STEP 2: Prepare binary classification jobs for parallel execution
    # Each label gets its own binary classifier prompt
    
    # Data Category: UGC, Device, Transaction, Survey/Statistical
    category_jobs = [
        ("UGC", f"prompts/dsrp/{dimension_name}/v3_binary/category/ugc.yaml"),
        ("Device", f"prompts/dsrp/{dimension_name}/v3_binary/category/device.yaml"),
        ("Transaction", f"prompts/dsrp/{dimension_name}/v3_binary/category/transaction.yaml"),
        ("Survey/Statistical", f"prompts/dsrp/{dimension_name}/v3_binary/category/survey_statistical.yaml"),
    ]
    
    # Data Format: Structured, Unstructured, Semi-Structured
    format_jobs = [
        ("Structured", f"prompts/dsrp/{dimension_name}/v3_binary/format/structured.yaml"),
        ("Unstructured", f"prompts/dsrp/{dimension_name}/v3_binary/format/unstructured.yaml"),
        ("Semi-Structured", f"prompts/dsrp/{dimension_name}/v3_binary/format/semi_structured.yaml"),
    ]
    
    # Data Characteristics: Temporal, Spatial, Textual, Visual, Networked
    characteristics_jobs = [
        ("Temporal", f"prompts/dsrp/{dimension_name}/v3_binary/characteristics/temporal.yaml"),
        ("Spatial", f"prompts/dsrp/{dimension_name}/v3_binary/characteristics/spatial.yaml"),
        ("Textual", f"prompts/dsrp/{dimension_name}/v3_binary/characteristics/textual.yaml"),
        ("Visual", f"prompts/dsrp/{dimension_name}/v3_binary/characteristics/visual.yaml"),
        ("Networked", f"prompts/dsrp/{dimension_name}/v3_binary/characteristics/networked.yaml"),
    ]
    
    # STEP 3: Run all binary classifiers in parallel
    #print("Running binary classifiers in parallel...") # remove

    # vector search for category

    category_results = _run_binary_classifiers_parallel(
        llm=llm,
        jobs=category_jobs,
        context_text=context_text,
        max_workers=parallel_workers,
    )
    
    format_results = _run_binary_classifiers_parallel(
        llm=llm,
        jobs=format_jobs,
        context_text=context_text,
        max_workers=parallel_workers,
    )
    
    characteristics_results = _run_binary_classifiers_parallel(
        llm=llm,
        jobs=characteristics_jobs,
        context_text=context_text,
        max_workers=parallel_workers,
    )

    # print raw results for debugging
    #print("Category Results:", category_results)
    #print("Format Results:", format_results)  
    #print("Characteristics Results:", characteristics_results)
        
    # STEP 4: Aggregate results from binary classifiers
    # Select labels where is_present=true and confidence > 0
    
    selected_categories = []
    category_bibliography = []
    category_confidence = 0.0
    
    for label, result in category_results.items():
        
        # 1. Access the Pydantic property directly using dot notation
        if result.label:  
            selected_categories.append(label)
            
            # 2. Extract bibliography items and convert them back to dictionaries
            for item in result.bibliography:
                category_bibliography.append(item.model_dump())
                
            # 3. Access confidence score and calculate the maximum
            category_confidence = max(category_confidence, float(result.confidence or 0.0))

    selected_formats = []
    format_bibliography = []
    format_confidence = 0.0
    
    for label, result in format_results.items():
        if result.label:
            selected_formats.append(label)
            for item in result.bibliography:
                format_bibliography.append(item.model_dump())
            format_confidence = max(format_confidence, float(result.confidence or 0.0))
    
    selected_characteristics = []
    characteristics_bibliography = []
    characteristics_confidence = 0.0
    
    for label, result in characteristics_results.items():
        if result.label:
            selected_characteristics.append(label)
            for item in result.bibliography:
                characteristics_bibliography.append(item.model_dump())
            characteristics_confidence = max(characteristics_confidence, float(result.confidence or 0.0))
    
    # Combine all evidence and results
    all_evidence = {
        "category_results": category_results,
        "format_results": format_results,
        "characteristics_results": characteristics_results,
    }
    
    combined_classification = {
        "data_category": selected_categories,
        "data_format": selected_formats,
        "data_characteristics": selected_characteristics,
        "confidence": max(category_confidence, format_confidence, characteristics_confidence),
        "reasoning_explanation": "Aggregated from binary classification of all labels.",
        "evidence": all_evidence,
    }
    
    # STEP 5: Unified auditor validation
    #auditor_prompt = load_yaml_prompt(f"prompts/dsrp/{dimension_name}/v3_binary/auditor.yaml")
    
    
    # Enforce structured output schema for auditor
    #llm = base_llm.with_structured_output(AuditorSchema)

    
    #audit_response = llm.invoke(auditor_prompt.format_messages(input=combined_classification))
    
    #audit_json = parse_llm_json(audit_response.content)
    #audit_json = {label: result for label, result in audit_response}

    #use direct results
    audit_json = combined_classification
    
    state["dsrp_outputs"][f"{dimension_name}"] = audit_json
    
    return {"dsrp_outputs": state["dsrp_outputs"]}


def data_understanding_node(state: DSRPState):
    return data_understanding_node_v3(state)
