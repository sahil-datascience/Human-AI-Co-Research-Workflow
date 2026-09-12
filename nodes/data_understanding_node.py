

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


def _run_prompt_pair(llm, retriever_path: str, classifier_path: str, context_text: str):
    retriever_prompt = load_yaml_prompt(retriever_path)
    retrieval_response = llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )
    evidence_json = parse_llm_json(retrieval_response.content)

    classifier_prompt = load_yaml_prompt(classifier_path)
    classification_response = llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = parse_llm_json(classification_response.content)

    return evidence_json, classification_json


def _run_classifier_only(llm, classifier_path: str, context_text: str):
    classifier_prompt = load_yaml_prompt(classifier_path)
    classification_response = llm.invoke(
        classifier_prompt.format_messages(input=context_text)
    )
    return parse_llm_json(classification_response.content)


def _run_prompt_pairs_parallel(llm, jobs, context_text: str, max_workers: int):
    evidence_by_key = {}
    classification_by_key = {}

    # Keep this bounded to reduce rate-limit pressure.
    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_key = {
            executor.submit(_run_prompt_pair, llm, retriever_path, classifier_path, context_text): key
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


def _run_classifiers_parallel(llm, jobs, context_text: str, max_workers: int):
    classification_by_key = {}

    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_key = {
            executor.submit(_run_classifier_only, llm, classifier_path, context_text): key
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


def data_understanding_node(state: DSRPState):

    dimension_name = "data_understanding"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")
    parallel_workers = max(1, int(os.getenv("DSRP_PARALLEL_WORKERS", "4")))

    # STEP 1: Load vector query (once)
    vector_config = load_vector_query(
        f"prompts/dsrp/{dimension_name}/vector_query.yaml"
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

    ## -- Re-ranking --
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


    # Format context
    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # Set up LLM
    llm = set_llm(model=llm_model, reasoning=llm_reasoning)

    # STEP 2: Evidence extraction (optional legacy prompt).
    # New prompt layout may only contain dimension-specific retrievers.
    evidence_json = {}
    base_retriever_path = Path(f"prompts/dsrp/{dimension_name}/retriever.yaml")
    if base_retriever_path.exists():
        retriever_prompt = load_yaml_prompt(str(base_retriever_path))
        evidence_response = llm.invoke(
            retriever_prompt.format_messages(input=context_text)
        )
        evidence_json = parse_llm_json(evidence_response.content)

    # STEP 3: Run specialized classifier pipelines.
    dimension_configs = [
        ("category", "data_category"),
        ("format", "data_format"),
    ]

    classification_outputs = {}
    per_dimension_evidence = {}

    base_jobs = []
    for dimension_folder, output_key in dimension_configs:
        retriever_path = f"prompts/dsrp/{dimension_name}/{dimension_folder}/retriever.yaml"
        classifier_path = f"prompts/dsrp/{dimension_name}/{dimension_folder}/classifier.yaml"
        base_jobs.append((output_key, retriever_path, classifier_path))

    base_evidence, base_classifications = _run_prompt_pairs_parallel(
        llm=llm,
        jobs=base_jobs,
        context_text=context_text,
        max_workers=parallel_workers,
    )
    per_dimension_evidence.update(base_evidence)
    classification_outputs.update(base_classifications)

    # STEP 3b: Data-characteristics split pipeline (6 prompt folders)
    characteristics_base = Path(f"prompts/dsrp/{dimension_name}/characteristics")
    characteristics_label_units = [
        ("temporal", "Temporal"),
        ("spatial", "Spatial"),
        ("textual", "Textual"),
        ("visual", "Visual"),
        ("networked", "Networked"),
    ]
    split_prompt_available = all(
        (characteristics_base / folder / "classifier.yaml").exists()
        for folder, _ in characteristics_label_units
    )

    if split_prompt_available:
        characteristic_jobs = []
        for folder, label_name in characteristics_label_units:
            classifier_path = str(characteristics_base / folder / "classifier.yaml")
            characteristic_jobs.append((label_name, classifier_path))

        characteristics_by_label = _run_classifiers_parallel(
            llm=llm,
            jobs=characteristic_jobs,
            context_text=context_text,
            max_workers=parallel_workers,
        )

        # Optional folder-level auditor for split data_characteristics outputs.
        characteristics_auditor_path = characteristics_base / "auditor.yaml"
        if characteristics_auditor_path.exists():
            characteristics_auditor_prompt = load_yaml_prompt(str(characteristics_auditor_path))
            characteristics_auditor_input = {
                "label_outputs": characteristics_by_label,
                "labels": [label_name for _, label_name in characteristics_label_units],
            }
            characteristics_audit_response = llm.invoke(
                characteristics_auditor_prompt.format_messages(
                    input=json.dumps(characteristics_auditor_input)
                )
            )
            characteristics_audit_json = parse_llm_json(characteristics_audit_response.content)

            audited_labels = characteristics_audit_json.get("data_characteristics", [])
            audited_confidence = float(characteristics_audit_json.get("confidence", 0.0) or 0.0)
            audited_reasoning = characteristics_audit_json.get("validated_reasoning", "")
            audited_bibliography = characteristics_audit_json.get("validated_bibliography", [])
        else:
            audited_labels = None
            audited_confidence = 0.0
            audited_reasoning = ""
            audited_bibliography = []

        selected_labels = []
        max_conf = 0.0
        bibliography = []
        for _, label_name in characteristics_label_units:
            out = characteristics_by_label.get(label_name, {})
            label_biblio = out.get("bibliography", [])
            label_present = bool(out.get("is_present", False))
            label_supported = bool(label_biblio)

            if label_present and label_supported:
                selected_labels.append(label_name)
                bibliography.extend(label_biblio)

            max_conf = max(max_conf, float(out.get("confidence", 0.0) or 0.0))

        if isinstance(audited_labels, list):
            selected_labels = [
                label for label in audited_labels
                if label in {label_name for _, label_name in characteristics_label_units}
            ]

            bibliography = audited_bibliography if isinstance(audited_bibliography, list) else bibliography
            max_conf = max(max_conf, audited_confidence)
            reasoning_explanation = (
                audited_reasoning
                if isinstance(audited_reasoning, str) and audited_reasoning.strip()
                else "Validated by characteristics-level auditor and aggregated from split label-specific classifiers."
            )
        else:
            reasoning_explanation = "Strictly aggregated from split label-specific classifiers with explicit-evidence gating."

        characteristics_classification = {
            "data_characteristics": selected_labels,
            "confidence": max_conf,
            "reasoning_explanation": reasoning_explanation,
            "bibliography": bibliography,
        }

        classification_outputs["data_characteristics"] = characteristics_classification
        per_dimension_evidence["data_characteristics"] = {
            "by_label": {
                label_name: characteristics_by_label.get(label_name, {}).get("bibliography", [])
                for _, label_name in characteristics_label_units
            },
            "per_label_classifiers": characteristics_by_label,  # Store full label outputs for debugging
        }
    else:
        # Backward-compatible fallback to the legacy single characteristics prompts.
        legacy_retriever_path = f"prompts/dsrp/{dimension_name}/characteristics/retriever.yaml"
        legacy_classifier_path = f"prompts/dsrp/{dimension_name}/characteristics/classifier.yaml"
        characteristics_evidence_json, characteristics_classification_json = _run_prompt_pair(
            llm=llm,
            retriever_path=legacy_retriever_path,
            classifier_path=legacy_classifier_path,
            context_text=context_text,
        )
        per_dimension_evidence["data_characteristics"] = characteristics_evidence_json
        classification_outputs["data_characteristics"] = characteristics_classification_json

    if not evidence_json:
        evidence_json = {
            "data_category": per_dimension_evidence.get("data_category", {}),
            "data_format": per_dimension_evidence.get("data_format", {}),
            "data_characteristics": per_dimension_evidence.get("data_characteristics", {}),
        }

    # STEP 4: Combine all three classifier outputs
    combined_classification = {
        "data_category": classification_outputs.get("data_category", {}).get("data_category", []),
        "data_format": classification_outputs.get("data_format", {}).get("data_format", []),
        "data_characteristics": classification_outputs.get("data_characteristics", {}).get("data_characteristics", []),
        "confidence": max(
            classification_outputs.get("data_category", {}).get("confidence", 0),
            classification_outputs.get("data_format", {}).get("confidence", 0),
            classification_outputs.get("data_characteristics", {}).get("confidence", 0),
        ),
        "reasoning_explanation": "",
        "evidence": evidence_json,
    }

    # STEP 5: Unified auditor validation
    auditor_prompt = load_yaml_prompt(
        f"prompts/dsrp/{dimension_name}/auditor.yaml"
    )

    auditor_input = {
        "classification_outputs": combined_classification,
        "evidence": evidence_json,
    }

    audit_response = llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(auditor_input)
        )
    )

    audit_json = parse_llm_json(audit_response.content)

    # Store both pre-audit and audited outputs for notebook-level debugging.
    state["dsrp_outputs"][f"{dimension_name}_pre_audit"] = {
        "classification_outputs": classification_outputs,
        "combined_classification": combined_classification,
        "evidence": evidence_json,
    }
    state["dsrp_outputs"][dimension_name] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}