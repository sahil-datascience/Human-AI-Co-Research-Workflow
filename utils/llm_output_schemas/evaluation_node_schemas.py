from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


EvaluationStrategy = Literal[
    "Statistical Model Diagnostics",
    "Machine Learning Evaluation",
    "Mixed Evaluation",
]


class EvidenceItem(BaseModel):
    id: int = Field(default=0)
    page: str = Field(default="")
    section: str = Field(default="")
    direct_quote: str = Field(default="")

    @field_validator("page", "section", "direct_quote", mode="before")
    @classmethod
    def stringify_optional_text(cls, value):
        if value is None:
            return ""
        return str(value)


class MetricsEvidenceSchema(BaseModel):
    evaluation_evidence: List[EvidenceItem] = Field(default_factory=list)


class TheoryEvidenceSchema(BaseModel):
    theory_evidence: List[EvidenceItem] = Field(default_factory=list)


class InterpretabilityEvidenceSchema(BaseModel):
    interpretability_evidence: List[EvidenceItem] = Field(default_factory=list)


class EthicalEvidenceSchema(BaseModel):
    ethical_evidence: List[EvidenceItem] = Field(default_factory=list)


class MetricsEvaluationSchema(BaseModel):
    evaluation_strategy: EvaluationStrategy = Field(default="Machine Learning Evaluation")
    learning_type: List[str] = Field(default_factory=list)
    problem_type: List[str] = Field(default_factory=list)
    evaluation_metrics_present: str = Field(default="No")
    validation_procedure: List[str] = Field(default_factory=lambda: ["Not reported"])
    effect_size_reported: str = Field(default="Not applicable")
    assumption_checks_reported: str = Field(default="Not applicable")
    confidence: float = Field(default=0.0)
    reasoning: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("evaluation_strategy", mode="before")
    @classmethod
    def normalize_strategy(cls, value):
        mapping = {
            "statistical model diagnostics": "Statistical Model Diagnostics",
            "machine learning evaluation": "Machine Learning Evaluation",
            "mixed evaluation": "Mixed Evaluation",
        }
        if value is None:
            return "Machine Learning Evaluation"
        return mapping.get(str(value).strip().lower(), value)

    @field_validator("learning_type", "problem_type", "validation_procedure", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class TheoryOrientationSchema(BaseModel):
    explicit_theory: str = Field(default="No")
    implicit_theory_detected: str = Field(default="No")
    epistemological_orientation: str = Field(default="Data-driven discovery")
    primary_research_orientation: str = Field(default="Method-oriented research")
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("explicit_theory", "implicit_theory_detected", mode="before")
    @classmethod
    def normalize_yes_no(cls, value):
        if value is None:
            return "No"
        text = str(value).strip()
        lowered = text.lower()
        if lowered.startswith("no evidence") or lowered in {"no", "false", "0"}:
            return "No"
        if lowered in {"yes", "true", "1"}:
            return "Yes"
        return text

    @field_validator("epistemological_orientation", mode="before")
    @classmethod
    def normalize_epistemological_orientation(cls, value):
        if value is None:
            return "Data-driven discovery"
        text = str(value).strip()
        lowered = text.lower()
        if "theory-driven" in lowered:
            return "Theory-driven research"
        if "hybrid" in lowered and "data" in lowered and "theory" in lowered:
            return "Hybrid (data + theory)"
        if "data-driven" in lowered:
            return "Data-driven discovery"
        return text

    @field_validator("primary_research_orientation", mode="before")
    @classmethod
    def normalize_primary_orientation(cls, value):
        if value is None:
            return "Method-oriented research"
        text = str(value).strip()
        lowered = text.lower()
        if "hybrid" in lowered and "method" in lowered and "knowledge" in lowered:
            return "Hybrid (method + knowledge)"
        if "knowledge-oriented" in lowered:
            return "Knowledge-oriented research"
        if "method-oriented" in lowered:
            return "Method-oriented research"
        return text


class InterpretabilitySchema(BaseModel):
    interpretability_discussed: str = Field(default="No")
    interpretability_approach: str = Field(default="")
    interpretability_method: str = Field(default="")
    model_transparency_level: str = Field(default="Low transparency")
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("interpretability_discussed", mode="before")
    @classmethod
    def normalize_discussed(cls, value):
        if value is None:
            return "No"
        text = str(value).strip()
        lowered = text.lower()
        if lowered.startswith("no evidence") or lowered in {"no", "false", "0"}:
            return "No"
        if lowered in {"yes", "true", "1"}:
            return "Yes"
        return text

    @field_validator("interpretability_approach", "interpretability_method", mode="before")
    @classmethod
    def normalize_approach(cls, value):
        if value is None:
            return ""
        text = str(value).strip()
        lowered = text.lower()
        if "post-hoc" in lowered or "post hoc" in lowered or "feature importance" in lowered:
            return "Post-hoc explanation"
        if "inherently interpretable" in lowered:
            return "Inherently interpretable model"
        if "none" in lowered or lowered.startswith("no evidence"):
            return "None reported"
        return text

    @field_validator("model_transparency_level", mode="before")
    @classmethod
    def normalize_transparency(cls, value):
        if value is None:
            return "Low transparency"
        text = str(value).strip()
        lowered = text.lower()
        if "high" in lowered:
            return "High transparency"
        if "moderate" in lowered:
            return "Moderate transparency"
        if "low" in lowered:
            return "Low transparency"
        return text


class EthicalSocialSchema(BaseModel):
    privacy_protection_reported: str = Field(default="No")
    bias_fairness_considered: str = Field(default="No")
    societal_impact_discussed: str = Field(default="No")
    ethical_reflection_level: str = Field(default="Low")
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("privacy_protection_reported", "bias_fairness_considered", "societal_impact_discussed", mode="before")
    @classmethod
    def normalize_yes_no(cls, value):
        if value is None:
            return "No"
        text = str(value).strip()
        lowered = text.lower()
        if lowered.startswith("no evidence") or lowered in {"no", "false", "0"}:
            return "No"
        if lowered in {"yes", "true", "1"}:
            return "Yes"
        return text

    @field_validator("ethical_reflection_level", mode="before")
    @classmethod
    def normalize_reflection_level(cls, value):
        if value is None:
            return "Low"
        text = str(value).strip()
        lowered = text.lower()
        if "high" in lowered:
            return "High"
        if "moderate" in lowered:
            return "Moderate"
        if "low" in lowered:
            return "Low"
        return text
