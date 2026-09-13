from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


FoundationalParadigm = Literal[
    "Classical Statistical Modelling",
    "Machine Learning",
    "Mixed",
]


class EvidenceItem(BaseModel):
    id: int = Field(default=0)
    page: str = Field(default="")
    section: str = Field(default="")
    direct_quote: str = Field(default="")
    method_used: bool = Field(default=True)

    @field_validator("page", "section", "direct_quote", mode="before")
    @classmethod
    def stringify_optional_text(cls, value):
        if value is None:
            return ""
        return str(value)


class ModellingEvidenceSchema(BaseModel):
    modelling_evidence: List[EvidenceItem] = Field(default_factory=list)


class FoundationalClassifierSchema(BaseModel):
    foundational_paradigm: FoundationalParadigm = Field(default="Machine Learning")
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)

    @field_validator("foundational_paradigm", mode="before")
    @classmethod
    def normalize_foundational(cls, value):
        mapping = {
            "classical statistical modelling": "Classical Statistical Modelling",
            "classical statistical modeling": "Classical Statistical Modelling",
            "statistical model diagnostics": "Classical Statistical Modelling",
            "machine learning": "Machine Learning",
            "mixed": "Mixed",
        }
        if value is None:
            return "Machine Learning"
        return mapping.get(str(value).strip().lower(), value)


class MLLearningClassifierSchema(BaseModel):
    ml_learning_type: List[str] = Field(default_factory=list)
    deep_learning_used: bool = Field(default=False)
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)

    @field_validator("ml_learning_type", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class MLProblemClassifierSchema(BaseModel):
    ml_problem_type: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)

    @field_validator("ml_problem_type", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class FoundationalAuditSchema(BaseModel):
    foundational_paradigm: FoundationalParadigm = Field(default="Machine Learning")
    ml_learning_type: List[str] = Field(default_factory=list)
    ml_problem_type: List[str] = Field(default_factory=list)
    deep_learning_used: bool = Field(default=False)
    confidence: float = Field(default=0.0)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("foundational_paradigm", mode="before")
    @classmethod
    def normalize_foundational(cls, value):
        return FoundationalClassifierSchema.normalize_foundational(value)

    @field_validator("ml_learning_type", "ml_problem_type", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class SpecialisedClassifierSchema(BaseModel):
    specialised_paradigms: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    reasoning: str = Field(default="")
    bibliography: List[EvidenceItem] = Field(default_factory=list)

    @field_validator("specialised_paradigms", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class SpecialisedAuditSchema(BaseModel):
    specialised_paradigms: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("specialised_paradigms", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class GlobalModellingAuditSchema(BaseModel):
    foundational_paradigm: FoundationalParadigm = Field(default="Machine Learning")
    ml_learning_type: List[str] = Field(default_factory=list)
    ml_problem_type: List[str] = Field(default_factory=list)
    deep_learning_used: bool = Field(default=False)
    specialised_paradigms: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[EvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("foundational_paradigm", mode="before")
    @classmethod
    def normalize_foundational(cls, value):
        return FoundationalClassifierSchema.normalize_foundational(value)

    @field_validator("ml_learning_type", "ml_problem_type", "specialised_paradigms", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
