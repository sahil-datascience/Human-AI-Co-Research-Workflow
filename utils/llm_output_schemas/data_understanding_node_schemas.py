#-------------------------------------- Output Schema --------------------------------------#
# Use a structured output schema for all binary classifiers to ensure consistency
from typing import List
from pydantic import BaseModel, Field

class BibliographyItem(BaseModel):
    id: int = Field(description="Unique incremental identifier starting from 1")
    page: str = Field(description="Page number or range where the information was found")
    section: str = Field(description="Name or number of the section/chapter")
    direct_quote: str = Field(description="Exact verbatim quote from the text source")

class Primary_labels_Schema(BaseModel):
    label: bool = Field(description="True if the statement is verified, False otherwise")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning_explanation: str = Field(description="Detailed step-by-step reasoning for the label")
    bibliography: List[BibliographyItem] = Field(description="List of text sources used to verify the statement")

#----------------- Auditor Schema -----------------#

class ValidatedBibliographyItem(BaseModel):
    id: int = Field(description="Unique incremental identifier starting from 1")
    page: str = Field(description="Page number or range reference")
    section: str = Field(description="Section, header, or chapter name")
    direct_quote: str = Field(description="Exact quote from the text source")

class AuditorSchema(BaseModel):
    data_category: List[str] = Field(description="List of matching data categories")
    data_format: List[str] = Field(description="List of identified data formats")
    data_characteristics: List[str] = Field(description="List of observed data characteristics")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    validated_reasoning: str = Field(description="Reasoning explaining the selections")
    validated_bibliography: List[ValidatedBibliographyItem] = Field(description="Sources validating this entry")
    audit_commentary: str = Field(description="Additional auditor flags, notes, or comments")

