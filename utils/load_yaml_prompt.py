import yaml
import os
from langchain_core.prompts import ChatPromptTemplate


GLOBAL_OUTPUT_CONSISTENCY_ADDENDUM = """

GLOBAL OUTPUT FORMAT ENFORCEMENT
════════════════════════════════
1) Return only a single valid JSON object. No markdown fences, no prose outside JSON.
2) Use strict JSON syntax: double-quoted keys/strings, no trailing commas.
3) Escape quotes inside strings properly.
4) If you provide reasoning text, include inline citation markers such as [1], [2].
5) If bibliography is present, keep it as a JSON array of objects with fields:
   id, page, section, direct_quote.
6) Prefer 8-10 high-quality evidence items when sufficient evidence exists in context.
7) Do not fabricate bibliography entries; include only evidence grounded in supplied context.
"""

def load_yaml_prompt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    instructions = config["instructions"]
    enforce_global = os.getenv("DSRP_PROMPT_CONSISTENCY", "1").strip().lower() not in {"0", "false", "off"}
    if enforce_global:
        instructions = f"{instructions}\n{GLOBAL_OUTPUT_CONSISTENCY_ADDENDUM}"

    return ChatPromptTemplate.from_messages([
        ("system", instructions),
        ("human", "{{input}}")
    ],
    template_format="jinja2") # Escape curly braces in the prompt template

