
import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _extract_first_json_object_or_array(text: str) -> str:
    """Extract first balanced JSON object/array from mixed model output."""
    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not start_positions:
        return text

    start = min(start_positions)
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _escape_invalid_backslashes(text: str) -> str:
    # \X -> \\X only when X is not a valid JSON escape character.
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)


def _escape_control_chars(text: str) -> str:
    # Remove raw control chars that frequently break strict JSON parsing.
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)


def parse_llm_json(text: str) -> Any:
    s = _strip_code_fences((text or "").strip())
    s = _extract_first_json_object_or_array(s)

    candidates = [
        s,
        _remove_trailing_commas(s),
        _escape_invalid_backslashes(s),
        _escape_control_chars(s),
        _escape_control_chars(_escape_invalid_backslashes(_remove_trailing_commas(s))),
    ]

    last_error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e

    raise ValueError(f"LLM returned invalid JSON: {last_error}\nRaw:\n{s[:1600]}")