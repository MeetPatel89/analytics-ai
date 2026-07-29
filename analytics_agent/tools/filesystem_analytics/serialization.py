"""Bounded JSON serialization shared by filesystem tools."""

from __future__ import annotations

import base64
import json
import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import cast

from analytics_agent.tools.filesystem_analytics.models import (
    MAX_TOOL_OUTPUT_BYTES,
)

MAX_VALUE_CHARACTERS = 1000
MAX_NESTED_ITEMS = 50
MAX_NESTING_DEPTH = 4


def json_result(payload: object) -> str:
    """Serialize one JSON-safe tool result."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def bounded_list_result(
    payload: dict[str, object],
    key: str,
    values: list[object],
    *,
    requested_truncated: bool = False,
) -> str:
    """Fit a result list within the global tool-output byte cap."""
    accepted: list[object] = []
    for value in values:
        candidate = {**payload, key: [*accepted, value], "truncated": False}
        if _json_size(candidate) > MAX_TOOL_OUTPUT_BYTES:
            break
        accepted.append(value)
    truncated = requested_truncated or len(accepted) < len(values)
    result = {**payload, key: accepted, "truncated": truncated}
    if _json_size(result) > MAX_TOOL_OUTPUT_BYTES:
        raise ValueError("Tool-result metadata exceeds the output byte limit.")
    return json_result(result)


def bounded_rows_result(
    payload: dict[str, object],
    rows: list[list[object]],
    *,
    requested_truncated: bool,
) -> str:
    """Fit positional rows within the global tool-output byte cap."""
    accepted: list[list[object]] = []
    for row in rows:
        candidate = {
            **payload,
            "rows": [*accepted, row],
            "returned_row_count": len(accepted) + 1,
            "truncated": False,
        }
        if _json_size(candidate) > MAX_TOOL_OUTPUT_BYTES:
            break
        accepted.append(row)
    truncated = requested_truncated or len(accepted) < len(rows)
    result = {
        **payload,
        "rows": accepted,
        "returned_row_count": len(accepted),
        "truncated": truncated,
    }
    if _json_size(result) > MAX_TOOL_OUTPUT_BYTES:
        raise ValueError("Tool-result metadata exceeds the output byte limit.")
    return json_result(result)


def bounded_text_result(payload: dict[str, object], content: str) -> str:
    """Fit decoded text within the global tool-output byte cap."""
    candidate = {**payload, "content": content}
    if _json_size(candidate) <= MAX_TOOL_OUTPUT_BYTES:
        return json_result(candidate)

    low = 0
    high = len(content)
    while low < high:
        middle = (low + high + 1) // 2
        shortened = {
            **payload,
            "content": content[:middle],
            "truncated": True,
            "output_truncated": True,
        }
        if _json_size(shortened) <= MAX_TOOL_OUTPUT_BYTES:
            low = middle
        else:
            high = middle - 1
    result = {
        **payload,
        "content": content[:low],
        "truncated": True,
        "output_truncated": True,
    }
    return json_result(result)


def normalize_value(value: object, *, depth: int = 0) -> object:
    """Convert DuckDB and Arrow scalar values into bounded JSON values."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _bounded_string(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return _bounded_string(f"base64:{encoded}")
    if depth >= MAX_NESTING_DEPTH:
        return _bounded_string(str(value))
    if isinstance(value, dict):
        items = list(value.items())[:MAX_NESTED_ITEMS]
        return {
            _bounded_string(str(key)): normalize_value(item, depth=depth + 1)
            for key, item in items
        }
    if isinstance(value, (list, tuple)):
        return [
            normalize_value(item, depth=depth + 1) for item in value[:MAX_NESTED_ITEMS]
        ]

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        return normalize_value(scalar_item(), depth=depth)
    return _bounded_string(str(value))


def normalize_rows(rows: list[tuple[object, ...]]) -> list[list[object]]:
    """Normalize positional query rows."""
    return [
        [normalize_value(value) for value in row]
        for row in cast(list[tuple[object, ...]], rows)
    ]


def _bounded_string(value: str) -> str:
    if len(value) <= MAX_VALUE_CHARACTERS:
        return value
    return f"{value[: MAX_VALUE_CHARACTERS - 1]}…"


def _json_size(payload: object) -> int:
    return len(json_result(payload).encode("utf-8"))
