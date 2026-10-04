"""Leakage-resistant normalization of ordered SmartVMI API records."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import PureWindowsPath
from typing import Any, cast

API_NORMALIZATION_VERSION = "smartvmi_api_events/v1"

_HASH = re.compile(r"(?i)\b[0-9a-f]{32,64}\b")
_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
_WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:\\|\\\\)[^\s\"']+")
_WINDOWS_USER = re.compile(r"(?i)([a-z]:\\users\\)[^\\\s]+")
_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret)\s*[:=]\s*[^\s,;&]+"
)
_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,16}\b")
_LONG_ID = re.compile(r"\b\d{7,}\b")
_WHITESPACE = re.compile(r"\s+")

_FUNCTION_KEYS = ("vmi_FunctionName", "function", "function_name", "api")
_MODULE_KEYS = ("vmi_ModuleName", "module", "module_name")
_PARAMETER_KEYS = ("vmi_Parameterlist", "parameters", "parameter_list", "args")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return cast(object, record[key])
    return None


def _path_token(match: re.Match[str]) -> str:
    raw = match.group(0).rstrip(",.;)")
    suffix = PureWindowsPath(raw).suffix.lower()
    return f"<PATH:{suffix}>" if suffix else "<PATH>"


def _normalize_string(value: str) -> str:
    text = _WINDOWS_USER.sub(r"\1<USER>", value)
    text = _SECRET.sub(lambda match: f"{match.group(1)}=<REDACTED>", text)
    text = _WINDOWS_PATH.sub(_path_token, text)
    text = _UUID.sub("<UUID>", text)
    text = _HASH.sub("<HASH>", text)
    text = _ADDRESS.sub("<ADDRESS>", text)
    text = _LONG_ID.sub("<ID>", text)
    return _WHITESPACE.sub(" ", text).strip()


def _normalize_parameter(value: object, key: str = "") -> object:
    key_lower = key.casefold()
    if any(token in key_lower for token in ("password", "secret", "token", "api_key")):
        return "<REDACTED>"
    if any(
        token in key_lower
        for token in ("handle", "address", "pointer", "processid", "threadid", "_pid", "_tid")
    ):
        return "<IDENTIFIER>"
    if isinstance(value, dict):
        return {
            str(child_key): _normalize_parameter(value[child_key], str(child_key))
            for child_key in sorted(value, key=str)
        }
    if isinstance(value, list):
        return [_normalize_parameter(item, key) for item in value]
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "<NONFINITE>"
    return value


def _context_token(value: object, mapping: dict[str, str], prefix: str) -> str:
    if value in (None, ""):
        return ""
    raw = str(value)
    if raw not in mapping:
        mapping[raw] = f"<{prefix}_{len(mapping)}>"
    return mapping[raw]


def normalize_api_event(
    record: dict[str, Any],
    process_tokens: dict[str, str],
    thread_tokens: dict[str, str],
) -> dict[str, object] | None:
    function_value = _first(record, _FUNCTION_KEYS)
    if function_value is None:
        return None
    function = _normalize_string(str(function_value))
    if not function:
        return None
    module_value = _first(record, _MODULE_KEYS)
    module = _normalize_string(str(module_value)).lower() if module_value is not None else ""
    parameters_value = _first(record, _PARAMETER_KEYS)
    parameters_known = int(parameters_value is not None)
    normalized_parameters = _normalize_parameter(
        parameters_value if parameters_value is not None else {}
    )
    parameters_text = json.dumps(
        normalized_parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    process = _context_token(
        record.get("vmi_ProcessDtb") or record.get("process_id"),
        process_tokens,
        "PROCESS",
    )
    thread = _context_token(
        record.get("vmi_ProcessTeb") or record.get("thread_id"),
        thread_tokens,
        "THREAD",
    )
    collector_ts = str(record.get("ts") or record.get("timestamp") or "")
    vmi_ts = str(record.get("vmi_ts") or "")
    canonical = f"api={function}"
    if module:
        canonical += f" module={module}"
    if process:
        canonical += f" process={process}"
    if thread:
        canonical += f" thread={thread}"
    canonical += f" parameters={parameters_text}"
    return {
        "api_name": function,
        "module_name": module,
        "process_token": process,
        "thread_token": thread,
        "parameters_normalized": parameters_text,
        "canonical_text": canonical,
        "collector_ts": collector_ts,
        "vmi_ts": vmi_ts,
        "function_known": 1,
        "module_known": int(bool(module)),
        "parameters_known": parameters_known,
        "timestamp_known": int(bool(vmi_ts or collector_ts)),
    }


def parse_trace_payload(payload: bytes) -> tuple[list[dict[str, Any]], int]:
    """Parse JSON array/object or JSON-lines, returning records and parse errors."""
    text = payload.decode("utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        errors = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                errors += 1
        return records, errors
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)], sum(
            not isinstance(item, dict) for item in value
        )
    if isinstance(value, dict):
        for key in ("events", "records", "data"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)], sum(
                    not isinstance(item, dict) for item in items
                )
        return [value], 0
    return [], 1


def timestamp_sort_value(value: str) -> float | None:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return number if math.isfinite(number) else None
