"""Convert heterogeneous VT behaviour summaries into label-free canonical events."""

from __future__ import annotations

import ipaddress
import json
import re
import urllib.parse
from collections.abc import Mapping
from typing import Any

from pe_research.data.au_pemal_2025.records import EventRecord

NORMALIZATION_VERSION = "vt_behavior_events/v1"

_HASH = re.compile(r"(?i)\b[0-9a-f]{32,64}\b")
_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
_WINDOWS_USER = re.compile(r"(?i)([a-z]:\\users\\)[^\\\s]+")
_UNIX_USER = re.compile(r"(?i)(/home/)[^/\s]+")
_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret)\s*[:=]\s*[^\s,;&]+"
)
_PID = re.compile(r"(?i)\b(pid|process[_ ]?id)\s*[:= ]\s*\d+")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_WHITESPACE = re.compile(r"\s+")


EVENT_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("calls_highlighted", "api", "call", "api"),
    ("command_executions", "command", "execute", "command"),
    ("processes_created", "process", "create", "process"),
    ("processes_terminated", "process", "terminate", "process"),
    ("processes_killed", "process", "kill", "process"),
    ("processes_injected", "process", "inject", "process"),
    ("files_opened", "file", "open", "file"),
    ("files_written", "file", "write", "file"),
    ("files_deleted", "file", "delete", "file"),
    ("files_attribute_changed", "file", "change_attribute", "file"),
    ("registry_keys_opened", "registry", "open", "registry_key"),
    ("registry_keys_set", "registry", "set", "registry_key"),
    ("registry_keys_deleted", "registry", "delete", "registry_key"),
    ("services_opened", "service", "open", "service"),
    ("services_created", "service", "create", "service"),
    ("services_started", "service", "start", "service"),
    ("services_stopped", "service", "stop", "service"),
    ("services_deleted", "service", "delete", "service"),
    ("modules_loaded", "module", "load", "module"),
    ("dns_lookups", "network", "dns_lookup", "domain"),
    ("http_conversations", "network", "http", "url"),
    ("ip_traffic", "network", "connect", "endpoint"),
)


def _replace_ip(match: re.Match[str]) -> str:
    text = match.group(0)
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return "<IP>"
    return "<PRIVATE_IP>" if address.is_private else "<PUBLIC_IP>"


def _normalize_urls(text: str) -> str:
    words = text.split()
    normalized: list[str] = []
    for word in words:
        stripped = word.strip("\"'(),")
        if stripped.startswith(("http://", "https://")):
            parsed = urllib.parse.urlsplit(stripped)
            path = re.sub(r"/\d+(?=/|$)", "/<ID>", parsed.path)
            replacement = urllib.parse.urlunsplit(
                (parsed.scheme.lower(), parsed.netloc.lower(), path, "", "")
            )
            word = word.replace(stripped, replacement)
        normalized.append(word)
    return " ".join(normalized)


def normalize_text(value: object) -> str:
    """Normalize identifiers while retaining behaviorally useful context."""
    if isinstance(value, Mapping):
        serializable = {str(key): value[key] for key in sorted(value, key=str)}
        text = json.dumps(serializable, sort_keys=True, ensure_ascii=False)
    elif isinstance(value, list):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    else:
        text = str(value)
    text = _normalize_urls(text)
    text = _WINDOWS_USER.sub(r"\1<USER>", text)
    text = _UNIX_USER.sub(r"\1<USER>", text)
    text = _SECRET.sub(lambda match: f"{match.group(1)}=<REDACTED>", text)
    text = _PID.sub(lambda match: f"{match.group(1)}=<PID>", text)
    text = _UUID.sub("<UUID>", text)
    text = _HASH.sub("<HASH>", text)
    text = _IPV4.sub(_replace_ip, text)
    return _WHITESPACE.sub(" ", text).strip()


def _value_timestamp(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    for key in ("timestamp", "time", "datetime", "first_seen"):
        item = value.get(key)
        if item not in (None, ""):
            return str(item)
    return ""


def _value_result(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    for key in ("result", "status", "response_status_code", "success"):
        item = value.get(key)
        if item not in (None, ""):
            return normalize_text(item)
    return ""


def behavior_coverage(attributes: Mapping[str, Any]) -> int:
    return sum(
        1
        for field, _, _, _ in EVENT_FIELDS
        if isinstance(attributes.get(field), list) and attributes[field]
    )


def events_from_behavior(trace_id: str, attributes: Mapping[str, Any]) -> list[EventRecord]:
    pending: list[tuple[str, str, str, str, str, str]] = []
    for field, event_type, operation, object_role in EVENT_FIELDS:
        values = attributes.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            normalized = normalize_text(value)
            if not normalized:
                continue
            result = _value_result(value)
            observed_at = _value_timestamp(value)
            canonical = (
                f"event_type={event_type} operation={operation} "
                f"object_role={object_role} object={normalized}"
            )
            if result:
                canonical += f" result={result}"
            pending.append(
                (event_type, operation, object_role, result, canonical, observed_at)
            )

    ordering_known = bool(pending) and all(item[5] for item in pending)
    if ordering_known:
        pending.sort(key=lambda item: item[5])
    return [
        EventRecord(
            trace_id=trace_id,
            event_index=index,
            event_type=event_type,
            operation=operation,
            object_role=object_role,
            result=result,
            canonical_text=canonical,
            observed_at=observed_at,
            ordering_known=int(ordering_known),
            object_known=1,
            result_known=int(bool(result)),
        )
        for index, (
            event_type,
            operation,
            object_role,
            result,
            canonical,
            observed_at,
        ) in enumerate(pending)
    ]
