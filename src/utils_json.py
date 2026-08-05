from __future__ import annotations

import ast
import json
import math
from typing import Any, Dict

import pandas as pd


def parse_json_like(value: Any) -> Dict[str, Any]:
    if value is None or pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _stringify_nested(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=True)
        except Exception:
            return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def flatten_dict_one_level(data: Any, prefix: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        out_key = f"{prefix}_{key_str}" if prefix else key_str
        flattened[out_key] = _stringify_nested(value)
    return flattened
