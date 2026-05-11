#!/usr/bin/env python3
"""Create a runtime YAML config by overriding dot-path keys."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Render a runtime YAML config from a base config plus dot-path overrides.')
    parser.add_argument('--base', required=True, help='Base YAML config.')
    parser.add_argument('--output', required=True, help='Output YAML config path.')
    parser.add_argument('--set', dest='sets', action='append', default=[], help='Override in KEY=VALUE format. Can be repeated.')
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {'none', 'null', '~'}:
        return None
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return yaml.safe_load(text)
    try:
        if any(ch in text for ch in ['.', 'e', 'E']):
            return float(text)
        return int(text)
    except ValueError:
        return text


def set_by_dot_path(cfg: Dict[str, Any], key: str, value: Any):
    parts = [p for p in key.split('.') if p]
    if not parts:
        raise ValueError(f'Invalid empty key: {key!r}')
    cursor = cfg
    for part in parts[:-1]:
        if part not in cursor or cursor[part] is None:
            cursor[part] = {}
        if not isinstance(cursor[part], dict):
            raise TypeError(f'Cannot set {key}: {part} already exists and is not a mapping.')
        cursor = cursor[part]
    cursor[parts[-1]] = value


def main():
    args = parse_args()
    with open(args.base, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    for item in args.sets:
        if '=' not in item:
            raise ValueError(f'--set must be KEY=VALUE, got: {item}')
        key, value = item.split('=', 1)
        set_by_dot_path(cfg, key.strip(), parse_scalar(value))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f'[OK] wrote runtime config: {out}')


if __name__ == '__main__':
    main()
