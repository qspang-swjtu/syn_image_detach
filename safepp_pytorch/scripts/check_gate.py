#!/usr/bin/env python3
"""Compare candidate metrics against baseline metrics and enforce promotion gates."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Model promotion gate checker.')
    parser.add_argument('--candidate_dir', required=True, help='Directory containing candidate metrics JSON files.')
    parser.add_argument('--baseline_dir', default='', help='Directory containing baseline metrics JSON files.')
    parser.add_argument('--gate', default='configs/gate.yaml', help='Gate YAML config.')
    parser.add_argument('--out', default='', help='Optional gate report JSON path.')
    parser.add_argument('--soft_fail', action='store_true', help='Always exit 0, even when gates fail.')
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_rule(split: str, metric: str, rule: Dict[str, Any], candidate_metrics: Dict[str, Any], baseline_metrics: Optional[Dict[str, Any]], missing_baseline_policy: str) -> Dict[str, Any]:
    candidate_value = to_float(candidate_metrics.get(metric))
    baseline_value = to_float(baseline_metrics.get(metric)) if baseline_metrics else None
    result = {'split': split, 'metric': metric, 'candidate': candidate_value, 'baseline': baseline_value, 'rule': rule, 'status': 'pass', 'messages': []}
    if candidate_value is None:
        result['status'] = 'fail'
        result['messages'].append('candidate metric is missing or non-numeric')
        return result
    if rule.get('min_value') is not None and candidate_value < float(rule['min_value']):
        result['status'] = 'fail'
        result['messages'].append(f'candidate {candidate_value:.8g} < min_value {float(rule["min_value"]):.8g}')
    if rule.get('max_value') is not None and candidate_value > float(rule['max_value']):
        result['status'] = 'fail'
        result['messages'].append(f'candidate {candidate_value:.8g} > max_value {float(rule["max_value"]):.8g}')
    for delta_key, comparator in [('min_delta', 'min'), ('max_delta', 'max')]:
        if delta_key not in rule:
            continue
        if baseline_value is None:
            if missing_baseline_policy == 'fail':
                result['status'] = 'fail'
                result['messages'].append(f'baseline metric missing for {delta_key}')
            else:
                result['messages'].append(f'skipped {delta_key}; baseline metric missing')
            continue
        threshold = baseline_value + float(rule[delta_key])
        if comparator == 'min' and candidate_value < threshold:
            result['status'] = 'fail'
            result['messages'].append(f'candidate {candidate_value:.8g} < baseline {baseline_value:.8g} + min_delta {float(rule[delta_key]):.8g}')
        if comparator == 'max' and candidate_value > threshold:
            result['status'] = 'fail'
            result['messages'].append(f'candidate {candidate_value:.8g} > baseline {baseline_value:.8g} + max_delta {float(rule[delta_key]):.8g}')
    return result


def main():
    args = parse_args()
    gate_path = Path(args.gate)
    if not gate_path.is_absolute():
        gate_path = Path.cwd() / gate_path
    gate = load_yaml(gate_path)
    candidate_dir = Path(args.candidate_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    rules = gate.get('rules', {})
    behavior = gate.get('behavior', {})
    missing_baseline_policy = behavior.get('missing_baseline_for_delta_rules', 'skip')
    if missing_baseline_policy not in {'skip', 'fail'}:
        raise ValueError('behavior.missing_baseline_for_delta_rules must be skip or fail')
    checks: List[Dict[str, Any]] = []
    for split, split_rules in rules.items():
        candidate_metrics = load_json(candidate_dir / f'{split}.json')
        if candidate_metrics is None:
            checks.append({'split': split, 'metric': '*', 'candidate': None, 'baseline': None, 'rule': {}, 'status': 'fail', 'messages': [f'missing candidate metrics file: {candidate_dir / f"{split}.json"}']})
            continue
        baseline_metrics = load_json(baseline_dir / f'{split}.json') if baseline_dir else None
        for metric, rule in split_rules.items():
            checks.append(evaluate_rule(split, metric, rule or {}, candidate_metrics, baseline_metrics, missing_baseline_policy))
    failed = [x for x in checks if x['status'] != 'pass']
    report = {'status': 'fail' if failed else 'pass', 'num_checks': len(checks), 'num_failed': len(failed), 'candidate_dir': str(candidate_dir), 'baseline_dir': str(baseline_dir) if baseline_dir else '', 'gate': str(gate_path), 'checks': checks}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed and not args.soft_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
