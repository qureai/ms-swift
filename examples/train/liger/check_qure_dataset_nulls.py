#!/usr/bin/env python3
"""Check Qure SFT datasets for null values that Swift will reject.

This script reads DATASET_PATHS from qure_full_4b.sh, streams each JSON/JSONL
dataset, and reports:
  - top-level columns whose value is null
  - messages/conversation/conversations fields that are null or malformed
  - null message roles/content, including ShareGPT-style from/value records
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


MESSAGE_KEYS = ("messages", "conversation", "conversations")
ROLE_KEYS = ("role", "from")
CONTENT_KEYS = ("content", "value")


def parse_dataset_paths(script_path: Path) -> List[Path]:
    text = script_path.read_text(encoding="utf-8")
    match = re.search(r"DATASET_PATHS=\((.*?)\)", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find DATASET_PATHS=(...) in {script_path}")

    body = "\n".join(line.split("#", 1)[0] for line in match.group(1).splitlines())
    paths = [Path(token) for token in shlex.split(body)]
    if not paths:
        raise ValueError(f"DATASET_PATHS is empty in {script_path}")
    return paths


def first_nonspace_char(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                return ""
            stripped = chunk.lstrip()
            if stripped:
                return stripped[0]


def iter_json_records(path: Path) -> Iterator[Tuple[int, Any]]:
    first_char = first_nonspace_char(path)
    if first_char == "[":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            yield 1, data
            return
        for idx, row in enumerate(data, start=1):
            yield idx, row
        return

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def truncate(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def message_role(message: Dict[str, Any]) -> Optional[Any]:
    for key in ROLE_KEYS:
        if key in message:
            return message[key]
    return None


def message_content(message: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    for key in CONTENT_KEYS:
        if key in message:
            return key, message[key]
    return None, None


def check_row(row: Any) -> List[str]:
    issues: List[str] = []
    if row is None:
        return ["row_is_null"]
    if not isinstance(row, dict):
        return [f"row_is_{type(row).__name__}"]

    for key, value in row.items():
        if value is None:
            issues.append(f"top_level_null:{key}")

    present_message_key = False
    for key in MESSAGE_KEYS:
        if key not in row:
            continue
        present_message_key = True
        messages = row[key]
        if messages is None:
            issues.append(f"{key}_is_null")
            continue
        if not isinstance(messages, list):
            issues.append(f"{key}_is_{type(messages).__name__}")
            continue
        if len(messages) == 0:
            issues.append(f"{key}_is_empty")
            continue

        for idx, message in enumerate(messages):
            prefix = f"{key}[{idx}]"
            if message is None:
                issues.append(f"{prefix}_is_null")
                continue
            if not isinstance(message, dict):
                issues.append(f"{prefix}_is_{type(message).__name__}")
                continue

            for role_key in ROLE_KEYS:
                if role_key in message and message[role_key] is None:
                    issues.append(f"{prefix}.{role_key}_is_null")
            for content_key in CONTENT_KEYS:
                if content_key in message and message[content_key] is None:
                    issues.append(f"{prefix}.{content_key}_is_null")

            role = message_role(message)
            content_key, content = message_content(message)
            if content_key is None:
                issues.append(f"{prefix}.content_or_value_missing")
            elif role in {"assistant", "gpt"} and content is None:
                issues.append(f"{prefix}.assistant_content_is_null")

    if not present_message_key:
        issues.append("messages_conversation_field_missing")

    return issues


def scan_file(path: Path, max_examples: int, max_records: Optional[int]) -> Tuple[int, Counter, List[Tuple[int, str, Any]]]:
    counts: Counter = Counter()
    examples: List[Tuple[int, str, Any]] = []
    scanned = 0

    for line_no, row in iter_json_records(path):
        scanned += 1
        if max_records is not None and scanned > max_records:
            break
        issues = check_row(row)
        for issue in issues:
            counts[issue] += 1
            if len(examples) < max_examples:
                examples.append((line_no, issue, row))

    return scanned, counts, examples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check Qure training datasets for null message/content fields.")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("qure_full_4b.sh"),
        help="Training shell script containing DATASET_PATHS.",
    )
    parser.add_argument("--max-examples-per-file", type=int, default=20)
    parser.add_argument("--max-records-per-file", type=int, default=None, help="Debug option; default scans all rows.")
    args = parser.parse_args(argv)

    dataset_paths = parse_dataset_paths(args.script)
    total_issues = 0

    for path in dataset_paths:
        print(f"\n=== {path} ===")
        if not path.exists():
            print(f"ERROR: file does not exist: {path}")
            total_issues += 1
            continue

        try:
            scanned, counts, examples = scan_file(path, args.max_examples_per_file, args.max_records_per_file)
        except Exception as exc:
            print(f"ERROR: failed to read {path}: {exc}")
            total_issues += 1
            continue

        issue_count = sum(counts.values())
        total_issues += issue_count
        print(f"scanned_rows={scanned} issue_count={issue_count}")

        if counts:
            print("issue_summary:")
            for issue, count in counts.most_common():
                print(f"  {issue}: {count}")
            print("examples:")
            for line_no, issue, row in examples:
                print(f"  line={line_no} issue={issue} row={truncate(row)}")
        else:
            print("OK: no null/malformed message issues found")

    if total_issues:
        print(f"\nFAILED: found {total_issues} issue(s).")
        return 1
    print("\nPASSED: no issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
