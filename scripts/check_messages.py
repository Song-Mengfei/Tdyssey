"""Check shared commit/PR formats without external packages, network, or writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


TITLE = re.compile(
    r"(?:feat|fix|docs|test|refactor|perf|build|ci|chore|style|revert)"
    r"(?:\([a-z0-9][a-z0-9._/-]*\))?!?: \S.*"
)
SECTIONS = ("Summary", "Changes", "Validation", "Limitations", "Related issues")


def check_title(title: str) -> list[str]:
    errors = []
    if "\n" in title or "\r" in title or not TITLE.fullmatch(title):
        errors.append("Title must use type(scope): description with a supported lowercase type")
    if len(title) > 72:
        errors.append("Title must be at most 72 characters")
    return errors


def check_commit(message: str) -> list[str]:
    lines = message.splitlines()
    if not lines:
        return ["Commit message is empty"]
    errors = check_title(lines[0])
    if len(lines) > 1 and lines[1].strip():
        errors.append("Separate the commit title and body with a blank line")
    return errors


def check_pr(title: str, body: str) -> list[str]:
    errors = check_title(title)
    # Template comments do not count as completed information.
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    headers = list(re.finditer(r"(?m)^## ([^\n]+?)[ \t]*$", body))
    sections: dict[str, str] = {}
    for index, header in enumerate(headers):
        name = header.group(1).strip()
        if name in sections:
            errors.append(f"Duplicate PR section: {name}")
        end = headers[index + 1].start() if index + 1 < len(headers) else len(body)
        sections[name] = body[header.end():end].strip()
    for name in SECTIONS:
        if not re.sub(r"[\s*#-]", "", sections.get(name, "")):
            errors.append(f"Fill the PR section: {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--commit-message", type=Path, help="Read a prepared commit-message file")
    mode.add_argument("--event", type=Path, help="Read a GitHub pull_request event file")
    mode.add_argument("--title", help="PR title for a local check")
    parser.add_argument("--body-file", type=Path, help="PR description for a local check")
    args = parser.parse_args()
    if args.title is not None and args.body_file is None:
        parser.error("--title requires --body-file")
    if args.body_file is not None and args.title is None:
        parser.error("--body-file requires --title")
    try:
        if args.commit_message:
            errors = check_commit(args.commit_message.read_text(encoding="utf-8"))
        elif args.event:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            pr = event["pull_request"]
            errors = check_pr(pr["title"], pr.get("body") or "")
        else:
            errors = check_pr(args.title, args.body_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: Cannot read message input: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if errors:
        return 1
    print("PASS: message format (content and evidence still need review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
