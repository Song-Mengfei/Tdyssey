"""Read-only scaffold checks using Python's standard library; no environment needed."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []
    try:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        pilot = tomllib.loads((ROOT / "configs/pilot.toml").read_text())
        tomllib.loads((ROOT / ".codex/config.toml").read_text())
        upstream = json.loads((ROOT / "manifests/upstream.json").read_text())
        example = json.loads((ROOT / "manifests/experiment.example.json").read_text())
        gym_metadata = json.loads((ROOT / "manifests/gym-metadata.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    packages = lock.get("package", [])
    requirements = list(project["project"]["dependencies"])
    for extra in project["project"].get("optional-dependencies", {}).values():
        requirements.extend(extra)
    for requirement in requirements:
        match = re.fullmatch(r"([\w.-]+)==([\w.+-]+)", requirement)
        if not match:
            errors.append(f"Static checker needs updating for requirement: {requirement}")
            continue
        name, version = match.groups()
        name = name.lower().replace("_", "-")
        if not any(
            item["name"] == name and item["version"].split("+")[0] == version
            for item in packages
        ):
            errors.append(f"Direct pin absent from lock: {requirement}")

    for repo in upstream["repositories"]:
        if not re.fullmatch(r"[0-9a-f]{40}", repo["commit"]):
            errors.append(f"Upstream commit must be a full SHA: {repo['name']}")
    gym_entries = [item for item in project["tool"]["uv"].get("dependency-metadata", [])
                   if item["name"] == "gym" and item["version"] == gym_metadata["version"]]
    if len(gym_entries) != 1 or gym_entries[0]["requires-dist"] != gym_metadata["requires_dist"]:
        errors.append("Gym resolution metadata differs from recorded source provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", gym_metadata["source_sha256"]):
        errors.append("Gym metadata archive needs a SHA-256 provenance record")
    if example["status"] != "template_not_executed":
        errors.append("Example experiment must remain labelled as a template")
    if pilot["budgets"]["long_jobs_authorized"]:
        errors.append("Shared pilot must not grant machine-specific training authorization")

    for path in (ROOT / "scripts").glob("*.py"):
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))

    # Catch stale local documentation links without requesting remote pages.
    markdown = [ROOT / "README.md", ROOT / "AGENTS.md", *(ROOT / "docs").glob("*.md")]
    for path in markdown:
        for target in re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if not (path.parent / unquote(parsed.path)).exists():
                errors.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")

    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"PASS: configuration, {len(packages)} locked package entries, upstream SHAs, "
          "Python syntax, and local documentation links")
    print("Static checks only; no dependency installation or runtime compatibility claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
