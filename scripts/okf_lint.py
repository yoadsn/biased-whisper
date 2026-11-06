#!/usr/bin/env python3
"""Minimal OKF v0.2 conformance check for ./knowledge (frontmatter + type, reserved files, index coverage)."""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "knowledge"
errors, concepts = [], []
for p in sorted(ROOT.rglob("*.md")):
    rel = p.relative_to(ROOT).as_posix()
    text = p.read_text()
    if p.name == "log.md":
        if not re.search(r"^## \d{4}-\d{2}-\d{2}$", text, re.M):
            errors.append(f"{rel}: log.md needs '## YYYY-MM-DD' headings")
        continue
    if p.name == "index.md":
        if rel != "index.md" and text.startswith("---"):
            errors.append(f"{rel}: only the root index.md may have frontmatter")
        continue
    concepts.append(rel)
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        errors.append(f"{rel}: missing frontmatter")
        continue
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        errors.append(f"{rel}: bad YAML ({e})")
        continue
    if not fm.get("type"):
        errors.append(f"{rel}: missing 'type'")
    for k in ("title", "description"):
        if not fm.get(k):
            print(f"warn: {rel}: no {k}")
idx = (ROOT / "index.md").read_text() if (ROOT / "index.md").exists() else ""
for c in concepts:
    if c not in idx:
        errors.append(f"{c}: not listed in knowledge/index.md")
print(f"{len(concepts)} concepts checked, {len(errors)} problems")
for e in errors:
    print("ERROR", e)
sys.exit(1 if errors else 0)
