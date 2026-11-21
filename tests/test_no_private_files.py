"""
test_no_private_files.py

Fails if `git ls-files` contains any path matching patterns
that indicate private or derived-from-private documents.
This test must be fast (no OCR) and always runs.
"""

import re
import subprocess


PRIVATE_PATTERNS = [
    re.compile(r".*-real\\..*"),        # *-real.*
    re.compile(r"images/saudi-.*"),      # images/saudi-*
    re.compile(r"images/WhatsApp.*"),    # images/WhatsApp*
    re.compile(r"vendra_private/.*"),    # vendra_private/
    re.compile(r"private_data/.*"),      # private_data/
    re.compile(r"result.*\\.json"),      # result*.json
    re.compile(r"out_.*\\.json"),        # out_*.json
]


def test_no_private_files_in_git():
    """Assert that no tracked file matches a private-data pattern."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = result.stdout.strip().splitlines()
    violations = []
    for path in tracked:
        for pat in PRIVATE_PATTERNS:
            if pat.match(path):
                violations.append((path, pat.pattern))
                break  # one match per file is enough

    if violations:
        lines = "\n".join(f"  {p!r}  matched  {pat!r}" for p, pat in violations)
        raise AssertionError(
            f"Private or derived files found in git index:\n{lines}\n"
            "Remove them from tracking (add to .gitignore and use git rm --cached)."
        )
