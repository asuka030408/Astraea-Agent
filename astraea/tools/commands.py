"""Bounded command execution for inspection and tests."""

import shlex
import subprocess

from langchain.tools import tool

from astraea.tools.files import workspace


@tool
def run_command(command: str) -> str:
    """Run pytest or a read-only git command in the workspace (no shell syntax)."""
    try:
        argv = shlex.split(command)
        if not argv:
            return "ERROR: Empty command."
        name = argv[0].lower()
        if name == "git" and len(argv) >= 2 and argv[1] in {"status", "diff", "log"}:
            if any(arg.startswith(("--git-dir", "--work-tree", "-C")) for arg in argv[2:]):
                return "ERROR: Git path overrides are not allowed."
        elif name == "pytest":
            pass
        elif name in {"python", "py"} and argv[1:3] == ["-m", "pytest"]:
            pass
        else:
            return "ERROR: Allowed commands are pytest, python -m pytest, and git status/diff/log."
        result = subprocess.run(argv, cwd=workspace(), capture_output=True, text=True, errors="replace", timeout=30, shell=False)
        output = (result.stdout + result.stderr)[:12_000]
        return f"exit_code={result.returncode}\n{output or '[no output]'}"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"
