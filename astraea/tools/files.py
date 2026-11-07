"""File tools confined to the configured workspace."""

from pathlib import Path

from langchain.tools import tool

from astraea.config import PROJECT_DIR, get_config


MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 100_000
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def workspace() -> Path:
    return Path(get_config("ASTRAEA_WORKSPACE", str(PROJECT_DIR / "workspace"))).expanduser().resolve()


def safe_path(relative_path: str) -> Path:
    root = workspace()
    path = (root / relative_path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path must name a file inside the workspace.")
    return path


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file in the workspace by relative path."""
    try:
        target = safe_path(path)
        content = target.read_text(encoding="utf-8")
        if len(content) > MAX_READ_CHARS:
            return content[:MAX_READ_CHARS] + "\n[Output truncated]"
        return content
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Create or replace a UTF-8 text file in the workspace by relative path."""
    try:
        if len(content) > MAX_WRITE_CHARS:
            return f"ERROR: Content exceeds {MAX_WRITE_CHARS} characters."
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {target.relative_to(workspace()).as_posix()} ({len(content)} characters)."
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def search_files(query: str) -> str:
    """Search workspace file names and UTF-8 text contents for a literal phrase."""
    if not query.strip():
        return "ERROR: Search query is empty."
    root = workspace()
    matches = []
    checked = 0
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.resolve().is_relative_to(root):
            continue
        checked += 1
        if checked > 2000:
            break
        relative = path.relative_to(root).as_posix()
        if query.casefold() in relative.casefold():
            matches.append(f"{relative} (file name)")
        try:
            if path.stat().st_size > 1_000_000:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query.casefold() in line.casefold():
                    matches.append(f"{relative}:{number}: {line[:200]}")
                    if len(matches) >= 50:
                        return "\n".join(matches)
        except (OSError, UnicodeError):
            continue
    return "\n".join(matches) if matches else "No matches found."
