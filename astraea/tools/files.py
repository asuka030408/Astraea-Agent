"""File tools confined to the configured workspace."""

from pathlib import Path
import json
import shutil
from uuid import uuid4

from langchain.tools import tool

from astraea.config import PROJECT_DIR, get_config


MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 100_000
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".astraea-trash"}


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


@tool
def list_files(directory: str = ".", limit: int = 100) -> str:
    """List files and folders beneath a workspace directory, with a bounded result."""
    try:
        root = workspace()
        target = root if directory in {"", "."} else safe_path(directory)
        if not target.is_dir():
            return "ERROR: Directory does not exist."
        entries = []
        for path in sorted(target.iterdir(), key=lambda p: p.name.casefold()):
            if path.name in SKIP_DIRS or not path.resolve().is_relative_to(root):
                continue
            entries.append({"path": path.relative_to(root).as_posix(), "type": "directory" if path.is_dir() else "file"})
            if len(entries) >= max(1, min(limit, 200)):
                break
        return json.dumps(entries, ensure_ascii=False)
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def file_info(path: str) -> str:
    """Show size, type and modification time for a workspace path."""
    try:
        target = safe_path(path)
        stat = target.stat()
        return json.dumps({"path": target.relative_to(workspace()).as_posix(),
                           "type": "directory" if target.is_dir() else "file",
                           "size_bytes": stat.st_size, "modified_at": stat.st_mtime}, ensure_ascii=False)
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def read_file_lines(path: str, start_line: int = 1, line_count: int = 80) -> str:
    """Read a bounded line range from a UTF-8 workspace text file."""
    try:
        target = safe_path(path)
        if target.stat().st_size > 2_000_000:
            return "ERROR: File is too large for line reading."
        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, start_line)
        end = start + max(1, min(line_count, 200)) - 1
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def append_file(path: str, content: str) -> str:
    """Append UTF-8 text to an existing workspace file."""
    try:
        if len(content) > MAX_WRITE_CHARS:
            return "ERROR: Content is too large."
        target = safe_path(path)
        if not target.is_file():
            return "ERROR: File does not exist."
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"Appended {len(content)} characters to {target.relative_to(workspace()).as_posix()}."
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one unique exact text occurrence in a UTF-8 workspace file."""
    try:
        if not old_text or len(new_text) > MAX_WRITE_CHARS:
            return "ERROR: Invalid replacement text."
        target = safe_path(path)
        if target.stat().st_size > 2_000_000:
            return "ERROR: File is too large for replacement."
        content = target.read_text(encoding="utf-8")
        if content.count(old_text) != 1:
            return "ERROR: The old text must occur exactly once."
        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Replaced text in {target.relative_to(workspace()).as_posix()}."
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def copy_file(source: str, destination: str) -> str:
    """Copy one workspace file to a new workspace path without overwriting."""
    try:
        src, dst = safe_path(source), safe_path(destination)
        if not src.is_file() or dst.exists():
            return "ERROR: Source must be a file and destination must not exist."
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return f"Copied {source} to {destination}."
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move one workspace file to a new workspace path without overwriting."""
    try:
        src, dst = safe_path(source), safe_path(destination)
        if not src.is_file() or dst.exists():
            return "ERROR: Source must be a file and destination must not exist."
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved {source} to {destination}."
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"


@tool
def delete_file(path: str) -> str:
    """Remove a workspace file by moving it to the workspace's recoverable trash."""
    try:
        target = safe_path(path)
        if not target.is_file():
            return "ERROR: File does not exist."
        trash = safe_path(".astraea-trash")
        trash.mkdir(exist_ok=True)
        destination = trash / f"{uuid4().hex}-{target.name}"
        shutil.move(str(target), str(destination))
        return f"Moved {path} to {destination.relative_to(workspace()).as_posix()} (recoverable)."
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"
