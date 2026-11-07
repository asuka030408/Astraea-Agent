"""Exercise the local API, model, and a file tool with a temporary workspace."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.web import app


def main() -> None:
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
        client = TestClient(app)
        session = client.post("/api/session")
        assert session.status_code == 200, session.text
        response = client.post("/api/chat", json={
            "session_id": session.json()["session_id"],
            "message": "Use write_file to create hello.txt containing exactly ASTRAEA_OK, then answer Done.",
        })
        assert response.status_code == 200, response.text
        assert (Path(directory) / "hello.txt").read_text(encoding="utf-8") == "ASTRAEA_OK"
        assert any(item["name"] == "write_file" for item in response.json()["events"])
        print("PASS: web API, model tool call, file write, and tool trace.")


if __name__ == "__main__":
    main()
