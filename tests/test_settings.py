"""Web settings overrides: precedence, secret masking, validation, runtime reload."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import astraea.config as config
import astraea.web as web
from astraea.config import get_config, load_settings, mask_secret, save_user_settings, user_settings
from astraea.memory import MemoryManager


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "settings.json"
        self.patcher = patch.object(config, "USER_SETTINGS_FILE", self.path)
        self.patcher.start()
        web.sessions.clear()
        web.reset_runtime()

    def tearDown(self):
        self.patcher.stop()
        self.temporary.cleanup()
        web.reset_runtime()
        web.sessions.clear()

    def test_overrides_win_empty_keeps_and_none_removes(self):
        save_user_settings({"ASTRAEA_API_KEY": "sk-test-123456", "ASTRAEA_MODEL": "custom-model"})
        self.assertEqual(get_config("ASTRAEA_API_KEY"), "sk-test-123456")
        self.assertEqual(get_config("ASTRAEA_MODEL"), "custom-model")
        save_user_settings({"ASTRAEA_API_KEY": "", "NOT_ALLOWED": "x"})
        self.assertEqual(get_config("ASTRAEA_API_KEY"), "sk-test-123456")
        self.assertNotIn("NOT_ALLOWED", user_settings())
        save_user_settings({"ASTRAEA_MODEL": None})
        self.assertNotIn("ASTRAEA_MODEL", user_settings())

    def test_mask_secret_never_exposes_full_key(self):
        self.assertEqual(mask_secret(""), "")
        self.assertNotIn("abcdefgh", mask_secret("sk-abcdefgh"))

    def test_get_settings_masks_secrets(self):
        save_user_settings({"ASTRAEA_API_KEY": "sk-super-secret-value"})
        data = TestClient(web.app).get("/api/settings").json()
        self.assertNotIn("sk-super-secret-value", json.dumps(data, ensure_ascii=False))
        self.assertIn("****", data["values"]["ASTRAEA_API_KEY"])
        self.assertTrue(data["configured"])
        self.assertIn("ASTRAEA_API_KEY", data["saved_keys"])

    def test_saving_reloads_runtime_and_model_settings(self):
        web._runtime = ("stale", "stale", "stale")
        response = TestClient(web.app).post("/api/settings", json={"values": {
            "ASTRAEA_API_KEY": "sk-new-key-999", "ASTRAEA_MODEL": "my-model"}})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(web._runtime)
        settings = load_settings()
        self.assertEqual(settings.api_key, "sk-new-key-999")
        self.assertEqual(settings.model, "my-model")

    def test_invalid_values_are_rejected(self):
        client = TestClient(web.app)
        bad_window = client.post("/api/settings", json={"values": {"ASTRAEA_CONTEXT_WINDOW": "abc"}})
        self.assertEqual(bad_window.status_code, 400)
        bad_url = client.post("/api/settings", json={"values": {"ASTRAEA_BASE_URL": "ftp://example.com"}})
        self.assertEqual(bad_url.status_code, 400)
        self.assertFalse(self.path.exists())

    def test_embedding_key_from_ui_reaches_memory_manager(self):
        save_user_settings({"ASTRAEA_EMBEDDING_API_KEY": "embed-test-key"})
        manager = MemoryManager(None, Path(self.temporary.name) / "memory",
                                embedder=lambda texts: [])
        self.assertEqual(manager.embedding_key, "embed-test-key")


if __name__ == "__main__":
    unittest.main()
