"""Unit tests for Settings configuration."""
import os
import unittest
from unittest.mock import patch

from config import Settings


class TestConfigSettings(unittest.TestCase):
    """Tests for application settings and defaults."""

    def test_default_model_is_gemini_3_8_flash(self):
        """Verify the default model in Settings is Gemini 3.8 Flash."""
        settings = Settings()
        assert settings.agy_default_model == "Gemini 3.8 Flash"

    def test_default_model_env_override(self):
        """Verify the default model can be overridden by environment variable."""
        with patch.dict(os.environ, {"AGY_DEFAULT_MODEL": "Gemini 3.5 Pro"}):
            settings = Settings()
            assert settings.agy_default_model == "Gemini 3.5 Pro"

    def test_default_model_alias_override(self):
        """Verify the default model alias AGY_AGY_DEFAULT_MODEL works."""
        with patch.dict(os.environ, {"AGY_AGY_DEFAULT_MODEL": "Custom-Model"}):
            settings = Settings()
            assert settings.agy_default_model == "Custom-Model"
