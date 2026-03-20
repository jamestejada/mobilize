"""Tests for src/settings.py — verifies env-var loading and class structure."""
import os
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


class TestTelegramBotCredentials:
    def test_token_read_from_env(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test:token999"}):
            # Reload to pick up patched env
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert settings.TelegramBotCredentials.TOKEN == "test:token999"

    def test_channel_id_cast_to_int(self):
        with patch.dict(os.environ, {"BOT_CHANNEL_ID": "-100123456789"}):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert settings.TelegramBotCredentials.CHANNEL_ID == -100123456789
            assert isinstance(settings.TelegramBotCredentials.CHANNEL_ID, int)

    def test_allowed_user_ids_parsed_as_set_of_ints(self):
        with patch.dict(os.environ, {"ALLOWED_USER_IDS": "111,222,333"}):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert settings.TelegramBotCredentials.ALLOWED_USER_IDS == {111, 222, 333}

    def test_allowed_user_ids_empty_string_gives_empty_set(self):
        with patch.dict(os.environ, {"ALLOWED_USER_IDS": ""}, clear=False):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert settings.TelegramBotCredentials.ALLOWED_USER_IDS == set()

    def test_allowed_user_ids_whitespace_stripped(self):
        with patch.dict(os.environ, {"ALLOWED_USER_IDS": " 42 , 99 "}):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert 42 in settings.TelegramBotCredentials.ALLOWED_USER_IDS
            assert 99 in settings.TelegramBotCredentials.ALLOWED_USER_IDS


def reload_settings_without_dotenv(**env_overrides):
    """Reload settings with load_dotenv mocked out, using only env_overrides."""
    import importlib
    import src.settings as settings
    base_env = {k: v for k, v in os.environ.items()}
    base_env.update(env_overrides)
    with patch.dict(os.environ, base_env, clear=True), \
         patch("src.settings.load_dotenv"):
        importlib.reload(settings)
    return settings


REQUIRED_URLS = {
    "OLLAMA_ROOT_URL": "http://localhost:11434",
    "MOBILIZE_US_ROOT_URL": "http://localhost:8080",
    "BOT_CHANNEL_ID": "-100000001",
}


def stripped_env(*drop_keys: str) -> dict:
    """Build env dict suitable for reload: drop specified keys but keep required URLs."""
    env = {k: v for k, v in os.environ.items() if k not in drop_keys}
    env.update(REQUIRED_URLS)
    return env


class TestFECCredentials:
    def test_default_key_is_demo(self):
        import importlib
        import src.settings as settings
        with patch.dict(os.environ, stripped_env("FEC_API_KEY"), clear=True), \
             patch("dotenv.load_dotenv"):
            importlib.reload(settings)
            assert settings.FECCredentials.API_KEY == "DEMO_KEY"

    def test_custom_key_loaded(self):
        with patch.dict(os.environ, {"FEC_API_KEY": "my_real_key"}):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert settings.FECCredentials.API_KEY == "my_real_key"


class TestMaxHistory:
    def test_default_is_30(self):
        import importlib
        import src.settings as settings
        with patch.dict(os.environ, stripped_env("MAX_HISTORY"), clear=True), \
             patch("dotenv.load_dotenv"):
            importlib.reload(settings)
            assert settings.MAX_HISTORY == 30

    def test_custom_value_loaded(self):
        with patch.dict(os.environ, {"MAX_HISTORY": "50"}):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert settings.MAX_HISTORY == 50


class TestModels:
    def test_default_coordinate_model(self):
        import importlib
        import src.settings as settings
        with patch.dict(os.environ, stripped_env("RESEARCH_COORDINATE_MODEL"), clear=True), \
             patch("dotenv.load_dotenv"):
            importlib.reload(settings)
            assert settings.Models.Research_Coordinate == "qwen3:8b-q4_K_M"

    def test_default_reflect_write_model(self):
        import importlib
        import src.settings as settings
        with patch.dict(os.environ, stripped_env("REFLECT_WRITE_MODEL"), clear=True), \
             patch("dotenv.load_dotenv"):
            importlib.reload(settings)
            assert settings.Models.Reflect_Write == "qwen2.5:14b-instruct-q4_k_m"


class TestLogDir:
    def test_default_log_dir(self):
        env = {k: v for k, v in os.environ.items() if k != "LOG_DIRECTORY"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import src.settings as settings
            importlib.reload(settings)
            assert str(settings.LOG_DIR) == "logs"

    def test_log_dir_created(self):
        import importlib
        import src.settings as settings
        importlib.reload(settings)
        assert settings.LOG_DIR.exists()
