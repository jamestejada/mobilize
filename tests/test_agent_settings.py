"""Tests for src/agent_settings.py — verifies per-agent model defaults and model_settings shape."""
import importlib
import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def reload_agent_settings(**env_overrides):
    import src.settings as settings
    import src.agent_settings as agent_settings
    with patch.dict(os.environ, env_overrides, clear=False):
        importlib.reload(settings)
        importlib.reload(agent_settings)
    return agent_settings


class TestAgentsConfigurationDefaults:
    @pytest.mark.parametrize("agent,env_var,expected_default", [
        ("PRAETOR",    "PRAETOR_MODEL",    "qwen3.5:latest"),
        ("EXPLORATOR", "EXPLORATOR_MODEL", "qwen3:8b-q4_K_M"),
        ("TABULARIUS", "TABULARIUS_MODEL", "hermes3:8b"),
        ("NUNTIUS",    "NUNTIUS_MODEL",    "qwen3:14b"),
        ("COGITATOR",  "COGITATOR_MODEL",  "qwen3.5:latest"),
        ("PROBATOR",   "PROBATOR_MODEL",   "qwen3:14b"),
    ])
    def test_default_model(self, agent, env_var, expected_default):
        env = {k: v for k, v in os.environ.items() if k != env_var}
        with patch.dict(os.environ, env, clear=True):
            import src.settings as settings
            import src.agent_settings as agent_settings
            importlib.reload(settings)
            importlib.reload(agent_settings)
            actual = getattr(agent_settings.AgentsConfiguration, agent).model
            assert actual == expected_default

    @pytest.mark.parametrize("agent,env_var", [
        ("PRAETOR",    "PRAETOR_MODEL"),
        ("EXPLORATOR", "EXPLORATOR_MODEL"),
    ])
    def test_custom_model_loaded(self, agent, env_var):
        with patch.dict(os.environ, {env_var: "custom:model"}):
            import src.settings as settings
            import src.agent_settings as agent_settings
            importlib.reload(settings)
            importlib.reload(agent_settings)
            actual = getattr(agent_settings.AgentsConfiguration, agent).model
            assert actual == "custom:model"


class TestAgentSettingsModelSettings:
    def test_model_settings_shape(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", think=False, temperature=0.5)
        ms = s.model_settings
        assert "extra_body" in ms
        assert "options" in ms["extra_body"]
        opts = ms["extra_body"]["options"]
        assert "temperature" in opts
        assert "top_p" in opts
        assert "num_ctx" in opts

    def test_think_false_adds_think_key(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", think=False, temperature=0.1)
        assert s.model_settings["extra_body"]["think"] is False

    def test_think_true_omits_think_key(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", think=True, temperature=0.1)
        assert "think" not in s.model_settings["extra_body"]

    def test_num_ctx_from_env(self):
        env = {k: v for k, v in os.environ.items()}
        env["OLLAMA_CTX_SIZE"] = "4096"
        with patch.dict(os.environ, env, clear=True):
            import src.settings as settings
            import src.agent_settings as agent_settings
            importlib.reload(settings)
            importlib.reload(agent_settings)
            s = agent_settings.AgentSettings(
                model="x",
                think=False,
                temperature=0.1,
                num_ctx=settings.OLLAMA_NUM_CTX,
            )
            assert s.model_settings["extra_body"]["options"]["num_ctx"] == 4096
