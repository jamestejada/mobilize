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
    @pytest.mark.parametrize("agent,model_env_var,expected_default_model,prompt_env_var,default_prompt_file", [
        ("PRAETOR",    "PRAETOR_MODEL",    "gemma4:latest", "COORDINATOR_PROMPT", "coordinator_gemma.md"),
        ("EXPLORATOR", "EXPLORATOR_MODEL", "gemma4:latest", "EXPLORATOR_PROMPT", "explorator_gemma.md"),
        ("TABULARIUS", "TABULARIUS_MODEL", "gemma4:latest", "TABULARIUS_PROMPT", "tabularius_gemma.md"),
        ("NUNTIUS",    "NUNTIUS_MODEL",    "gemma4:latest", "WRITER_PROMPT", "writer_gemma.md"),
        ("COGITATOR",  "COGITATOR_MODEL",  "qwen3:14b", "REFLECTION_PROMPT", "reflection.md"),
        ("PROBATOR",   "PROBATOR_MODEL",   "gemma4:latest", "GAP_ANALYSIS_PROMPT", "gap_analysis_gemma.md"),
    ])
    def test_default_model_and_prompt_file(
        self, agent, model_env_var, expected_default_model, prompt_env_var, default_prompt_file
    ):
        env = {
            k: v for k, v in os.environ.items()
            if k not in {model_env_var, prompt_env_var}
        }
        with patch.dict(os.environ, env, clear=True):
            import src.settings as settings
            import src.agent_settings as agent_settings
            importlib.reload(settings)
            importlib.reload(agent_settings)
            actual = getattr(agent_settings.AgentsConfiguration, agent)
            assert actual.model == expected_default_model
            assert actual.prompt_file == os.getenv(prompt_env_var, default_prompt_file)

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

    @pytest.mark.parametrize("agent,env_var", [
        ("PRAETOR", "COORDINATOR_PROMPT"),
        ("NUNTIUS", "WRITER_PROMPT"),
    ])
    def test_custom_prompt_loaded(self, agent, env_var):
        custom_prompt = "coordinator.md" if agent == "PRAETOR" else "writer.md"
        with patch.dict(os.environ, {env_var: custom_prompt}):
            import src.settings as settings
            import src.agent_settings as agent_settings
            importlib.reload(settings)
            importlib.reload(agent_settings)
            actual = getattr(agent_settings.AgentsConfiguration, agent).prompt_file
            assert actual == custom_prompt


class TestAgentSettingsModelSettings:
    def test_model_settings_shape(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", prompt_file="writer.md", think=False, temperature=0.5)
        ms = s.model_settings
        assert "extra_body" in ms
        assert "options" in ms["extra_body"]
        opts = ms["extra_body"]["options"]
        assert "temperature" in opts
        assert "top_p" in opts
        assert "num_ctx" in opts

    def test_think_false_adds_think_key(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", prompt_file="writer.md", think=False, temperature=0.1)
        assert s.model_settings["extra_body"]["think"] is False

    def test_think_true_omits_think_key(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", prompt_file="writer.md", think=True, temperature=0.1)
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
                prompt_file="writer.md",
                think=False,
                temperature=0.1,
                num_ctx=settings.OLLAMA_NUM_CTX,
            )
            assert s.model_settings["extra_body"]["options"]["num_ctx"] == 4096

    def test_instructions_read_from_prompt_file(self):
        from src.agent_settings import AgentSettings
        s = AgentSettings(model="x", prompt_file="writer.md", think=False, temperature=0.1)
        assert s.instructions.startswith("# ")
