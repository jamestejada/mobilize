import os
from dataclasses import dataclass, field

from .settings import OLLAMA_NUM_CTX, OllamaEndpoints


@dataclass
class AgentSettings:
    model: str
    think: bool
    temperature: float
    top_p: float = 0.7
    num_ctx: int = field(default_factory=lambda: OLLAMA_NUM_CTX)
    repeat_penalty: float = 1.1
    kv_cache_type: str = "q8_0"

    @property
    def model_settings(self) -> dict:
        body: dict = {
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "kv_cache_type": self.kv_cache_type,
            }
        }
        if not self.think:
            body["think"] = False
        return {"extra_body": body}

    def make_model(self):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider
        from .ollama_transport import ollama_http_client
        return OpenAIChatModel(
            model_name=self.model,
            provider=OllamaProvider(
                base_url=str(OllamaEndpoints.CHAT),
                http_client=ollama_http_client,
            )
        )


class AgentsConfiguration:
    PRAETOR = AgentSettings(
            model=os.getenv("PRAETOR_MODEL", "qwen3.5:latest"),
            think=True,
            temperature=0.1
        )
    EXPLORATOR = AgentSettings(
            model=os.getenv("EXPLORATOR_MODEL", "qwen3:14b"),
            think=False,
            temperature=0.1
        )
    TABULARIUS = AgentSettings(
            model=os.getenv("TABULARIUS_MODEL", "qwen3:14b"),
            think=False,
            temperature=0.1
        )
    NUNTIUS = AgentSettings(
            model=os.getenv("NUNTIUS_MODEL", "qwen3:14b"),
            think=True,
            temperature=0.35,
            top_p=0.9
        )
    COGITATOR = AgentSettings(
            model=os.getenv("COGITATOR_MODEL", "qwen2.5:14b-instruct-q4_k_m"),
            think=False,
            temperature=0.1
        )
    PROBATOR = AgentSettings(
            model=os.getenv("PROBATOR_MODEL", "qwen3:14b"),
            think=True,
            temperature=0.1
        )
