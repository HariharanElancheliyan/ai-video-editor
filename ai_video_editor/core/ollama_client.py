import json
import logging
from typing import Any
import ollama
from ollama import AsyncClient
from ..config.settings import Settings
from .types import Message, ChatResponse, ToolDefinition

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.model = model or self.settings.default_model
        self._client = AsyncClient(host=self.settings.ollama_base_url)
        logger.info("OllamaClient initialized (model=%s, host=%s)", self.model, self.settings.ollama_base_url)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False
    ) -> Any:
        msg_dicts = [msg.model_dump(exclude_none=True) for msg in messages]
        
        tool_dicts = None
        if tools:
            tool_dicts = [t.model_dump(exclude_none=True) for t in tools]

        logger.debug("Chat request: model=%s, messages=%d, tools=%d", self.model, len(msg_dicts), len(tool_dicts or []))
        
        if stream:
            async def _stream():
                async for chunk in self._client.chat(
                    model=self.model,
                    messages=msg_dicts,
                    tools=tool_dicts,
                    stream=True
                ):
                    yield ChatResponse(**dict(chunk))
            return _stream()
        
        response = await self._client.chat(
            model=self.model,
            messages=msg_dicts,
            tools=tool_dicts
        )
        # ollama returns pydantic models; dump to plain dict for re-validation
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = dict(response)
        chat_resp = ChatResponse(**data)
        logger.debug("Chat response: content_length=%d, tool_calls=%d",
                      len(chat_resp.message.content or ""),
                      len(chat_resp.message.tool_calls or []))
        return chat_resp

    async def generate(
        self,
        prompt: str,
        system: str = "",
        images: list[bytes] | None = None
    ) -> dict[str, Any]:
        logger.debug("Generate request: model=%s, prompt_length=%d", self.model, len(prompt))
        result = await self._client.generate(
            model=self.model,
            prompt=prompt,
            system=system or None,
            images=images
        )
        logger.debug("Generate response received")
        return result

    def list_models(self) -> list[dict[str, Any]]:
        result = ollama.list()
        # Handle both old dict API and new pydantic model API
        if hasattr(result, "models"):
            models = result.models
        elif isinstance(result, dict):
            models = result.get("models", [])
        else:
            models = []
        return [m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else {"name": str(m)}) for m in models]


def create_tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any]
) -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function={
            "name": name,
            "description": description,
            "parameters": parameters
        }
    )