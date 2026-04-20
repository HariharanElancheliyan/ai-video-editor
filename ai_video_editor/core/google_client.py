import json
import logging
import uuid
from typing import Any, AsyncIterator

from ..config.settings import Settings
from .types import (
    BaseLLMClient,
    ChatResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class GoogleGenAIClient(BaseLLMClient):
    """LLM client backed by the Google Gen AI SDK (google-genai)."""

    def __init__(self, model: str | None = None, settings: Settings | None = None):
        try:
            from google import genai
            from google.genai import types as gtypes  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required for Google AI support. "
                "Install it with: pip install google-genai"
            )
        self.settings = settings or Settings()
        self.model = model or self.settings.default_model
        self._genai = genai
        self._gtypes = gtypes
        self._client = genai.Client(api_key=self.settings.google_api_key)
        logger.info("GoogleGenAIClient initialized (model=%s)", self.model)

    # ── internal helpers ─────────────────────────────────────────────────

    def _build_contents(self, messages: list[Message]) -> list[Any]:
        """Convert internal Message list to google-genai Content objects."""
        gtypes = self._gtypes
        contents: list[Any] = []
        system_parts: list[Any] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(gtypes.Part.from_text(text=msg.content))
                continue

            if msg.role == "user":
                contents.append(gtypes.Content(
                    role="user",
                    parts=[gtypes.Part.from_text(text=msg.content)],
                ))

            elif msg.role == "assistant":
                parts: list[Any] = []
                if msg.content:
                    parts.append(gtypes.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.function or {}
                        name = fn.get("name", "")
                        raw_args = fn.get("arguments", {})
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {}
                        parts.append(gtypes.Part.from_function_call(
                            name=name, args=raw_args,
                        ))
                if parts:
                    contents.append(gtypes.Content(role="model", parts=parts))

            elif msg.role == "tool":
                fn_name = msg.tool_call_id or "unknown"
                try:
                    result = json.loads(msg.content) if msg.content else {}
                except json.JSONDecodeError:
                    result = {"raw": msg.content}
                contents.append(gtypes.Content(
                    role="user",
                    parts=[gtypes.Part.from_function_response(
                        name=fn_name, response=result,
                    )],
                ))

        return contents, system_parts

    def _build_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        """Convert internal ToolDefinition list to google-genai Tool objects."""
        gtypes = self._gtypes
        declarations = []
        for t in tools:
            fn = t.function
            declarations.append(gtypes.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters_json_schema=fn.get("parameters", {}),
            ))
        return [gtypes.Tool(function_declarations=declarations)]

    @staticmethod
    def _parse_function_calls(response: Any) -> list[ToolCall]:
        """Extract ToolCall list from a google-genai response."""
        tool_calls: list[ToolCall] = []
        fc_list = getattr(response, "function_calls", None) or []
        for fc in fc_list:
            call_id = getattr(fc, "id", None) or uuid.uuid4().hex
            tool_calls.append(ToolCall(
                id=call_id,
                function={
                    "name": fc.name,
                    "arguments": dict(fc.args) if fc.args else {},
                },
            ))
        return tool_calls

    # ── BaseLLMClient interface ──────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> ChatResponse | AsyncIterator[ChatResponse]:
        gtypes = self._gtypes
        contents, system_parts = self._build_contents(messages)

        config_kwargs: dict[str, Any] = {}
        if system_parts:
            config_kwargs["system_instruction"] = system_parts
        if tools:
            config_kwargs["tools"] = self._build_tools(tools)
            config_kwargs["automatic_function_calling"] = (
                gtypes.AutomaticFunctionCallingConfig(disable=True)
            )

        config = gtypes.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        logger.debug(
            "Chat request: model=%s, contents=%d, tools=%d",
            self.model, len(contents), len(tools or []),
        )

        if stream:
            async def _stream() -> AsyncIterator[ChatResponse]:
                async for chunk in await self._client.aio.models.generate_content_stream(
                    model=self.model, contents=contents, config=config,
                ):
                    yield ChatResponse(
                        message=Message(
                            role="assistant",
                            content=chunk.text or "",
                        ),
                        done=False,
                    )
            return _stream()

        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config,
        )

        tool_calls = self._parse_function_calls(response)
        text = response.text if not tool_calls else (response.text or "")

        msg = Message(
            role="assistant",
            content=text or "",
            tool_calls=tool_calls or None,
        )
        chat_resp = ChatResponse(message=msg, done=not tool_calls)
        logger.debug(
            "Chat response: content_length=%d, tool_calls=%d",
            len(msg.content or ""), len(tool_calls),
        )
        return chat_resp

    async def generate(
        self,
        prompt: str,
        system: str = "",
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        gtypes = self._gtypes
        logger.debug("Generate request: model=%s, prompt_length=%d", self.model, len(prompt))

        parts: list[Any] = [gtypes.Part.from_text(text=prompt)]
        if images:
            for img_bytes in images:
                parts.append(gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        config = None
        if system:
            config = gtypes.GenerateContentConfig(
                system_instruction=system,
            )

        response = await self._client.aio.models.generate_content(
            model=self.model, contents=parts, config=config,
        )
        logger.debug("Generate response received")
        return {"response": response.text or "", "done": True}

    def list_models(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for m in self._client.models.list():
            result.append({
                "name": getattr(m, "name", str(m)),
                "display_name": getattr(m, "display_name", ""),
            })
        return result
