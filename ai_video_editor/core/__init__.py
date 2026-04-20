from .ollama_client import OllamaClient
from .types import BaseLLMClient, Message, ChatResponse, ToolDefinition
from ..config.settings import Settings


def create_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Factory: return the appropriate LLM client based on settings.llm_provider."""
    settings = settings or Settings()
    if settings.llm_provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(settings=settings)
    if settings.llm_provider == "google":
        from .google_client import GoogleGenAIClient
        return GoogleGenAIClient(settings=settings)
    return OllamaClient(settings=settings)


__all__ = [
    "BaseLLMClient",
    "OllamaClient",
    "Message",
    "ChatResponse",
    "ToolDefinition",
    "create_llm_client",
]