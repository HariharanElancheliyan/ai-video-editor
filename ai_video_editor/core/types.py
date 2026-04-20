from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    function: dict[str, Any] | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolParameter(BaseModel):
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: Message
    done: bool = False
    total_duration: int | None = None