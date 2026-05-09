"""OpenAI-compatible API types.

Port of src/api-types.hpp from distributed-llama-reference.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class ToolChoiceKind(IntEnum):
    TOOL_CHOICE_AUTO = 0
    TOOL_CHOICE_NONE = 1
    TOOL_CHOICE_REQUIRED = 2
    TOOL_CHOICE_NAMED = 3


@dataclass
class ToolFunctionDef:
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass
class Tool:
    type: str = "function"
    function: ToolFunctionDef = field(default_factory=ToolFunctionDef)


@dataclass
class ToolCallFunction:
    name: str = ""
    arguments: str = ""


@dataclass
class ToolCall:
    id: str = ""
    type: str = "function"
    function: ToolCallFunction = field(default_factory=ToolCallFunction)


@dataclass
class ToolCallDelta:
    id: str = ""
    type: str = "function"
    function: ToolCallFunction = field(default_factory=ToolCallFunction)
    has_function: bool = False


@dataclass
class ToolChoice:
    kind: ToolChoiceKind = ToolChoiceKind.TOOL_CHOICE_NONE
    tool_name: str = ""


@dataclass
class ChatMessage:
    role: str = ""
    content: str = ""
    tool_call_id: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class ChatMessageDelta:
    role: str = ""
    content: str = ""
    tool_calls: List[ToolCallDelta] = field(default_factory=list)
    has_tool_calls: bool = False


@dataclass
class Choice:
    index: int = 0
    message: ChatMessage = field(default_factory=ChatMessage)
    finish_reason: str = ""


@dataclass
class ChunkChoice:
    index: int = 0
    delta: ChatMessageDelta = field(default_factory=ChatMessageDelta)
    has_delta: bool = False
    finish_reason: str = ""


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletionChunk:
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[ChunkChoice] = field(default_factory=list)


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: ChatUsage = field(default_factory=ChatUsage)


@dataclass
class Model:
    id: str = ""
    object: str = "model"
    created: int = 0
    owned_by: str = ""


@dataclass
class ModelList:
    object: str = "list"
    data: List[Model] = field(default_factory=list)


@dataclass
class InferenceParams:
    messages: List[ChatMessage] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    stop: List[str] = field(default_factory=list)
    stream: bool = False
    seed: int = 0
    tools: List[Tool] = field(default_factory=list)
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
