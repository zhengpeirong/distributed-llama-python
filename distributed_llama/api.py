"""HTTP API server for distributed-llama-python.

Port of src/dllama-api.cpp — OpenAI-compatible API with streaming,
tool calling, prompt caching, and multi-content message support.

Uses Python stdlib http.server for zero-dependency operation.
"""

import json
import sys
import time
import struct
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict

from .inference import AppCliArgs, AppInferenceContext, run_inference_app
from .api_types import (
    ChatMessage, ChatUsage, ChatCompletion, Choice,
    ChatCompletionChunk, ChunkChoice, ChatMessageDelta,
    Tool, ToolChoice, ToolChoiceKind, InferenceParams,
    Model as ApiModel, ModelList,
)


# ======================================================================
# Content helpers (port of normalizeMessageContent / parseChatMessages)
# ======================================================================

def normalize_message_content(content) -> str:
    """Normalize message content from string, array, or object formats."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = ""
        for part in content:
            piece = ""
            if isinstance(part, str):
                piece = part
            elif isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype in ("text", "input_text") and "text" in part:
                    piece = str(part["text"])
                elif ptype == "text" and "content" in part:
                    piece = str(part["content"])
                elif "text" in part:
                    piece = str(part["text"])
            if not piece:
                continue
            if result and result[-1] != '\n':
                result += ' '
            result += piece
        return result
    if isinstance(content, dict):
        return json.dumps(content)
    return str(content)


def parse_chat_messages(messages: list) -> List[ChatMessage]:
    """Parse OpenAI-format chat messages including tool_calls."""
    result = []
    for item in messages:
        msg = ChatMessage(
            role=item.get("role", ""),
            content=normalize_message_content(item.get("content")),
            tool_call_id=item.get("tool_call_id", ""),
        )
        if "tool_calls" in item and isinstance(item["tool_calls"], list):
            from .api_types import ToolCall, ToolCallFunction
            for ci in item["tool_calls"]:
                fn_data = ci.get("function", {})
                call = ToolCall(
                    id=ci.get("id", ""),
                    type=ci.get("type", "function"),
                    function=ToolCallFunction(
                        name=fn_data.get("name", ""),
                        arguments=fn_data.get("arguments", "")
                        if isinstance(fn_data.get("arguments"), str)
                        else json.dumps(fn_data.get("arguments", {})),
                    ),
                )
                msg.tool_calls.append(call)
        result.append(msg)
    return result


# ======================================================================
# JSON fragment detection (port of tryFindJsonFragmentAtEnd)
# ======================================================================

def try_find_json_fragment_at_end(text: str):
    """Try to parse complete JSON, or find the last balanced {...} fragment."""
    import json as _json
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    # Walk backwards looking for balanced braces
    in_string = False
    depth = 0
    end = None
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == '"':
            backslashes = 0
            j = i
            while j > 0 and text[j - 1] == '\\':
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
            continue
        if in_string:
            continue
        if c == '}':
            if end is None:
                end = i
            depth += 1
        elif c == '{':
            depth -= 1
            if depth == 0:
                try:
                    return _json.loads(text[i:end + 1])
                except _json.JSONDecodeError:
                    break
    return None


# ======================================================================
# Tool helpers (port of tryParseToolCallsFromJson / tryBuildToolsSystemPrompt)
# ======================================================================

def try_parse_tool_calls_from_json(data) -> list:
    """Try to extract tool calls from parsed JSON."""
    from .api_types import ToolCall, ToolCallFunction

    def _valid(call_item):
        if not isinstance(call_item, dict):
            return False
        fn = call_item.get("function")
        if isinstance(fn, dict):
            return bool(fn.get("name"))
        return bool(call_item.get("name"))

    def _parse(call_item, index):
        fn = call_item.get("function", {})
        name = fn.get("name", "") or call_item.get("name", "")
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args)
        return ToolCall(
            id=call_item.get("id", f"call_{index + 1}"),
            type=call_item.get("type", "function"),
            function=ToolCallFunction(name=name, arguments=args),
        )

    if isinstance(data, dict):
        calls = data.get("tool_calls")
        if isinstance(calls, list):
            parsed = [_parse(c, i) for i, c in enumerate(calls) if _valid(c)]
            if parsed:
                return parsed
        func_call = data.get("function_call")
        if isinstance(func_call, dict) and _valid(func_call):
            return [_parse(func_call, 0)]
    return []


def build_tools_system_prompt(tools: list, choice: ToolChoice) -> str:
    """Build system prompt describing available tools."""
    tools_json = json.dumps([{
        "type": t.type,
        "function": {
            "name": t.function.name,
            "description": t.function.description,
            "parameters": t.function.parameters,
        },
    } for t in tools])

    prompt = "You have access to the following tools:\n"
    prompt += tools_json
    prompt += "\n\nWhen you decide to call a tool, respond with a JSON object in this format:\n"
    prompt += '{"tool_calls":[{"id":"call_1","type":"function",'
    prompt += '"function":{"name":"tool_name","arguments":"{...}"}}]}'

    if choice.kind == ToolChoiceKind.TOOL_CHOICE_NONE:
        prompt += "\nDo not call any tools."
    elif choice.kind == ToolChoiceKind.TOOL_CHOICE_REQUIRED:
        prompt += "\nYou must call a tool."
    elif choice.kind == ToolChoiceKind.TOOL_CHOICE_NAMED:
        prompt += f"\nYou must call the tool named: {choice.tool_name}."

    return prompt


# ======================================================================
# NaiveCache (port of C++ NaiveCache)
# ======================================================================

class NaiveCache:
    """Caches prompt prefix positions across requests."""

    def __init__(self):
        self._cache: list = []  # list of (end_pos, ChatMessage)

    def push(self, end_pos: int, message: ChatMessage):
        self._cache.append((end_pos, message))

    def clear(self):
        self._cache.clear()

    def resolve_delta_prompt(self, messages: list) -> int:
        """Try to resolve cached prefix. Returns start_pos (0 if no cache)."""
        if not self._cache:
            return 0
        if len(messages) <= len(self._cache):
            return 0

        cache_len = len(self._cache)
        for i in range(cache_len):
            if (self._cache[i][1].role != messages[i].role or
                    self._cache[i][1].content != messages[i].content):
                return 0

        start_pos = self._cache[cache_len - 1][0]
        print(f"  Found naive cache for {cache_len} messages, pos={start_pos}")
        # Remove cached prefix from messages
        messages[:] = messages[cache_len:]
        return start_pos


# ======================================================================
# Inference params parsing
# ======================================================================

def parse_inference_params(body: dict, default_temp: float,
                           default_topp: float, default_seed: int) -> InferenceParams:
    """Parse OpenAI-compatible request body into InferenceParams."""
    params = InferenceParams()
    params.temperature = body.get("temperature", default_temp)
    params.top_p = body.get("top_p", default_topp)
    params.seed = body.get("seed", default_seed)
    params.stream = body.get("stream", False)
    params.max_tokens = body.get("max_tokens", 256)
    params.messages = parse_chat_messages(body.get("messages", []))

    # Stop sequences
    stop = body.get("stop")
    if isinstance(stop, str):
        params.stop = [stop]
    elif isinstance(stop, list):
        params.stop = stop
    else:
        params.stop = ["<|eot_id|>"]

    # Tools
    tools_data = body.get("tools")
    if isinstance(tools_data, list):
        for t in tools_data:
            fn_data = t.get("function", {})
            from .api_types import ToolFunctionDef
            tool = Tool(
                type=t.get("type", "function"),
                function=ToolFunctionDef(
                    name=fn_data.get("name", ""),
                    description=fn_data.get("description", ""),
                    parameters=fn_data.get("parameters", {}),
                ),
            )
            params.tools.append(tool)
        if params.tools:
            params.tool_choice.kind = ToolChoiceKind.TOOL_CHOICE_AUTO

    # Tool choice
    tc = body.get("tool_choice")
    if isinstance(tc, str):
        if tc == "none":
            params.tool_choice.kind = ToolChoiceKind.TOOL_CHOICE_NONE
        elif tc == "required":
            params.tool_choice.kind = ToolChoiceKind.TOOL_CHOICE_REQUIRED
        else:
            params.tool_choice.kind = ToolChoiceKind.TOOL_CHOICE_AUTO
    elif isinstance(tc, dict):
        fn = tc.get("function", {})
        if tc.get("type") == "function" and fn.get("name"):
            params.tool_choice.kind = ToolChoiceKind.TOOL_CHOICE_NAMED
            params.tool_choice.tool_name = fn["name"]

    return params


# ======================================================================
# HTTP request handler
# ======================================================================

class LlamaAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for LLM inference API."""

    inference_ctx: Optional[AppInferenceContext] = None
    lock = threading.Lock()

    def log_message(self, format, *args):
        sys.stderr.write(f"  {format % args}\n")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _start_sse_stream(self):
        """Send SSE stream headers and initial chunk."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_sse_chunk(self, data: str):
        chunk = data.encode("utf-8")
        header = f"{len(chunk):x}\r\n".encode("utf-8")
        self.wfile.write(header + chunk + b"\r\n")

    def _end_sse_stream(self):
        self.wfile.write(b"0\r\n\r\n")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        if self.path == "/v1/completions":
            self._handle_completions()
        elif self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_models(self):
        ctx = self.inference_ctx
        model_id = "distributed-llama"
        if ctx and ctx.header:
            model_id = f"llama-{ctx.header.dim}d-{ctx.header.n_layers}l"
        model = ApiModel(id=model_id)
        model_list = ModelList(data=[model])
        self._send_json({
            "object": model_list.object,
            "data": [{"id": m.id, "object": m.object,
                      "created": m.created, "owned_by": m.owned_by}
                     for m in model_list.data],
        })

    def _handle_completions(self):
        ctx = self.inference_ctx
        if ctx is None:
            self._send_json({"error": "Model not loaded"}, 503)
            return

        body = self._read_body()
        prompt = normalize_message_content(body.get("prompt", ""))
        params = parse_inference_params(
            body, ctx.args.temperature, ctx.args.topp, ctx.args.seed,
        )
        max_tokens = params.max_tokens or ctx.args.steps
        stream = params.stream

        with self.lock:
            ctx.sampler.set_temp(params.temperature)
            if params.seed:
                ctx.sampler.set_seed(params.seed)

            encode_result = ctx.tokenizer.encode(
                prompt, is_start=True, add_special_tokens=True,
            )
            tokens = encode_result
            n_input = len(tokens)

            # Eval phase
            pos = 0
            ctx.inference.set_batch_size(1)
            for pos in range(n_input - 1):
                ctx.inference.set_position(pos)
                ctx.inference.set_token(0, tokens[pos])
                ctx.inference.forward()
            pos = n_input - 1

            ctx.tokenizer.reset_decoder()
            ctx.inference.set_batch_size(1)

            # Generate
            generated = ""
            token = tokens[pos] if tokens else 0
            completion_tokens = 0

            if stream:
                self._start_sse_stream()

            stop_sequences = params.stop

            for _ in range(max_tokens):
                ctx.inference.set_position(pos)
                ctx.inference.set_token(0, token)
                ctx.inference.forward()

                logits = ctx.inference.logits_pipe
                token = ctx.sampler.sample(logits.tolist())

                if ctx.tokenizer.is_eos(token):
                    break

                piece = ctx.tokenizer.decode(token)
                if piece:
                    generated += piece

                    if stream:
                        chunk = ChatCompletionChunk(
                            id=f"cmpl-{int(time.time())}",
                            model="distributed-llama",
                            created=int(time.time()),
                            choices=[ChunkChoice(
                                index=0,
                                delta=ChatMessageDelta(content=piece),
                                has_delta=True,
                            )],
                        )
                        self._write_sse_chunk(
                            f"data: {json.dumps(_chunk_to_dict(chunk))}\n\n"
                        )

                completion_tokens += 1
                pos += 1

                # Check stop sequences
                for stop_seq in stop_sequences:
                    if stop_seq and stop_seq in generated:
                        break
                else:
                    if pos < ctx.header.seq_len:
                        continue
                break

            if stream:
                chunk = ChatCompletionChunk(
                    id=f"cmpl-{int(time.time())}",
                    model="distributed-llama",
                    created=int(time.time()),
                    choices=[ChunkChoice(index=0, finish_reason="stop")],
                )
                self._write_sse_chunk(
                    f"data: {json.dumps(_chunk_to_dict(chunk))}\n\n"
                )
                self._write_sse_chunk("data: [DONE]\n\n")
                self._end_sse_stream()
            else:
                self._send_json({
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "distributed-llama",
                    "choices": [{
                        "text": generated,
                        "index": 0,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": n_input,
                        "completion_tokens": completion_tokens,
                        "total_tokens": n_input + completion_tokens,
                    },
                })

    def _handle_chat_completions(self):
        ctx = self.inference_ctx
        if ctx is None:
            self._send_json({"error": "Model not loaded"}, 503)
            return

        body = self._read_body()
        params = parse_inference_params(
            body, ctx.args.temperature, ctx.args.topp, ctx.args.seed,
        )
        max_tokens = params.max_tokens or ctx.args.steps

        from .tokenizer import ChatItem, ChatTemplateGenerator, TokenizerChatStops, EosDetector

        with self.lock:
            ctx.sampler.set_temp(params.temperature)
            if params.seed:
                ctx.sampler.set_seed(params.seed)

            # Build tool system prompt if needed
            messages = params.messages
            if params.tools:
                tool_prompt = build_tools_system_prompt(
                    params.tools, params.tool_choice,
                )
                messages.insert(0, ChatMessage(role="system", content=tool_prompt))

            # Naive cache resolution
            naive_cache = getattr(self.inference_ctx, '_naive_cache', None)
            if naive_cache is None:
                naive_cache = NaiveCache()
                self.inference_ctx._naive_cache = naive_cache

            # Work on copies for cache resolution
            msg_copies = [ChatMessage(role=m.role, content=m.content)
                          for m in messages]
            start_pos = naive_cache.resolve_delta_prompt(msg_copies)
            delta_messages = msg_copies if start_pos > 0 else messages

            stops = TokenizerChatStops(ctx.tokenizer)
            template_gen = ChatTemplateGenerator(
                ctx.args.chat_template_type,
                ctx.tokenizer.chat_template,
                stops.stops[0] if stops.stops else "",
            )

            eos_tokens = ctx.tokenizer.eos_token_ids
            eos_pieces = [ctx.tokenizer.vocab[t] for t in eos_tokens]
            eos_detector = EosDetector(
                len(eos_tokens), eos_tokens,
                eos_pieces if eos_pieces else [""],
                stops.max_stop_length, stops.max_stop_length,
            )

            input_items = [ChatItem(m.role, m.content) for m in delta_messages]
            content, public_prompt = template_gen.generate(input_items, True)

            is_start = start_pos == 0
            encode_result = ctx.tokenizer.encode(
                content, is_start=is_start, add_special_tokens=True,
            )
            tokens = encode_result
            n_input = len(tokens)

            # Eval phase
            prompt_end_pos = start_pos + n_input - 1
            if prompt_end_pos > ctx.header.seq_len:
                prompt_end_pos = ctx.header.seq_len
            max_pred_pos = (prompt_end_pos + max_tokens
                            if max_tokens > 0 else ctx.header.seq_len)
            if max_pred_pos > ctx.header.seq_len:
                max_pred_pos = ctx.header.seq_len

            # Push to cache
            for msg in messages:
                naive_cache.push(prompt_end_pos, msg)

            pos = start_pos
            i = 0
            while True:
                remaining = prompt_end_pos - pos
                if remaining <= 0:
                    break
                batch_size = min(remaining, ctx.args.n_batches)

                ctx.inference.set_batch_size(batch_size)
                ctx.inference.set_position(pos)
                for j in range(batch_size):
                    ctx.inference.set_token(j, tokens[i + j])

                ctx.inference.forward()

                i += batch_size
                pos += batch_size
                token = tokens[i + 1] if i + 1 < len(tokens) else 0

            ctx.inference.set_batch_size(1)
            ctx.tokenizer.reset_decoder()
            eos_detector.reset()

            # Stream setup
            if params.stream:
                self._start_sse_stream()

            buffer = ""
            if public_prompt:
                if params.stream:
                    chunk = ChatCompletionChunk(
                        id=f"chatcmpl-{int(time.time())}",
                        model="distributed-llama",
                        created=int(time.time()),
                        choices=[ChunkChoice(
                            index=0,
                            delta=ChatMessageDelta(
                                role="assistant", content=public_prompt,
                            ),
                            has_delta=True,
                        )],
                    )
                    self._write_sse_chunk(
                        f"data: {json.dumps(_chunk_to_dict(chunk))}\n\n"
                    )
                buffer += public_prompt

            # Prediction phase
            for pos in range(prompt_end_pos, max_pred_pos):
                ctx.inference.set_position(pos)
                ctx.inference.set_token(0, token)
                ctx.inference.forward()

                logits = ctx.inference.logits_pipe
                token = ctx.sampler.sample(logits.tolist())
                piece = ctx.tokenizer.decode(token) or ""

                eos_type = eos_detector.append(token, piece)
                if eos_type in (EosDetector.MAYBE_EOS, EosDetector.EOS):
                    delta = eos_detector.get_delta()
                    if delta:
                        if params.stream:
                            chunk = ChatCompletionChunk(
                                id=f"chatcmpl-{int(time.time())}",
                                model="distributed-llama",
                                created=int(time.time()),
                                choices=[ChunkChoice(
                                    index=0,
                                    delta=ChatMessageDelta(content=delta),
                                    has_delta=True,
                                )],
                            )
                            self._write_sse_chunk(
                                f"data: {json.dumps(_chunk_to_dict(chunk))}\n\n"
                            )
                        buffer += delta
                    eos_detector.reset()
                elif piece:
                    if params.stream:
                        chunk = ChatCompletionChunk(
                            id=f"chatcmpl-{int(time.time())}",
                            model="distributed-llama",
                            created=int(time.time()),
                            choices=[ChunkChoice(
                                index=0,
                                delta=ChatMessageDelta(content=piece),
                                has_delta=True,
                            )],
                        )
                        self._write_sse_chunk(
                            f"data: {json.dumps(_chunk_to_dict(chunk))}\n\n"
                        )
                    buffer += piece

                if eos_type == EosDetector.EOS:
                    break

            # Update cache
            reply = ChatMessage(role="assistant", content=buffer)
            if pos >= ctx.header.seq_len:
                naive_cache.clear()
            else:
                naive_cache.push(pos, reply)

            if params.stream:
                chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{int(time.time())}",
                    model="distributed-llama",
                    created=int(time.time()),
                    choices=[ChunkChoice(index=0, finish_reason="stop")],
                )
                self._write_sse_chunk(
                    f"data: {json.dumps(_chunk_to_dict(chunk))}\n\n"
                )
                self._write_sse_chunk("data: [DONE]\n\n")
                self._end_sse_stream()
            else:
                choice = Choice(
                    index=0,
                    message=reply,
                    finish_reason="stop",
                )

                # Tool call detection
                if params.tools:
                    parsed = try_find_json_fragment_at_end(buffer)
                    if parsed:
                        tool_calls = try_parse_tool_calls_from_json(parsed)
                        if tool_calls:
                            choice.message.tool_calls = tool_calls
                            choice.finish_reason = "tool_calls"

                completion_tokens = pos - prompt_end_pos
                usage = ChatUsage(
                    prompt_tokens=n_input,
                    completion_tokens=completion_tokens,
                    total_tokens=n_input + completion_tokens,
                )
                self._send_json({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "distributed-llama",
                    "choices": [_choice_to_dict(choice)],
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                })


def _choice_to_dict(choice: Choice) -> dict:
    msg = {
        "role": choice.message.role,
        "content": choice.message.content,
    }
    if choice.message.tool_calls:
        msg["tool_calls"] = [{
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        } for tc in choice.message.tool_calls]
    if choice.message.tool_call_id:
        msg["tool_call_id"] = choice.message.tool_call_id
    return {
        "index": choice.index,
        "message": msg,
        "finish_reason": choice.finish_reason,
    }


def _chunk_to_dict(chunk: ChatCompletionChunk) -> dict:
    result = {
        "id": chunk.id,
        "object": chunk.object,
        "created": chunk.created,
        "model": chunk.model,
        "choices": [],
    }
    for cc in chunk.choices:
        c = {"index": cc.index, "finish_reason": cc.finish_reason}
        if cc.has_delta:
            delta = {"role": cc.delta.role} if cc.delta.role else {}
            if cc.delta.content:
                delta["content"] = cc.delta.content
            if cc.delta.has_tool_calls and cc.delta.tool_calls:
                delta["tool_calls"] = [{
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                } for tc in cc.delta.tool_calls]
            c["delta"] = delta
        result["choices"].append(c)
    return result


# ======================================================================
# Setup and entry point
# ======================================================================

def _setup_inference_context(args: AppCliArgs) -> AppInferenceContext:
    """Set up model, tokenizer, and sampler for API serving."""
    from .model import load_llm_header, print_llm_header
    from .graph_builder import build_llm_net
    from .executor import (
        NnNetExecution, NnCpuDevice,
        NnExecutor, NnFakeNodeSynchronizer,
    )
    from .tokenizer import Tokenizer, Sampler

    print("Loading model...")
    header = load_llm_header(args.model_path, args.max_seq_len, args.sync_type)
    print_llm_header(header)

    n_nodes = args.n_workers + 1
    net = build_llm_net(header, n_nodes, args.n_batches)
    execution = NnNetExecution(args.n_threads, net.net_config)

    root_config = net.node_configs[0]
    device = NnCpuDevice(net.net_config, root_config, execution)
    synchronizer = NnFakeNodeSynchronizer()
    executor = NnExecutor(
        net.net_config, root_config, device,
        execution, synchronizer, False,
    )

    print("Loading weights...")
    from .inference import _load_weights
    _load_weights(args.model_path, net, executor, None)

    inference = RootLlmInference(net, execution, executor, None)
    tokenizer = Tokenizer(args.tokenizer_path)
    tokenizer.print_header()
    sampler = Sampler(header.vocab_size, args.temperature, args.topp, args.seed)

    ctx = AppInferenceContext()
    ctx.args = args
    ctx.header = header
    ctx.inference = inference
    ctx.tokenizer = tokenizer
    ctx.sampler = sampler
    return ctx


# Need import here to avoid circular dependency
from .inference import RootLlmInference


def run_api_server(args: AppCliArgs):
    """Start the HTTP API server."""
    ctx = _setup_inference_context(args)
    LlamaAPIHandler.inference_ctx = ctx

    server = HTTPServer((args.host, args.port), LlamaAPIHandler)
    print(f"API server listening on {args.host}:{args.port}")
    print(f"  POST /v1/completions")
    print(f"  POST /v1/chat/completions")
    print(f"  GET  /v1/models")
    print(f"  GET  /health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    import argparse
    from .inference import AppCliArgs

    parser = argparse.ArgumentParser(prog="dllama-api", description="OpenAI-compatible LLM API server")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer file")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--nthreads", type=int, default=1, help="Number of threads")
    parser.add_argument("--nbatches", type=int, default=32, help="Batch size for eval")
    parser.add_argument("--max-seq-len", type=int, default=0, help="Max sequence length")
    parser.add_argument("--buffer-float-type", default="q80", choices=["f32", "f16", "q40", "q80"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--topp", type=float, default=0.9)

    args_ns = parser.parse_args()

    from .quants import F_32, F_16, F_Q40, F_Q80
    sync_map = {"f32": F_32, "f16": F_16, "q40": F_Q40, "q80": F_Q80}
    cli_args = AppCliArgs()
    cli_args.model_path = args_ns.model
    cli_args.tokenizer_path = args_ns.tokenizer
    cli_args.host = args_ns.host
    cli_args.port = args_ns.port
    cli_args.n_threads = args_ns.nthreads
    cli_args.n_batches = args_ns.nbatches
    cli_args.max_seq_len = args_ns.max_seq_len
    cli_args.sync_type = sync_map.get(args_ns.buffer_float_type, F_Q80)
    cli_args.seed = args_ns.seed
    cli_args.temperature = args_ns.temperature
    cli_args.topp = args_ns.topp
    run_api_server(cli_args)


if __name__ == "__main__":
    main()
