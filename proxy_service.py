import asyncio
import hmac
import json
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


class FairGate:
    """Small FIFO semaphore used to serialize or bound upstream generations."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.active = 0
        self.waiters: Deque[asyncio.Future] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        async with self.lock:
            self.waiters.append(waiter)
            self._wake_locked()
        try:
            await waiter
        except BaseException:
            async with self.lock:
                try:
                    self.waiters.remove(waiter)
                except ValueError:
                    if waiter.done() and not waiter.cancelled():
                        self.active = max(0, self.active - 1)
                self._wake_locked()
            raise

    async def release(self) -> None:
        async with self.lock:
            self.active = max(0, self.active - 1)
            self._wake_locked()

    def _wake_locked(self) -> None:
        while self.active < self.capacity and self.waiters:
            waiter = self.waiters.popleft()
            if waiter.cancelled():
                continue
            self.active += 1
            waiter.set_result(True)

    @property
    def queued(self) -> int:
        return sum(1 for waiter in self.waiters if not waiter.cancelled())


class InferenceProxy:
    def __init__(
        self,
        manager: Any,
        models: Dict[str, Dict[str, Any]],
        api_key_getter: Callable[[], str],
        mock_mode_getter: Callable[[], bool],
        mock_completion_factory: Callable[[dict], dict],
        auto_wake_getter: Callable[[], bool],
        relay_hub: Any = None,
    ):
        self.manager = manager
        self.models = models
        self.api_key_getter = api_key_getter
        self.mock_mode_getter = mock_mode_getter
        self.mock_completion_factory = mock_completion_factory
        self.auto_wake_getter = auto_wake_getter
        self.relay_hub = relay_hub
        self.gates = {1: FairGate(1), 3: FairGate(3)}
        self._client: Optional[httpx.AsyncClient] = None
        self._wake_lock = asyncio.Lock()
        self._wake_task: Optional[asyncio.Task] = None

    def closeable_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(600.0, connect=30.0, read=600.0, write=600.0, pool=600.0)
            limits = httpx.Limits(max_connections=32, max_keepalive_connections=16, keepalive_expiry=60.0)
            self._client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)
        return self._client

    def authenticate(self, request: Request) -> None:
        expected = self.api_key_getter()
        auth = request.headers.get("authorization", "")
        supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "").strip()
        if not expected or not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing Client API Key")

    def active_model(self) -> Dict[str, Any]:
        try:
            return self.manager.model_spec(self.manager.active_model_id)
        except Exception:
            return self.models.get(self.manager.active_model_id) or next(iter(self.models.values()))

    def normalize_request_body(self, body: dict) -> dict:
        """Use the active upstream alias and keep requested output inside context."""
        normalized = dict(body or {})
        normalized["model"] = self.manager.active_model_id
        model = self.active_model()
        context = int(model.get("runtime_context_length") or model.get("context_length") or 32768)
        prompt_chars = len(json.dumps(normalized.get("messages", normalized.get("prompt", "")), ensure_ascii=False))
        estimated_prompt_tokens = max(1, prompt_chars // 4)
        safe_max = max(1, context - estimated_prompt_tokens - 256)
        for key in ("max_tokens", "max_completion_tokens"):
            if key in normalized:
                try:
                    normalized[key] = max(1, min(int(normalized[key]), safe_max))
                except (TypeError, ValueError):
                    normalized[key] = min(4096, safe_max)
        if "max_tokens" not in normalized and "max_completion_tokens" not in normalized:
            normalized["max_tokens"] = min(4096, safe_max)
        return normalized

    def gate(self) -> FairGate:
        capacity = int(self.active_model().get("max_concurrency", 1))
        if capacity not in self.gates:
            self.gates[capacity] = FairGate(capacity)
        return self.gates[capacity]

    def queue_state(self) -> Dict[str, int]:
        gate = self.gate()
        return {"capacity": gate.capacity, "active": gate.active, "queued": gate.queued}

    async def _wake_and_wait(self) -> str:
        if self.manager.tunnel_url:
            return self.manager.tunnel_url
        if self.manager.status in {"idle", "error", "quota_wait"}:
            result = await asyncio.to_thread(self.manager.push_kernel, self.manager.active_model_id)
            if not result.get("success") and not result.get("failover_scheduled"):
                raise HTTPException(
                    status_code=503,
                    detail=str(result.get("error") or "GPU auto-wake failed"),
                    headers=self.manager.warmup_headers(),
                )
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            if self.manager.tunnel_url:
                return self.manager.tunnel_url
            if self.manager.status == "error" and not (self.manager._failover_thread and self.manager._failover_thread.is_alive()):
                raise HTTPException(status_code=503, detail=self.manager.status_detail, headers=self.manager.warmup_headers())
            await asyncio.sleep(2.0)
        raise HTTPException(
            status_code=504,
            detail="Accelerator warm-up exceeded 15 minutes",
            headers=self.manager.warmup_headers(),
        )

    async def ensure_tunnel(self) -> str:
        if self.manager.tunnel_url:
            return self.manager.tunnel_url
        if not self.auto_wake_getter():
            raise HTTPException(
                status_code=503,
                detail=f"Kaggle accelerator is not ready ({self.manager.status})",
                headers=self.manager.warmup_headers(),
            )
        async with self._wake_lock:
            if self.manager.tunnel_url:
                return self.manager.tunnel_url
            if self._wake_task is None or self._wake_task.done():
                self._wake_task = asyncio.create_task(self._wake_and_wait())
            task = self._wake_task
        return await asyncio.wait_for(asyncio.shield(task), timeout=905.0)

    def upstream_headers(self) -> Dict[str, str]:
        key = self.manager.upstream_api_key or self.api_key_getter()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
            headers["x-api-key"] = key
        return headers

    async def _begin(self) -> tuple[FairGate, str]:
        self.manager.note_request_queued()
        gate = None
        acquired = False
        try:
            await self.ensure_tunnel()
            gate = self.gate()
            await asyncio.wait_for(gate.acquire(), timeout=600.0)
            acquired = True
            upstream_url = self.manager.reserve_upstream()
        except asyncio.TimeoutError as exc:
            self.manager.note_request_aborted()
            raise HTTPException(status_code=504, detail="Inference queue wait exceeded 600 seconds") from exc
        except BaseException:
            if acquired and gate:
                await gate.release()
            self.manager.note_request_aborted()
            raise
        return gate, upstream_url

    async def _finish(self, gate: FairGate, upstream_url: str) -> None:
        self.manager.release_upstream(upstream_url)
        await gate.release()

    async def _open_with_retry(self, path: str, body: dict, is_stream: bool):
        transient = {502, 503, 504}
        last_error = None
        for attempt in range(4):
            gate, upstream_url = await self._begin()
            client = self.closeable_client()
            upstream = f"{upstream_url.rstrip('/')}{path}"
            try:
                if upstream_url.startswith("relay://"):
                    if not self.relay_hub:
                        raise ConnectionError("Outbound relay is not configured")
                    session_id = upstream_url.split("://", 1)[1]
                    relayed = await self.relay_hub.submit(
                        session_id, path, body, self.upstream_headers(), timeout=600.0
                    )
                    relay_content = relayed.get("content") or b""
                    relay_response_args = {
                        "status_code": int(relayed.get("status_code") or 502),
                        "headers": relayed.get("headers") or {"content-type": "application/json"},
                    }
                    if is_stream:
                        relay_response_args["stream"] = httpx.ByteStream(relay_content)
                    else:
                        relay_response_args["content"] = relay_content
                    response = httpx.Response(**relay_response_args)
                elif is_stream:
                    request = client.build_request("POST", upstream, json=body, headers=self.upstream_headers())
                    response = await client.send(request, stream=True)
                else:
                    response = await client.post(upstream, json=body, headers=self.upstream_headers())
            except (httpx.TransportError, OSError, ConnectionError, asyncio.TimeoutError) as exc:
                last_error = exc
                await self._finish(gate, upstream_url)
                if attempt >= 3:
                    raise
                await asyncio.sleep(2)
                continue
            if response.status_code in transient and attempt < 3:
                if is_stream:
                    await response.aclose()
                await self._finish(gate, upstream_url)
                await asyncio.sleep(2)
                continue
            return response, gate, upstream_url
        raise last_error or RuntimeError("Upstream retry attempts exhausted")

    async def raw_proxy(self, path: str, body: dict, is_stream: bool) -> Response:
        body = self.normalize_request_body(body)
        response, gate, upstream_url = await self._open_with_retry(path, body, is_stream)
        if not is_stream:
            try:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type", "application/json"),
                )
            finally:
                await self._finish(gate, upstream_url)

        async def stream():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await self._finish(gate, upstream_url)

        media_type = response.headers.get("content-type", "text/event-stream").split(";", 1)[0]
        return StreamingResponse(
            stream(),
            status_code=response.status_code,
            media_type=media_type,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def chat(self, request: Request) -> Response:
        self.authenticate(request)
        body = await request.json()
        body = self.normalize_request_body(body)
        is_stream = bool(body.get("stream", False))
        body.setdefault("presence_penalty", 0.3)
        body.setdefault("frequency_penalty", 0.3)
        body.setdefault("repeat_penalty", 1.15)
        if self.mock_mode_getter():
            return self._mock_openai(body, is_stream)
        try:
            return await self.raw_proxy("/v1/chat/completions", body, is_stream)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to communicate with Kaggle upstream: {exc}") from exc

    async def completions(self, request: Request) -> Response:
        self.authenticate(request)
        body = await request.json()
        body = self.normalize_request_body(body)
        is_stream = bool(body.get("stream", False))
        if self.mock_mode_getter():
            prompt = body.get("prompt", "")
            mock = self.mock_completion_factory({"model": body.get("model"), "messages": [{"role": "user", "content": prompt}]})
            text = mock["choices"][0]["message"]["content"]
            return JSONResponse({
                "id": mock["id"], "object": "text_completion", "created": mock["created"], "model": mock["model"],
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}], "usage": mock["usage"],
            })
        return await self.raw_proxy("/v1/completions", body, is_stream)

    def _mock_openai(self, body: dict, is_stream: bool) -> Response:
        mock = self.mock_completion_factory(body)
        if not is_stream:
            return JSONResponse(mock)

        async def stream():
            for word in mock["choices"][0]["message"]["content"].split(" "):
                chunk = {
                    "id": mock["id"], "object": "chat.completion.chunk", "created": mock["created"], "model": mock["model"],
                    "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'id': mock['id'], 'object': 'chat.completion.chunk', 'created': mock['created'], 'model': mock['model'], 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @staticmethod
    def anthropic_to_openai(body: dict) -> Dict[str, Any]:
        messages = []
        system = body.get("system")
        if system:
            if isinstance(system, list):
                system = "\n".join(item.get("text", "") for item in system if isinstance(item, dict) and item.get("type") == "text")
            messages.append({"role": "system", "content": str(system)})
        for message in body.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            messages.append({"role": message.get("role", "user"), "content": content})
        result = {
            "model": body.get("model"),
            "messages": messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 4096),
            "stream": bool(body.get("stream", False)),
            "presence_penalty": body.get("presence_penalty", 0.3),
            "frequency_penalty": body.get("frequency_penalty", 0.3),
            "repeat_penalty": body.get("repeat_penalty", 1.15),
        }
        if "top_p" in body:
            result["top_p"] = body["top_p"]
        if "stop_sequences" in body:
            result["stop"] = body["stop_sequences"]
        return result

    async def messages(self, request: Request) -> Response:
        self.authenticate(request)
        body = await request.json()
        body = self.normalize_request_body(body)
        is_stream = bool(body.get("stream", False))
        if self.active_model().get("native_anthropic") and not self.mock_mode_getter():
            return await self.raw_proxy("/v1/messages", body, is_stream)
        openai_body = self.anthropic_to_openai(body)
        model_name = body.get("model") or self.manager.active_model_id
        message_id = f"msg_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if self.mock_mode_getter():
            mock = self.mock_completion_factory(openai_body)
            return self._anthropic_mock(message_id, model_name, mock["choices"][0]["message"]["content"], is_stream)
        if is_stream:
            return await self._anthropic_stream(message_id, model_name, openai_body)
        response, gate, upstream_url = await self._open_with_retry("/v1/chat/completions", openai_body, False)
        try:
            if response.status_code != 200:
                return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))
            data = response.json()
            choice = data.get("choices", [{}])[0]
            content = (choice.get("message") or {}).get("content", "")
            return JSONResponse({
                "id": message_id, "type": "message", "role": "assistant", "model": model_name,
                "content": [{"type": "text", "text": content}],
                "stop_reason": "max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": data.get("usage", {}).get("prompt_tokens", 0), "output_tokens": data.get("usage", {}).get("completion_tokens", 0)},
            })
        finally:
            await self._finish(gate, upstream_url)

    async def _anthropic_stream(self, message_id: str, model_name: str, openai_body: dict) -> Response:
        response, gate, upstream_url = await self._open_with_retry("/v1/chat/completions", openai_body, True)

        async def stream():
            output_tokens = 0
            try:
                yield self._anthropic_event("message_start", {"type": "message_start", "message": {"id": message_id, "type": "message", "role": "assistant", "content": [], "model": model_name, "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
                yield self._anthropic_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                        text = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except Exception:
                        text = ""
                    if text:
                        output_tokens += 1
                        yield self._anthropic_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
                yield self._anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                yield self._anthropic_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": output_tokens}})
                yield self._anthropic_event("message_stop", {"type": "message_stop"})
            finally:
                await response.aclose()
                await self._finish(gate, upstream_url)

        return StreamingResponse(stream(), status_code=response.status_code, media_type="text/event-stream")

    @staticmethod
    def _anthropic_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _anthropic_mock(self, message_id: str, model_name: str, text: str, is_stream: bool) -> Response:
        if not is_stream:
            return JSONResponse({
                "id": message_id, "type": "message", "role": "assistant", "model": model_name,
                "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": max(1, len(text.split()))},
            })

        async def stream():
            yield self._anthropic_event("message_start", {"type": "message_start", "message": {"id": message_id, "type": "message", "role": "assistant", "content": [], "model": model_name, "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
            yield self._anthropic_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            for word in text.split(" "):
                yield self._anthropic_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": word + " "}})
                await asyncio.sleep(0.01)
            yield self._anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield self._anthropic_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": max(1, len(text.split()))}})
            yield self._anthropic_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def count_tokens(self, request: Request) -> Response:
        self.authenticate(request)
        body = await request.json()
        if self.active_model().get("native_anthropic") and not self.mock_mode_getter():
            return await self.raw_proxy("/v1/messages/count_tokens", body, False)
        text = json.dumps(body.get("messages", []), ensure_ascii=False) + str(body.get("system", ""))
        return JSONResponse({"input_tokens": max(1, len(text) // 4)})
