import asyncio
import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RelaySession:
    token: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pending: Dict[str, asyncio.Future] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)


class OutboundRelayHub:
    """Authenticated controller/Kaggle request relay.

    Kaggle workers poll this controller over ordinary outbound HTTP. No inbound
    tunnel process runs inside the GPU worker, which avoids platform tunnel
    cancellation while retaining the permanent Tailscale client endpoint.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, RelaySession] = {}

    def register(self, session_id: str, token: str) -> None:
        current = self.sessions.get(session_id)
        if current and current.token == token:
            current.last_seen = time.time()
            return
        self.sessions[session_id] = RelaySession(token=token)

    def unregister(self, session_id: str, reason: str = "Relay session closed") -> None:
        session = self.sessions.pop(session_id, None)
        if not session:
            return
        for future in list(session.pending.values()):
            if not future.done():
                future.set_exception(ConnectionError(reason))

    def authenticate(self, session_id: str, token: str) -> bool:
        session = self.sessions.get(session_id)
        if not session or not token or token != session.token:
            return False
        session.last_seen = time.time()
        return True

    def is_online(self, session_id: str, max_age: float = 45.0) -> bool:
        session = self.sessions.get(session_id)
        return bool(session and time.time() - session.last_seen <= max_age)

    async def poll(self, session_id: str, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if not session:
            return None
        session.last_seen = time.time()
        try:
            job = await asyncio.wait_for(session.queue.get(), timeout=max(1.0, min(timeout, 25.0)))
        except asyncio.TimeoutError:
            session.last_seen = time.time()
            return None
        session.last_seen = time.time()
        return job

    async def submit(
        self,
        session_id: str,
        path: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session or not self.is_online(session_id):
            raise ConnectionError("Kaggle outbound relay is offline")
        loop = asyncio.get_running_loop()
        job_id = uuid.uuid4().hex
        future = loop.create_future()
        session.pending[job_id] = future
        await session.queue.put({
            "id": job_id,
            "path": path,
            "method": "POST",
            "body": body,
            "headers": headers,
        })
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            session.pending.pop(job_id, None)

    def resolve(
        self,
        session_id: str,
        job_id: str,
        status_code: int,
        headers: Optional[Dict[str, str]],
        body_b64: str,
    ) -> bool:
        session = self.sessions.get(session_id)
        if not session:
            return False
        session.last_seen = time.time()
        future = session.pending.get(job_id)
        if not future or future.done():
            return False
        try:
            content = base64.b64decode(body_b64 or "", validate=False)
        except Exception:
            content = b""
        future.set_result({
            "status_code": int(status_code or 502),
            "headers": dict(headers or {}),
            "content": content,
        })
        return True
