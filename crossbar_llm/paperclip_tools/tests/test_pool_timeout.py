"""Connection-pool saturation must fail fast and say what happened.

httpx expands a bare float timeout to connect/read/write/pool all at once, so
queueing for a free connection used to be charged against the 60s (or 480s)
call budget. Under load that surfaced as "Paperclip is slow" rather than "this
process is out of connections", and it quietly consumed the caller's per-tool
timeout.
"""
from __future__ import annotations

import httpx
import pytest

from crossbar_llm.paperclip_tools.adapter import (
    PaperclipAdapter,
    PaperclipRestUnavailable,
)


class _RecordingClient:
    """Captures the Timeout object the adapter hands to httpx."""

    def __init__(self, raises: Exception | None = None):
        self.raises = raises
        self.timeout = None

    async def post(self, url, **kwargs):
        self.timeout = kwargs.get("timeout")
        if self.raises is not None:
            raise self.raises
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"output": "ok"}, request=request)


async def test_pool_timeout_is_separate_from_the_call_timeout(monkeypatch):
    adapter = PaperclipAdapter(
        api_key="k", disable_rest=False, timeout_s=60.0, pool_timeout_s=10.0
    )
    client = _RecordingClient()
    monkeypatch.setattr(adapter, "_rest_client", lambda: client)

    await adapter._run_rest("search", "anything")

    assert isinstance(client.timeout, httpx.Timeout)
    assert client.timeout.read == 60.0
    # The whole point: queueing gets its own, much shorter budget.
    assert client.timeout.pool == 10.0


async def test_saturation_is_reported_as_pool_exhaustion(monkeypatch):
    adapter = PaperclipAdapter(
        api_key="k", disable_rest=False, max_connections=4, pool_timeout_s=10.0
    )
    client = _RecordingClient(raises=httpx.PoolTimeout("no free connection"))
    monkeypatch.setattr(adapter, "_rest_client", lambda: client)

    with pytest.raises(PaperclipRestUnavailable) as excinfo:
        await adapter._run_rest("search", "anything")

    message = str(excinfo.value)
    assert "pool exhausted" in message
    # Names the knob an operator would actually turn.
    assert "max_connections=4" in message


async def test_pool_size_is_configurable(monkeypatch):
    adapter = PaperclipAdapter(api_key="k", max_connections=64)
    client = adapter._rest_client()
    try:
        transport = client._transport
        pool = transport._pool
        assert pool._max_connections == 64
        assert pool._max_keepalive_connections == 32
    finally:
        await client.aclose()
