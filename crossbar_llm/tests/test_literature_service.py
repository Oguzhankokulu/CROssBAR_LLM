import asyncio
import os
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from crossbar_llm.agent_tools.callback_handler import UsageMetricsCallback
from crossbar_llm.api.schemas.requests import DbSearchRequest, LiteratureToolsConfig
from crossbar_llm.api.core.settings import Settings
from crossbar_llm.api.services.literature_service import LiteratureService
from crossbar_llm.paperclip_tools.adapter import PaperclipConfigError


def _settings(timeout: float = 1.0) -> Settings:
    # The real Settings object, not a stub: these tests exercise code that
    # reads a growing set of tuning fields, and a hand-rolled namespace would
    # drift out of sync with it silently.
    return Settings(literature_tool_timeout_seconds=timeout)


def _payload(tools: LiteratureToolsConfig) -> DbSearchRequest:
    return DbSearchRequest(
        provider="openai",
        model="gpt-4o-mini",
        question="What is the role of EGFR in cancer?",
        execution_mode="generate_and_run",
        literature_tools=tools,
    )


@pytest.mark.asyncio
async def test_disabled_tools_are_not_run(monkeypatch):
    service = LiteratureService(_settings())

    async def unexpected(*args, **kwargs):
        raise AssertionError("disabled literature tool was invoked")

    monkeypatch.setattr(service, "_run_paperclip", unexpected)
    monkeypatch.setattr(service, "_run_pubtator3", unexpected)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig()),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result == {}


@pytest.mark.asyncio
async def test_enabled_tools_run_in_parallel(monkeypatch):
    service = LiteratureService(_settings())
    started = {"paperclip": asyncio.Event(), "pubtator3": asyncio.Event()}
    release = asyncio.Event()

    async def paperclip(*args, **kwargs):
        started["paperclip"].set()
        await release.wait()
        return {"final_answer": "Paperclip answer", "citations": [], "warnings": []}

    async def pubtator3(*args, **kwargs):
        started["pubtator3"].set()
        await release.wait()
        return {"final_answer": "PubTator3 answer", "documents": [], "warnings": []}

    monkeypatch.setattr(service, "_run_paperclip", paperclip)
    monkeypatch.setattr(service, "_run_pubtator3", pubtator3)

    task = asyncio.create_task(
        service.run(
            question="test",
            payload=_payload(LiteratureToolsConfig(paperclip=True, pubtator3=True)),
            callback=UsageMetricsCallback("session", strict=False),
        )
    )
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 1)
    release.set()
    result = await task

    assert list(result) == ["paperclip", "pubtator3"]
    assert result["paperclip"].answer == "Paperclip answer"
    assert result["pubtator3"].answer == "PubTator3 answer"


@pytest.mark.asyncio
async def test_one_tool_failure_does_not_discard_the_other(monkeypatch):
    service = LiteratureService(_settings())

    async def paperclip(*args, **kwargs):
        raise RuntimeError("Paperclip unavailable")

    async def pubtator3(*args, **kwargs):
        return {"final_answer": "PubTator3 answer", "documents": [], "warnings": []}

    monkeypatch.setattr(service, "_run_paperclip", paperclip)
    monkeypatch.setattr(service, "_run_pubtator3", pubtator3)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig(paperclip=True, pubtator3=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result["paperclip"].status == "failed"
    assert result["pubtator3"].status == "completed"


@pytest.mark.asyncio
async def test_tool_timeout_is_reported_per_tool(monkeypatch):
    service = LiteratureService(_settings(timeout=0.01))

    async def slow(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(service, "_run_paperclip", slow)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig(paperclip=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result["paperclip"].status == "failed"
    assert "timed out" in result["paperclip"].warnings[0]


@pytest.mark.asyncio
async def test_malformed_tool_result_is_isolated(monkeypatch):
    service = LiteratureService(_settings())

    async def malformed(*args, **kwargs):
        return None

    async def valid(*args, **kwargs):
        return {"final_answer": "PubTator3 answer", "documents": [], "warnings": []}

    monkeypatch.setattr(service, "_run_paperclip", malformed)
    monkeypatch.setattr(service, "_run_pubtator3", valid)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig(paperclip=True, pubtator3=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result["paperclip"].status == "failed"
    assert result["pubtator3"].status == "completed"


@pytest.mark.asyncio
async def test_paperclip_adapter_is_lazy_reused_and_closed(monkeypatch):
    created = []

    class Adapter:
        def __init__(self, **kwargs):
            self.close_count = 0
            self.kwargs = kwargs
            created.append(self)

        async def aclose(self):
            self.close_count += 1

    monkeypatch.setattr(
        "crossbar_llm.api.services.literature_service.PaperclipAdapter",
        Adapter,
    )
    settings = _settings()
    settings.env_settings = SimpleNamespace(
        paperclip_api_key=SecretStr("paperclip-test-key"),
        paperclip_disable_rest=True,
    )
    monkeypatch.delenv("PAPERCLIP_API_KEY", raising=False)
    monkeypatch.delenv("PAPERCLIP_DISABLE_REST", raising=False)
    service = LiteratureService(settings)

    assert service.paperclip_adapter is None
    assert service._get_paperclip_adapter() is service._get_paperclip_adapter()
    assert len(created) == 1

    # Credentials are handed to the adapter directly. Exporting them to
    # os.environ instead would be a process-global side effect from a request
    # path, and would leak between tests and between tenants.
    assert created[0].kwargs["api_key"] == "paperclip-test-key"
    assert created[0].kwargs["disable_rest"] is True
    assert "PAPERCLIP_API_KEY" not in os.environ
    assert "PAPERCLIP_DISABLE_REST" not in os.environ

    await service.aclose()
    assert created[0].close_count == 1
    assert service.paperclip_adapter is None


@pytest.mark.asyncio
async def test_missing_paperclip_credentials_fail_only_that_tool(monkeypatch):
    """An unconfigured tool degrades to its own failure, not a 500."""
    service = LiteratureService(_settings())

    async def unconfigured(*args, **kwargs):
        raise PaperclipConfigError("PAPERCLIP_API_KEY is not set")

    async def pubtator3(*args, **kwargs):
        return {"final_answer": "PubTator3 answer", "documents": [], "warnings": []}

    monkeypatch.setattr(service, "_run_paperclip", unconfigured)
    monkeypatch.setattr(service, "_run_pubtator3", pubtator3)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig(paperclip=True, pubtator3=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result["paperclip"].status == "failed"
    # A missing key is safe and actionable, so it is surfaced verbatim.
    assert "PAPERCLIP_API_KEY is not set" in result["paperclip"].warnings[0]
    assert result["pubtator3"].status == "completed"


@pytest.mark.asyncio
async def test_unexpected_failures_do_not_leak_upstream_detail(monkeypatch):
    """Arbitrary upstream error text must not reach the HTTP response."""
    service = LiteratureService(_settings())

    async def boom(*args, **kwargs):
        raise RuntimeError("https://paperclip.example/x?token=SUPERSECRET failed")

    monkeypatch.setattr(service, "_run_paperclip", boom)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig(paperclip=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    warning = result["paperclip"].warnings[0]
    assert "SUPERSECRET" not in warning
    assert "RuntimeError" in warning


@pytest.mark.asyncio
async def test_enabled_tool_without_a_question_is_reported_as_skipped(monkeypatch):
    """Resuming with no checkpointed question must say so, not stay silent."""
    service = LiteratureService(_settings())

    async def unexpected(*args, **kwargs):
        raise AssertionError("literature tool ran without a question")

    monkeypatch.setattr(service, "_run_paperclip", unexpected)

    result = await service.run(
        question="   ",
        payload=_payload(LiteratureToolsConfig(paperclip=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result["paperclip"].status == "skipped"
    assert "no question" in result["paperclip"].warnings[0]


def test_tool_usage_slices_by_node_name_prefix():
    """Pins the contract `_tool_usage` depends on.

    Per-tool usage is recovered by matching the `node_name` prefix each agent
    tags its LLM calls with. Renaming a node without updating the prefix would
    otherwise empty this out with no test failing.
    """
    service = LiteratureService(_settings())
    summary = {
        "per_node_usage": {
            "paperclip.router": {"total_tokens": 10, "call_count": 1},
            "paperclip.synthesize": {"total_tokens": 30, "call_count": 2},
            "pubtator3.router": {"total_tokens": 7, "call_count": 1},
        },
        "aggregated_usage": {
            "models_by_node": {"paperclip.router": ["gpt-4o-mini"]},
        },
    }

    paperclip = service._tool_usage(summary, "paperclip.")

    assert set(paperclip["per_node_usage"]) == {
        "paperclip.router",
        "paperclip.synthesize",
    }
    assert paperclip["aggregated_usage"]["totals"]["total_tokens"] == 40
    assert paperclip["call_count"] == 3
    # `totals` keeps exactly the core agent's shape, so the two are comparable.
    assert "call_count" not in paperclip["aggregated_usage"]["totals"]
    assert service._tool_usage(summary, "nothing.") == {}


@pytest.mark.asyncio
async def test_admission_control_bounds_concurrent_runs(monkeypatch):
    """Only `literature_max_concurrent_runs` tools may be in flight at once."""
    settings = _settings()
    settings.literature_max_concurrent_runs = 1
    service = LiteratureService(settings)

    in_flight = 0
    peak = 0
    release = asyncio.Event()

    async def tool(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await release.wait()
            return {"final_answer": "answer", "documents": [], "warnings": []}
        finally:
            in_flight -= 1

    monkeypatch.setattr(service, "_run_paperclip", tool)
    monkeypatch.setattr(service, "_run_pubtator3", tool)

    task = asyncio.create_task(
        service.run(
            question="test",
            payload=_payload(LiteratureToolsConfig(paperclip=True, pubtator3=True)),
            callback=UsageMetricsCallback("session", strict=False),
        )
    )
    await asyncio.sleep(0.05)
    assert peak == 1, "both tools started despite a concurrency limit of 1"
    release.set()
    await task


@pytest.mark.asyncio
async def test_saturation_reports_skipped_not_timed_out(monkeypatch):
    """A rejected run must not masquerade as a slow upstream service."""
    settings = _settings()
    settings.literature_max_concurrent_runs = 1
    settings.literature_admission_wait_seconds = 0.01
    service = LiteratureService(settings)

    release = asyncio.Event()

    async def blocker(*args, **kwargs):
        await release.wait()
        return {"final_answer": "answer", "citations": [], "warnings": []}

    monkeypatch.setattr(service, "_run_paperclip", blocker)
    monkeypatch.setattr(service, "_run_pubtator3", blocker)

    # Occupy the only slot, then ask for both tools.
    hog = asyncio.create_task(
        service.run(
            question="test",
            payload=_payload(LiteratureToolsConfig(paperclip=True)),
            callback=UsageMetricsCallback("session", strict=False),
        )
    )
    await asyncio.sleep(0.05)

    result = await service.run(
        question="test",
        payload=_payload(LiteratureToolsConfig(pubtator3=True)),
        callback=UsageMetricsCallback("session", strict=False),
    )

    assert result["pubtator3"].status == "skipped"
    assert "capacity" in result["pubtator3"].warnings[0]
    release.set()
    await hog
