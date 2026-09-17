import json
import sys
from pathlib import Path

import pytest

from backend.llm.base import LLMResponse
from backend.llm.providers.openai_compat import OpenAICompatibleProvider
from backend.llm.rate_limit import RateLimiter, RateLimitExhausted, parse_duration

LIMITS = {"default": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}, "small": {"rpm": 3, "tpm": 1000, "tpd": 3000}}


@pytest.fixture()
def clock(monkeypatch):
    now = {"t": 1_800_000_000.0}
    slept = []

    def sleep(s):
        slept.append(s)
        now["t"] += s

    monkeypatch.setattr("backend.llm.rate_limit.time.time", lambda: now["t"])
    monkeypatch.setattr("backend.llm.rate_limit.time.monotonic", lambda: now["t"])
    monkeypatch.setattr("backend.llm.rate_limit.time.sleep", sleep)
    return now, slept


def test_parse_duration():
    assert parse_duration("10m48s") == pytest.approx(648)
    assert parse_duration("255ms") == pytest.approx(0.255)
    assert parse_duration("1h2m3s") == pytest.approx(3723)


def test_minute_window_waits_then_day_budget_raises(tmp_path, clock):
    now, slept = clock
    rl = RateLimiter(tmp_path / "u.json", LIMITS, safety=1.0, max_wait_s=90)
    t1 = rl.acquire("small", 600)
    rl.record("small", t1, 600)
    now["t"] += 10
    rl.acquire("small", 600)  # 600 + 600 > 1000 TPM → waits for the first request to leave the window
    assert slept and 45 <= slept[0] <= 51
    rl2 = RateLimiter(tmp_path / "u.json", LIMITS, safety=1.0, max_wait_s=90)  # state persisted across instances
    assert rl2.usage("small")["day"]["tokens"] == 1200
    with pytest.raises(RateLimitExhausted) as e:
        for _ in range(5):
            now["t"] += 61
            rl2.acquire("small", 600)
    assert e.value.scope == "day"


def test_rpm_and_short_wait_budget(tmp_path, clock):
    now, _ = clock
    rl = RateLimiter(None, LIMITS, safety=1.0, max_wait_s=5)
    for _ in range(3):
        rl.acquire("small", 10)
    with pytest.raises(RateLimitExhausted) as e:
        rl.acquire("small", 10)  # 4th request in the minute; wait of ~60 s exceeds the 5 s budget
    assert e.value.scope == "minute"


def test_429_body_blocks_model_until_retry_time(tmp_path, clock):
    now, _ = clock
    rl = RateLimiter(tmp_path / "u.json", LIMITS)
    body = '{"error":{"message":"Rate limit reached for model `m` on tokens per day (TPD): Limit 200000, Used 199087, Requested 2413. Please try again in 10m48s."}}'
    secs = rl.block_from_429("m", body)
    assert secs == pytest.approx(648)
    with pytest.raises(RateLimitExhausted):
        rl.acquire("m", 10)
    now["t"] += 700
    assert rl.usage("m")["blocked_until"] is None


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if isinstance(body, dict) else body
        self.headers = headers or {}

    def json(self):
        return self._body


def test_provider_fails_over_to_next_model_on_daily_limit(tmp_path, monkeypatch):
    calls = []
    ok = {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 50}}
    tpd = {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000, Used 199999. Please try again in 30m0s."}}

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            calls.append(json["model"])
            return _Resp(429, tpd) if json["model"] == "primary" else _Resp(200, ok, {"x-ratelimit-remaining-tokens": "7000", "x-ratelimit-reset-tokens": "2s"})

    monkeypatch.setattr("backend.llm.providers.openai_compat.httpx.Client", Client)
    rl = RateLimiter(tmp_path / "u.json", LIMITS)
    p = OpenAICompatibleProvider("http://x", "k", "primary", fallback_models=["backup"], model_params={"backup": {"reasoning_effort": "low"}}, limiter=rl)
    r = p.chat([{"role": "user", "content": "hello"}])
    assert isinstance(r, LLMResponse) and r.content == "hi"
    assert calls == ["primary", "backup"] and p.describe()["model"] == "backup"
    calls.clear()
    p.chat([{"role": "user", "content": "again"}])
    assert calls == ["backup"], "a model blocked by a daily 429 is not retried"


def test_tool_selection_covers_benchmark_requests(merged_session):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from run_llm_eval import CASES

    from backend.llm.tool_selection import select_tools
    from backend.llm.tools import TOOLS

    missed = []
    for _kind, msg, groups, _args in CASES:
        sel = set(select_tools(merged_session, msg, 12))
        assert len(sel) <= 12
        if any(not (g & sel) for g in groups):
            missed.append(msg)
    assert not missed, missed
    assert len(select_tools(merged_session, "anything", 0)) == len(TOOLS)


def test_follow_up_rounds_only_offer_requested_mutations(merged_session):
    from backend.llm.tool_selection import requested_mutations

    assert "execute_merge" in requested_mutations(merged_session, "Only merge matches above 95%, then merge them.")
    assert "set_preferences" in requested_mutations(merged_session, "Only merge matches above 95%, then merge them.")
    assert not requested_mutations(merged_session, "Show me conflicts.") & {"execute_merge", "set_preferences", "undo_merge"}


def test_agent_skips_follow_up_round_when_budget_is_tight(merged_session):
    from backend.llm.agent import ChatAgent
    from backend.llm.base import LLMProvider, ToolCall

    class Tight(LLMProvider):
        name = "tight"
        model = "t"
        is_mock = False

        def __init__(self):
            self.calls = []

        def budget_ok(self, messages, tools=None, max_tokens=256):
            return False

        def chat(self, messages, tools=None, temperature=0.1, max_tokens=2048, max_wait_s=None):
            self.calls.append([t["name"] for t in tools or []])
            return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name="show_conflicts", arguments={})])

    p = Tight()
    agent = ChatAgent(p)
    res = agent.handle(merged_session, "Show me conflicts.")
    assert [c["tool"] for c in res["tool_calls"]] == ["show_conflicts"]
    assert len(p.calls) == 1 and agent.last_turn_stats["skipped_followup"] == "rate-limit budget"
    assert len(p.calls[0]) <= 12


def test_dataset_change_outdates_merge_and_agent_nudges_for_action(tmp_path, sample_files):
    from backend.llm.agent import ChatAgent, state_summary
    from backend.llm.base import LLMProvider, ToolCall
    from backend.session.state import Session

    s = Session(workspace_root=tmp_path)
    for f in sample_files:
        s.add_file(f)
    s.discover()
    old = s.execute(write_csv=False)
    s.remove_dataset("sample_sales")
    assert old.status == "outdated" and not any(r.status == "active" for r in s.merges)
    assert "outdated" in state_summary(s) and "Current merge:" not in state_summary(s)

    class Lazy(LLMProvider):
        """First answers from memory (no tool), then complies when nudged."""
        name, model, is_mock = "lazy", "l", False

        def __init__(self):
            self.n = 0

        def chat(self, messages, tools=None, temperature=0.1, max_tokens=2048):
            self.n += 1
            if self.n == 1:
                return LLMResponse(content="The merge is already complete.", tool_calls=[])
            if self.n == 2:
                return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name="execute_merge", arguments={})])
            return LLMResponse(content="Done.", tool_calls=[])

    agent = ChatAgent(Lazy())
    res = agent.handle(s, "merge all of it")
    assert [c["tool"] for c in res["tool_calls"]] == ["execute_merge"] and agent.last_turn_stats.get("nudged")
    assert s.current_merge().result.merge_id != old.result.merge_id


def test_agent_retries_with_all_tools_when_model_calls_unoffered_tool(merged_session):
    from backend.core.errors import LLMProviderError
    from backend.llm.agent import ChatAgent
    from backend.llm.base import LLMProvider, ToolCall
    from backend.llm.tools import TOOLS

    class Strict(LLMProvider):
        name, model, is_mock = "strict", "s", False

        def __init__(self):
            self.offered = []

        def chat(self, messages, tools=None, temperature=0.1, max_tokens=2048):
            names = [t["name"] for t in tools or []]
            self.offered.append(len(names))
            if len(self.offered) == 1:
                raise LLMProviderError("groq returned HTTP 400: tool call validation failed: attempted to call tool 'lineage_report' which was not in request.tools")
            if len(self.offered) == 2:
                return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name="lineage_report", arguments={})])
            raise LLMProviderError("groq returned HTTP 400: attempted to call tool 'quality_report' which was not in request.tools")

    p = Strict()
    agent = ChatAgent(p)
    res = agent.handle(merged_session, "Show me conflicts.")
    assert [c["tool"] for c in res["tool_calls"]] == ["lineage_report"] and not res["provider"].get("fallback")
    assert p.offered[0] <= 12 and p.offered[1] == len(TOOLS) and agent.last_turn_stats["retried_with_all_tools"]
