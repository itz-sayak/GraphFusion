import pytest

from backend.core.errors import LLMProviderError, ToolArgumentError
from backend.llm.agent import ChatAgent
from backend.llm.base import LLMProvider, LLMResponse, ToolCall
from backend.llm.nl_fallback import parse
from backend.llm.providers.mock import MockProvider
from backend.llm.providers.openai_compat import OpenAICompatibleProvider
from backend.llm.tools import TOOLS, execute_tool, validate_arguments


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Find relationships between the datasets", ["find_candidate_relationships"]),
        ("Merge everything.", ["execute_merge"]),
        ("Only merge matches above 95%", ["set_preferences"]),
        ("Merge them, but only matches above 95%", ["set_preferences", "execute_merge"]),
        ("Don't merge uncertain records.", ["set_preferences"]),
        ("Use customer_id as the primary identifier", ["set_preferences"]),
        ("Show me conflicts", ["show_conflicts"]),
        ("Show me duplicate entities", ["show_duplicates"]),
        ("Undo the previous merge", ["undo_merge"]),
        ("Export the final dataset", ["export_dataset"]),
        ("Generate a data lineage report", ["lineage_report"]),
        ("What percentage of rows were matched?", ["match_statistics"]),
        ("What mappings did you infer?", ["show_mappings"]),
        ("Which columns refer to the same entity?", ["show_mappings"]),
        ("Why did you choose this join?", ["explain_relationship"]),
        ("Show me the graph of dataset relationships", ["build_graph"]),
        ("Explain why sample_customers and sample_sales were linked", ["explain_relationship"]),
        ("Why do city and location match?", ["explain_match"]),
    ],
)
def test_intent_parser(merged_session, message, expected):
    assert [name for name, _ in parse(merged_session, message)] == expected


def test_parser_extracts_arguments(merged_session):
    calls = dict(parse(merged_session, "Merge them, but only matches above 95% and prefer the latest values"))
    assert calls["set_preferences"] == {"confidence_threshold": 0.95, "conflict_strategy": "prefer_latest"}
    assert dict(parse(merged_session, "Use customer_id as the primary identifier"))["set_preferences"]["primary_key"] == "customer_id"


def test_tool_argument_validation():
    tool = TOOLS["set_preferences"]
    assert validate_arguments(tool, {"mode": "strict", "confidence_threshold": "95%"}) == {"mode": "strict", "confidence_threshold": 95.0}
    with pytest.raises(ToolArgumentError):
        validate_arguments(tool, {"mode": "reckless"})
    # unknown argument names are dropped, never passed on: injected keys cannot reach the session method
    assert validate_arguments(tool, {"python_code": "import os"}) == {}
    assert validate_arguments(TOOLS["match_statistics"], {"dataset": "merge_x"}) == {}
    with pytest.raises(ToolArgumentError):
        validate_arguments(TOOLS["explain_match"], {"left": "a"})


def test_unknown_tool_is_refused(merged_session):
    res = execute_tool(merged_session, "run_python", {"code": "print(1)"})
    assert not res["ok"] and "not an available operation" in res["rendered"]


class ScriptedProvider(LLMProvider):
    """Stands in for a remote model: emits a tool call, then a short commentary."""

    name = "scripted"
    model = "scripted"

    def __init__(self, calls, fail_after_tools=False):
        self.calls = calls
        self.turn = 0
        self.fail_after_tools = fail_after_tools

    def chat(self, messages, tools=None, temperature=0.1, max_tokens=2048):
        self.turn += 1
        if self.turn == 1:
            return LLMResponse(content=None, tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(self.calls)])
        if self.fail_after_tools:
            raise LLMProviderError("timed out")
        assert messages[-1]["role"] == "tool"
        return LLMResponse(content="Most rows matched; review the orphan transactions next.")


def test_llm_loop_executes_backend_tools(merged_session):
    agent = ChatAgent(ScriptedProvider([("match_statistics", {})]))
    res = agent.handle(merged_session, "how did the merge go?")
    assert res["tool_calls"][0]["tool"] == "match_statistics" and res["tool_calls"][0]["ok"]
    assert "%" in res["reply"] and "orphan transactions" in res["reply"]


def test_llm_failure_after_tools_does_not_rerun_them(merged_session):
    merges_before = len(merged_session.merges)
    agent = ChatAgent(ScriptedProvider([("execute_merge", {})], fail_after_tools=True))
    res = agent.handle(merged_session, "merge them")
    assert len(merged_session.merges) == merges_before + 1, "the merge must run exactly once"
    assert "stopped responding" in res["reply"]


def test_provider_unavailable_falls_back(merged_session):
    class Down(LLMProvider):
        name = "down"

        def chat(self, *a, **k):
            raise LLMProviderError("connection refused")

    res = ChatAgent(Down()).handle(merged_session, "What percentage of rows were matched?")
    assert res["provider"]["fallback"] and res["tool_calls"][0]["tool"] == "match_statistics"


def test_openai_compatible_parsing():
    data = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "x", "function": {"name": "execute_merge", "arguments": '{"confidence_threshold": 0.95}'}}]}}]}
    r = OpenAICompatibleProvider._parse(data)
    assert r.tool_calls[0].name == "execute_merge" and r.tool_calls[0].arguments == {"confidence_threshold": 0.95}


def test_mock_provider_is_default_capable(merged_session):
    res = ChatAgent(MockProvider()).handle(merged_session, "Show me conflicts")
    assert res["cards"] and res["cards"][0]["type"] == "conflicts"


def test_groq_provider_factory_and_rate_limit_parsing(monkeypatch):
    from backend.config import Settings
    from backend.llm.factory import make_provider
    from backend.llm.providers.openai_compat import GroqProvider, _parse_duration

    p = make_provider("groq", Settings(groq_api_key="test-key", dfg_groq_model="openai/gpt-oss-120b", _env_file=None))
    assert isinstance(p, GroqProvider) and p.base_url == "https://api.groq.com/openai/v1" and p.model == "openai/gpt-oss-120b"
    assert p.models[0] == "openai/gpt-oss-120b" and "openai/gpt-oss-20b" in p.models and p.limiter is not None
    assert p.extra_body["reasoning_effort"] == "medium"
    with pytest.raises(LLMProviderError):
        make_provider("groq", Settings(groq_api_key="", _env_file=None))
    assert _parse_duration("15m50.4s") == pytest.approx(950.4)
    assert _parse_duration("42.195s") == pytest.approx(42.195)
    assert _parse_duration("120ms") == pytest.approx(0.12)


def test_tool_specs_allow_null_for_optional_arguments():
    """Providers that validate tool calls server-side reject explicit nulls unless the schema allows them."""
    spec = TOOLS["generate_merge_plan"].spec()["parameters"]
    assert spec["properties"]["mode"]["type"] == ["string", "null"] and None in spec["properties"]["mode"]["enum"]
    assert "required" not in spec
    explain = TOOLS["explain_match"].spec()["parameters"]
    assert explain["properties"]["left"]["type"] == "string" and explain["required"] == ["left", "right"]
    assert validate_arguments(TOOLS["generate_merge_plan"], {"mode": None, "datasets": None}) == {}
