"""Shared API dependencies: session registry, agent, session resolution."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Header, Query, Request

from backend.llm.agent import ChatAgent, make_agent
from backend.session.state import Session, SessionRegistry


@lru_cache
def registry() -> SessionRegistry:
    return SessionRegistry()


_agents: dict[str, ChatAgent] = {}


def agent(provider: str | None = None) -> ChatAgent:
    key = provider or "__default__"
    if key not in _agents:
        from backend.llm.factory import make_provider

        _agents[key] = make_agent(make_provider(provider) if provider else None)
    return _agents[key]


def get_session(
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    session_id: str | None = Query(default=None),
) -> Session:
    """Resolve the session from the X-Session-ID header or ``session_id`` query (created on first use)."""
    sid = x_session_id or session_id
    session = registry().get_or_create(sid, owner=current_user(request))
    request.state.session_id = session.session_id
    return session


def current_user(request: Request) -> str | None:
    """User id set by the auth middleware (None when authentication is not configured)."""
    return getattr(request.state, "user", None)
