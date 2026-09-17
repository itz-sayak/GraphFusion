"""FastAPI application entry point.

    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from backend import __version__
from backend.api.routes import router
from backend.config import get_settings
from backend.core.errors import DFGError
from backend.logging_conf import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.dfg_log_level, settings.dfg_log_json)
    log = get_logger("dfg.api")
    app = FastAPI(
        title="GraphFusion API",
        version=__version__,
        description="Conversational multi-format dataset integration: schema matching, entity resolution, graph-based merge planning, provenance.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins + ["http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID", "Content-Disposition"],
    )

    open_paths = ("/health", "/docs", "/redoc", "/openapi.json")

    @app.middleware("http")
    async def session_header(request: Request, call_next):
        started = time.perf_counter()
        tokens = get_settings().api_tokens
        if tokens and request.method != "OPTIONS" and not request.url.path.startswith(open_paths):
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.query_params.get("token", "")
            user = tokens.get(token)
            if user is None:
                return JSONResponse(status_code=401, content={"error": "unauthorized", "message": "A valid API token is required (Authorization: Bearer <token>)", "details": {}},
                                    headers={"WWW-Authenticate": "Bearer"})
            request.state.user = user
        response = await call_next(request)
        sid = getattr(request.state, "session_id", None)
        if sid and request.method in ("POST", "PUT", "PATCH", "DELETE") and response.status_code < 500:
            # persist after every mutation so a restart never loses datasets, labels, decisions or merges
            try:
                from backend.api import deps

                await run_in_threadpool(deps.registry().persist, sid)
            except Exception as exc:  # noqa: BLE001 - persistence problems must not fail the request
                log.warning("Session could not be persisted", session_id=sid, error=repr(exc))
        if sid:
            response.headers["X-Session-ID"] = sid
        if request.url.path not in ("/health",):
            log.info("HTTP request", method=request.method, path=request.url.path, status=response.status_code, elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return response

    @app.exception_handler(DFGError)
    async def dfg_error(_request: Request, exc: DFGError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.message, "details": exc.details})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": "invalid_request", "message": "Request validation failed", "details": {"errors": exc.errors()}})

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        log.error("Unhandled error", error=repr(exc))
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)[:500], "details": {}})

    app.include_router(router)
    return app


app = create_app()
