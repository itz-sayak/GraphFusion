"""Typed exceptions mapped to HTTP responses by the API layer."""
from __future__ import annotations


class DFGError(Exception):
    status_code = 400
    code = "dfg_error"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details


class UnsupportedFormatError(DFGError):
    code = "unsupported_format"


class IngestionError(DFGError):
    code = "ingestion_failed"


class NotFoundError(DFGError):
    status_code = 404
    code = "not_found"


class InvalidStateError(DFGError):
    status_code = 409
    code = "invalid_state"


class ValidationFailedError(DFGError):
    status_code = 422
    code = "validation_failed"


class ToolArgumentError(DFGError):
    status_code = 422
    code = "invalid_tool_arguments"


class LLMProviderError(DFGError):
    status_code = 502
    code = "llm_provider_error"
