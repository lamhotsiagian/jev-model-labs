"""jevkit -- production toolkit for building decision systems on TypeSafe Jev.

Companion code for "JEV Model System Design" (AI Engineering Insider).
Runs offline by default against a deterministic simulator; set
JEV_BACKEND=http and TYPESAFE_API_KEY to call the real API.
"""
from .client import AsyncJevClient, JevClient, JevError
from .contract import (ChoiceAnswer, ChoiceQ, ContractError, JevResult, NoulAnswer, NoulQ,
                       ScoreAnswer, ScoreQ, build_request, normalize, parse_response)

__all__ = ["JevClient", "AsyncJevClient", "JevError", "NoulQ", "ChoiceQ", "ScoreQ",
           "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "JevResult", "ContractError",
           "build_request", "parse_response", "normalize"]
__version__ = "1.0.0"
