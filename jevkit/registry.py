"""jevkit.registry -- version-controlled question registry (Chapter 4).

Questions are code. A question is identified by a stable ID, carries a
semantic version and an owner, and every wording change is recorded in a
changelog. The content hash lets logs, caches and evaluations detect a silent
edit that forgot to bump the version.

YAML format (see labs/ch04_state/questions.yaml):

    workflow: support_triage
    questions:
      - id: checkout_outage
        type: noul
        version: 2
        owner: payments-oncall
        instructions: Is checkout unavailable to more than one customer?
        criteria: {true: ..., false: ...}
        changelog:
          - "v2: narrowed from 'Is this urgent?'"
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .contract import ChoiceQ, NoulQ, Question, ScoreQ


@dataclass(frozen=True)
class RegisteredQuestion:
    id: str
    type: str
    version: int
    owner: str
    instructions: Any
    criteria: Any = None
    changelog: tuple[str, ...] = field(default_factory=tuple)

    @property
    def content_hash(self) -> str:
        body = json.dumps({"t": self.type, "i": self.instructions, "c": self.criteria},
                          sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:12]

    @property
    def versioned_id(self) -> str:
        """What we log: ``checkout_outage@v2#a1b2c3d4e5f6``."""
        return f"{self.id}@v{self.version}#{self.content_hash}"

    def to_question(self) -> Question:
        if self.type == "noul":
            return NoulQ(self.instructions, self.criteria)
        if self.type == "choice":
            return ChoiceQ(self.instructions, self.criteria)
        if self.type == "score":
            return ScoreQ(self.instructions, tuple(self.criteria))
        raise ValueError(f"{self.id}: unknown type {self.type}")


class QuestionRegistry:
    def __init__(self, workflow: str, questions: list[RegisteredQuestion]):
        ids = [q.id for q in questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        self.workflow = workflow
        self._q = {q.id: q for q in questions}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuestionRegistry":
        doc = yaml.safe_load(Path(path).read_text())
        qs = []
        for d in doc["questions"]:
            qs.append(RegisteredQuestion(
                id=d["id"], type=d["type"], version=int(d.get("version", 1)),
                owner=d.get("owner", "unowned"), instructions=d["instructions"],
                criteria=d.get("criteria"), changelog=tuple(d.get("changelog", []))))
        return cls(doc["workflow"], qs)

    def __getitem__(self, qid: str) -> RegisteredQuestion:
        return self._q[qid]

    def __iter__(self):
        return iter(self._q.values())

    def __len__(self) -> int:
        return len(self._q)

    def questions(self, ids: list[str] | None = None) -> dict[str, Question]:
        sel = ids or list(self._q)
        return {i: self._q[i].to_question() for i in sel}

    def fingerprint(self) -> str:
        """One hash for the whole workflow; part of every cache key."""
        return hashlib.sha256("|".join(sorted(q.versioned_id for q in self)).encode()).hexdigest()[:12]

    def lint(self) -> list[str]:
        """Static checks that catch the most common question-design bugs."""
        issues = []
        vague = ("urgent", "important", "good", "bad", "appropriate", "problematic")
        for q in self:
            text = json.dumps(q.instructions).lower()
            if any(f" {w}" in text or text.startswith(f'"{w}') for w in vague):
                issues.append(f"{q.id}: vague adjective in instructions; split into atomic questions")
            if " and " in text and q.type == "noul":
                issues.append(f"{q.id}: Noul with 'and' may be two judgments in one")
            if "not" in text.split() and ("n't" in text or " no " in text):
                issues.append(f"{q.id}: possible double negative (Jev 1.13 weak spot)")
            if q.type == "choice" and q.criteria and all(v is None for v in q.criteria.values()):
                issues.append(f"{q.id}: Choice options have no descriptions")
            if not q.changelog and q.version > 1:
                issues.append(f"{q.id}: version {q.version} with empty changelog")
        return issues
