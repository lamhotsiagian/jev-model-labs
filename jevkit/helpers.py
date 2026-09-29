"""jevkit.helpers -- patterns for working within Jev's constraints (Chapter 6).

    select_span            no text generation  -> find candidates in code, Jev picks one
    hierarchical_classify  > 255 classes       -> beam search down a label tree
    count_items            no reliable counting -> one Noul per item, sum in code
    rank_pairs             pairwise judgments  -> one question per pair, one call

Every helper keeps arithmetic, ordering and string manipulation in code and
asks Jev only for the judgment it is good at.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .client import JevClient
from .contract import ChoiceQ, MAX_CHOICE_OPTIONS, NoulQ

NONE_OPTION = "none_of_these"


# ---------------------------------------------------------------------------
# 1. Extraction by candidates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpanResult:
    value: str | None          # normalized value, or None if Jev picked "none"
    raw: str | None
    confidence: float
    n_candidates: int


def select_span(client: JevClient, state: str, pattern: str | re.Pattern, role: str,
                normalize: Callable[[str], str] = str.strip, context_chars: int = 40,
                max_candidates: int = MAX_CHOICE_OPTIONS - 1) -> SpanResult:
    """Extract the span that plays ``role`` (e.g. "the order the customer wants refunded").

    Step 1 (code): regex/parser proposes candidates, each shown with context.
    Step 2 (Jev):  a Choice picks the right candidate or 'none_of_these'.
    Step 3 (code): normalize the chosen raw string.
    """
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern
    seen, cands = set(), []
    for m in rx.finditer(state):
        if m.group(0) in seen:
            continue
        seen.add(m.group(0))
        lo, hi = max(0, m.start() - context_chars), min(len(state), m.end() + context_chars)
        cands.append((m.group(0), state[lo:hi].replace("\n", " ")))
    if not cands:
        return SpanResult(None, None, 1.0, 0)
    cands = cands[:max_candidates]
    options = {f"c{i}": f"'{raw}' in: ...{ctx}..." for i, (raw, ctx) in enumerate(cands)}
    options[NONE_OPTION] = "None of the candidates fits the role"
    ans = client.ask(state, {"span": ChoiceQ(f"Which candidate is {role}?", options)})["span"]
    if ans.choice == NONE_OPTION:
        return SpanResult(None, None, ans.confidence, len(cands))
    raw = cands[int(ans.choice[1:])][0]
    return SpanResult(normalize(raw), raw, ans.confidence, len(cands))


# ---------------------------------------------------------------------------
# 2. Hierarchical classification (cardinality > 255)
# ---------------------------------------------------------------------------
Tree = Mapping[str, Any]   # {"label": description | subtree}


def hierarchical_classify(client: JevClient, state: Any, tree: Tree, beam: int = 2,
                          instructions: str = "Which category best describes this item?") -> list[tuple[str, float]]:
    """Beam search down a label taxonomy using Choice probabilities.

    At each level we keep the ``beam`` most probable branches and multiply path
    probabilities. Beam > 1 recovers from a near-tie at a coarse level, the
    main failure of greedy top-down classification. All sibling Choices at one
    depth are sent in ONE call (fan-out), so latency grows with depth, not width.
    Returns leaf paths ("a/b/c") ranked by path probability.
    """
    frontier: list[tuple[str, float, Tree]] = [("", 1.0, tree)]
    leaves: list[tuple[str, float]] = []
    while frontier:
        qs, meta, nxt = {}, {}, []
        for i, (path, p, node) in enumerate(frontier):
            opts = {k: (v if isinstance(v, str) else v.get("_desc", k)) for k, v in node.items()
                    if k != "_desc"}
            if len(opts) == 1:                      # single child: no question needed
                (label, child), = ((k, v) for k, v in node.items() if k != "_desc")
                full = f"{path}/{label}".lstrip("/")
                (leaves.append((full, p)) if isinstance(child, str) else nxt.append((full, p, child)))
                continue
            if len(opts) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"node '{path}' has {len(opts)} children; split it")
            qs[f"n{i}"] = ChoiceQ(f"{instructions} (within: {path or 'all'})", opts)
            meta[f"n{i}"] = (path, p, node)
        res = client.ask(state, qs) if qs else None
        for qid, (path, p, node) in meta.items():
            ans = res[qid]
            for label, q in sorted(ans.probabilities.items(), key=lambda kv: -kv[1])[:beam]:
                child, full = node[label], f"{path}/{label}".lstrip("/")
                if isinstance(child, str):
                    leaves.append((full, p * q))
                else:
                    nxt.append((full, p * q, child))
        frontier = sorted(nxt, key=lambda t: -t[1])[:beam]
    return sorted(leaves, key=lambda t: -t[1])


def two_stage_choose(client: JevClient, state: Any, labels: Mapping[str, str], shortlist: int = 20,
                     instructions: str = "Which label fits best?") -> tuple[str, float]:
    """Alternative for flat label sets > 255: Noul relevance per label, then one Choice.

    Stage 1 fans out one Noul per label in a single call (cheap, output is free);
    stage 2 runs a Choice over the top ``shortlist`` labels.
    """
    items = list(labels.items())
    scores: dict[str, float] = {}
    for start in range(0, len(items), 200):              # keep requests reasonably sized
        batch = items[start:start + 200]
        res = client.ask(state, {f"l{i}": NoulQ(f"Is this item about: {desc}?")
                                 for i, (_, desc) in enumerate(batch)})
        for i, (name, _) in enumerate(batch):
            scores[name] = res[f"l{i}"].p_true
    top = sorted(scores, key=scores.get, reverse=True)[:shortlist]
    ans = client.ask(state, {"pick": ChoiceQ(instructions, {k: labels[k] for k in top})})["pick"]
    return ans.choice, ans.confidence


# ---------------------------------------------------------------------------
# 3. Counting
# ---------------------------------------------------------------------------
def count_items(client: JevClient, items: Sequence[str], predicate: str,
                threshold: float = 0.5, context: str = "",
                criteria: Mapping[str, str] | None = None) -> dict:
    """Count items satisfying ``predicate`` with one Noul per item, summed in code.

    Returns the hard count (p >= threshold) AND the expected count (sum of p),
    whose gap is a free uncertainty signal for the aggregate.
    """
    if not items:
        return {"count": 0, "expected": 0.0, "per_item": []}
    state = (context + "\n" if context else "") + "\n".join(f"item {i}: {t}" for i, t in enumerate(items))
    qs = {f"i{i}": NoulQ(f"Does item {i} satisfy: {predicate}? Consider only item {i}.", criteria)
          for i in range(len(items))}
    res = client.ask(state, qs)
    ps = [res[f"i{i}"].p_true for i in range(len(items))]
    return {"count": sum(p >= threshold for p in ps), "expected": round(sum(ps), 2), "per_item": ps}


# ---------------------------------------------------------------------------
# 4. Pairwise judgments
# ---------------------------------------------------------------------------
def rank_pairs(client: JevClient, query: str, candidates: Mapping[str, str],
               relation: str = "answers the query") -> list[tuple[str, float]]:
    """Rerank / match with one Noul per (query, candidate) pair, batched in one call.

    Each pair gets its own state slot and its own question naming that slot, so
    the judgment is about that pair only. The query is restated inside each
    question: a question should carry the exact condition it tests.
    Sorting by P(yes) in code yields the ranking.
    """
    ids = list(candidates)
    state = "\n".join(f"candidate {cid}: {candidates[cid]}" for cid in ids)
    qs = {f"p{i}": NoulQ(f"Candidate {cid} {relation}: '{query}'") for i, cid in enumerate(ids)}
    res = client.ask(state, qs)
    return sorted(((cid, res[f"p{i}"].p_true) for i, cid in enumerate(ids)), key=lambda t: -t[1])
