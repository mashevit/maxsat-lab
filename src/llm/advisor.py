from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
import json


@dataclass
class LLMAdvice:
    # Either "flip" variables, or set them explicitly.
    flip_vars: List[int]             # variable indices (positive ints)
    set_true: List[int]              # variable indices
    set_false: List[int]             # variable indices
    note: str = ""


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    # try to find JSON object in messy output
    text = text.strip()
    # quick path
    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: find first '{' ... last '}' block
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(text[i : j + 1])
        except Exception:
            return None
    return None


def apply_advice(assign01: List[bool], advice: LLMAdvice) -> List[bool]:
    """
    Applies advice to a 1-based assignment (index 0 unused).
    Returns a new list.
    """
    out = assign01[:]

    for v in advice.flip_vars:
        if v <= 0 or v >= len(out):
            continue
        out[v] = not out[v]

    for v in advice.set_true:
        if v <= 0 or v >= len(out):
            continue
        out[v] = True

    for v in advice.set_false:
        if v <= 0 or v >= len(out):
            continue
        out[v] = False

    out[0] = False
    return out


class LLMAdvisor:
    """
    High-level interface used by memetic.py.
    It builds a prompt and converts model output into an LLMAdvice object.
    """

    def __init__(self, provider: LLMProvider, max_clause_examples: int = 20):
        self.provider = provider
        self.max_clause_examples = max_clause_examples

    def propose(
        self,
        *,
        wcnf: Any,
        child_assign01: List[bool],
        violated_hard_clause_idxs: List[int],
        rng_seed: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMAdvice:
        from .prompt import build_prompt, extract_clause_examples

        clause_examples = extract_clause_examples(
            wcnf=wcnf,
            assign01=child_assign01,
            hard_clause_idxs=violated_hard_clause_idxs,
            max_k=self.max_clause_examples,
        )

        prompt = build_prompt(
            n_vars=getattr(wcnf, "n_vars", None) or getattr(wcnf, "nv", None),
            violated_hard=len(violated_hard_clause_idxs),
            clause_examples=clause_examples,
            rng_seed=rng_seed,
            extra=extra or {},
        )

        raw = self.provider.complete(prompt)
        obj = _safe_parse_json(raw) or {}

        return LLMAdvice(
            flip_vars=[int(x) for x in obj.get("flip", []) if str(x).lstrip("-").isdigit()],
            set_true=[int(x) for x in obj.get("set_true", []) if str(x).lstrip("-").isdigit()],
            set_false=[int(x) for x in obj.get("set_false", []) if str(x).lstrip("-").isdigit()],
            note=str(obj.get("note", ""))[:500],
        )
