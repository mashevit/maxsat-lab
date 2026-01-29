from __future__ import annotations

class NoopProvider:
    def complete(self, prompt: str) -> str:
        # Always return "do nothing" advice
        return '{"flip": [], "set_true": [], "set_false": [], "note": "noop"}'
