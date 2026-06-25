"""Treatment-prompt output schema (C4 output)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from schema.finding import Urgency

DISCLAIMER = (
    "⚠ AI second opinion only. "
    "Final diagnosis and treatment plan subject to clinical judgment."
)


class TreatmentOption(BaseModel):
    rank: int
    text: str
    when_to_use: str


class PromptOutput(BaseModel):
    tooth_fdi: str
    pattern: str
    options: List[TreatmentOption]
    tests_suggested: List[str]
    urgency: Urgency
    red_flags: List[str]
    tier: str   # "template" | "vlm_advisory"
    disclaimer: str = DISCLAIMER

    def to_text(self) -> str:
        """Render a single per-tooth treatment card page."""
        lines = [
            f"=== Tooth {self.tooth_fdi} | {self.pattern} "
            f"| urgency: {self.urgency.value} | tier: {self.tier} ===",
            "",
            "Treatment options (ranked):",
        ]
        for opt in sorted(self.options, key=lambda o: o.rank):
            lines.append(f"  {opt.rank}. {opt.text}")
            lines.append(f"     when to use: {opt.when_to_use}")
        if self.tests_suggested:
            lines.append("")
            lines.append("Suggested tests:")
            lines.extend(f"  - {t}" for t in self.tests_suggested)
        if self.red_flags:
            lines.append("")
            lines.append("Red flags:")
            lines.extend(f"  ! {r}" for r in self.red_flags)
        lines.append("")
        lines.append(self.disclaimer)
        return "\n".join(lines)
