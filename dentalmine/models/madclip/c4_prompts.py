"""MadClip C4 — generate ranked treatment prompts from per-tooth clusters.

Tier 1 (default, always safe): fill a YAML template matched by cluster.pattern.
Tier 2 (--use-vlm): MMKD-CLIP grounding + Qwen2-VL text generation. Stubbed in
Phase 1 — falls back to Tier 1 with a clear notice (heavy VLM not wired/tested
locally). TODO: implement two-step grounding + generation.
"""
from __future__ import annotations

from typing import List, Optional

import yaml
from rich.console import Console

from models.madclip.c3_clustering import PATTERN_LIBRARY
from paths import DEFAULT_TEMPLATES, resolve
from schema.finding import ToothCluster, Urgency
from schema.patient import PatientContext
from schema.prompt import DISCLAIMER, PromptOutput, TreatmentOption

_console = Console()

_URGENCY_TEXT = {
    Urgency.IMMEDIATE: "address within 24 hours",
    Urgency.SOON: "schedule within ~2 weeks",
    Urgency.ROUTINE: "handle at the next routine visit",
    Urgency.MONITOR: "monitor and re-evaluate in ~6 months",
}


class TreatmentPromptGenerator:
    def __init__(self, templates_path=None, medclip=None):
        path = resolve(templates_path) if templates_path else DEFAULT_TEMPLATES
        with open(path, "r", encoding="utf-8") as fh:
            self.templates = yaml.safe_load(fh) or {}
        self.medclip = medclip
        self._warn_missing_templates()

    def _warn_missing_templates(self):
        missing = [p for p in PATTERN_LIBRARY if p not in self.templates]
        if missing:
            _console.print(f"[yellow]C4: templates missing for patterns: {missing}[/yellow]")

    # ---- public --------------------------------------------------------
    def generate(
        self,
        clusters: List[ToothCluster],
        patient_context: Optional[PatientContext] = None,
        use_vlm: bool = False,
        image_path: Optional[str] = None,
    ) -> List[PromptOutput]:
        prompts = []
        for c in clusters:
            if use_vlm:
                prompts.append(self.generate_from_vlm(c, image_path, patient_context))
            else:
                prompts.append(self.generate_from_template(c, patient_context))
        return prompts

    # ---- Tier 1 --------------------------------------------------------
    def generate_from_template(
        self, cluster: ToothCluster, patient_context: Optional[PatientContext] = None
    ) -> PromptOutput:
        tpl = self.templates.get(cluster.pattern)
        if tpl is None:
            return self._fallback_prompt(cluster)

        fmt = self._format_kwargs(cluster, patient_context)
        options = [
            TreatmentOption(
                rank=o["rank"],
                text=self._safe_format(o["text"], fmt),
                when_to_use=self._safe_format(o.get("when_to_use", ""), fmt),
            )
            for o in tpl.get("options", [])
        ]
        return PromptOutput(
            tooth_fdi=cluster.tooth_fdi,
            pattern=cluster.pattern,
            options=options,
            tests_suggested=[self._safe_format(t, fmt) for t in tpl.get("tests_suggested", [])],
            urgency=cluster.urgency,
            red_flags=[self._safe_format(r, fmt) for r in tpl.get("red_flags", [])],
            tier="template",
            disclaimer=DISCLAIMER,
        )

    # ---- Tier 2 (stub) -------------------------------------------------
    def generate_from_vlm(
        self,
        cluster: ToothCluster,
        image_path: Optional[str],
        patient_context: Optional[PatientContext] = None,
    ) -> PromptOutput:
        # TODO(Phase 2): Step A MMKD-CLIP grounding + Step B Qwen2-VL generation.
        _console.print(
            "[yellow]C4 Tier-2 VLM not implemented in Phase 1 — using Tier-1 template.[/yellow]"
        )
        out = self.generate_from_template(cluster, patient_context)
        return out

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _format_kwargs(cluster: ToothCluster, pc: Optional[PatientContext]) -> dict:
        surface = ", ".join(cluster.surfaces_affected) if cluster.surfaces_affected else "the affected"
        return {
            "tooth": cluster.tooth_fdi,
            "surface": surface,
            "bone_mm": f"{cluster.bone_loss_mm:.1f}" if cluster.bone_loss_mm else "n/a",
            "age": pc.age if pc and pc.age is not None else "n/a",
            "urgency_text": _URGENCY_TEXT.get(cluster.urgency, ""),
        }

    @staticmethod
    def _safe_format(text: str, kwargs: dict) -> str:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text

    @staticmethod
    def _fallback_prompt(cluster: ToothCluster) -> PromptOutput:
        return PromptOutput(
            tooth_fdi=cluster.tooth_fdi,
            pattern=cluster.pattern,
            options=[TreatmentOption(
                rank=1,
                text="Clinical evaluation recommended (no template for this pattern).",
                when_to_use="General",
            )],
            tests_suggested=["Clinical examination"],
            urgency=cluster.urgency,
            red_flags=[],
            tier="template",
            disclaimer=DISCLAIMER,
        )
