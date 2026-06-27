"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const STEPS = [
  { title: "Image Input → Quality Gate", detail: "Rejects underexposed, cropped, or wrong-orientation images before they reach the model." },
  { title: "Modality Router (MMKD-CLIP zero-shot)", detail: "Classifies bitewing / panoramic / periapical / FMS / CBCT without a dedicated classifier." },
  { title: "Shared DentVFM-2D Encoder", detail: "ViT-L/14, DINOv2-pretrained on 1.6M dental images — one backbone for every 2D modality." },
  { title: "C1: Cross-slice Attention", detail: "CBCT and full-mouth series only — attends across neighboring slices, not a single frame." },
  { title: "Modality-specific Detection Head", detail: "Each modality branches into its own fine-tuned head after the shared encoder." },
  { title: "C2: Consistency Filter", detail: "Cross-checks detections against neighboring views before they count as a finding." },
  { title: "NMS + FDI Assignment", detail: "De-duplicates overlapping boxes and maps every detection to an FDI tooth number." },
  { title: "C3: Per-tooth Clustering", detail: "Groups multiple findings on one tooth into a single compound clinical pattern." },
  { title: "Overlay Renderer", detail: "Draws the color-coded annotation layer dentists actually see on the image." },
  { title: "C4: Treatment Prompt", detail: "Template Tier 1 for common patterns, DentVLM Tier 2 for ranked, reasoned options." },
];

export function PipelineDiagram() {
  return (
    <div className="relative mx-auto max-w-2xl">
      <div className="absolute left-5 top-0 h-full w-px bg-border" aria-hidden />
      <div className="flex flex-col gap-8">
        {STEPS.map((step, i) => (
          <motion.div
            key={step.title}
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.4 }}
            className="relative flex gap-5 pl-0"
          >
            <div
              className={cn(
                "relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-pill border-2 font-mono text-xs font-bold",
                step.title.startsWith("C")
                  ? "border-secondary bg-secondary/10 text-secondary"
                  : "border-primary bg-background text-primary",
              )}
            >
              {i + 1}
            </div>
            <div className="rounded-card-lg border border-border bg-surface px-5 py-4">
              <h3 className="mb-1 text-sm font-bold text-text">{step.title}</h3>
              <p className="text-sm leading-relaxed text-text-muted">{step.detail}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
