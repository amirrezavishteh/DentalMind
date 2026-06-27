"use client";

import { motion } from "framer-motion";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { ToothIcon } from "@/components/shared/ToothIcon";
import { Badge } from "@/components/ui/badge";

const MODALITIES = [
  {
    name: "Bitewing",
    model: "YOLOv8x-seg",
    metricLabel: "Caries accuracy",
    metric: "87–95%",
  },
  {
    name: "Panoramic",
    model: "YOLOv11 + HierarchicalDet",
    metricLabel: "mAP",
    metric: "0.848",
  },
  {
    name: "Periapical",
    model: "Swin-T Hybrid",
    metricLabel: "Accuracy range",
    metric: "70–99%",
  },
  {
    name: "Full-Mouth Series",
    model: "Shared DentVFM-2D",
    metricLabel: "Per-series findings",
    metric: "Up to 18 images",
  },
  {
    name: "CBCT 3D",
    model: "nnU-Net v2",
    metricLabel: "Dice score",
    metric: "0.87",
  },
];

export function ModalitiesSection() {
  return (
    <section className="bg-surface/30 py-20">
      <div className="container">
        <SectionHeading
          eyebrow="Modalities"
          title="One pipeline. All five modalities."
          description="Every modality runs through the same shared encoder before branching into a specialized detection head."
        />
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-5">
          {MODALITIES.map((m, i) => (
            <motion.div
              key={m.name}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ delay: i * 0.08, duration: 0.5 }}
              className="flex flex-col rounded-card-lg border border-border bg-surface p-5"
            >
              <div className="mb-4 flex h-24 items-center justify-center rounded-card border border-border bg-background">
                <ToothIcon className="h-12 w-12 text-primary/30" />
              </div>
              <h3 className="mb-1 text-base font-bold text-text">{m.name}</h3>
              <p className="mb-3 font-mono text-xs text-text-muted">{m.model}</p>
              <Badge variant="cyan" className="w-fit">
                {m.metricLabel}: {m.metric}
              </Badge>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
