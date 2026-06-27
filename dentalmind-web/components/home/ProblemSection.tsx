"use client";

import { motion } from "framer-motion";
import { BoxSelect, ScanEye, ShieldAlert } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";

const PROBLEMS = [
  {
    icon: BoxSelect,
    title: "Bounding boxes, not decisions",
    description: "Current AI tools give you bounding boxes. You still have to think.",
  },
  {
    icon: ScanEye,
    title: "Single-modality blind spots",
    description: "Bitewing only. OPG only. Never the full picture across modalities.",
  },
  {
    icon: ShieldAlert,
    title: "False positives everywhere",
    description: "Dentists stop trusting the tool when every scan flags noise.",
  },
];

export function ProblemSection() {
  return (
    <section className="py-20">
      <div className="container">
        <SectionHeading
          eyebrow="The problem"
          title="Dental AI promised less work. It delivered more review."
          description="Three structural problems with today's dental imaging AI — and why they keep dentists from trusting the output."
        />
        <div className="grid gap-6 sm:grid-cols-3">
          {PROBLEMS.map((p, i) => (
            <motion.div
              key={p.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ delay: i * 0.12, duration: 0.5 }}
              className="rounded-card-lg border border-border bg-surface p-6"
            >
              <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-card border border-danger/30 bg-danger/10 text-danger">
                <p.icon className="h-5 w-5" />
              </div>
              <h3 className="mb-2 text-lg font-bold text-text">{p.title}</h3>
              <p className="text-sm leading-relaxed text-text-muted">{p.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
