"use client";

import { motion } from "framer-motion";
import { ArrowRight, Layers, ShieldCheck, GitMerge, ClipboardList } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { cn } from "@/lib/utils";

const MADCLIP = [
  {
    code: "C1",
    icon: Layers,
    title: "Cross-slice 3D awareness",
    description: "Sees neighbors, not just one slice",
    accent: "success",
  },
  {
    code: "C2",
    icon: ShieldCheck,
    title: "Consistency filter",
    description: "Kills false positives before they reach you",
    accent: "success",
  },
  {
    code: "C3",
    icon: GitMerge,
    title: "Per-tooth clustering",
    description: "Tooth 26: caries + bone loss = one compound finding",
    accent: "purple",
  },
  {
    code: "C4",
    icon: ClipboardList,
    title: "Treatment prompt",
    description: "Ranked options, not just a label",
    accent: "purple",
  },
] as const;

const accentMap = {
  success: "border-success/30 bg-success/10 text-success",
  purple: "border-secondary/30 bg-secondary/10 text-secondary",
};

export function MadClipSection() {
  return (
    <section className="py-20">
      <div className="container">
        <SectionHeading
          eyebrow="The solution"
          title="MadClip: four components, one coherent output"
          description="Each finding passes through all four before it ever reaches a dentist's screen."
        />
        <div className="grid gap-4 lg:grid-cols-4">
          {MADCLIP.map((c, i) => (
            <div key={c.code} className="flex items-center gap-4">
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ delay: i * 0.15, duration: 0.5 }}
                className="flex-1 rounded-card-lg border border-border bg-surface p-6"
              >
                <div className="mb-4 flex items-center justify-between">
                  <span className="font-mono text-xs font-bold tracking-widest text-text-muted">
                    {c.code}
                  </span>
                  <div
                    className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-card border",
                      accentMap[c.accent],
                    )}
                  >
                    <c.icon className="h-5 w-5" />
                  </div>
                </div>
                <h3 className="mb-2 text-base font-bold text-text">{c.title}</h3>
                <p className="text-sm leading-relaxed text-text-muted">{c.description}</p>
              </motion.div>
              {i < MADCLIP.length - 1 && (
                <ArrowRight className="hidden h-5 w-5 shrink-0 text-text-muted lg:block" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
