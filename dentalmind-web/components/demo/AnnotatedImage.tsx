"use client";

import { motion } from "framer-motion";
import type { DemoResult } from "@/lib/demo-data";

const TYPE_LABEL: Record<string, string> = {
  caries: "Caries",
  bone_loss: "Bone loss",
  calculus: "Calculus",
  impaction: "Impaction",
  lesion: "Periapical lesion",
  restoration: "Restoration",
};

export function AnnotatedImage({ result }: { result: DemoResult }) {
  return (
    <div className="relative overflow-hidden rounded-card-lg border border-border bg-background">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={result.image} alt={`${result.modality} sample radiograph`} className="w-full" />

      {result.overlays.map((dot, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.15 * i, duration: 0.3 }}
          className="group absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${dot.x}%`, top: `${dot.y}%` }}
        >
          <span
            className="block h-3.5 w-3.5 animate-pulse-glow rounded-full ring-2 ring-background"
            style={{ backgroundColor: dot.color }}
          />
          <span className="absolute left-1/2 top-full z-10 mt-2 -translate-x-1/2 whitespace-nowrap rounded-pill border border-border bg-background px-2 py-1 text-[10px] font-mono text-text opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
            FDI {dot.tooth} &middot; {TYPE_LABEL[dot.type] ?? dot.type}
          </span>
        </motion.div>
      ))}

      <div className="absolute bottom-3 right-3 rounded-pill border border-border bg-background/90 px-3 py-1 text-[10px] font-mono uppercase tracking-wide text-text-muted">
        FDI numbering
      </div>
    </div>
  );
}
