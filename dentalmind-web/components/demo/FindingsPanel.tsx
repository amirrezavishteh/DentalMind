"use client";

import { motion } from "framer-motion";
import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Disclaimer } from "@/components/shared/Disclaimer";
import { severityMeta, urgencyLabel, type DemoResult } from "@/lib/demo-data";

export function FindingsPanel({ result }: { result: DemoResult }) {
  return (
    <div className="flex flex-col gap-4">
      {result.clusters.map((cluster, i) => {
        const meta = severityMeta[cluster.severity];
        return (
          <motion.div
            key={cluster.tooth_fdi + i}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i, duration: 0.4 }}
            className="rounded-card-lg border border-border bg-surface p-5"
          >
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-mono text-base font-bold text-text">
                Tooth {cluster.tooth_fdi}
              </h3>
              <div className="flex items-center gap-2">
                <Badge variant={meta.badge}>{meta.label}</Badge>
                <span className="text-xs text-text-muted">
                  {urgencyLabel[cluster.urgency] ?? cluster.urgency}
                </span>
              </div>
            </div>

            <p className="mb-3 font-mono text-xs text-text-muted">
              pattern: <span className="text-secondary">{cluster.pattern}</span>
            </p>

            <ul className="mb-4 space-y-1.5 border-l border-border pl-4">
              {cluster.findings.map((f, fi) => (
                <li key={fi} className="font-mono text-xs text-text">
                  {f.class_name}
                  {f.surface ? ` (${f.surface})` : ""} — confidence{" "}
                  <span className="text-primary">{Math.round(f.confidence * 100)}%</span>
                  {f.bone_loss_mm ? (
                    <span className="text-warning"> &middot; {f.bone_loss_mm}mm</span>
                  ) : null}
                </li>
              ))}
            </ul>

            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
              Treatment options
            </p>
            <ol className="mb-3 space-y-1.5">
              {cluster.prompts.options.map((opt) => (
                <li key={opt.rank} className="text-sm text-text">
                  <span className="mr-2 font-mono text-primary">{opt.rank}.</span>
                  {opt.text}
                </li>
              ))}
            </ol>

            {cluster.prompts.red_flags.length > 0 && (
              <div className="mb-3 flex items-start gap-2 text-xs text-danger">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{cluster.prompts.red_flags.join(" · ")}</span>
              </div>
            )}
          </motion.div>
        );
      })}

      <Disclaimer />
    </div>
  );
}
