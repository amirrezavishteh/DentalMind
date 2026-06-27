"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Info } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { UploadZone } from "@/components/demo/UploadZone";
import { AnnotatedImage } from "@/components/demo/AnnotatedImage";
import { FindingsPanel } from "@/components/demo/FindingsPanel";
import { cn } from "@/lib/utils";
import { demoResults, type DemoResult, type Modality } from "@/lib/demo-data";

const SAMPLES: { modality: Modality; label: string }[] = [
  { modality: "bitewing", label: "BW sample" },
  { modality: "panoramic", label: "OPG sample" },
  { modality: "periapical", label: "PA sample" },
];

export default function DemoPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [activeSample, setActiveSample] = useState<Modality | null>(null);

  const runAnalysis = async (modality: Modality) => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/demo/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modality }),
      });
      const data = await res.json();
      await new Promise((resolve) => setTimeout(resolve, 1600));
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="py-16">
      <div className="container">
        <SectionHeading
          eyebrow="Live demo"
          title="See the pipeline output"
          description="Upload a radiograph or pick a sample below. This demo uses pre-computed results — no real AI runs in your browser."
        />

        <div className="mx-auto mb-8 flex max-w-3xl items-start gap-2 rounded-card border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-primary">
          <Info className="mt-0.5 h-4 w-4 shrink-0" />
          <span>This demo uses pre-computed results. Contact us to connect your PACS.</span>
        </div>

        <div className="mx-auto mb-10 flex max-w-3xl flex-wrap justify-center gap-3">
          {SAMPLES.map((sample) => (
            <button
              key={sample.modality}
              onClick={() => {
                setActiveSample(sample.modality);
                runAnalysis(sample.modality);
              }}
              className={cn(
                "rounded-pill border px-4 py-2 text-sm font-medium transition-colors",
                activeSample === sample.modality
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-surface text-text-muted hover:border-primary/50 hover:text-text",
              )}
            >
              {sample.label}
            </button>
          ))}
        </div>

        <div className="mx-auto max-w-3xl">
          <UploadZone
            loading={loading}
            onAnalyze={(modality) => {
              setActiveSample(modality);
              runAnalysis(modality);
            }}
          />
        </div>

        <AnimatePresence mode="wait">
          {loading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mt-16 flex flex-col items-center gap-4 text-text-muted"
            >
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm">
                Running quality gate &middot; modality router &middot; detection heads &middot;
                C2/C3/C4...
              </p>
            </motion.div>
          )}

          {!loading && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="mt-16 grid gap-8 lg:grid-cols-2"
            >
              <AnnotatedImage result={result} />
              <FindingsPanel result={result} />
            </motion.div>
          )}
        </AnimatePresence>

        {!loading && !result && (
          <p className="mt-16 text-center text-sm text-text-muted">
            Demo results for {demoResults.bitewing.clusters.length +
              demoResults.panoramic.clusters.length +
              demoResults.periapical.clusters.length}{" "}
            pre-computed findings across 3 modalities will appear here.
          </p>
        )}
      </div>
    </main>
  );
}
