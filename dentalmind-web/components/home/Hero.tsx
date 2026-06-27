"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ToothIcon } from "@/components/shared/ToothIcon";

const HEADLINE = "AI That Thinks Like a Dentist".split(" ");

const OVERLAY_DOTS = [
  { top: "28%", left: "32%", color: "#EF4444", delay: 0.2 },
  { top: "44%", left: "58%", color: "#F59E0B", delay: 0.6 },
  { top: "62%", left: "40%", color: "#00D4FF", delay: 1.0 },
  { top: "36%", left: "70%", color: "#7C3AED", delay: 1.4 },
  { top: "55%", left: "22%", color: "#10B981", delay: 1.8 },
];

export function Hero() {
  return (
    <section className="relative overflow-hidden bg-grid-glow">
      <div className="absolute inset-0 grid-pattern opacity-30" />
      <div className="container relative grid gap-12 py-20 lg:grid-cols-2 lg:py-28">
        <div className="flex flex-col justify-center">
          <motion.span
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 inline-flex w-fit items-center rounded-pill border border-primary/30 bg-primary/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-widest text-primary"
          >
            5 modalities &middot; 1 pipeline
          </motion.span>

          <h1 className="text-balance text-4xl font-extrabold leading-tight text-text sm:text-5xl lg:text-6xl">
            {HEADLINE.map((word, i) => (
              <motion.span
                key={word + i}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08, duration: 0.5 }}
                className={`mr-3 inline-block ${word === "Dentist" ? "text-gradient-cyan" : ""}`}
              >
                {word}
              </motion.span>
            ))}
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.5 }}
            className="mt-6 max-w-xl text-balance text-lg text-text-muted"
          >
            DentalMind analyzes all 5 X-ray modalities in one pipeline, clusters findings per
            tooth, and gives your team ranked treatment prompts — not a list of bounding boxes.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.5 }}
            className="mt-8 flex flex-wrap gap-4"
          >
            <Link href="/demo">
              <Button size="lg" variant="primary">
                Try live demo
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/research">
              <Button size="lg" variant="ghost">
                Read the research
              </Button>
            </Link>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="relative flex items-center justify-center"
        >
          <div className="relative flex aspect-square w-full max-w-md items-center justify-center rounded-card-lg border border-border bg-surface">
            <ToothIcon className="h-40 w-40 text-primary/20" />
            {OVERLAY_DOTS.map((dot, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: dot.delay, duration: 0.4 }}
                className="absolute h-3 w-3 animate-pulse-glow rounded-full"
                style={{ top: dot.top, left: dot.left, backgroundColor: dot.color }}
              />
            ))}
            <span className="absolute bottom-4 right-4 rounded-pill border border-border bg-background/80 px-3 py-1 text-xs font-mono text-text-muted">
              FDI 26 &middot; 91% conf
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
