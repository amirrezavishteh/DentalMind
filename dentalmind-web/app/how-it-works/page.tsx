import type { Metadata } from "next";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { PipelineDiagram } from "@/components/how-it-works/PipelineDiagram";
import { MadClipAccordion } from "@/components/how-it-works/MadClipAccordion";

export const metadata: Metadata = {
  title: "How It Works",
  description:
    "The DentalMind pipeline: quality gate, modality router, shared encoder, MadClip C1-C4, and ranked treatment prompts.",
};

const TRAINING_PHASES = [
  {
    phase: "Phase 1",
    title: "2D backbone pretraining",
    detail: "DentVFM-2D learns general dental representations across 1.6M images from public + licensed datasets.",
  },
  {
    phase: "Phase 2",
    title: "DRR generation",
    detail: "Every labeled CBCT volume is projected into synthetic 2D digitally-reconstructed radiographs at multiple angles.",
  },
  {
    phase: "Phase 3",
    title: "Joint 2D + 3D fine-tune",
    detail: "Real 2D radiographs and DRR-derived 2D images train the same encoder side by side.",
  },
  {
    phase: "Phase 4",
    title: "Modality head specialization",
    detail: "Detection heads fine-tune per modality on top of the shared, now richly-pretrained backbone.",
  },
];

export default function HowItWorksPage() {
  return (
    <main className="py-16">
      <div className="container">
        <SectionHeading
          eyebrow="Pipeline"
          title="From upload to treatment prompt"
          description="Ten steps, four of them MadClip components, all running on a shared encoder."
        />
        <PipelineDiagram />
      </div>

      <div className="container mt-28">
        <SectionHeading
          eyebrow="MadClip deep-dive"
          title="What each component actually does"
          description="No math — just the problem it solves, how it solves it, and where it runs."
        />
        <MadClipAccordion />
      </div>

      <div className="container mt-28">
        <SectionHeading
          eyebrow="Training strategy"
          title="1 labeled CBCT = 500 free 2D training images"
          description="Joint 2D + 3D training means every expensive 3D annotation pays for itself many times over."
        />
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {TRAINING_PHASES.map((p) => (
            <div key={p.phase} className="rounded-card-lg border border-border bg-surface p-6">
              <span className="mb-3 inline-block rounded-pill border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                {p.phase}
              </span>
              <h3 className="mb-2 text-base font-bold text-text">{p.title}</h3>
              <p className="text-sm leading-relaxed text-text-muted">{p.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
