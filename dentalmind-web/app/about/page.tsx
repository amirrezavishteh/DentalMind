import type { Metadata } from "next";
import { Layers, GitMerge, ShieldCheck, ArrowUpRight } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";

const RESEARCH_BLOG_URL = "https://amirrezavishteh.github.io/blog/";

export const metadata: Metadata = {
  title: "About",
  description:
    "DentalMind's mission, the gap in dental AI we're closing, and our regulatory-safe approach to clinical decision support.",
};

const PRINCIPLES = [
  {
    icon: Layers,
    title: "One encoder, not four",
    description:
      "Every 2D modality shares the same DentVFM backbone instead of training isolated models per modality. Shared representations mean a finding in one modality strengthens detection in another.",
  },
  {
    icon: GitMerge,
    title: "Cluster, don't dump",
    description:
      "A list of unrelated detections is not a clinical picture. C3 groups findings by tooth so what reaches the dentist is a compound pattern, not six disconnected rows.",
  },
  {
    icon: ShieldCheck,
    title: "Suggest, don't diagnose",
    description:
      "Every output is framed as a ranked option set with a second-opinion disclaimer — never a final diagnosis. This is the regulatory-safe foundation the rest of the product is built on.",
  },
];

export default function AboutPage() {
  return (
    <main className="py-16">
      <div className="container max-w-3xl">
        <SectionHeading
          eyebrow="About"
          title="Our mission"
          align="left"
          className="mx-0"
        />
        <p className="text-balance text-xl font-medium leading-relaxed text-text">
          We believe a dentist should spend their expertise on decisions, not on manually
          correlating 18 X-ray images. DentalMind does the correlation.
        </p>

        <div className="mt-16 flex flex-col gap-6 text-base leading-relaxed text-text-muted">
          <p>
            Dental AI has spent the last five years competing on detection accuracy — who can find
            the most caries, the most bone loss, the most subtle periapical lesion. That race
            produced genuinely good detectors. It also produced a UX problem: a dentist reviewing
            a full-mouth series now gets back dozens of unconnected bounding boxes, one per
            finding, with no sense of which ones describe the same clinical situation on the same
            tooth.
          </p>
          <p>
            The result is a tool that adds review work instead of removing it. Dentists either
            ignore the output or spend as long correlating AI detections as they would have spent
            reading the films themselves — and every false positive erodes trust a little further,
            until the tool gets switched off.
          </p>
          <p>
            DentalMind starts from a different premise: detection is necessary but not sufficient.
            The real product is the layer above detection — cross-checking findings for
            consistency, clustering them per tooth into compound patterns, and turning each pattern
            into a ranked, explainable set of treatment options. We built the full per-tooth
            pipeline first, even on top of a placeholder detector, because integration is the part
            that actually changes how a clinic works.
          </p>
        </div>
      </div>

      <div className="container mt-24">
        <SectionHeading eyebrow="Tech philosophy" title="Three principles" />
        <div className="grid gap-6 sm:grid-cols-3">
          {PRINCIPLES.map((p) => (
            <div key={p.title} className="rounded-card-lg border border-border bg-surface p-6">
              <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-card border border-primary/30 bg-primary/10 text-primary">
                <p.icon className="h-5 w-5" />
              </div>
              <h3 className="mb-2 text-lg font-bold text-text">{p.title}</h3>
              <p className="text-sm leading-relaxed text-text-muted">{p.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="container mt-24 max-w-3xl">
        <SectionHeading
          eyebrow="Beyond DentalMind"
          title="A broader focus: trustworthy AI for healthcare"
          align="left"
          className="mx-0"
        />
        <div className="flex flex-col gap-6 text-base leading-relaxed text-text-muted">
          <p>
            DentalMind is the first proof point of a wider research direction: building AI that
            healthcare practitioners can actually trust — systems that are transparent about their
            limits, resistant to the failure modes that erode confidence, and designed to support a
            clinician&apos;s judgment rather than override it. The &ldquo;suggest, don&apos;t
            diagnose&rdquo; principle behind this product is one expression of that thesis.
          </p>
          <p>
            I write about that broader research agenda — trustworthy AI and AI safety, applied to
            healthcare and beyond — on my personal research blog.
          </p>
        </div>
        <a
          href={RESEARCH_BLOG_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-6 inline-flex items-center gap-1.5 text-sm font-semibold text-primary transition-colors hover:text-primary/80"
        >
          Read the research blog
          <ArrowUpRight className="h-4 w-4" />
        </a>
      </div>

      <div className="container mt-24 max-w-3xl">
        <div className="rounded-card-lg border border-warning/30 bg-warning/10 p-8">
          <h3 className="mb-3 text-lg font-bold text-warning">Disclaimer</h3>
          <p className="text-sm leading-relaxed text-warning/90">
            DentalMind is a clinical decision support tool intended to provide a second opinion
            only. It is not a diagnostic device and does not replace the clinical judgment of a
            licensed dentist or radiologist. All findings and treatment suggestions must be
            independently reviewed and confirmed by a qualified clinician before any treatment
            decision is made.
          </p>
        </div>
      </div>
    </main>
  );
}
