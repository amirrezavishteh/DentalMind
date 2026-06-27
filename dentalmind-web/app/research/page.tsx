import type { Metadata } from "next";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { PaperCard } from "@/components/shared/PaperCard";
import { ArrowUpRight } from "lucide-react";

export const metadata: Metadata = {
  title: "Research",
  description:
    "The peer-reviewed datasets and papers behind DentalMind's detection, segmentation, and shared-encoder architecture.",
};

const PAPERS = [
  {
    title: "DENTEX: Dental Enumeration and Diagnosis on Panoramic X-rays",
    venue: "MICCAI Challenge",
    year: "2023",
    summary: "Quadrant + tooth enumeration + disease annotations on panoramic X-rays — the basis of our OPG detection head.",
    href: "#",
    tag: "Dataset",
  },
  {
    title: "HUNT4 Oral Health Study",
    venue: "Population cohort",
    year: "2017–2019",
    summary: "Large-scale bitewing radiograph cohort used to validate caries and calculus detection at population scale.",
    href: "#",
    tag: "Dataset",
  },
  {
    title: "DentalSegmentator: open-source segmentation of CBCT images",
    venue: "Journal of Dentistry",
    year: "2024",
    summary: "Open-source CBCT segmentation model and weights — the starting point for our 3D pipeline.",
    href: "#",
    tag: "CBCT",
  },
  {
    title: "MMKD-CLIP: Multimodal Knowledge Distillation for Medical CLIP",
    venue: "arXiv 2506.22567",
    year: "2025",
    summary: "Zero-shot modality classification backbone — powers our automatic modality router.",
    href: "https://arxiv.org/abs/2506.22567",
    tag: "Backbone",
  },
  {
    title: "DentVFM: A Dental Vision Foundation Model",
    venue: "Preprint",
    year: "2024",
    summary: "Shared ViT-L/14 encoder pretrained on 1.6M dental images across modalities — our shared 2D backbone.",
    href: "#",
    tag: "Backbone",
  },
  {
    title: "nnU-Net: a self-configuring method for biomedical image segmentation",
    venue: "Nature Methods",
    year: "2021",
    summary: "The self-configuring 3D segmentation framework underlying our CBCT detection head (nnU-Net v2).",
    href: "#",
    tag: "CBCT",
  },
  {
    title: "ToothFairy3: 43-structure CBCT segmentation challenge",
    venue: "MICCAI Challenge",
    year: "2025",
    summary: "Anatomical structure segmentation benchmark guiding our CBCT structure taxonomy.",
    href: "#",
    tag: "CBCT",
  },
];

const DATASETS = [
  { name: "DENTEX", modality: "Panoramic (OPG)", size: "~4,000 images", license: "CC-BY", source: "Zenodo 7812323" },
  { name: "HUNT4 Oral Health", modality: "Bitewing", size: "Cohort-scale", license: "Research", source: "Population study" },
  { name: "Zenodo Dental Cavity", modality: "Periapical / caries", size: "2,038 images", license: "CC", source: "Zenodo 4907880" },
  { name: "Mendeley Mandible", modality: "Panoramic + mandible seg", size: "348 images", license: "CC-BY", source: "Mendeley hxt48yk462" },
  { name: "Tufts Dental DB 2022", modality: "Panoramic + masks", size: "9,005 images", license: "Research", source: "Kaggle mirror" },
  { name: "DentalSegmentator CBCT", modality: "CBCT 3D", size: "Multi-center", license: "CC-BY", source: "Zenodo" },
];

const OPEN_SOURCE = [
  { name: "DENTEX dataset", href: "https://zenodo.org/records/7812323", note: "Zenodo · CC-BY" },
  { name: "DentalSegmentator weights", href: "https://zenodo.org", note: "Zenodo" },
  { name: "MMKD-CLIP", href: "https://arxiv.org/abs/2506.22567", note: "arXiv + GitHub" },
];

export default function ResearchPage() {
  return (
    <main className="py-16">
      <div className="container">
        <SectionHeading
          eyebrow="Research"
          title="Built on peer-reviewed science"
          description="Every component in the DentalMind pipeline traces back to a published dataset, model, or benchmark."
        />

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {PAPERS.map((paper) => (
            <PaperCard key={paper.title} {...paper} />
          ))}
        </div>
      </div>

      <div className="container mt-24">
        <SectionHeading
          eyebrow="Datasets"
          title="Training data inventory"
          align="left"
          className="mx-0"
        />
        <div className="overflow-x-auto rounded-card-lg border border-border">
          <table className="w-full min-w-[640px] border-collapse text-left">
            <thead>
              <tr className="border-b border-border bg-surface">
                <th className="px-6 py-4 text-sm font-bold text-text">Dataset</th>
                <th className="px-6 py-4 text-sm font-bold text-text">Modality</th>
                <th className="px-6 py-4 text-sm font-bold text-text">Size</th>
                <th className="px-6 py-4 text-sm font-bold text-text">License</th>
                <th className="px-6 py-4 text-sm font-bold text-text">Source</th>
              </tr>
            </thead>
            <tbody>
              {DATASETS.map((d, i) => (
                <tr key={d.name} className={i % 2 === 0 ? "bg-background" : "bg-surface/40"}>
                  <td className="px-6 py-4 text-sm font-medium text-text">{d.name}</td>
                  <td className="px-6 py-4 text-sm text-text-muted">{d.modality}</td>
                  <td className="px-6 py-4 text-sm text-text-muted">{d.size}</td>
                  <td className="px-6 py-4 text-sm text-text-muted">{d.license}</td>
                  <td className="px-6 py-4 text-sm text-text-muted">{d.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="container mt-24">
        <SectionHeading
          eyebrow="Open source"
          title="What we built on"
          align="left"
          className="mx-0"
        />
        <div className="grid gap-4 sm:grid-cols-3">
          {OPEN_SOURCE.map((item) => (
            <a
              key={item.name}
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center justify-between rounded-card-lg border border-border bg-surface p-5 transition-colors hover:border-primary/40"
            >
              <div>
                <p className="text-sm font-semibold text-text">{item.name}</p>
                <p className="text-xs text-text-muted">{item.note}</p>
              </div>
              <ArrowUpRight className="h-4 w-4 text-text-muted transition-colors group-hover:text-primary" />
            </a>
          ))}
        </div>
      </div>
    </main>
  );
}
