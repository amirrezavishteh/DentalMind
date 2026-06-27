import { Check, X } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";

const ROWS = [
  { feature: "All 5 modalities", others: false, dentalmind: true },
  { feature: "Per-tooth clustering", others: false, dentalmind: true, dentalmindNote: "C3" },
  { feature: "Treatment prompt", others: false, dentalmind: true, dentalmindNote: "C4" },
  {
    feature: "FP reduction",
    others: "Clinical layer",
    dentalmind: "Image-level",
    dentalmindNote: "C2",
  },
  { feature: "Open source backbone", others: false, dentalmind: true, dentalmindNote: "MMKD-CLIP" },
  { feature: "2D + 3D joint train", others: false, dentalmind: true },
];

function Cell({ value }: { value: boolean | string }) {
  if (typeof value === "string") {
    return <span className="text-sm text-text">{value}</span>;
  }
  return value ? (
    <Check className="h-5 w-5 text-success" />
  ) : (
    <X className="h-5 w-5 text-danger" />
  );
}

export function ComparisonTable() {
  return (
    <section className="py-20">
      <div className="container">
        <SectionHeading
          eyebrow="Comparison"
          title="How DentalMind stacks up"
          description="Against the incumbent dental imaging AI vendors."
        />
        <div className="overflow-x-auto rounded-card-lg border border-border">
          <table className="w-full min-w-[640px] border-collapse text-left">
            <thead>
              <tr className="border-b border-border bg-surface">
                <th className="px-6 py-4 text-sm font-bold text-text">Feature</th>
                <th className="px-6 py-4 text-sm font-bold text-text-muted">
                  Pearl / Overjet / Videa
                </th>
                <th className="px-6 py-4 text-sm font-bold text-primary">DentalMind</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row, i) => (
                <tr
                  key={row.feature}
                  className={i % 2 === 0 ? "bg-background" : "bg-surface/40"}
                >
                  <td className="px-6 py-4 text-sm font-medium text-text">{row.feature}</td>
                  <td className="px-6 py-4">
                    <Cell value={row.others} />
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <Cell value={row.dentalmind} />
                      {row.dentalmindNote && (
                        <span className="rounded-pill border border-secondary/30 bg-secondary/10 px-2 py-0.5 font-mono text-[10px] text-secondary">
                          {row.dentalmindNote}
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
