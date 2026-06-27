import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";

const ITEMS = [
  {
    code: "C1",
    title: "Cross-slice 3D awareness",
    problem: "A single CBCT slice or panoramic crop can miss pathology that's only visible across neighboring slices or the full arch.",
    how: "C1 applies cross-slice attention so the model weighs evidence from adjacent slices and views before committing to a detection, instead of judging each frame in isolation.",
    where: "CBCT volumes and full-mouth series (FMS) only — single-image modalities like an isolated bitewing skip this step.",
  },
  {
    code: "C2",
    title: "Consistency filter",
    problem: "Detection models hallucinate findings that look plausible in one view but don't hold up against corroborating evidence — this is the main source of dentist distrust.",
    how: "C2 cross-checks every candidate detection against neighboring slices, repeat views, or symmetry priors. A finding that disappears under scrutiny gets dropped before it ever reaches the report.",
    where: "Runs on every modality, with the strongest effect on CBCT and FMS where multiple views of the same tooth are available.",
  },
  {
    code: "C3",
    title: "Per-tooth clustering",
    problem: "A dentist doesn't want six separate detection rows for one tooth — caries, bone loss, and a periapical lesion on tooth 26 are one clinical story, not three.",
    how: "C3 groups all findings by FDI tooth number and matches the combination against known compound patterns (e.g. 'caries_with_bone'), producing a single clinical cluster per tooth.",
    where: "Applies after FDI assignment, across all modalities.",
  },
  {
    code: "C4",
    title: "Treatment prompt",
    problem: "A label like 'dentin caries, 91% confidence' tells a dentist what was found, not what to do about it.",
    how: "C4 turns each cluster into ranked treatment options. Common patterns use fast template-based prompts (Tier 1); ambiguous or compound patterns escalate to DentVLM (Tier 2) for a reasoned, ranked recommendation.",
    where: "All modalities, every cluster — every output carries the second-opinion disclaimer.",
  },
];

export function MadClipAccordion() {
  return (
    <Accordion type="single" collapsible className="mx-auto flex max-w-3xl flex-col gap-4">
      {ITEMS.map((item) => (
        <AccordionItem key={item.code} value={item.code}>
          <AccordionTrigger>
            <span className="flex items-center gap-3">
              <Badge variant="purple" className="font-mono">
                {item.code}
              </Badge>
              {item.title}
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="flex flex-col gap-3">
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-text">
                  Problem
                </p>
                <p>{item.problem}</p>
              </div>
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-text">
                  How it works
                </p>
                <p>{item.how}</p>
              </div>
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-text">
                  Where it activates
                </p>
                <p>{item.where}</p>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}
