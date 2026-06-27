"use client";

import { useState } from "react";
import Link from "next/link";
import { Check, Minus } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

interface Tier {
  name: string;
  monthly: number | null;
  modalities: string;
  findings: string;
  treatmentPrompts: string;
  pacs: boolean;
  cbct: boolean;
  sla: string;
  support: string;
  highlight?: boolean;
}

const TIERS: Tier[] = [
  {
    name: "Starter",
    monthly: 199,
    modalities: "BW + OPG",
    findings: "500 / month",
    treatmentPrompts: "Tier 1 only",
    pacs: false,
    cbct: false,
    sla: "—",
    support: "Email",
  },
  {
    name: "Clinic",
    monthly: 599,
    modalities: "All 5",
    findings: "Unlimited",
    treatmentPrompts: "Tier 1 + 2",
    pacs: true,
    cbct: true,
    sla: "99.9%",
    support: "Priority",
    highlight: true,
  },
  {
    name: "Enterprise",
    monthly: null,
    modalities: "All 5 + API",
    findings: "Unlimited",
    treatmentPrompts: "Full",
    pacs: true,
    cbct: true,
    sla: "99.99%",
    support: "Dedicated",
  },
];

function Cell({ value }: { value: boolean | string }) {
  if (typeof value === "string") return <span className="text-sm text-text">{value}</span>;
  return value ? (
    <Check className="h-4 w-4 text-success" />
  ) : (
    <Minus className="h-4 w-4 text-text-muted" />
  );
}

export default function PricingPage() {
  const [annual, setAnnual] = useState(false);

  return (
    <main className="py-16">
      <div className="container">
        <SectionHeading
          eyebrow="Pricing"
          title="Plans that scale with your clinic"
          description="All plans include FDA-compliant second-opinion framing, audit logs, and DICOM support."
        />

        <div className="mb-12 flex items-center justify-center gap-3">
          <span className={cn("text-sm", !annual ? "text-text" : "text-text-muted")}>Monthly</span>
          <Switch checked={annual} onCheckedChange={setAnnual} />
          <span className={cn("text-sm", annual ? "text-text" : "text-text-muted")}>
            Annual <span className="text-success">(save 20%)</span>
          </span>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {TIERS.map((tier) => {
            const price =
              tier.monthly === null
                ? null
                : annual
                  ? Math.round(tier.monthly * 0.8)
                  : tier.monthly;
            return (
              <div
                key={tier.name}
                className={cn(
                  "flex flex-col rounded-card-lg border bg-surface p-8",
                  tier.highlight ? "border-primary shadow-glow-cyan" : "border-border",
                )}
              >
                {tier.highlight && (
                  <span className="mb-4 w-fit rounded-pill border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                    Most popular
                  </span>
                )}
                <h3 className="text-xl font-bold text-text">{tier.name}</h3>
                <div className="mt-4 flex items-baseline gap-1">
                  {price === null ? (
                    <span className="text-3xl font-extrabold text-text">Custom</span>
                  ) : (
                    <>
                      <span className="text-4xl font-extrabold text-text">${price}</span>
                      <span className="text-sm text-text-muted">/mo</span>
                    </>
                  )}
                </div>
                {annual && price !== null && (
                  <p className="mt-1 text-xs text-text-muted">billed annually</p>
                )}

                <ul className="mt-6 flex flex-1 flex-col gap-3 text-sm">
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">Modalities</span>
                    <Cell value={tier.modalities} />
                  </li>
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">Findings / month</span>
                    <Cell value={tier.findings} />
                  </li>
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">Treatment prompts</span>
                    <Cell value={tier.treatmentPrompts} />
                  </li>
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">PACS integration</span>
                    <Cell value={tier.pacs} />
                  </li>
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">CBCT 3D</span>
                    <Cell value={tier.cbct} />
                  </li>
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">SLA</span>
                    <Cell value={tier.sla} />
                  </li>
                  <li className="flex items-center justify-between">
                    <span className="text-text-muted">Support</span>
                    <Cell value={tier.support} />
                  </li>
                </ul>

                <Link href="/contact" className="mt-8">
                  <Button
                    variant={tier.highlight ? "primary" : "ghost"}
                    className="w-full"
                  >
                    Start free 14-day trial
                  </Button>
                </Link>
              </div>
            );
          })}
        </div>

        <p className="mt-10 text-center text-xs text-text-muted">
          No payment integration in this demo — every &ldquo;Start trial&rdquo; button routes to
          our contact form.
        </p>
      </div>
    </main>
  );
}
