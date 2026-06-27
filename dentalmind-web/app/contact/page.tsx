"use client";

import { useState, type FormEvent } from "react";
import { Mail, Calendar, CheckCircle2 } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const ROLES = ["Dentist", "Radiologist", "IT Director", "Investor", "Researcher", "Other"];
const SOURCES = [
  "Search engine",
  "Conference / event",
  "Referral",
  "Social media",
  "Research paper",
  "Other",
];

export default function ContactPage() {
  const [role, setRole] = useState("Dentist");
  const [source, setSource] = useState(SOURCES[0]);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("loading");
    const form = new FormData(e.currentTarget);
    const payload = {
      name: form.get("name"),
      email: form.get("email"),
      organization: form.get("organization"),
      country: form.get("country"),
      role,
      message: form.get("message"),
      source,
    };

    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("failed");
      setStatus("success");
    } catch {
      setStatus("error");
    }
  }

  return (
    <main className="py-16">
      <div className="container">
        <SectionHeading
          eyebrow="Contact"
          title="Talk to the team"
          description="Tell us about your clinic, hospital, or research group — we'll follow up within one business day."
        />

        <div className="grid gap-10 lg:grid-cols-3">
          <div className="lg:col-span-2 rounded-card-lg border border-border bg-surface p-8">
            {status === "success" ? (
              <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
                <CheckCircle2 className="h-12 w-12 text-success" />
                <h3 className="text-xl font-bold text-text">Message sent</h3>
                <p className="max-w-sm text-sm text-text-muted">
                  Thanks for reaching out — someone from the DentalMind team will follow up
                  shortly.
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="grid gap-5 sm:grid-cols-2">
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="name" className="text-sm font-medium text-text">
                    Name
                  </label>
                  <input
                    id="name"
                    name="name"
                    required
                    className="h-11 rounded-card border border-border bg-background px-4 text-sm text-text outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="email" className="text-sm font-medium text-text">
                    Email
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required
                    className="h-11 rounded-card border border-border bg-background px-4 text-sm text-text outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="organization" className="text-sm font-medium text-text">
                    Clinic / Hospital name
                  </label>
                  <input
                    id="organization"
                    name="organization"
                    className="h-11 rounded-card border border-border bg-background px-4 text-sm text-text outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="country" className="text-sm font-medium text-text">
                    Country
                  </label>
                  <input
                    id="country"
                    name="country"
                    className="h-11 rounded-card border border-border bg-background px-4 text-sm text-text outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-sm font-medium text-text">Role</label>
                  <Select value={role} onValueChange={setRole}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ROLES.map((r) => (
                        <SelectItem key={r} value={r}>
                          {r}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-sm font-medium text-text">How did you hear about us?</label>
                  <Select value={source} onValueChange={setSource}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {SOURCES.map((s) => (
                        <SelectItem key={s} value={s}>
                          {s}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-1.5 sm:col-span-2">
                  <label htmlFor="message" className="text-sm font-medium text-text">
                    Message
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    rows={5}
                    required
                    className="rounded-card border border-border bg-background px-4 py-3 text-sm text-text outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                {status === "error" && (
                  <p className="sm:col-span-2 text-sm text-danger">
                    Something went wrong sending your message. Please try again.
                  </p>
                )}

                <Button
                  type="submit"
                  variant="primary"
                  size="lg"
                  disabled={status === "loading"}
                  className="sm:col-span-2"
                >
                  {status === "loading" ? "Sending..." : "Send message"}
                </Button>
              </form>
            )}
          </div>

          <div className="flex flex-col gap-4">
            <a
              href="mailto:hello@dentalmind.ai"
              className="flex items-center gap-3 rounded-card-lg border border-border bg-surface p-5 transition-colors hover:border-primary/40"
            >
              <Mail className="h-5 w-5 text-primary" />
              <div>
                <p className="text-sm font-semibold text-text">Email</p>
                <p className="text-xs text-text-muted">hello@dentalmind.ai</p>
              </div>
            </a>
            <a
              href="https://www.linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 rounded-card-lg border border-border bg-surface p-5 transition-colors hover:border-primary/40"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5 text-primary">
                <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.03-1.85-3.03-1.85 0-2.14 1.44-2.14 2.94v5.66H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.38-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45z" />
              </svg>
              <div>
                <p className="text-sm font-semibold text-text">LinkedIn</p>
                <p className="text-xs text-text-muted">Follow DentalMind</p>
              </div>
            </a>
            <div className="rounded-card-lg border border-border bg-surface p-5">
              <div className="mb-3 flex items-center gap-3">
                <Calendar className="h-5 w-5 text-primary" />
                <p className="text-sm font-semibold text-text">Book a call</p>
              </div>
              <p className="text-xs text-text-muted">
                Calendar booking embed (Cal.com) goes here once connected.
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
