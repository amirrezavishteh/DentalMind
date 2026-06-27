import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

interface PaperCardProps {
  title: string;
  venue: string;
  year: string;
  summary: string;
  href: string;
  tag: string;
}

export function PaperCard({ title, venue, year, summary, href, tag }: PaperCardProps) {
  return (
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex flex-col rounded-card-lg border border-border bg-surface p-6 transition-all duration-300 hover:-translate-y-1 hover:border-primary/40"
    >
      <div className="mb-3 flex items-center justify-between">
        <span className="rounded-pill border border-secondary/30 bg-secondary/10 px-3 py-1 text-xs font-semibold text-secondary">
          {tag}
        </span>
        <ArrowUpRight className="h-4 w-4 text-text-muted transition-colors group-hover:text-primary" />
      </div>
      <h3 className="mb-1 text-base font-bold leading-snug text-text">{title}</h3>
      <p className="mb-3 text-xs font-medium uppercase tracking-wide text-text-muted">
        {venue} &middot; {year}
      </p>
      <p className="text-sm leading-relaxed text-text-muted">{summary}</p>
    </Link>
  );
}
