import Link from "next/link";
import { Sparkles } from "lucide-react";

const COLUMNS = [
  {
    title: "Product",
    links: [
      { href: "/demo", label: "Live Demo" },
      { href: "/how-it-works", label: "How It Works" },
      { href: "/pricing", label: "Pricing" },
    ],
  },
  {
    title: "Company",
    links: [
      { href: "/about", label: "About" },
      { href: "/research", label: "Research" },
      { href: "/blog", label: "Blog" },
      { href: "/contact", label: "Contact" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t border-border bg-surface/40">
      <div className="container py-12">
        <div className="grid grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-4">
          <div className="col-span-1 sm:col-span-2 lg:col-span-2">
            <Link href="/" className="flex items-center gap-2 font-extrabold text-text">
              <span className="flex h-8 w-8 items-center justify-center rounded-card bg-primary/10 text-primary">
                <Sparkles className="h-4 w-4" />
              </span>
              <span className="text-lg">
                Dental<span className="text-primary">Mind</span>
              </span>
            </Link>
            <p className="mt-4 max-w-sm text-sm text-text-muted">
              AI-powered radiograph analysis for dental clinics and hospitals — one pipeline
              across Bitewing, Panoramic, Periapical, Full-Mouth Series and CBCT.
            </p>
          </div>

          {COLUMNS.map((col) => (
            <div key={col.title}>
              <h4 className="mb-4 text-sm font-bold uppercase tracking-wide text-text">
                {col.title}
              </h4>
              <ul className="space-y-3">
                {col.links.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      className="text-sm text-text-muted transition-colors hover:text-primary"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-10 rounded-card border border-warning/30 bg-warning/10 px-4 py-3 text-xs text-warning">
          DentalMind is a clinical decision support tool intended for second-opinion use only.
          It does not replace the clinical judgment of a licensed dentist or radiologist.
        </div>

        <div className="mt-8 flex flex-col items-center justify-between gap-4 border-t border-border pt-6 text-xs text-text-muted sm:flex-row">
          <p>&copy; {new Date().getFullYear()} DentalMind. All rights reserved.</p>
          <p>Built on open dental imaging research.</p>
        </div>
      </div>
    </footer>
  );
}
