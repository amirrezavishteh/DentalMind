"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/demo", label: "Live Demo" },
  { href: "/how-it-works", label: "How It Works" },
  { href: "/research", label: "Research" },
  { href: "/pricing", label: "Pricing" },
  { href: "/blog", label: "Blog" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <header
      className={cn(
        "sticky top-0 z-50 w-full transition-all duration-300",
        scrolled
          ? "border-b border-border bg-background/80 shadow-lg shadow-black/20 backdrop-blur-lg"
          : "border-b border-transparent bg-transparent",
      )}
    >
      <nav className="container flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-extrabold text-text">
          <span className="flex h-8 w-8 items-center justify-center rounded-card bg-primary/10 text-primary">
            <Sparkles className="h-4 w-4" />
          </span>
          <span className="text-lg">
            Dental<span className="text-primary">Mind</span>
          </span>
        </Link>

        <div className="hidden items-center gap-8 lg:flex">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "text-sm font-medium text-text-muted transition-colors hover:text-primary",
                pathname === link.href && "text-primary",
              )}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="hidden items-center gap-3 lg:flex">
          <Link href="/contact">
            <Button variant="ghost" size="sm">
              Contact
            </Button>
          </Link>
          <Link href="/demo">
            <Button variant="primary" size="sm">
              Try live demo
            </Button>
          </Link>
        </div>

        <button
          aria-label="Toggle menu"
          className="text-text lg:hidden"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-border bg-background px-6 pb-6 pt-2 lg:hidden">
          <div className="flex flex-col gap-4">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "text-sm font-medium text-text-muted transition-colors hover:text-primary",
                  pathname === link.href && "text-primary",
                )}
              >
                {link.label}
              </Link>
            ))}
            <Link href="/contact">
              <Button variant="ghost" className="w-full">
                Contact
              </Button>
            </Link>
            <Link href="/demo">
              <Button variant="primary" className="w-full">
                Try live demo
              </Button>
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
