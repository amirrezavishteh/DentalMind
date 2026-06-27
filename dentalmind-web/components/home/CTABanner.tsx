import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export function CTABanner() {
  return (
    <section className="py-20">
      <div className="container">
        <div className="relative overflow-hidden rounded-card-lg border border-primary/30 bg-surface px-8 py-16 text-center">
          <div className="absolute inset-0 bg-grid-glow" />
          <div className="relative">
            <h2 className="text-balance text-3xl font-extrabold text-text sm:text-4xl">
              Ready to stop reading bounding boxes?
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-balance text-text-muted">
              Book a demo and see how DentalMind clusters findings per tooth and ranks
              treatment options across all five modalities.
            </p>
            <div className="mt-8 flex justify-center gap-4">
              <Link href="/contact">
                <Button size="lg" variant="primary">
                  Book a demo
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/demo">
                <Button size="lg" variant="ghost">
                  Try it yourself
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
