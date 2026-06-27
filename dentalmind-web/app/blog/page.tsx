import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { Badge } from "@/components/ui/badge";
import { blogPosts } from "@/lib/blog";

export const metadata: Metadata = {
  title: "Blog",
  description: "Research notes and engineering deep-dives from the DentalMind team.",
};

export default function BlogPage() {
  return (
    <main className="py-16">
      <div className="container">
        <SectionHeading
          eyebrow="Blog"
          title="Research notes"
          description="How the DentalMind pipeline is built, why it's built that way, and how it compares to the rest of dental AI."
        />

        <div className="mx-auto flex max-w-3xl flex-col gap-6">
          {blogPosts.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="group rounded-card-lg border border-border bg-surface p-6 transition-colors hover:border-primary/40"
            >
              <div className="mb-3 flex items-center gap-3">
                <Badge variant="cyan">{post.tag}</Badge>
                <span className="text-xs text-text-muted">
                  {new Date(post.date).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </span>
                <span className="text-xs text-text-muted">&middot; {post.readingTime}</span>
              </div>
              <h2 className="mb-2 text-xl font-bold text-text transition-colors group-hover:text-primary">
                {post.title}
              </h2>
              <p className="mb-3 text-sm leading-relaxed text-text-muted">{post.description}</p>
              <span className="inline-flex items-center gap-1 text-sm font-medium text-primary">
                Read post <ArrowRight className="h-3.5 w-3.5" />
              </span>
            </Link>
          ))}
        </div>
      </div>
    </main>
  );
}
