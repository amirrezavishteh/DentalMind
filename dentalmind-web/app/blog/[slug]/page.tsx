import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { MDXRemote } from "next-mdx-remote/rsc";
import { Badge } from "@/components/ui/badge";
import { blogPosts, getPostBySlug } from "@/lib/blog";

export function generateStaticParams() {
  return blogPosts.map((post) => ({ slug: post.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const post = getPostBySlug(params.slug);
  if (!post) return {};
  return {
    title: post.meta.title,
    description: post.meta.description,
  };
}

const mdxComponents = {
  h2: (props: React.ComponentProps<"h2">) => (
    <h2 className="mb-4 mt-10 text-2xl font-bold text-text" {...props} />
  ),
  p: (props: React.ComponentProps<"p">) => (
    <p className="mb-5 leading-relaxed text-text-muted" {...props} />
  ),
  ul: (props: React.ComponentProps<"ul">) => (
    <ul className="mb-5 list-disc space-y-2 pl-6 text-text-muted" {...props} />
  ),
  ol: (props: React.ComponentProps<"ol">) => (
    <ol className="mb-5 list-decimal space-y-2 pl-6 text-text-muted" {...props} />
  ),
  li: (props: React.ComponentProps<"li">) => <li className="leading-relaxed" {...props} />,
  strong: (props: React.ComponentProps<"strong">) => (
    <strong className="font-semibold text-text" {...props} />
  ),
  em: (props: React.ComponentProps<"em">) => <em className="italic text-text" {...props} />,
  a: (props: React.ComponentProps<"a">) => (
    <a className="text-primary underline-offset-4 hover:underline" {...props} />
  ),
};

export default function BlogPostPage({ params }: { params: { slug: string } }) {
  const post = getPostBySlug(params.slug);
  if (!post) notFound();

  return (
    <main className="py-16">
      <article className="container max-w-2xl">
        <Link
          href="/blog"
          className="mb-8 inline-flex items-center gap-1 text-sm text-text-muted transition-colors hover:text-primary"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to blog
        </Link>

        <div className="mb-4 flex items-center gap-3">
          <Badge variant="cyan">{post.meta.tag}</Badge>
          <span className="text-xs text-text-muted">
            {new Date(post.meta.date).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </span>
          <span className="text-xs text-text-muted">&middot; {post.meta.readingTime}</span>
        </div>

        <h1 className="mb-8 text-balance text-3xl font-extrabold text-text sm:text-4xl">
          {post.meta.title}
        </h1>

        <div className="prose-dentalmind">
          <MDXRemote source={post.content} components={mdxComponents} />
        </div>
      </article>
    </main>
  );
}
