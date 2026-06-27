import fs from "fs";
import path from "path";
import matter from "gray-matter";

const BLOG_DIR = path.join(process.cwd(), "content/blog");

export interface BlogPostMeta {
  slug: string;
  title: string;
  description: string;
  date: string;
  readingTime: string;
  tag: string;
}

function readSlugs(): string[] {
  if (!fs.existsSync(BLOG_DIR)) return [];
  return fs
    .readdirSync(BLOG_DIR)
    .filter((file) => file.endsWith(".mdx"))
    .map((file) => file.replace(/\.mdx$/, ""));
}

export const blogPosts: BlogPostMeta[] = readSlugs()
  .map((slug) => {
    const raw = fs.readFileSync(path.join(BLOG_DIR, `${slug}.mdx`), "utf8");
    const { data } = matter(raw);
    return {
      slug,
      title: data.title as string,
      description: data.description as string,
      date: data.date as string,
      readingTime: data.readingTime as string,
      tag: data.tag as string,
    };
  })
  .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

export function getPostBySlug(slug: string) {
  const filePath = path.join(BLOG_DIR, `${slug}.mdx`);
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf8");
  const { content, data } = matter(raw);
  return {
    slug,
    content,
    meta: {
      slug,
      title: data.title as string,
      description: data.description as string,
      date: data.date as string,
      readingTime: data.readingTime as string,
      tag: data.tag as string,
    } satisfies BlogPostMeta,
  };
}
