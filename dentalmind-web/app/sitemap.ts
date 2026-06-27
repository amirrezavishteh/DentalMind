import type { MetadataRoute } from "next";
import { blogPosts } from "@/lib/blog";

const SITE_URL = "https://dentalmind.ai";

export default function sitemap(): MetadataRoute.Sitemap {
  const staticRoutes = [
    "",
    "/demo",
    "/how-it-works",
    "/research",
    "/pricing",
    "/contact",
    "/about",
    "/blog",
  ].map((route) => ({
    url: `${SITE_URL}${route}`,
    lastModified: new Date(),
  }));

  const blogRoutes = blogPosts.map((post) => ({
    url: `${SITE_URL}/blog/${post.slug}`,
    lastModified: new Date(post.date),
  }));

  return [...staticRoutes, ...blogRoutes];
}
