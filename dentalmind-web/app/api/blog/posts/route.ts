import { NextResponse } from "next/server";
import { blogPosts } from "@/lib/blog";

export async function GET() {
  return NextResponse.json(blogPosts);
}
