import { NextRequest, NextResponse } from "next/server";
import { demoResults, type Modality } from "@/lib/demo-data";

const VALID_MODALITIES: Modality[] = ["bitewing", "panoramic", "periapical"];

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const modality = body?.modality;

  if (!VALID_MODALITIES.includes(modality)) {
    return NextResponse.json(
      { error: "Unknown modality. Expected one of: bitewing, panoramic, periapical." },
      { status: 400 },
    );
  }

  await new Promise((resolve) => setTimeout(resolve, 400));

  return NextResponse.json(demoResults[modality as Modality]);
}
