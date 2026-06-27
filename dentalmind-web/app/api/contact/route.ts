import { NextRequest, NextResponse } from "next/server";
import { Resend } from "resend";

interface ContactPayload {
  name: string;
  email: string;
  organization: string;
  country: string;
  role: string;
  message: string;
  source: string;
}

function isValidPayload(body: unknown): body is ContactPayload {
  if (!body || typeof body !== "object") return false;
  const b = body as Record<string, unknown>;
  return (
    typeof b.name === "string" &&
    b.name.trim().length > 0 &&
    typeof b.email === "string" &&
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(b.email) &&
    typeof b.message === "string" &&
    b.message.trim().length > 0
  );
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!isValidPayload(body)) {
    return NextResponse.json({ error: "Missing required fields." }, { status: 400 });
  }

  const apiKey = process.env.RESEND_API_KEY;
  const toEmail = process.env.CONTACT_TO_EMAIL ?? "hello@dentalmind.ai";

  if (!apiKey) {
    console.log("[contact] RESEND_API_KEY not set — logging submission instead of sending", body);
    return NextResponse.json({ ok: true, simulated: true });
  }

  try {
    const resend = new Resend(apiKey);
    await resend.emails.send({
      from: "DentalMind Website <onboarding@resend.dev>",
      to: toEmail,
      replyTo: body.email,
      subject: `New contact form submission from ${body.name}`,
      text: [
        `Name: ${body.name}`,
        `Email: ${body.email}`,
        `Organization: ${body.organization || "—"}`,
        `Country: ${body.country || "—"}`,
        `Role: ${body.role || "—"}`,
        `Heard about us via: ${body.source || "—"}`,
        "",
        "Message:",
        body.message,
      ].join("\n"),
    });
    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error("[contact] Resend send failed", error);
    return NextResponse.json({ error: "Failed to send message." }, { status: 502 });
  }
}
