import { ImageResponse } from "next/og";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#0A0F1E",
          backgroundImage:
            "radial-gradient(circle at 50% 0%, rgba(0,212,255,0.25), transparent 60%)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            fontSize: 72,
            fontWeight: 800,
            color: "#E6EDF3",
          }}
        >
          <span>Dental</span>
          <span style={{ color: "#00D4FF" }}>Mind</span>
        </div>
        <div style={{ marginTop: 24, fontSize: 28, color: "#8B949E" }}>
          AI That Thinks Like a Dentist
        </div>
      </div>
    ),
    { ...size },
  );
}
