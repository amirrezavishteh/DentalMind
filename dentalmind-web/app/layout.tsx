import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

const SITE_URL = "https://dentalmind.ai";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "DentalMind — AI-Powered Dental Radiograph Analysis",
    template: "%s — DentalMind",
  },
  description:
    "DentalMind analyzes Bitewing, Panoramic, Periapical, FMS and CBCT in one pipeline. Per-tooth clustering and ranked treatment prompts for dentists — built on a trustworthy-AI approach to healthcare.",
  keywords: [
    "dental AI",
    "caries detection",
    "CBCT segmentation",
    "panoramic AI",
    "dental radiology AI",
    "trustworthy AI",
    "healthcare AI",
  ],
  authors: [{ name: "DentalMind" }],
  openGraph: {
    title: "DentalMind — AI-Powered Dental Radiograph Analysis",
    description:
      "DentalMind analyzes all 5 X-ray modalities in one pipeline, clusters findings per tooth, and gives your team ranked treatment prompts.",
    url: SITE_URL,
    siteName: "DentalMind",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "DentalMind — AI-Powered Dental Radiograph Analysis",
    description:
      "DentalMind analyzes all 5 X-ray modalities in one pipeline, clusters findings per tooth, and gives your team ranked treatment prompts.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "DentalMind",
  applicationCategory: "MedicalApplication",
  operatingSystem: "Web",
  description:
    "DentalMind analyzes Bitewing, Panoramic, Periapical, FMS and CBCT dental radiographs in one pipeline, clustering findings per tooth and producing ranked treatment prompts as a second opinion for dentists.",
  offers: {
    "@type": "Offer",
    price: "199",
    priceCurrency: "USD",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} bg-background font-sans text-text antialiased`}
      >
        <Navbar />
        {children}
        <Footer />
      </body>
    </html>
  );
}
