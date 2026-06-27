import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./content/**/*.{md,mdx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "1.5rem",
      screens: {
        "2xl": "1280px",
      },
    },
    extend: {
      colors: {
        background: "#0A0F1E",
        surface: "#161b2e",
        border: "#2a3350",
        primary: {
          DEFAULT: "#00D4FF",
          foreground: "#04141a",
        },
        secondary: {
          DEFAULT: "#7C3AED",
          foreground: "#f5f3ff",
        },
        success: "#10B981",
        warning: "#F59E0B",
        danger: "#EF4444",
        text: {
          DEFAULT: "#E6EDF3",
          muted: "#8B949E",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      borderRadius: {
        card: "8px",
        "card-lg": "12px",
        pill: "999px",
      },
      boxShadow: {
        "glow-cyan": "0 0 24px rgba(0, 212, 255, 0.35)",
        "glow-purple": "0 0 24px rgba(124, 58, 237, 0.35)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "pulse-glow": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.6", transform: "scale(1.3)" },
        },
        "count-up": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "pulse-glow": "pulse-glow 1.8s ease-in-out infinite",
      },
      backgroundImage: {
        "grid-glow":
          "radial-gradient(circle at 50% 0%, rgba(0,212,255,0.12), transparent 60%)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
