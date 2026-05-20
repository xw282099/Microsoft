import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Institutional dark terminal palette
        bg: {
          base: "#05070d",
          panel: "#0a0e18",
          card: "#0f1422",
          elev: "#141b2d",
          line: "#1d2538",
        },
        ink: {
          DEFAULT: "#e6ebf5",
          muted: "#8a93a6",
          dim: "#5a6378",
          faint: "#3a4256",
        },
        accent: {
          cyan: "#00e5ff",
          blue: "#3b82f6",
          neon: "#39ff88",
          ai: "#7c5cff",
          gold: "#f5b400",
        },
        risk: {
          bull: "#22d39a",
          bear: "#ff4d6d",
          neutral: "#f5b400",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SF Mono", "Menlo", "Consolas", "monospace"],
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(0, 229, 255, 0.18)",
        ai: "0 0 24px rgba(124, 92, 255, 0.25)",
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(29,37,56,.5) 1px, transparent 1px), linear-gradient(90deg, rgba(29,37,56,.5) 1px, transparent 1px)",
        radialAi:
          "radial-gradient(circle at top, rgba(124,92,255,.18), transparent 60%)",
      },
      animation: {
        pulseSoft: "pulseSoft 2.2s ease-in-out infinite",
        scan: "scan 2.6s linear infinite",
        tickerSlide: "tickerSlide 40s linear infinite",
        flicker: "flicker 4s linear infinite",
      },
      keyframes: {
        pulseSoft: {
          "0%,100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(120%)" },
        },
        tickerSlide: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        flicker: {
          "0%,18%,22%,25%,53%,57%,100%": { opacity: "1" },
          "20%,24%,55%": { opacity: "0.6" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
