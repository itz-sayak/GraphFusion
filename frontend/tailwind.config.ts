import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#0b1020", 900: "#111831", 800: "#1a2342", 700: "#26305a", 600: "#3a4577", 400: "#8b94c7", 300: "#b5bce3", 100: "#e8ebf8" },
        accent: { 500: "#5b8cff", 400: "#7aa2ff", 300: "#a6c1ff" },
        good: "#34d399",
        warn: "#fbbf24",
        bad: "#f87171",
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
    },
  },
  plugins: [],
};
export default config;
