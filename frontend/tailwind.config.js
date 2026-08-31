/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surfaces: an almost-white canvas with plain white panels.
        canvas: "#fafafa",
        surface: "#ffffff",
        // Subtle gray borders. `line` is the default hairline everywhere.
        line: {
          subtle: "#f1f1f3",
          DEFAULT: "#e7e7ea",
          strong: "#d6d6dc",
        },
        // Near-black text with three quieter steps.
        ink: {
          DEFAULT: "#0c0c0f",
          secondary: "#42424d",
          muted: "#6f6f7a",
          faint: "#9b9ba5",
        },
        // Blue is the only decorative accent.
        accent: {
          50: "#eff5ff",
          100: "#dce8fd",
          200: "#c0d7fc",
          300: "#93bbf9",
          400: "#5f97f5",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1b3fa8",
        },
        // Status only. Never used decoratively.
        positive: {
          50: "#ecfdf3",
          100: "#d3f8e0",
          500: "#16a34a",
          600: "#12813c",
          700: "#0f6b33",
        },
        caution: {
          50: "#fffaeb",
          100: "#fdf0c8",
          500: "#e0900a",
          600: "#b97407",
          700: "#93590a",
        },
        critical: {
          50: "#fef3f2",
          100: "#fde5e2",
          500: "#e5484d",
          600: "#cc2f36",
          700: "#a91d24",
        },
      },
      fontFamily: {
        sans: [
          "Geist",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "Geist Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        // Almost no shadows: one hairline lift, one for true overlays.
        xs: "0 1px 2px rgba(12, 12, 15, 0.04)",
        overlay: "0 16px 48px rgba(12, 12, 15, 0.14)",
      },
      maxWidth: {
        page: "78rem",
        prose: "44rem",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "slide-in-right": {
          from: { transform: "translateX(100%)" },
          to: { transform: "translateX(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 120ms ease-out",
        "slide-in-right": "slide-in-right 180ms cubic-bezier(0.32, 0.72, 0, 1)",
      },
    },
  },
  plugins: [],
};
