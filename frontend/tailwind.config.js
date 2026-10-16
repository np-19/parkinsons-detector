/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        clinical: {
          50: "#F0F7FF",
          100: "#E0EFFF",
          500: "#2563EB",
          600: "#1D4ED8",
          700: "#1E40AF",
          900: "#0F172A",
        },
      },
    },
  },
  plugins: [],
};
