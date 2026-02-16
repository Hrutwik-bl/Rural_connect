/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#059669',
        'primary-dark': '#047857',
        'primary-darker': '#065f46',
        secondary: '#1e293b',
        accent: '#f59e0b',
        'accent-dark': '#d97706',
        surface: '#f8f7f4',
        'surface-warm': '#faf5ee',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #059669 0%, #047857 100%)',
        'gradient-dark': 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
        'gradient-accent': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'gradient-hero': 'linear-gradient(135deg, #065f46 0%, #059669 50%, #047857 100%)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'card-hover': '0 10px 25px rgba(0,0,0,0.08)',
        'elevated': '0 20px 40px rgba(0,0,0,0.1)',
      },
    },
  },
  plugins: [],
}
