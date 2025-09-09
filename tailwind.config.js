/** @type {import('tailwindcss').Config} */
module.exports = {
	prefix: 'tw-',
	important: false,
	content: [
		"**/*.{html, jsx, js}",
		"**/*.js",
		"**/*.html",
	],
	theme: {
		extend: {
			colors: {
				primary: "#2563eb",
				'data-blue': "#2563eb",
				'accent-blue': "#0ea5e9",
				'chart-purple': "#8b5cf6",
				'success-green': "#10b981",
				'slate-dark': "#1e293b",
				'bg-light': "#f8fafc"
			},
			animation: {
				'fade-in': 'fadeIn 0.6s ease-in-out',
				'slide-up': 'slideUp 0.8s ease-out',
				'pulse-slow': 'pulse 3s infinite'
			},
			keyframes: {
				fadeIn: {
					'0%': { opacity: '0', transform: 'translateY(20px)' },
					'100%': { opacity: '1', transform: 'translateY(0)' }
				},
				slideUp: {
					'0%': { opacity: '0', transform: 'translateY(50px)' },
					'100%': { opacity: '1', transform: 'translateY(0)' }
				}
			}
		},
	},
	plugins: [],
}

