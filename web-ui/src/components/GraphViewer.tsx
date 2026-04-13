export function GraphViewer() {
    return (
        <div className="w-full h-full flex items-center justify-center relative p-8">
            {/* Node 1 */}
            <div className="absolute top-[20%] left-[20%] w-32 h-12 bg-white/10 border border-pink-500/30 rounded-lg flex items-center justify-center text-sm font-medium text-pink-200 shadow-[0_0_15px_rgba(236,72,153,0.1)] hover:bg-white/20 transition-all cursor-pointer hover:scale-105">
                AuthContext.tsx
            </div>

            {/* Target Node */}
            <div className="absolute bottom-[20%] right-[20%] w-32 h-12 bg-white/10 border border-emerald-500/30 rounded-lg flex items-center justify-center text-sm font-medium text-emerald-200 shadow-[0_0_15px_rgba(16,185,129,0.1)] hover:bg-white/20 transition-all cursor-pointer hover:scale-105">
                UseAuthStore.tsx
            </div>

            {/* SVG Connecting Line */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
                <defs>
                    <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#ec4899" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="#10b981" stopOpacity="0.5" />
                    </linearGradient>
                </defs>
                <path
                    d="M 30% 30% C 50% 30%, 50% 60%, 70% 70%"
                    fill="none"
                    stroke="url(#lineGrad)"
                    strokeWidth="3"
                    strokeDasharray="6 4"
                    className="animate-pulse"
                />
            </svg>
        </div>
    );
}
