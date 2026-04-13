import Link from 'next/link';
import { GraphViewer } from '@/components/GraphViewer'; // Use the Next.js @ alias

export default function RulesReviewPage() {
    const pendingRules = [
        {
            id: "uuid-1123",
            title: "Dependency Injection over Mixins",
            confidence: 0.88,
            source: "org/repo #42",
            path: "/src/services/**",
        },
        {
            id: "uuid-1124",
            title: "Migrate AuthContext to Zustand",
            confidence: 0.95,
            source: "org/auth-service #19",
            path: "/src/components/**",
        }
    ];

    return (
        <div className="min-h-screen bg-gray-950 text-white font-sans selection:bg-indigo-500 selection:text-white">
            <nav className="border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center">
                            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                        </div>
                        <span className="font-bold text-xl tracking-tight text-white/90">RuleExtractor</span>
                    </div>
                    <div className="flex items-center gap-6 text-sm font-medium">
                        <Link href="/" className="text-white/60 hover:text-white transition-colors">Dashboard</Link>
                        <Link href="/rules" className="text-white">Pending Rules</Link>
                        <button className="px-4 py-1.5 rounded-full bg-white/10 hover:bg-white/20 transition-all cursor-pointer">Settings</button>
                    </div>
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-6 py-12">
                <header className="mb-12 flex justify-between items-end">
                    <div>
                        <h1 className="text-3xl font-extrabold tracking-tight mb-2">Needs Review</h1>
                        <p className="text-gray-400">Human-in-the-loop approval queue for extracted architectural patterns.</p>
                    </div>
                    <button className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 px-6 py-2.5 rounded-lg font-medium shadow-[0_0_20px_rgba(99,102,241,0.3)] transition-all cursor-pointer">
                        Bulk Approve
                    </button>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                    {/* Rules List */}
                    <div className="space-y-4">
                        {pendingRules.map((rule) => (
                            <div key={rule.id} className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:border-white/20 cursor-pointer transition-all hover:bg-white/[0.07] group">
                                <div className="flex justify-between items-start mb-4">
                                    <h3 className="font-bold text-lg text-indigo-100 group-hover:text-indigo-300 transition-colors">{rule.title}</h3>
                                    <span className="text-xs bg-indigo-500/20 text-indigo-300 px-2.5 py-1 rounded-full border border-indigo-500/30">
                                        Conf: {(rule.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div className="text-sm text-gray-400 mb-6 space-y-1">
                                    <p><span className="text-gray-500 w-16 inline-block">Source:</span> {rule.source}</p>
                                    <p><span className="text-gray-500 w-16 inline-block">Scope:</span> <code className="bg-black/50 px-1 py-0.5 rounded text-indigo-200">{rule.path}</code></p>
                                </div>
                                <div className="flex gap-2">
                                    <button className="flex-1 border border-white/10 bg-white/5 hover:bg-emerald-500/20 hover:text-emerald-400 hover:border-emerald-500/30 py-2 rounded-lg text-sm font-medium transition-all">Approve</button>
                                    <button className="flex-1 border border-white/10 bg-white/5 hover:bg-pink-500/20 hover:text-pink-400 hover:border-pink-500/30 py-2 rounded-lg text-sm font-medium transition-all">Reject</button>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Graph Visualization */}
                    <div className="rounded-2xl border border-white/10 bg-black/40 overflow-hidden relative min-h-[400px]">
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(99,102,241,0.15),transparent)] pointer-events-none"></div>
                        <div className="absolute top-4 left-4 z-10 text-[10px] font-bold uppercase tracking-[0.2em] text-gray-500">
                            Substitution Graph (LightRAG)
                        </div>
                        <GraphViewer />
                    </div>
                </div>
            </main>
        </div>
    );
}
