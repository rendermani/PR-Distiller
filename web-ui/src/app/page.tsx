'use client';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Home() {
  // Pipeline Rules State
  const [rules, setRules] = useState<any[]>([]);

  // Orchestrator State
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("Idle");
  const [jobProgress, setJobProgress] = useState<number>(0);

  // Multi-Repo States
  const [selectedRepo, setSelectedRepo] = useState<string>("");
  const [months, setMonths] = useState(2);
  const [threshold, setThreshold] = useState(0.45);

  // UI Modal States
  const [activeTab, setActiveTab] = useState<"database" | "telemetry">("database");
  const [showSettings, setShowSettings] = useState(false);
  const [showAddRepo, setShowAddRepo] = useState(false);
  const [newRepoString, setNewRepoString] = useState("");

  // Add Rule Modal
  const [showAddRule, setShowAddRule] = useState(false);
  const [newRuleTitle, setNewRuleTitle] = useState("");
  const [newRuleDesc, setNewRuleDesc] = useState("");
  const [newRuleEnforce, setNewRuleEnforce] = useState("");

  // Export Modal States
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportType, setExportType] = useState("prompt");
  const [targetArch, setTargetArch] = useState("openai");
  const [exportResult, setExportResult] = useState("");
  const [isExporting, setIsExporting] = useState(false);

  // Dev Cache
  const [useCache, setUseCache] = useState(false);
  const [cacheInfo, setCacheInfo] = useState<any>(null);

  // Global Config
  const [config, setConfig] = useState<any>({
    github_token: "", llm_provider: "ollama", llm_api_base: "http://localhost:11434/v1", llm_model: "ollama/qwen3:8b", llm_api_key: "", provider_api_keys: {} as Record<string, string>, repos: {}, provider_models: {}
  });

  // Treat "local" (old) and "ollama" as the same provider so URL-vs-API-key
  // conditional rendering works for both existing saved configs and new ones.
  const isLocalProvider = (p: string) => p === "ollama" || p === "local";

  // All backend calls go through a same-origin Next.js proxy that injects
  // the API auth token server-side. The token never reaches the client bundle.
  // See web-ui/src/app/api/proxy/[...path]/route.ts.
  const apiFetch = (path: string, init: RequestInit = {}) =>
    fetch(`/api/proxy${path.startsWith('/') ? path : `/${path}`}`, init);

  const [isCustomModel, setIsCustomModel] = useState(false);
  const [customModelString, setCustomModelString] = useState("");

  // Error-handling state
  const [tokenStatus, setTokenStatus] = useState<{state: "idle" | "checking" | "valid" | "invalid"; message?: string; login?: string; scopes?: string[]}>({state: "idle"});
  const [llmHealth, setLlmHealth] = useState<{reachable: boolean; api_base?: string; error?: string} | null>(null);
  const [bannerDismissed, setBannerDismissed] = useState(false);

  const detectOS = (): "mac" | "windows" | "linux" | "other" => {
    if (typeof navigator === "undefined") return "other";
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("mac")) return "mac";
    if (ua.includes("win")) return "windows";
    if (ua.includes("linux")) return "linux";
    return "other";
  };

  const ollamaInstructions = (os: string): {command: string; hint: string} => {
    switch (os) {
      case "mac":
        return {command: "brew install ollama && ollama serve", hint: "or launch the Ollama.app from Applications"};
      case "windows":
        return {command: "winget install Ollama.Ollama", hint: "then run 'ollama serve' or launch the Ollama app from the Start menu"};
      case "linux":
        return {command: "curl -fsSL https://ollama.com/install.sh | sh && ollama serve", hint: "or: systemctl --user start ollama"};
      default:
        return {command: "See https://ollama.com/download", hint: "Install Ollama for your OS, then start the server"};
    }
  };

  // Load Loop
  useEffect(() => {
    apiFetch(`/api/config`).then(r => r.json()).then(data => {
      const hasRepos = Object.keys(data.repos || {}).length > 0;
      if (hasRepos && !selectedRepo) {
        const firstRepo = Object.keys(data.repos)[0];
        setSelectedRepo(firstRepo);
        setMonths(data.repos[firstRepo].months || 2);
        setThreshold(data.repos[firstRepo].threshold || 0.45);
      }

      // Normalize legacy "local" to "ollama" so the dropdown has a matching option.
      const rawProvider = data.llm_provider || "ollama";
      const providerStr = rawProvider === "local" ? "ollama" : rawProvider;
      const providerMap = data.provider_models?.[providerStr] || [];
      const isStandard = providerMap.some((m: any) => m.id === data.llm_model);

      setIsCustomModel(!isStandard);
      if (!isStandard) setCustomModelString(data.llm_model || "");

      setConfig({ ...data, repos: data.repos || {}, llm_provider: providerStr, provider_models: data.provider_models || {} });
    }).catch(console.error);
  }, []);

  const fetchRules = () => {
    const path = selectedRepo
      ? `/api/rules?repo=${encodeURIComponent(selectedRepo)}`
      : `/api/rules`;
    apiFetch(path)
      .then(r => r.json())
      .then(d => setRules(d.rules || []))
      .catch(console.error);
  };

  // Dynamically bounce HTTP payload when specific repo toggles natively
  useEffect(() => {
    fetchRules();
    if (selectedRepo) {
      apiFetch(`/api/cache/${encodeURIComponent(selectedRepo)}`)
        .then(r => r.json())
        .then(d => setCacheInfo(d.cached ? d : null))
        .catch(() => setCacheInfo(null));
    } else {
      setCacheInfo(null);
    }
  }, [selectedRepo]);

  // LLM health — poll every 20s. Used to show a top-of-page banner when unreachable.
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const r = await apiFetch(`/api/health/llm`);
        const d = await r.json();
        if (!cancelled) setLlmHealth(d);
      } catch {
        if (!cancelled) setLlmHealth({reachable: false, error: "Backend unreachable"});
      }
    };
    check();
    const t = setInterval(check, 20000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  // GitHub token validator — debounced on value change.
  useEffect(() => {
    const token = config.github_token;
    if (!token || token === "***") { setTokenStatus({state: "idle"}); return; }
    setTokenStatus({state: "checking"});
    const handle = setTimeout(async () => {
      try {
        const r = await apiFetch(`/api/github/validate`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({token}),
        });
        const d = await r.json();
        if (d.valid) {
          setTokenStatus({state: "valid", login: d.login, scopes: d.scopes});
        } else {
          setTokenStatus({state: "invalid", message: d.error});
        }
      } catch (e: any) {
        setTokenStatus({state: "invalid", message: "Could not reach backend."});
      }
    }, 600);
    return () => clearTimeout(handle);
  }, [config.github_token]);

  // Telemetry Polling
  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const res = await apiFetch(`/api/jobs/status/${jobId}`);
        const data = await res.json();

        if (data.status === "not_found") {
          setJobId(null);
          setJobStatus("Idle");
          setJobProgress(0);
          return window.alert("Backend connection memory dropped. Please re-run the distillation.");
        }

        setJobStatus(data.status);
        setJobProgress(data.progress || 0);

        if (data.progress === 100 || data.progress === -1) {
          setTimeout(() => {
            setJobId(null);
            fetchRules();
          }, 3500); // Wait 3.5s so the user can read the "Completed" message natively
        }
      } catch (err) { }
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId]);

  // ---------------------------------
  // HANDLERS
  // ---------------------------------
  const handleSaveConfig = async (e: any) => {
    e.preventDefault();
    const activeModel = isCustomModel ? customModelString : config.llm_model;
    const { llm_api_key: _dropped, ...rest } = config;
    const payload = { ...rest, llm_model: activeModel };
    await apiFetch(`/api/config`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    setConfig(payload);
    setShowSettings(false);
  };

  const handleRepoChange = (e: any) => {
    const repoName = e.target.value;
    setSelectedRepo(repoName);
    if (config.repos && config.repos[repoName]) {
      setMonths(config.repos[repoName].months);
      setThreshold(config.repos[repoName].threshold);
    }
  };

  const handleAddRepoSubmit = async (e: any) => {
    e.preventDefault();
    const repoPath = newRepoString.trim();
    if (repoPath) {
      const updatedRepos = { ...config.repos, [repoPath]: { months: 2, threshold: 0.45 } };
      const updatedConfig = { ...config, repos: updatedRepos };
      setConfig(updatedConfig);
      setSelectedRepo(repoPath);
      setMonths(2);
      setThreshold(0.45);
      setNewRepoString("");
      setShowAddRepo(false);
      await apiFetch(`/api/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(updatedConfig) });
    }
  };

  const handleDeleteRepo = async (repoToDelete: string) => {
    if (window.confirm(`Are you sure you want to completely remove ${repoToDelete}?`)) {
      const updatedRepos = { ...config.repos };
      delete updatedRepos[repoToDelete];
      const updatedConfig = { ...config, repos: updatedRepos };
      setConfig(updatedConfig);

      const keys = Object.keys(updatedRepos);
      if (keys.length > 0) {
        setSelectedRepo(keys[0]);
        setMonths(updatedRepos[keys[0]].months);
        setThreshold(updatedRepos[keys[0]].threshold);
      } else {
        setSelectedRepo("");
      }
      await apiFetch(`/api/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(updatedConfig) });
    }
  };

  const updateCurrentRepoSettings = async () => {
    if (!selectedRepo) return;
    const updatedConfig = {
      ...config,
      repos: { ...config.repos, [selectedRepo]: { months, threshold } }
    };
    setConfig(updatedConfig);
    await apiFetch(`/api/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(updatedConfig) });
  };

  const handleStartJob = async (e: any) => {
    e.preventDefault();
    if (!selectedRepo) return alert("Select or Add a Repository first!");

    try {
      await updateCurrentRepoSettings();
      setActiveTab("telemetry");
      setJobProgress(0);

      const res = await apiFetch(`/api/jobs/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo: selectedRepo, months, threshold, use_cache: useCache })
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Server dropped connection");
      }

      const data = await res.json();
      setJobId(data.job_id);
    } catch (err: any) {
      alert(`Pipeline failed to start: ${err.message}`);
    }
  };

  const handleCancelJob = async () => {
    if (!jobId) return;
    try {
      await apiFetch(`/api/jobs/cancel/${jobId}`, { method: 'POST' });
    } catch (err) { console.error("Failed to cancel loop:", err); }
    setJobId(null);
    setJobStatus("Idle");
    setJobProgress(0);
  };

  const handleExportContext = async () => {
    setIsExporting(true);
    try {
      const res = await apiFetch(`/api/export?repo=${encodeURIComponent(selectedRepo)}&type=${exportType}&arch=${targetArch}`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setExportResult(data.content || "No rules ready to export.");
    } catch (err) {
      setExportResult("Failed to perform pipeline synthesis.");
    }
    setIsExporting(false);
  };


  const getApiKeyLabel = () => {
    if (config.llm_provider === "google") return "Google API Key";
    if (config.llm_provider === "anthropic") return "Anthropic API Key";
    if (config.llm_provider === "openrouter") return "OpenRouter API Key";
    if (config.llm_provider === "openai") return "OpenAI API Key";
    return "Custom Provider API Key";
  };

  const llmBannerVisible = llmHealth && !llmHealth.reachable && !bannerDismissed;
  const os = detectOS();
  const instructions = ollamaInstructions(os);

  return (
    <main className="min-h-screen bg-[#0A0A0A] text-white p-8 selection:bg-purple-500/30 font-sans flex flex-col md:flex-row gap-8 relative overflow-hidden">

      {/* LLM HEALTH BANNER */}
      {llmBannerVisible && (
        <div className="fixed top-0 left-0 right-0 z-40 bg-amber-500/10 border-b border-amber-500/30 backdrop-blur-md px-6 py-3 flex items-start gap-4">
          <div className="flex-1">
            <div className="text-sm font-semibold text-amber-300 mb-1">
              LLM server unreachable at {llmHealth!.api_base || "configured endpoint"}
            </div>
            <div className="text-xs text-amber-200/80 mb-2">
              {llmHealth!.error || "Inference will fail until the server is running."} Start it for your OS ({os}):
            </div>
            <code className="block bg-black/50 px-3 py-1.5 rounded text-xs font-mono text-amber-100 select-all">{instructions.command}</code>
            <div className="text-xs text-amber-200/60 mt-1">{instructions.hint}</div>
          </div>
          <button onClick={() => setBannerDismissed(true)} className="text-amber-300/60 hover:text-amber-300 text-sm" aria-label="Dismiss">✕</button>
        </div>
      )}

      {/* LEFT SIDEBAR: CONTROL CENTER */}
      <div className="w-full md:w-96 flex flex-col gap-6 flex-shrink-0 relative z-10">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-light tracking-tight">PR-Distiller Control.</h1>
          <button onClick={() => setShowSettings(true)} className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm transition font-medium tracking-wide border border-white/10">
            ⚙️ Global Settings
          </button>
        </div>

        <form onSubmit={handleStartJob} className="bg-white/5 border border-white/10 p-6 rounded-2xl flex flex-col gap-5">
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs text-neutral-400 uppercase tracking-widest block">Target Repository</label>
              <button type="button" onClick={() => setShowAddRepo(true)} className="text-xs text-purple-400 hover:text-purple-300 font-semibold">+ Add Repo</button>
            </div>

            {Object.keys(config.repos || {}).length === 0 ? (
              <div className="text-sm p-4 border border-dashed border-white/20 rounded-lg text-center text-neutral-500 cursor-pointer hover:bg-white/5 transition" onClick={() => setShowAddRepo(true)}>
                No Repositories Mapped.<br />Click to add one.
              </div>
            ) : (
              <div className="flex gap-2">
                <select
                  className="flex-1 bg-black/50 border border-white/10 rounded-lg p-3 text-sm text-white focus:border-purple-500 outline-none transition cursor-pointer"
                  value={selectedRepo} onChange={handleRepoChange}
                >
                  {Object.keys(config.repos).map(r => <option key={r} value={r}>{r}</option>)}
                </select>
                <button type="button" onClick={() => handleDeleteRepo(selectedRepo)} className="px-4 bg-red-500/10 hover:bg-red-500/20 text-red-500 rounded-lg border border-red-500/20 transition" title="Delete Repository">
                  🗑️
                </button>
              </div>
            )}
          </div>

          <div className="flex gap-4">
            <div className="flex-1">
              <label className="text-xs text-neutral-400 uppercase tracking-widest mb-1 block">Months Back</label>
              <input type="number" value={months} onChange={e => { setMonths(parseFloat(e.target.value)); updateCurrentRepoSettings(); }} disabled={!selectedRepo} className="w-full bg-black/50 border border-white/10 rounded-lg p-3 text-sm text-white outline-none disabled:opacity-50" />
            </div>
            <div className="flex-1">
              <label className="text-xs text-neutral-400 uppercase tracking-widest mb-1 block cursor-help" title="Controls structural strictness mapping. Lower binds tighter deduplication matrices.">Similarity</label>
              <input type="number" step="0.01" value={threshold} onChange={e => { setThreshold(parseFloat(e.target.value)); updateCurrentRepoSettings(); }} disabled={!selectedRepo} className="w-full bg-black/50 border border-white/10 rounded-lg p-3 text-sm text-white outline-none disabled:opacity-50" />
            </div>
          </div>

          {cacheInfo && (
            <label className="flex items-center gap-3 mt-2 cursor-pointer group">
              <input type="checkbox" checked={useCache} onChange={e => setUseCache(e.target.checked)} className="accent-amber-500 w-4 h-4" />
              <span className="text-xs text-amber-400/80 group-hover:text-amber-300 transition">
                Use dev cache ({cacheInfo.count} tuples, {cacheInfo.updated_at?.slice(0, 10)})
              </span>
            </label>
          )}

          <button disabled={!!jobId || !selectedRepo} type="submit" className="w-full bg-white text-black py-3 rounded-lg font-semibold shadow-[0_0_15px_rgba(255,255,255,0.1)] hover:scale-[1.02] transition disabled:opacity-50 disabled:hover:scale-100 disabled:cursor-not-allowed mt-2 text-sm tracking-wide">
            {jobId ? "Orchestrating Architecture..." : useCache ? "Replay from Cache" : "Start Distillation"}
          </button>
        </form>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white/5 border border-white/10 p-4 rounded-2xl flex flex-col items-start">
            <p className="text-xs text-neutral-400 uppercase tracking-widest">Active Rules</p>
            <p className="text-3xl font-light mt-1 text-white">{rules.length}</p>
          </div>
          <div className="bg-white/5 border border-white/10 p-4 rounded-2xl flex flex-col items-start">
            <p className="text-xs text-neutral-400 uppercase tracking-widest">Engine Loop</p>
            <p className="text-3xl font-light mt-1 text-green-400">{jobId ? "Active" : "Stable"}</p>
          </div>
        </div>
      </div>

      {/* RIGHT MAIN PANEL: RESULTS */}
      <div className="flex-1 min-w-0 bg-black/30 border border-white/10 rounded-2xl p-6 md:p-10 flex flex-col items-start justify-start overflow-y-auto relative z-10">
        {/* TAB SWITCHER */}
        <div className="flex gap-2 mb-8 border-b border-white/10 w-full pb-px relative z-0">
          <button onClick={() => setActiveTab("database")} className={`pb-3 px-2 text-sm font-medium tracking-wide transition border-b-2 ${activeTab === "database" ? "text-white border-purple-500" : "text-neutral-500 border-transparent hover:text-neutral-300"}`}>
            Rules Database
          </button>
          <button onClick={() => setActiveTab("telemetry")} className={`pb-3 px-2 text-sm font-medium tracking-wide transition border-b-2 flex gap-2 items-center ${activeTab === "telemetry" ? "text-white border-purple-500" : "text-neutral-500 border-transparent hover:text-neutral-300"}`}>
            Pipeline State {jobId && <span className="h-2 w-2 rounded-full bg-purple-500 animate-pulse"></span>}
          </button>
        </div>

        {/* LOGIC VIEWS */}
        {activeTab === "database" && (
          <div className="w-full flex flex-col flex-1 relative z-0">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-light text-white tracking-widest uppercase">Verified Rule Schema</h2>
              <div className="flex gap-2">
                <button onClick={() => setShowAddRule(true)} disabled={!selectedRepo} className="px-4 py-2 bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 rounded-lg text-xs tracking-wider uppercase font-semibold transition disabled:opacity-50">
                  + Add Rule
                </button>
                <button onClick={() => setShowExportModal(true)} disabled={rules.length === 0} className="px-4 py-2 bg-purple-500/10 text-purple-400 border border-purple-500/20 hover:bg-purple-500/20 rounded-lg text-xs tracking-wider uppercase font-semibold transition disabled:opacity-50">
                  Export AI Context
                </button>
              </div>
            </div>

            {rules.length === 0 ? (
              <div className="text-center text-neutral-500 mt-20 font-light">No structural rules extracted yet. Start Distillation to map constraints.</div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                <AnimatePresence>
                  {rules.map((rule) => {
                    const match = rule.document.match(/Rule: (.*?) - Context:/);
                    const title = match ? match[1] : "Extracted Constraint";
                    const desc = rule.document.replace(/Rule: (.*?) - Context:/, "").trim();
                    const status = rule.metadata?.status || "needs_review";
                    const isApproved = status === "active";
                    const isBlocked = status === "blocked";
                    const category: string | undefined = rule.metadata?.category;
                    const confidence: number | undefined = rule.metadata?.confidence;
                    const categoryStyles: Record<string, string> = {
                      security:     "bg-red-500/10 text-red-400 border-red-500/20",
                      performance:  "bg-amber-500/10 text-amber-400 border-amber-500/20",
                      testing:      "bg-blue-500/10 text-blue-400 border-blue-500/20",
                      "code-style": "bg-neutral-500/10 text-neutral-400 border-neutral-500/20",
                      architecture: "bg-purple-500/10 text-purple-400 border-purple-500/20",
                      correctness:  "bg-green-500/10 text-green-400 border-green-500/20",
                    };
                    const categoryClass = category ? (categoryStyles[category] ?? "bg-white/5 text-neutral-400 border-white/10") : null;
                    return (
                      <motion.div key={rule.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`bg-white/[0.03] hover:bg-white/[0.06] transition border rounded-2xl p-6 ${isApproved ? 'border-green-500/20' : isBlocked ? 'border-amber-500/20 opacity-60' : 'border-white/10'}`}>
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3 flex-wrap">
                            <span className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs font-semibold text-neutral-300">
                              Density: {rule.metadata.occurrence_count || 1}x
                            </span>
                            {categoryClass && (
                              <span className={`px-3 py-1 border rounded-full text-xs font-semibold capitalize ${categoryClass}`}>
                                {category}
                              </span>
                            )}
                            {confidence !== undefined && (
                              <span className="text-xs text-neutral-500 font-mono">
                                {Math.round(confidence * 100)}% confidence
                              </span>
                            )}
                            <h3 className="text-lg font-medium">{title}</h3>
                          </div>
                          {isBlocked ? (
                            <div className="flex gap-2 relative">
                              <span className="text-xs text-amber-400 font-semibold px-4 py-2 border border-amber-500/20 rounded-full">Blocked</span>
                              <button onClick={async () => {
                                await apiFetch(`/api/rules/${rule.id}/approve`, { method: 'POST' });
                                fetchRules();
                              }} className="text-xs bg-white/5 text-neutral-300 border border-white/10 px-4 py-2 hover:bg-white hover:text-black flex-shrink-0 rounded-full font-semibold shadow hover:scale-105 transition">Unblock</button>
                              <button onClick={async () => {
                                await apiFetch(`/api/rules/${rule.id}`, { method: 'DELETE' });
                                fetchRules();
                              }} className="text-xs bg-red-500/10 text-red-400 border border-red-500/20 px-4 py-2 hover:bg-red-500 hover:text-white flex-shrink-0 rounded-full font-semibold shadow hover:scale-105 transition">Delete</button>
                            </div>
                          ) : !isApproved ? (
                            <div className="flex gap-2 relative">
                              <button onClick={async () => {
                                await apiFetch(`/api/rules/${rule.id}/approve`, { method: 'POST' });
                                fetchRules();
                              }} className="text-xs bg-white text-black px-4 py-2 flex-shrink-0 rounded-full font-semibold shadow hover:scale-105 transition">Approve Rule</button>
                              <button onClick={async () => {
                                await apiFetch(`/api/rules/${rule.id}/block`, { method: 'POST' });
                                fetchRules();
                              }} className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 px-4 py-2 hover:bg-amber-500 hover:text-black flex-shrink-0 rounded-full font-semibold shadow hover:scale-105 transition">Block</button>
                              <button onClick={async () => {
                                await apiFetch(`/api/rules/${rule.id}`, { method: 'DELETE' });
                                fetchRules();
                              }} className="text-xs bg-red-500/10 text-red-400 border border-red-500/20 px-4 py-2 hover:bg-red-500 hover:text-white flex-shrink-0 rounded-full font-semibold shadow hover:scale-105 transition">Reject</button>
                            </div>
                          ) : (
                            <div className="flex gap-2 relative">
                              <span className="text-xs text-green-400 font-semibold px-4 py-2 border border-green-500/20 rounded-full">Verified</span>
                              <button onClick={async () => {
                                await apiFetch(`/api/rules/${rule.id}/block`, { method: 'POST' });
                                fetchRules();
                              }} className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 px-4 py-2 hover:bg-amber-500 hover:text-black flex-shrink-0 rounded-full font-semibold shadow hover:scale-105 transition">Block</button>
                            </div>
                          )}
                        </div>
                        <p className="text-sm text-neutral-400 leading-relaxed max-w-4xl">{desc}</p>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            )}
          </div>
        )}

        {activeTab === "telemetry" && (
          <div className="w-full flex-1 flex flex-col items-center justify-center p-12 text-center h-full relative z-0">
            {!jobId ? (
              <div className="text-neutral-500 font-light flex flex-col items-center gap-4">
                <span className="text-4xl">💤</span>
                <p>Pipeline is currently asleep. Select a repository to initiate distillation.</p>
              </div>
            ) : (
              <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center max-w-md w-full">
                {jobProgress === 100 ? (
                  <div className="h-16 w-16 rounded-full bg-green-500/20 border border-green-500 flex items-center justify-center mb-8 text-2xl">✓</div>
                ) : jobProgress === -1 ? (
                  <div className="h-16 w-16 rounded-full bg-red-500/20 border border-red-500 flex items-center justify-center mb-8 text-2xl">✕</div>
                ) : (
                  <div className="h-16 w-16 rounded-full border-4 border-white/10 border-t-purple-500 animate-spin mb-8"></div>
                )}

                <h3 className="text-3xl font-light text-white mb-2">
                  {jobProgress === 100 ? "Matrix Integrated" : jobProgress === -1 ? "Pipeline Aborted" : "Executing Spider"}
                </h3>
                <p className={`text-lg transition-colors ${jobProgress === -1 ? 'text-red-400' : jobProgress === 100 ? 'text-green-400' : 'text-neutral-400'}`}>{jobStatus}</p>
                <div className="w-full bg-white/5 h-2 mt-8 rounded-full overflow-hidden relative">
                  <div className={`h-full transition-all duration-1000 ease-in-out ${jobProgress === 100 ? 'bg-green-500' : jobProgress === -1 ? 'bg-red-500' : 'bg-gradient-to-r from-purple-600 to-purple-400'} ${jobProgress > 0 && jobProgress < 100 ? 'animate-pulse' : ''}`} style={{ width: `${Math.max(5, jobProgress)}%` }}></div>
                </div>
                <p className="text-xs text-neutral-500 mt-4 uppercase tracking-widest flex justify-between w-full">
                  <span>Bridging Matrix Engine</span>
                  <span>{jobProgress}%</span>
                </p>

                {jobProgress > 0 && jobProgress < 100 && jobProgress !== -1 && (
                  <button onClick={handleCancelJob} className="mt-8 text-xs px-4 py-2 text-neutral-400 border border-neutral-700/50 hover:border-red-500/50 hover:text-red-400 hover:bg-red-500/10 rounded-full transition-all">
                    Cancel Distillation
                  </button>
                )}
              </motion.div>
            )}
          </div>
        )}
      </div>

      {/* ENTIRELY ESCAPED ROOT LEVEL OVERLAYS - BYPASSING ANY RIGHT PANEL Z-INDEX OR SCROLL CLIPPING */}
      {/* ADD REPO OVERLAY */}
      <AnimatePresence>
        {showAddRepo && (
          <motion.div initial={{ opacity: 0, scale: 0.95, y: 30 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95, y: 30 }} className="w-full max-w-sm mx-auto bg-[#1A1A1A] border border-white/10 p-8 rounded-2xl fixed z-50 top-1/4 left-1/2 -translate-x-1/2 shadow-[0_30px_100px_rgba(0,0,0,0.8)] backdrop-blur-3xl">
            <div className="flex justify-between mb-4 items-center">
              <h2 className="text-xl font-light text-white">Add Target Repository</h2>
              <button type="button" onClick={() => setShowAddRepo(false)} className="text-neutral-500 hover:text-white transition text-lg bg-white/5 w-8 h-8 rounded-full flex items-center justify-center">✕</button>
            </div>
            <form onSubmit={handleAddRepoSubmit} className="flex flex-col gap-5">
              <div>
                <label className="text-xs text-neutral-400 uppercase tracking-widest mb-2 block">GitHub Namespace Path</label>
                <input autoFocus required value={newRepoString} onChange={e => setNewRepoString(e.target.value)} placeholder="e.g. tiangolo/fastapi" className="w-full bg-black/80 border border-white/10 rounded-lg p-3 text-sm focus:border-purple-500 outline-none transition text-white shadow-inner" />
              </div>
              <button type="submit" className="w-full bg-purple-600 hover:bg-purple-500 text-white py-3 rounded-lg font-semibold transition tracking-wide mt-2 shadow-[0_0_20px_rgba(168,85,247,0.3)]">Connect Engine Root ⚡</button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>

      {/* SETTINGS OVERLAY */}
      <AnimatePresence>
        {showSettings && (
          <motion.div initial={{ opacity: 0, scale: 0.95, y: 30 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95, y: 30 }} className="w-full max-w-xl mx-auto bg-[#1A1A1A] border border-white/10 p-8 rounded-2xl fixed z-50 top-[10%] left-1/2 -translate-x-1/2 shadow-[0_30px_100px_rgba(0,0,0,0.8)] backdrop-blur-3xl max-h-[85vh] overflow-y-auto">
            <div className="flex justify-between mb-6 items-center">
              <h2 className="text-2xl font-light text-white">Global Control Architecture</h2>
              <button type="button" onClick={() => setShowSettings(false)} className="text-neutral-500 hover:text-white transition font-semibold bg-white/5 px-3 py-1 rounded-full text-sm">✕ ESC</button>
            </div>

            <form onSubmit={handleSaveConfig} className="flex flex-col gap-6">
              <div>
                <label className="text-xs text-neutral-400 uppercase tracking-widest mb-2 block">GitHub Auth Token (.env Bypass)</label>
                <input
                  value={config.github_token}
                  onChange={e => setConfig({ ...config, github_token: e.target.value })}
                  type="password"
                  placeholder="ghp_xxx..."
                  className={`w-full bg-black/80 border rounded-lg p-3 text-sm outline-none text-white shadow-inner ${
                    tokenStatus.state === "valid" ? "border-green-500/60 focus:border-green-500"
                    : tokenStatus.state === "invalid" ? "border-red-500/60 focus:border-red-500"
                    : "border-white/10 focus:border-purple-500"
                  }`}
                />
                {tokenStatus.state === "checking" && (
                  <div className="text-xs text-neutral-400 mt-2">Validating…</div>
                )}
                {tokenStatus.state === "valid" && (
                  <div className="text-xs text-green-400 mt-2">
                    ✓ Valid — authenticated as <span className="font-mono">{tokenStatus.login}</span>
                    {tokenStatus.scopes && tokenStatus.scopes.length > 0 && (
                      <span className="text-neutral-500"> · scopes: {tokenStatus.scopes.join(", ")}</span>
                    )}
                  </div>
                )}
                {tokenStatus.state === "invalid" && (
                  <div className="text-xs text-red-400 mt-2">✗ {tokenStatus.message}</div>
                )}
              </div>

              <div className="border border-white/10 p-6 rounded-xl bg-black/30 shadow-inner">
                <h3 className="text-sm font-semibold mb-5 text-purple-400 uppercase tracking-widest flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse"></span>
                  LLM Generation Matrix
                </h3>

                <div className="mb-5">
                  <label className="text-xs text-neutral-500 uppercase tracking-widest mb-2 block">1. Target Provider Array</label>
                  <select
                    value={config.llm_provider}
                    onChange={e => {
                      const newProvider = e.target.value;
                      setConfig({ ...config, llm_provider: newProvider, llm_model: "" });
                      setIsCustomModel(false);
                    }}
                    className="w-full bg-black/50 border border-white/10 rounded-lg p-3 text-sm text-white outline-none cursor-pointer focus:border-purple-500"
                  >
                    <option value="ollama">Self-hosted (Ollama / vLLM / any OpenAI-compatible)</option>
                    <option value="openai">OpenAI</option>
                    <option value="google">Google Gemini</option>
                    <option value="anthropic">Anthropic Claude</option>
                    <option value="openrouter">OpenRouter</option>
                  </select>
                </div>

                <div className="mb-5">
                  <label className="text-xs text-neutral-500 uppercase tracking-widest mb-2 block">2. Associated Model Boundary</label>
                  <select
                    value={isCustomModel ? "custom" : config.llm_model}
                    onChange={e => {
                      const val = e.target.value;
                      if (val === "custom") setIsCustomModel(true);
                      else {
                        setIsCustomModel(false);
                        setConfig({ ...config, llm_model: val });
                      }
                    }}
                    className="w-full bg-black/50 border border-white/10 rounded-lg p-3 text-sm text-purple-300 outline-none cursor-pointer focus:border-purple-500 mb-2"
                  >
                    <option value="" disabled>Select internal map...</option>
                    {(config.provider_models?.[config.llm_provider] || []).map((m: any) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                    <option value="custom">⚙️ Custom Explicit Endpoint...</option>
                  </select>

                  {isCustomModel && (
                    <input
                      value={customModelString} onChange={e => setCustomModelString(e.target.value)}
                      placeholder="e.g. huggingface/databricks"
                      className="w-full bg-black/80 border border-purple-500/50 rounded-lg p-3 text-sm text-purple-400 outline-none mt-2 shadow-inner"
                    />
                  )}
                </div>

                {/* Conditional URL Field for Local Mode */}
                {isLocalProvider(config.llm_provider) && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}>
                    <label className="text-xs text-neutral-500 uppercase tracking-widest mb-2 block">3. API Base URL (Ollama / vLLM / OpenAI-compatible)</label>
                    <input value={config.llm_api_base} onChange={e => setConfig({ ...config, llm_api_base: e.target.value })} placeholder="http://localhost:11434/v1" className="w-full bg-black/80 border border-white/10 rounded-lg p-3 text-sm font-mono text-neutral-300 outline-none focus:border-purple-500 shadow-inner" />
                  </motion.div>
                )}

                {/* Conditional API Key Field for Online Providers */}
                {!isLocalProvider(config.llm_provider) && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}>
                    <label className="text-xs text-amber-500/80 uppercase tracking-widest mb-2 block flex items-center gap-2">
                      <span>🔒</span> {getApiKeyLabel()} (Symmetrically Encrypted)
                    </label>
                    <input
                      value={config.provider_api_keys?.[config.llm_provider] || ""}
                      onChange={e => setConfig({
                        ...config,
                        provider_api_keys: { ...config.provider_api_keys, [config.llm_provider]: e.target.value }
                      })}
                      type="password"
                      placeholder={`Authorize Connection...`}
                      className="w-full bg-black/80 border border-amber-500/40 rounded-lg p-3 text-sm font-mono text-white outline-none focus:border-amber-500 shadow-inner"
                    />
                  </motion.div>
                )}
              </div>

              <div className="flex gap-4 mt-2">
                <button type="button" onClick={() => setShowSettings(false)} className="flex-1 bg-white/5 hover:bg-white/10 text-white py-4 rounded-lg font-semibold transition tracking-wide text-sm border border-white/5">Cancel Edit</button>
                <button type="submit" className="flex-[2] bg-white text-black hover:bg-neutral-200 py-4 rounded-lg font-bold shadow-[0_0_20px_rgba(255,255,255,0.2)] transition tracking-wide text-sm">✓ Submit Architecture Settings</button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showExportModal && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }} exit={{ scale: 0.95 }} className="bg-[#111] border border-white/10 p-8 rounded-2xl w-full max-w-4xl shadow-2xl overflow-y-auto max-h-[90vh]">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-light tracking-widest text-purple-400 uppercase flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse"></span>
                  Generative Profile Builder
                </h2>
                <button onClick={() => { setShowExportModal(false); setExportResult(""); }} className="text-neutral-500 hover:text-white transition">✕</button>
              </div>

              <div className="grid grid-cols-5 gap-4 mb-6">
                <div onClick={() => { setExportType("prompt"); setExportResult(""); }} className={`cursor-pointer p-4 rounded-xl border ${exportType === 'prompt' ? 'border-purple-500 bg-purple-500/10' : 'border-white/5 bg-black/30 hover:border-white/20'} transition`}>
                  <h3 className="font-semibold mb-1">Standard Prompt</h3>
                  <p className="text-xs text-neutral-400">Universal System text boundaries</p>
                </div>
                <div onClick={() => { setExportType("skill"); setExportResult(""); }} className={`cursor-pointer p-4 rounded-xl border ${exportType === 'skill' ? 'border-purple-500 bg-purple-500/10' : 'border-white/5 bg-black/30 hover:border-white/20'} transition`}>
                  <h3 className="font-semibold mb-1">Markdown Skill</h3>
                  <p className="text-xs text-neutral-400">IDEs (.cursorrules / .github)</p>
                </div>
                <div onClick={() => { setExportType("agent"); setExportResult(""); }} className={`cursor-pointer p-4 rounded-xl border ${exportType === 'agent' ? 'border-purple-500 bg-purple-500/10' : 'border-white/5 bg-black/30 hover:border-white/20'} transition`}>
                  <h3 className="font-semibold mb-1">Agent Schema</h3>
                  <p className="text-xs text-neutral-400">Machine-readable Logic Models</p>
                </div>
                <div onClick={() => { setExportType("cursorrules"); setExportResult(""); }} className={`cursor-pointer p-4 rounded-xl border ${exportType === 'cursorrules' ? 'border-purple-500 bg-purple-500/10' : 'border-white/5 bg-black/30 hover:border-white/20'} transition`}>
                  <h3 className="font-semibold mb-1">Cursor Rules</h3>
                  <p className="text-xs text-neutral-400">Drop-in .cursorrules file</p>
                </div>
                <div onClick={() => { setExportType("claude_md"); setExportResult(""); }} className={`cursor-pointer p-4 rounded-xl border ${exportType === 'claude_md' ? 'border-purple-500 bg-purple-500/10' : 'border-white/5 bg-black/30 hover:border-white/20'} transition`}>
                  <h3 className="font-semibold mb-1">Claude Code</h3>
                  <p className="text-xs text-neutral-400">CLAUDE.md for Claude Code</p>
                </div>
              </div>

              {exportType === "agent" && (
                <div className="mb-6 flex gap-6 p-4 border border-white/5 rounded-xl bg-black/30">
                  <label className="text-sm text-neutral-300 font-semibold uppercase tracking-widest flex items-center gap-3 cursor-pointer">
                    <input type="radio" name="arch_sel" className="accent-purple-500 w-4 h-4 cursor-pointer" checked={targetArch === "openai"} onChange={() => { setTargetArch("openai"); setExportResult(""); }} /> OpenAI JSON Map
                  </label>
                  <label className="text-sm text-neutral-300 font-semibold uppercase tracking-widest flex items-center gap-3 cursor-pointer">
                    <input type="radio" name="arch_sel" className="accent-purple-500 w-4 h-4 cursor-pointer" checked={targetArch === "anthropic"} onChange={() => { setTargetArch("anthropic"); setExportResult(""); }} /> Anthropic XML Map
                  </label>
                </div>
              )}

              <button onClick={handleExportContext} disabled={isExporting} className="w-full py-4 bg-white text-black font-semibold rounded-lg text-sm mb-6 disabled:opacity-50 hover:bg-neutral-200 transition uppercase tracking-widest">
                {isExporting ? "Synthesizing Profiles..." : "Generate Architecture"}
              </button>

              {exportResult && (
                <div className="relative group mt-4">
                  <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition">
                    <button onClick={() => navigator.clipboard.writeText(exportResult)} className="text-xs font-semibold bg-white/10 hover:bg-white/20 border border-white/10 text-white px-4 py-2 rounded-lg shadow-lg backdrop-blur-md transition uppercase tracking-widest">Copy to Clipboard</button>
                  </div>
                  <pre className="bg-black/80 shadow-inner border border-white/10 p-6 rounded-xl text-sm text-green-400 font-mono overflow-auto whitespace-pre-wrap max-h-96 w-full leading-relaxed custom-scrollbar">
                    {exportResult}
                  </pre>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add Rule Modal */}
      <AnimatePresence>
        {showAddRule && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }} className="bg-[#111] border border-white/10 rounded-2xl p-8 w-full max-w-lg relative">
              <button onClick={() => setShowAddRule(false)} className="absolute top-4 right-4 text-neutral-500 hover:text-white transition text-lg">x</button>
              <h2 className="text-xl font-light tracking-widest uppercase mb-6">Add Rule</h2>
              <form onSubmit={async (e) => {
                e.preventDefault();
                await apiFetch(`/api/rules`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ repo: selectedRepo, title: newRuleTitle, description: newRuleDesc, enforcement: newRuleEnforce })
                });
                setNewRuleTitle(""); setNewRuleDesc(""); setNewRuleEnforce("");
                setShowAddRule(false);
                fetchRules();
              }} className="flex flex-col gap-4">
                <div>
                  <label className="text-xs text-neutral-400 uppercase tracking-widest block mb-1">Rule Title</label>
                  <input value={newRuleTitle} onChange={e => setNewRuleTitle(e.target.value)} required placeholder="e.g. No direct DB calls in handlers" className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:border-white/30" />
                </div>
                <div>
                  <label className="text-xs text-neutral-400 uppercase tracking-widest block mb-1">Context / Description</label>
                  <textarea value={newRuleDesc} onChange={e => setNewRuleDesc(e.target.value)} required rows={3} placeholder="Why this rule exists and when it applies..." className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:border-white/30 resize-none" />
                </div>
                <div>
                  <label className="text-xs text-neutral-400 uppercase tracking-widest block mb-1">Enforcement Prompt</label>
                  <textarea value={newRuleEnforce} onChange={e => setNewRuleEnforce(e.target.value)} required rows={2} placeholder="What the AI should do when this rule is violated..." className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:border-white/30 resize-none" />
                </div>
                <button type="submit" className="w-full py-3 bg-blue-500 hover:bg-blue-400 text-white rounded-xl font-semibold text-sm tracking-wider uppercase transition">Create Rule</button>
              </form>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Universal Backdrop Blocker */}
      <AnimatePresence>
        {(showSettings || showAddRepo || showExportModal || showAddRule) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-40 bg-[#0A0A0A]/90 backdrop-blur-lg" />
        )}
      </AnimatePresence>
    </main >
  );
}
