'use client';
import { useEffect, useState } from "react";

const apiFetch = (path: string, init: RequestInit = {}) =>
  fetch(`/api/proxy${path.startsWith("/") ? path : `/${path}`}`, init);

type Config = {
  github_token: string;
  huggingface_token: string;
  github_webhook_secret: string;
  llm_provider: string;
  provider_api_keys: Record<string, string>;
  env_overrides: Record<string, boolean>;
  webhook_url: string;
};

const EMPTY_CONFIG: Config = {
  github_token: "",
  huggingface_token: "",
  github_webhook_secret: "",
  llm_provider: "",
  provider_api_keys: {},
  env_overrides: {},
  webhook_url: "",
};

export default function SecretsPage() {
  const [config, setConfig] = useState<Config>(EMPTY_CONFIG);

  useEffect(() => {
    apiFetch("/api/config")
      .then((r) => r.json())
      .then((d) => setConfig({ ...EMPTY_CONFIG, ...d }));
  }, []);

  const updateField = (field: keyof Config, value: string) => {
    setConfig({ ...config, [field]: value });
  };

  const save = async (payload: Partial<Config>) => {
    await apiFetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  };

  const isEnvOverride = (key: string) => Boolean(config.env_overrides[key]);

  return (
    <div className="min-h-screen bg-black text-white p-12 max-w-3xl mx-auto">
      <header className="mb-10">
        <h1 className="text-3xl font-light flex items-center gap-3">
          <span>🔒</span>
          <span>Encrypted Vault — AES-256</span>
        </h1>
        <p className="text-sm text-neutral-400 mt-2">
          Stored encrypted at rest with Fernet (AES-128 CBC + HMAC-SHA256). Keys
          set via environment variables are read-only here.
        </p>
      </header>

      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-4">GitHub</h2>
        <SecretInput
          label="Personal access token"
          value={isEnvOverride("github_token") ? "" : config.github_token}
          onChange={(v) => updateField("github_token", v)}
          onBlur={() => save({ github_token: config.github_token })}
          envOverride={isEnvOverride("github_token")}
          envVarName="GITHUB_TOKEN"
        />
      </section>

      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-2">HuggingFace</h2>
        <p className="text-xs text-neutral-500 mb-4">
          Optional. Without a token, downloads are rate-limited and gated models
          will fail.
        </p>
        <SecretInput
          label="HuggingFace token"
          value={isEnvOverride("huggingface_token") ? "" : config.huggingface_token}
          onChange={(v) => updateField("huggingface_token", v)}
          onBlur={() => save({ huggingface_token: config.huggingface_token })}
          envOverride={isEnvOverride("huggingface_token")}
          envVarName="HUGGINGFACE_HUB_TOKEN"
        />
      </section>

      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-4">LLM Providers</h2>
        {(["openai", "anthropic", "google", "openrouter"] as const).map((p) => {
          const envKey = `provider_api_keys.${p}`;
          const isActive = config.llm_provider === p;
          return (
            <div key={p} className="mb-4">
              <SecretInput
                label={
                  isActive
                    ? `${p[0].toUpperCase() + p.slice(1)} ★ active`
                    : p[0].toUpperCase() + p.slice(1)
                }
                value={
                  isEnvOverride(envKey)
                    ? ""
                    : config.provider_api_keys[p] || ""
                }
                onChange={(v) =>
                  setConfig({
                    ...config,
                    provider_api_keys: {
                      ...config.provider_api_keys,
                      [p]: v,
                    },
                  })
                }
                onBlur={() =>
                  save({
                    provider_api_keys: {
                      [p]: config.provider_api_keys[p] || "",
                    } as Record<string, string>,
                  })
                }
                envOverride={isEnvOverride(envKey)}
                envVarName={`${p.toUpperCase()}_API_KEY`}
              />
            </div>
          );
        })}
      </section>

      {/* Webhook section appended in P21 */}
    </div>
  );
}

function SecretInput({
  label,
  value,
  onChange,
  onBlur,
  envOverride,
  envVarName,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onBlur?: () => void;
  envOverride: boolean;
  envVarName: string;
  hint?: string;
}) {
  return (
    <div className="mb-2">
      <label className="text-xs uppercase tracking-widest text-neutral-400 mb-2 flex items-center gap-2">
        {label}
        {envOverride && (
          <span
            className="text-[10px] bg-amber-500/20 border border-amber-500/40 text-amber-300 px-2 py-0.5 rounded-full"
            title={`Set via ${envVarName} environment variable.`}
          >
            from environment
          </span>
        )}
      </label>
      <input
        type="password"
        disabled={envOverride}
        value={envOverride ? "••••••••••••••••••••" : value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        className={`w-full bg-black/80 border rounded-lg p-3 text-sm font-mono outline-none ${
          envOverride
            ? "border-white/5 text-neutral-500 cursor-not-allowed"
            : "border-white/10 focus:border-purple-500"
        }`}
      />
      {hint && <p className="text-xs text-neutral-500 mt-1">{hint}</p>}
    </div>
  );
}
