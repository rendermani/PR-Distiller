'use client';
import { useEffect, useState } from "react";

const apiFetch = (path: string, init: RequestInit = {}) =>
  fetch(`/api/proxy${path.startsWith("/") ? path : `/${path}`}`, init);

type Config = {
  github_token: string;
  huggingface_token: string;
  github_webhook_secret: string;
  llm_models: LlmModel[];
  llm_models_active: string[];
  provider_api_keys: Record<string, string>;
  env_overrides: Record<string, boolean>;
  webhook_url: string;
};

type LlmModel = {
  id: string;
  model: string;
};

const EMPTY_CONFIG: Config = {
  github_token: "",
  huggingface_token: "",
  github_webhook_secret: "",
  llm_models: [],
  llm_models_active: [],
  provider_api_keys: {},
  env_overrides: {},
  webhook_url: "",
};

/** LiteLLM provider prefix of a model string: "openai/gpt-4o" -> "openai".
 *  An unprefixed model means OpenAI, matching _provider_from_model() in
 *  backend/pipeline/model_registry.py. */
const providerOf = (model: string): string =>
  model.includes("/") ? model.split("/", 1)[0] : "openai";

export default function SecretsPage() {
  const [config, setConfig] = useState<Config>(EMPTY_CONFIG);
  const [saveError, setSaveError] = useState("");

  useEffect(() => {
    apiFetch("/api/config")
      .then((r) => r.json())
      .then((d) => setConfig({ ...EMPTY_CONFIG, ...d }));
  }, []);

  const updateField = (field: keyof Config, value: string) => {
    setConfig({ ...config, [field]: value });
  };

  const save = async (payload: Partial<Config>) => {
    // Report a rejected save. Discarding the response meant a 4xx looked
    // identical to success: the field kept the typed value on screen while
    // nothing was persisted.
    const res = await apiFetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      setSaveError(`Save failed (HTTP ${res.status}). ${detail.slice(0, 300)}`);
      return;
    }
    setSaveError("");
  };

  const isEnvOverride = (key: string) => Boolean(config.env_overrides[key]);

  // Which providers the next job will actually reach, derived from the active
  // models' LiteLLM prefixes. Replaces a single llm_provider field that the
  // multi-model refactor removed, leaving the marker permanently unlit.
  const activeProviders = new Set(
    config.llm_models
      .filter((m) => config.llm_models_active.includes(m.id))
      .map((m) => providerOf(m.model)),
  );

  return (
    <div className="min-h-screen bg-black text-white p-12 max-w-3xl mx-auto">
      <header className="mb-10">
        <h1 className="text-3xl font-light flex items-center gap-3">
          <span>🔒</span>
          {/* Fernet is AES-128-CBC; the old "AES-256" heading contradicted the
              accurate description directly below it. */}
          <span>Encrypted Vault — AES-128</span>
        </h1>
        <p className="text-sm text-neutral-400 mt-2">
          Stored encrypted at rest with Fernet (AES-128 CBC + HMAC-SHA256). Keys
          set via environment variables are read-only here.
        </p>
      </header>

      {saveError && (
        <div
          className="mb-6 text-xs text-red-400 border border-red-500/40 rounded-lg p-3 bg-red-500/5"
          role="alert"
        >
          {saveError}
        </div>
      )}

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
          const isActive = activeProviders.has(p);
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

      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-2">GitHub Webhook</h2>
        <p className="text-xs text-neutral-500 mb-4">
          In GitHub's webhook config, paste the URL below and the secret. Subscribe to: <strong>Pull requests</strong>.
        </p>

        <div className="mb-4">
          <label className="text-xs uppercase tracking-widest text-neutral-400 mb-2 block">
            Webhook URL
          </label>
          <div className="flex gap-2">
            <input
              readOnly
              value={config.webhook_url}
              className="flex-1 bg-black/80 border border-white/10 rounded-lg p-3 text-sm font-mono text-neutral-300"
            />
            <button
              onClick={() => navigator.clipboard.writeText(config.webhook_url)}
              className="px-4 bg-white/5 border border-white/10 rounded-lg text-sm hover:bg-white/10"
            >
              Copy
            </button>
          </div>
        </div>

        <WebhookSecret
          value={config.github_webhook_secret}
          envOverride={isEnvOverride("github_webhook_secret")}
          onChange={(v) => updateField("github_webhook_secret", v)}
          onSave={(v) => save({ github_webhook_secret: v })}
        />
      </section>
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

function WebhookSecret({
  value,
  envOverride,
  onChange,
  onSave,
}: {
  value: string;
  envOverride: boolean;
  onChange: (v: string) => void;
  onSave: (v: string) => void;
}) {
  const [revealed, setRevealed] = useState(false);

  const generate = () => {
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    const hex = Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
    onChange(hex);
    onSave(hex);
  };

  const revealAndCopy = async () => {
    setRevealed(true);
    await navigator.clipboard.writeText(value);
    setTimeout(() => setRevealed(false), 3000);
  };

  return (
    <div>
      <label className="text-xs uppercase tracking-widest text-neutral-400 mb-2 flex items-center gap-2">
        Webhook secret
        {envOverride && (
          <span
            className="text-[10px] bg-amber-500/20 border border-amber-500/40 text-amber-300 px-2 py-0.5 rounded-full"
            title="Set via GITHUB_WEBHOOK_SECRET environment variable."
          >
            from environment
          </span>
        )}
      </label>
      <div className="flex gap-2">
        <input
          type={revealed ? "text" : "password"}
          readOnly={envOverride}
          value={value || ""}
          onChange={(e) => !envOverride && onChange(e.target.value)}
          className={`flex-1 bg-black/80 border rounded-lg p-3 text-sm font-mono ${
            envOverride
              ? "border-white/5 text-neutral-500 cursor-not-allowed"
              : "border-white/10 focus:border-purple-500"
          }`}
        />
        {value && (
          <button
            onClick={revealAndCopy}
            className="px-4 bg-amber-500/10 border border-amber-500/40 rounded-lg text-amber-300 text-sm hover:bg-amber-500/20"
          >
            {revealed ? "Copied" : "Reveal & Copy"}
          </button>
        )}
        {!envOverride && (
          <button
            onClick={generate}
            className="px-4 bg-purple-500/20 border border-purple-500/40 rounded-lg text-purple-200 text-sm hover:bg-purple-500/30"
          >
            {value ? "Rotate" : "Generate"}
          </button>
        )}
      </div>
    </div>
  );
}
