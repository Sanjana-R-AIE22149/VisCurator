import { useEffect, useRef, useState } from 'react';
import { BASE_URL } from '../../lib/api';

// ── Types ──────────────────────────────────────────────────────────────────────

type LineStatus = 'pending' | 'ok' | 'warn' | 'error' | 'running';

interface BootLine {
  id: string;
  label: string;
  status: LineStatus;
  detail?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function statusColor(s: LineStatus) {
  return s === 'ok'
    ? '#34d399'
    : s === 'warn'
    ? '#fbbf24'
    : s === 'error'
    ? '#f87171'
    : s === 'running'
    ? '#60a5fa'
    : '#475569';
}

function statusGlyph(s: LineStatus) {
  if (s === 'ok') return '✓';
  if (s === 'warn') return '⚠';
  if (s === 'error') return '✗';
  if (s === 'running') return '◌';
  return '·';
}

function delay(ms: number) {
  return new Promise<void>((res) => setTimeout(res, ms));
}

// ── Cursor blink ───────────────────────────────────────────────────────────────

function Cursor() {
  const [vis, setVis] = useState(true);
  useEffect(() => {
    const id = setInterval(() => setVis((v) => !v), 500);
    return () => clearInterval(id);
  }, []);
  return (
    <span
      style={{
        display: 'inline-block',
        width: 8,
        height: 14,
        background: vis ? '#60a5fa' : 'transparent',
        verticalAlign: 'middle',
        borderRadius: 1,
        marginLeft: 2,
      }}
    />
  );
}

// ── Boot progress bar ──────────────────────────────────────────────────────────

function ProgressBar({ pct }: { pct: number }) {
  return (
    <div
      style={{
        width: '100%',
        height: 2,
        background: '#1e293b',
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          height: '100%',
          width: `${pct}%`,
          background: 'linear-gradient(90deg, #3b82f6, #06b6d4)',
          transition: 'width 0.35s ease',
          borderRadius: 2,
          boxShadow: '0 0 8px #3b82f680',
        }}
      />
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

interface BootScreenProps {
  onDone: () => void;
}

export default function BootScreen({ onDone }: BootScreenProps) {
  const [lines, setLines] = useState<BootLine[]>([]);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<'booting' | 'done' | 'fading'>('booting');
  const [terminalText, setTerminalText] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const doneRef = useRef(false);

  const updateLine = (id: string, patch: Partial<BootLine>) =>
    setLines((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));

  const addLine = (line: BootLine) =>
    setLines((prev) => [...prev, line]);

  // Typewriter for header
  useEffect(() => {
    const msg = 'VisCurator v0.2 · System Boot Sequence';
    let i = 0;
    const id = setInterval(() => {
      setTerminalText(msg.slice(0, i + 1));
      i++;
      if (i >= msg.length) clearInterval(id);
    }, 28);
    return () => clearInterval(id);
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines]);

  // Boot sequence
  useEffect(() => {
    let cancelled = false;

    async function run() {
      await delay(400);

      // ── Step 1: Frontend init ──────────────────────────────────
      addLine({ id: 'ui', label: 'Initializing React runtime', status: 'running' });
      setProgress(8);
      await delay(420);
      if (cancelled) return;
      updateLine('ui', { status: 'ok', detail: 'React 19 · Vite · Zustand · XYFlow' });
      setProgress(15);

      // ── Step 2: Ping backend ───────────────────────────────────
      addLine({ id: 'backend', label: 'Connecting to FastAPI backend (localhost:8000)', status: 'running' });
      setProgress(25);
      await delay(250);

      let healthData: Record<string, unknown> | null = null;
      let backendOnline = false;
      try {
        const res = await fetch(`${BASE_URL}/api/health`, {
          signal: AbortSignal.timeout(4000),
        });
        if (res.ok) {
          healthData = await res.json();
          backendOnline = true;
        }
      } catch {
        /* offline */
      }
      if (cancelled) return;

      if (backendOnline) {
        updateLine('backend', { status: 'ok', detail: 'FastAPI online · Uvicorn ASGI' });
      } else {
        updateLine('backend', {
          status: 'error',
          detail: 'Backend unreachable — start the server with python VisCurator_launch.py',
        });
      }
      setProgress(38);

      // ── Step 3: NIM / LLM ─────────────────────────────────────
      addLine({ id: 'nim', label: 'Checking NVIDIA NIM connection', status: 'running' });
      await delay(350);
      if (cancelled) return;

      const nimOk = backendOnline && (healthData?.nim_connected as boolean);
      const nimError = backendOnline ? (healthData?.nim_error as string | undefined) : undefined;
      updateLine('nim', {
        status: nimOk ? 'ok' : backendOnline ? 'warn' : 'error',
        detail: nimOk
          ? `LLaMA 3.1 70B · ${(healthData?.nim_model as string) ?? 'NIM'}`
          : nimError ?? 'NIM not configured — set NVIDIA_API_KEY in .env',
      });
      setProgress(52);

      // ── Step 4: Python packages ────────────────────────────────
      addLine({ id: 'pkgs', label: 'Verifying Python ML packages', status: 'running' });
      await delay(300);
      if (cancelled) return;

      const pkgs = (healthData?.python_packages as Record<string, boolean>) ?? {};
      const missing = Object.entries(pkgs)
        .filter(([, v]) => !v)
        .map(([k]) => k);

      updateLine('pkgs', {
        status: missing.length === 0 ? 'ok' : missing.length <= 2 ? 'warn' : 'error',
        detail:
          missing.length === 0
            ? 'torch · cv2 · albumentations · imagehash · PIL'
            : `Missing: ${missing.join(', ')}`,
      });
      setProgress(65);

      // ── Step 5: Env vars ───────────────────────────────────────
      addLine({ id: 'env', label: 'Checking environment variables', status: 'running' });
      await delay(280);
      if (cancelled) return;

      const envVars = (healthData?.env_vars as Record<string, string>) ?? {};
      const missingEnv = Object.entries(envVars)
        .filter(([k, v]) => k !== 'HF_TOKEN' && v === 'MISSING')
        .map(([k]) => k);

      updateLine('env', {
        status: missingEnv.length === 0 ? 'ok' : 'warn',
        detail:
          missingEnv.length === 0
            ? 'NVIDIA_API_KEY · HF_TOKEN · Kaggle · Roboflow'
            : `Not set: ${missingEnv.join(', ')}`,
      });
      setProgress(78);

      // ── Step 6: Active jobs ────────────────────────────────────
      addLine({ id: 'jobs', label: 'Loading active jobs & output dirs', status: 'running' });
      await delay(240);
      if (cancelled) return;

      const activeJobs = (healthData?.active_jobs as number) ?? 0;
      updateLine('jobs', {
        status: 'ok',
        detail: backendOnline
          ? `${activeJobs} active job${activeJobs !== 1 ? 's' : ''} · cvagent_output ready`
          : 'Will sync when backend comes online',
      });
      setProgress(90);

      // ── Step 7: Auth ───────────────────────────────────────────
      addLine({ id: 'auth', label: 'Validating auth session', status: 'running' });
      await delay(220);
      if (cancelled) return;

      const token = localStorage.getItem('vc_token');
      updateLine('auth', {
        status: token ? 'ok' : 'warn',
        detail: token ? 'JWT session active · Bearer token found' : 'No session — login required',
      });
      setProgress(100);

      // ── Done ───────────────────────────────────────────────────
      await delay(500);
      if (cancelled) return;
      if (!doneRef.current) {
        doneRef.current = true;
        setPhase('done');
        await delay(900);
        if (!cancelled) {
          setPhase('fading');
          await delay(550);
          onDone();
        }
      }
    }

    void run();
    return () => {
      cancelled = true;
    };
  }, [onDone]);

  const overallStatus = lines.some((l) => l.status === 'error')
    ? 'error'
    : lines.some((l) => l.status === 'warn')
    ? 'warn'
    : 'ok';

  const statusBadgeColor =
    phase === 'booting'
      ? '#60a5fa'
      : overallStatus === 'error'
      ? '#f87171'
      : overallStatus === 'warn'
      ? '#fbbf24'
      : '#34d399';

  const statusBadgeLabel =
    phase === 'booting'
      ? 'BOOTING'
      : overallStatus === 'error'
      ? 'DEGRADED'
      : overallStatus === 'warn'
      ? 'PARTIAL'
      : 'READY';

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: '#020617',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        opacity: phase === 'fading' ? 0 : 1,
        transition: 'opacity 0.55s ease',
        padding: '24px',
      }}
    >
      {/* Ambient glow */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 480,
          height: 280,
          background:
            'radial-gradient(ellipse at center, rgba(59,130,246,0.09) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      {/* Terminal window */}
      <div
        style={{
          width: '100%',
          maxWidth: 680,
          borderRadius: 14,
          border: '1px solid #1e293b',
          background: '#0a0f1e',
          boxShadow: '0 0 60px rgba(59,130,246,0.08), 0 24px 80px rgba(0,0,0,0.7)',
          overflow: 'hidden',
          fontFamily: '"Fira Code", "Cascadia Code", ui-monospace, monospace',
        }}
      >
        {/* Title bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 16px',
            borderBottom: '1px solid #1e293b',
            background: '#070d1a',
          }}
        >
          <div style={{ display: 'flex', gap: 6 }}>
            {['#ff5f57', '#febc2e', '#28c840'].map((c, i) => (
              <div
                key={i}
                style={{ width: 12, height: 12, borderRadius: '50%', background: c, opacity: 0.8 }}
              />
            ))}
          </div>
          <span style={{ flex: 1, textAlign: 'center', fontSize: 11, color: '#475569' }}>
            viscurator · boot.log
          </span>
          <div
            style={{
              fontSize: 9,
              fontWeight: 700,
              letterSpacing: '0.1em',
              color: statusBadgeColor,
              background: `${statusBadgeColor}18`,
              border: `1px solid ${statusBadgeColor}40`,
              borderRadius: 4,
              padding: '2px 7px',
            }}
          >
            {statusBadgeLabel}
          </div>
        </div>

        {/* Terminal body */}
        <div style={{ padding: '20px 22px 16px', minHeight: 280, maxHeight: 420, overflowY: 'auto' }}>
          {/* Header line */}
          <div style={{ marginBottom: 18 }}>
            <span style={{ color: '#3b82f6', fontSize: 12 }}>$ </span>
            <span style={{ color: '#e2e8f0', fontSize: 12 }}>
              {terminalText}
            </span>
            {terminalText.length < 38 && <Cursor />}
          </div>

          {/* Divider */}
          <div style={{ borderTop: '1px solid #1e293b', marginBottom: 14 }} />

          {/* Boot lines */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {lines.map((line) => (
              <div key={line.id} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                {/* Status glyph */}
                <span
                  style={{
                    color: statusColor(line.status),
                    fontSize: 12,
                    lineHeight: 1.6,
                    minWidth: 12,
                    animation: line.status === 'running' ? 'spin 1s linear infinite' : undefined,
                  }}
                >
                  {statusGlyph(line.status)}
                </span>

                {/* Label + detail */}
                <div style={{ flex: 1 }}>
                  <span
                    style={{
                      color: line.status === 'running' ? '#94a3b8' : '#cbd5e1',
                      fontSize: 12,
                      lineHeight: 1.6,
                    }}
                  >
                    {line.label}
                  </span>
                  {line.detail && (
                    <span
                      style={{
                        display: 'block',
                        color: statusColor(line.status),
                        fontSize: 10.5,
                        opacity: 0.85,
                        marginTop: 1,
                        paddingLeft: 2,
                      }}
                    >
                      {line.detail}
                    </span>
                  )}
                </div>
              </div>
            ))}

            {/* Blinking cursor while booting */}
            {phase === 'booting' && lines.length > 0 && (
              <div style={{ marginTop: 2 }}>
                <Cursor />
              </div>
            )}

            {/* Done message */}
            {phase === 'done' && (
              <div style={{ marginTop: 6, color: '#34d399', fontSize: 12 }}>
                ✓ All systems checked — launching VisCurator…
              </div>
            )}
          </div>

          <div ref={bottomRef} />
        </div>

        {/* Progress bar */}
        <div style={{ padding: '0 22px 18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ color: '#334155', fontSize: 10 }}>
              boot progress
            </span>
            <span style={{ color: '#475569', fontSize: 10 }}>{progress}%</span>
          </div>
          <ProgressBar pct={progress} />
        </div>
      </div>

      {/* Bottom tag */}
      <div
        style={{
          marginTop: 18,
          fontSize: 10,
          color: '#1e293b',
          fontFamily: 'ui-monospace, monospace',
          letterSpacing: '0.05em',
        }}
      >
        VISCURATOR · CV ENGINEERING PLATFORM · v0.2.0
      </div>

      {/* Spin keyframe via style tag */}
      <style>{`
        @keyframes spin {
          0% { opacity: 1; }
          25% { opacity: 0.4; }
          50% { opacity: 1; }
          75% { opacity: 0.4; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
