import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Wand2, Play, Download, RefreshCw, ChevronDown, BarChart2,
  Layers, Zap, Shield, Stethoscope, AlertTriangle, CheckCircle2,
  XCircle, Info, ArrowRight, Loader2
} from 'lucide-react';
import { authHeaders, connectPipelineWebSocket, BASE_URL, getDatasetJobs, type JobListItem } from '../lib/api';

const API = BASE_URL;

/* ── Types ──────────────────────────────────────────────────────────────── */
interface ClassSummary { original: number; augmented: number; total: number }
interface AugReport {
  strategy: string; multiplier: number; target_px: number; balance: boolean;
  total_input: number; total_output: number; expansion_ratio: number;
  class_summary: Record<string, ClassSummary>;
}
interface LogEntry { type: string; message: string; data?: any; timestamp: string }

type Strategy = 'light' | 'medium' | 'heavy' | 'medical' | 'adversarial';
type Phase = 'idle' | 'running' | 'done' | 'error';

/* ── Strategy definitions ────────────────────────────────────────────────── */
const STRATEGIES: {
  id: Strategy; label: string; icon: any;
  desc: string; color: string; glow: string
}[] = [
  { id: 'light', label: 'Light', icon: Zap, desc: 'Flip, gentle rotate & brightness. Safe for any domain.', color: 'from-sky-500/20 to-sky-500/5 border-sky-500/30 text-sky-400', glow: 'shadow-sky-500/20' },
  { id: 'medium', label: 'Medium', icon: Layers, desc: 'Rich spatial + photometric mix. Best general-purpose choice.', color: 'from-teal-500/20 to-teal-500/5 border-teal-500/30 text-teal-400', glow: 'shadow-teal-500/20' },
  { id: 'heavy', label: 'Heavy', icon: BarChart2, desc: 'Aggressive distortion, noise & dropout. Maximum diversity.', color: 'from-violet-500/20 to-violet-500/5 border-violet-500/30 text-violet-400', glow: 'shadow-violet-500/20' },
  { id: 'medical', label: 'Medical', icon: Stethoscope, desc: 'Structure-preserving. Avoids hue shifts for medical images.', color: 'from-rose-500/20 to-rose-500/5 border-rose-500/30 text-rose-400', glow: 'shadow-rose-500/20' },
  { id: 'adversarial', label: 'Adversarial', icon: Shield, desc: 'Extreme transforms to stress-test model robustness.', color: 'from-amber-500/20 to-amber-500/5 border-amber-500/30 text-amber-400', glow: 'shadow-amber-500/20' },
];

export default function AugmentationPage() {
  const [jobId, setJobId]             = useState('');
  const [inputDir, setInputDir]       = useState('');
  const [jobs, setJobs]               = useState<JobListItem[]>([]);
  const [strategy, setStrategy]       = useState<Strategy>('medium');
  const [multiplier, setMultiplier]   = useState(3);
  const [targetPx, setTargetPx]       = useState(224);
  const [balance, setBalance]         = useState(true);
  const [maxWorkers, setMaxWorkers]   = useState(4);
  const [phase, setPhase]             = useState<Phase>('idle');
  const [logs, setLogs]               = useState<LogEntry[]>([]);
  const [report, setReport]           = useState<AugReport | null>(null);
  const [progress, setProgress]       = useState(0);
  const [error, setError]             = useState('');
  const wsRef  = useRef<ReturnType<typeof connectPipelineWebSocket> | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  /* scroll log to bottom */
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  /* cleanup WS on unmount */
  useEffect(() => () => { wsRef.current?.close(); }, []);

  /* page title */
  useEffect(() => { document.title = 'Augmentation — VisCurator'; }, []);

  /* fetch recent jobs for picker */
  useEffect(() => {
    getDatasetJobs().then(list => setJobs(list)).catch(() => {});
  }, []);

  const connectWs = useCallback((jid: string) => {
    wsRef.current?.close();
    const ws = connectPipelineWebSocket(
      jid,
      (msg) => {
        setLogs(prev => [...prev.slice(-300), msg as any]);
        if ((msg.data as any)?.augmentation_report) {
          const r = (msg.data as any).augmentation_report;
          setReport(r.total_output ? r : r);
        }
        if ((msg.data as any)?.progress !== undefined) setProgress((msg.data as any).progress as number);
        if (msg.type === 'done') { setPhase('done'); setProgress(100); }
        if (msg.type === 'error') { setPhase('error'); setError(msg.message); }
      },
      () => {},
      () => setPhase('error'),
    );
    wsRef.current = ws;
  }, []);

  const handleRun = async () => {
    if (!jobId.trim() && !inputDir.trim()) { setError('Provide a Job ID or input directory path.'); return; }
    setError(''); setLogs([]); setReport(null); setProgress(0); setPhase('running');
    const activeJobId = jobId.trim() || '00000000-0000-0000-0000-000000000000';
    connectWs(activeJobId);
    try {
      const res = await fetch(`${API}/api/dataset/augment-agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({
          job_id: activeJobId,
          input_dir: inputDir.trim() || undefined,
          strategy, multiplier, target_px: targetPx, balance, max_workers: maxWorkers,
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `HTTP ${res.status}`);
      }
    } catch (err: any) {
      setPhase('error'); setError(err.message);
    }
  };

  const handleDownload = async () => {
    const jid = jobId.trim() || '00000000-0000-0000-0000-000000000000';
    try {
      const res = await fetch(`${API}/api/dataset/download-augmented-agent/${jid}`, {
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error(`Download failed (${res.status})`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `augmented_${jid.slice(0, 8)}.zip`;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
    } catch (err: any) { setError(err.message); }
  };

  const selectedStrat = STRATEGIES.find(s => s.id === strategy)!;

  return (
    <div className="h-full overflow-y-auto bg-[#0a0a0a] text-slate-100 p-6 space-y-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500/20 to-violet-500/5 border border-violet-500/30 flex items-center justify-center shadow-lg shadow-violet-500/10">
            <Wand2 className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-100 tracking-tight">Augmentation Agent</h1>
            <p className="text-xs text-slate-500">Expand any dataset with intelligent, class-aware transforms</p>
          </div>
        </div>
        {phase === 'done' && (
          <button
            id="aug-download-btn"
            onClick={handleDownload}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-400 hover:bg-teal-500/20 transition-all text-sm font-medium"
          >
            <Download className="w-4 h-4" /> Download ZIP
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

        {/* ── Config Panel ── */}
        <div className="xl:col-span-1 space-y-4">

          {/* Source */}
          <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-4 space-y-3">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Data Source</h2>

            {/* Job picker from existing uploads */}
            {jobs.length > 0 && (
              <div className="space-y-1.5">
                <label className="block text-xs text-slate-500">Pick a recent job</label>
                <select
                  id="aug-job-picker"
                  value={jobId}
                  onChange={e => setJobId(e.target.value)}
                  className="w-full bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-violet-500/50 transition"
                >
                  <option value="">— select a job —</option>
                  {jobs.map(j => (
                    <option key={j.job_id} value={j.job_id}>
                      {j.job_id.slice(0, 8)}… · {j.query || 'uploaded'} · {new Date(j.created_at).toLocaleDateString()}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="space-y-2">
              <label className="block text-xs text-slate-500">{jobs.length > 0 ? 'Or enter Job ID manually' : 'Job ID (from upload / annotator)'}</label>
              <input
                id="aug-job-id-input"
                value={jobId}
                onChange={e => setJobId(e.target.value)}
                placeholder="e.g. 3f8a1b2c-..."
                className="w-full bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-violet-500/50 transition"
              />
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-600">
              <div className="flex-1 h-px bg-slate-800" /> <span>or</span> <div className="flex-1 h-px bg-slate-800" />
            </div>
            <div className="space-y-2">
              <label className="block text-xs text-slate-500">Direct path to ImageFolder</label>
              <input
                id="aug-input-dir-input"
                value={inputDir}
                onChange={e => setInputDir(e.target.value)}
                placeholder="C:\datasets\my_dataset"
                className="w-full bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-violet-500/50 transition"
              />
            </div>
          </div>

          {/* Strategy picker */}
          <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-4 space-y-3">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Augmentation Strategy</h2>
            <div className="grid grid-cols-1 gap-2">
              {STRATEGIES.map(s => {
                const Icon = s.icon;
                const active = strategy === s.id;
                return (
                  <button
                    key={s.id}
                    id={`aug-strategy-${s.id}`}
                    onClick={() => setStrategy(s.id)}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border bg-gradient-to-r transition-all text-left ${
                      active ? s.color + ' shadow-lg ' + s.glow : 'border-slate-800/50 text-slate-500 hover:text-slate-300 hover:border-slate-700/50'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <div className="min-w-0">
                      <div className="text-xs font-semibold">{s.label}</div>
                      <div className="text-[10px] opacity-70 truncate">{s.desc}</div>
                    </div>
                    {active && <CheckCircle2 className="w-3.5 h-3.5 ml-auto shrink-0" />}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Parameters */}
          <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-4 space-y-4">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Parameters</h2>

            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Multiplier</span>
                <span className="text-violet-400 font-mono">{multiplier}×</span>
              </div>
              <input type="range" min="1.5" max="10" step="0.5" value={multiplier}
                onChange={e => setMultiplier(+e.target.value)}
                className="w-full accent-violet-500" id="aug-multiplier-slider" />
              <p className="text-[10px] text-slate-600">Output ≈ input × {multiplier}</p>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Output Resolution</span>
                <span className="text-teal-400 font-mono">{targetPx}×{targetPx}</span>
              </div>
              <input type="range" min="64" max="512" step="32" value={targetPx}
                onChange={e => setTargetPx(+e.target.value)}
                className="w-full accent-teal-500" id="aug-resolution-slider" />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-400">Balance Classes</div>
                <div className="text-[10px] text-slate-600">Minority classes get more variants</div>
              </div>
              <button
                id="aug-balance-toggle"
                onClick={() => setBalance(b => !b)}
                className={`relative w-10 h-5 rounded-full transition-colors ${balance ? 'bg-teal-500' : 'bg-slate-700'}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${balance ? 'translate-x-5' : ''}`} />
              </button>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Workers</span>
                <span className="text-slate-300 font-mono">{maxWorkers}</span>
              </div>
              <input type="range" min="1" max="8" step="1" value={maxWorkers}
                onChange={e => setMaxWorkers(+e.target.value)}
                className="w-full accent-slate-400" id="aug-workers-slider" />
            </div>
          </div>

          {/* Run button */}
          <button
            id="aug-run-btn"
            onClick={handleRun}
            disabled={phase === 'running'}
            className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-500 hover:to-violet-400 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm shadow-lg shadow-violet-500/25 transition-all"
          >
            {phase === 'running'
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Running…</>
              : <><Play className="w-4 h-4" /> Run Augmentation Agent</>}
          </button>

          {error && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
              <XCircle className="w-4 h-4 shrink-0 mt-0.5" /> {error}
            </div>
          )}
        </div>

        {/* ── Right Panel: Live log + Report ── */}
        <div className="xl:col-span-2 space-y-4">

          {/* Progress bar */}
          {phase === 'running' && (
            <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-4">
              <div className="flex justify-between text-xs text-slate-400 mb-2">
                <span>Augmenting dataset…</span>
                <span className="font-mono text-violet-400">{progress}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-violet-600 to-teal-500 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Live log */}
          <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-800/60 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Live Log</span>
              {phase === 'running' && <span className="flex items-center gap-1.5 text-xs text-teal-400"><span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />Live</span>}
            </div>
            <div ref={logRef} className="h-64 overflow-y-auto p-3 space-y-1 font-mono text-[11px]">
              {logs.length === 0
                ? <div className="flex items-center gap-2 text-slate-600 py-8 justify-center"><Info className="w-4 h-4" /> Configure settings and click Run.</div>
                : logs.map((l, i) => (
                  <div key={i} className={`flex gap-2 ${
                    l.type === 'error' ? 'text-rose-400' :
                    l.type === 'done'  ? 'text-teal-400' :
                    l.type === 'tool_result' ? 'text-violet-400' :
                    'text-slate-400'
                  }`}>
                    <span className="text-slate-700 shrink-0">{l.timestamp?.slice(11, 19) ?? ''}</span>
                    <span className="break-all">{l.message}</span>
                  </div>
                ))
              }
            </div>
          </div>

          {/* Report card */}
          {report && (
            <div className="rounded-xl bg-slate-900/60 border border-teal-500/20 p-4 space-y-4">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-400" />
                <h2 className="text-sm font-semibold text-teal-400">Augmentation Complete</h2>
              </div>

              {/* Stat pills */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: 'Input Images',  value: report.total_input,    color: 'text-slate-300' },
                  { label: 'Output Images', value: report.total_output,   color: 'text-teal-400' },
                  { label: 'Expansion',     value: `${report.expansion_ratio}×`, color: 'text-violet-400' },
                  { label: 'Resolution',    value: `${report.target_px}px`, color: 'text-sky-400' },
                ].map(stat => (
                  <div key={stat.label} className="bg-slate-800/50 rounded-lg p-3 text-center">
                    <div className={`text-xl font-bold font-mono ${stat.color}`}>{stat.value}</div>
                    <div className="text-[10px] text-slate-500 mt-0.5">{stat.label}</div>
                  </div>
                ))}
              </div>

              {/* Per-class table */}
              {report.class_summary && Object.keys(report.class_summary).length > 0 && (
                <div>
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">Per-Class Breakdown</h3>
                  <div className="space-y-2">
                    {Object.entries(report.class_summary).map(([cls, info]) => {
                      const pct = info.total ? Math.round((info.augmented / info.total) * 100) : 0;
                      return (
                        <div key={cls}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-300 font-medium truncate max-w-[40%]">{cls}</span>
                            <span className="text-slate-500 font-mono">
                              {info.original} <ArrowRight className="w-3 h-3 inline" /> {info.total}
                              <span className="text-teal-400 ml-1">(+{info.augmented})</span>
                            </span>
                          </div>
                          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-violet-500 to-teal-500 transition-all duration-700"
                              style={{ width: `${Math.min(pct, 100)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <button
                id="aug-download-report-btn"
                onClick={handleDownload}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-400 hover:bg-teal-500/20 transition-all text-sm font-medium"
              >
                <Download className="w-4 h-4" /> Download Augmented Dataset
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
