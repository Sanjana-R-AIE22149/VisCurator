import { useEffect, useState } from 'react';
import {
  TrendingUp,
  Database,
  Cpu,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  WifiOff,
  HardDrive,
  MemoryStick,
  Server,
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { fetchHealth, BASE_URL } from '../lib/api';

// ── Status badge config ───────────────────────────────────────────────────────

type SystemStatus = 'nim_online' | 'nim_offline' | 'backend_offline';

const STATUS_CONFIG: Record<SystemStatus, { dot: string; label: string }> = {
  nim_online:       { dot: 'bg-emerald-400', label: 'NIM Online' },
  nim_offline:      { dot: 'bg-amber-400',   label: 'NIM Offline' },
  backend_offline:  { dot: 'bg-red-500',     label: 'Backend Offline' },
};

// ── Telemetry types ───────────────────────────────────────────────────────────

interface Telemetry {
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  gpu_available: boolean;
  gpu_name: string | null;
  gpu_memory_used_mb: number | null;
  gpu_memory_total_mb: number | null;
  gpu_utilization_percent: number | null;
  disk_used_gb: number;
  disk_total_gb: number;
  active_jobs: number;
  total_training_runs: number;
}

// ── Activity row helpers ──────────────────────────────────────────────────────

type RowStatus = 'done' | 'running' | 'warn';

function msgTypeToStatus(msgType: string | undefined): RowStatus {
  if (msgType === 'done') return 'done';
  if (msgType === 'error') return 'warn';
  return 'running';
}

const STATUS_ICON: Record<RowStatus, React.ReactNode> = {
  done:    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />,
  running: <Loader2 className="w-3.5 h-3.5 text-teal-400 animate-spin" />,
  warn:    <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />,
};

const STATUS_BADGE: Record<RowStatus, string> = {
  done:    'bg-emerald-400/10 text-emerald-400 border-emerald-400/20',
  running: 'bg-teal-400/10 text-teal-400 border-teal-400/20',
  warn:    'bg-amber-400/10 text-amber-400 border-amber-400/20',
};

// ── Bar helper ────────────────────────────────────────────────────────────────

function UsageBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${Math.min(100, pct)}%`, background: color }}
      />
    </div>
  );
}

// ── Skeleton row ─────────────────────────────────────────────────────────────

function SkeletonBar() {
  return <div className="h-3 rounded bg-slate-800 skeleton-shimmer w-full" />;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const user           = useAppStore((s) => s.user);
  const imagesIngested = useAppStore((s) => s.imagesIngested);
  const qualityScore   = useAppStore((s) => s.qualityScore);
  const terminalLogs   = useAppStore((s) => s.terminalLogs);

  const [visible, setVisible]           = useState(false);
  const [isLoading, setIsLoading]       = useState(true);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>('backend_offline');
  const [telemetry, setTelemetry]       = useState<Telemetry | null>(null);
  const [telemetryLoading, setTelemetryLoading] = useState(true);

  // ── Initial fade-in ──
  useEffect(() => {
    const t = setTimeout(() => { setIsLoading(false); setVisible(true); }, 600);
    return () => clearTimeout(t);
  }, []);

  // ── Health polling every 30 s ──
  useEffect(() => {
    const poll = async () => {
      const health = await fetchHealth();
      if (!health.online) {
        setSystemStatus('backend_offline');
      } else {
        setSystemStatus(health.nim_connected ? 'nim_online' : 'nim_offline');
      }
    };
    poll();
    const id = setInterval(poll, 30_000);
    return () => clearInterval(id);
  }, []);

  // ── Telemetry polling every 3 s ──
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`${BASE_URL}/api/system/telemetry`, {
          signal: AbortSignal.timeout(4000),
        });
        if (!res.ok) return;
        const data: Telemetry = await res.json();
        if (!cancelled) {
          setTelemetry(data);
          setTelemetryLoading(false);
        }
      } catch {
        // backend offline — keep showing last known values
      }
    };
    poll();
    const id = setInterval(poll, 3_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // ── Metric cards ──
  const METRIC_CARDS = [
    {
      id: 'datasets',
      label: 'Images Ingested',
      value: imagesIngested !== null ? imagesIngested.toLocaleString() : '—',
      unit: '',
      delta: imagesIngested !== null ? 'from last pipeline run' : 'run a pipeline first',
      trend: 'up' as const,
      icon: Database,
      glow: 'rgba(45,212,191,0.15)',
      border: 'rgba(45,212,191,0.25)',
      iconColor: '#2dd4bf',
    },
    {
      id: 'gpu-util',
      label: 'GPU Utilization',
      value: telemetry?.gpu_available && telemetry.gpu_utilization_percent !== null
        ? String(telemetry.gpu_utilization_percent)
        : '—',
      unit: telemetry?.gpu_available ? '%' : '',
      delta: telemetry?.gpu_available
        ? (telemetry.gpu_name ?? 'GPU detected')
        : 'No GPU detected',
      trend: 'up' as const,
      icon: Cpu,
      glow: 'rgba(56,189,248,0.12)',
      border: 'rgba(56,189,248,0.22)',
      iconColor: '#38bdf8',
    },
    {
      id: 'quality',
      label: 'Quality Score',
      value: qualityScore !== null ? qualityScore.toFixed(1) : '—',
      unit: '',
      delta: qualityScore !== null
        ? qualityScore >= 70 ? 'Good quality' : qualityScore >= 40 ? 'Fair quality' : 'Poor quality'
        : 'awaiting analysis',
      trend: (qualityScore !== null && qualityScore >= 70 ? 'up' : 'down') as 'up' | 'down',
      icon: TrendingUp,
      glow: 'rgba(167,139,250,0.12)',
      border: 'rgba(167,139,250,0.22)',
      iconColor: '#a78bfa',
    },
  ];

  // ── Activity rows from terminal logs ──
  const activityRows = [...terminalLogs]
    .reverse()
    .slice(0, 6)
    .map((log) => ({
      id: log.id,
      task: log.message.length > 55 ? log.message.slice(0, 55) + '…' : log.message,
      status: msgTypeToStatus(log.msgType),
      ts: log.timestamp,
    }));

  const { dot, label: statusLabel } = STATUS_CONFIG[systemStatus];

  // ── Derived telemetry values ──
  const ramPct   = telemetry ? (telemetry.ram_used_gb / telemetry.ram_total_gb) * 100 : 0;
  const diskPct  = telemetry ? (telemetry.disk_used_gb / telemetry.disk_total_gb) * 100 : 0;
  const gpuMemPct = telemetry?.gpu_available && telemetry.gpu_memory_total_mb
    ? (telemetry.gpu_memory_used_mb! / telemetry.gpu_memory_total_mb) * 100
    : 0;

  // ── Page skeleton ──
  if (isLoading) {
    return (
      <div className="h-full overflow-y-auto max-w-7xl mx-auto px-6 py-8 space-y-8">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <div className="h-3 w-24 bg-slate-800 rounded skeleton-shimmer" />
            <div className="h-8 w-64 bg-slate-800 rounded skeleton-shimmer" />
          </div>
          <div className="h-10 w-48 bg-slate-800 rounded-xl skeleton-shimmer" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-40 rounded-2xl bg-slate-900/50 border border-slate-800/60 skeleton-shimmer" />
          ))}
        </div>
        <div className="h-96 rounded-2xl bg-slate-900/50 border border-slate-800/60 skeleton-shimmer" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div
        className="max-w-7xl mx-auto px-6 py-8 space-y-8"
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.45s ease, transform 0.45s ease',
        }}
      >
        {/* ── Header ── */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[11px] font-mono text-teal-400/80 uppercase tracking-widest mb-1">
              Command Center
            </p>
            <h1 className="text-3xl font-bold text-slate-100 tracking-tight">
              Good evening,{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-sky-400">
                {user?.name?.split(' ')[0] ?? 'Engineer'}
              </span>
            </h1>
            <p className="text-sm text-slate-500 mt-1">
              Here's your CVAgent infrastructure at a glance.
            </p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800/60 backdrop-blur-sm">
            {systemStatus === 'backend_offline'
              ? <WifiOff className="w-3.5 h-3.5 text-red-500" />
              : <div className={`w-2 h-2 rounded-full ${dot} status-pulse`} />
            }
            <span className="text-xs font-mono text-slate-400">{statusLabel}</span>
          </div>
        </div>

        {/* ── Metric Cards ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {METRIC_CARDS.map(({ id, label, value, unit, delta, trend, icon: Icon, glow, border, iconColor }) => (
            <div
              key={id}
              id={`metric-card-${id}`}
              className="relative rounded-2xl border p-5 overflow-hidden group cursor-default"
              style={{
                background: `radial-gradient(ellipse at 0% 0%, ${glow} 0%, transparent 70%), #0f172a66`,
                borderColor: border,
                backdropFilter: 'blur(12px)',
              }}
            >
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 skeleton-shimmer pointer-events-none rounded-2xl" />
              <div className="flex items-start justify-between mb-4">
                <div className="p-2.5 rounded-xl" style={{ background: glow, border: `1px solid ${border}` }}>
                  <Icon className="w-5 h-5" style={{ color: iconColor }} />
                </div>
                <div className={`flex items-center gap-1 text-xs font-medium ${trend === 'up' ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {trend === 'up' ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                  <span>{delta}</span>
                </div>
              </div>
              <p className="text-3xl font-bold text-slate-100 font-mono tracking-tight">
                {value}
                <span className="text-lg text-slate-500 ml-1">{unit}</span>
              </p>
              <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest">{label}</p>
            </div>
          ))}
        </div>

        {/* ── Infrastructure Telemetry ── */}
        <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 backdrop-blur-md overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-sky-400" />
              <h2 className="text-sm font-semibold text-slate-200">Infrastructure</h2>
              {/* LIVE badge */}
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-[9px] font-mono uppercase tracking-widest">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Live
              </span>
            </div>
            {telemetry && (
              <span className="text-[10px] font-mono text-slate-600">
                {telemetry.active_jobs} active job{telemetry.active_jobs !== 1 ? 's' : ''} · {telemetry.total_training_runs} training run{telemetry.total_training_runs !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-px bg-slate-800/30">
            {/* CPU */}
            <div className="bg-slate-900/60 px-5 py-4 space-y-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-sky-400" />
                <span className="text-[10px] uppercase tracking-widest text-slate-500 font-mono">CPU</span>
              </div>
              {telemetryLoading ? (
                <SkeletonBar />
              ) : (
                <>
                  <p className="text-2xl font-bold font-mono text-slate-100">
                    {telemetry!.cpu_percent}
                    <span className="text-sm text-slate-500 ml-0.5">%</span>
                  </p>
                  <UsageBar pct={telemetry!.cpu_percent} color="#38bdf8" />
                </>
              )}
            </div>

            {/* RAM */}
            <div className="bg-slate-900/60 px-5 py-4 space-y-3">
              <div className="flex items-center gap-2">
                <MemoryStick className="w-3.5 h-3.5 text-violet-400" />
                <span className="text-[10px] uppercase tracking-widest text-slate-500 font-mono">RAM</span>
              </div>
              {telemetryLoading ? (
                <SkeletonBar />
              ) : (
                <>
                  <p className="text-2xl font-bold font-mono text-slate-100">
                    {telemetry!.ram_used_gb}
                    <span className="text-sm text-slate-500 ml-0.5">/ {telemetry!.ram_total_gb} GB</span>
                  </p>
                  <UsageBar pct={ramPct} color="#a78bfa" />
                </>
              )}
            </div>

            {/* GPU */}
            <div className="bg-slate-900/60 px-5 py-4 space-y-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-teal-400" />
                <span className="text-[10px] uppercase tracking-widest text-slate-500 font-mono">GPU</span>
              </div>
              {telemetryLoading ? (
                <SkeletonBar />
              ) : !telemetry!.gpu_available ? (
                <p className="text-xs text-slate-600 font-mono pt-1">No GPU detected</p>
              ) : (
                <>
                  <p className="text-2xl font-bold font-mono text-slate-100">
                    {telemetry!.gpu_utilization_percent}
                    <span className="text-sm text-slate-500 ml-0.5">%</span>
                  </p>
                  <UsageBar pct={telemetry!.gpu_utilization_percent!} color="#2dd4bf" />
                  <p className="text-[10px] text-slate-600 font-mono truncate">
                    {telemetry!.gpu_memory_used_mb} / {telemetry!.gpu_memory_total_mb} MB · {telemetry!.gpu_name}
                  </p>
                </>
              )}
            </div>

            {/* Disk */}
            <div className="bg-slate-900/60 px-5 py-4 space-y-3">
              <div className="flex items-center gap-2">
                <HardDrive className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-[10px] uppercase tracking-widest text-slate-500 font-mono">Disk</span>
              </div>
              {telemetryLoading ? (
                <SkeletonBar />
              ) : (
                <>
                  <p className="text-2xl font-bold font-mono text-slate-100">
                    {telemetry!.disk_used_gb}
                    <span className="text-sm text-slate-500 ml-0.5">/ {telemetry!.disk_total_gb} GB</span>
                  </p>
                  <UsageBar pct={diskPct} color="#fbbf24" />
                </>
              )}
            </div>
          </div>

          {/* GPU memory bar (only when GPU present) */}
          {!telemetryLoading && telemetry?.gpu_available && (
            <div className="px-6 py-3 border-t border-slate-800/40 flex items-center gap-3">
              <span className="text-[10px] font-mono text-slate-600 w-20 shrink-0">VRAM</span>
              <div className="flex-1">
                <UsageBar pct={gpuMemPct} color="#2dd4bf" />
              </div>
              <span className="text-[10px] font-mono text-slate-500 w-32 text-right shrink-0">
                {telemetry.gpu_memory_used_mb} / {telemetry.gpu_memory_total_mb} MB
              </span>
            </div>
          )}
        </div>

        {/* ── Recent Activity Table ── */}
        <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 backdrop-blur-md overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-teal-400" />
              <h2 className="text-sm font-semibold text-slate-200">Recent Activity</h2>
              <span className="ml-1 px-2 py-0.5 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-400 text-[10px] font-mono">
                {activityRows.length} entries
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            {activityRows.length === 0 ? (
              <div className="px-6 py-10 text-center text-xs text-slate-600 font-mono">
                No pipeline activity yet — run a dataset job to see logs here.
              </div>
            ) : (
              <table className="w-full text-sm" id="activity-table">
                <thead>
                  <tr className="border-b border-slate-800/40">
                    {['Message', 'Status', 'Time'].map((col) => (
                      <th
                        key={col}
                        className="px-6 py-3 text-left text-[10px] uppercase tracking-widest text-slate-600 font-medium"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {activityRows.map((row, i) => (
                    <tr
                      key={row.id}
                      className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                      style={{ animationDelay: `${i * 60}ms` }}
                    >
                      <td className="px-6 py-3.5 text-slate-300 text-xs font-mono max-w-xs truncate">
                        {row.task}
                      </td>
                      <td className="px-6 py-3.5">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-medium uppercase tracking-wider ${STATUS_BADGE[row.status]}`}>
                          {STATUS_ICON[row.status]}
                          {row.status}
                        </span>
                      </td>
                      <td className="px-6 py-3.5">
                        <div className="flex items-center gap-1.5 text-slate-600 text-xs">
                          <Clock className="w-3 h-3" />
                          {row.ts}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="px-6 py-4 border-t border-slate-800/30">
            <div className="flex items-center gap-3">
              <div className={`h-1.5 w-1.5 rounded-full ${dot} status-pulse`} />
              <span className="text-[10px] font-mono text-slate-600">
                {statusLabel} — polling every 30s
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
