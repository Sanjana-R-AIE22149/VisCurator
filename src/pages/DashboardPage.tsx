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
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

/* ── Mock data ── */
const METRIC_CARDS = [
  {
    id: 'datasets',
    label: 'Total Datasets',
    value: '48',
    unit: '',
    delta: '+3 this week',
    trend: 'up' as const,
    icon: Database,
    accent: 'teal',
    glow: 'rgba(45,212,191,0.15)',
    border: 'rgba(45,212,191,0.25)',
  },
  {
    id: 'gpu-util',
    label: 'GPU Utilization',
    value: '73',
    unit: '%',
    delta: '+5% vs yesterday',
    trend: 'up' as const,
    icon: Cpu,
    accent: 'sky',
    glow: 'rgba(56,189,248,0.12)',
    border: 'rgba(56,189,248,0.22)',
  },
  {
    id: 'map-score',
    label: 'Best mAP Score',
    value: '0.847',
    unit: '',
    delta: '-0.012 vs baseline',
    trend: 'down' as const,
    icon: TrendingUp,
    accent: 'purple',
    glow: 'rgba(167,139,250,0.12)',
    border: 'rgba(167,139,250,0.22)',
  },
];

const ACTIVITY_ROWS = [
  { id: 1, task: 'HuggingFace ingestion — coco-annotated-v2', status: 'done', ts: '2m ago', count: '12,400 imgs' },
  { id: 2, task: 'Blur filter pass — wheat-disease-set', status: 'done', ts: '18m ago', count: '8,910 imgs' },
  { id: 3, task: 'Auto-annotation — corn-segmentation', status: 'running', ts: '34m ago', count: '5,200 imgs' },
  { id: 4, task: 'Model export — YOLOv8-nano.onnx', status: 'done', ts: '1h ago', count: '—' },
  { id: 5, task: 'Quality audit — rice-blast-2024', status: 'warn', ts: '2h ago', count: '3,100 imgs' },
  { id: 6, task: 'Training run — ResNet50 pretrain', status: 'done', ts: '5h ago', count: '50 epochs' },
];

const STATUS_ICON = {
  done: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />,
  running: <Loader2 className="w-3.5 h-3.5 text-teal-400 animate-spin" />,
  warn: <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />,
};

const STATUS_BADGE: Record<string, string> = {
  done: 'bg-emerald-400/10 text-emerald-400 border-emerald-400/20',
  running: 'bg-teal-400/10 text-teal-400 border-teal-400/20',
  warn: 'bg-amber-400/10 text-amber-400 border-amber-400/20',
};

export default function DashboardPage() {
  const user = useAppStore((s) => s.user);
  const [visible, setVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const t = setTimeout(() => {
      setIsLoading(false);
      setVisible(true);
    }, 600);
    return () => clearTimeout(t);
  }, []);

  if (isLoading) {
    return (
      <div className="h-full overflow-y-auto max-w-7xl mx-auto px-6 py-8 space-y-8 animate-fade-in">
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
            <div className="w-2 h-2 rounded-full bg-emerald-400 status-pulse" />
            <span className="text-xs font-mono text-slate-400">System Status: Online</span>
          </div>
        </div>

        {/* ── Metric Cards ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {METRIC_CARDS.map(({ id, label, value, unit, delta, trend, icon: Icon, glow, border }) => (
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
              {/* Subtle shimmer on hover */}
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 skeleton-shimmer pointer-events-none rounded-2xl" />

              <div className="flex items-start justify-between mb-4">
                <div
                  className="p-2.5 rounded-xl"
                  style={{ background: glow, border: `1px solid ${border}` }}
                >
                  <Icon className="w-5 h-5" style={{ color: glow.includes('45,212') ? '#2dd4bf' : glow.includes('56,189') ? '#38bdf8' : '#a78bfa' }} />
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

        {/* ── Recent Activity Table ── */}
        <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 backdrop-blur-md overflow-hidden">
          {/* Table header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-teal-400" />
              <h2 className="text-sm font-semibold text-slate-200">Recent Activity</h2>
              <span className="ml-1 px-2 py-0.5 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-400 text-[10px] font-mono">
                {ACTIVITY_ROWS.length} tasks
              </span>
            </div>
            <button
              id="view-all-activity-btn"
              className="text-xs text-slate-500 hover:text-teal-400 transition-colors font-medium"
            >
              View all →
            </button>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm" id="activity-table">
              <thead>
                <tr className="border-b border-slate-800/40">
                  {['Task', 'Status', 'Images / Epochs', 'Time'].map((col) => (
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
                {ACTIVITY_ROWS.map((row, i) => (
                  <tr
                    key={row.id}
                    className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors group"
                    style={{ animationDelay: `${i * 60}ms` }}
                  >
                    <td className="px-6 py-3.5 text-slate-300 text-xs font-mono max-w-xs truncate">
                      {row.task}
                    </td>
                    <td className="px-6 py-3.5">
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-medium uppercase tracking-wider ${STATUS_BADGE[row.status]}`}
                      >
                        {STATUS_ICON[row.status as keyof typeof STATUS_ICON]}
                        {row.status}
                      </span>
                    </td>
                    <td className="px-6 py-3.5 text-slate-500 text-xs font-mono">{row.count}</td>
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
          </div>

          {/* Skeleton loading rows (decorative) */}
          <div className="px-6 py-4 border-t border-slate-800/30">
            <div className="flex items-center gap-3">
              <div className="h-1.5 w-1.5 rounded-full bg-teal-500 status-pulse" />
              <span className="text-[10px] font-mono text-slate-600">Live updates enabled — polling every 30s</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
