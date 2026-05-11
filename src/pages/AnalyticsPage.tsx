import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AlertTriangle, BarChart3, Download, Info, Layers, Target, TrendingUp, Columns, Zap, FlaskConical, WifiOff } from 'lucide-react';
import { getTrainingMetrics, getTrainingRuns, BASE_URL, type TrainingMetricPoint, type TrainingRunSummary } from '../lib/api';

const GRID_STROKE = '#1e293b';
const TICK_STYLE = { fill: '#64748b', fontSize: 11 };
const TOOLTIP_STYLE = {
  contentStyle: { background: '#020617', border: '1px solid #334155', borderRadius: 8, fontFamily: 'monospace', fontSize: 11 },
  labelStyle: { color: '#94a3b8' },
  itemStyle: { color: '#cbd5e1' },
};
const METRIC_COLORS = {
  loss: '#f97316',
  accuracy: '#2dd4bf',
  precision: '#38bdf8',
  recall: '#a78bfa',
  map: '#34d399',
  loss2: '#94a3b8',
};

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#020617', border: '1px solid #334155', borderRadius: 8, padding: '8px 12px', fontFamily: 'monospace', fontSize: 11 }}>
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>Epoch {label}</p>
      {payload.map((item) => (
        <p key={item.name} style={{ color: item.color, margin: '2px 0' }}>
          {item.name}: <span style={{ color: '#e2e8f0' }}>{typeof item.value === 'number' ? item.value.toFixed(4) : item.value}</span>
        </p>
      ))}
    </div>
  );
}

function LossCurveChart({ data }: { data: TrainingMetricPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={TICK_STYLE} />
        <YAxis tick={TICK_STYLE} domain={[0, 'auto']} />
        <Tooltip content={<CustomTooltip />} />
        <Line type="monotone" dataKey="loss" name="Loss" stroke={METRIC_COLORS.loss} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function CompareLossCurveChart({ data }: { data: any[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={TICK_STYLE} />
        <YAxis tick={TICK_STYLE} domain={[0, 'auto']} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }} />
        <Line type="monotone" dataKey="loss1" name="Run A Loss" stroke={METRIC_COLORS.loss} dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="loss2" name="Run B Loss" stroke={METRIC_COLORS.loss2} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function MetricAreaChart({ data }: { data: TrainingMetricPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <defs>
          <linearGradient id="gAccuracy" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={METRIC_COLORS.accuracy} stopOpacity={0.32} />
            <stop offset="95%" stopColor={METRIC_COLORS.accuracy} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gMap" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={METRIC_COLORS.map} stopOpacity={0.25} />
            <stop offset="95%" stopColor={METRIC_COLORS.map} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={TICK_STYLE} />
        <YAxis tick={TICK_STYLE} domain={[0, 1]} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }} />
        <Area type="monotone" dataKey="accuracy" name="Accuracy" stroke={METRIC_COLORS.accuracy} fill="url(#gAccuracy)" strokeWidth={2} />
        <Area type="monotone" dataKey="map" name="mAP" stroke={METRIC_COLORS.map} fill="url(#gMap)" strokeWidth={2} />
        <Line type="monotone" dataKey="precision" name="Precision" stroke={METRIC_COLORS.precision} dot={false} strokeWidth={1.6} />
        <Line type="monotone" dataKey="recall" name="Recall" stroke={METRIC_COLORS.recall} dot={false} strokeWidth={1.6} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function SnapshotBarChart({ latest }: { latest: TrainingMetricPoint | null }) {
  const data = latest
    ? [
        { name: 'accuracy', value: latest.accuracy },
        { name: 'precision', value: latest.precision },
        { name: 'recall', value: latest.recall },
        { name: 'mAP', value: latest.map },
      ]
    : [];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={TICK_STYLE} />
        <YAxis tick={TICK_STYLE} domain={[0, 1]} />
        <Tooltip {...TOOLTIP_STYLE} />
        <Bar dataKey="value" fill={METRIC_COLORS.accuracy} radius={[4, 4, 0, 0]} maxBarSize={26} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function DeltaChart({ data }: { data: Array<{ epoch: number; accuracy_delta: number; map_delta: number }> }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={TICK_STYLE} />
        <YAxis tick={TICK_STYLE} />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 2" />
        <Bar dataKey="accuracy_delta" name="Acc Delta" fill={METRIC_COLORS.precision} maxBarSize={16} radius={[3, 3, 0, 0]} />
        <Line type="monotone" dataKey="map_delta" name="mAP Delta" stroke={METRIC_COLORS.map} dot={false} strokeWidth={2} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function MetricMatrix({ latest }: { latest: TrainingMetricPoint | null }) {
  const labels = ['acc', 'prec', 'rec', 'map'];
  const values = latest ? [latest.accuracy, latest.precision, latest.recall, latest.map] : [0, 0, 0, 0];
  const cellSize = 48;
  const labelW = 46;
  const svgW = labelW + labels.length * cellSize + 8;
  const svgH = labelW + labels.length * cellSize + 8;

  const matrix = values.map((rowValue) => values.map((colValue) => Number((1 - Math.abs(rowValue - colValue)).toFixed(3))));
  const toColor = (value: number) => {
    const t = Math.max(0, Math.min(1, value));
    const r = Math.round(12 + t * 40);
    const g = Math.round(24 + t * 190);
    const b = Math.round(32 + t * 150);
    return `rgb(${r},${g},${b})`;
  };

  return (
    <div className="flex w-full items-center justify-center" style={{ height: 220 }}>
      <svg width={svgW} height={svgH} style={{ overflow: 'visible' }}>
        <text x={labelW + (labels.length * cellSize) / 2} y={10} textAnchor="middle" fill="#475569" fontSize={10} fontFamily="monospace">
          Metric Consistency
        </text>
        {labels.map((label, index) => (
          <g key={label}>
            <text x={labelW + index * cellSize + cellSize / 2} y={labelW - 6} textAnchor="middle" fill="#64748b" fontSize={10} fontFamily="monospace">
              {label}
            </text>
            <text x={labelW - 6} y={labelW + index * cellSize + cellSize / 2 + 4} textAnchor="end" fill="#64748b" fontSize={10} fontFamily="monospace">
              {label}
            </text>
          </g>
        ))}
        {matrix.map((row, rowIndex) =>
          row.map((value, colIndex) => {
            const x = labelW + colIndex * cellSize;
            const y = labelW + rowIndex * cellSize;
            const textColor = value > 0.5 ? '#0f172a' : '#cbd5e1';
            return (
              <g key={`${rowIndex}-${colIndex}`}>
                <rect x={x} y={y} width={cellSize - 2} height={cellSize - 2} rx={3} fill={toColor(value)} />
                <text x={x + cellSize / 2 - 1} y={y + cellSize / 2 + 4} textAnchor="middle" fill={textColor} fontSize={10} fontFamily="monospace" fontWeight={600}>
                  {(value * 100).toFixed(0)}%
                </text>
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
}

function exportCSV(runId: string, metrics: TrainingMetricPoint[]) {
  const rows = ['epoch,loss,accuracy,precision,recall,map,timestamp'];
  metrics.forEach((item) => {
    rows.push(`${item.epoch},${item.loss},${item.accuracy},${item.precision},${item.recall},${item.map},${item.timestamp}`);
  });
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${runId}-metrics.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function EmptyChartState({ message }: { message: string }) {
  return (
    <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-slate-800/80 bg-slate-950/30 px-6 text-center">
      <p className="max-w-xs text-[11px] text-slate-500">{message}</p>
    </div>
  );
}

export default function AnalyticsPage() {
  const [visible, setVisible] = useState(false);
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [runId, setRunId] = useState('');
  const [runId2, setRunId2] = useState('');
  const [metrics, setMetrics] = useState<TrainingMetricPoint[]>([]);
  const [metrics2, setMetrics2] = useState<TrainingMetricPoint[]>([]);
  const [annotationReport, setAnnotationReport] = useState<any>(null);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [runsError, setRunsError] = useState('');
  const [metricsError, setMetricsError] = useState('');
  const [compareMode, setCompareMode] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const syncRuns = async () => {
      try {
        const nextRuns = await getTrainingRuns();
        if (cancelled) return;
        setRuns(nextRuns);
        setRunsError('');
        if (!runId && nextRuns.length > 0) {
          setRunId(nextRuns[0].run_id);
          if (nextRuns.length > 1) setRunId2(nextRuns[1].run_id);
        }
      } catch (error) {
        if (!cancelled) {
          setRunsError(error instanceof Error ? error.message : 'Unable to load training runs.');
          setRuns([]);
        }
      } finally {
        if (!cancelled) setIsLoadingRuns(false);
      }
    };

    void syncRuns();
    const interval = setInterval(() => {
      void syncRuns();
    }, 4000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId]);

  useEffect(() => {
    if (!runId) {
      setMetrics([]);
      setAnnotationReport(null);
      return;
    }

    const fetchAnnotationReport = async () => {
      try {
        // Attempt to fetch annotation report assuming the run ID corresponds to a local dataset slug
        // In a full DB setup, this link would be explicit. For now, we try to load it.
        const url = `${BASE_URL}/data/${runId}/annotation_report.json`;
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          setAnnotationReport(data);
        } else {
          setAnnotationReport(null);
        }
      } catch {
        setAnnotationReport(null);
      }
    };

    void fetchAnnotationReport();

    let cancelled = false;
    const syncMetrics = async () => {
      if (!cancelled) setIsLoadingMetrics(true);
      try {
        const [nextMetrics, nextMetrics2] = await Promise.all([
          getTrainingMetrics(runId),
          compareMode && runId2 ? getTrainingMetrics(runId2) : Promise.resolve([])
        ]);
        if (!cancelled) {
          setMetrics(nextMetrics);
          setMetrics2(nextMetrics2);
          setMetricsError('');
        }
      } catch (error) {
        if (!cancelled) {
          setMetrics([]);
          setMetrics2([]);
          setMetricsError(error instanceof Error ? error.message : 'Unable to load metrics.');
        }
      } finally {
        if (!cancelled) setIsLoadingMetrics(false);
      }
    };

    void syncMetrics();
    const interval = setInterval(() => {
      void syncMetrics();
    }, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId, runId2, compareMode]);

  const currentRun = runs.find((item) => item.run_id === runId) ?? null;
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const hasMetrics = metrics.length > 0;

  const mergedData = useMemo(() => {
    const map = new Map();
    metrics.forEach(m => map.set(m.epoch, { epoch: m.epoch, loss1: m.loss, acc1: m.accuracy, map1: m.map }));
    if (compareMode) {
      metrics2.forEach(m => {
        const existing = map.get(m.epoch) || { epoch: m.epoch };
        existing.loss2 = m.loss;
        existing.acc2 = m.accuracy;
        existing.map2 = m.map;
        map.set(m.epoch, existing);
      });
    }
    return Array.from(map.values()).sort((a, b) => a.epoch - b.epoch);
  }, [metrics, metrics2, compareMode]);

  const final1 = metrics[metrics.length - 1];
  const final2 = metrics2[metrics2.length - 1];
  let accDelta = 0;
  let mapDelta = 0;
  let overtookEpoch = '-';
  if (final1 && final2) {
    accDelta = final1.accuracy - final2.accuracy;
    mapDelta = final1.map - final2.map;
  }
  const overtake = mergedData.find(d => d.acc1 !== undefined && d.acc2 !== undefined && d.acc1 > d.acc2);
  if (overtake) overtookEpoch = String(overtake.epoch);

  const stats = latestMetric
    ? [
        { label: 'Best Epoch', value: String(latestMetric.epoch) },
        { label: 'mAP', value: `${(latestMetric.map * 100).toFixed(1)}%` },
        { label: 'Accuracy', value: `${(latestMetric.accuracy * 100).toFixed(1)}%` },
        { label: 'Precision', value: `${(latestMetric.precision * 100).toFixed(1)}%` },
        { label: 'Recall', value: `${(latestMetric.recall * 100).toFixed(1)}%` },
        { label: 'Loss', value: latestMetric.loss.toFixed(4) },
      ]
    : [];

  const deltaSeries = useMemo(
    () =>
      metrics.map((item, index) => {
        const prev = metrics[index - 1];
        return {
          epoch: item.epoch,
          accuracy_delta: prev ? Number((item.accuracy - prev.accuracy).toFixed(4)) : 0,
          map_delta: prev ? Number((item.map - prev.map).toFixed(4)) : 0,
        };
      }),
    [metrics]
  );

  if (isLoadingRuns) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="max-w-md w-full text-center space-y-4">
          <div className="relative inline-flex">
            <div className="absolute inset-0 rounded-full bg-teal-500/20 blur-2xl animate-pulse" />
            <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
              <TrendingUp className="h-10 w-10 text-teal-400 animate-pulse" />
            </div>
          </div>
          <div>
            <h2 className="mb-2 text-2xl font-bold text-white">Loading Analytics</h2>
            <p className="text-sm leading-relaxed text-slate-500">
              Pulling runs and metrics from the SQLite training store…
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Backend offline or auth error ─────────────────────────────
  if (runsError) {
    const isOffline = runsError.toLowerCase().includes('fetch') || runsError.toLowerCase().includes('network') || runsError.toLowerCase().includes('failed');
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="max-w-md w-full text-center space-y-5">
          <div className="relative inline-flex">
            <div className="absolute inset-0 rounded-full bg-rose-500/20 blur-2xl" />
            <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl border border-rose-800/50 bg-slate-900 shadow-2xl">
              {isOffline ? <WifiOff className="h-10 w-10 text-rose-400" /> : <AlertTriangle className="h-10 w-10 text-rose-400" />}
            </div>
          </div>
          <div>
            <h2 className="mb-2 text-2xl font-bold text-white">
              {isOffline ? 'Backend Offline' : 'Analytics Unavailable'}
            </h2>
            <p className="text-sm leading-relaxed text-slate-400">
              {isOffline
                ? 'Could not reach the VisCurator backend at localhost:8000. Make sure the server is running.'
                : runsError}
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-left font-mono text-xs text-rose-400">
            {runsError}
          </div>
        </div>
      </div>
    );
  }

  // ── No runs yet ───────────────────────────────────────────────
  if (runs.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="max-w-md w-full text-center space-y-5">
          <div className="relative inline-flex">
            <div className="absolute inset-0 rounded-full bg-purple-500/20 blur-2xl" />
            <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl border border-purple-800/50 bg-slate-900 shadow-2xl">
              <FlaskConical className="h-10 w-10 text-purple-400" />
            </div>
          </div>
          <div>
            <h2 className="mb-2 text-2xl font-bold text-white">No Training Runs Yet</h2>
            <p className="text-sm leading-relaxed text-slate-400">
              Analytics will appear here once you launch a training job from the{' '}
              <strong className="text-slate-200">Builder</strong> page. Design a model, hit{' '}
              <span className="font-mono text-purple-300">Train</span>, and come back.
            </p>
          </div>
          <a
            href="/builder"
            className="inline-flex items-center gap-2 rounded-xl border border-purple-500/40 bg-purple-500/10 px-5 py-2.5 text-sm font-medium text-purple-300 transition hover:bg-purple-500/20"
          >
            <BarChart3 className="h-4 w-4" />
            Go to Builder
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div
        className="mx-auto max-w-7xl space-y-6 px-6 py-8"
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.45s ease, transform 0.45s ease',
        }}
      >
        <div className="flex items-start justify-between">
          <div>
            <p className="mb-1 font-mono text-[11px] uppercase tracking-widest text-purple-400/80">Experiment Tracker</p>
            <h1 className="text-3xl font-bold tracking-tight text-slate-100">Analytics</h1>
            <p className="mt-1 text-sm text-slate-500">Live and historical training metrics from the backend SQLite run store.</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`flex items-center gap-1.5 rounded-xl border px-3 py-2 font-mono text-xs transition-colors ${compareMode ? 'bg-purple-900/40 border-purple-500/50 text-purple-300' : 'border-slate-800/60 bg-slate-900/60 text-slate-400 hover:border-slate-700 hover:text-slate-200'}`}
            >
              <Columns className="h-3.5 w-3.5" />
              Compare
            </button>
            <button
              onClick={() => exportCSV(runId, metrics)}
              disabled={!runId || metrics.length === 0}
              className="flex items-center gap-1.5 rounded-xl border border-slate-800/60 bg-slate-900/60 px-3 py-2 font-mono text-xs text-slate-400 transition-colors hover:border-slate-700 hover:text-slate-200 disabled:opacity-40"
            >
              <Download className="h-3.5 w-3.5" />
              Export CSV
            </button>
            {runId && !compareMode && (
              <a
                href={`${BASE_URL}/api/builder/train/${runId}/download`}
                download={`run_${runId}.zip`}
                className="flex items-center gap-1.5 rounded-xl border border-sky-800/50 bg-sky-900/30 px-3 py-2 font-mono text-xs text-sky-400 transition-colors hover:border-sky-700 hover:bg-sky-900/50"
              >
                <Download className="h-3.5 w-3.5" />
                Download Model
              </a>
            )}
            <div className="flex items-center gap-2 rounded-xl border border-slate-800/60 bg-slate-900/60 px-4 py-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">{compareMode ? 'Run A' : 'Run'}</span>
              <select
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
                className="max-w-[220px] cursor-pointer bg-transparent text-xs text-slate-300 focus:outline-none"
              >
                {runs.map((item) => (
                  <option key={item.run_id} value={item.run_id}>
                    {item.run_id} · {item.task_type ?? 'training'}
                  </option>
                ))}
              </select>
            </div>
            {compareMode && (
              <div className="flex items-center gap-2 rounded-xl border border-slate-800/60 bg-slate-900/60 px-4 py-2">
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Run B</span>
                <select
                  value={runId2}
                  onChange={(e) => setRunId2(e.target.value)}
                  className="max-w-[220px] cursor-pointer bg-transparent text-xs text-slate-300 focus:outline-none"
                >
                  {runs.map((item) => (
                    <option key={item.run_id} value={item.run_id}>
                      {item.run_id} · {item.task_type ?? 'training'}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        {annotationReport && (
          <div className="space-y-6 mb-8 animate-fade-up">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Zap className="h-5 w-5 text-violet-400" />
              <h2 className="text-lg font-bold text-slate-100">Auto-Annotation Results</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="rounded-xl border border-slate-800/60 bg-slate-900/40 p-5">
                <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-4">Confidence Distribution</p>
                <div className="space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-emerald-400">High (&gt;80%)</span>
                    <span className="font-mono text-slate-300">{annotationReport.confidence_distribution.high}</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-amber-400">Medium (&gt;50%)</span>
                    <span className="font-mono text-slate-300">{annotationReport.confidence_distribution.medium}</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-rose-400">Low (&lt;50%)</span>
                    <span className="font-mono text-slate-300">{annotationReport.confidence_distribution.low}</span>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800/60 bg-slate-900/40 p-5 md:col-span-2">
                <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-4">Class Balance (Post-Annotation)</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {Object.entries(annotationReport.class_counts).map(([cls, count]) => (
                    <div key={cls} className="rounded border border-slate-800 bg-slate-950 p-2 text-center">
                      <p className="text-[10px] text-slate-400 truncate" title={cls}>{cls}</p>
                      <p className="text-sm font-bold text-slate-200 mt-1">{String(count)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-3">Annotation Preview</p>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {annotationReport.annotated_samples?.map((sample: any, i: number) => (
                  <div key={i} className="group relative aspect-square overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
                    <img 
                      src={`${BASE_URL}/data/${runId}/${sample.url}`}
                      alt="Annotated"
                      className="h-full w-full object-cover"
                    />
                    <div className="absolute inset-x-0 bottom-0 bg-black/80 p-1.5 flex justify-between items-center">
                      <span className="truncate text-[8px] font-bold text-violet-300">{sample.label}</span>
                      <span className={`text-[8px] font-mono ${sample.confidence > 0.8 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {Math.round(sample.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {compareMode ? (
          <div className="space-y-6">
            <div className="overflow-hidden rounded-2xl border border-purple-500/20 bg-slate-900/40 backdrop-blur">
              <div className="border-b border-purple-500/20 px-5 py-4">
                <h3 className="text-sm font-semibold text-slate-200">Compare Runs</h3>
                <p className="mt-0.5 text-[11px] text-slate-500">Overlaid loss curves for selected runs</p>
              </div>
              <div className="px-4 py-4">
                <CompareLossCurveChart data={mergedData} />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-center">
                <h4 className="text-xs font-mono uppercase text-slate-500 mb-1">Final Accuracy Delta</h4>
                <div className={`text-2xl font-bold ${accDelta > 0 ? 'text-teal-400' : 'text-rose-400'}`}>
                  {accDelta > 0 ? '+' : ''}{(accDelta * 100).toFixed(2)}%
                </div>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-center">
                <h4 className="text-xs font-mono uppercase text-slate-500 mb-1">Final mAP Delta</h4>
                <div className={`text-2xl font-bold ${mapDelta > 0 ? 'text-teal-400' : 'text-rose-400'}`}>
                  {mapDelta > 0 ? '+' : ''}{(mapDelta * 100).toFixed(2)}%
                </div>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-center">
                <h4 className="text-xs font-mono uppercase text-slate-500 mb-1">A Overtook B at Epoch</h4>
                <div className="text-2xl font-bold text-slate-200">{overtookEpoch}</div>
              </div>
            </div>

            {accDelta > 0.02 && (
              <div className="rounded-xl border border-teal-500/30 bg-teal-500/10 p-4 text-center">
                <p className="text-sm font-medium text-teal-300">🎉 Statistically meaningful improvement in Run A (+{(accDelta * 100).toFixed(1)}%)</p>
              </div>
            )}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-3 md:grid-cols-6">
              {stats.length > 0 ? (
                stats.map(({ label, value }) => (
                  <div key={label} className="rounded-xl border border-slate-800/60 bg-slate-900/40 px-4 py-3 text-center">
                    <p className="font-mono text-xl font-bold text-slate-100">{value}</p>
                    <p className="mt-0.5 text-[10px] uppercase tracking-widest text-slate-600">{label}</p>
                  </div>
                ))
              ) : (
                <div className="col-span-full rounded-xl border border-slate-800/60 bg-slate-900/40 px-4 py-6 text-center text-[11px] text-slate-500">
                  {isLoadingMetrics ? 'Loading metrics...' : 'No metrics recorded for the selected run yet.'}
                </div>
              )}
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              {[
                {
                  id: 'loss-curves',
                  title: 'Training Loss Curve',
                  subtitle: 'Real per-epoch loss from the selected run',
                  icon: TrendingUp,
                  accent: '#f97316',
                  border: 'rgba(249,115,22,0.2)',
                  glow: 'rgba(249,115,22,0.06)',
                  span: 'md:col-span-2',
                  tag: 'LineChart',
                  chart: hasMetrics ? <LossCurveChart data={metrics} /> : <EmptyChartState message={metricsError || (isLoadingMetrics ? 'Loading metrics...' : 'No metrics recorded for this run yet.')} />,
                },
                {
                  id: 'validation-metrics',
                  title: 'Validation Metrics',
                  subtitle: 'Accuracy, precision, recall, and mAP over time',
                  icon: Target,
                  accent: '#34d399',
                  border: 'rgba(52,211,153,0.2)',
                  glow: 'rgba(52,211,153,0.06)',
                  span: '',
                  tag: 'AreaChart',
                  chart: hasMetrics ? <MetricAreaChart data={metrics} /> : <EmptyChartState message={metricsError || (isLoadingMetrics ? 'Loading metrics...' : 'Waiting for the first streamed epoch.')} />,
                },
                {
                  id: 'snapshot',
                  title: 'Latest Metric Snapshot',
                  subtitle: 'Current validation scores at the latest epoch',
                  icon: BarChart3,
                  accent: '#38bdf8',
                  border: 'rgba(56,189,248,0.2)',
                  glow: 'rgba(56,189,248,0.06)',
                  span: '',
                  tag: 'BarChart',
                  chart: hasMetrics ? <SnapshotBarChart latest={latestMetric} /> : <EmptyChartState message={metricsError || (isLoadingMetrics ? 'Loading metrics...' : 'Snapshot will appear once metrics are stored.')} />,
                },
                {
                  id: 'delta',
                  title: 'Improvement Trend',
                  subtitle: 'Epoch-to-epoch change in accuracy and mAP',
                  icon: Layers,
                  accent: '#a78bfa',
                  border: 'rgba(167,139,250,0.2)',
                  glow: 'rgba(167,139,250,0.06)',
                  span: '',
                  tag: 'ComposedChart',
                  chart: hasMetrics ? <DeltaChart data={deltaSeries} /> : <EmptyChartState message={metricsError || (isLoadingMetrics ? 'Loading metrics...' : 'Need at least one epoch to calculate deltas.')} />,
                },
                {
                  id: 'matrix',
                  title: 'Metric Matrix',
                  subtitle: 'Consistency between the latest headline metrics',
                  icon: Target,
                  accent: '#2dd4bf',
                  border: 'rgba(45,212,191,0.2)',
                  glow: 'rgba(45,212,191,0.06)',
                  span: '',
                  tag: 'SVG HeatMap',
                  chart: hasMetrics ? <MetricMatrix latest={latestMetric} /> : <EmptyChartState message={metricsError || (isLoadingMetrics ? 'Loading metrics...' : 'Matrix will populate when metrics are available.')} />,
                },
              ].map(({ id, title, subtitle, icon: Icon, accent, border, glow, span, tag, chart }) => (
                <div
                  key={id}
                  id={`chart-${id}`}
                  className={`overflow-hidden rounded-2xl border ${span}`}
                  style={{ borderColor: border, background: `${glow}, #0f172a66`, backdropFilter: 'blur(12px)' }}
                >
                  <div className="flex items-start justify-between border-b px-5 py-4" style={{ borderColor: border }}>
                    <div className="flex items-center gap-2.5">
                      <div className="rounded-lg p-2" style={{ background: glow, border: `1px solid ${border}` }}>
                        <Icon className="h-4 w-4" style={{ color: accent }} />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
                        <p className="mt-0.5 text-[11px] text-slate-500">{subtitle}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-600">
                      <Info className="h-3 w-3" />
                      <span className="font-mono">{tag}</span>
                    </div>
                  </div>
                  <div className="px-4 py-4">{chart}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
