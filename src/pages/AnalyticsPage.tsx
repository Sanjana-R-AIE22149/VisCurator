import { useEffect, useState, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';
import { BarChart3, TrendingUp, Target, Layers, Info, Download, Plus } from 'lucide-react';

// ─── Constants ───────────────────────────────────────────────────────────────

const CLASSES = ['cat', 'dog', 'car', 'person', 'bicycle'];
const CLASS_COLORS = ['#2dd4bf', '#a78bfa', '#f59e0b', '#38bdf8', '#34d399'];

const GRID_STROKE = '#1e293b';
const TICK_STYLE = { fill: '#64748b', fontSize: 11 };
const TOOLTIP_STYLE = {
  contentStyle: { background: '#020617', border: '1px solid #334155', borderRadius: 8, fontFamily: 'monospace', fontSize: 11 },
  labelStyle: { color: '#94a3b8' },
  itemStyle: { color: '#cbd5e1' },
};

// ─── Mock data generators ─────────────────────────────────────────────────────

type RunConfig = { epochs: number; bestEpoch: number; finalMap50: number; finalMap5095: number; noise: number };

const RUN_CONFIGS: Record<string, RunConfig> = {
  'run-003': { epochs: 150, bestEpoch: 127, finalMap50: 0.847, finalMap5095: 0.623, noise: 0.008 },
  'run-002': { epochs: 120, bestEpoch: 98,  finalMap50: 0.791, finalMap5095: 0.571, noise: 0.012 },
  'run-001': { epochs: 100, bestEpoch: 81,  finalMap50: 0.734, finalMap5095: 0.512, noise: 0.018 },
};

const RUN_STATS: Record<string, typeof STAT_STRIP_BASE> = {
  'run-003': [
    { label: 'Best Epoch', value: '127' }, { label: 'mAP@50', value: '84.7%' },
    { label: 'mAP@50-95', value: '62.3%' }, { label: 'Precision', value: '91.2%' },
    { label: 'Recall', value: '87.8%' }, { label: 'F1 Score', value: '0.894' },
  ],
  'run-002': [
    { label: 'Best Epoch', value: '98' }, { label: 'mAP@50', value: '79.1%' },
    { label: 'mAP@50-95', value: '57.1%' }, { label: 'Precision', value: '86.4%' },
    { label: 'Recall', value: '83.2%' }, { label: 'F1 Score', value: '0.848' },
  ],
  'run-001': [
    { label: 'Best Epoch', value: '81' }, { label: 'mAP@50', value: '73.4%' },
    { label: 'mAP@50-95', value: '51.2%' }, { label: 'Precision', value: '80.1%' },
    { label: 'Recall', value: '77.5%' }, { label: 'F1 Score', value: '0.788' },
  ],
};

const STAT_STRIP_BASE = [
  { label: 'Best Epoch', value: '127' }, { label: 'mAP@50', value: '84.7%' },
  { label: 'mAP@50-95', value: '62.3%' }, { label: 'Precision', value: '91.2%' },
  { label: 'Recall', value: '87.8%' }, { label: 'F1 Score', value: '0.894' },
];

function seededRand(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s - 1) / 2147483646; };
}

function genLossCurves(cfg: RunConfig) {
  const rand = seededRand(42);
  return Array.from({ length: cfg.epochs }, (_, i) => {
    const e = i + 1;
    const decay = (start: number, end: number, k: number) =>
      end + (start - end) * Math.exp(-k * e) + (rand() - 0.5) * cfg.noise * 2;
    return {
      epoch: e,
      boxLoss: +decay(3.2, 0.04, 0.035).toFixed(4),
      objLoss: +decay(2.8, 0.03, 0.030).toFixed(4),
      clsLoss: +decay(1.6, 0.01, 0.040).toFixed(4),
    };
  });
}

function genMapCurves(cfg: RunConfig) {
  const rand = seededRand(99);
  return Array.from({ length: cfg.epochs }, (_, i) => {
    const e = i + 1;
    const t = e / cfg.epochs;
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
    const map50 = +(cfg.finalMap50 * sigmoid(8 * t - 2) + (rand() - 0.5) * cfg.noise * 3).toFixed(4);
    const map5095 = +(cfg.finalMap5095 * sigmoid(8 * t - 2.5) + (rand() - 0.5) * cfg.noise * 3).toFixed(4);
    return { epoch: e, map50: Math.max(0, map50), map5095: Math.max(0, map5095) };
  });
}

function genClassDist(cfg: RunConfig) {
  const rand = seededRand(7);
  const scale = cfg.finalMap50 / 0.847;
  return CLASSES.map((cls) => ({
    cls,
    train: Math.round((800 + rand() * 400) * scale),
    val:   Math.round((100 + rand() * 60)  * scale),
    test:  Math.round((80  + rand() * 50)  * scale),
  }));
}

function genPRCurves(cfg: RunConfig) {
  const rand = seededRand(13);
  const scale = cfg.finalMap50 / 0.847;
  return Array.from({ length: 21 }, (_, i) => {
    const recall = +(i / 20).toFixed(2);
    const row: Record<string, number> = { recall };
    CLASSES.forEach((cls, ci) => {
      const base = (0.97 - recall * 0.55) * (0.88 + ci * 0.02) * scale;
      row[cls] = +Math.max(0, Math.min(1, base + (rand() - 0.5) * 0.04)).toFixed(4);
    });
    return row;
  });
}

function genConfusionMatrix(cfg: RunConfig) {
  const rand = seededRand(55);
  const scale = cfg.finalMap50 / 0.847;
  return CLASSES.map((actual, ai) =>
    CLASSES.map((_, pi) => {
      if (ai === pi) return +Math.min(1, (0.82 + rand() * 0.15) * scale).toFixed(3);
      return +Math.max(0, (rand() * 0.08) * (1 - scale * 0.5)).toFixed(3);
    })
  );
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label, unit = '' }: {
  active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string | number; unit?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#020617', border: '1px solid #334155', borderRadius: 8, padding: '8px 12px', fontFamily: 'monospace', fontSize: 11 }}>
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>{unit}{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color, margin: '2px 0' }}>
          {p.name}: <span style={{ color: '#e2e8f0' }}>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</span>
        </p>
      ))}
    </div>
  );
}

// ─── Chart components ─────────────────────────────────────────────────────────

function LossCurveChart({ data, bestEpoch }: { data: ReturnType<typeof genLossCurves>; bestEpoch: number }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={TICK_STYLE} interval={24} label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#475569', fontSize: 10 }} />
        <YAxis tick={TICK_STYLE} domain={[0, 'auto']} />
        <Tooltip content={<CustomTooltip unit="Epoch " />} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }} />
        <ReferenceLine x={bestEpoch} stroke="#64748b" strokeDasharray="4 2" label={{ value: `best`, position: 'top', fill: '#64748b', fontSize: 10 }} />
        <Line type="monotone" dataKey="boxLoss" name="Box Loss" stroke="#2dd4bf" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="objLoss" name="Obj Loss" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="clsLoss" name="Cls Loss" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function MapChart({ data }: { data: ReturnType<typeof genMapCurves> }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <defs>
          <linearGradient id="gMap50" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gMap5095" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={TICK_STYLE} interval={24} />
        <YAxis tick={TICK_STYLE} domain={[0, 1]} />
        <Tooltip content={<CustomTooltip unit="Epoch " />} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }} />
        <Area type="monotone" dataKey="map50" name="mAP@50" stroke="#2dd4bf" fill="url(#gMap50)" strokeWidth={2} dot={false} />
        <Area type="monotone" dataKey="map5095" name="mAP@50-95" stroke="#a78bfa" fill="url(#gMap5095)" strokeWidth={2} strokeDasharray="5 3" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function ClassDistChart({ data }: { data: ReturnType<typeof genClassDist> }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="cls" tick={TICK_STYLE} />
        <YAxis tick={TICK_STYLE} />
        <Tooltip {...TOOLTIP_STYLE} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }} />
        <Bar dataKey="train" name="Train" fill="#2dd4bf" radius={[3, 3, 0, 0]} maxBarSize={18} />
        <Bar dataKey="val"   name="Val"   fill="#38bdf8" radius={[3, 3, 0, 0]} maxBarSize={18} />
        <Bar dataKey="test"  name="Test"  fill="#f59e0b" radius={[3, 3, 0, 0]} maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function PRChart({ data }: { data: ReturnType<typeof genPRCurves> }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
        <XAxis dataKey="recall" tick={TICK_STYLE} domain={[0, 1]} label={{ value: 'Recall', position: 'insideBottomRight', offset: -4, fill: '#475569', fontSize: 10 }} />
        <YAxis tick={TICK_STYLE} domain={[0, 1]} label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 14, fill: '#475569', fontSize: 10 }} />
        <Tooltip content={<CustomTooltip unit="Recall " />} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }} />
        {CLASSES.map((cls, i) => (
          <Line key={cls} type="monotone" dataKey={cls} stroke={CLASS_COLORS[i]} dot={false} strokeWidth={1.5} />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const cellSize = 52;
  const labelW = 52;
  const pad = 8;
  const total = CLASSES.length;
  const svgW = labelW + total * cellSize + pad;
  const svgH = labelW + total * cellSize + pad;

  const toColor = (v: number) => {
    const t = Math.pow(Math.max(0, Math.min(1, v)), 0.6);
    const r = Math.round(2 + t * (45 - 2));
    const g = Math.round(12 + t * (212 - 12));
    const b = Math.round(23 + t * (191 - 23));
    return `rgb(${r},${g},${b})`;
  };

  return (
    <div className="flex items-center justify-center w-full" style={{ height: 220 }}>
      <svg width={svgW} height={svgH} style={{ overflow: 'visible' }}>
        {/* Axis labels */}
        <text x={labelW + (total * cellSize) / 2} y={10} textAnchor="middle" fill="#475569" fontSize={10} fontFamily="monospace">Predicted</text>
        <text x={10} y={labelW + (total * cellSize) / 2} textAnchor="middle" fill="#475569" fontSize={10} fontFamily="monospace" transform={`rotate(-90, 10, ${labelW + (total * cellSize) / 2})`}>Actual</text>
        {CLASSES.map((cls, i) => (
          <g key={cls}>
            <text x={labelW + i * cellSize + cellSize / 2} y={labelW - 6} textAnchor="middle" fill="#64748b" fontSize={10} fontFamily="monospace">{cls}</text>
            <text x={labelW - 6} y={labelW + i * cellSize + cellSize / 2 + 4} textAnchor="end" fill="#64748b" fontSize={10} fontFamily="monospace">{cls}</text>
          </g>
        ))}
        {matrix.map((row, ai) =>
          row.map((val, pi) => {
            const x = labelW + pi * cellSize;
            const y = labelW + ai * cellSize;
            const textColor = val > 0.5 ? '#0f172a' : '#94a3b8';
            return (
              <g key={`${ai}-${pi}`}>
                <rect x={x} y={y} width={cellSize - 2} height={cellSize - 2} rx={3} fill={toColor(val)} />
                <text x={x + cellSize / 2 - 1} y={y + cellSize / 2 + 4} textAnchor="middle" fill={textColor} fontSize={10} fontFamily="monospace" fontWeight={600}>
                  {(val * 100).toFixed(0)}%
                </text>
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
}

// ─── CSV export ───────────────────────────────────────────────────────────────

function exportCSV(run: string, lossCurves: ReturnType<typeof genLossCurves>, mapCurves: ReturnType<typeof genMapCurves>) {
  const rows = ['epoch,boxLoss,objLoss,clsLoss,map50,map5095'];
  lossCurves.forEach((l, i) => {
    const m = mapCurves[i];
    rows.push(`${l.epoch},${l.boxLoss},${l.objLoss},${l.clsLoss},${m.map50},${m.map5095}`);
  });
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${run}-metrics.csv`;
  a.click();
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [visible, setVisible] = useState(false);
  const [run, setRun] = useState('run-003');

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, []);

  const cfg = RUN_CONFIGS[run];

  const lossCurves    = useMemo(() => genLossCurves(cfg),    [cfg]);
  const mapCurves     = useMemo(() => genMapCurves(cfg),     [cfg]);
  const classDist     = useMemo(() => genClassDist(cfg),     [cfg]);
  const prCurves      = useMemo(() => genPRCurves(cfg),      [cfg]);
  const confMatrix    = useMemo(() => genConfusionMatrix(cfg),[cfg]);
  const stats         = RUN_STATS[run];

  if (Object.keys(RUN_CONFIGS).length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6 animate-fade-up">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="relative inline-flex">
            <div className="absolute inset-0 bg-purple-500/20 blur-2xl rounded-full animate-pulse" />
            <div className="relative w-20 h-20 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center shadow-2xl">
              <TrendingUp className="w-10 h-10 text-purple-400" />
            </div>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">No Training Runs Yet</h2>
            <p className="text-sm text-slate-500 leading-relaxed">
              Your experiment metrics will appear here once you initialize your first model training pipeline.
            </p>
          </div>
          <button className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold transition-all shadow-lg shadow-purple-900/20 active:scale-95">
            <Plus className="w-5 h-5" />
            Initialize New Run
          </button>
        </div>
      </div>
    );
  }

  const CHART_SECTIONS = [
    {
      id: 'loss-curves', title: 'Training Loss Curves', subtitle: 'Box, Object, Class loss vs. Epoch',
      icon: TrendingUp, accent: '#2dd4bf', border: 'rgba(45,212,191,0.2)', glow: 'rgba(45,212,191,0.06)',
      span: 'md:col-span-2', tag: 'LineChart',
      chart: <LossCurveChart data={lossCurves} bestEpoch={cfg.bestEpoch} />,
    },
    {
      id: 'val-map', title: 'Validation mAP', subtitle: 'mAP@50 & mAP@50-95 over time',
      icon: Target, accent: '#a78bfa', border: 'rgba(167,139,250,0.2)', glow: 'rgba(167,139,250,0.06)',
      span: '', tag: 'AreaChart',
      chart: <MapChart data={mapCurves} />,
    },
    {
      id: 'class-dist', title: 'Class Distribution', subtitle: 'Label balance across dataset splits',
      icon: BarChart3, accent: '#38bdf8', border: 'rgba(56,189,248,0.2)', glow: 'rgba(56,189,248,0.06)',
      span: '', tag: 'BarChart',
      chart: <ClassDistChart data={classDist} />,
    },
    {
      id: 'precision-recall', title: 'Precision–Recall Curve', subtitle: 'Per-class P–R at threshold 0.5',
      icon: Layers, accent: '#f59e0b', border: 'rgba(245,158,11,0.2)', glow: 'rgba(245,158,11,0.06)',
      span: '', tag: 'ComposedChart',
      chart: <PRChart data={prCurves} />,
    },
    {
      id: 'confusion-matrix', title: 'Confusion Matrix', subtitle: 'Predicted vs. Ground truth (normalized)',
      icon: Target, accent: '#34d399', border: 'rgba(52,211,153,0.2)', glow: 'rgba(52,211,153,0.06)',
      span: '', tag: 'SVG HeatMap',
      chart: <ConfusionMatrix matrix={confMatrix} />,
    },
  ];

  return (
    <div className="h-full overflow-y-auto">
      <div
        className="max-w-7xl mx-auto px-6 py-8 space-y-6"
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.45s ease, transform 0.45s ease',
        }}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[11px] font-mono text-purple-400/80 uppercase tracking-widest mb-1">Experiment Tracker</p>
            <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Analytics</h1>
            <p className="text-sm text-slate-500 mt-1">Loss curves, mAP scores, and per-class metrics across training runs.</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => exportCSV(run, lossCurves, mapCurves)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800/60 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition-colors text-xs font-mono"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
            <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800/60">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">Run</span>
              <select
                value={run}
                onChange={(e) => setRun(e.target.value)}
                className="bg-transparent text-slate-300 text-xs focus:outline-none cursor-pointer"
              >
                <option value="run-003">run-003 · YOLOv8n</option>
                <option value="run-002">run-002 · ResNet50</option>
                <option value="run-001">run-001 · EfficientDet</option>
              </select>
            </div>
          </div>
        </div>

        {/* Stat strip */}
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {stats.map(({ label, value }) => (
            <div key={label} className="rounded-xl border border-slate-800/60 bg-slate-900/40 px-4 py-3 text-center">
              <p className="text-xl font-bold font-mono text-slate-100">{value}</p>
              <p className="text-[10px] uppercase tracking-widest text-slate-600 mt-0.5">{label}</p>
            </div>
          ))}
        </div>

        {/* Chart Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {CHART_SECTIONS.map(({ id, title, subtitle, icon: Icon, accent, border, glow, span, tag, chart }) => (
            <div
              key={id}
              id={`chart-${id}`}
              className={`rounded-2xl border overflow-hidden ${span}`}
              style={{ borderColor: border, background: `${glow}, #0f172a66`, backdropFilter: 'blur(12px)' }}
            >
              <div className="flex items-start justify-between px-5 py-4 border-b" style={{ borderColor: border }}>
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg" style={{ background: glow, border: `1px solid ${border}` }}>
                    <Icon className="w-4 h-4" style={{ color: accent }} />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
                    <p className="text-[11px] text-slate-500 mt-0.5">{subtitle}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-slate-600">
                  <Info className="w-3 h-3" />
                  <span className="font-mono">{tag}</span>
                </div>
              </div>
              <div className="px-4 py-4">{chart}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
