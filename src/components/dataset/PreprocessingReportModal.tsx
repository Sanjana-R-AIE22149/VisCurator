import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, ReferenceLine,
} from 'recharts';
import { X, FileText, Loader2, AlertTriangle } from 'lucide-react';
import { BASE_URL, authHeaders } from '../../lib/api';
import { useAppStore } from '../../store/useAppStore';

interface BlurPoint {
  id: number;
  laplacian: number;
  resolution: number;
  accepted: boolean;
}

interface PreprocessingReport {
  dataset_id?: string;
  output_dir?: string;
  before_stats?: { images?: number; class_distribution?: Record<string, number> };
  after_stats?: {
    images?: number;
    class_distribution?: Record<string, number>;
    blur_filtered?: number;
    duplicates_removed?: number;
  };
  class_distribution?: Record<string, number>;
  augmentation_summary?: {
    operations?: string[];
    augmented_images?: number;
    synthetic_generation_recommended?: boolean;
  };
  blur_scatter?: BlurPoint[];
}

const GRID  = '#1e293b';
const TICK  = { fill: '#64748b', fontSize: 10 };
const TT    = {
  contentStyle: { background: '#020617', border: '1px solid #334155', borderRadius: 6, fontFamily: 'monospace', fontSize: 11 },
  labelStyle: { color: '#94a3b8' },
};

function ClassDistChart({ dist }: { dist: Record<string, number> }) {
  const data = Object.entries(dist)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([name, count]) => ({ name, count }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 40 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={{ ...TICK, fontSize: 9 }} angle={-35} textAnchor="end" interval={0} />
        <YAxis tick={TICK} />
        <Tooltip {...TT} />
        <Bar dataKey="count" fill="#2dd4bf" radius={[3, 3, 0, 0]} maxBarSize={32} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function BlurScatterPlot({ points }: { points: BlurPoint[] }) {
  const accepted = points.filter((p) => p.accepted);
  const rejected = points.filter((p) => !p.accepted);
  return (
    <ResponsiveContainer width="100%" height={200}>
      <ScatterChart margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="id" name="Image index" type="number" tick={TICK}
          label={{ value: 'Image index', position: 'insideBottom', offset: -2, style: { fontSize: 9, fill: '#475569' } }} />
        <YAxis dataKey="laplacian" name="Laplacian σ²" type="number" tick={TICK}
          label={{ value: 'Laplacian σ²', angle: -90, position: 'insideLeft', offset: 18, style: { fontSize: 9, fill: '#475569' } }} />
        <Tooltip
          cursor={{ strokeDasharray: '3 3', stroke: '#475569' }}
          contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11, fontFamily: 'monospace' }}
          formatter={(val: any, name: any) => [Number(val).toFixed(1), name]}
        />
        <ReferenceLine y={80} stroke="#f59e0b" strokeDasharray="4 2" strokeOpacity={0.7}
          label={{ value: 'threshold=80', position: 'right', fill: '#f59e0b', fontSize: 9 }} />
        <Scatter name="Accepted" data={accepted}>
          {accepted.map((_, i) => <Cell key={i} fill="#34d39966" />)}
        </Scatter>
        <Scatter name="Rejected" data={rejected}>
          {rejected.map((_, i) => <Cell key={i} fill="#f8717166" />)}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

interface Props {
  jobId: string;
  onClose: () => void;
}

export default function PreprocessingReportModal({ jobId, onClose }: Props) {
  const { preprocessingReport: storeReport } = useAppStore();
  const [report, setReport]   = useState<PreprocessingReport | null>(storeReport as PreprocessingReport | null);
  const [loading, setLoading] = useState(!storeReport);
  const [error, setError]     = useState('');

  useEffect(() => {
    // If we already have it from the store, show immediately while fetching fresh copy
    if (storeReport) setReport(storeReport as PreprocessingReport);

    let cancelled = false;
    fetch(`${BASE_URL}/api/dataset/jobs/${jobId}/report`, {
      headers: authHeaders(),
      signal: AbortSignal.timeout(8000),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} — ${r.statusText}`);
        return r.json() as Promise<PreprocessingReport>;
      })
      .then((data) => { if (!cancelled) { setReport(data); setLoading(false); } })
      .catch((err) => {
        if (!cancelled) {
          // Don't show error if we already have store data
          if (!storeReport) setError((err as Error).message);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [jobId, storeReport]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const afterDist  = report?.after_stats?.class_distribution ?? report?.class_distribution ?? {};
  const beforeDist = report?.before_stats?.class_distribution ?? {};
  const blurPoints = report?.blur_scatter ?? [];
  const accepted   = blurPoints.filter((p) => p.accepted).length;
  const rejected   = blurPoints.length - accepted;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="flex flex-col w-full max-w-3xl max-h-[90vh] rounded-2xl border border-slate-700/60 bg-slate-950 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-800/60 shrink-0">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-teal-400" />
            <span className="text-sm font-semibold text-slate-200">Preprocessing Report</span>
            {report?.dataset_id && (
              <span className="font-mono text-[10px] text-slate-500">{report.dataset_id}</span>
            )}
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-800 hover:text-slate-200 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {loading && (
            <div className="flex items-center justify-center py-16 gap-2 text-slate-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-xs font-mono">Loading report…</span>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-300">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              {error}
            </div>
          )}

          {report && (
            <>
              {/* Stats row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: 'Before', value: report.before_stats?.images?.toLocaleString() ?? '—', sub: 'images' },
                  { label: 'After',  value: report.after_stats?.images?.toLocaleString()  ?? '—', sub: 'images' },
                  { label: 'Blur filtered',      value: String(report.after_stats?.blur_filtered      ?? 0), sub: 'removed' },
                  { label: 'Duplicates removed', value: String(report.after_stats?.duplicates_removed ?? 0), sub: 'removed' },
                ].map(({ label, value, sub }) => (
                  <div key={label} className="rounded-xl border border-slate-800/50 bg-slate-900/50 px-4 py-3 text-center">
                    <p className="text-[10px] uppercase tracking-widest text-slate-500">{label}</p>
                    <p className="text-xl font-bold font-mono text-slate-100 mt-0.5">{value}</p>
                    <p className="text-[10px] text-slate-600">{sub}</p>
                  </div>
                ))}
              </div>

              {/* Augmentation ops */}
              {(report.augmentation_summary?.operations ?? []).length > 0 && (
                <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 px-4 py-3 space-y-2">
                  <p className="text-[10px] uppercase tracking-widest text-slate-500">Augmentation operations</p>
                  <div className="flex flex-wrap gap-1.5">
                    {report.augmentation_summary!.operations!.map((op) => (
                      <span key={op} className="rounded-full border border-teal-500/25 bg-teal-500/10 px-2.5 py-0.5 text-[10px] font-mono text-teal-400">
                        {op}
                      </span>
                    ))}
                    {report.augmentation_summary?.augmented_images != null && (
                      <span className="rounded-full border border-slate-700 bg-slate-800/60 px-2.5 py-0.5 text-[10px] font-mono text-slate-400">
                        +{report.augmentation_summary.augmented_images} augmented
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Class distribution */}
              {Object.keys(afterDist).length > 0 && (
                <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 px-4 pt-3 pb-2">
                  <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-3">
                    Class distribution (after) — {Object.keys(afterDist).length} classes
                  </p>
                  <ClassDistChart dist={afterDist} />
                  {Object.keys(beforeDist).length > 0 && (
                    <p className="text-[10px] font-mono text-slate-600 mt-1">
                      Before: {Object.values(beforeDist).reduce((a, b) => a + b, 0).toLocaleString()} images across {Object.keys(beforeDist).length} classes
                    </p>
                  )}
                </div>
              )}

              {/* Blur scatter */}
              {blurPoints.length > 0 && (
                <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 px-4 pt-3 pb-2">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-[10px] uppercase tracking-widest text-slate-500">
                      Blur scatter — {blurPoints.length} images
                    </p>
                    <div className="flex items-center gap-3 text-[10px] font-mono">
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400/60 inline-block" />accepted {accepted}</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400/60 inline-block" />rejected {rejected}</span>
                    </div>
                  </div>
                  <BlurScatterPlot points={blurPoints} />
                </div>
              )}

              {/* Output path */}
              {report.output_dir && (
                <p className="text-[10px] font-mono text-slate-600 break-all">
                  Output: {report.output_dir}
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
