import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { useAppStore } from '../../store/useAppStore';

export default function BlurFilterChart() {
  const { blurData, isProcessing, isPaused } = useAppStore();

  const accepted = useMemo(() => blurData.filter((d) => d.accepted), [blurData]);
  const rejected = useMemo(() => blurData.filter((d) => !d.accepted), [blurData]);

  const isEmpty = blurData.length === 0;
  const statusText = isProcessing
    ? 'Awaiting quality analysis from agent…'
    : isPaused
    ? 'Agent paused — select a dataset to continue'
    : 'Run the pipeline to populate this chart';

  return (
    <div className="flex flex-col rounded-xl border border-slate-800/60 bg-slate-900/30 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/50">
        <div>
          <h3 className="text-xs font-semibold text-slate-300 tracking-wide">
            Blur Filter Dashboard
          </h3>
          <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
            Laplacian variance vs resolution · threshold = 80
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-[10px] text-slate-500">Accepted ({accepted.length})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-red-400" />
            <span className="text-[10px] text-slate-500">Rejected ({rejected.length})</span>
          </div>
        </div>
      </div>

      <div className="h-64 p-4">
        {isEmpty ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-xs text-slate-600 font-mono">{statusText}</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" strokeOpacity={0.5} />
              <XAxis
                dataKey="resolution"
                name="Resolution (kpx)"
                type="number"
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={{ stroke: '#334155' }}
                label={{ value: 'Resolution (kpx)', position: 'insideBottom', offset: -2, style: { fontSize: 9, fill: '#475569' } }}
              />
              <YAxis
                dataKey="laplacian"
                name="Laplacian σ²"
                type="number"
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={{ stroke: '#334155' }}
                label={{ value: 'Laplacian σ²', angle: -90, position: 'insideLeft', offset: 18, style: { fontSize: 9, fill: '#475569' } }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3', stroke: '#475569' }}
                contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
                formatter={(val, name) => [Number(val ?? 0).toFixed(1), String(name)]}
              />
              <ReferenceLine y={80} stroke="#f59e0b" strokeDasharray="4 2" strokeOpacity={0.6}
                label={{ value: 'threshold=80', position: 'right', fill: '#f59e0b', fontSize: 9 }} />
              <Scatter name="Accepted" data={accepted} fill="#34d399">
                {accepted.map((_, i) => <Cell key={i} fill="#34d39966" />)}
              </Scatter>
              <Scatter name="Rejected" data={rejected} fill="#f87171">
                {rejected.map((_, i) => <Cell key={i} fill="#f8717166" />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>

      {!isEmpty && (
        <div className="px-4 py-2 border-t border-slate-800/30 flex items-center gap-4 text-[10px] font-mono text-slate-600">
          <span>Total: {blurData.length}</span>
          <span className="text-emerald-500">✓ {accepted.length} accepted</span>
          <span className="text-red-500">✗ {rejected.length} rejected</span>
          <span>Rejection rate: {blurData.length > 0 ? ((rejected.length / blurData.length) * 100).toFixed(1) : 0}%</span>
        </div>
      )}
    </div>
  );
}
