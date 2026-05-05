import { useMemo, useEffect } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from 'recharts';
import { useAppStore } from '../../store/useAppStore';

/* ── Generate mock blur data ── */
function generateBlurData(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    laplacian: Math.random() * 300,
    resolution: 100 + Math.random() * 900,
    accepted: Math.random() * 300 > 80,
  }));
}

export default function BlurFilterChart() {
  const { blurData, setBlurData, isProcessing } = useAppStore();

  /* Generate data when processing starts */
  useEffect(() => {
    if (isProcessing && blurData.length === 0) {
      const timer = setTimeout(() => {
        setBlurData(generateBlurData(220));
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [isProcessing, blurData.length, setBlurData]);

  const accepted = useMemo(
    () => blurData.filter((d) => d.accepted),
    [blurData]
  );
  const rejected = useMemo(
    () => blurData.filter((d) => !d.accepted),
    [blurData]
  );

  return (
    <div className="flex flex-col rounded-xl border border-slate-800/60 bg-slate-900/30 overflow-hidden">
      {/* Chart header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/50">
        <div>
          <h3 className="text-xs font-semibold text-slate-300 tracking-wide">
            Real-time Blur Filtering Dashboard
          </h3>
          <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
            OpenCV Laplacian Variance vs Resolution
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-[10px] text-slate-500">
              Accepted ({accepted.length})
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-red-400" />
            <span className="text-[10px] text-slate-500">
              Rejected ({rejected.length})
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64 p-4">
        {blurData.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-xs text-slate-600 font-mono">
              {isProcessing
                ? 'Awaiting blur analysis data...'
                : 'Run pipeline to populate chart'}
            </span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#1e293b"
                strokeOpacity={0.5}
              />
              <XAxis
                dataKey="resolution"
                name="Resolution"
                type="number"
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={{ stroke: '#334155' }}
                label={{
                  value: 'Resolution (px)',
                  position: 'insideBottom',
                  offset: -2,
                  style: { fontSize: 9, fill: '#475569' },
                }}
              />
              <YAxis
                dataKey="laplacian"
                name="Laplacian"
                type="number"
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={{ stroke: '#334155' }}
                label={{
                  value: 'Laplacian σ²',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 18,
                  style: { fontSize: 9, fill: '#475569' },
                }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3', stroke: '#475569' }}
                contentStyle={{
                  background: '#0f172a',
                  border: '1px solid #1e293b',
                  borderRadius: 8,
                  fontSize: 11,
                  color: '#94a3b8',
                  fontFamily: 'var(--font-mono)',
                }}
                labelStyle={{ display: 'none' }}
              />
              {/* Threshold line at y=80 */}
              <Scatter name="Accepted" data={accepted} fill="#34d399">
                {accepted.map((_, i) => (
                  <Cell key={i} fill="#34d39980" />
                ))}
              </Scatter>
              <Scatter name="Rejected" data={rejected} fill="#f87171">
                {rejected.map((_, i) => (
                  <Cell key={i} fill="#f8717180" />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
