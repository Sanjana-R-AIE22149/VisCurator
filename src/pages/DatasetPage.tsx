import { Database, Sparkles, ShieldCheck, BarChart3 } from 'lucide-react';
import DatasetControls from '../components/dataset/DatasetControls';
import LiveTerminal from '../components/dataset/LiveTerminal';
import BlurFilterChart from '../components/dataset/BlurFilterChart';

const stats = [
  { label: 'Images Ingested', value: '—', icon: Database, accent: 'text-teal-400' },
  { label: 'Auto-Annotated', value: '—', icon: Sparkles, accent: 'text-purple-400' },
  { label: 'Quality Score', value: '—', icon: ShieldCheck, accent: 'text-emerald-400' },
  { label: 'Class Balance', value: '—', icon: BarChart3, accent: 'text-amber-400' },
];

export default function DatasetPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fade-up">
        {/* Page Header */}
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px flex-1 bg-gradient-to-r from-teal-500/40 to-transparent" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100">
            Autonomous Data Ingestor
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Configure, execute, and monitor the end-to-end dataset curation pipeline
          </p>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {stats.map(({ label, value, icon: Icon, accent }) => (
            <div
              key={label}
              className="flex items-center gap-3 px-4 py-3 rounded-xl border border-slate-800/50 bg-slate-900/20"
            >
              <div className={`p-2 rounded-lg bg-slate-800/50 ${accent}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-widest text-slate-500">
                  {label}
                </p>
                <p className="text-lg font-bold text-slate-200 font-mono">
                  {value}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Controls */}
        <DatasetControls />

        {/* Terminal + Chart Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <LiveTerminal />
          <BlurFilterChart />
        </div>
      </div>
    </div>
  );
}
