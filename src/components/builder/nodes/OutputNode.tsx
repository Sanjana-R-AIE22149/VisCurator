import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { BarChart3 } from 'lucide-react';

function OutputNode({ data }: NodeProps) {
  return (
    <div className="group relative w-56">
      {/* Glow */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-600/20 to-emerald-600/5 rounded-xl blur-sm opacity-60 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative bg-slate-900/90 border border-emerald-600/30 rounded-xl p-4 backdrop-blur-md">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-600/15 border border-emerald-600/25">
            <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-emerald-300 leading-tight">
              {(data as any).label}
            </p>
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
              Output Layer
            </p>
          </div>
        </div>

        {/* Specs */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Classes</span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-600/10 px-1.5 py-0.5 rounded">
              {(data as any).num_classes}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Activation</span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-600/10 px-1.5 py-0.5 rounded">
              {(data as any).activation}
            </span>
          </div>
        </div>

        {/* Class output visual */}
        <div className="mt-3 flex items-end justify-center gap-1.5 h-8 p-1.5 bg-emerald-600/5 rounded-lg border border-emerald-600/10">
          {[0.4, 0.9, 0.3, 0.6, 0.2].map((val, i) => (
            <div
              key={i}
              className="w-2 rounded-t-sm bg-emerald-500/40"
              style={{ height: `${val * 100}%` }}
            />
          ))}
        </div>

        {/* Bottom accent */}
        <div className="absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent via-emerald-600/40 to-transparent" />
      </div>

      {/* Handles - Only target */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-emerald-500 !border-emerald-600/50 !w-2.5 !h-2.5"
      />
    </div>
  );
}

export default memo(OutputNode);
