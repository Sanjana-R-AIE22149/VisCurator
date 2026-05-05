import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Shuffle } from 'lucide-react';

function BatchNormNode({ data }: NodeProps) {
  return (
    <div className="group relative w-56">
      {/* Glow */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-slate-500/20 to-slate-500/5 rounded-xl blur-sm opacity-60 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative bg-slate-900/90 border border-slate-500/30 rounded-xl p-4 backdrop-blur-md">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-slate-500/15 border border-slate-500/25">
            <Shuffle className="w-3.5 h-3.5 text-slate-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-300 leading-tight">
              {(data as any).label}
            </p>
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
              Normalization
            </p>
          </div>
        </div>

        {/* Specs */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Num Features</span>
            <span className="text-[10px] font-mono text-slate-400 bg-slate-500/10 px-1.5 py-0.5 rounded">
              {(data as any).num_features}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Eps</span>
            <span className="text-[10px] font-mono text-slate-400 bg-slate-500/10 px-1.5 py-0.5 rounded">
              {(data as any).eps}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Momentum</span>
            <span className="text-[10px] font-mono text-slate-400 bg-slate-500/10 px-1.5 py-0.5 rounded">
              {(data as any).momentum}
            </span>
          </div>
        </div>

        {/* Waveform visualization */}
        <div className="mt-4 flex items-center justify-center gap-1.5 h-6">
          {[0, 1, 2].map((i) => (
            <svg key={i} className="w-8 h-4 overflow-visible" viewBox="0 0 40 20">
              <path
                d="M 0 10 Q 10 0 20 10 T 40 10"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="text-slate-500/40"
              />
            </svg>
          ))}
        </div>

        {/* Bottom accent */}
        <div className="absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent via-slate-500/40 to-transparent" />
      </div>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-slate-400 !border-slate-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-slate-400 !border-slate-500/50 !w-2.5 !h-2.5"
      />
    </div>
  );
}

export default memo(BatchNormNode);
