import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { ArrowDownToLine } from 'lucide-react';

function PoolingNode({ data }: NodeProps) {
  return (
    <div className="group relative w-56">
      {/* Glow */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-rose-500/20 to-rose-500/5 rounded-xl blur-sm opacity-60 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative bg-slate-900/90 border border-rose-500/30 rounded-xl p-4 backdrop-blur-md">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-rose-500/15 border border-rose-500/25">
            <ArrowDownToLine className="w-3.5 h-3.5 text-rose-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-rose-300 leading-tight">
              {(data as any).label}
            </p>
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
              Pooling
            </p>
          </div>
        </div>

        {/* Specs */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Type</span>
            <span className="text-[10px] font-mono text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded">
              {(data as any).pool_type}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Kernel</span>
            <span className="text-[10px] font-mono text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded">
              {(data as any).kernel}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Stride</span>
            <span className="text-[10px] font-mono text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded">
              {(data as any).stride}
            </span>
          </div>
        </div>

        {/* Bottom accent */}
        <div className="absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent via-rose-500/40 to-transparent" />
      </div>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-rose-400 !border-rose-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-rose-400 !border-rose-500/50 !w-2.5 !h-2.5"
      />
    </div>
  );
}

export default memo(PoolingNode);
