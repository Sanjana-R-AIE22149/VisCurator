import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { ImageIcon } from 'lucide-react';

function InputNode({ data }: NodeProps) {
  return (
    <div className="group relative w-56">
      {/* Glow effect */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500/20 to-emerald-500/5 rounded-xl blur-sm opacity-60 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative bg-slate-900/90 border border-emerald-500/30 rounded-xl p-4 backdrop-blur-md">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-500/15 border border-emerald-500/25">
            <ImageIcon className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-emerald-300 leading-tight">
              {(data as any).label}
            </p>
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
              Input Layer
            </p>
          </div>
        </div>

        {/* Specs */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Channels</span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
              {(data as any).channels}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Resolution</span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
              {(data as any).resolution}
            </span>
          </div>
        </div>

        {/* Bottom accent bar */}
        <div className="absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent via-emerald-500/40 to-transparent" />
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-emerald-400 !border-emerald-500/50 !w-2.5 !h-2.5"
      />
    </div>
  );
}

export default memo(InputNode);
