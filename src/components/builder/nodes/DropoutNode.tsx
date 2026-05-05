import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Scissors } from 'lucide-react';

function DropoutNode({ data }: NodeProps) {
  const p = (data as any).p || 0.5;
  const pPercent = Math.round(p * 100);

  return (
    <div className="group relative w-56">
      {/* Glow */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-yellow-500/20 to-yellow-500/5 rounded-xl blur-sm opacity-60 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative bg-slate-900/90 border border-yellow-500/30 rounded-xl p-4 backdrop-blur-md">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-yellow-500/15 border border-yellow-500/25">
            <Scissors className="w-3.5 h-3.5 text-yellow-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-yellow-300 leading-tight">
              {(data as any).label}
            </p>
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
              Regularization
            </p>
          </div>
        </div>

        {/* Specs */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500">Rate (p)</span>
            <span className="text-[10px] font-mono text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
              {pPercent}%
            </span>
          </div>
        </div>

        {/* Dropout visual */}
        <div className="mt-3 flex flex-wrap gap-1 justify-center p-2 bg-yellow-500/5 rounded-lg border border-yellow-500/10">
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={i}
              className={`w-1.5 h-1.5 rounded-full transition-all duration-500 ${
                Math.random() > p ? 'bg-yellow-500/40' : 'bg-slate-800'
              }`}
            />
          ))}
          <div className="w-full text-center mt-1">
             <span className="text-[8px] font-mono text-yellow-500/50 uppercase">{pPercent}% dropped</span>
          </div>
        </div>

        {/* Bottom accent */}
        <div className="absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent via-yellow-500/40 to-transparent" />
      </div>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-yellow-400 !border-yellow-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-yellow-400 !border-yellow-500/50 !w-2.5 !h-2.5"
      />
    </div>
  );
}

export default memo(DropoutNode);
