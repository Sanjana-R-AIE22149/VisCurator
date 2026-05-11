import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Sparkles } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'heads', label: 'Heads', type: 'number', min: 1 },
  { key: 'dimK', label: 'Key Dim (d_k)', type: 'number', min: 1 },
];

function AttentionBlock({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-violet-500/20"
      accentTo="to-violet-500/5"
      borderColor="border-violet-500/30"
      accentLine="via-violet-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-violet-500/15 border border-violet-500/25">
          <Sparkles className="w-3.5 h-3.5 text-violet-400" />
        </div>
        <div>
          <p className="text-xs font-semibold text-violet-300 leading-tight">
            {(data as any).label}
          </p>
          <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
            Attention
          </p>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Heads</span>
          <span className="text-[10px] font-mono text-violet-400 bg-violet-500/10 px-1.5 py-0.5 rounded">
            {(data as any).heads}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Key Dim</span>
          <span className="text-[10px] font-mono text-violet-400 bg-violet-500/10 px-1.5 py-0.5 rounded">
            d_k={String((data as any).dimK)}
          </span>
        </div>
      </div>

      <div className="mt-3 flex gap-1 justify-center">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="w-1 rounded-full bg-violet-500/30"
            style={{
              height: `${8 + Math.sin(i * 0.8) * 6}px`,
              animationDelay: `${i * 100}ms`,
            }}
          />
        ))}
      </div>

      <Handle
        type="target"
        position={Position.Left}
        className="!bg-violet-400 !border-violet-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-violet-400 !border-violet-500/50 !w-2.5 !h-2.5"
      />
    </NodeWrapper>
  );
}

export default memo(AttentionBlock);
