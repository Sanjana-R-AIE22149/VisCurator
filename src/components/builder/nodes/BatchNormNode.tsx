import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Shuffle } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'num_features', label: 'Num Features', type: 'number', min: 1 },
  { key: 'eps', label: 'Eps (e.g. 1e-5)', type: 'number' },
  { key: 'momentum', label: 'Momentum', type: 'number' },
];

function BatchNormNode({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-slate-500/20"
      accentTo="to-slate-500/5"
      borderColor="border-slate-500/30"
      accentLine="via-slate-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
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
    </NodeWrapper>
  );
}

export default memo(BatchNormNode);
