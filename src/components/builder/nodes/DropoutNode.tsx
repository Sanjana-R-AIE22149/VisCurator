import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Scissors } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'p', label: 'Rate (p)', type: 'number', min: 0, max: 1 },
];

function DropoutNode({ id, data }: NodeProps) {
  const p = (data as any).p || 0.5;
  const pPercent = Math.round(p * 100);

  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-yellow-500/20"
      accentTo="to-yellow-500/5"
      borderColor="border-yellow-500/30"
      accentLine="via-yellow-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
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

      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Rate (p)</span>
          <span className="text-[10px] font-mono text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
            {pPercent}%
          </span>
        </div>
      </div>

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
    </NodeWrapper>
  );
}

export default memo(DropoutNode);
