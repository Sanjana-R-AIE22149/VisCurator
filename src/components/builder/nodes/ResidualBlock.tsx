import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { GitBranch } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'in_channels', label: 'In Channels', type: 'number', min: 1 },
  { key: 'out_channels', label: 'Out Channels', type: 'number', min: 1 },
  { key: 'stride', label: 'Stride', type: 'number', min: 1 },
];

function ResidualBlock({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-amber-500/20"
      accentTo="to-amber-500/5"
      borderColor="border-amber-500/30"
      accentLine="via-amber-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-amber-500/15 border border-amber-500/25">
          <GitBranch className="w-3.5 h-3.5 text-amber-400" />
        </div>
        <div>
          <p className="text-xs font-semibold text-amber-300 leading-tight">
            {(data as any).label}
          </p>
          <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
            Skip Connection
          </p>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">In Channels</span>
          <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">
            {(data as any).in_channels}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Out Channels</span>
          <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">
            {(data as any).out_channels}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Stride</span>
          <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">
            {(data as any).stride ?? 1}
          </span>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-center gap-2 px-2 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
          <div className="flex-1 h-px bg-amber-500/20" />
          <span className="text-[8px] font-mono text-amber-500/60 uppercase tracking-tighter">→ bypass</span>
          <div className="flex-1 h-px bg-amber-500/20" />
      </div>

      <Handle
        type="target"
        position={Position.Left}
        className="!bg-amber-400 !border-amber-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-amber-400 !border-amber-500/50 !w-2.5 !h-2.5"
      />
    </NodeWrapper>
  );
}

export default memo(ResidualBlock);
