import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { ArrowDownToLine } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'pool_type', label: 'Type', type: 'select', options: ['Max', 'Average', 'AdaptiveAvg'] },
  { key: 'kernel', label: 'Kernel Size', type: 'select', options: ['2x2', '3x3', 'N/A'] },
  { key: 'stride', label: 'Stride', type: 'number', min: 1 },
];

function PoolingNode({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-rose-500/20"
      accentTo="to-rose-500/5"
      borderColor="border-rose-500/30"
      accentLine="via-rose-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
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
    </NodeWrapper>
  );
}

export default memo(PoolingNode);
