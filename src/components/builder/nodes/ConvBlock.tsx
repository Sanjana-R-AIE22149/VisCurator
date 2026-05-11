import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Layers } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'filters', label: 'Filters', type: 'number', min: 1 },
  { key: 'kernel', label: 'Kernel Size', type: 'select', options: ['1x1', '3x3', '5x5', '7x7'] },
  { key: 'activation', label: 'Activation', type: 'select', options: ['None', 'ReLU', 'GELU', 'SiLU'] },
];

function ConvBlock({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-sky-500/20"
      accentTo="to-sky-500/5"
      borderColor="border-sky-500/30"
      accentLine="via-sky-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-sky-500/15 border border-sky-500/25">
          <Layers className="w-3.5 h-3.5 text-sky-400" />
        </div>
        <div>
          <p className="text-xs font-semibold text-sky-300 leading-tight">
            {(data as any).label}
          </p>
          <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
            Convolution
          </p>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Filters</span>
          <span className="text-[10px] font-mono text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded">
            {(data as any).filters}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Kernel</span>
          <span className="text-[10px] font-mono text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded">
            {(data as any).kernel}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Activation</span>
          <span className="text-[10px] font-mono text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded">
            {(data as any).activation}
          </span>
        </div>
      </div>

      <Handle
        type="target"
        position={Position.Left}
        className="!bg-sky-400 !border-sky-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-sky-400 !border-sky-500/50 !w-2.5 !h-2.5"
      />
    </NodeWrapper>
  );
}

export default memo(ConvBlock);
