import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Box } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'in_features', label: 'In Features', type: 'number', min: 1 },
  { key: 'out_features', label: 'Out Features', type: 'number', min: 1 },
  { key: 'bias', label: 'Bias', type: 'select', options: ['true', 'false'] },
  { key: 'activation', label: 'Activation', type: 'select', options: ['None', 'ReLU', 'GELU', 'SiLU', 'Softmax'] },
];

function LinearNode({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-teal-500/20"
      accentTo="to-teal-500/5"
      borderColor="border-teal-500/30"
      accentLine="via-teal-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-teal-500/15 border border-teal-500/25">
          <Box className="w-3.5 h-3.5 text-teal-400" />
        </div>
        <div>
          <p className="text-xs font-semibold text-teal-300 leading-tight">
            {(data as any).label}
          </p>
          <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
            Fully Connected
          </p>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">In Features</span>
          <span className="text-[10px] font-mono text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded">
            {(data as any).in_features}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Out Features</span>
          <span className="text-[10px] font-mono text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded">
            {(data as any).out_features}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Bias</span>
          <span className="text-[10px] font-mono text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded">
            {String((data as any).bias)}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Activation</span>
          <span className="text-[10px] font-mono text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded">
            {(data as any).activation}
          </span>
        </div>
      </div>

      <Handle
        type="target"
        position={Position.Left}
        className="!bg-teal-400 !border-teal-500/50 !w-2.5 !h-2.5"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-teal-400 !border-teal-500/50 !w-2.5 !h-2.5"
      />
    </NodeWrapper>
  );
}

export default memo(LinearNode);
