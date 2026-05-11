import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { BarChart3 } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'num_classes', label: 'Num Classes', type: 'number', min: 1 },
  { key: 'activation', label: 'Activation', type: 'select', options: ['None', 'Softmax', 'Sigmoid'] },
];

function OutputNode({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-emerald-600/20"
      accentTo="to-emerald-600/5"
      borderColor="border-emerald-600/30"
      accentLine="via-emerald-600/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-600/15 border border-emerald-600/25">
          <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
        </div>
        <div>
          <p className="text-xs font-semibold text-emerald-300 leading-tight">
            {(data as any).label}
          </p>
          <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
            Output Layer
          </p>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Classes</span>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-600/10 px-1.5 py-0.5 rounded">
            {(data as any).num_classes}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">Activation</span>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-600/10 px-1.5 py-0.5 rounded">
            {(data as any).activation}
          </span>
        </div>
      </div>

      <div className="mt-3 flex items-end justify-center gap-1.5 h-8 p-1.5 bg-emerald-600/5 rounded-lg border border-emerald-600/10">
        {[0.4, 0.9, 0.3, 0.6, 0.2].map((val, i) => (
          <div
            key={i}
            className="w-2 rounded-t-sm bg-emerald-500/40"
            style={{ height: `${val * 100}%` }}
          />
        ))}
      </div>

      <Handle
        type="target"
        position={Position.Left}
        className="!bg-emerald-500 !border-emerald-600/50 !w-2.5 !h-2.5"
      />
    </NodeWrapper>
  );
}

export default memo(OutputNode);
