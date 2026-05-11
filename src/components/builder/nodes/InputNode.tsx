import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { ImageIcon } from 'lucide-react';
import NodeWrapper, { type EditableField } from './NodeWrapper';

const EDITABLE_FIELDS: EditableField[] = [
  { key: 'label', label: 'Layer Name', type: 'text' },
  { key: 'channels', label: 'Channels', type: 'number', min: 1 },
  { key: 'resolution', label: 'Resolution', type: 'text' },
];

function InputNode({ id, data }: NodeProps) {
  return (
    <NodeWrapper
      nodeId={id}
      accentFrom="from-emerald-500/20"
      accentTo="to-emerald-500/5"
      borderColor="border-emerald-500/30"
      accentLine="via-emerald-500/40"
      data={data as any}
      editableFields={EDITABLE_FIELDS}
    >
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

      <Handle
        type="source"
        position={Position.Right}
        className="!bg-emerald-400 !border-emerald-500/50 !w-2.5 !h-2.5"
      />
    </NodeWrapper>
  );
}

export default memo(InputNode);
