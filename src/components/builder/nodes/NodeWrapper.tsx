/**
 * NodeWrapper — shared container for all builder nodes.
 *
 * Features:
 *  • Delete button (×) on hover — removes node + connected edges from the graph
 *  • Click-to-edit inline param panel (fields passed as `editableFields`)
 *  • Resolution propagation: when input node changes resolution, all downstream
 *    nodes display their computed feature-map size
 */
import { useState, useCallback } from 'react';
import { X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { useAppStore } from '../../../store/useAppStore';

export interface EditableField {
  key: string;
  label: string;
  type: 'text' | 'number' | 'select';
  options?: string[];           // for select type
  min?: number; max?: number;   // for number type
}

interface NodeWrapperProps {
  nodeId: string;
  accentFrom: string;   // e.g. 'from-emerald-500/20'
  accentTo: string;     // e.g. 'to-emerald-500/5'
  borderColor: string;  // e.g. 'border-emerald-500/30'
  accentLine: string;   // e.g. 'via-emerald-500/40'
  data: Record<string, any>;
  editableFields: EditableField[];
  children: React.ReactNode;
}

export default function NodeWrapper({
  nodeId,
  accentFrom,
  accentTo,
  borderColor,
  accentLine,
  data,
  editableFields,
  children,
}: NodeWrapperProps) {
  const [editing, setEditing] = useState(false);
  const { getEdges, deleteElements } = useReactFlow();
  const { updateNodeData } = useAppStore();

  const handleDelete = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const connectedEdges = getEdges().filter(
        (ed) => ed.source === nodeId || ed.target === nodeId
      );
      deleteElements({ nodes: [{ id: nodeId }], edges: connectedEdges });
    },
    [nodeId, getEdges, deleteElements]
  );

  const handleFieldChange = useCallback(
    (key: string, value: string | number) => {
      updateNodeData(nodeId, { [key]: value });
    },
    [nodeId, updateNodeData]
  );

  return (
    <div className="group relative w-56">
      {/* Glow */}
      <div className={`absolute -inset-0.5 bg-gradient-to-r ${accentFrom} ${accentTo} rounded-xl blur-sm opacity-60 group-hover:opacity-100 transition-opacity duration-300`} />

      {/* Delete button */}
      <button
        onClick={handleDelete}
        className="absolute -top-2.5 -right-2.5 z-20 hidden group-hover:flex items-center justify-center w-5 h-5 rounded-full bg-rose-500 border border-rose-400/60 text-white shadow-lg transition-transform hover:scale-110"
        title="Delete node"
      >
        <X className="w-2.5 h-2.5" />
      </button>

      {/* Card */}
      <div
        className={`relative bg-slate-900/90 border ${borderColor} rounded-xl p-4 backdrop-blur-md cursor-pointer select-none`}
        onClick={() => setEditing((e) => !e)}
      >
        {children}

        {/* Inline editor */}
        {editing && (
          <div
            className="mt-3 space-y-2 border-t border-slate-800/60 pt-3"
            onClick={(e) => e.stopPropagation()}
          >
            {editableFields.map((field) => (
              <div key={field.key} className="flex flex-col gap-0.5">
                <label className="text-[9px] uppercase tracking-widest text-slate-500">
                  {field.label}
                </label>
                {field.type === 'select' ? (
                  <select
                    value={data[field.key] ?? ''}
                    onChange={(e) => handleFieldChange(field.key, e.target.value)}
                    className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-[11px] text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500/40"
                  >
                    {field.options?.map((o) => (
                      <option key={o} value={o}>{o}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type={field.type}
                    min={field.min}
                    max={field.max}
                    value={data[field.key] ?? ''}
                    onChange={(e) =>
                      handleFieldChange(
                        field.key,
                        field.type === 'number' ? Number(e.target.value) : e.target.value
                      )
                    }
                    className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-[11px] font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500/40"
                  />
                )}
              </div>
            ))}
            <p className="text-[9px] text-slate-600 text-right">click card to close</p>
          </div>
        )}

        {/* Computed Shape Output */}
        {data.computed_res && (
          <div className="mt-2 pt-2 border-t border-slate-700/50 flex justify-between items-center">
            <span className="text-[9px] uppercase tracking-widest text-slate-500">Output Shape</span>
            <span className="text-[10px] font-mono text-slate-300 bg-slate-800/80 px-1.5 py-0.5 rounded">
              {data.computed_channels} × {data.computed_res}
            </span>
          </div>
        )}

        {/* Bottom accent */}
        <div className={`absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent ${accentLine} to-transparent`} />
      </div>
    </div>
  );
}
