import { useState } from 'react';
import { 
  ImageIcon, Layers, Sparkles, Box, ArrowDownToLine, Shuffle, 
  GitBranch, Scissors, BarChart3, Search 
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

interface PaletteItem {
  type: string;
  label: string;
  sublabel: string;
  icon: React.ComponentType<{ className?: string }>;
  accent: string;
  borderAccent: string;
  defaultData: Record<string, unknown>;
  category: 'Input/Output' | 'Feature Extraction' | 'Attention' | 'Normalization & Regularization' | 'Readout';
}

const PALETTE_ITEMS: PaletteItem[] = [
  {
    type: 'inputNode',
    label: 'Input',
    sublabel: 'Image Tensor',
    icon: ImageIcon,
    accent: 'text-emerald-400',
    borderAccent: 'border-emerald-500/20 hover:border-emerald-500/40',
    defaultData: { label: 'Image Input', channels: 3, resolution: '224×224' },
    category: 'Input/Output',
  },
  {
    type: 'outputNode',
    label: 'Output',
    sublabel: 'Final Layer',
    icon: BarChart3,
    accent: 'text-emerald-500',
    borderAccent: 'border-emerald-600/20 hover:border-emerald-600/40',
    defaultData: { label: 'Output Layer', num_classes: 10, activation: 'Softmax' },
    category: 'Input/Output',
  },
  {
    type: 'convBlock',
    label: 'Conv2d',
    sublabel: 'Convolution',
    icon: Layers,
    accent: 'text-sky-400',
    borderAccent: 'border-sky-500/20 hover:border-sky-500/40',
    defaultData: { label: 'Conv2d Block', filters: 64, kernel: '3×3', activation: 'ReLU' },
    category: 'Feature Extraction',
  },
  {
    type: 'residualBlock',
    label: 'Residual',
    sublabel: 'Skip Connection',
    icon: GitBranch,
    accent: 'text-amber-400',
    borderAccent: 'border-amber-500/20 hover:border-amber-500/40',
    defaultData: { label: 'ResBlock', in_channels: 64, out_channels: 64, stride: 1 },
    category: 'Feature Extraction',
  },
  {
    type: 'poolingNode',
    label: 'Pooling',
    sublabel: 'MaxPool2d',
    icon: ArrowDownToLine,
    accent: 'text-rose-400',
    borderAccent: 'border-rose-500/20 hover:border-rose-500/40',
    defaultData: { label: 'MaxPool2d', pool_type: 'Max', kernel: '2×2', stride: 2 },
    category: 'Feature Extraction',
  },
  {
    type: 'attentionBlock',
    label: 'Attention',
    sublabel: 'Self-Attention',
    icon: Sparkles,
    accent: 'text-violet-400',
    borderAccent: 'border-violet-500/20 hover:border-violet-500/40',
    defaultData: { label: 'Self-Attention', heads: 8, dimK: 64 },
    category: 'Attention',
  },
  {
    type: 'batchNormNode',
    label: 'BatchNorm',
    sublabel: 'Normalization',
    icon: Shuffle,
    accent: 'text-slate-400',
    borderAccent: 'border-slate-500/20 hover:border-slate-500/40',
    defaultData: { label: 'BatchNorm2d', num_features: 64, eps: 1e-5, momentum: 0.1 },
    category: 'Normalization & Regularization',
  },
  {
    type: 'dropoutNode',
    label: 'Dropout',
    sublabel: 'Regularization',
    icon: Scissors,
    accent: 'text-yellow-400',
    borderAccent: 'border-yellow-500/20 hover:border-yellow-500/40',
    defaultData: { label: 'Dropout', p: 0.5 },
    category: 'Normalization & Regularization',
  },
  {
    type: 'linearNode',
    label: 'Linear',
    sublabel: 'Fully Connected',
    icon: Box,
    accent: 'text-teal-400',
    borderAccent: 'border-teal-500/20 hover:border-teal-500/40',
    defaultData: { label: 'Linear', in_features: 512, out_features: 10, bias: true, activation: 'ReLU' },
    category: 'Readout',
  },
];

const CATEGORIES = [
  'Input/Output',
  'Feature Extraction',
  'Attention',
  'Normalization & Regularization',
  'Readout',
] as const;

export default function ComponentPalette() {
  const addNode = useAppStore((s) => s.addNode);
  const nodes = useAppStore((s) => s.nodes);
  const [search, setSearch] = useState('');

  const handleAdd = (item: PaletteItem) => {
    const id = `${item.type}-${Date.now()}`;
    const lastNode = nodes[nodes.length - 1];
    const x = lastNode ? lastNode.position.x + 320 : 100;
    const y = lastNode ? lastNode.position.y : 200;

    addNode({
      id,
      type: item.type,
      position: { x, y: y + (Math.random() * 60 - 30) },
      data: { ...item.defaultData },
    });
  };

  const onDragStart = (event: React.DragEvent, nodeType: string, data: any) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('application/reactflow-data', JSON.stringify(data));
    event.dataTransfer.effectAllowed = 'move';
  };

  const filteredItems = PALETTE_ITEMS.filter(
    (item) =>
      item.label.toLowerCase().includes(search.toLowerCase()) ||
      item.sublabel.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="w-64 h-full border-r border-slate-800/60 bg-slate-950/80 backdrop-blur-xl flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800/50">
        <h3 className="text-xs font-semibold text-slate-300 tracking-wide uppercase">Components</h3>
        <p className="text-[9px] text-slate-500 mt-0.5">Drag or click to add to canvas</p>
      </div>

      {/* Search */}
      <div className="p-3 border-b border-slate-800/30">
        <div className="relative group">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-600 group-focus-within:text-sky-400 transition-colors" />
          <input
            type="text"
            placeholder="Search layers..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-900/50 border border-slate-800 rounded-md py-1.5 pl-8 pr-3 text-[11px] text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 transition-all"
          />
        </div>
      </div>

      {/* Palette items */}
      <div className="flex-1 overflow-y-auto p-3 custom-scrollbar">
        {CATEGORIES.map((cat) => {
          const items = filteredItems.filter((i) => i.category === cat);
          if (items.length === 0) return null;

          return (
            <div key={cat} className="mb-4 last:mb-0">
              <h4 className="text-[9px] font-bold text-slate-600 uppercase tracking-widest mb-2 px-1">
                {cat}
              </h4>
              <div className="space-y-1">
                {items.map((item, idx) => {
                  const Icon = item.icon;
                  return (
                    <div
                      key={idx}
                      draggable
                      onDragStart={(e) => onDragStart(e, item.type, item.defaultData)}
                      onClick={() => handleAdd(item)}
                      className={`
                        w-full flex items-center gap-3 px-3 py-2
                        rounded-lg border bg-slate-900/30
                        ${item.borderAccent}
                        transition-all duration-200
                        hover:bg-slate-800/50
                        group cursor-grab active:cursor-grabbing
                      `}
                    >
                      <div className={`p-1.5 rounded-md bg-slate-800/80 ${item.accent}`}>
                        <Icon className="w-3.5 h-3.5" />
                      </div>
                      <div className="text-left">
                        <p className="text-[11px] font-medium text-slate-300 group-hover:text-slate-100 transition-colors">
                          {item.label}
                        </p>
                        <p className="text-[9px] text-slate-600 font-mono leading-tight">{item.sublabel}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer hint */}
      <div className="px-4 py-3 border-t border-slate-800/50 bg-slate-950/40">
        <p className="text-[9px] text-slate-600 text-center">
          Building ResNet-50 v2 Architecture
        </p>
      </div>
    </div>
  );
}
