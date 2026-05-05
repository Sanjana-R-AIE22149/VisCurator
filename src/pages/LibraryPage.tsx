import { useEffect, useState } from 'react';
import { Copy, Star, Cpu, Layers, Zap, ExternalLink } from 'lucide-react';

interface RecipeCard {
  id: string;
  name: string;
  variant: string;
  task: string;
  params: string;
  mapScore: string;
  speed: string;
  tags: string[];
  accent: string;
  border: string;
  glow: string;
  stars: number;
}

const RECIPES: RecipeCard[] = [
  {
    id: 'yolov8-nano',
    name: 'YOLOv8',
    variant: 'Nano',
    task: 'Object Detection',
    params: '3.2M',
    mapScore: '37.3',
    speed: '1.47ms',
    tags: ['Real-time', 'Edge'],
    accent: '#2dd4bf',
    border: 'rgba(45,212,191,0.25)',
    glow: 'rgba(45,212,191,0.07)',
    stars: 1284,
  },
  {
    id: 'yolov8-small',
    name: 'YOLOv8',
    variant: 'Small',
    task: 'Object Detection',
    params: '11.2M',
    mapScore: '44.9',
    speed: '2.66ms',
    tags: ['Balanced', 'GPU'],
    accent: '#38bdf8',
    border: 'rgba(56,189,248,0.25)',
    glow: 'rgba(56,189,248,0.07)',
    stars: 892,
  },
  {
    id: 'resnet50',
    name: 'ResNet',
    variant: '50',
    task: 'Classification',
    params: '25.6M',
    mapScore: '76.1',
    speed: '4.10ms',
    tags: ['Classic', 'Pretrained'],
    accent: '#a78bfa',
    border: 'rgba(167,139,250,0.25)',
    glow: 'rgba(167,139,250,0.07)',
    stars: 2103,
  },
  {
    id: 'efficientdet-d0',
    name: 'EfficientDet',
    variant: 'D0',
    task: 'Detection',
    params: '3.9M',
    mapScore: '33.8',
    speed: '2.54ms',
    tags: ['Efficient', 'Mobile'],
    accent: '#f59e0b',
    border: 'rgba(245,158,11,0.25)',
    glow: 'rgba(245,158,11,0.07)',
    stars: 674,
  },
  {
    id: 'detr',
    name: 'DETR',
    variant: 'ResNet-50',
    task: 'Detection',
    params: '41M',
    mapScore: '42.0',
    speed: '28ms',
    tags: ['Transformer', 'Attention'],
    accent: '#34d399',
    border: 'rgba(52,211,153,0.25)',
    glow: 'rgba(52,211,153,0.07)',
    stars: 543,
  },
  {
    id: 'maskrcnn',
    name: 'Mask R-CNN',
    variant: 'X101',
    task: 'Segmentation',
    params: '63M',
    mapScore: '44.3',
    speed: '15ms',
    tags: ['Instance Seg', 'Heavy'],
    accent: '#fb7185',
    border: 'rgba(251,113,133,0.25)',
    glow: 'rgba(251,113,133,0.07)',
    stars: 921,
  },
];

export default function LibraryPage() {
  const [visible, setVisible] = useState(false);
  const [cloned, setCloned] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, []);

  const filtered = RECIPES.filter(
    (r) =>
      r.name.toLowerCase().includes(search.toLowerCase()) ||
      r.task.toLowerCase().includes(search.toLowerCase()) ||
      r.variant.toLowerCase().includes(search.toLowerCase())
  );

  const handleClone = (id: string) => {
    setCloned(id);
    setTimeout(() => setCloned(null), 2000);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div
        className="max-w-7xl mx-auto px-6 py-8 space-y-6"
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.45s ease, transform 0.45s ease',
        }}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-[11px] font-mono text-sky-400/80 uppercase tracking-widest mb-1">
              Model Registry
            </p>
            <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Library</h1>
            <p className="text-sm text-slate-500 mt-1">
              Pre-built model recipes. Clone to your workspace and start training instantly.
            </p>
          </div>
          {/* Search */}
          <div className="relative">
            <input
              id="library-search-input"
              type="text"
              placeholder="Search recipes…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 px-4 py-2.5 pl-10 rounded-xl bg-slate-900/60 border border-slate-800/60 text-slate-300 text-sm placeholder-slate-600 focus:outline-none focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 transition-all"
            />
            <Layers className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
          </div>
        </div>

        {/* Recipe Card Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((recipe) => (
            <div
              key={recipe.id}
              id={`recipe-card-${recipe.id}`}
              className="group relative rounded-2xl border overflow-hidden transition-transform duration-300 hover:-translate-y-1"
              style={{
                borderColor: recipe.border,
                background: `${recipe.glow}, #0f172a80`,
                backdropFilter: 'blur(12px)',
              }}
            >
              {/* Top color band */}
              <div
                className="absolute top-0 left-0 right-0 h-px"
                style={{ background: `linear-gradient(90deg, transparent, ${recipe.accent}, transparent)` }}
              />

              <div className="p-5">
                {/* Header row */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-base font-bold text-slate-100">{recipe.name}</h3>
                      <span
                        className="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider"
                        style={{ background: recipe.glow, color: recipe.accent, border: `1px solid ${recipe.border}` }}
                      >
                        {recipe.variant}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500">{recipe.task}</p>
                  </div>
                  <button
                    type="button"
                    id={`recipe-details-${recipe.id}`}
                    className="p-1.5 rounded-lg text-slate-600 hover:text-slate-400 transition-colors"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Metric row */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  {[
                    { label: 'Params', value: recipe.params, icon: Cpu },
                    { label: 'mAP', value: `${recipe.mapScore}%`, icon: Zap },
                    { label: 'Speed', value: recipe.speed, icon: Layers },
                  ].map(({ label, value, icon: Icon }) => (
                    <div
                      key={label}
                      className="rounded-lg px-2.5 py-2 text-center"
                      style={{ background: recipe.glow, border: `1px solid ${recipe.border}` }}
                    >
                      <Icon className="w-3 h-3 mx-auto mb-1 opacity-60" style={{ color: recipe.accent }} />
                      <p className="text-xs font-bold font-mono text-slate-300">{value}</p>
                      <p className="text-[9px] uppercase tracking-wider text-slate-600">{label}</p>
                    </div>
                  ))}
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {recipe.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 rounded-md text-[10px] text-slate-500 border border-slate-800/60 bg-slate-900/40"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between pt-3 border-t border-slate-800/40">
                  <div className="flex items-center gap-1 text-slate-600">
                    <Star className="w-3 h-3" style={{ color: recipe.accent, fill: recipe.accent }} />
                    <span className="text-xs font-mono text-slate-500">{recipe.stars.toLocaleString()}</span>
                  </div>
                  <button
                    id={`clone-btn-${recipe.id}`}
                    type="button"
                    onClick={() => handleClone(recipe.id, `${recipe.name} ${recipe.variant}`)}
                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
                    style={
                      cloned === recipe.id
                        ? { background: recipe.glow, color: recipe.accent, border: `1px solid ${recipe.border}` }
                        : { background: recipe.accent, color: '#0a0a0a', boxShadow: `0 0 16px 2px ${recipe.glow}` }
                    }
                  >
                    <Copy className="w-3.5 h-3.5" />
                    {cloned === recipe.id ? 'Cloned!' : 'Clone'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 animate-fade-up">
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-sky-500/10 blur-3xl rounded-full" />
              <div className="relative w-20 h-20 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center shadow-2xl">
                <SearchX className="w-10 h-10 text-slate-700" />
              </div>
            </div>
            <h2 className="text-xl font-bold text-white mb-2">No Recipes Found</h2>
            <p className="text-sm text-slate-500 max-w-xs mx-auto">
              We couldn't find any models matching "<span className="text-slate-300 italic">{search}</span>". Try adjusting your filters.
            </p>
            <button 
              onClick={() => setSearch('')}
              className="mt-6 text-xs font-bold text-sky-400 hover:text-sky-300 uppercase tracking-widest transition-colors"
            >
              Reset Search
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
