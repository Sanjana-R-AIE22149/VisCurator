import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Copy, Star, Cpu, Layers, Zap, ExternalLink, Search, X, Database, History, TrendingUp, Download } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { getRecipeGraphTemplate, type RecipeTemplateId } from '../lib/graphTemplates';
import { BASE_URL } from '../lib/api';

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
  url: string;
  hfId: string;
}

const RECIPES: RecipeCard[] = [
  {
    id: 'yolov8-nano', name: 'YOLOv8', variant: 'Nano', task: 'Object Detection',
    params: '3.2M', mapScore: '37.3', speed: '1.47ms', tags: ['Real-time', 'Edge'],
    accent: '#2dd4bf', border: 'rgba(45,212,191,0.25)', glow: 'rgba(45,212,191,0.07)', stars: 1284,
    url: 'https://docs.ultralytics.com/models/yolov8/', hfId: 'Ultralytics/assets',
  },
  {
    id: 'yolov8-small', name: 'YOLOv8', variant: 'Small', task: 'Object Detection',
    params: '11.2M', mapScore: '44.9', speed: '2.66ms', tags: ['Balanced', 'GPU'],
    accent: '#38bdf8', border: 'rgba(56,189,248,0.25)', glow: 'rgba(56,189,248,0.07)', stars: 892,
    url: 'https://docs.ultralytics.com/models/yolov8/', hfId: 'Ultralytics/assets',
  },
  {
    id: 'resnet50', name: 'ResNet', variant: '50', task: 'Classification',
    params: '25.6M', mapScore: '76.1', speed: '4.10ms', tags: ['Classic', 'Pretrained'],
    accent: '#a78bfa', border: 'rgba(167,139,250,0.25)', glow: 'rgba(167,139,250,0.07)', stars: 2103,
    url: 'https://huggingface.co/microsoft/resnet-50', hfId: 'microsoft/resnet-50',
  },
  {
    id: 'efficientdet-d0', name: 'EfficientDet', variant: 'D0', task: 'Detection',
    params: '3.9M', mapScore: '33.8', speed: '2.54ms', tags: ['Efficient', 'Mobile'],
    accent: '#f59e0b', border: 'rgba(245,158,11,0.25)', glow: 'rgba(245,158,11,0.07)', stars: 674,
    url: 'https://github.com/google/automl/tree/master/efficientdet', hfId: '',
  },
  {
    id: 'detr', name: 'DETR', variant: 'ResNet-50', task: 'Detection',
    params: '41M', mapScore: '42.0', speed: '28ms', tags: ['Transformer', 'Attention'],
    accent: '#34d399', border: 'rgba(52,211,153,0.25)', glow: 'rgba(52,211,153,0.07)', stars: 543,
    url: 'https://huggingface.co/facebook/detr-resnet-50', hfId: 'facebook/detr-resnet-50',
  },
  {
    id: 'maskrcnn', name: 'Mask R-CNN', variant: 'X101', task: 'Segmentation',
    params: '63M', mapScore: '44.3', speed: '15ms', tags: ['Instance Seg', 'Heavy'],
    accent: '#fb7185', border: 'rgba(251,113,133,0.25)', glow: 'rgba(251,113,133,0.07)', stars: 921,
    url: 'https://github.com/facebookresearch/detectron2', hfId: '',
  },
];

export default function LibraryPage() {
  const navigate = useNavigate();
  const { setClonedRecipe, setNodes, setEdges, addToast } = useAppStore();
  const [visible, setVisible] = useState(false);
  const [cloned, setCloned] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [localDatasets, setLocalDatasets] = useState<any[]>([]);
  const [localModels, setLocalModels] = useState<any[]>([]);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    const fetchLocalData = async () => {
      try {
        const res = await fetch(`${BASE_URL}/api/library`);
        if (res.ok) {
          const data = await res.json();
          setLocalDatasets(data.datasets || []);
          setLocalModels(data.models || []);
        }
      } catch (err) {
        console.error('Failed to fetch library:', err);
      }
    };
    fetchLocalData();
  }, []);

  const filtered = RECIPES.filter(
    (r) =>
      r.name.toLowerCase().includes(search.toLowerCase()) ||
      r.task.toLowerCase().includes(search.toLowerCase()) ||
      r.variant.toLowerCase().includes(search.toLowerCase()) ||
      r.tags.some((t) => t.toLowerCase().includes(search.toLowerCase()))
  );

  const handleClone = (recipe: RecipeCard) => {
    const supportedTemplates = new Set<RecipeTemplateId>([
      'yolov8-nano',
      'resnet50',
      'efficientdet-d0',
      'detr',
    ]);

    setCloned(recipe.id);
    if (!supportedTemplates.has(recipe.id as RecipeTemplateId)) {
      addToast(`${recipe.name} ${recipe.variant} template is not available yet`, 'warning');
      setCloned(null);
      return;
    }

    const label = `${recipe.name} ${recipe.variant}`;
    const graph = getRecipeGraphTemplate(recipe.id as RecipeTemplateId);
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setClonedRecipe(label);
    addToast(`${label} graph loaded into Builder`, 'success');
    setTimeout(() => {
      setCloned(null);
      navigate('/builder');
    }, 250);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div
        className="max-w-7xl mx-auto px-6 py-8 space-y-12"
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.45s ease, transform 0.45s ease',
        }}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-[11px] font-mono text-sky-400/80 uppercase tracking-widest mb-1">Model Registry</p>
            <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Library</h1>
            <p className="text-sm text-slate-500 mt-1">
              Your curated datasets and trained models, along with community recipes.
            </p>
          </div>
          <div className="relative">
            <input
              type="text"
              placeholder="Search library…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 px-4 py-2.5 pl-10 rounded-xl bg-slate-900/60 border border-slate-800/60 text-slate-300 text-sm placeholder-slate-600 focus:outline-none focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 transition-all"
            />
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
            {search && (
              <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2">
                <X className="w-3.5 h-3.5 text-slate-500 hover:text-slate-300" />
              </button>
            )}
          </div>
        </div>

        {/* Local Datasets Section */}
        {localDatasets.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Database className="h-4 w-4 text-teal-400" />
              <h2 className="text-lg font-bold text-slate-200">Curated Datasets</h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {localDatasets.map((ds) => (
                <div key={ds.id} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-3">
                  <div className="flex justify-between items-start">
                    <div className="h-8 w-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
                      <Database className="h-4 w-4 text-teal-400" />
                    </div>
                    <a 
                      href={`${BASE_URL}/api/dataset/download/${ds.id}`}
                      className="text-slate-500 hover:text-teal-400 transition-colors"
                      title="Download Zip"
                    >
                      <Download className="h-4 w-4" />
                    </a>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-200 truncate">{ds.name}</h3>
                    <p className="text-[10px] text-slate-500 font-mono mt-0.5">{ds.images} images · {ds.classes.length} classes</p>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {ds.classes.slice(0, 3).map((c: string) => (
                      <span key={c} className="px-1.5 py-0.5 rounded bg-slate-800 text-[8px] text-slate-400 uppercase tracking-tighter">
                        {c}
                      </span>
                    ))}
                    {ds.classes.length > 3 && <span className="text-[8px] text-slate-600">+{ds.classes.length - 3}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Local Models Section */}
        {localModels.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <History className="h-4 w-4 text-purple-400" />
              <h2 className="text-lg font-bold text-slate-200">Training History</h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {localModels.map((run) => (
                <div key={run.id} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-3">
                  <div className="flex justify-between items-start">
                    <div className="h-8 w-8 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                      <TrendingUp className="h-4 w-4 text-purple-400" />
                    </div>
                    <button 
                      onClick={() => navigate(`/analytics?runId=${run.id}`)}
                      className="text-slate-500 hover:text-purple-400 transition-colors"
                      title="View Analytics"
                    >
                      <Zap className="h-4 w-4" />
                    </button>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-200 truncate">{run.id}</h3>
                    <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                      Acc: {(run.accuracy * 100).toFixed(1)}% · Epochs: {run.epochs}
                    </p>
                  </div>
                  <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                    <div 
                      className="bg-purple-500 h-full" 
                      style={{ width: `${run.accuracy * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recipes Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Cpu className="h-4 w-4 text-sky-400" />
            <h2 className="text-lg font-bold text-slate-200">Model Recipes</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
            {filtered.map((recipe) => (
              <div
                key={recipe.id}
                className="group relative rounded-2xl border overflow-hidden transition-transform duration-300 hover:-translate-y-1"
                style={{ borderColor: recipe.border, background: `${recipe.glow}, #0f172a80`, backdropFilter: 'blur(12px)' }}
              >
                <div className="absolute top-0 left-0 right-0 h-px"
                  style={{ background: `linear-gradient(90deg, transparent, ${recipe.accent}, transparent)` }} />
                <div className="p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="text-base font-bold text-slate-100">{recipe.name}</h3>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider"
                          style={{ background: recipe.glow, color: recipe.accent, border: `1px solid ${recipe.border}` }}>
                          {recipe.variant}
                        </span>
                      </div>
                      <p className="text-xs text-slate-500">{recipe.task}</p>
                    </div>
                    <a href={recipe.url} target="_blank" rel="noopener noreferrer"
                      className="p-1.5 rounded-lg text-slate-600 hover:text-slate-400 transition-colors">
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mb-4">
                    {[
                      { label: 'Params', value: recipe.params, icon: Cpu },
                      { label: 'mAP', value: `${recipe.mapScore}%`, icon: Zap },
                      { label: 'Speed', value: recipe.speed, icon: Layers },
                    ].map(({ label, value, icon: Icon }) => (
                      <div key={label} className="rounded-lg px-2.5 py-2 text-center"
                        style={{ background: recipe.glow, border: `1px solid ${recipe.border}` }}>
                        <Icon className="w-3 h-3 mx-auto mb-1 opacity-60" style={{ color: recipe.accent }} />
                        <p className="text-xs font-bold font-mono text-slate-300">{value}</p>
                        <p className="text-[9px] uppercase tracking-wider text-slate-600">{label}</p>
                      </div>
                    ))}
                  </div>

                  <div className="flex flex-wrap gap-1.5 mb-4">
                    {recipe.tags.map((tag) => (
                      <span key={tag} className="px-2 py-0.5 rounded-md text-[10px] text-slate-500 border border-slate-800/60 bg-slate-900/40">
                        {tag}
                      </span>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-3 border-t border-slate-800/40">
                    <div className="flex items-center gap-1">
                      <Star className="w-3 h-3" style={{ color: recipe.accent, fill: recipe.accent }} />
                      <span className="text-xs font-mono text-slate-500">{recipe.stars.toLocaleString()}</span>
                    </div>
                    <button
                      onClick={() => handleClone(recipe)}
                      className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
                      style={
                        cloned === recipe.id
                          ? { background: recipe.glow, color: recipe.accent, border: `1px solid ${recipe.border}` }
                          : { background: recipe.accent, color: '#0a0a0a', boxShadow: `0 0 16px 2px ${recipe.glow}` }
                      }
                    >
                      <Copy className="w-3.5 h-3.5" />
                      {cloned === recipe.id ? 'Loading…' : 'Clone'}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {filtered.length === 0 && localDatasets.length === 0 && localModels.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24">
            <div className="w-20 h-20 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-6">
              <Search className="w-10 h-10 text-slate-700" />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">No Items Found</h2>
            <p className="text-sm text-slate-500 max-w-xs text-center">
              Nothing matches your search criteria.
            </p>
            <button onClick={() => setSearch('')}
              className="mt-6 text-xs font-bold text-sky-400 hover:text-sky-300 uppercase tracking-widest transition-colors">
              Clear search
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
