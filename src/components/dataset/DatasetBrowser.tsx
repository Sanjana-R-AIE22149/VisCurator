import { useState, useEffect, useRef } from 'react';
import { Search, ExternalLink, ThumbsUp, Download, ChevronRight, Loader2, BookOpen } from 'lucide-react';
import { BASE_URL } from '../../lib/api';
import { useAppStore } from '../../store/useAppStore';

interface BrowseResult {
  source: string;
  dataset_id: string;
  name: string;
  description: string;
  downloads: number;
  likes: number;
  tags: string[];
  url: string;
  size_estimate: string;
  unavailable?: boolean;
}

const SOURCE_BADGE: Record<string, string> = {
  HuggingFace:    'text-yellow-400 border-yellow-400/30 bg-yellow-400/10',
  Kaggle:         'text-sky-400 border-sky-400/30 bg-sky-400/10',
  Roboflow:       'text-purple-400 border-purple-400/30 bg-purple-400/10',
  PapersWithCode: 'text-emerald-400 border-emerald-400/30 bg-emerald-400/10',
};

const SOURCES = [
  { value: 'huggingface,paperswithcode', label: 'All available' },
  { value: 'huggingface',               label: 'HuggingFace' },
  { value: 'paperswithcode',            label: 'Papers With Code' },
  { value: 'huggingface,kaggle,roboflow,paperswithcode', label: 'All (incl. Kaggle/RF)' },
];

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export default function DatasetBrowser() {
  const { setDatasetQuery, setDatasetSource } = useAppStore();

  const [query, setQuery]           = useState('');
  const [sources, setSources]       = useState(SOURCES[0].value);
  const [results, setResults]       = useState<BrowseResult[]>([]);
  const [loading, setLoading]       = useState(false);
  const [searched, setSearched]     = useState(false);
  const [usedId, setUsedId]         = useState<string | null>(null);
  const debounceRef                 = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef                    = useRef<AbortController | null>(null);

  // Debounced fetch — fires 400 ms after the user stops typing
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!query.trim()) {
      setResults([]);
      setSearched(false);
      setLoading(false);
      return;
    }
    debounceRef.current = setTimeout(() => void fetchResults(query, sources), 400);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, sources]);

  async function fetchResults(q: string, srcs: string) {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    try {
      const params = new URLSearchParams({ q, sources: srcs, max_results: '6' });
      const res = await fetch(`${BASE_URL}/api/dataset/browse?${params}`, {
        signal: abortRef.current.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults((data.datasets as BrowseResult[]) ?? []);
      setSearched(true);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setResults([]);
        setSearched(true);
      }
    } finally {
      setLoading(false);
    }
  }

  function handleUse(result: BrowseResult) {
    // Pre-fill the agent query with the dataset name and set source
    setDatasetQuery(result.name);
    const srcMap: Record<string, 'HuggingFace' | 'OpenImages'> = {
      HuggingFace: 'HuggingFace',
    };
    setDatasetSource(srcMap[result.source] ?? 'HuggingFace');
    setUsedId(result.dataset_id);
    // Scroll to / focus the agent query input
    window.dispatchEvent(new Event('focus-dataset-query'));
  }

  const realResults = results.filter((r) => !r.unavailable);
  const unavailable = results.filter((r) => r.unavailable);

  return (
    <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 backdrop-blur-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-3.5 border-b border-slate-800/50">
        <BookOpen className="w-3.5 h-3.5 text-sky-400" />
        <span className="text-xs font-semibold text-slate-200">Dataset Browser</span>
        <span className="text-[10px] font-mono text-slate-600 ml-1">— explore before running the agent</span>
      </div>

      {/* Search bar */}
      <div className="flex gap-2 px-5 py-3 border-b border-slate-800/40">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search datasets… e.g. plant disease, COCO, face detection"
            className="w-full rounded-lg border border-slate-800 bg-slate-950/80 py-2 pl-9 pr-3 font-mono text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
          />
          {loading && (
            <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-sky-400 animate-spin" />
          )}
        </div>
        <select
          value={sources}
          onChange={(e) => setSources(e.target.value)}
          className="rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
        >
          {SOURCES.map((s) => (
            <option key={s.value} value={s.value}>{s.label}</option>
          ))}
        </select>
      </div>

      {/* Results */}
      <div className="px-5 py-4">
        {!searched && !loading && (
          <p className="text-center text-[11px] font-mono text-slate-600 py-6">
            Type to browse datasets — results appear as you type
          </p>
        )}

        {searched && realResults.length === 0 && !loading && (
          <p className="text-center text-[11px] font-mono text-slate-600 py-6">
            No results found for "{query}"
          </p>
        )}

        {realResults.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {realResults.map((r) => {
              const badge = SOURCE_BADGE[r.source] ?? 'text-slate-400 border-slate-400/30 bg-slate-400/10';
              const isUsed = usedId === r.dataset_id;
              return (
                <div
                  key={`${r.source}-${r.dataset_id}`}
                  className="flex flex-col rounded-xl border border-slate-800/60 bg-slate-950/50 p-4 gap-3 hover:border-sky-500/30 transition-colors"
                >
                  {/* Top row */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-1.5 min-w-0">
                      <span className="text-xs font-semibold text-slate-200 truncate">{r.name}</span>
                      <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${badge}`}>
                        {r.source}
                      </span>
                    </div>
                    <a
                      href={r.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 text-slate-600 hover:text-sky-400 transition-colors"
                    >
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>

                  {/* Description */}
                  <p className="text-[11px] text-slate-500 leading-relaxed line-clamp-2 flex-1">
                    {r.description || 'No description available.'}
                  </p>

                  {/* Tags */}
                  {r.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {r.tags.slice(0, 4).map((tag) => (
                        <span key={tag} className="rounded px-1.5 py-0.5 bg-slate-800/60 text-[9px] font-mono text-slate-500">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Stats + action */}
                  <div className="flex items-center justify-between gap-2 pt-1 border-t border-slate-800/40">
                    <div className="flex items-center gap-3 text-[10px] font-mono text-slate-600">
                      {r.size_estimate && r.size_estimate !== 'unknown' && (
                        <span>{r.size_estimate}</span>
                      )}
                      {r.downloads > 0 && (
                        <span className="flex items-center gap-1">
                          <Download className="w-2.5 h-2.5" />{fmt(r.downloads)}
                        </span>
                      )}
                      {r.likes > 0 && (
                        <span className="flex items-center gap-1">
                          <ThumbsUp className="w-2.5 h-2.5" />{fmt(r.likes)}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => handleUse(r)}
                      className={`flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-[10px] font-semibold transition-all ${
                        isUsed
                          ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-400'
                          : 'border-sky-500/30 bg-sky-500/10 text-sky-400 hover:bg-sky-500/20 hover:border-sky-500/50'
                      }`}
                    >
                      {isUsed ? 'Queued ✓' : <>Use This <ChevronRight className="w-3 h-3" /></>}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Unavailable source notices */}
        {unavailable.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {unavailable.map((r) => (
              <span key={r.source} className="text-[10px] font-mono text-slate-600">
                {r.source}: {r.description}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
