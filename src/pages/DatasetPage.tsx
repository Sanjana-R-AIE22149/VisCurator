import {
  Database, ScanSearch, CopyCheck, Wand2, Sparkles, ShieldCheck, BarChart3,
  AlertTriangle, CheckCircle2, Download, RefreshCw, FileText, FolderDown,
  Play, RotateCcw, ArrowRight, Layers, Cpu, TrendingUp, ImageIcon, Filter, Info,
} from 'lucide-react';
import DatasetControls from '../components/dataset/DatasetControls';
import DatasetBrowser from '../components/dataset/DatasetBrowser';
import LiveTerminal from '../components/dataset/LiveTerminal';
import BlurFilterChart from '../components/dataset/BlurFilterChart';
import StagePreview from '../components/dataset/StagePreview';
import PreprocessingReportModal from '../components/dataset/PreprocessingReportModal';
import { useAppStore } from '../store/useAppStore';
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { BASE_URL, authHeaders } from '../lib/api';

export default function DatasetPage() {
  const navigate = useNavigate();
  const {
    imagesIngested,
    qualityScore,
    classBalance,
    duplicatePercentage,
    processingPlan,
    preprocessingReport,
    jobId,
    targetSize,
    isProcessing,
  } = useAppStore();

  const [reportOpen, setReportOpen] = useState(false);
  const [processedDatasets, setProcessedDatasets] = useState<Array<{
    slug: string; dataset_id: string; class_count: number; image_count: number; has_report: boolean
  }>>([]);
  const [loadingProcessed, setLoadingProcessed] = useState(false);

  // Short-dataset augmentation state
  const [augmentPromptDismissed, setAugmentPromptDismissed] = useState(false);
  const [isAugmenting, setIsAugmenting] = useState(false);
  const [augDone, setAugDone] = useState(false);

  const fetchProcessed = async () => {
    setLoadingProcessed(true);
    try {
      const res = await fetch(`${BASE_URL}/api/dataset/processed`, { headers: authHeaders() });
      if (res.ok) setProcessedDatasets(await res.json());
    } catch { /* backend offline */ }
    finally { setLoadingProcessed(false); }
  };

  useEffect(() => { void fetchProcessed(); }, []);

  useEffect(() => {
    document.title = 'Dataset Pipeline — VisCurator';
    const handler = () => { void fetchProcessed(); };
    window.addEventListener('viscurator:pipeline-done', handler);
    return () => window.removeEventListener('viscurator:pipeline-done', handler);
  }, []);

  // Reset short-dataset banner when a new job starts
  useEffect(() => {
    if (isProcessing) {
      setAugmentPromptDismissed(false);
      setAugDone(false);
    }
  }, [isProcessing]);

  const currentSlug = preprocessingReport?.dataset_id
    ? preprocessingReport.dataset_id.replace(/\//g, '_')
    : null;

  // ── Determine if dataset is "short" ──────────────────────────────────────────
  const actualCount  = preprocessingReport?.after_stats?.images ?? null;
  const isShort      = actualCount !== null && actualCount < targetSize;
  const deficit      = isShort ? targetSize - actualCount : 0;
  const showShortBanner = isShort && !augmentPromptDismissed && !isAugmenting && !augDone && !isProcessing;

  // ── Inline augmentation trigger ───────────────────────────────────────────────
  const handleAugmentShortDataset = async () => {
    if (!jobId && !currentSlug) return;
    setIsAugmenting(true);
    setAugmentPromptDismissed(true);
    try {
      const token = localStorage.getItem('viscurator_token') || '';
      // Use augmentation agent — pick multiplier to reach targetSize
      const multiplier = Math.ceil(targetSize / Math.max(actualCount ?? 1, 1)) + 1;
      const body = {
        job_id: jobId ?? `slug_${currentSlug}`,
        strategy: 'medium',
        multiplier,
        target_size: targetSize,
        target_px: 224,
        balance: true,
        max_workers: 4,
      };
      await fetch(`${BASE_URL}/api/dataset/augment-agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(body),
      });
      // Poll for a bit then mark done
      setTimeout(() => { setIsAugmenting(false); setAugDone(true); void fetchProcessed(); }, 4000);
    } catch {
      setIsAugmenting(false);
    }
  };

  // ── Blur / dup stats ─────────────────────────────────────────────────────────
  const blurFiltered = preprocessingReport?.after_stats?.blur_filtered ?? 0;
  const dupRemoved   = preprocessingReport?.after_stats?.duplicates_removed ?? 0;
  const totalBefore  = preprocessingReport?.before_stats?.images ?? 0;
  const blurPct      = totalBefore > 0 ? ((blurFiltered / totalBefore) * 100).toFixed(1) : '0';
  const dupPct       = totalBefore > 0 ? ((dupRemoved / totalBefore) * 100).toFixed(1) : '0';

  // ── Filtered samples for blur grid ───────────────────────────────────────────
  const filteredSamples = preprocessingReport?.stage_samples?.filtered ?? [];
  const annotationSummary = preprocessingReport?.annotation_summary;
  const deblurPreview = preprocessingReport?.deblur_preview ?? preprocessingReport?.stage_samples?.recovered ?? [];
  const deblurSummary = preprocessingReport?.deblur_summary;

  // ── Class distribution bars ───────────────────────────────────────────────────
  const classDist = preprocessingReport?.class_distribution ?? {};
  const classMax  = Math.max(...Object.values(classDist), 1);

  const statCards = [
    {
      label: 'Images Exported',
      value: imagesIngested !== null ? imagesIngested.toLocaleString() : '—',
      icon: Database,
      accent: 'text-teal-400',
      sub: targetSize ? `of ${targetSize.toLocaleString()} requested` : '',
    },
    {
      label: 'Blur Rejected',
      value: preprocessingReport ? blurFiltered.toLocaleString() : (qualityScore !== null ? '—' : '—'),
      icon: ScanSearch,
      accent: 'text-amber-400',
      sub: preprocessingReport ? `${blurPct}% of raw` : '',
    },
    {
      label: 'Duplicates',
      value: preprocessingReport ? dupRemoved.toLocaleString() : (duplicatePercentage !== null ? `${(duplicatePercentage ?? 0).toFixed(1)}%` : '—'),
      icon: CopyCheck,
      accent: 'text-rose-400',
      sub: preprocessingReport ? `${dupPct}% of raw` : 'estimated',
    },
    {
      label: 'Class Balance',
      value: classBalance ?? '—',
      icon: BarChart3,
      accent: 'text-emerald-400',
      sub: Object.keys(classDist).length > 0 ? `${Object.keys(classDist).length} classes` : '',
    },
    {
      label: 'Annotations',
      value: annotationSummary?.available ? `${annotationSummary.count?.toLocaleString() ?? 0}` : 'Pending',
      icon: ShieldCheck,
      accent: annotationSummary?.available ? 'text-violet-400' : 'text-slate-400',
      sub: annotationSummary?.available ? 'SAM boxes ready' : (annotationSummary?.error ? 'SAM unavailable' : ''),
    },
  ];

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fade-up">

        {/* ── Header ── */}
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px flex-1 bg-gradient-to-r from-teal-500/40 to-transparent" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100">
            Dataset Pipeline
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Search → detect blur &amp; duplicates → auto-annotate &amp; augment → download.
            Set your target image count and the pipeline handles the rest.
          </p>
        </div>

        {/* ── Search + Controls ── */}
        <DatasetControls />
        <DatasetBrowser />

        {/* ── Stat Cards (live during + after pipeline) ── */}
        {(imagesIngested !== null || preprocessingReport) && (
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
            {statCards.map(({ label, value, icon: Icon, accent, sub }) => (
              <div
                key={label}
                className="flex items-center gap-3 px-4 py-3 rounded-xl border border-slate-800/50 bg-slate-900/20"
              >
                <div className={`p-2 rounded-lg bg-slate-800/50 ${accent}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-widest text-slate-500">{label}</p>
                  <p className="text-lg font-bold text-slate-200 font-mono">{value}</p>
                  {sub && <p className="text-[9px] text-slate-600">{sub}</p>}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── SHORT DATASET BANNER ── */}
        {showShortBanner && (
          <div className="rounded-xl border border-amber-500/40 bg-gradient-to-r from-amber-500/10 to-amber-500/5 p-5">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center shrink-0 mt-0.5">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-bold text-amber-300">
                  Dataset is shorter than requested
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Only <span className="text-amber-300 font-bold font-mono">{actualCount?.toLocaleString()}</span> images
                  available after filtering — you asked for{' '}
                  <span className="text-white font-bold font-mono">{targetSize.toLocaleString()}</span>
                  {' '}({deficit.toLocaleString()} short).
                  Want to augment with rotations, flips &amp; brightness changes to reach your target?
                </p>
                <div className="flex gap-3 mt-4">
                  <button
                    id="btn-augment-short"
                    onClick={handleAugmentShortDataset}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-black text-sm font-bold transition-all shadow-lg shadow-amber-500/25"
                  >
                    <Sparkles className="w-4 h-4" />
                    Yes, augment to {targetSize.toLocaleString()}
                  </button>
                  <button
                    onClick={() => setAugmentPromptDismissed(true)}
                    className="px-4 py-2 rounded-lg border border-slate-700 bg-slate-800/60 text-slate-400 hover:text-slate-200 text-sm transition-all"
                  >
                    No, keep {actualCount?.toLocaleString()}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Augmenting in progress */}
        {isAugmenting && (
          <div className="flex items-center gap-3 rounded-xl border border-violet-500/30 bg-violet-500/5 px-5 py-4">
            <RefreshCw className="w-5 h-5 text-violet-400 animate-spin shrink-0" />
            <div>
              <p className="text-sm font-semibold text-violet-300">Augmentation running…</p>
              <p className="text-xs text-slate-500 mt-0.5">
                Generating rotations, flips, brightness variants. This may take a minute.
              </p>
            </div>
          </div>
        )}

        {/* Augmentation done */}
        {augDone && (
          <div className="flex items-center gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-5 py-4">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <p className="text-sm font-semibold text-emerald-300">
              Augmentation complete — refresh the Processed Datasets panel to see the updated count.
            </p>
            <button
              onClick={() => void fetchProcessed()}
              className="ml-auto flex items-center gap-1.5 text-[11px] text-emerald-400 hover:text-emerald-300 shrink-0"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Refresh
            </button>
          </div>
        )}

        {/* ── Blur Detection Grid ── */}
        {filteredSamples.length > 0 && (
          <div className="rounded-xl border border-amber-500/25 bg-slate-900/30 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ScanSearch className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-semibold text-slate-200">Blur &amp; Duplicate Detection</h2>
              </div>
              <div className="flex gap-3 text-[10px]">
                <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-amber-300 font-semibold">
                  {blurFiltered} blurry
                </span>
                <span className="rounded-full border border-rose-500/30 bg-rose-500/10 px-2.5 py-1 text-rose-300 font-semibold">
                  {dupRemoved} duplicates
                </span>
              </div>
            </div>
            <p className="text-[11px] text-slate-500">
              Preview of rejected images so you can inspect what was filtered out for blur or duplication.
            </p>
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-2">
              {filteredSamples.slice(0, 24).map((s, i) => {
                const baseUrl = `${BASE_URL}/data/${preprocessingReport?.dataset_id?.replace(/\//g, '_')}`;
                return (
                  <div
                    key={i}
                    className="group relative rounded-lg overflow-hidden border border-amber-500/20 bg-slate-950"
                  >
                    <img
                      src={`${baseUrl}/${s.url}`}
                      alt={s.label}
                      className="w-full aspect-square object-cover opacity-70 group-hover:opacity-100 transition-opacity"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src =
                          'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><rect fill="%231e293b" width="80" height="80"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="%23475569" font-size="8">blur</text></svg>';
                      }}
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-1.5 py-1">
                      <p className="text-[8px] font-bold text-amber-300 uppercase tracking-wide">{s.reason}</p>
                    </div>
                  </div>
                );
              })}
            </div>
            {filteredSamples.length > 24 && (
              <p className="text-[10px] text-slate-500">
                Showing 24 of {filteredSamples.length} rejected previews in this panel. The full sample set is available in the stage preview and JSON report.
              </p>
            )}
          </div>
        )}

        {deblurPreview.length > 0 && (
          <div className="rounded-xl border border-sky-500/25 bg-slate-900/30 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Wand2 className="w-4 h-4 text-sky-400" />
                <h2 className="text-sm font-semibold text-slate-200">Deblur Comparison</h2>
              </div>
              {deblurSummary && (
                <div className="flex gap-3 text-[10px]">
                  <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-1 text-sky-300 font-semibold">
                    recovered {deblurSummary.recovered_for_export ?? 0}
                  </span>
                  <span className="rounded-full border border-slate-700/60 bg-slate-900/60 px-2.5 py-1 text-slate-300 font-semibold">
                    blur {deblurSummary.avg_before ?? 0} → {deblurSummary.avg_after ?? 0}
                  </span>
                </div>
              )}
            </div>
            <p className="text-[11px] text-slate-500">
              Preview of blurry rejects and their recovered versions. Recovered images can be folded back into the curated export when the target count is short.
            </p>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {deblurPreview.slice(0, 8).map((s, i) => {
                const baseUrl = `${BASE_URL}/data/${preprocessingReport?.dataset_id?.replace(/\//g, '_')}`;
                return (
                  <div key={i} className="rounded-xl border border-slate-800/60 bg-slate-950/40 p-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <p className="text-[10px] uppercase tracking-widest text-rose-300">Before</p>
                        <img
                          src={`${baseUrl}/${s.before_url}`}
                          alt={`${s.label} before deblur`}
                          className="w-full aspect-square object-cover rounded-lg border border-rose-500/20"
                        />
                        <p className="text-[10px] text-slate-500">Laplacian {s.before_blur}</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-[10px] uppercase tracking-widest text-emerald-300">After</p>
                        <img
                          src={`${baseUrl}/${s.after_url}`}
                          alt={`${s.label} after deblur`}
                          className="w-full aspect-square object-cover rounded-lg border border-emerald-500/20"
                        />
                        <p className="text-[10px] text-slate-500">Laplacian {s.after_blur}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Class Distribution Bars ── */}
        {Object.keys(classDist).length > 0 && (
          <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-5 space-y-3">
            <div className="flex items-center gap-2 mb-1">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-slate-200">Class Distribution</h2>
              <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                classBalance === 'Balanced'
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                  : classBalance === 'Moderately Imbalanced'
                  ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                  : 'border-rose-500/30 bg-rose-500/10 text-rose-300'
              }`}>
                {classBalance ?? 'Unknown'}
              </span>
            </div>
            <div className="space-y-2">
              {Object.entries(classDist)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, count]) => (
                  <div key={cls} className="flex items-center gap-3">
                    <p className="text-[11px] font-mono text-slate-300 w-32 truncate shrink-0">{cls}</p>
                    <div className="flex-1 h-4 rounded-full bg-slate-800/60 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-teal-500 to-teal-400 transition-all duration-700"
                        style={{ width: `${(count / classMax) * 100}%` }}
                      />
                    </div>
                    <p className="text-[11px] font-bold font-mono text-teal-300 w-10 text-right shrink-0">
                      {count.toLocaleString()}
                    </p>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* ── Preprocessing Recommendation ── */}
        {processingPlan && (
          <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-5 space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-200">Preprocessing Plan</h2>
                <p className="text-[11px] text-slate-500 mt-1">Auto-generated by the CVAgent planner.</p>
              </div>
              <p className="text-sm font-semibold text-teal-300 shrink-0">{processingPlan.recommended_model}</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                {
                  label: 'Blur Filtering',
                  value: processingPlan.needs_blur_filtering ? 'Enabled' : 'Not needed',
                  icon: ScanSearch,
                  accent: 'text-sky-400',
                },
                {
                  label: 'Deduplication',
                  value: processingPlan.needs_deduplication
                    ? `${(duplicatePercentage ?? 0).toFixed(1)}% estimated dupes`
                    : 'Low duplicate risk',
                  icon: CopyCheck,
                  accent: 'text-emerald-400',
                },
                {
                  label: 'Augmentation',
                  value: processingPlan.needs_augmentation
                    ? processingPlan.recommended_augmentations.join(', ')
                    : 'Not required',
                  icon: Wand2,
                  accent: 'text-violet-400',
                },
              ].map(({ label, value, icon: Icon, accent }) => (
                <div key={label} className="rounded-lg border border-slate-800/50 bg-slate-950/50 p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className={`w-4 h-4 ${accent}`} />
                    <p className="text-[10px] uppercase tracking-widest text-slate-500">{label}</p>
                  </div>
                  <p className="text-xs text-slate-300">{value}</p>
                </div>
              ))}
            </div>
            <div className="rounded-lg border border-teal-500/15 bg-teal-500/5 p-4 flex gap-2">
              <Info className="w-3.5 h-3.5 text-teal-400 shrink-0 mt-0.5" />
              <p className="text-xs text-teal-200">{processingPlan.reasoning}</p>
            </div>

            {/* Before / After / Filtered counts */}
            {preprocessingReport && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: 'Raw Input', value: totalBefore, color: 'text-slate-300' },
                  { label: 'Exported', value: actualCount ?? 0, color: 'text-teal-300' },
                  {
                    label: 'Filtered Out',
                    value: blurFiltered + dupRemoved,
                    color: 'text-amber-300',
                    sub: `blur: ${blurFiltered} | dupes: ${dupRemoved}`,
                  },
                ].map(({ label, value, color, sub }) => (
                  <div key={label} className="rounded-lg border border-slate-800/50 bg-slate-950/50 p-4">
                    <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">{label}</p>
                    <p className={`text-lg font-mono font-bold ${color}`}>{value.toLocaleString()}</p>
                    {sub && <p className="text-[10px] text-slate-600 mt-1">{sub}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Stage Sample Preview ── */}
        <StagePreview />

        {/* ── Terminal + Blur Chart ── */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <LiveTerminal />
          <BlurFilterChart />
        </div>

        {/* ── Download bar (after active job) ── */}
        {preprocessingReport && jobId && currentSlug && (
          <div className="flex items-center gap-3 rounded-xl border border-teal-500/20 bg-teal-500/5 px-4 py-3">
            <div className="flex-1">
              <p className="text-xs font-semibold text-teal-300">Pipeline complete — ready to download</p>
              {preprocessingReport.output_dir && (
                <p className="text-[10px] font-mono text-slate-500 mt-0.5 truncate">{preprocessingReport.output_dir}</p>
              )}
            </div>
            <div className="flex gap-2 flex-wrap">
              {(['zip', 'coco', 'yolo'] as const).map((fmt) => (
                <a
                  key={fmt}
                  href={`${BASE_URL}/api/dataset/download/${currentSlug}?format=${fmt}`}
                  download
                  className="flex items-center gap-1.5 rounded-lg border border-teal-500/30 bg-teal-500/15 px-3 py-2 text-xs font-semibold text-teal-400 hover:bg-teal-500/25 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  {fmt.toUpperCase()}
                </a>
              ))}
              <button
                onClick={() => setReportOpen(true)}
                className="flex items-center gap-1.5 rounded-lg border border-teal-500/30 bg-teal-500/15 px-3 py-2 text-xs font-semibold text-teal-400 hover:bg-teal-500/25 transition-colors"
              >
                <FileText className="w-3.5 h-3.5" />
                Report
              </button>
            </div>
          </div>
        )}

        {/* ── Go to Builder CTA (after pipeline) ── */}
        {preprocessingReport && !isProcessing && (
          <div className="flex items-center gap-4 rounded-xl border border-violet-500/30 bg-gradient-to-r from-violet-500/10 to-violet-500/5 px-5 py-4">
            <div className="w-9 h-9 rounded-xl bg-violet-500/20 border border-violet-500/30 flex items-center justify-center shrink-0">
              <Cpu className="w-4 h-4 text-violet-400" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-violet-300">Dataset ready — design your model</p>
              <p className="text-xs text-slate-500 mt-0.5">
                Drag &amp; drop layers in the Builder, compile PyTorch code, and launch training.
              </p>
            </div>
            <button
              id="cta-go-builder"
              onClick={() => navigate('/builder')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold transition-all shadow-lg shadow-violet-500/20 shrink-0"
            >
              Builder <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* ── Persistent Processed Datasets ── */}
        {processedDatasets.length > 0 && (
          <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FolderDown className="w-4 h-4 text-teal-400" />
                <p className="text-xs font-semibold text-slate-300">Processed Datasets</p>
              </div>
              <button
                onClick={() => void fetchProcessed()}
                disabled={loadingProcessed}
                className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
              >
                <RefreshCw className={`w-3 h-3 ${loadingProcessed ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
            <div className="space-y-2">
              {processedDatasets.map((ds) => (
                <div key={ds.slug} className="flex items-center justify-between rounded-lg border border-slate-800/40 bg-slate-950/50 px-4 py-2.5 gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-mono text-slate-300 truncate">{ds.dataset_id}</p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      {ds.class_count} classes · {ds.image_count.toLocaleString()} images
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {(['zip', 'coco', 'yolo'] as const).map((fmt) => (
                      <a
                        key={fmt}
                        href={`${BASE_URL}/api/dataset/download/${ds.slug}?format=${fmt}`}
                        download
                        className="rounded border border-slate-700/60 bg-slate-800/60 px-2 py-1 text-[10px] font-mono text-slate-300 hover:bg-slate-700/60 hover:text-slate-100 transition-colors"
                      >
                        {fmt.toUpperCase()}
                      </a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>

      {reportOpen && jobId && (
        <PreprocessingReportModal jobId={jobId} onClose={() => setReportOpen(false)} />
      )}
    </div>
  );
}
