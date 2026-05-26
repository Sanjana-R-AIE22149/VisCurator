import { BarChart3, CopyCheck, Database, FileText, FolderDown, ScanSearch, ShieldCheck, Sparkles, Wand2, Download, RefreshCw, ArrowRight, Layers } from 'lucide-react';
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

import AnnotationSeedUI from '../components/dataset/AnnotationSeedUI';

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
  } = useAppStore();

  const [reportOpen, setReportOpen] = useState(false);
  const [processedDatasets, setProcessedDatasets] = useState<Array<{ slug: string; dataset_id: string; class_count: number; image_count: number; has_report: boolean }>>([]);
  const [loadingProcessed, setLoadingProcessed] = useState(false);

  // Load available processed datasets (persists across page refreshes)
  const fetchProcessed = async () => {
    setLoadingProcessed(true);
    try {
      const res = await fetch(`${BASE_URL}/api/dataset/processed`, { headers: authHeaders() });
      if (res.ok) setProcessedDatasets(await res.json());
    } catch { /* backend offline */ }
    finally { setLoadingProcessed(false); }
  };

  useEffect(() => { void fetchProcessed(); }, []);

  // Auto-refresh when pipeline finishes
  useEffect(() => {
    document.title = 'Datasets — VisCurator';
    const handler = () => { void fetchProcessed(); };
    window.addEventListener('viscurator:pipeline-done', handler);
    return () => window.removeEventListener('viscurator:pipeline-done', handler);
  }, []);

  // Recompute slug from current preprocessingReport
  const currentSlug = preprocessingReport?.dataset_id
    ? preprocessingReport.dataset_id.replace(/\//g, '_')
    : null;

  const stats = [
    {
      label: 'Images Ingested',
      value: imagesIngested !== null ? imagesIngested.toLocaleString() : '—',
      icon: Database,
      accent: 'text-teal-400',
    },
    {
      label: 'Augmented',
      value: preprocessingReport?.augmentation_summary?.augmented_images
        ? preprocessingReport.augmentation_summary.augmented_images.toLocaleString()
        : '—',
      icon: Sparkles,
      accent: 'text-purple-400',
    },
    {
      label: 'Quality Score',
      value: qualityScore !== null ? qualityScore.toFixed(1) : '—',
      icon: ShieldCheck,
      accent: 'text-emerald-400',
    },
    {
      label: 'Class Balance',
      value: classBalance ?? '—',
      icon: BarChart3,
      accent: 'text-amber-400',
    },
  ];

  const recommendationCards = processingPlan
    ? [
        {
          label: 'Blur Filtering',
          value: processingPlan.needs_blur_filtering ? 'Enabled' : 'Skip',
          icon: ScanSearch,
          accent: 'text-sky-400',
        },
        {
          label: 'Deduplication',
          value: processingPlan.needs_deduplication ? `${(duplicatePercentage ?? 0).toFixed(1)}% estimated duplicates` : 'Low duplicate risk',
          icon: CopyCheck,
          accent: 'text-emerald-400',
        },
        {
          label: 'Augmentation',
          value: processingPlan.needs_augmentation ? processingPlan.recommended_augmentations.join(', ') : 'Not required',
          icon: Wand2,
          accent: 'text-violet-400',
        },
      ]
    : [];

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fade-up">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px flex-1 bg-gradient-to-r from-teal-500/40 to-transparent" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100">
            Autonomous Data Ingestor
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Describe your task — CVAgent searches HuggingFace, Kaggle, Roboflow & more, recommends preprocessing, and exports a processed dataset.
          </p>
        </div>

        {/* ── Next Step CTA (after fetch completes) ── */}
        {preprocessingReport && jobId && (
          <div className="flex items-center gap-4 rounded-xl border border-violet-500/30 bg-gradient-to-r from-violet-500/10 to-violet-500/5 px-5 py-4">
            <div className="w-9 h-9 rounded-xl bg-violet-500/20 border border-violet-500/30 flex items-center justify-center shrink-0">
              <Layers className="w-4 h-4 text-violet-400" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-violet-300">Step 1 complete — dataset fetched & cleaned</p>
              <p className="text-xs text-slate-500 mt-0.5">
                {imagesIngested ? `${imagesIngested.toLocaleString()} images ready.` : 'Dataset ready.'} Continue to define classes and auto-annotate with SAM+CLIP.
              </p>
            </div>
            <button
              id="cta-go-annotate"
              onClick={() => navigate('/annotator')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold transition-all shadow-lg shadow-violet-500/20 shrink-0"
            >
              Annotate <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {stats.map(({ label, value, icon: Icon, accent }) => (
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
              </div>
            </div>
          ))}
        </div>

        {/* Persistent download section for all processed datasets */}
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
                    <p className="text-[10px] text-slate-500 mt-0.5">{ds.class_count} classes · {ds.image_count.toLocaleString()} images</p>
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

        {/* Inline download section after active curation job */}
        {preprocessingReport && jobId && currentSlug && (
          <div className="flex items-center gap-3 rounded-xl border border-teal-500/20 bg-teal-500/5 px-4 py-3">
            <div className="flex-1">
              <p className="text-xs font-semibold text-teal-300">Preprocessing complete — ready to download</p>
              {preprocessingReport.output_dir && (
                <p className="text-[10px] font-mono text-slate-500 mt-0.5 truncate">{preprocessingReport.output_dir}</p>
              )}
            </div>
            <div className="flex gap-2">
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
                View Report
              </button>
            </div>
          </div>
        )}

        <DatasetBrowser />

        <DatasetControls />
        <AnnotationSeedUI />

        {processingPlan && (
          <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-5 space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-200">Preprocessing Recommendation</h2>
                <p className="text-[11px] text-slate-500 mt-1">
                  Deterministic planner output from the backend tools.
                </p>
              </div>
              <div className="text-right">
                <p className="text-[10px] uppercase tracking-widest text-slate-500">Recommended Model</p>
                <p className="text-sm font-semibold text-teal-300">{processingPlan.recommended_model}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {recommendationCards.map(({ label, value, icon: Icon, accent }) => (
                <div key={label} className="rounded-lg border border-slate-800/50 bg-slate-950/50 p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className={`w-4 h-4 ${accent}`} />
                    <p className="text-[10px] uppercase tracking-widest text-slate-500">{label}</p>
                  </div>
                  <p className="text-xs text-slate-300">{value}</p>
                </div>
              ))}
            </div>

            <div className="rounded-lg border border-teal-500/15 bg-teal-500/5 p-4">
              <p className="text-xs text-teal-200">{processingPlan.reasoning}</p>
            </div>

            {preprocessingReport && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="rounded-lg border border-slate-800/50 bg-slate-950/50 p-4">
                  <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Before</p>
                  <p className="text-lg font-mono font-bold text-slate-200">
                    {preprocessingReport.before_stats?.images?.toLocaleString() ?? '—'}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1">images before processing</p>
                </div>
                <div className="rounded-lg border border-slate-800/50 bg-slate-950/50 p-4">
                  <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">After</p>
                  <p className="text-lg font-mono font-bold text-slate-200">
                    {preprocessingReport.after_stats?.images?.toLocaleString() ?? '—'}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1">processed export images</p>
                </div>
                <div className="rounded-lg border border-slate-800/50 bg-slate-950/50 p-4">
                  <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Filtered</p>
                  <p className="text-lg font-mono font-bold text-slate-200">
                    {(preprocessingReport.after_stats?.blur_filtered ?? 0) + (preprocessingReport.after_stats?.duplicates_removed ?? 0)}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    blur: {preprocessingReport.after_stats?.blur_filtered ?? 0} | dupes: {preprocessingReport.after_stats?.duplicates_removed ?? 0}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        <StagePreview />

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <LiveTerminal />
          <BlurFilterChart />
        </div>
      </div>

      {reportOpen && jobId && (
        <PreprocessingReportModal jobId={jobId} onClose={() => setReportOpen(false)} />
      )}
    </div>
  );
}
