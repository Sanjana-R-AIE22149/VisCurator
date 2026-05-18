import { useState, useEffect, useRef } from 'react';
import {
  Upload, X, Zap, ImageIcon,
  Sparkles, AlertTriangle, CheckCircle2,
  Play, RefreshCw, Layers, Wand2, Filter,
  TrendingUp, AlertCircle, Download,
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import {
  uploadSeedClass, startAnnotation, startAugmentation,
  connectPipelineWebSocket, BASE_URL, downloadAugmentedDataset,
} from '../lib/api';
import type { PipelineMessage, AnnotationReport } from '../lib/api';


interface LogLine {
  id: string;
  text: string;
  kind: 'info' | 'ok' | 'error' | 'progress';
}

interface AugReport {
  stats: { recovered: number; augmented: number; failed: number;
           avg_blur_before: number; avg_blur_after: number };
  total_input: number;
  n_aug: number;
}

/* ══════════════════════════════════════════════════════════════════════════ */
export default function AnnotatorPage() {
  const { localDataset, jobId } = useAppStore();

  /* seed management */
  const [classes, setClasses]       = useState<string[]>([]);
  const [newClass, setNewClass]     = useState('');
  const [seedCounts, setSeedCounts] = useState<Record<string, number>>({});
  const [uploading, setUploading]   = useState<Record<string, boolean>>({});
  const [dragOver, setDragOver]     = useState<string | null>(null);

  /* pipeline state */
  const [phase, setPhase]  = useState<'idle' | 'annotating' | 'done' | 'augmenting' | 'aug_done'>('idle');
  const [logs, setLogs]    = useState<LogLine[]>([]);
  const [progress, setProgress] = useState(0);
  const [blurStats, setBlurStats] = useState<{ rejected: number; total: number } | null>(null);

  /* annotation settings */
  const [minConfidence, setMinConfidence] = useState(0.30);
  const [blurThreshold, setBlurThreshold] = useState(80);

  /* annotation report (preview grid) */
  const [annoReport, setAnnoReport] = useState<AnnotationReport | null>(null);
  const [previewTab, setPreviewTab] = useState<'annotated' | 'low_confidence'>('annotated');

  /* augmenter settings */
  const [nAug, setNAug]             = useState(4);
  const [targetSize, setTargetSize]  = useState(224);
  const [augReport, setAugReport]    = useState<AugReport | null>(null);

  const logsEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  const pushLog = (text: string, kind: LogLine['kind'] = 'info') =>
    setLogs(prev => [...prev.slice(-200), { id: `${Date.now()}-${Math.random()}`, text, kind }]);

  /* ── WS listener (reused for both annotation + augmentation) ── */
  const wsRef = useRef<ReturnType<typeof connectPipelineWebSocket> | null>(null);

  const attachWS = (jid: string, onDone: () => void) => {
    wsRef.current?.close();
    wsRef.current = connectPipelineWebSocket(
      jid,
      (msg: PipelineMessage) => {
        if (msg.type === 'done') {
          pushLog(msg.message, 'ok');
          // annotation_report embedded in completed event
          const d = msg.data as Record<string, unknown> | undefined;
          const ar = d?.annotation_report as AnnotationReport | undefined;
          if (ar) setAnnoReport(ar);
          // blur stats from preprocessing_report
          const report = d?.preprocessing_report as Record<string, unknown> | undefined;
          if (report) {
            const after = report.after_stats as Record<string, number> | undefined;
            if (after) setBlurStats({ rejected: after.blur_filtered ?? 0, total: after.images ?? 0 });
          }
          // augmentation_report
          const augRep = d?.augmentation_report as AugReport | undefined;
          if (augRep) setAugReport(augRep);
          onDone();
        } else if (msg.type === 'error') {
          pushLog(msg.message, 'error');
        } else if (msg.type === 'log' || msg.type === 'script_log') {
          pushLog(msg.message, 'info');
          const d = msg.data as Record<string, unknown> | undefined;
          if (typeof d?.progress === 'number') setProgress(d.progress as number);
        } else if (msg.type === 'tool_result') {
          const d = msg.data as Record<string, unknown> | undefined;
          const augRep2 = d?.augmentation_report as AugReport | undefined;
          if (augRep2) setAugReport(augRep2);
        }
      },
      () => {},
    );
  };

  /* ── class management ── */
  const addClass = () => {
    const cls = newClass.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_');
    if (!cls || classes.includes(cls)) return;
    setClasses(p => [...p, cls]);
    setNewClass('');
  };
  const removeClass = (cls: string) => {
    setClasses(p => p.filter(c => c !== cls));
    setSeedCounts(p => { const n = { ...p }; delete n[cls]; return n; });
  };

  const handleDrop = async (files: FileList | File[], cls: string) => {
    if (!jobId) return;
    setUploading(p => ({ ...p, [cls]: true }));
    try {
      await uploadSeedClass(jobId, cls, files);
      setSeedCounts(p => ({ ...p, [cls]: (p[cls] || 0) + files.length }));
      pushLog(`Uploaded ${files.length} seed(s) for "${cls}"`, 'ok');
    } catch (e) {
      pushLog(`Seed upload failed: ${e instanceof Error ? e.message : e}`, 'error');
    } finally {
      setUploading(p => ({ ...p, [cls]: false }));
    }
  };

  /* ── run annotation ── */
  const runAnnotation = async () => {
    if (!jobId || classes.length === 0) return;
    setPhase('annotating');
    setProgress(0);
    setLogs([]);
    setAnnoReport(null);
    pushLog(`Launching SAM + CLIP annotation (confidence ≥ ${(minConfidence * 100).toFixed(0)}%, blur threshold ${blurThreshold})…`);
    attachWS(jobId, () => setPhase('done'));
    try {
      await startAnnotation(jobId, minConfidence, blurThreshold);
    } catch (e) {
      pushLog(`Error: ${e instanceof Error ? e.message : e}`, 'error');
      setPhase('idle');
    }
  };

  /* ── download augmented ZIP ── */
  const [downloading, setDownloading] = useState(false);
  const handleDownload = async () => {
    if (!jobId) return;
    setDownloading(true);
    try {
      await downloadAugmentedDataset(jobId);
    } catch (e) {
      pushLog(`Download failed: ${e instanceof Error ? e.message : e}`, 'error');
    } finally {
      setDownloading(false);
    }
  };

  /* ── run augmentation ── */
  const runAugmentation = async () => {
    if (!jobId) return;
    setPhase('augmenting');
    setProgress(0);
    pushLog(`Launching Anti-Blur + Augmentation (×${nAug}, ${targetSize}px)…`);
    attachWS(jobId, () => setPhase('aug_done'));
    try {
      await startAugmentation(jobId, { nAug, targetSize });
    } catch (e) {
      pushLog(`Error: ${e instanceof Error ? e.message : e}`, 'error');
      setPhase('done');
    }
  };

  const noJob = !localDataset || !jobId;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fade-up">

        {/* Header */}
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px flex-1 bg-gradient-to-r from-violet-500/40 to-transparent" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
            <Layers className="w-6 h-6 text-violet-400" />
            Annotator &amp; Augmenter
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Define classes → upload seeds → auto-annotate with SAM+CLIP → recover &amp; augment blur-rejected images.
          </p>
        </div>

        {noJob && (
          <div className="flex items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 px-5 py-4">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
            <p className="text-sm text-amber-300">
              Upload a dataset ZIP on the <strong>Dataset</strong> page first to enable annotation.
            </p>
          </div>
        )}

        {/* Two-column layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

          {/* ── LEFT: Seed Manager ── */}
          <div className="space-y-4">
            <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-bold text-violet-300 flex items-center gap-2">
                    <Zap className="h-4 w-4" /> Define Classes &amp; Upload Seeds
                  </h2>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Add classes then drop seed images or ZIPs into each bucket.
                  </p>
                </div>
                <button
                  id="btn-run-annotation"
                  onClick={runAnnotation}
                  disabled={noJob || classes.length === 0 || phase === 'annotating' || phase === 'augmenting'}
                  className="flex items-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500
                             px-4 py-2 text-xs font-bold text-white transition-colors disabled:opacity-40"
                >
                  {phase === 'annotating' ? (
                    <><RefreshCw className="h-3.5 w-3.5 animate-spin" /> Annotating…</>
                  ) : (
                    <><Play className="h-3.5 w-3.5" /> Annotate</>
                  )}
                </button>
              </div>

              {/* Confidence threshold slider */}
              <div className="mb-4 rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3 space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] uppercase tracking-widest text-slate-500 flex items-center gap-1.5">
                    <Filter className="h-3 w-3" /> Min Confidence Threshold
                  </label>
                  <span className={`text-xs font-bold font-mono ${
                    minConfidence >= 0.6 ? 'text-emerald-400' :
                    minConfidence >= 0.35 ? 'text-amber-400' : 'text-rose-400'
                  }`}>
                    {(minConfidence * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  id="slider-min-confidence"
                  type="range" min={0.05} max={0.90} step={0.05}
                  value={minConfidence}
                  onChange={e => setMinConfidence(Number(e.target.value))}
                  disabled={phase === 'annotating'}
                  className="w-full accent-violet-500 disabled:opacity-40"
                />
                <p className="text-[10px] text-slate-600">
                  Images with CLIP similarity below this go to <span className="font-mono text-slate-400">low_confidence/</span> for review instead of the training set.
                </p>
              </div>

              {/* Blur threshold slider */}
              <div className="mb-4 rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3 space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] uppercase tracking-widest text-slate-500 flex items-center gap-1.5">
                    <Filter className="h-3 w-3" /> Blur Rejection Threshold
                  </label>
                  <div className="flex items-center gap-1.5">
                    <span className={`text-xs font-bold font-mono ${
                      blurThreshold <= 30  ? 'text-rose-400'   :
                      blurThreshold <= 100 ? 'text-amber-400'  : 'text-emerald-400'
                    }`}>
                      {blurThreshold}
                    </span>
                    <span className="text-[9px] text-slate-600">Laplacian var.</span>
                  </div>
                </div>
                <input
                  id="slider-blur-threshold"
                  type="range" min={5} max={300} step={5}
                  value={blurThreshold}
                  onChange={e => setBlurThreshold(Number(e.target.value))}
                  disabled={phase === 'annotating'}
                  className="w-full accent-amber-500 disabled:opacity-40"
                />
                <div className="flex justify-between text-[9px] text-slate-700">
                  <span>Strict (5) — rejects more</span>
                  <span>Lenient (300) — keeps more</span>
                </div>
              </div>

              {/* Add class */}
              <div className="flex gap-2 mb-4">
                <input
                  id="input-new-class"
                  type="text"
                  value={newClass}
                  onChange={e => setNewClass(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && addClass()}
                  placeholder="e.g. apple_scab"
                  disabled={noJob}
                  className="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm
                             text-slate-200 focus:border-violet-500 focus:outline-none disabled:opacity-40"
                />
                <button
                  id="btn-add-class"
                  onClick={addClass}
                  disabled={noJob}
                  className="rounded bg-slate-800 hover:bg-slate-700 px-4 py-2 text-sm font-bold
                             text-slate-300 transition-colors disabled:opacity-40"
                >
                  Add
                </button>
              </div>

              {/* Seed drop zones */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {classes.map(cls => (
                  <div
                    key={cls}
                    className={`relative rounded-xl border-2 border-dashed p-4 transition-colors
                      ${dragOver === cls
                        ? 'border-violet-400 bg-violet-400/10'
                        : 'border-slate-700 bg-slate-900/40 hover:border-slate-600'}`}
                    onDragOver={e => { e.preventDefault(); setDragOver(cls); }}
                    onDragLeave={() => setDragOver(null)}
                    onDrop={e => { e.preventDefault(); setDragOver(null); void handleDrop(e.dataTransfer.files, cls); }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs font-bold text-violet-300">{cls}</span>
                      <button onClick={() => removeClass(cls)} className="text-slate-600 hover:text-rose-400">
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    <div className="flex flex-col items-center py-3 text-slate-500">
                      {uploading[cls] ? (
                        <div className="h-5 w-5 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
                      ) : (
                        <>
                          <Upload className="mb-1 h-5 w-5" />
                          <p className="text-[10px] text-center">Drop images or ZIP</p>
                          <label className="mt-2 cursor-pointer rounded border border-slate-700
                                           bg-slate-800 px-3 py-1 text-[10px] font-semibold
                                           text-slate-300 hover:bg-slate-700 transition-colors">
                            Browse
                            <input
                              type="file" multiple accept="image/*,.zip" className="hidden"
                              onChange={e => {
                                if (e.target.files && e.target.files.length > 0)
                                  void handleDrop(e.target.files, cls);
                              }}
                            />
                          </label>
                          <p className="mt-1.5 text-[10px] font-bold text-violet-400">
                            {seedCounts[cls] || 0} files
                          </p>
                        </>
                      )}
                    </div>
                  </div>
                ))}

                {classes.length === 0 && (
                  <div className="col-span-full py-8 text-center text-slate-600 border border-dashed border-slate-800 rounded-xl">
                    <ImageIcon className="mx-auto h-7 w-7 mb-2 opacity-40" />
                    <p className="text-xs">Add a class above to get started.</p>
                  </div>
                )}
              </div>
            </div>

            {/* ── Augmenter Settings ── */}
            <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 p-5 space-y-4">
              <h2 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
                <Wand2 className="h-4 w-4" /> Anti-Blur &amp; Augmenter
              </h2>
              <p className="text-[11px] text-slate-400">
                After annotation, blur-rejected images are sharpened (unsharp mask + Richardson-Lucy) and
                multiplied into augmented variants to boost dataset size.
              </p>

              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <span className="text-[10px] uppercase tracking-widest text-slate-500">Augments per image</span>
                  <input
                    id="input-n-aug"
                    type="number" min={1} max={16} value={nAug}
                    onChange={e => setNAug(Number(e.target.value))}
                    className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-3 py-1.5
                               text-sm text-slate-200 focus:border-emerald-500 focus:outline-none"
                  />
                </label>
                <label className="block">
                  <span className="text-[10px] uppercase tracking-widest text-slate-500">Target size (px)</span>
                  <select
                    id="select-target-size"
                    value={targetSize}
                    onChange={e => setTargetSize(Number(e.target.value))}
                    className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-3 py-1.5
                               text-sm text-slate-200 focus:border-emerald-500 focus:outline-none"
                  >
                    {[128, 224, 256, 320, 512].map(s => (
                      <option key={s} value={s}>{s}×{s}</option>
                    ))}
                  </select>
                </label>
              </div>

              <button
                id="btn-run-augmentation"
                onClick={runAugmentation}
                disabled={noJob || (phase !== 'done' && phase !== 'aug_done' && phase !== 'idle') }
                className="w-full flex items-center justify-center gap-2 rounded-lg
                           bg-emerald-600 hover:bg-emerald-500 px-4 py-2.5
                           text-sm font-bold text-white transition-colors disabled:opacity-40"
              >
                {phase === 'augmenting' ? (
                  <><RefreshCw className="h-4 w-4 animate-spin" /> Augmenting…</>
                ) : (
                  <><Sparkles className="h-4 w-4" /> Run Anti-Blur &amp; Augment</>
                )}
              </button>

              {blurStats && (
                <div className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-950/50 px-4 py-2.5">
                  <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
                  <div>
                    <p className="text-xs font-semibold text-amber-300">
                      {blurStats.rejected} blur-rejected image{blurStats.rejected !== 1 ? 's' : ''} detected
                    </p>
                    <p className="text-[10px] text-slate-500">Click "Run Anti-Blur &amp; Augment" to recover them.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── RIGHT: Live Terminal + Stats ── */}
          <div className="space-y-4">

            {/* Progress bar */}
            {(phase === 'annotating' || phase === 'augmenting') && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>{phase === 'annotating' ? 'Annotation' : 'Augmentation'} Progress</span>
                  <span className="font-mono text-slate-200">{progress}%</span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500
                      ${phase === 'annotating' ? 'bg-violet-500' : 'bg-emerald-500'}`}
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Augmentation report */}
            {augReport && (
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-5 space-y-3">
                <h3 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" /> Augmentation Complete
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {[
                    { label: 'Input Blurry',  value: augReport.total_input,          accent: 'text-amber-400' },
                    { label: 'Recovered',     value: augReport.stats.recovered,       accent: 'text-sky-400'  },
                    { label: 'Augmented',     value: augReport.stats.augmented,       accent: 'text-emerald-400' },
                    { label: 'Failed',        value: augReport.stats.failed,          accent: 'text-rose-400' },
                    { label: 'Blur Before',   value: augReport.stats.avg_blur_before, accent: 'text-slate-400' },
                    { label: 'Blur After',    value: augReport.stats.avg_blur_after,  accent: 'text-slate-300' },
                  ].map(({ label, value, accent }) => (
                    <div key={label} className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 text-center">
                      <p className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">{label}</p>
                      <p className={`text-lg font-bold font-mono ${accent}`}>{value}</p>
                    </div>
                  ))}
                </div>
                <p className="text-[11px] text-emerald-400/80">
                  Laplacian variance: {augReport.stats.avg_blur_before} → {augReport.stats.avg_blur_after} (higher = sharper)
                </p>
              </div>
            )}

            {/* Live terminal */}
            <div className="rounded-xl border border-slate-800 bg-slate-950 overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-800 bg-slate-900/60">
                <div className="flex gap-1.5">
                  <div className="h-2.5 w-2.5 rounded-full bg-rose-500/60" />
                  <div className="h-2.5 w-2.5 rounded-full bg-amber-500/60" />
                  <div className="h-2.5 w-2.5 rounded-full bg-emerald-500/60" />
                </div>
                <span className="text-[10px] font-mono text-slate-500 ml-1">pipeline.log</span>
                {(phase === 'annotating' || phase === 'augmenting') && (
                  <div className="ml-auto flex items-center gap-1.5 text-[10px] text-emerald-400">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    LIVE
                  </div>
                )}
              </div>
              <div className="h-80 overflow-y-auto p-4 font-mono text-[11px] space-y-1">
                {logs.length === 0 && (
                  <p className="text-slate-700 italic">Waiting for pipeline events…</p>
                )}
                {logs.map(l => (
                  <p
                    key={l.id}
                    className={
                      l.kind === 'ok'       ? 'text-emerald-400' :
                      l.kind === 'error'    ? 'text-rose-400'    :
                      l.kind === 'progress' ? 'text-sky-400'     :
                                              'text-slate-400'
                    }
                  >
                    {l.text}
                  </p>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>

            {/* Phase status badge + Download button */}
            {(phase === 'done' || phase === 'aug_done') && (
              <div className="space-y-2">
                <div className="flex items-center gap-3 rounded-xl border border-teal-500/25
                                bg-teal-500/5 px-4 py-3">
                  <CheckCircle2 className="h-5 w-5 text-teal-400 shrink-0" />
                  <div className="flex-1">
                    <p className="text-xs font-semibold text-teal-300">
                      {phase === 'aug_done' ? 'Augmentation complete!' : 'Annotation complete!'}
                    </p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      {phase === 'done'
                        ? 'You can now run Anti-Blur & Augment to recover blur-rejected images.'
                        : 'Augmented images are saved in cvagent_output/<slug>/augmented/.'}
                    </p>
                  </div>
                  {/* Download button — shown once augmentation has run */}
                  {phase === 'aug_done' && (
                    <button
                      id="btn-download-augmented"
                      onClick={handleDownload}
                      disabled={downloading}
                      className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40
                                 bg-emerald-600/20 hover:bg-emerald-600/40 px-3 py-2
                                 text-xs font-bold text-emerald-300 transition-colors disabled:opacity-50 shrink-0"
                    >
                      {downloading
                        ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" /> Packing…</>
                        : <><Download className="h-3.5 w-3.5" /> Download ZIP</>}
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Annotation Preview Grid ── */}
        {annoReport && (
          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-5 space-y-4 animate-fade-up">

            {/* Header + class distribution */}
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-violet-400" />
                  Annotation Preview
                </h2>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Min confidence used: <span className="font-mono text-violet-300">{(annoReport.min_confidence * 100).toFixed(0)}%</span>
                </p>
              </div>

              {/* Confidence distribution pills */}
              <div className="flex items-center gap-2 flex-wrap">
                {[
                  { label: 'High ≥80%',  count: annoReport.confidence_distribution.high,   color: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30' },
                  { label: 'Mid 50–80%', count: annoReport.confidence_distribution.medium, color: 'bg-amber-500/15 text-amber-300 border-amber-500/30' },
                  { label: 'Low <50%',   count: annoReport.confidence_distribution.low,    color: 'bg-rose-500/15 text-rose-300 border-rose-500/30' },
                  { label: 'Filtered',   count: annoReport.low_confidence_count,            color: 'bg-slate-700/40 text-slate-400 border-slate-600/40' },
                ].map(({ label, count, color }) => (
                  <span key={label} className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold ${color}`}>
                    {count} {label}
                  </span>
                ))}
              </div>
            </div>

            {/* Class counts bar */}
            <div className="flex flex-wrap gap-2">
              {Object.entries(annoReport.class_counts).map(([cls, count]) => (
                <div key={cls} className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-1.5">
                  <div className="h-2 w-2 rounded-full bg-violet-500" />
                  <span className="text-[10px] font-mono text-slate-300">{cls}</span>
                  <span className="text-[10px] font-bold text-violet-300 ml-1">{count}</span>
                </div>
              ))}
            </div>

            {/* Low-confidence warning */}
            {annoReport.low_confidence_count > 0 && (
              <div className="flex items-center gap-3 rounded-lg border border-amber-500/25 bg-amber-500/5 px-4 py-2.5">
                <AlertCircle className="h-4 w-4 text-amber-400 shrink-0" />
                <p className="text-xs text-amber-300">
                  <strong>{annoReport.low_confidence_count}</strong> image{annoReport.low_confidence_count !== 1 ? 's' : ''} fell below
                  the {(annoReport.min_confidence * 100).toFixed(0)}% threshold and were saved to{' '}
                  <span className="font-mono">low_confidence/</span> for manual review.
                </p>
              </div>
            )}

            {/* Tab switcher */}
            <div className="flex gap-1 rounded-lg border border-slate-800 bg-slate-950/50 p-1 w-fit">
              {(['annotated', 'low_confidence'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setPreviewTab(tab)}
                  className={`px-3 py-1.5 rounded-md text-[11px] font-semibold transition-colors ${
                    previewTab === tab
                      ? 'bg-violet-600 text-white'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  {tab === 'annotated'
                    ? `Annotated (${annoReport.annotated_samples.length})`
                    : `Low Confidence (${annoReport.low_confidence_samples.length})`}
                </button>
              ))}
            </div>

            {/* Image grid */}
            {(() => {
              const samples = previewTab === 'annotated'
                ? annoReport.annotated_samples
                : annoReport.low_confidence_samples;

              const slug = jobId ? `local_${jobId.replace(/-/g, '').slice(0, 8)}` : '';
              const baseImageUrl = `${BASE_URL}/data/${slug}`;

              if (samples.length === 0) {
                return (
                  <div className="py-10 text-center text-slate-600 border border-dashed border-slate-800 rounded-xl">
                    <ImageIcon className="mx-auto h-8 w-8 mb-2 opacity-30" />
                    <p className="text-xs">No samples in this category.</p>
                  </div>
                );
              }

              return (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                  {samples.map((s, i) => {
                    const bandColor =
                      s.confidence_band === 'high'   ? 'bg-emerald-500 text-white' :
                      s.confidence_band === 'medium' ? 'bg-amber-500 text-white'   :
                                                        'bg-rose-500 text-white';
                    return (
                      <div
                        key={i}
                        className="group relative rounded-xl overflow-hidden border border-slate-800
                                   bg-slate-950 hover:border-violet-500/50 transition-colors"
                      >
                        <img
                          src={`${baseImageUrl}/${s.url}`}
                          alt={s.label}
                          className="w-full aspect-square object-cover"
                          onError={e => { (e.target as HTMLImageElement).src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect fill="%231e293b" width="100" height="100"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="%23475569" font-size="10">No img</text></svg>'; }}
                        />
                        {/* Confidence badge */}
                        <span className={`absolute top-1.5 right-1.5 rounded-full px-1.5 py-0.5 text-[9px] font-bold ${bandColor}`}>
                          {(s.confidence * 100).toFixed(0)}%
                        </span>
                        {/* Label tooltip */}
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-2">
                          <p className="text-[9px] font-mono text-white truncate">{s.label}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </div>
        )}

      </div>
    </div>
  );
}
