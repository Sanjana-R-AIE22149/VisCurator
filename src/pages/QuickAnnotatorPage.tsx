import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Upload, X, Zap, Sparkles, Download, RefreshCw,
  Tag, CheckCircle2, AlertTriangle, ScanSearch, ChevronDown,
} from 'lucide-react';

/* ─── Types ──────────────────────────────────────────────────────────────── */
interface Detection {
  label: string;        // DETR model label
  score: number;        // confidence 0-1
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
  confirmedClass: string;
  color: string;
}

interface ImageResult {
  file: File;
  dataUrl: string;
  detections: Detection[];
}

/* ─── Constants ──────────────────────────────────────────────────────────── */
const PALETTE = [
  '#7c6fff','#4fc3f7','#69f0ae','#ffd740','#ff6e40',
  '#f06292','#81c784','#64b5f6','#ffb74d','#ba68c8',
  '#4db6ac','#ff8a65','#a1887f','#90a4ae','#dce775',
];

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _vcDetector: any;
    JSZip: {
      new(): {
        file(name: string, content: string): void;
        generateAsync(opts: { type: string }): Promise<Blob>;
      };
    };
  }
}

/* ─── helpers ────────────────────────────────────────────────────────────── */
function fileToDataUrl(file: File): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = e => res(e.target!.result as string);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

function mapToCustomClass(modelLabel: string, customs: string[]): string {
  if (customs.length === 0) return modelLabel;
  const lbl = modelLabel.toLowerCase();
  const match = customs.find(c =>
    lbl.includes(c.toLowerCase()) || c.toLowerCase().includes(lbl)
  );
  return match ?? customs[0];
}

function drawBoxes(
  canvas: HTMLCanvasElement,
  dataUrl: string,
  detections: Detection[],
  highlightIdx: number,
) {
  const img = new Image();
  img.onload = () => {
    const wrap = canvas.parentElement!;
    const W = wrap.clientWidth  || 360;
    const H = wrap.clientHeight || 270;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d')!;

    const scale = Math.min(W / img.width, H / img.height);
    const sw = img.width  * scale;
    const sh = img.height * scale;
    const ox = (W - sw) / 2;
    const oy = (H - sh) / 2;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(img, ox, oy, sw, sh);

    detections.forEach((d, i) => {
      const { xmin, ymin, xmax, ymax } = d.box;
      const bx = ox + xmin * scale;
      const by = oy + ymin * scale;
      const bw = (xmax - xmin) * scale;
      const bh = (ymax - ymin) * scale;
      const isHL = i === highlightIdx;

      ctx.strokeStyle = d.color;
      ctx.lineWidth   = isHL ? 3 : 2;
      if (isHL) { ctx.shadowColor = d.color; ctx.shadowBlur = 14; }
      ctx.strokeRect(bx, by, bw, bh);
      ctx.shadowBlur = 0;

      const lbl = d.confirmedClass;
      ctx.font = `${isHL ? 700 : 600} 11px Inter,sans-serif`;
      const tw  = ctx.measureText(lbl).width;
      ctx.globalAlpha = isHL ? 0.95 : 0.85;
      ctx.fillStyle   = d.color;
      ctx.fillRect(bx - 1, by - 20, tw + 12, 18);
      ctx.globalAlpha = 1;
      ctx.fillStyle   = '#000';
      ctx.fillText(lbl, bx + 5, by - 6);
    });
  };
  img.src = dataUrl;
}

/* ══════════════════════════════════════════════════════════════════════════ */
export default function QuickAnnotatorPage() {

  /* ── state ── */
  const [modelStatus, setModelStatus] = useState<'idle'|'loading'|'ready'|'error'>('idle');
  const [modelProgress, setModelProgress] = useState(0);
  const [imageFiles,  setImageFiles]  = useState<File[]>([]);
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const [classInput,  setClassInput]  = useState('');
  const [results,     setResults]     = useState<ImageResult[]>([]);
  const [activeImg,   setActiveImg]   = useState<number>(-1);
  const [activeDet,   setActiveDet]   = useState<number>(-1);
  const [phase, setPhase] = useState<'idle'|'running'|'done'>('idle');
  const [statusMsg, setStatusMsg] = useState('Upload images, add class names, then click Auto-Detect.');
  const [dragOver,  setDragOver]  = useState(false);
  const [exportDone, setExportDone] = useState(false);

  const canvasRefs = useRef<Record<number, HTMLCanvasElement | null>>({});
  const dropRef    = useRef<HTMLDivElement>(null);

  /* ── page title ── */
  useEffect(() => { document.title = 'Quick Annotator — VisCurator'; }, []);

  /* ── load transformers.js on demand ── */
  const loadModel = useCallback(async () => {
    if (window._vcDetector) return window._vcDetector as unknown;
    setModelStatus('loading');
    setModelProgress(0);

    // Dynamically inject the ESM script by using a blob re-export trick
    // (Vite/React can't statically import CDN ESM directly in prod without config)
    const src = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
    const { pipeline, env } = await import(/* @vite-ignore */ src) as {
      pipeline: (task: string, model: string, opts?: object) => Promise<unknown>;
      env: { allowLocalModels: boolean };
    };
    env.allowLocalModels = false;

    const det = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
      quantized: true,
      progress_callback: (p: { status: string; progress?: number; file?: string }) => {
        if (p.status === 'progress') {
          setModelProgress(Math.round(p.progress ?? 0));
        }
      },
    });
    window._vcDetector = det;
    setModelStatus('ready');
    return det;
  }, []);

  /* ── drag-and-drop ── */
  const handleFiles = useCallback((files: FileList | File[]) => {
    const imgs = [...files].filter(f => f.type.startsWith('image/'));
    setImageFiles(prev => [...prev, ...imgs]);
  }, []);

  useEffect(() => {
    const el = dropRef.current;
    if (!el) return;
    const over  = (e: DragEvent) => { e.preventDefault(); setDragOver(true);  };
    const leave = ()              => setDragOver(false);
    const drop  = (e: DragEvent) => {
      e.preventDefault(); setDragOver(false);
      if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
    };
    el.addEventListener('dragover',  over);
    el.addEventListener('dragleave', leave);
    el.addEventListener('drop',      drop);
    return () => {
      el.removeEventListener('dragover',  over);
      el.removeEventListener('dragleave', leave);
      el.removeEventListener('drop',      drop);
    };
  }, [handleFiles]);

  /* ── class management ── */
  const addClass = () => {
    const names = classInput.split(',').map(s => s.trim()).filter(Boolean);
    setCustomClasses(prev => {
      const next = [...prev];
      names.forEach(n => { if (!next.includes(n)) next.push(n); });
      return next;
    });
    setClassInput('');
  };

  /* ── run detection ── */
  const runDetection = async () => {
    if (imageFiles.length === 0) return;
    setPhase('running');
    setResults([]);
    setActiveImg(-1);
    setActiveDet(-1);
    setExportDone(false);

    let det: unknown;
    try {
      det = await loadModel();
    } catch (e) {
      setModelStatus('error');
      setStatusMsg('Failed to load model: ' + (e instanceof Error ? e.message : String(e)));
      setPhase('idle');
      return;
    }

    type RawDet = { label: string; score: number; box: { xmin: number; ymin: number; xmax: number; ymax: number } };
    const detFn = det as (url: string, opts: object) => Promise<RawDet[]>;

    const newResults: ImageResult[] = [];

    for (let i = 0; i < imageFiles.length; i++) {
      const file = imageFiles[i];
      setStatusMsg(`Analysing ${file.name} (${i + 1}/${imageFiles.length})…`);
      const dataUrl = await fileToDataUrl(file);

      let rawDets: RawDet[] = [];
      try {
        rawDets = await detFn(dataUrl, { threshold: 0.4 });
      } catch { rawDets = []; }

      const detections: Detection[] = rawDets.map((d, idx) => ({
        label: d.label,
        score: d.score,
        box: d.box,
        confirmedClass: mapToCustomClass(d.label, customClasses),
        color: PALETTE[idx % PALETTE.length],
      }));

      const result: ImageResult = { file, dataUrl, detections };
      newResults.push(result);

      // Update state incrementally so cards appear as they finish
      setResults(prev => [...prev, result]);
    }

    setPhase('done');
    setStatusMsg(`Done! ${newResults.length} image(s) annotated. Review & confirm classes below.`);
    setActiveImg(0);
  };

  /* ── redraw canvas when results/active change ── */
  useEffect(() => {
    results.forEach((r, i) => {
      const canvas = canvasRefs.current[i];
      if (canvas) {
        drawBoxes(canvas, r.dataUrl, r.detections, activeImg === i ? activeDet : -1);
      }
    });
  }, [results, activeImg, activeDet]);

  /* ── update class on a detection ── */
  const updateClass = (imgIdx: number, detIdx: number, newClass: string) => {
    setResults(prev => {
      const next = [...prev];
      const dets = [...next[imgIdx].detections];
      dets[detIdx] = { ...dets[detIdx], confirmedClass: newClass };
      next[imgIdx] = { ...next[imgIdx], detections: dets };
      return next;
    });
    setActiveImg(imgIdx);
    setActiveDet(detIdx);
  };

  /* ── YOLO export ── */
  const exportYOLO = async () => {
    const allClasses = new Set<string>();
    results.forEach(r => r.detections.forEach(d => allClasses.add(d.confirmedClass)));
    const classList = [...allClasses];

    const files: { name: string; content: string }[] = [];

    // data.yaml
    files.push({
      name: 'data.yaml',
      content: `path: ./dataset\ntrain: images/train\nval: images/val\nnc: ${classList.length}\nnames: [${classList.map(c => `'${c}'`).join(', ')}]\n`,
    });

    // label .txt files
    for (const r of results) {
      const img = new Image();
      await new Promise<void>(res => { img.onload = () => res(); img.src = r.dataUrl; });
      const W = img.naturalWidth, H = img.naturalHeight;
      const lines = r.detections.map(d => {
        const { xmin, ymin, xmax, ymax } = d.box;
        const id = classList.indexOf(d.confirmedClass);
        const cx = ((xmin + xmax) / 2) / W;
        const cy = ((ymin + ymax) / 2) / H;
        const nw = (xmax - xmin) / W;
        const nh = (ymax - ymin) / H;
        return `${id} ${cx.toFixed(6)} ${cy.toFixed(6)} ${nw.toFixed(6)} ${nh.toFixed(6)}`;
      });
      const stem = r.file.name.replace(/\.[^.]+$/, '');
      files.push({ name: `labels/${stem}.txt`, content: lines.join('\n') + '\n' });
    }

    files.push({
      name: 'README.txt',
      content: `VisCurator Quick Annotator Export\nClasses (${classList.length}): ${classList.join(', ')}\nImages: ${results.length}\nFormat: YOLO normalized (cx cy w h)\n\nUsage:\n  yolo train data=data.yaml model=yolov8n.pt epochs=100\n`,
    });

    // Load JSZip
    if (!window.JSZip) {
      await new Promise<void>((res, rej) => {
        const s = document.createElement('script');
        s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
        s.onload = () => res(); s.onerror = rej;
        document.head.appendChild(s);
      });
    }
    const zip = new window.JSZip();
    files.forEach(f => zip.file(f.name, f.content));
    const blob = await zip.generateAsync({ type: 'blob' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'viscurator_quick_annotations.zip'; a.click();
    URL.revokeObjectURL(url);
    setExportDone(true);
    setStatusMsg(`✅ Exported ${files.length} files as viscurator_quick_annotations.zip`);
  };

  const reset = () => {
    setImageFiles([]); setCustomClasses([]); setClassInput('');
    setResults([]); setActiveImg(-1); setActiveDet(-1);
    setPhase('idle'); setExportDone(false);
    setStatusMsg('Upload images, add class names, then click Auto-Detect.');
    canvasRefs.current = {};
  };

  /* ─────────────────────────────────────────── render ─────────────────── */
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fade-up">

        {/* ── Header ── */}
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px flex-1 bg-gradient-to-r from-violet-500/40 to-transparent" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
            <ScanSearch className="w-6 h-6 text-violet-400" />
            Quick Annotator
            <span className="text-xs font-normal px-2 py-0.5 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300">
              🤖 AI-Powered · Runs in Browser
            </span>
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Drop images → AI auto-detects objects &amp; predicts classes → confirm or correct → export YOLO labels.
          </p>
        </div>

        {/* ── Model status banner ── */}
        {modelStatus === 'loading' && (
          <div className="rounded-xl border border-sky-500/30 bg-sky-500/5 px-5 py-4 space-y-2">
            <div className="flex items-center gap-3">
              <RefreshCw className="w-4 h-4 text-sky-400 animate-spin shrink-0" />
              <p className="text-sm font-semibold text-sky-300">Loading DETR model via Transformers.js…</p>
              <span className="ml-auto text-xs font-mono text-sky-400">{modelProgress}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500 transition-all duration-300"
                style={{ width: `${modelProgress}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-500">First-time download (~50 MB). Cached by your browser after that.</p>
          </div>
        )}

        {/* ── Setup Row ── */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">

          {/* Drop zone */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
            <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3 flex items-center gap-2">
              <Upload className="w-3.5 h-3.5" /> Upload Images
            </p>
            <div
              ref={dropRef}
              className={`relative rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-all
                ${dragOver
                  ? 'border-violet-400 bg-violet-400/10 shadow-lg shadow-violet-500/10'
                  : 'border-slate-700 bg-slate-950/40 hover:border-slate-600'}`}
            >
              <input
                type="file" multiple accept="image/*"
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                onChange={e => e.target.files && handleFiles(e.target.files)}
              />
              <div className="text-3xl mb-2 opacity-60">🖼️</div>
              <p className="text-sm font-semibold text-slate-300">Drop images here</p>
              <p className="text-xs text-slate-600 mt-1">or click to browse · JPG, PNG, WEBP</p>
            </div>

            {imageFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {imageFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-1.5 bg-violet-500/10 border border-violet-500/25 rounded-md px-2.5 py-1">
                    <span className="text-[11px] font-mono text-violet-300 truncate max-w-[140px]">{f.name}</span>
                    <button
                      onClick={() => setImageFiles(prev => prev.filter((_, j) => j !== i))}
                      className="text-slate-600 hover:text-rose-400 transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Class names */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
            <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3 flex items-center gap-2">
              <Tag className="w-3.5 h-3.5" /> Target Classes
            </p>
            <p className="text-xs text-slate-500 mb-3">
              Define the classes for your dataset. The AI maps its detections to your classes.
              Leave empty to use COCO default labels.
            </p>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={classInput}
                onChange={e => setClassInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && addClass()}
                placeholder="e.g. cat, dog, car  (comma-separated)"
                className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm
                           text-slate-200 focus:border-violet-500 focus:outline-none"
              />
              <button
                onClick={addClass}
                className="rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700
                           px-4 py-2 text-sm font-semibold text-slate-300 transition-colors"
              >
                Add
              </button>
            </div>

            {customClasses.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {customClasses.map((c, i) => {
                  const color = PALETTE[i % PALETTE.length];
                  return (
                    <span
                      key={c}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border"
                      style={{ color, borderColor: color + '40', background: color + '18' }}
                    >
                      {c}
                      <button onClick={() => setCustomClasses(prev => prev.filter((_, j) => j !== i))} className="opacity-50 hover:opacity-100">
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-slate-700 italic">No custom classes — using COCO-91 defaults</p>
            )}
          </div>
        </div>

        {/* ── Action bar ── */}
        <div className="flex items-center gap-3 flex-wrap">
          <button
            id="btn-quick-annotate"
            onClick={runDetection}
            disabled={imageFiles.length === 0 || phase === 'running'}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-violet-500
                       hover:from-violet-500 hover:to-violet-400 disabled:opacity-40
                       px-5 py-2.5 text-sm font-bold text-white transition-all shadow-lg shadow-violet-500/25"
          >
            {phase === 'running'
              ? <><RefreshCw className="w-4 h-4 animate-spin" /> Detecting…</>
              : <><ScanSearch className="w-4 h-4" /> Auto-Detect &amp; Annotate</>}
          </button>

          {results.length > 0 && (
            <button
              id="btn-export-yolo"
              onClick={exportYOLO}
              className="flex items-center gap-2 rounded-xl border border-emerald-500/40 bg-emerald-600/15
                         hover:bg-emerald-600/30 px-5 py-2.5 text-sm font-bold text-emerald-300 transition-all"
            >
              <Download className="w-4 h-4" />
              {exportDone ? 'Re-export YOLO ZIP' : 'Export YOLO ZIP'}
            </button>
          )}

          <button
            onClick={reset}
            className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/50
                       hover:border-slate-600 px-4 py-2.5 text-sm font-semibold text-slate-400 transition-all"
          >
            <RefreshCw className="w-4 h-4" /> Reset
          </button>

          <div className="ml-auto text-xs text-slate-600 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            DETR + Transformers.js · runs locally in-browser
          </div>
        </div>

        {/* ── Status bar ── */}
        <div className={`flex items-center gap-2.5 rounded-lg border px-4 py-2.5 text-sm transition-colors
          ${phase === 'running'
            ? 'border-sky-500/30 bg-sky-500/5 text-sky-300'
            : phase === 'done'
            ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-300'
            : 'border-slate-800 bg-slate-900/30 text-slate-500'}`}
        >
          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
            phase === 'running' ? 'bg-sky-400 animate-pulse' :
            phase === 'done'    ? 'bg-emerald-400' : 'bg-slate-600'
          }`} />
          {statusMsg}
        </div>

        {/* ── Results grid ── */}
        {results.length > 0 && (
          <div className="space-y-4">
            <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-violet-400" />
              Annotated Images — click a detection to highlight its bounding box
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
              {results.map((r, imgIdx) => (
                <div
                  key={imgIdx}
                  className={`rounded-xl border overflow-hidden transition-all
                    ${activeImg === imgIdx
                      ? 'border-violet-500/50 shadow-lg shadow-violet-500/10'
                      : 'border-slate-800 hover:border-slate-700'}`}
                >
                  {/* Canvas */}
                  <div
                    className="relative w-full bg-slate-950 overflow-hidden"
                    style={{ aspectRatio: '4/3' }}
                    onClick={() => { setActiveImg(imgIdx); setActiveDet(-1); }}
                  >
                    <canvas
                      ref={el => { canvasRefs.current[imgIdx] = el; }}
                      className="absolute inset-0 w-full h-full"
                    />
                  </div>

                  {/* Card body */}
                  <div className="p-4 bg-slate-900/60 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-mono text-slate-400 truncate max-w-[200px]">
                        {r.file.name}
                      </span>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300">
                        {r.detections.length} object{r.detections.length !== 1 ? 's' : ''}
                      </span>
                    </div>

                    {/* Detection rows */}
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      {r.detections.length === 0 ? (
                        <div className="flex items-center gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                          <p className="text-xs text-amber-300">No objects detected in this image.</p>
                        </div>
                      ) : (
                        r.detections.map((d, di) => {
                          const isActive = activeImg === imgIdx && activeDet === di;
                          return (
                            <div
                              key={di}
                              onClick={() => { setActiveImg(imgIdx); setActiveDet(di); }}
                              className={`flex items-center gap-2 rounded-lg px-3 py-2 cursor-pointer transition-all border
                                ${isActive
                                  ? 'border-violet-500/50 bg-violet-500/10'
                                  : 'border-slate-800 bg-slate-950/50 hover:border-slate-700'}`}
                            >
                              <div
                                className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                                style={{ background: d.color }}
                              />
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-semibold text-slate-200 truncate">{d.confirmedClass}</p>
                                <p className="text-[10px] text-slate-500">{d.label} · {(d.score * 100).toFixed(0)}% conf</p>
                              </div>

                              {/* Class selector dropdown */}
                              <div className="relative flex-shrink-0" onClick={e => e.stopPropagation()}>
                                <select
                                  value={d.confirmedClass}
                                  onChange={e => updateClass(imgIdx, di, e.target.value)}
                                  className="appearance-none rounded-md border border-slate-700 bg-slate-900
                                             text-slate-300 text-[11px] pl-2 pr-6 py-1 cursor-pointer
                                             focus:outline-none focus:border-violet-500"
                                >
                                  {/* Always show model label as an option */}
                                  {!customClasses.includes(d.label) && (
                                    <option value={d.label}>{d.label}</option>
                                  )}
                                  {(customClasses.length > 0 ? customClasses : [d.label]).map(c => (
                                    <option key={c} value={c}>{c}</option>
                                  ))}
                                </select>
                                <ChevronDown className="absolute right-1 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-500 pointer-events-none" />
                              </div>

                              {/* Confirmed icon */}
                              <CheckCircle2 className={`w-4 h-4 flex-shrink-0 transition-colors ${
                                d.confirmedClass !== d.label ? 'text-emerald-400' : 'text-slate-700'
                              }`} />
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Export success ── */}
        {exportDone && (
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-5 py-4 flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-emerald-300">YOLO dataset exported!</p>
              <p className="text-xs text-slate-500 mt-0.5">
                Your zip contains <code className="text-emerald-400">data.yaml</code> and per-image label files ready for YOLOv8 training.
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
