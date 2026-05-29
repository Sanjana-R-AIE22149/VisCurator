import { useAppStore } from '../../store/useAppStore';
import { Layers, Image as ImageIcon, Filter, Zap, ExternalLink } from 'lucide-react';
import { BASE_URL } from '../../lib/api';

export default function StagePreview() {
  const { preprocessingReport } = useAppStore();

  if (!preprocessingReport?.stage_samples) {
    return null;
  }

  const { raw, filtered, processed } = preprocessingReport.stage_samples;
  const previewLimit = 12;

  const getImageUrl = (url: string) => {
    // The url in the report is relative to the dataset output dir
    // We need to map it to /data/{dataset_id_slug}/{url}
    const slug = preprocessingReport.dataset_id.replace(/\//g, '_');
    return `${BASE_URL}/data/${slug}/${url}`;
  };

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
        <Layers className="h-4 w-4 text-teal-400" />
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">Processing Stages</h3>
        <span className="ml-auto text-[10px] text-slate-500 font-mono">Previewing up to {previewLimit} per stage</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Stage 1: Raw */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 px-1">
            <ImageIcon className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-xs font-semibold text-slate-300">1. Original Samples</span>
          </div>
          <div className="flex items-center justify-between px-1 text-[10px] text-slate-500">
            <span>{raw.length} captured</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {raw.slice(0, previewLimit).map((img, i) => (
              <div key={i} className="group relative aspect-square overflow-hidden rounded-lg border border-slate-800 bg-slate-900/50">
                <img 
                  src={getImageUrl(img.url)} 
                  alt="Raw" 
                  className="h-full w-full object-cover transition-transform group-hover:scale-110"
                />
                <div className="absolute inset-x-0 bottom-0 bg-black/60 p-1 px-2">
                  <p className="truncate text-[8px] font-mono text-slate-300">{img.label}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Stage 2: Filtered */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 px-1">
            <Filter className="h-3.5 w-3.5 text-rose-400" />
            <span className="text-xs font-semibold text-rose-300">2. Rejected (Blur/Dup)</span>
          </div>
          <div className="flex items-center justify-between px-1 text-[10px] text-rose-300/80">
            <span>{filtered.length} rejected shown</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {filtered.length > 0 ? (
              filtered.slice(0, previewLimit).map((img, i) => (
                <div key={i} className="group relative aspect-square overflow-hidden rounded-lg border border-rose-500/20 bg-rose-500/5">
                  <img 
                    src={getImageUrl(img.url)} 
                    alt="Filtered" 
                    className="h-full w-full object-cover grayscale opacity-60"
                  />
                  <div className="absolute inset-x-0 bottom-0 bg-rose-900/80 p-1 px-2">
                    <p className="truncate text-[8px] font-mono text-white font-bold">{img.reason.toUpperCase()}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-2 flex h-32 flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-900/20 text-slate-600">
                <p className="text-[10px]">No rejections found</p>
              </div>
            )}
          </div>
        </div>

        {/* Stage 3: Processed */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 px-1">
            <Zap className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-xs font-semibold text-emerald-300">3. Curated & Augmented</span>
          </div>
          <div className="flex items-center justify-between px-1 text-[10px] text-emerald-300/80">
            <span>{processed.length} curated previews</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {processed.slice(0, previewLimit).map((img, i) => (
              <div key={i} className="group relative aspect-square overflow-hidden rounded-lg border border-emerald-500/30 bg-emerald-500/5 ring-1 ring-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.1)]">
                <img 
                  src={getImageUrl(img.url)} 
                  alt="Processed" 
                  className="h-full w-full object-cover transition-transform group-hover:scale-110"
                />
                <div className="absolute inset-x-0 bottom-0 bg-emerald-950/80 p-1 px-2">
                  <p className="truncate text-[8px] font-mono text-emerald-100 font-bold">READY</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="rounded-lg bg-slate-900/40 p-3 border border-slate-800 flex items-center justify-between">
        <p className="text-[10px] text-slate-400 leading-relaxed max-w-[80%]">
          The processing pipeline performed <span className="text-teal-400">blur detection</span>, 
          <span className="text-teal-400"> perceptual hash deduplication</span>, and 
          <span className="text-teal-400"> minority-class augmentation</span>. 
          Samples above represent the state of data at each checkpoint.
        </p>
        <a 
          href={`${BASE_URL}/data/${preprocessingReport.dataset_id.replace(/\//g, '_')}/preprocessing_report.json`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 rounded bg-slate-800 px-3 py-1.5 text-[10px] font-bold text-slate-300 hover:bg-slate-700 transition-colors"
        >
          JSON Report <ExternalLink className="h-3 w-3" />
        </a>
      </div>
    </div>
  );
}
