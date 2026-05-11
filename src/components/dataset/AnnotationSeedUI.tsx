import { useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { Upload, X, Zap, ChevronRight, Image as ImageIcon } from 'lucide-react';
import { uploadSeedClass, startAnnotation } from '../../lib/api';

export default function AnnotationSeedUI() {
  const { localDataset, jobId, addTerminalLog, setPreprocessingReport } = useAppStore();
  const [classes, setClasses] = useState<string[]>([]);
  const [currentClass, setCurrentClass] = useState('');
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [dragActive, setDragActive] = useState<string | null>(null);
  const [uploading, setUploading] = useState<Record<string, boolean>>({});
  const [seedCounts, setSeedCounts] = useState<Record<string, number>>({});

  if (!localDataset || !jobId) return null;

  const handleAddClass = () => {
    const cls = currentClass.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_');
    if (!cls) return;
    if (!classes.includes(cls)) {
      setClasses([...classes, cls]);
    }
    setCurrentClass('');
  };

  const removeClass = (cls: string) => {
    setClasses(classes.filter((c) => c !== cls));
    setSeedCounts((prev) => {
      const next = { ...prev };
      delete next[cls];
      return next;
    });
  };

  const handleDrop = async (e: React.DragEvent, cls: string) => {
    e.preventDefault();
    setDragActive(null);
    if (!e.dataTransfer.files || e.dataTransfer.files.length === 0) return;

    setUploading({ ...uploading, [cls]: true });
    try {
      await uploadSeedClass(jobId, cls, e.dataTransfer.files);
      setSeedCounts({ ...seedCounts, [cls]: (seedCounts[cls] || 0) + e.dataTransfer.files.length });
      addTerminalLog({
        id: `seed-up-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `> Uploaded ${e.dataTransfer.files.length} seed(s) for class '${cls}'.`,
        status: 'ok',
        msgType: 'log',
      });
    } catch (err) {
      addTerminalLog({
        id: `seed-err-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `Failed to upload seeds: ${err instanceof Error ? err.message : 'Unknown'}`,
        status: 'error',
        msgType: 'error',
      });
    } finally {
      setUploading({ ...uploading, [cls]: false });
    }
  };

  const handleRunAnnotation = async () => {
    if (classes.length === 0) return;
    setIsAnnotating(true);
    addTerminalLog({
      id: `anno-start-${Date.now()}`,
      timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
      message: '> Starting Clean & Auto-Annotate pipeline (SAM & CLIP)...',
      status: 'info',
      msgType: 'log',
    });

    try {
      setPreprocessingReport(null);
      await startAnnotation(jobId);
    } catch (err) {
      addTerminalLog({
        id: `anno-err-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `Annotation failed: ${err instanceof Error ? err.message : 'Unknown'}`,
        status: 'error',
        msgType: 'error',
      });
      setIsAnnotating(false);
    }
  };

  return (
    <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-5 animate-fade-up">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-violet-300 flex items-center gap-2">
            <Zap className="h-4 w-4" /> Define Classes & Upload Seeds
          </h2>
          <p className="text-[11px] text-slate-400 mt-1">
            Add classes and drop seed images/ZIPs into them. We will clean the dataset and use your seeds to auto-annotate the rest.
          </p>
        </div>
        <button
          onClick={handleRunAnnotation}
          disabled={isAnnotating || classes.length === 0}
          className="flex items-center gap-2 rounded-lg bg-violet-500 hover:bg-violet-600 px-4 py-2 text-xs font-bold text-white transition-colors disabled:opacity-50"
        >
          {isAnnotating ? 'Processing...' : 'Clean & Annotate'}
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={currentClass}
          onChange={(e) => setCurrentClass(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
          placeholder="New class (e.g. apple_scab)"
          className="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 focus:border-violet-500 focus:outline-none"
        />
        <button
          onClick={handleAddClass}
          className="rounded bg-slate-800 px-4 py-2 text-sm font-bold text-slate-300 hover:bg-slate-700"
        >
          Add Class
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {classes.map((cls) => (
          <div
            key={cls}
            className={`relative rounded-xl border-2 border-dashed p-4 transition-colors ${
              dragActive === cls ? 'border-violet-400 bg-violet-400/10' : 'border-slate-700 bg-slate-900/40 hover:border-slate-600'
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(cls);
            }}
            onDragLeave={() => setDragActive(null)}
            onDrop={(e) => handleDrop(e, cls)}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-xs font-bold text-violet-300">{cls}</span>
              <button onClick={() => removeClass(cls)} className="text-slate-500 hover:text-rose-400">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-col items-center justify-center py-4 text-slate-500">
              {uploading[cls] ? (
                <div className="h-6 w-6 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
              ) : (
                <>
                  <Upload className="mb-2 h-6 w-6" />
                  <p className="text-[10px] text-center">Drop seed images or ZIP here</p>
                  
                  <label className="mt-2 cursor-pointer rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-[10px] font-semibold text-slate-300 transition-colors hover:bg-slate-700">
                    Browse Files
                    <input 
                      type="file" 
                      multiple 
                      accept="image/*,.zip" 
                      className="hidden" 
                      onChange={(e) => {
                        if (e.target.files && e.target.files.length > 0) {
                          handleDrop({ preventDefault: () => {}, dataTransfer: { files: e.target.files } } as any, cls);
                        }
                      }} 
                    />
                  </label>

                  <p className="mt-2 text-[10px] font-bold text-violet-400">
                    {seedCounts[cls] || 0} files uploaded
                  </p>
                </>
              )}
            </div>
          </div>
        ))}
        {classes.length === 0 && (
          <div className="col-span-full py-8 text-center text-slate-500 border border-dashed border-slate-800 rounded-xl">
            <ImageIcon className="mx-auto h-8 w-8 mb-2 opacity-50" />
            <p className="text-xs">Add a class above to start uploading seeds.</p>
          </div>
        )}
      </div>
    </div>
  );
}
