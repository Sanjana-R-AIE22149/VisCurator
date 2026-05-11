import { useRef, useEffect, useState } from 'react';
import { Search, ChevronDown, Play, RotateCcw, ExternalLink, CheckCircle, ChevronRight, Upload } from 'lucide-react';
import { useAppStore, type PipelineQuestion, type DatasetOption, type PreprocessingReport, type ProcessingPlan } from '../../store/useAppStore';
import {
  startDatasetPipeline,
  connectPipelineWebSocket,
  replyToJob,
  getHealth,
  type PipelineMessage,
} from '../../lib/api';

function DatasetOptionCard({
  option,
  index,
  onSelect,
}: {
  option: DatasetOption;
  index: number;
  onSelect: (opt: DatasetOption) => void;
}) {
  const sourceColors: Record<string, string> = {
    HuggingFace: 'text-yellow-400 border-yellow-400/30 bg-yellow-400/10',
    Kaggle: 'text-sky-400 border-sky-400/30 bg-sky-400/10',
    Roboflow: 'text-purple-400 border-purple-400/30 bg-purple-400/10',
    PapersWithCode: 'text-emerald-400 border-emerald-400/30 bg-emerald-400/10',
  };
  const badgeClass = sourceColors[option.source] || 'text-slate-400 border-slate-400/30 bg-slate-400/10';

  return (
    <div
      className="group relative cursor-pointer rounded-xl border border-slate-700/60 bg-slate-900/50 p-4 transition-all hover:border-teal-500/40 hover:bg-slate-900/80"
      onClick={() => onSelect(option)}
    >
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-bold text-slate-200">{option.name}</span>
          <span className={`rounded-full border px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${badgeClass}`}>
            {option.source}
          </span>
          <span className="font-mono text-[10px] text-slate-500">{option.size_estimate}</span>
        </div>
        <span className="shrink-0 font-mono text-[10px] text-slate-600">#{index + 1}</span>
      </div>
      <p className="mb-3 line-clamp-2 text-[11px] leading-relaxed text-slate-400">{option.description}</p>
      {(option.pros?.length > 0 || option.cons?.length > 0) && (
        <div className="mb-3 grid grid-cols-2 gap-2">
          {option.pros?.slice(0, 2).map((pro, i) => (
            <div key={i} className="flex items-start gap-1">
              <span className="mt-0.5 text-[10px] text-emerald-400">+</span>
              <span className="text-[10px] text-slate-500">{pro}</span>
            </div>
          ))}
          {option.cons?.slice(0, 2).map((con, i) => (
            <div key={i} className="flex items-start gap-1">
              <span className="mt-0.5 text-[10px] text-rose-400">-</span>
              <span className="text-[10px] text-slate-500">{con}</span>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center justify-between">
        <a
          href={option.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-[10px] text-slate-600 transition-colors hover:text-teal-400"
          onClick={(e) => e.stopPropagation()}
        >
          <ExternalLink className="h-3 w-3" /> View dataset
        </a>
        <button className="flex items-center gap-1.5 rounded-lg border border-teal-500/30 bg-teal-500/15 px-3 py-1.5 text-[10px] font-semibold text-teal-400 transition-all group-hover:border-teal-500/50 hover:bg-teal-500/25">
          Select <ChevronRight className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}

function ClarificationPanel({
  question,
  onReply,
}: {
  question: PipelineQuestion;
  onReply: (reply: string) => void;
}) {
  const [customReply, setCustomReply] = useState('');

  return (
    <div className="mt-3 rounded-xl border border-violet-500/30 bg-violet-500/5 p-4">
      <p className="mb-3 text-xs font-semibold text-violet-300">{question.question}</p>
      {question.context && <p className="mb-3 text-[10px] text-slate-500">{question.context}</p>}
      {Array.isArray(question.options) && question.options.length > 0 && typeof question.options[0] === 'string' && (
        <div className="mb-3 flex flex-wrap gap-2">
          {(question.options as string[]).map((option, i) => (
            <button
              key={i}
              onClick={() => onReply(option)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-[11px] text-slate-300 transition-all hover:border-violet-400/50 hover:text-violet-300"
            >
              {option}
            </button>
          ))}
        </div>
      )}
      <div className="flex gap-2">
        <input
          type="text"
          value={customReply}
          onChange={(e) => setCustomReply(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && customReply.trim()) {
              onReply(customReply);
              setCustomReply('');
            }
          }}
          placeholder="Or type a custom answer..."
          className="flex-1 rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
        />
        <button
          onClick={() => {
            if (customReply.trim()) {
              onReply(customReply);
              setCustomReply('');
            }
          }}
          className="rounded-lg border border-violet-500/40 bg-violet-500/20 px-3 py-2 text-xs font-semibold text-violet-400 transition-all hover:bg-violet-500/30"
        >
          Send
        </button>
      </div>
    </div>
  );
}

function DatasetSelectionPanel({
  question,
  onSelect,
}: {
  question: PipelineQuestion;
  onSelect: (reply: string) => void;
}) {
  const options = question.options as DatasetOption[];

  return (
    <div className="mt-3 space-y-3">
      <div className="flex items-start justify-between">
        <p className="text-xs font-semibold text-teal-300">{question.question}</p>
      </div>
      {question.recommendation && (
        <div className="flex items-start gap-2 rounded-lg border border-teal-500/20 bg-teal-500/10 px-3 py-2">
          <CheckCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-teal-400" />
          <p className="text-[11px] text-teal-300/80">{question.recommendation}</p>
        </div>
      )}
      {options.length === 0 && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2">
          <p className="text-[11px] text-amber-300/80">
            No alternate options were returned for this paused job. Start a new search or reset this run.
          </p>
        </div>
      )}
      <div className="grid grid-cols-1 gap-3">
        {options.map((option, i) => (
          <DatasetOptionCard
            key={i}
            option={option}
            index={i}
            onSelect={(selected) => onSelect(`I'll use option ${i + 1}: ${selected.name} from ${selected.source} (dataset_id: ${selected.dataset_id})`)}
          />
        ))}
      </div>
    </div>
  );
}

export default function DatasetControls() {
  const {
    datasetQuery,
    setDatasetQuery,
    datasetSource,
    setDatasetSource,
    targetSize,
    setTargetSize,
    isProcessing,
    isPaused,
    pendingQuestion,
    startProcessing,
    resumeProcessing,
    stopProcessing,
    setPaused,
    clearPendingQuestion,
    clearTerminalLogs,
    addTerminalLog,
    setJobId,
    jobId,
    setDatasetResults,
    setBlurData,
    setImagesIngested,
    setQualityScore,
    setClassBalance,
    setDuplicatePercentage,
    setProcessingPlan,
    setPreprocessingReport,
  } = useAppStore();

  const wsRef = useRef<any>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const focusInput = () => inputRef.current?.focus();
    window.addEventListener('focus-dataset-query', focusInput);
    return () => window.removeEventListener('focus-dataset-query', focusInput);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const mapStatus = (type: PipelineMessage['type']): 'ok' | 'warn' | 'error' | 'info' => {
    switch (type) {
      case 'done':
        return 'ok';
      case 'error':
        return 'error';
      case 'tool_result':
        return 'ok';
      default:
        return 'info';
    }
  };

  const formatMessage = (msg: PipelineMessage): string => {
    switch (msg.type) {
      case 'thought':
        return `Thought: ${msg.message}`;
      case 'tool_call':
        return `Tool: ${msg.message}`;
      case 'tool_result':
        return `Result: ${msg.message}`;
      case 'done':
        return msg.data?.paused ? `Paused: ${msg.message}` : `Complete: ${msg.message}`;
      case 'error':
        return `Error: ${msg.message}`;
      case 'script_log':
        return msg.message;
      default:
        return `> ${msg.message}`;
    }
  };

  const handleToolResult = (data: Record<string, unknown>) => {
    if (data.live_sample && typeof data.live_sample === 'object') {
      const { stage, sample } = data.live_sample as any;
      const currentReport = useAppStore.getState().preprocessingReport || { 
        dataset_id: datasetQuery, 
        stage_samples: { raw: [], filtered: [], processed: [] } 
      };
      const stageSamples = { ...(currentReport.stage_samples || { raw: [], filtered: [], processed: [] }) };
      
      if (!stageSamples[stage as keyof typeof stageSamples]) {
        (stageSamples as any)[stage] = [];
      }
      
      // Avoid duplicates
      if (!stageSamples[stage as keyof typeof stageSamples].some((s: any) => s.id === sample.id)) {
        stageSamples[stage as keyof typeof stageSamples] = [...stageSamples[stage as keyof typeof stageSamples], sample].slice(-8);
        setPreprocessingReport({ ...currentReport, stage_samples: stageSamples });
      }
      return;
    }

    if (typeof data.quality_score === 'number') {
      setQualityScore(data.quality_score as number);
      setImagesIngested((data.images_sampled as number | null) ?? null);
    }
    if (typeof data.class_balance === 'string') {
      setClassBalance(data.class_balance as string);
    }
    if (typeof data.duplicate_percentage === 'number') {
      setDuplicatePercentage(data.duplicate_percentage as number);
    }
    if (Array.isArray(data.blur_scatter)) {
      setBlurData(data.blur_scatter as Array<{ id: number; laplacian: number; resolution: number; accepted: boolean }>);
    }
    if (typeof data.needs_blur_filtering === 'boolean' && Array.isArray(data.recommended_augmentations)) {
      setProcessingPlan(data as unknown as ProcessingPlan);
    }
    if (data.preprocessing_report && typeof data.preprocessing_report === 'object') {
      const report = data.preprocessing_report as PreprocessingReport;
      setPreprocessingReport(report);
      if (typeof report.after_stats?.images === 'number') {
        setImagesIngested(report.after_stats.images);
      }
      if (report.class_distribution) {
        const counts = Object.values(report.class_distribution);
        const max = Math.max(...counts, 0);
        const min = Math.min(...counts, max || 0);
        setClassBalance(max > 0 && min > 0 ? (max / min <= 1.35 ? 'Balanced' : max / min <= 2.25 ? 'Moderately Imbalanced' : 'Highly Imbalanced') : 'Unknown');
      }
      if (Array.isArray(report.blur_scatter)) {
        setBlurData(report.blur_scatter);
      }
    }
  };

  const handleWsMessage = (msg: PipelineMessage) => {
    addTerminalLog({
      id: msg.id,
      timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
      message: formatMessage(msg),
      status: mapStatus(msg.type),
      msgType: msg.type,
    });

    if (msg.type === 'tool_result') {
      handleToolResult(msg.data);
    }

    if (msg.type === 'done') {
      const data = msg.data as Record<string, unknown>;
      if (data?.paused) {
        const pauseType = data.type as string;
        let question = null;
        if (pauseType === 'clarification') {
          question = {
            type: 'clarification' as const,
            question: data.question as string,
            options: (data.options as string[]) || [],
            context: (data.context as string) || '',
          };
        } else if (pauseType === 'dataset_selection') {
          question = {
            type: 'dataset_selection' as const,
            question: data.question as string,
            options: (data.options as DatasetOption[]) || [],
            recommendation: (data.recommendation as string) || '',
          };
        }
        setPaused(true, question);
      } else {
        // ── Extract preprocessing_report from completed job ──────────────
        // Check 1: direct field on done data
        if (data?.preprocessing_report && typeof data.preprocessing_report === 'object') {
          const report = data.preprocessing_report as PreprocessingReport;
          setPreprocessingReport(report);
          if (typeof report.after_stats?.images === 'number') setImagesIngested(report.after_stats.images);
        }
        // Check 2: scan through tool_results for clean_and_augment result
        if (data?.tool_results && Array.isArray(data.tool_results)) {
          setDatasetResults(data.tool_results as any[]);
          for (const tr of data.tool_results as any[]) {
            if (tr?.tool === 'clean_and_augment_dataset') {
              const report = tr?.result?.preprocessing_report;
              if (report && typeof report === 'object') {
                setPreprocessingReport(report as PreprocessingReport);
                if (typeof report.after_stats?.images === 'number') setImagesIngested(report.after_stats.images);
                if (report.class_distribution) {
                  const counts = Object.values(report.class_distribution) as number[];
                  const mx = Math.max(...counts, 0);
                  const mn = Math.min(...counts, mx || 0);
                  setClassBalance(mx > 0 && mn > 0 ? (mx / mn <= 1.35 ? 'Balanced' : mx / mn <= 2.25 ? 'Moderately Imbalanced' : 'Highly Imbalanced') : 'Unknown');
                }
              }
              break;
            }
          }
        }
        stopProcessing();
        wsRef.current?.close();
        // Trigger a refresh of the processed datasets panel via custom event
        window.dispatchEvent(new CustomEvent('viscurator:pipeline-done'));
      }
    }

    if (msg.type === 'error') {
      stopProcessing();
      if (msg.message.includes('Job not found')) {
        setJobId(null);
      }
      wsRef.current?.close();
    }
  };

  const handleReply = async (reply: string) => {
    if (!jobId) return;

    clearPendingQuestion();
    addTerminalLog({
      id: `user-reply-${Date.now()}`,
      timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
      message: `You: ${reply}`,
      status: 'info',
      msgType: 'log',
    });

    try {
      await replyToJob(jobId, reply);
      resumeProcessing();
      
      // The SafeWebSocket in wsRef.current will handle reconnection if it's already there
      // If we closed it, we need to re-open or the manager on backend will handle it.
      if (!wsRef.current) {
        wsRef.current = connectPipelineWebSocket(
          jobId,
          handleWsMessage,
          (event) => {
            if (useAppStore.getState().isProcessing) {
              addTerminalLog({
                id: `ws-close-${Date.now()}`,
                timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
                message: event && event.code !== 1000 ? `Pipeline connection closed (${event.code}). Retrying...` : 'Pipeline connection closed.',
                status: 'info',
                msgType: 'log',
              });
            }
          },
          () => {
            addTerminalLog({
              id: `ws-error-${Date.now()}`,
              timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
              message: 'Pipeline websocket error. Reconnecting...',
              status: 'warn',
              msgType: 'error',
            });
          }
        );
      }
    } catch (err) {
      addTerminalLog({
        id: `reply-error-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `Failed to send reply: ${err instanceof Error ? err.message : 'Unknown error'}`,
        status: 'error',
        msgType: 'error',
      });
      stopProcessing();
    }
  };

  const handleRun = async () => {
    if (!datasetQuery.trim()) return;

    startProcessing();
    setBlurData([]);
    setImagesIngested(null);
    setQualityScore(null);
    setClassBalance(null);
    setDuplicatePercentage(null);
    setProcessingPlan(null);
    setPreprocessingReport(null);

    const health = await getHealth();
    if (!health.online) {
      addTerminalLog({
        id: `offline-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: '> Backend offline. Dataset pipeline unavailable.',
        status: 'warn',
        msgType: 'log',
      });
      stopProcessing();
      return;
    }

    try {
      addTerminalLog({
        id: `init-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: '> Connecting to CVAgent backend...',
        status: 'info',
        msgType: 'log',
      });

      const { job_id } = await startDatasetPipeline(datasetQuery, datasetSource, targetSize);
      setJobId(job_id);

      wsRef.current?.close();
      wsRef.current = connectPipelineWebSocket(
        job_id,
        handleWsMessage,
        (event) => {
          if (useAppStore.getState().isProcessing) {
            // SafeWebSocket will auto-reconnect, so we don't necessarily stopProcessing here
            // unless it's a permanent closure.
          }
        },
        () => {
          addTerminalLog({
            id: `ws-error-${Date.now()}`,
            timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
            message: 'Pipeline websocket error. Reconnecting...',
            status: 'warn',
            msgType: 'error',
          });
        }
      );
    } catch (err) {
      addTerminalLog({
        id: `err-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `Failed: ${err instanceof Error ? err.message : 'Unknown error'}`,
        status: 'error',
        msgType: 'error',
      });
      stopProcessing();
    }
  };

  const handleReset = () => {
    wsRef.current?.close();
    wsRef.current = null;
    clearTerminalLogs();
    setJobId(null);
    setDatasetResults([]);
    setBlurData([]);
    setImagesIngested(null);
    setQualityScore(null);
    setClassBalance(null);
    setDuplicatePercentage(null);
    setProcessingPlan(null);
    setPreprocessingReport(null);
    useAppStore.setState({ clonedRecipe: null, localDataset: null });
  };

  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.zip')) {
      await processUpload(file);
    } else {
      addTerminalLog({
        id: `upload-warn-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: 'Only .zip files are supported for local upload.',
        status: 'warn',
        msgType: 'log',
      });
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await processUpload(file);
    }
  };

  const processUpload = async (file: File) => {
    startProcessing();
    addTerminalLog({
      id: `upload-${Date.now()}`,
      timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
      message: `> Uploading and extracting ${file.name}...`,
      status: 'info',
      msgType: 'log',
    });

    try {
      // Import dynamically to avoid circular dependencies if any
      const { uploadDataset } = await import('../../lib/api');
      const { job_id, files, slug } = await uploadDataset(file);
      
      setJobId(job_id);
      useAppStore.getState().setLocalDataset({ slug, path: `cvagent_output/${slug}/raw`, files });
      
      addTerminalLog({
        id: `upload-ok-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `> Successfully extracted ${files.length} images. Ready for Auto-Annotation.`,
        status: 'ok',
        msgType: 'log',
      });
      
      // Stop the regular pipeline processing state, as we switch to the Annotation UI
      stopProcessing();
      
    } catch (err) {
      addTerminalLog({
        id: `upload-err-${Date.now()}`,
        timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
        message: `Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`,
        status: 'error',
        msgType: 'error',
      });
      stopProcessing();
    }
  };

  const isRunning = isProcessing && !isPaused;

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-5 backdrop-blur-sm">
        <div className="grid grid-cols-1 items-end gap-4 lg:grid-cols-4 mb-4">
          <div className="lg:col-span-1">
            <label className="mb-2 block text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              Dataset Query
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />
              <input
                ref={inputRef}
                type="text"
                value={datasetQuery}
                onChange={(e) => setDatasetQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isRunning && !isPaused) handleRun();
                }}
                placeholder="e.g. Apple Leaf Disease"
                disabled={isRunning}
                className="w-full rounded-lg border border-slate-800 bg-slate-950/80 py-2.5 pl-9 pr-3 font-mono text-xs text-slate-200 placeholder:text-slate-600 transition-all focus:outline-none focus:ring-1 focus:ring-teal-500/50 disabled:opacity-50"
              />
            </div>
          </div>

          <div>
            <label className="mb-2 block text-[10px] font-semibold uppercase tracking-widest text-slate-500">Source</label>
            <div className="relative">
              <select
                value={datasetSource}
                onChange={(e) => setDatasetSource(e.target.value as 'HuggingFace' | 'OpenImages')}
                disabled={isRunning}
                className="w-full appearance-none rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2.5 text-xs text-slate-200 transition-all focus:outline-none focus:ring-1 focus:ring-teal-500/50 disabled:opacity-50"
              >
                <option value="HuggingFace">All Sources (auto)</option>
                <option value="OpenImages">HuggingFace only</option>
              </select>
              <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />
            </div>
          </div>

          <div>
            <label className="mb-2 block text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              Target: <span className="text-teal-400">{targetSize.toLocaleString()} imgs</span>
            </label>
            <input
              type="range"
              min={100}
              max={5000}
              step={100}
              value={targetSize}
              onChange={(e) => setTargetSize(Number(e.target.value))}
              disabled={isRunning}
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-800 disabled:opacity-50 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-teal-400"
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleRun}
              disabled={isRunning || isPaused || !datasetQuery.trim()}
              className="btn-glow flex flex-1 items-center justify-center gap-2 rounded-lg border border-teal-500/30 bg-teal-500/15 px-4 py-2.5 text-xs font-semibold text-teal-400 transition-all hover:bg-teal-500/25 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Play className="h-3.5 w-3.5" />
              {isRunning ? 'Running...' : isPaused ? 'Paused' : 'Execute'}
            </button>
            <button
              onClick={handleReset}
              disabled={isRunning}
              className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2.5 text-xs text-slate-400 transition-all hover:bg-slate-800 disabled:opacity-40"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-center">
          <div className="h-px flex-1 bg-slate-800" />
          <span className="px-4 text-[10px] uppercase tracking-widest text-slate-500 font-bold">OR</span>
          <div className="h-px flex-1 bg-slate-800" />
        </div>

        <div 
          className={`mt-4 flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-colors ${
            isDragging ? 'border-teal-400 bg-teal-400/10' : 'border-slate-700 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/60'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload className={`mb-3 h-8 w-8 ${isDragging ? 'text-teal-400' : 'text-slate-500'}`} />
          <p className="text-sm font-semibold text-slate-300">Drag & Drop a local dataset ZIP</p>
          <p className="mt-1 text-[11px] text-slate-500">Unstructured raw images. We will Auto-Annotate them.</p>
          
          <label className="mt-4 cursor-pointer rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 transition-colors hover:bg-slate-700">
            Browse Files
            <input type="file" accept=".zip" className="hidden" onChange={handleFileSelect} />
          </label>
        </div>
      </div>

      {isPaused && pendingQuestion && (
        <div className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-4 backdrop-blur-sm">
          {pendingQuestion.type === 'clarification' ? (
            <ClarificationPanel question={pendingQuestion} onReply={handleReply} />
          ) : (
            <DatasetSelectionPanel question={pendingQuestion} onSelect={handleReply} />
          )}
        </div>
      )}

      {isPaused && !pendingQuestion && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
          <p className="font-mono text-xs text-amber-300/80">
            Agent is waiting for your input. See the terminal above.
          </p>
        </div>
      )}
    </div>
  );
}
