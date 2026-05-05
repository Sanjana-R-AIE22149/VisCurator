import { useCallback, useEffect, useRef } from 'react';
import { useAppStore, type TerminalLog } from '../../store/useAppStore';

/* ── Mock pipeline messages (fallback when backend is offline) ── */
const PIPELINE_MESSAGES: Array<{ message: string; status: TerminalLog['status']; delay: number }> = [
  { message: '> Initializing CVAgent autonomous pipeline...', status: 'info', delay: 400 },
  { message: '> Connecting to data source API...', status: 'info', delay: 800 },
  { message: '> Authentication successful                           [ok]', status: 'ok', delay: 600 },
  { message: '> Querying dataset repository...', status: 'info', delay: 1000 },
  { message: '> Found 2,847 candidate images                       [ok]', status: 'ok', delay: 700 },
  { message: '> Downloading batch 1/6 (512 images)...', status: 'info', delay: 1200 },
  { message: '> Downloading batch 2/6 (512 images)...', status: 'info', delay: 900 },
  { message: '> Downloading batch 3/6 (512 images)...', status: 'info', delay: 800 },
  { message: '> Running Laplacian variance check...', status: 'info', delay: 1400 },
  { message: '> Filtered 42 blurry images (threshold < 80)         [filtered]', status: 'warn', delay: 600 },
  { message: '> Running duplicate hash detection (dHash)...', status: 'info', delay: 1100 },
  { message: '> Removed 17 near-duplicates (hamming dist < 6)      [cleaned]', status: 'warn', delay: 500 },
  { message: '> Initializing Grounded-SAM for segmentation...', status: 'info', delay: 1300 },
  { message: '> Loading ViT-H backbone weights...', status: 'info', delay: 900 },
  { message: '> SAM model loaded                                   [ok]', status: 'ok', delay: 500 },
  { message: '> Generating bounding box annotations...', status: 'info', delay: 1600 },
  { message: '> 1,847 / 2,788 images annotated                    [progress]', status: 'info', delay: 1200 },
  { message: '> 2,788 / 2,788 images annotated                    [complete]', status: 'ok', delay: 800 },
  { message: '> Running quality scoring (BRISQUE + NIQE)...', status: 'info', delay: 1000 },
  { message: '> 23 low-quality images flagged (score < 30)         [warn]', status: 'warn', delay: 500 },
  { message: '> Class distribution: balanced within 8% tolerance   [ok]', status: 'ok', delay: 600 },
  { message: '> Generating train/val/test splits (70/20/10)...', status: 'info', delay: 700 },
  { message: '> Writing COCO-format annotations to disk...', status: 'info', delay: 900 },
  { message: '> Dataset exported: ./output/dataset_v1/             [ok]', status: 'ok', delay: 500 },
  { message: '> ═══════════════════════════════════════════════════════', status: 'info', delay: 200 },
  { message: '> Pipeline complete. 2,788 curated images ready.     [done]', status: 'ok', delay: 300 },
  { message: '> Total time: 4m 32s | Rejected: 82 | Accepted: 2,788', status: 'ok', delay: 100 },
];

export default function LiveTerminal() {
  const { terminalLogs, isProcessing, addTerminalLog, jobId } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const hasRunRef = useRef(false);

  /* Auto-scroll terminal to bottom */
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [terminalLogs]);

  /* Run the simulated pipeline (fallback only — when no backend jobId) */
  const runMockPipeline = useCallback(() => {
    if (hasRunRef.current) return;
    hasRunRef.current = true;

    let cumulativeDelay = 0;

    PIPELINE_MESSAGES.forEach((msg, idx) => {
      cumulativeDelay += msg.delay;
      setTimeout(() => {
        addTerminalLog({
          id: `log-${idx}-${Date.now()}`,
          timestamp: new Date().toISOString().split('T')[1].slice(0, 12),
          message: msg.message,
          status: msg.status,
        });
      }, cumulativeDelay);
    });

    setTimeout(() => {
      hasRunRef.current = false;
    }, cumulativeDelay + 500);
  }, [addTerminalLog]);

  /* Start mock pipeline when isProcessing becomes true AND there's no real job */
  useEffect(() => {
    if (isProcessing && !jobId) {
      runMockPipeline();
    }
  }, [isProcessing, jobId, runMockPipeline]);

  /**
   * Color coding for terminal log entries.
   * Agent message types get distinct colors; regular logs use the
   * existing status-based palette.
   */
  const getLogColor = (log: TerminalLog): string => {
    // If the log carries an agent message type, use type-based coloring
    if (log.msgType) {
      switch (log.msgType) {
        case 'thought':     return 'text-violet-400';
        case 'tool_call':   return 'text-sky-400';
        case 'tool_result': return 'text-teal-400';
        case 'script_log':  return 'text-emerald-300 font-mono';  // raw subprocess stdout
        case 'done':        return 'text-emerald-400';
        case 'error':       return 'text-red-400';
        default:            break; // fall through to status-based
      }
    }

    // Fallback: status-based coloring (existing behaviour)
    switch (log.status) {
      case 'ok':    return 'text-emerald-400';
      case 'warn':  return 'text-amber-400';
      case 'error': return 'text-red-400';
      default:      return 'text-slate-400';
    }
  };

  return (
    <div className="flex flex-col rounded-xl border border-slate-800/60 bg-[#0c0c0c] overflow-hidden">
      {/* Terminal header bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 bg-slate-900/60 border-b border-slate-800/50">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
        </div>
        <span className="ml-2 text-[10px] font-mono text-slate-500 uppercase tracking-widest">
          cvagent://pipeline/live
        </span>
        {isProcessing && (
          <div className="ml-auto flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
            <span className="text-[10px] font-mono text-teal-400">streaming</span>
          </div>
        )}
      </div>

      {/* Terminal body */}
      <div
        ref={scrollRef}
        className="h-72 overflow-y-auto p-4 font-mono text-xs leading-relaxed flex flex-col"
      >
        {terminalLogs.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center space-y-4 opacity-40">
            <div className="relative">
              <div className="absolute inset-0 bg-sky-500/20 rounded-full blur-xl animate-pulse" />
              <div className="relative w-12 h-12 rounded-full border border-slate-800 flex items-center justify-center">
                <div className="w-1.5 h-4 bg-sky-400 animate-[pulse_1s_infinite] rounded-sm" />
              </div>
            </div>
            <div className="text-center">
              <p className="text-[11px] font-bold text-slate-300 uppercase tracking-widest mb-1">
                Awaiting Execution
              </p>
              <p className="text-[10px] text-slate-600">
                Pipeline parameters not yet initialized
              </p>
            </div>
          </div>
        ) : (
          terminalLogs.map((log) => (
            <div key={log.id} className="flex gap-3 animate-fade-up">
              <span className="text-slate-600 shrink-0 select-none">
                {log.timestamp}
              </span>
              <span className={getLogColor(log)}>
                {log.message}
              </span>
            </div>
          ))
        )}
        {terminalLogs.length > 0 && isProcessing && (
          <div className="mt-1">
            <span className="terminal-cursor text-slate-500" />
          </div>
        )}
      </div>
    </div>
  );
}
