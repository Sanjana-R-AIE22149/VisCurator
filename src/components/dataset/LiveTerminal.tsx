import { useEffect, useRef } from 'react';
import { useAppStore, type TerminalLog } from '../../store/useAppStore';

export default function LiveTerminal() {
  const { terminalLogs, isProcessing, jobId } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  /* Auto-scroll terminal to bottom */
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [terminalLogs]);

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
                {isProcessing && !jobId ? 'Connecting to backend pipeline...' : 'Pipeline parameters not yet initialized'}
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
