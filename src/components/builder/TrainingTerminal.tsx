import { useEffect, useRef } from 'react';
import { useAppStore, type TerminalLog } from '../../store/useAppStore';

function getLogColor(log: TerminalLog): string {
  if (log.msgType) {
    switch (log.msgType) {
      case 'thought': return 'text-violet-400';
      case 'tool_call': return 'text-sky-400';
      case 'tool_result': return 'text-teal-400';
      case 'script_log': return 'text-emerald-300 font-mono';
      case 'done': return 'text-emerald-400';
      case 'error': return 'text-red-400';
      default: break;
    }
  }

  switch (log.status) {
    case 'ok': return 'text-emerald-400';
    case 'warn': return 'text-amber-400';
    case 'error': return 'text-red-400';
    default: return 'text-slate-400';
  }
}

export default function TrainingTerminal() {
  const { trainingLogs, isTraining, trainingRunId } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [trainingLogs]);

  return (
    <div className="h-64 rounded-xl border border-slate-800/60 bg-[#0c0c0c] overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-slate-900/60 border-b border-slate-800/50">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
        </div>
        <span className="ml-2 text-[10px] font-mono text-slate-500 uppercase tracking-widest">
          {trainingRunId ? `train://${trainingRunId}` : 'train://idle'}
        </span>
        {isTraining && (
          <div className="ml-auto flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
            <span className="text-[10px] font-mono text-teal-400">streaming</span>
          </div>
        )}
      </div>

      <div ref={scrollRef} className="h-[calc(16rem-42px)] overflow-y-auto p-4 font-mono text-xs leading-relaxed flex flex-col">
        {trainingLogs.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center space-y-3 opacity-40">
            <div className="w-12 h-12 rounded-full border border-slate-800 flex items-center justify-center">
              <div className="w-1.5 h-4 bg-teal-400 rounded-sm" />
            </div>
            <div className="text-center">
              <p className="text-[11px] font-bold text-slate-300 uppercase tracking-widest mb-1">
                Awaiting Training Run
              </p>
              <p className="text-[10px] text-slate-600">
                Start a builder run to stream metrics here
              </p>
            </div>
          </div>
        ) : (
          trainingLogs.map((log) => (
            <div key={log.id} className="flex gap-3 animate-fade-up">
              <span className="text-slate-600 shrink-0 select-none">{log.timestamp}</span>
              <span className={getLogColor(log)}>{log.message}</span>
            </div>
          ))
        )}
        {trainingLogs.length > 0 && isTraining && (
          <div className="mt-1">
            <span className="terminal-cursor text-slate-500" />
          </div>
        )}
      </div>
    </div>
  );
}
