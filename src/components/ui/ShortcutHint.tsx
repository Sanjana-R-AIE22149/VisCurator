import { X, Keyboard } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export default function ShortcutHint() {
  const { shortcutHintDismissed, dismissShortcutHint } = useAppStore();
  const showShortcutHint = !shortcutHintDismissed;

  if (!showShortcutHint) return null;

  return (
    <div className="fixed bottom-20 right-8 z-50 animate-fade-up">
      <div className="relative w-64 bg-slate-900/90 border border-slate-800 p-4 rounded-xl backdrop-blur-md shadow-2xl overflow-hidden group">
        <div className="absolute top-0 left-0 w-1 h-full bg-sky-500/50" />
        
        <button
          onClick={dismissShortcutHint}
          className="absolute top-2 right-2 p-1 rounded-md hover:bg-white/5 text-slate-600 hover:text-slate-400 transition-colors"
        >
          <X className="w-3 h-3" />
        </button>

        <div className="flex items-center gap-2 mb-3">
          <Keyboard className="w-4 h-4 text-sky-400" />
          <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">Quick Commands</span>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-500">Go to Builder</span>
            <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">^B</kbd>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-500">Go to Dataset</span>
            <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">^D</kbd>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-500">Search Dataset</span>
            <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">^K</kbd>
          </div>
        </div>
      </div>
    </div>
  );
}
