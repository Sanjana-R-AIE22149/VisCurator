import { useEffect, useState } from 'react';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';
import { useAppStore, type Toast as ToastType } from '../../store/useAppStore';
import { SOUND_MAP } from '../../lib/sounds';

const TOAST_ICONS = {
  success: <CheckCircle2 className="w-5 h-5 text-emerald-400" />,
  error: <AlertCircle className="w-5 h-5 text-red-400" />,
  warning: <AlertTriangle className="w-5 h-5 text-amber-400" />,
  info: <Info className="w-5 h-5 text-teal-400" />,
};

const TOAST_STYLES = {
  success: 'border-emerald-500/20 bg-emerald-500/5',
  error: 'border-red-500/20 bg-red-500/5',
  warning: 'border-amber-500/20 bg-amber-500/5',
  info: 'border-teal-500/20 bg-teal-500/5',
};

function ToastItem({ toast }: { toast: ToastType }) {
  const removeToast = useAppStore((s) => s.removeToast);
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    // Play notification sound immediately when toast appears
    SOUND_MAP[toast.type]?.();

    const duration = 4000;
    const step = 100;
    const interval = setInterval(() => {
      setProgress((prev) => Math.max(0, prev - (step / duration) * 100));
    }, step);

    const timer = setTimeout(() => {
      removeToast(toast.id);
    }, duration);

    return () => {
      clearInterval(interval);
      clearTimeout(timer);
    };
  }, [toast.id, removeToast]);

  return (
    <div
      className={`
        relative w-80 p-4 rounded-xl border backdrop-blur-xl
        flex gap-3 items-start animate-fade-in-right overflow-hidden
        ${TOAST_STYLES[toast.type]}
      `}
    >
      <div className="shrink-0 mt-0.5">{TOAST_ICONS[toast.type]}</div>
      <div className="flex-1">
        <p className="text-sm text-slate-200 font-medium leading-tight">
          {toast.message}
        </p>
      </div>
      <button
        onClick={() => removeToast(toast.id)}
        className="shrink-0 p-1 rounded-md hover:bg-white/5 transition-colors text-slate-500 hover:text-slate-300"
      >
        <X className="w-4 h-4" />
      </button>

      {/* Progress bar */}
      <div className="absolute bottom-0 left-0 h-0.5 bg-white/10 w-full" />
      <div
        className={`absolute bottom-0 left-0 h-0.5 transition-all duration-100 ease-linear ${
          toast.type === 'success' ? 'bg-emerald-500' :
          toast.type === 'error' ? 'bg-red-500' :
          toast.type === 'warning' ? 'bg-amber-500' : 'bg-teal-500'
        }`}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}

export default function ToastContainer() {
  const toasts = useAppStore((s) => s.toasts);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-6 right-6 z-[9999] flex flex-col gap-3">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  );
}
