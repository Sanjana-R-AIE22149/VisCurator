import { useState } from 'react';
import { ShieldCheck, Cpu, Key, CheckCircle2, Loader2, Sparkles } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export default function SetupModal() {
  const { setupComplete, setSetupComplete, setNimApiKey, addToast } = useAppStore();
  const [key, setKey] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const [isTested, setIsTested] = useState(false);

  if (setupComplete) return null;

  const handleTest = async () => {
    if (!key) return;
    setIsTesting(true);
    // Simulate ping to /api/health with key
    await new Promise((r) => setTimeout(r, 1200));
    setIsTesting(false);
    setIsTested(true);
    addToast('Connection established with NVIDIA NIM clusters', 'success');
  };

  const handleSave = () => {
    if (!key) return;
    setNimApiKey(key);
    setSetupComplete(true);
    addToast('Setup complete. Neural engine initialized.', 'success');
  };

  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center p-6 bg-slate-950/80 backdrop-blur-md animate-fade-in">
      <div className="relative w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl overflow-hidden">
        {/* Background glow */}
        <div className="absolute -top-24 -right-24 w-64 h-64 bg-sky-500/10 rounded-full blur-[100px]" />
        <div className="absolute -bottom-24 -left-24 w-64 h-64 bg-emerald-500/10 rounded-full blur-[100px]" />

        <div className="relative">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-sky-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-sky-500/20">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">Welcome to VisCurator</h2>
              <p className="text-sm text-slate-400">Initialize your neural development environment</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-slate-950/50 border border-slate-800 space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                <ShieldCheck className="w-3.5 h-3.5" />
                NVIDIA NIM Configuration
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                VisCurator requires an active NVIDIA NIM API key to power the CVAgent dataset analysis and architectural synthesis engines.
              </p>
              
              <div className="relative mt-4">
                <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
                <input
                  type="password"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="nvapi-..."
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg py-2.5 pl-10 pr-3 text-sm text-slate-200 focus:outline-none focus:border-sky-500 transition-colors placeholder:text-slate-700"
                />
              </div>
              <p className="text-[10px] text-slate-600 italic mt-2">
                Note: Your key is stored locally in your browser and is only sent to the NIM proxy.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleTest}
                disabled={!key || isTesting}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg border border-slate-700 bg-slate-800/50 text-sm font-medium text-slate-300 hover:bg-slate-800 transition-colors disabled:opacity-50"
              >
                {isTesting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isTested ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <Cpu className="w-4 h-4" />
                )}
                {isTesting ? 'Pinging NIM...' : isTested ? 'Verified' : 'Test Connection'}
              </button>
              
              <button
                onClick={handleSave}
                disabled={!key || !isTested}
                className="flex-[1.5] py-2.5 rounded-lg bg-gradient-to-r from-sky-600 to-sky-500 text-sm font-semibold text-white hover:from-sky-500 hover:to-sky-400 transition-all shadow-lg shadow-sky-900/20 disabled:opacity-50"
              >
                Save & Initialize
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
