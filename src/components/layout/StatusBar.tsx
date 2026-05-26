import { useEffect, useState, useRef } from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import { Wifi, Monitor, Clock, Activity, ChevronRight, Cpu, LogOut, Volume2, VolumeX } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { clearToken } from '../../lib/api';
import { isMuted, toggleMuted } from '../../lib/sounds';

const ROUTE_LABELS: Record<string, string> = {
  '': 'Dashboard',
  dataset: 'Datasets',
  annotator: 'Annotator',
  augmentation: 'Augmentation',
  builder: 'Builder',
  analytics: 'Analytics',
  library: 'Library',
  settings: 'Settings',
};

function Breadcrumbs() {
  const { pathname } = useLocation();
  const segments = pathname.split('/').filter(Boolean);
  const crumbs = [{ label: 'CVAgent', path: '/' }, ...segments.map((seg, i) => ({
    label: ROUTE_LABELS[seg] ?? seg,
    path: `/${segments.slice(0, i + 1).join('/')}`,
  }))];

  return (
    <nav className="flex items-center gap-1" aria-label="Breadcrumb">
      {crumbs.map((crumb, i) => (
        <span key={crumb.path} className="flex items-center gap-1">
          {i > 0 && <ChevronRight className="h-3 w-3 text-slate-700" />}
          {i === crumbs.length - 1 ? (
            <span className="text-[11px] font-medium text-slate-400">{crumb.label}</span>
          ) : (
            <Link to={crumb.path} className="font-mono text-[11px] text-slate-600 transition-colors hover:text-slate-400">
              {crumb.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}

export default function StatusBar() {
  const hardwareStats = useAppStore((s) => s.hardwareStats);
  const { user, logout } = useAppStore();
  const [time, setTime] = useState(new Date());
  const [showHardware, setShowHardware] = useState(false);
  const [muted, setMutedState] = useState(isMuted);
  const popoverRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const handleToggleMute = () => {
    const next = toggleMuted();
    setMutedState(next);
  };

  // Stay in sync if mute is toggled from another component
  useEffect(() => {
    const handler = (e: Event) => setMutedState((e as CustomEvent).detail.muted);
    window.addEventListener('vc-mute-change', handler);
    return () => window.removeEventListener('vc-mute-change', handler);
  }, []);

  const handleLogout = () => {
    clearToken();
    logout();
    navigate('/login', { replace: true });
  };

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setShowHardware(false);
      }
    };
    if (showHardware) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showHardware]);

  return (
    <header
      id="status-bar"
      className="z-30 flex h-10 shrink-0 items-center justify-between border-b border-slate-800/60 bg-slate-950/90 px-4 backdrop-blur-md"
    >
      <div className="flex items-center gap-4">
        <Breadcrumbs />
        <div className="h-4 w-px bg-slate-800" />
        <div className="flex items-center gap-2">
          <div className="status-pulse h-1.5 w-1.5 rounded-full bg-emerald-400" />
          <span className="font-mono text-[11px] text-slate-500">
            System Status: <span className="font-medium text-emerald-400">Online</span>
          </span>
        </div>
      </div>

      <div className="relative flex items-center gap-4">
        <div className="hidden items-center gap-1.5 md:flex">
          <Monitor className="h-3 w-3 text-slate-500" />
          <span className="font-mono text-[10px] text-slate-500">CUDA 12.4 · PyTorch 2.3</span>
        </div>
        <div className="hidden h-4 w-px bg-slate-800 md:block" />

        <button
          onClick={() => setShowHardware((current) => !current)}
          className={`flex items-center gap-1.5 rounded px-2 py-1 transition-colors hover:bg-white/5 ${showHardware ? 'bg-white/5' : ''}`}
        >
          <Activity className="h-3 w-3 text-teal-500" />
          <span className="font-mono text-[10px] text-slate-500">
            VRAM: {hardwareStats.vramUsed.toFixed(1)}/{hardwareStats.vramTotal} GB
          </span>
        </button>

        {showHardware && (
          <div
            ref={popoverRef}
            className="absolute right-0 top-full z-50 mt-1 w-64 rounded-xl border border-slate-800 bg-slate-900 p-4 shadow-2xl animate-fade-in"
          >
            <div className="mb-4 flex items-center gap-2 border-b border-slate-800 pb-2">
              <Cpu className="h-4 w-4 text-sky-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Infrastructure Stats</span>
            </div>

            <div className="space-y-4">
              <div>
                <div className="mb-1 flex justify-between text-[10px]">
                  <span className="uppercase tracking-tighter text-slate-500">GPU Model</span>
                  <span className="font-mono text-sky-400">{hardwareStats.gpuModel}</span>
                </div>
                <div className="mb-1.5 flex justify-between text-[10px]">
                  <span className="uppercase tracking-tighter text-slate-500">VRAM Utilization</span>
                  <span className="font-mono text-slate-300">{((hardwareStats.vramUsed / hardwareStats.vramTotal) * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-slate-800">
                  <div className="h-full bg-sky-500 transition-all duration-300" style={{ width: `${(hardwareStats.vramUsed / hardwareStats.vramTotal) * 100}%` }} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="mb-1.5 flex justify-between text-[10px]">
                    <span className="uppercase tracking-tighter text-slate-500">CPU Load</span>
                    <span className="font-mono text-slate-300">{hardwareStats.cpuPercent.toFixed(0)}%</span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-slate-800">
                    <div className="h-full bg-teal-500 transition-all duration-300" style={{ width: `${hardwareStats.cpuPercent}%` }} />
                  </div>
                </div>
                <div>
                  <div className="mb-1.5 flex justify-between text-[10px]">
                    <span className="uppercase tracking-tighter text-slate-500">RAM Usage</span>
                    <span className="font-mono text-slate-300">{hardwareStats.ramPercent.toFixed(0)}%</span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-slate-800">
                    <div className="h-full bg-purple-500 transition-all duration-300" style={{ width: `${hardwareStats.ramPercent}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="h-4 w-px bg-slate-800" />
        <div className="flex items-center gap-1.5">
          <Wifi className="h-3 w-3 text-teal-500" />
          <span className="font-mono text-[10px] text-slate-500">local</span>
        </div>
        <div className="h-4 w-px bg-slate-800" />

        {/* Sound mute toggle */}
        <button
          id="statusbar-mute-btn"
          onClick={handleToggleMute}
          title={muted ? 'Notifications muted — click to unmute' : 'Notifications on — click to mute'}
          className={`flex items-center gap-1 rounded px-2 py-1 transition-all duration-200 ${
            muted
              ? 'text-slate-600 hover:text-slate-400 hover:bg-white/5'
              : 'text-teal-500 hover:text-teal-400 hover:bg-white/5'
          }`}
        >
          {muted
            ? <VolumeX className="h-3 w-3" />
            : <Volume2 className="h-3 w-3" />}
          <span className="font-mono text-[10px]">{muted ? 'muted' : 'sound'}</span>
        </button>

        <div className="h-4 w-px bg-slate-800" />
        <div className="flex items-center gap-1.5">
          <Clock className="h-3 w-3 text-slate-500" />
          <span className="font-mono text-[10px] text-slate-500">
            {time.toLocaleTimeString('en-US', { hour12: false })}
          </span>
        </div>
        <div className="h-4 w-px bg-slate-800" />
        {user && (
          <button
            onClick={handleLogout}
            title={`Logout ${user.name}`}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-slate-500 transition-colors hover:bg-white/5 hover:text-red-400"
          >
            <LogOut className="h-3 w-3" />
            <span className="font-mono text-[10px]">{user.name}</span>
          </button>
        )}
      </div>
    </header>
  );
}
