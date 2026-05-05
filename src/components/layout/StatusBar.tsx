import { useEffect, useState, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { Wifi, Monitor, Clock, Activity, ChevronRight, Cpu, HardDrive } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

/* ── Breadcrumb map ── */
const ROUTE_LABELS: Record<string, string> = {
  '':          'Dashboard',
  'dataset':   'Datasets',
  'builder':   'Builder',
  'analytics': 'Analytics',
  'library':   'Library',
  'settings':  'Settings',
};

function Breadcrumbs() {
  const { pathname } = useLocation();

  const segments = pathname.split('/').filter(Boolean);
  const crumbs = [
    { label: 'CVAgent', path: '/' },
    ...segments.map((seg, i) => ({
      label: ROUTE_LABELS[seg] ?? seg,
      path: '/' + segments.slice(0, i + 1).join('/'),
    })),
  ];

  return (
    <nav className="flex items-center gap-1" aria-label="Breadcrumb">
      {crumbs.map((crumb, i) => (
        <span key={crumb.path} className="flex items-center gap-1">
          {i > 0 && <ChevronRight className="w-3 h-3 text-slate-700" />}
          {i === crumbs.length - 1 ? (
            <span className="text-[11px] font-medium text-slate-400">{crumb.label}</span>
          ) : (
            <Link
              to={crumb.path}
              className="text-[11px] text-slate-600 hover:text-slate-400 transition-colors font-mono"
            >
              {crumb.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}

export default function StatusBar() {
  const { hardwareStats, updateHardwareStats } = useAppStore();
  const [time, setTime] = useState(new Date());
  const [ping, setPing] = useState(12);
  const [showHardware, setShowHardware] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date());
      setPing(Math.floor(Math.random() * 8) + 8);
      
      // Update hardware stats with slight variations
      updateHardwareStats({
        vramUsed: Math.max(0, Math.min(hardwareStats.vramTotal, hardwareStats.vramUsed + (Math.random() * 0.4 - 0.2))),
        cpuUsage: Math.max(0, Math.min(100, hardwareStats.cpuUsage + (Math.random() * 4 - 2))),
        ramUsage: Math.max(0, Math.min(100, hardwareStats.ramUsage + (Math.random() * 2 - 1))),
      });
    }, 2000);
    return () => clearInterval(interval);
  }, [hardwareStats, updateHardwareStats]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setShowHardware(false);
      }
    };
    if (showHardware) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showHardware]);

  return (
    <header
      id="status-bar"
      className="flex items-center justify-between h-10 px-4 border-b border-slate-800/60 bg-slate-950/90 backdrop-blur-md shrink-0 z-30"
    >
      {/* Left — Breadcrumbs */}
      <div className="flex items-center gap-4">
        <Breadcrumbs />
        <div className="w-px h-4 bg-slate-800" />
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 status-pulse" />
          <span className="text-[11px] font-mono text-slate-500">
            System Status: <span className="text-emerald-400 font-medium">Online</span>
          </span>
        </div>
      </div>

      {/* Right — System stats */}
      <div className="flex items-center gap-4 relative">
        <div className="hidden md:flex items-center gap-1.5">
          <Monitor className="w-3 h-3 text-slate-500" />
          <span className="text-[10px] font-mono text-slate-500">CUDA 12.4 · PyTorch 2.3</span>
        </div>
        <div className="w-px h-4 bg-slate-800 hidden md:block" />
        
        {/* Hardware Monitor Toggle */}
        <button 
          onClick={() => setShowHardware(!showHardware)}
          className={`flex items-center gap-1.5 hover:bg-white/5 px-2 py-1 rounded transition-colors ${showHardware ? 'bg-white/5' : ''}`}
        >
          <Activity className="w-3 h-3 text-teal-500" />
          <span className="text-[10px] font-mono text-slate-500">
            VRAM: {hardwareStats.vramUsed.toFixed(1)}/{hardwareStats.vramTotal} GB
          </span>
        </button>

        {/* Hardware Popover */}
        {showHardware && (
          <div 
            ref={popoverRef}
            className="absolute top-full right-0 mt-1 w-64 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl p-4 z-50 animate-fade-in"
          >
            <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
              <Cpu className="w-4 h-4 text-sky-400" />
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Infrastructure Stats</span>
            </div>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-slate-500 uppercase tracking-tighter">GPU Model</span>
                  <span className="text-sky-400 font-mono">{hardwareStats.gpuModel}</span>
                </div>
                <div className="flex justify-between text-[10px] mb-1.5">
                  <span className="text-slate-500 uppercase tracking-tighter">VRAM Utilization</span>
                  <span className="text-slate-300 font-mono">{((hardwareStats.vramUsed / hardwareStats.vramTotal) * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-sky-500 transition-all duration-1000" 
                    style={{ width: `${(hardwareStats.vramUsed / hardwareStats.vramTotal) * 100}%` }} 
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="flex justify-between text-[10px] mb-1.5">
                    <span className="text-slate-500 uppercase tracking-tighter">CPU Load</span>
                    <span className="text-slate-300 font-mono">{hardwareStats.cpuUsage.toFixed(0)}%</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-teal-500 transition-all duration-1000" style={{ width: `${hardwareStats.cpuUsage}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-[10px] mb-1.5">
                    <span className="text-slate-500 uppercase tracking-tighter">RAM Usage</span>
                    <span className="text-slate-300 font-mono">{hardwareStats.ramUsage.toFixed(0)}%</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-purple-500 transition-all duration-1000" style={{ width: `${hardwareStats.ramUsage}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="w-px h-4 bg-slate-800" />
        <div className="flex items-center gap-1.5">
          <Wifi className="w-3 h-3 text-teal-500" />
          <span className="text-[10px] font-mono text-slate-500">{ping}ms</span>
        </div>
        <div className="w-px h-4 bg-slate-800" />
        <div className="flex items-center gap-1.5">
          <Clock className="w-3 h-3 text-slate-500" />
          <span className="text-[10px] font-mono text-slate-500">
            {time.toLocaleTimeString('en-US', { hour12: false })}
          </span>
        </div>
      </div>
    </header>
  );
}
