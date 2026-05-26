import { NavLink, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  Layers,
  Workflow,
  BarChart3,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Zap,
  LogOut,
  Wand2,
  ScanSearch,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { clearToken } from '../../lib/api';

interface SidebarProps {
  expanded: boolean;
  onToggle: () => void;
}

const navItems = [
  { to: '/',                  icon: LayoutDashboard, label: 'Dashboard',       end: true  },
  { to: '/dataset',           icon: Database,        label: 'Datasets',        end: false },
  { to: '/annotator',         icon: Layers,          label: 'Annotator',       end: false },
  { to: '/quick-annotator',   icon: ScanSearch,      label: 'Quick Annotate',  end: false },
  { to: '/augmentation',      icon: Wand2,           label: 'Augmentation',    end: false },
  { to: '/builder',           icon: Workflow,        label: 'Builder',         end: false },
  { to: '/analytics',         icon: BarChart3,       label: 'Analytics',       end: false },
  { to: '/library',           icon: BookOpen,        label: 'Library',         end: false },
  { to: '/settings',          icon: Settings,        label: 'Settings',        end: false },
];

export default function Sidebar({ expanded, onToggle }: SidebarProps) {
  const { logout, user } = useAppStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    clearToken();
    logout();
    navigate('/login', { replace: true });
  };

  return (
    <aside
      className={`
        relative flex flex-col
        ${expanded ? 'w-56' : 'w-16'}
        h-full border-r border-slate-800/60
        bg-slate-950/80 backdrop-blur-xl
        transition-all duration-300 ease-out
        z-40
      `}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-slate-800/50">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500/20 to-teal-500/5 border border-teal-500/30 shrink-0">
          <Cpu className="w-4 h-4 text-teal-400" />
        </div>
        {expanded && (
          <div className="animate-fade-up overflow-hidden">
            <span className="text-sm font-semibold tracking-tight text-slate-100 block">CVAgent</span>
            <span className="text-[10px] text-slate-500 tracking-wider uppercase">Infrastructure</span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-1 px-2 py-4" aria-label="Main navigation">
        {navItems.map(({ to, icon: Icon, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            title={!expanded ? label : undefined}
            className={({ isActive }) =>
              `group flex items-center gap-3 px-3 py-2.5 rounded-lg
               transition-all duration-200
               ${
                 isActive
                   ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20'
                   : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 border border-transparent'
               }`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {expanded && (
              <span className="text-xs font-medium tracking-wide animate-fade-up">{label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom section */}
      <div className="px-2 pb-4 space-y-2">
        {/* User chip */}
        {expanded && user && (
          <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-slate-900/50 border border-slate-800/50 animate-fade-up">
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-teal-500/40 to-teal-400/10 border border-teal-500/30 flex items-center justify-center shrink-0">
              <span className="text-[9px] font-bold text-teal-300">{user.avatarInitials}</span>
            </div>
            <div className="overflow-hidden">
              <p className="text-[11px] font-medium text-slate-300 truncate">{user.name}</p>
              <p className="text-[9px] text-slate-600 uppercase tracking-wider">{user.role}</p>
            </div>
          </div>
        )}

        {/* Version */}
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-900/50 border border-slate-800/50">
          <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          {expanded && (
            <span className="text-[10px] text-slate-400 font-mono animate-fade-up">v1.0.0-beta</span>
          )}
        </div>

        {/* Logout */}
        <button
          id="sidebar-logout-btn"
          onClick={handleLogout}
          title={!expanded ? 'Logout' : undefined}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all duration-200"
        >
          <LogOut className="w-4 h-4 shrink-0" />
          {expanded && <span className="text-xs font-medium animate-fade-up">Logout</span>}
        </button>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        id="sidebar-toggle-btn"
        className="
          absolute -right-3 top-1/2 -translate-y-1/2
          w-6 h-6 rounded-full
          bg-slate-800 border border-slate-700
          flex items-center justify-center
          text-slate-400 hover:text-teal-400
          hover:border-teal-500/50
          transition-all duration-200
          z-50 shadow-lg
        "
        aria-label="Toggle sidebar"
      >
        {expanded ? <ChevronLeft className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>
    </aside>
  );
}
