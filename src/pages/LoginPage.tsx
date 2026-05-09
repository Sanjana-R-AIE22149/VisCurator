import { useState, useEffect } from 'react';
import { useNavigate, useLocation, Navigate } from 'react-router-dom';
import { Cpu, Eye, EyeOff, Loader2, Sparkles, ShieldCheck, Zap } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import type { AppUser } from '../store/useAppStore';
import { apiLogin, setToken } from '../lib/api';

/* ── Floating particle effect ── */
const PARTICLES = Array.from({ length: 24 }, (_, i) => ({
  id: i,
  x: Math.random() * 100,
  y: Math.random() * 100,
  size: Math.random() * 2 + 1,
  delay: Math.random() * 4,
  duration: Math.random() * 6 + 6,
}));

export default function LoginPage() {
  const { isAuthenticated, login } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? '/';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(t);
  }, []);

  if (isAuthenticated) {
    return <Navigate to={from} replace />;
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      const data = await apiLogin(username, password);
      setToken(data.access_token);
      const user: AppUser = {
        id: data.username,
        name: data.name,
        email: `${data.username}@viscurator`,
        role: data.role,
        avatarInitials: data.name.slice(0, 2).toUpperCase(),
      };
      login(user);
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full bg-slate-950 overflow-hidden flex items-center justify-center">
      {/* ── Radial gradient background glows ── */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute -top-40 -left-40 w-[700px] h-[700px] rounded-full opacity-20"
          style={{
            background: 'radial-gradient(circle, rgba(45,212,191,0.35) 0%, transparent 70%)',
          }}
        />
        <div
          className="absolute -bottom-60 -right-40 w-[600px] h-[600px] rounded-full opacity-15"
          style={{
            background: 'radial-gradient(circle, rgba(167,139,250,0.4) 0%, transparent 70%)',
          }}
        />
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] rounded-full opacity-[0.04]"
          style={{
            background: 'radial-gradient(circle, rgba(56,189,248,0.6) 0%, transparent 70%)',
          }}
        />
      </div>

      {/* ── Animated particles ── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {PARTICLES.map((p) => (
          <div
            key={p.id}
            className="absolute rounded-full bg-teal-400/30"
            style={{
              left: `${p.x}%`,
              top: `${p.y}%`,
              width: `${p.size}px`,
              height: `${p.size}px`,
              animation: `float-particle ${p.duration}s ${p.delay}s ease-in-out infinite alternate`,
            }}
          />
        ))}
      </div>

      {/* ── Grid lines ── */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(45,212,191,0.8) 1px, transparent 1px), linear-gradient(90deg, rgba(45,212,191,0.8) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* ── Top brand strip ── */}
      <div className="absolute top-6 left-8 flex items-center gap-2.5">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500/30 to-teal-400/10 border border-teal-500/40">
          <Cpu className="w-4 h-4 text-teal-400" />
        </div>
        <div>
          <span className="text-sm font-semibold text-slate-200 tracking-tight">CVAgent</span>
          <span className="block text-[10px] text-slate-500 tracking-widest uppercase">AI Infrastructure</span>
        </div>
      </div>

      {/* ── Feature pills (top-right) ── */}
      <div className="absolute top-6 right-8 flex items-center gap-2 hidden md:flex">
        {[
          { icon: Sparkles, label: 'Auto-Annotate' },
          { icon: ShieldCheck, label: 'Quality Guard' },
          { icon: Zap, label: 'GPU Accelerated' },
        ].map(({ icon: Icon, label }) => (
          <div
            key={label}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900/60 border border-slate-800/70 backdrop-blur-sm"
          >
            <Icon className="w-3 h-3 text-teal-400" />
            <span className="text-[10px] font-medium text-slate-400">{label}</span>
          </div>
        ))}
      </div>

      {/* ── Main frosted-glass card ── */}
      <div
        className="relative w-full max-w-md mx-4"
        style={{
          transform: mounted ? 'translateY(0)' : 'translateY(20px)',
          opacity: mounted ? 1 : 0,
          transition: 'transform 0.6s cubic-bezier(0.16,1,0.3,1), opacity 0.6s ease',
        }}
      >
        {/* Outer glow ring */}
        <div
          className="absolute -inset-px rounded-2xl opacity-60"
          style={{
            background: 'linear-gradient(135deg, rgba(45,212,191,0.3) 0%, transparent 50%, rgba(167,139,250,0.2) 100%)',
          }}
        />

        <div className="relative rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 p-8 shadow-2xl">
          {/* ── Card header ── */}
          <div className="mb-8 text-center">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-500/20 to-teal-400/5 border border-teal-500/30 mb-4 shadow-lg shadow-teal-500/10">
              <Cpu className="w-7 h-7 text-teal-400" />
            </div>
            <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Welcome back</h1>
            <p className="text-sm text-slate-500 mt-1.5">
              Sign in to your CVAgent workspace
            </p>
          </div>

          {/* ── Form ── */}
          <form onSubmit={handleLogin} className="space-y-4" id="login-form">
            {/* Error banner */}
            {error && (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-xs text-red-400">
                {error}
              </div>
            )}

            {/* Username */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-400 uppercase tracking-wider" htmlFor="login-username">
                Username
              </label>
              <input
                id="login-username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-slate-800/60 border border-slate-700/60 text-slate-200 text-sm placeholder-slate-600 focus:outline-none focus:border-teal-500/60 focus:ring-1 focus:ring-teal-500/30 transition-all duration-200"
                placeholder="admin"
                required
                autoComplete="username"
              />
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-400 uppercase tracking-wider" htmlFor="login-password">
                Password
              </label>
              <div className="relative">
                <input
                  id="login-password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-11 rounded-xl bg-slate-800/60 border border-slate-700/60 text-slate-200 text-sm placeholder-slate-600 focus:outline-none focus:border-teal-500/60 focus:ring-1 focus:ring-teal-500/30 transition-all duration-200"
                  placeholder="••••••••••"
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  id="toggle-password-visibility"
                  onClick={() => setShowPassword((p) => !p)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                  tabIndex={-1}
                  aria-label="Toggle password visibility"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Remember me + Forgot */}
            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 cursor-pointer group">
                <input
                  type="checkbox"
                  id="remember-me"
                  className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 accent-teal-500"
                  defaultChecked
                />
                <span className="text-xs text-slate-500 group-hover:text-slate-400 transition-colors">
                  Remember me
                </span>
              </label>
              <button type="button" id="forgot-password-btn" className="text-xs text-teal-500 hover:text-teal-400 transition-colors">
                Forgot password?
              </button>
            </div>

            {/* Submit */}
            <button
              id="login-submit-btn"
              type="submit"
              disabled={isLoading}
              className="w-full mt-2 py-3 px-4 rounded-xl font-semibold text-sm text-slate-950 
                         bg-gradient-to-r from-teal-400 to-teal-500
                         hover:from-teal-300 hover:to-teal-400
                         disabled:opacity-60 disabled:cursor-not-allowed
                         transition-all duration-200 shadow-lg shadow-teal-500/25
                         flex items-center justify-center gap-2
                         btn-glow"
              style={{ letterSpacing: '0.02em' }}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Authenticating…</span>
                </>
              ) : (
                'Sign in to CVAgent'
              )}
            </button>
          </form>

          {/* ── Divider + SSO hint ── */}
          <div className="mt-6 flex items-center gap-3">
            <div className="flex-1 h-px bg-slate-800/80" />
            <span className="text-[11px] text-slate-600">or continue with</span>
            <div className="flex-1 h-px bg-slate-800/80" />
          </div>

          <div className="mt-4 flex gap-2">
            {['GitHub', 'Google'].map((provider) => (
              <button
                key={provider}
                id={`sso-${provider.toLowerCase()}-btn`}
                type="button"
                className="flex-1 py-2.5 rounded-xl border border-slate-800 bg-slate-900/60 text-slate-400 text-xs font-medium hover:border-slate-700 hover:text-slate-300 transition-all duration-200"
              >
                {provider}
              </button>
            ))}
          </div>

          {/* ── Footer ── */}
          <p className="mt-6 text-center text-[11px] text-slate-600">
            No account?{' '}
            <button type="button" id="request-access-btn" className="text-teal-500 hover:text-teal-400 transition-colors font-medium">
              Request access
            </button>
          </p>
        </div>
      </div>

      {/* ── Bottom copyright ── */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2">
        <p className="text-[10px] text-slate-700 tracking-widest uppercase">
          © 2026 CVAgent AI Infrastructure · All rights reserved
        </p>
      </div>

      {/* ── Inline keyframes for particles ── */}
      <style>{`
        @keyframes float-particle {
          0%   { transform: translateY(0px) scale(1); opacity: 0.3; }
          100% { transform: translateY(-30px) scale(1.4); opacity: 0.6; }
        }
      `}</style>
    </div>
  );
}
