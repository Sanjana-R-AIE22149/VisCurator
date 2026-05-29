import { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import ProtectedRoute from './components/auth/ProtectedRoute';
import AppLayout from './components/layout/AppLayout';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import DatasetPage from './pages/DatasetPage';
import BuilderPage from './pages/BuilderPage';
import AnalyticsPage from './pages/AnalyticsPage';
import LibraryPage from './pages/LibraryPage';
import QuickAnnotatorPage from './pages/QuickAnnotatorPage';
import BootScreen from './components/ui/BootScreen';

/* ── Settings placeholder (stub) ── */
function SettingsPage() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center space-y-2">
        <p className="text-[11px] font-mono text-slate-600 uppercase tracking-widest">Coming soon</p>
        <h2 className="text-2xl font-bold text-slate-700">Settings</h2>
      </div>
    </div>
  );
}

export default function App() {
  // Show boot screen once per browser session (disappears after refresh-free navigation)
  const [booting, setBooting] = useState<boolean>(
    () => !sessionStorage.getItem('vc_booted')
  );

  function handleBootDone() {
    sessionStorage.setItem('vc_booted', '1');
    setBooting(false);
  }

  return (
    <>
      {booting && <BootScreen onDone={handleBootDone} />}

      <Routes>
        {/* ── Public ── */}
        <Route path="/login" element={<LoginPage />} />

        {/* ── Protected ── */}
        <Route element={<ProtectedRoute />}>
          <Route element={<AppLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="dataset" element={<DatasetPage />} />
            <Route path="quick-annotator" element={<QuickAnnotatorPage />} />
            <Route path="builder" element={<BuilderPage />} />
            <Route path="analytics" element={<AnalyticsPage />} />
            <Route path="library" element={<LibraryPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Route>

        {/* ── Catch-all ── */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}
