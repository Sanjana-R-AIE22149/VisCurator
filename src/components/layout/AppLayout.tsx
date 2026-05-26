import { useEffect } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import StatusBar from './StatusBar';
import { useAppStore } from '../../store/useAppStore';
import ToastContainer from '../ui/Toast';
import SetupModal from '../onboarding/SetupModal';
import ShortcutHint from '../ui/ShortcutHint';
import PipelineWizard from '../ui/PipelineWizard';

export default function AppLayout() {

  const { sidebarOpen, setSidebarOpen } = useAppStore();
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl shortcuts
      if (e.ctrlKey || e.metaKey) {
        if (e.key.toLowerCase() === 'k') {
          e.preventDefault();
          if (location.pathname !== '/dataset') {
            navigate('/dataset');
          }
          setTimeout(() => {
            window.dispatchEvent(new CustomEvent('focus-dataset-query'));
          }, 100);
        }
        if (e.key.toLowerCase() === 'b') {
          e.preventDefault();
          navigate('/builder');
        }
        if (e.key.toLowerCase() === 'd') {
          e.preventDefault();
          navigate('/dataset');
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate, location.pathname]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0a0a0a]">
      {/* Global UI Overlays */}
      <ToastContainer />
      <SetupModal />
      <ShortcutHint />

      {/* Left Sidebar */}
      <Sidebar
        expanded={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Top Status Bar */}
        <StatusBar />

        {/* Pipeline Step Wizard — visible whenever a job is active */}
        <PipelineWizard />

        {/* Page Content */}
        <main className="flex-1 overflow-auto relative" id="main-content">
          <div 
            key={location.pathname}
            className="h-full animate-fade-up"
            style={{ animationDuration: '250ms' }}
          >
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
