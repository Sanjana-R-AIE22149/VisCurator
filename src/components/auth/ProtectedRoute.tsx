import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';
import { clearToken, getToken } from '../../lib/api';

/**
 * Wraps all protected routes. If the user is not authenticated, they are
 * redirected to /login, and the original path is preserved in location state
 * so the login page can redirect them back after a successful login.
 */
export default function ProtectedRoute() {
  const { isAuthenticated, logout } = useAppStore();
  const location = useLocation();
  const hasToken = Boolean(getToken());

  if (!isAuthenticated || !hasToken) {
    if (isAuthenticated && !hasToken) {
      clearToken();
      logout();
    }
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
}
