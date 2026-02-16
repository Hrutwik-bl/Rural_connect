import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Navigation = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = React.useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const getDashboardLink = () => {
    if (!user) return null;
    
    switch (user.role) {
      case 'citizen':
        return '/citizen-dashboard';
      case 'department':
        return '/department-dashboard';
      case 'admin':
        return '/admin-dashboard';
      default:
        return '/';
    }
  };

  return (
    <nav className="sticky top-0 z-50 shadow-lg" style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center no-underline group">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-emerald-700 rounded-xl flex items-center justify-center mr-3 shadow-md group-hover:shadow-emerald-500/30 transition-shadow">
              <span className="text-xl">🌿</span>
            </div>
            <span className="text-xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-emerald-200 hidden sm:inline tracking-tight">
              RuralConnect
            </span>
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-3">
            {user ? (
              <>
                <Link to={getDashboardLink()} className="px-3 py-1.5 text-slate-300 font-medium hover:text-emerald-400 transition-colors rounded-lg hover:bg-white/5">
                  Dashboard
                </Link>
                <div className="flex items-center px-3 text-slate-300">
                  <span className="font-medium">{user.name}</span>
                  <span className="ml-2 inline-block bg-emerald-500/20 text-emerald-400 text-xs font-bold px-2.5 py-1 rounded-full border border-emerald-500/30">
                    {user.role}
                  </span>
                </div>
                <button 
                  onClick={handleLogout}
                  className="ml-2 px-5 py-1.5 border border-slate-500 text-slate-300 rounded-lg hover:bg-red-500/10 hover:border-red-400 hover:text-red-400 transition-all font-medium text-sm"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className="px-4 py-1.5 text-slate-300 hover:text-emerald-400 transition-colors font-medium rounded-lg hover:bg-white/5">
                  Login
                </Link>
                <Link 
                  to="/register"
                  className="px-5 py-1.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 transition-all font-bold text-sm shadow-md hover:shadow-emerald-500/20"
                >
                  Register
                </Link>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-slate-300 hover:text-emerald-400 focus:outline-none transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden bg-slate-800/95 backdrop-blur-sm px-3 pt-3 pb-4 space-y-2 rounded-b-xl border-t border-slate-700">
            {user ? (
              <>
                <Link to={getDashboardLink()} className="block px-4 py-2.5 text-slate-300 hover:text-emerald-400 hover:bg-white/5 transition rounded-lg font-medium">
                  Dashboard
                </Link>
                <div className="px-4 py-2 text-slate-400 text-sm">
                  {user.name} <span className="ml-1 text-emerald-400 font-bold">({user.role})</span>
                </div>
                <button 
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2.5 text-slate-300 hover:text-red-400 hover:bg-red-500/10 transition rounded-lg font-medium"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className="block px-4 py-2.5 text-slate-300 hover:text-emerald-400 hover:bg-white/5 transition rounded-lg font-medium">
                  Login
                </Link>
                <Link to="/register" className="block px-4 py-2.5 bg-emerald-600 text-white rounded-lg font-bold text-center hover:bg-emerald-500 transition">
                  Register
                </Link>
              </>
            )}
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
