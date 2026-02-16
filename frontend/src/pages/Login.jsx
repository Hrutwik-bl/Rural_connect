import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, user } = useAuth();
  const navigate = useNavigate();

  // Redirect if already logged in
  React.useEffect(() => {
    if (user) {
      const dashboardMap = {
        citizen: '/citizen-dashboard',
        department: '/department-dashboard',
        admin: '/admin-dashboard'
      };
      navigate(dashboardMap[user.role] || '/');
    }
  }, [user, navigate]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await login(formData.email, formData.password);
    
    if (result.success) {
      // Navigation will be handled by useEffect
    } else {
      setError(result.message);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-100 flex items-center justify-center px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-4xl flex rounded-2xl shadow-elevated overflow-hidden border border-stone-200">
        {/* Left decorative panel */}
        <div className="hidden md:flex md:w-5/12 flex-col justify-center items-center p-10 text-white relative" style={{ background: 'linear-gradient(135deg, #065f46 0%, #059669 100%)' }}>
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-10 right-10 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
            <div className="absolute bottom-10 left-10 w-40 h-40 bg-amber-400/10 rounded-full blur-2xl"></div>
          </div>
          <div className="relative z-10 text-center">
            <div className="w-16 h-16 bg-white/15 rounded-2xl flex items-center justify-center mx-auto mb-6 backdrop-blur-sm border border-white/20">
              <span className="text-3xl">🌿</span>
            </div>
            <h3 className="text-2xl font-bold mb-3">Welcome Back</h3>
            <p className="text-emerald-200 text-sm leading-relaxed">Sign in to access your dashboard and continue making a difference in your community.</p>
          </div>
        </div>

        {/* Right form panel */}
        <div className="w-full md:w-7/12 bg-white p-8 md:p-10">
          <h2 className="text-2xl font-extrabold text-slate-800 mb-1">Login</h2>
          <p className="text-slate-400 text-sm mb-6">Enter your credentials to continue</p>
          
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-5 flex items-center gap-2 text-sm">
              <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
              {error}
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Email</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="Enter your email"
                className="w-full px-4 py-2.5 bg-stone-50 border border-stone-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all text-sm"
              />
            </div>

            <div>
              <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Enter your password"
                className="w-full px-4 py-2.5 bg-stone-50 border border-stone-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all text-sm"
              />
            </div>

            <button 
              type="submit"
              disabled={loading}
              className="w-full bg-emerald-600 text-white font-bold py-3 rounded-xl hover:bg-emerald-700 hover:shadow-lg hover:shadow-emerald-600/20 transition-all disabled:opacity-50 text-sm"
            >
              {loading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className="text-center mt-6">
            <p className="text-slate-500 text-sm">
              Don't have an account? <Link to="/register" className="text-emerald-600 hover:text-emerald-700 font-bold">Register here</Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;

