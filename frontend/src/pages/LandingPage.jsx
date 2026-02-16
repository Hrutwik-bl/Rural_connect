import React from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  return (
    <>
      {/* Hero Section */}
      <div className="relative overflow-hidden text-white" style={{ background: 'linear-gradient(135deg, #065f46 0%, #059669 40%, #047857 100%)' }}>
        {/* Decorative elements */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-20 left-10 w-72 h-72 bg-white/5 rounded-full blur-3xl"></div>
          <div className="absolute bottom-10 right-10 w-96 h-96 bg-amber-400/10 rounded-full blur-3xl"></div>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-400/5 rounded-full blur-3xl"></div>
        </div>
        
        <div className="container mx-auto px-4 relative z-10 py-28 md:py-36 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/10 border border-white/20 rounded-full text-sm font-medium mb-6 backdrop-blur-sm">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
            AI-Powered Rural Service Portal
          </div>
          <h1 className="font-black text-4xl md:text-6xl leading-tight max-w-4xl mx-auto tracking-tight" style={{ animation: 'fadeInUp 0.8s ease-out' }}>
            Welcome to Rural Service Request Portal
          </h1>
          <p className="text-lg md:text-xl mt-5 opacity-90 max-w-2xl mx-auto leading-relaxed" style={{ animation: 'fadeInUp 1s ease-out' }}>
            Your Voice Matters - Report Issues, Track Progress, Build Better Communities
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-4" style={{ animation: 'fadeInUp 1.2s ease-out' }}>
            <Link 
              to="/register"
              className="px-8 py-3.5 bg-amber-500 hover:bg-amber-400 text-slate-900 font-bold rounded-xl shadow-lg hover:shadow-amber-500/30 hover:-translate-y-0.5 transform transition-all duration-300 text-lg"
            >
              Get Started
            </Link>
            <Link 
              to="/login"
              className="px-8 py-3.5 bg-white/10 border-2 border-white/30 text-white font-bold rounded-xl shadow-lg hover:bg-white/20 hover:-translate-y-0.5 transform transition-all duration-300 backdrop-blur-sm text-lg"
            >
              Login
            </Link>
          </div>
        </div>
        {/* Bottom wave */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 80" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full">
            <path d="M0 80L1440 80L1440 40C1440 40 1200 0 720 0C240 0 0 40 0 40L0 80Z" fill="#f5f5f4"/>
          </svg>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-stone-100 py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <span className="inline-block px-4 py-1.5 bg-emerald-100 text-emerald-700 rounded-full text-sm font-bold tracking-wide uppercase mb-4">What We Offer</span>
            <h2 className="text-4xl md:text-5xl font-extrabold text-slate-800 mb-4 tracking-tight">
              Key Features
            </h2>
            <p className="text-slate-500 text-lg max-w-xl mx-auto">
              Empowering citizens and government departments to work together
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {[
              { icon: '📝', title: 'Easy Complaint Filing', desc: 'Submit complaints with photos and descriptions in just a few clicks.', color: 'emerald' },
              { icon: '📡', title: 'Real-Time Tracking', desc: 'Monitor complaint status with live updates and progress notifications.', color: 'sky' },
              { icon: '👥', title: 'Multi-Role Support', desc: 'Separate dashboards for citizens, departments, and admins.', color: 'violet' },
              { icon: '🤖', title: 'AI-Powered Classification', desc: 'Automatic complaint categorization using advanced AI.', color: 'amber' },
              { icon: '🔒', title: 'Secure & Private', desc: 'Your data is protected with industry-standard encryption.', color: 'red' },
              { icon: '📊', title: 'Analytics Dashboard', desc: 'Comprehensive insights and statistics to monitor performance.', color: 'teal' },
            ].map((feature, idx) => (
              <div key={idx} className="group bg-white p-8 rounded-2xl border border-stone-200 shadow-sm hover:shadow-xl hover:-translate-y-2 hover:border-emerald-200 transition-all duration-500">
                <div className={`w-14 h-14 rounded-xl bg-${feature.color}-100 flex items-center justify-center text-2xl mb-5 group-hover:scale-110 transition-transform duration-300`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-3">{feature.title}</h3>
                <p className="text-slate-500 leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)' }}>
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-72 h-72 bg-amber-500/10 rounded-full blur-3xl"></div>
        </div>
        <div className="container mx-auto px-4 text-center py-20 relative z-10">
          <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-5 tracking-tight">Ready to Make a Difference?</h2>
          <p className="text-xl text-slate-300 mb-10">Join thousands of citizens working together</p>
          <Link 
            to="/register"
            className="inline-block px-10 py-4 bg-emerald-600 text-white font-bold rounded-xl text-lg shadow-xl hover:bg-emerald-500 hover:-translate-y-1 hover:shadow-emerald-500/30 transform transition-all duration-300"
          >
            Create Your Account Now
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-10 border-t border-slate-800">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <span className="text-xl">🌿</span>
            <span className="font-bold text-lg text-emerald-400">RuralConnect</span>
          </div>
          <p className="text-slate-500 text-sm">© 2026 Rural Service Request Portal. All rights reserved.</p>
        </div>
      </footer>
    </>
  );
};

export default LandingPage;
