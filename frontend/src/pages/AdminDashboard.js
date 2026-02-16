import React, { useState, useEffect } from 'react';
import apiClient from '../api/axiosConfig';
import { 
  BarChart, Bar, PieChart, Pie, LineChart, Line, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const AdminDashboard = () => {
  const [complaints, setComplaints] = useState([]);
  const [users, setUsers] = useState([]);
  const [pendingApprovals, setPendingApprovals] = useState([]);
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [progressUpdates, setProgressUpdates] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [showViewModal, setShowViewModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchComplaints();
    fetchUsers();
    fetchPendingApprovals();
  }, []);

  const fetchComplaints = async () => {
    try {
      console.log('Fetching complaints...');
      const response = await apiClient.get('/api/complaints/all');
      console.log('Complaints received:', response.data);
      setComplaints(response.data);
      setError('');
    } catch (error) {
      console.error('Error fetching complaints:', error);
      setError(`Failed to fetch complaints: ${error.message}`);
      setComplaints([]);
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await apiClient.get('/api/users');
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchPendingApprovals = async () => {
    try {
      const response = await apiClient.get('/api/users/pending-approvals');
      setPendingApprovals(response.data);
    } catch (error) {
      console.error('Error fetching pending approvals:', error);
      setError('Failed to load pending approvals. Please refresh the page.');
    }
  };

  const handleApproveUser = async (userId) => {
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await apiClient.put(`/api/users/${userId}/approve`);
      setSuccess('Department user approved successfully!');
      fetchPendingApprovals();
      fetchUsers();
    } catch (error) {
      setError(error.response?.data?.message || 'Failed to approve user');
    } finally {
      setLoading(false);
    }
  };

  const handleRejectUser = async (userId) => {
    if (!window.confirm('Are you sure you want to reject this department registration?')) {
      return;
    }
    setLoading(true);
    try {
      await apiClient.put(`/api/users/${userId}/reject`);
      setSuccess('Department registration rejected');
      fetchPendingApprovals();
    } catch (error) {
      setError(error.response?.data?.message || 'Failed to reject user');
    } finally {
      setLoading(false);
    }
  };

  const handleEscalate = async (complaintId) => {
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      await apiClient.put(`/api/complaints/${complaintId}/escalate`);
      setSuccess('Complaint escalated successfully!');
      fetchComplaints();
    } catch (error) {
      setError(error.response?.data?.message || 'Failed to escalate complaint');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (complaintId) => {
    if (!window.confirm('Are you sure you want to delete this complaint?')) {
      return;
    }

    setLoading(true);
    try {
      await apiClient.delete(`/api/complaints/${complaintId}`);
      setSuccess('Complaint deleted successfully!');
      fetchComplaints();
    } catch (error) {
      setError(error.response?.data?.message || 'Failed to delete complaint');
    } finally {
      setLoading(false);
    }
  };

  const viewComplaintDetails = (complaint) => {
    setSelectedComplaint(complaint);
    fetchProgressUpdates(complaint._id);
    setShowViewModal(true);
  };

  const fetchProgressUpdates = async (complaintId) => {
    try {
      const response = await apiClient.get(`/api/complaints/${complaintId}/progress`);
      setProgressUpdates(response.data);
    } catch (error) {
      console.error('Error fetching progress updates:', error);
    }
  };

  const getStatusBadge = (status) => {
    const colors = {
      'Pending': 'bg-amber-100 text-amber-700',
      'In Progress': 'bg-sky-100 text-sky-700',
      'Resolved': 'bg-emerald-100 text-emerald-700',
      'Rejected': 'bg-red-100 text-red-700',
      'Escalated': 'bg-red-100 text-red-700'
    };
    return <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${colors[status] || 'bg-stone-100 text-stone-600'}`}>{status}</span>;
  };

  const getImageSrc = (complaint) => {
    if (!complaint?.imageData) return null;
    if (complaint.imageData.startsWith('data:')) return complaint.imageData;
    const type = complaint.imageType || 'image/jpeg';
    return `data:${type};base64,${complaint.imageData}`;
  };

  const complaintStats = {
    total: complaints.length,
    pending: complaints.filter(c => c.status === 'Pending').length,
    inProgress: complaints.filter(c => c.status === 'In Progress').length,
    resolved: complaints.filter(c => c.status === 'Resolved').length,
    escalated: complaints.filter(c => c.escalated).length
  };

  const userStats = {
    total: users.length,
    citizens: users.filter(u => u.role === 'citizen').length,
    departments: users.filter(u => u.role === 'department').length,
    admins: users.filter(u => u.role === 'admin').length
  };

  const complaintsByCategory = complaints.reduce((acc, complaint) => {
    acc[complaint.category] = (acc[complaint.category] || 0) + 1;
    return acc;
  }, {});

  const COLORS = ['#059669', '#d97706', '#0ea5e9', '#8b5cf6', '#14b8a6', '#ef4444'];

  const categoryData = Object.entries(complaintsByCategory).map(([name, value]) => ({
    name,
    value,
    complaints: value
  }));

  const departmentData = [
    { name: 'Water', complaints: complaints.filter(c => c.category === 'Water').length },
    { name: 'Electricity', complaints: complaints.filter(c => c.category === 'Electricity').length },
    { name: 'Roads', complaints: complaints.filter(c => c.category === 'Roads').length }
  ];

  const tabClass = (tabName) => `px-5 py-2.5 font-semibold border-b-2 cursor-pointer transition-all text-sm ${
    activeTab === tabName 
      ? 'border-emerald-600 text-emerald-600' 
      : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-stone-300'
  }`;

  return (
    <div className="w-full min-h-screen bg-stone-100 py-4" style={{ paddingLeft: '1rem', paddingRight: '1rem' }}>
      <div className="mb-4">
        <h2 className="text-3xl font-extrabold mb-1 text-slate-800">Admin Dashboard</h2>
        <p className="text-slate-500 text-sm">System overview and management</p>
      </div>

      {success && (
        <div className="mb-4 p-4 bg-emerald-50 border border-emerald-300 text-emerald-700 rounded-xl flex justify-between items-center">
          <span>{success}</span>
          <button onClick={() => setSuccess('')} className="font-bold text-emerald-500 hover:text-emerald-700">&times;</button>
        </div>
      )}
      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-300 text-red-700 rounded-xl flex justify-between items-center">
          <span>{error}</span>
          <button onClick={() => setError('')} className="font-bold text-red-500 hover:text-red-700">&times;</button>
        </div>
      )}

      <div className="border-b border-stone-200 mb-4">
        <div className="flex gap-1">
          <button onClick={() => setActiveTab('overview')} className={tabClass('overview')}>
            Overview {pendingApprovals.length > 0 && <span className="ml-2 inline-block px-2 py-0.5 bg-red-500 text-white text-xs rounded-full font-bold">{pendingApprovals.length}</span>}
          </button>
          <button onClick={() => setActiveTab('complaints')} className={tabClass('complaints')}>
            All Complaints
          </button>
          <button onClick={() => setActiveTab('approvals')} className={tabClass('approvals')}>
            Pending Approvals {pendingApprovals.length > 0 && <span className="ml-2 inline-block px-2 py-0.5 bg-red-500 text-white text-xs rounded-full font-bold">{pendingApprovals.length}</span>}
          </button>
          <button onClick={() => setActiveTab('users')} className={tabClass('users')}>
            Users
          </button>
        </div>
      </div>

      {activeTab === 'overview' && (
        <>
          {pendingApprovals.length > 0 && (
            <div className="mb-4 p-4 bg-amber-50 border border-amber-300 text-amber-700 rounded-xl">
              <div className="font-bold">⚠️ Pending Department Approvals</div>
              <p className="mt-1 text-sm">You have {pendingApprovals.length} department registration(s) waiting for approval.</p>
              <button 
                onClick={() => setActiveTab('approvals')}
                className="mt-2 px-3 py-1.5 border border-amber-500 text-amber-600 rounded-lg hover:bg-amber-100 text-sm font-medium transition"
              >
                Review Now
              </button>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
              <div className="flex items-center gap-3">
                <span className="text-2xl">📋</span>
                <div>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Total Complaints</p>
                  <div className="text-3xl font-extrabold text-emerald-600">{complaintStats.total}</div>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
              <div className="flex items-center gap-3">
                <span className="text-2xl">⏳</span>
                <div>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Pending</p>
                  <div className="text-3xl font-extrabold text-amber-500">{complaintStats.pending}</div>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
              <div className="flex items-center gap-3">
                <span className="text-2xl">🔄</span>
                <div>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">In Progress</p>
                  <div className="text-3xl font-extrabold text-sky-500">{complaintStats.inProgress}</div>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
              <div className="flex items-center gap-3">
                <span className="text-2xl">🚨</span>
                <div>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Escalated</p>
                  <div className="text-3xl font-extrabold text-red-500">{complaintStats.escalated}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white rounded-2xl shadow-card border border-stone-200">
              <div className="px-6 py-4 border-b border-stone-100">
                <h5 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Complaints by Category</h5>
              </div>
              <div className="p-6">
                {categoryData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={categoryData}
                        cx="50%"
                        cy="50%"
                        labelLine={true}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        innerRadius={45}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {categoryData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => `${value} complaints`} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-center text-slate-400">No data available</p>
                )}
              </div>
            </div>
            <div className="bg-white rounded-2xl shadow-card border border-stone-200">
              <div className="px-6 py-4 border-b border-stone-100">
                <h5 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Department Performance</h5>
              </div>
              <div className="p-6">
                {departmentData.some(d => d.complaints > 0) ? (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={departmentData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                      <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 12 }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 12 }} />
                      <Tooltip formatter={(value) => `${value} complaints`} />
                      <Bar dataKey="complaints" fill="#059669" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-center text-slate-400">No data available</p>
                )}
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-card border border-stone-200">
            <div className="px-6 py-4 border-b border-stone-100">
              <h5 className="text-sm font-bold text-slate-700 uppercase tracking-wider">User Distribution</h5>
            </div>
            <div className="p-4">
              <table className="w-full">
                <tbody>
                  <tr className="border-b border-stone-100">
                    <td className="font-semibold py-3 px-2 text-sm text-slate-600">Total Users</td>
                    <td className="text-right py-3"><span className="inline-block px-3 py-1 bg-emerald-100 text-emerald-700 text-sm rounded-lg font-bold">{userStats.total}</span></td>
                  </tr>
                  <tr className="border-b border-stone-100">
                    <td className="font-semibold py-3 px-2 text-sm text-slate-600">Citizens</td>
                    <td className="text-right py-3"><span className="inline-block px-3 py-1 bg-sky-100 text-sky-700 text-sm rounded-lg font-bold">{userStats.citizens}</span></td>
                  </tr>
                  <tr className="border-b border-stone-100">
                    <td className="font-semibold py-3 px-2 text-sm text-slate-600">Departments</td>
                    <td className="text-right py-3"><span className="inline-block px-3 py-1 bg-amber-100 text-amber-700 text-sm rounded-lg font-bold">{userStats.departments}</span></td>
                  </tr>
                  <tr>
                    <td className="font-semibold py-3 px-2 text-sm text-slate-600">Admins</td>
                    <td className="text-right py-3"><span className="inline-block px-3 py-1 bg-red-100 text-red-700 text-sm rounded-lg font-bold">{userStats.admins}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {activeTab === 'complaints' && (
        <div className="bg-white rounded-2xl shadow-card border border-stone-200">
          <div className="px-6 py-4 border-b border-stone-100">
            <h5 className="text-sm font-bold text-slate-700 uppercase tracking-wider">All Complaints</h5>
          </div>
          <div className="p-4">
            {error && (
              <div className="bg-red-50 border border-red-300 text-red-700 px-4 py-3 rounded-xl mb-4 text-sm">
                {error}
              </div>
            )}
            {complaints.length === 0 ? (
              <div className="text-center py-8">
                <span className="text-3xl block mb-2">📭</span>
                <p className="text-slate-400">No complaints found</p>
              </div>
            ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">ID</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Title</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Category</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Citizen</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Priority</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Status</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Date</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {complaints.map((complaint) => (
                    <tr key={complaint._id} className="hover:bg-emerald-50/40 transition">
                      <td className="py-2.5 px-3 text-slate-500 font-mono text-xs">{complaint._id.slice(-6)}</td>
                      <td className="py-2.5 px-3 font-medium text-slate-700">{complaint.title}</td>
                      <td className="py-2.5 px-3 text-slate-600">{complaint.category}</td>
                      <td className="py-2.5 px-3 text-slate-600">{complaint.citizen?.name || 'Unknown'}</td>
                      <td className="py-2.5 px-3">
                        <span className={`priority-${(complaint.priority || 'low').toLowerCase()}`}>
                          {complaint.priority || 'Low'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">{getStatusBadge(complaint.status)}</td>
                      <td className="py-2.5 px-3 text-slate-500">{new Date(complaint.createdAt).toLocaleDateString()}</td>
                      <td className="py-2.5 px-3">
                        <div className="flex gap-1">
                          <button 
                            onClick={() => viewComplaintDetails(complaint)}
                            className="px-2.5 py-1 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 text-xs font-medium transition"
                          >
                            View
                          </button>
                          {!complaint.escalated && (
                            <button 
                              onClick={() => handleEscalate(complaint._id)}
                              disabled={loading}
                              className="px-2.5 py-1 bg-amber-100 text-amber-700 rounded-lg hover:bg-amber-200 text-xs font-medium disabled:opacity-40 transition"
                            >
                              Escalate
                            </button>
                          )}
                          <button 
                            onClick={() => handleDelete(complaint._id)}
                            disabled={loading}
                            className="px-2.5 py-1 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 text-xs font-medium disabled:opacity-40 transition"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'approvals' && (
        <div className="bg-white rounded-2xl shadow-card border border-stone-200">
          <div className="px-6 py-4 border-b border-stone-100">
            <h5 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Department Registrations Pending Approval</h5>
          </div>
          <div className="p-4">
            {pendingApprovals.length === 0 ? (
              <div className="text-center py-8">
                <span className="text-3xl block mb-2">✅</span>
                <p className="text-slate-400">No pending approvals</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-slate-50">
                      <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Name</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Email</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Department</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Phone</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Registered</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-stone-100">
                    {pendingApprovals.map((user) => (
                      <tr key={user._id} className="hover:bg-emerald-50/40 transition">
                        <td className="py-2.5 px-3 font-medium text-slate-700">{user.name}</td>
                        <td className="py-2.5 px-3 text-slate-600">{user.email}</td>
                        <td className="py-2.5 px-3"><span className="inline-block px-2.5 py-1 bg-sky-100 text-sky-700 text-xs rounded-lg font-semibold">{user.department}</span></td>
                        <td className="py-2.5 px-3 text-slate-600">{user.phone || '-'}</td>
                        <td className="py-2.5 px-3 text-slate-500">{new Date(user.createdAt).toLocaleDateString()}</td>
                        <td className="py-2.5 px-3">
                          <div className="flex gap-1">
                            <button 
                              onClick={() => handleApproveUser(user._id)}
                              disabled={loading}
                              className="px-2.5 py-1 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 text-xs font-medium disabled:opacity-40 transition"
                            >
                              Approve
                            </button>
                            <button 
                              onClick={() => handleRejectUser(user._id)}
                              disabled={loading}
                              className="px-2.5 py-1 bg-red-500 text-white rounded-lg hover:bg-red-600 text-xs font-medium disabled:opacity-40 transition"
                            >
                              Reject
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'users' && (
        <div className="bg-white rounded-2xl shadow-card border border-stone-200">
          <div className="px-6 py-4 border-b border-stone-100">
            <h5 className="text-sm font-bold text-slate-700 uppercase tracking-wider">👤 All Users</h5>
          </div>
          <div className="p-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Name</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Email</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Role</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Department</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Phone</th>
                    <th className="text-left py-3 px-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Registered</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {users.map((user) => (
                    <tr key={user._id} className="hover:bg-emerald-50/40 transition">
                      <td className="py-2.5 px-3 font-medium text-slate-700">{user.name}</td>
                      <td className="py-2.5 px-3 text-slate-600">{user.email}</td>
                      <td className="py-2.5 px-3">
                        <span className={`inline-block px-2.5 py-1 text-xs rounded-lg font-semibold ${
                          user.role === 'admin' ? 'bg-red-100 text-red-700' : 
                          user.role === 'department' ? 'bg-emerald-100 text-emerald-700' : 
                          'bg-sky-100 text-sky-700'
                        }`}>
                          {user.role}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-600">{user.department || '-'}</td>
                      <td className="py-2.5 px-3 text-slate-600">{user.phone || '-'}</td>
                      <td className="py-2.5 px-3 text-slate-500">{new Date(user.createdAt).toLocaleDateString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-elevated max-w-2xl w-full mx-4 max-h-96 overflow-y-auto border border-stone-200">
            <div className="px-6 py-4 border-b border-stone-100 flex justify-between items-center sticky top-0 bg-white rounded-t-2xl z-10">
              <h5 className="text-lg font-bold text-slate-800">Complaint Details</h5>
              <button 
                onClick={() => setShowModal(false)}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 text-slate-400 hover:text-slate-600 transition text-xl font-bold"
              >
                &times;
              </button>
            </div>
            <div className="p-6">
              {selectedComplaint && (
                <div>
                  <h5 className="text-lg font-bold text-slate-800 mb-3">{selectedComplaint.title}</h5>
                  <hr className="my-3 border-stone-100" />
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                    <div className="space-y-2">
                      <p className="text-sm"><span className="font-semibold text-slate-500">Category:</span> <span className="text-slate-700">{selectedComplaint.category}</span></p>
                      <p className="text-sm">
                        <span className="font-semibold text-slate-500">Priority:</span> 
                        <span className={`ml-2 priority-${(selectedComplaint.priority || 'low').toLowerCase()}`}>
                          {selectedComplaint.priority || 'Low'}
                        </span>
                      </p>
                      {selectedComplaint.severity && (
                        <p className="text-sm"><span className="font-semibold text-slate-500">Severity:</span> <span className="text-slate-700">{selectedComplaint.severity}</span></p>
                      )}
                      <p className="text-sm"><span className="font-semibold text-slate-500">Status:</span> {getStatusBadge(selectedComplaint.status)}</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm"><span className="font-semibold text-slate-500">Citizen:</span> <span className="text-slate-700">{selectedComplaint.citizen?.name || 'N/A'}</span></p>
                      <p className="text-sm"><span className="font-semibold text-slate-500">Email:</span> <span className="text-slate-700">{selectedComplaint.citizen?.email || 'N/A'}</span></p>
                      <p className="text-sm"><span className="font-semibold text-slate-500">Phone:</span> <span className="text-slate-700">{selectedComplaint.citizen?.phone || 'N/A'}</span></p>
                    </div>
                  </div>
                  <hr className="my-3 border-stone-100" />
                  <p className="font-semibold text-slate-500 mb-1.5 text-sm">Description:</p>
                  <p className="mb-3 text-slate-700 bg-stone-50 p-3 rounded-xl text-sm">{selectedComplaint.description}</p>
                  {selectedComplaint.imageData && (
                    <>
                      <hr className="my-3 border-stone-100" />
                      <p className="font-semibold text-slate-500 mb-2 text-sm">Photo Evidence:</p>
                      <img
                        src={getImageSrc(selectedComplaint)}
                        alt="Complaint"
                        className="w-64 h-64 object-contain rounded-xl mb-3 border border-stone-200"
                      />
                    </>
                  )}
                  {selectedComplaint.location && (
                    <p className="mb-3 text-sm"><span className="font-semibold text-slate-500">Location:</span> <span className="text-slate-700">{selectedComplaint.location}</span></p>
                  )}
                  {selectedComplaint.remarks && (
                    <>
                      <hr className="my-3 border-stone-100" />
                      <p className="font-semibold text-slate-500 mb-1.5 text-sm">Remarks:</p>
                      <p className="mb-3 text-slate-700 bg-emerald-50 p-3 rounded-xl text-sm">{selectedComplaint.remarks}</p>
                    </>
                  )}

                  <hr className="my-3 border-stone-100" />
                  <div className="bg-stone-50 p-4 rounded-xl mb-3">
                    <h6 className="font-bold mb-2 text-slate-700 text-sm">📝 Progress Updates:</h6>
                    {progressUpdates.length === 0 ? (
                      <p className="text-slate-400 mb-0 text-sm">No updates yet</p>
                    ) : (
                      <div className="max-h-72 overflow-y-auto">
                        {progressUpdates.map((update, idx) => (
                          <div key={idx} className="border-l-4 border-emerald-300 pl-3 mb-2 pb-2">
                            <div className="flex justify-between">
                              <strong className="text-sm text-slate-700">{update.updatedByName}</strong>
                              <small className="text-slate-400">
                                {new Date(update.timestamp).toLocaleString()}
                              </small>
                            </div>
                            <small className="text-slate-500 block mb-1">({update.role})</small>
                            <p className="mb-0 text-sm text-slate-600">{update.message}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <hr className="my-3 border-stone-100" />
                  <p className="text-sm text-slate-500"><span className="font-semibold">Created:</span> {new Date(selectedComplaint.createdAt).toLocaleString()}</p>
                  {selectedComplaint.escalated && (
                    <p className="text-red-600 font-bold mt-2 text-sm">⚠️ This complaint has been escalated</p>
                  )}
                </div>
              )}
            </div>
            <div className="px-6 py-4 border-t border-stone-100 flex justify-end">
              <button 
                onClick={() => setShowModal(false)}
                className="px-5 py-2 bg-stone-100 text-slate-600 rounded-xl hover:bg-stone-200 transition font-semibold text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* View Complaint Modal */}
      {showViewModal && selectedComplaint && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-elevated border border-stone-200">
            <div className="px-6 py-4 border-b border-stone-100 flex justify-between items-center sticky top-0 bg-white rounded-t-2xl z-10">
              <h3 className="text-lg font-bold text-slate-800">Complaint Details</h3>
              <button onClick={() => setShowViewModal(false)} className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 text-slate-400 hover:text-slate-600 transition">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-5">
              <div className="pb-4 border-b border-stone-100">
                <h4 className="text-xl font-bold text-slate-800 mb-2">{selectedComplaint.title}</h4>
                {getStatusBadge(selectedComplaint.status)}
                {selectedComplaint.escalated && (
                  <span className="ml-2 px-3 py-1 rounded-full text-sm font-semibold bg-red-100 text-red-700">
                    Escalated
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Category</label>
                  <p className="text-slate-700 font-medium">{selectedComplaint.category}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Location</label>
                  <p className="text-slate-700 font-medium">{selectedComplaint.location}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Priority</label>
                  <span className={`priority-${(selectedComplaint.priority || 'low').toLowerCase()}`}>
                    {selectedComplaint.priority || 'Low'}
                  </span>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Date Submitted</label>
                  <p className="text-slate-700 font-medium">{new Date(selectedComplaint.createdAt).toLocaleString()}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Submitted By</label>
                  <p className="text-slate-700 font-medium">{selectedComplaint.citizen?.name || selectedComplaint.user?.name || 'N/A'}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Contact Email</label>
                  <p className="text-slate-700 font-medium">{selectedComplaint.citizen?.email || selectedComplaint.user?.email || 'N/A'}</p>
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5">Description</label>
                <p className="text-slate-700 bg-stone-50 p-4 rounded-xl">{selectedComplaint.description}</p>
              </div>

              {selectedComplaint.remarks && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5">Department Remarks</label>
                  <p className="text-slate-700 bg-emerald-50 p-4 rounded-xl border-l-4 border-emerald-500">{selectedComplaint.remarks}</p>
                </div>
              )}

              {selectedComplaint.imageData && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Attached Image</label>
                  <div className="rounded-xl p-4 bg-stone-50 border border-stone-200">
                    <img 
                      src={getImageSrc(selectedComplaint)} 
                      alt="Complaint" 
                      className="w-64 h-64 object-contain rounded-lg cursor-pointer hover:opacity-90 transition"
                      onClick={() => window.open(getImageSrc(selectedComplaint), '_blank')}
                    />
                    <p className="text-xs text-slate-400 mt-2 text-center">Click image to view full size</p>
                  </div>
                </div>
              )}

              {selectedComplaint.severity && (
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Severity</label>
                  <p className="text-slate-700 font-medium">{selectedComplaint.severity}</p>
                </div>
              )}
            </div>

            <div className="px-6 py-4 border-t border-stone-100 flex justify-end gap-2">
              <button
                onClick={() => setShowViewModal(false)}
                className="px-6 py-2.5 bg-stone-100 text-slate-600 rounded-xl hover:bg-stone-200 transition font-semibold text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
