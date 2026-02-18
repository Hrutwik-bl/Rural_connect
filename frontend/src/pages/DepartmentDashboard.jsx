import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../api/axiosConfig';
import { useAuth } from '../context/AuthContext';

const DepartmentDashboard = () => {
  const { user } = useAuth();
  const [complaints, setComplaints] = useState([]);
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [showViewModal, setShowViewModal] = useState(false);
  const [status, setStatus] = useState('');
  const [remarks, setRemarks] = useState('');
  const [loading, setLoading] = useState(false);
  const [resolvedImage, setResolvedImage] = useState(null);
  const [resolvedImagePreview, setResolvedImagePreview] = useState(null);
  const [showResolvedCamera, setShowResolvedCamera] = useState(false);
  const resolvedVideoRef = useRef(null);
  const resolvedStreamRef = useRef(null);
  const [imageError, setImageError] = useState('');

  useEffect(() => {
    fetchComplaints();
  }, []);

  const fetchComplaints = async () => {
    try {
      console.log('Fetching department complaints...');
      const response = await apiClient.get('/api/complaints/all');
      console.log('Department complaints received:', response.data);
      setComplaints(response.data);
    } catch (error) {
      console.error('Error fetching complaints:', error);
    }
  };

  // Camera functions for resolved image
  const startResolvedCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      resolvedStreamRef.current = stream;
      setShowResolvedCamera(true);
      setTimeout(() => {
        if (resolvedVideoRef.current) {
          resolvedVideoRef.current.srcObject = stream;
        }
      }, 100);
    } catch (err) {
      alert('Could not access camera. Please check permissions or use file upload.');
    }
  };

  const captureResolvedPhoto = () => {
    if (!resolvedVideoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = resolvedVideoRef.current.videoWidth;
    canvas.height = resolvedVideoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(resolvedVideoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    setResolvedImagePreview(dataUrl);
    // Create a dummy file-like object so the existing base64 conversion works
    setResolvedImage(null); // We'll use resolvedImagePreview directly
    stopResolvedCamera();
  };

  const stopResolvedCamera = () => {
    if (resolvedStreamRef.current) {
      resolvedStreamRef.current.getTracks().forEach(t => t.stop());
      resolvedStreamRef.current = null;
    }
    setShowResolvedCamera(false);
  };

  const handleUpdateStatus = async (e) => {
    e.preventDefault();
    setLoading(true);
    setImageError('');

    try {
      let resolvedLocationCoords = null;

      if (status === 'Resolved') {
        resolvedLocationCoords = await new Promise((resolve, reject) => {
          if (!navigator.geolocation) {
            reject(new Error('Location services are not supported by this browser.'));
            return;
          }

          navigator.geolocation.getCurrentPosition(
            (position) => {
              resolve({
                lat: position.coords.latitude,
                lng: position.coords.longitude
              });
            },
            (err) => {
              reject(new Error(err.message || 'Unable to get location.'));
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
          );
        });
      }

      // Convert resolved image to base64 if provided
      let resolvedImageData = null;
      let resolvedImageType = null;
      if (resolvedImage) {
        resolvedImageData = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(resolvedImage);
        });
        resolvedImageType = resolvedImage.type;
      } else if (resolvedImagePreview) {
        // Camera-captured image is already a data URL
        resolvedImageData = resolvedImagePreview;
        resolvedImageType = 'image/jpeg';
      }

      if (status === 'Resolved' && !resolvedImageData) {
        setImageError('Please upload or capture a photo of the resolved issue.');
        setLoading(false);
        return;
      }

      await apiClient.put(`/api/complaints/${selectedComplaint._id}/status`, {
        status,
        remarks,
        resolvedLocationCoords,
        resolvedImageData,
        resolvedImageType
      });
      setShowModal(false);
      setSelectedComplaint(null);
      setStatus('');
      setRemarks('');
      setResolvedImage(null);
      setResolvedImagePreview(null);
      setImageError('');
      stopResolvedCamera();
      fetchComplaints();
    } catch (error) {
      const message = error.response?.data?.message || error.message || 'Error updating complaint';
      if (error.response?.data?.imageVerification) {
        setImageError(message);
      } else {
        alert(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    const badges = {
      Pending: 'bg-yellow-100 text-yellow-800',
      'In Progress': 'bg-blue-100 text-blue-800',
      Resolved: 'bg-green-100 text-green-800',
      Rejected: 'bg-red-100 text-red-800'
    };
    return badges[status] || 'bg-gray-100 text-gray-800';
  };

  const getImageSrc = (complaint) => {
    if (!complaint?.imageData) return null;
    if (complaint.imageData.startsWith('data:')) return complaint.imageData;
    const type = complaint.imageType || 'image/jpeg';
    return `data:${type};base64,${complaint.imageData}`;
  };

  const openImageInNewTab = (src) => {
    const w = window.open('', '_blank');
    if (w) {
      w.document.write(`<html><head><title>Image Preview</title><style>body{margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#1a1a1a;}</style></head><body><img src="${src}" style="max-width:100%;max-height:100vh;object-fit:contain;"/></body></html>`);
      w.document.close();
    }
  };

  const openModal = (complaint) => {
    setSelectedComplaint(complaint);
    setStatus(complaint.status);
    setRemarks(complaint.remarks || '');
    setResolvedImage(null);
    setResolvedImagePreview(null);
    setImageError('');
    stopResolvedCamera();
    setShowModal(true);
  };

  return (
    <div className="min-h-screen bg-stone-100 py-8">
      <div className="container mx-auto px-4">
        <div className="bg-white rounded-2xl shadow-card p-6 mb-6 border border-stone-200">
          <h2 className="text-3xl font-extrabold text-slate-800">Department Dashboard</h2>
          <p className="text-slate-500 mt-1">{user?.department} Department — {user?.name}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-6">
          <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📋</span>
              <div>
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Total</h3>
                <p className="text-3xl font-extrabold text-emerald-600">{complaints.length}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
            <div className="flex items-center gap-3">
              <span className="text-2xl">⏳</span>
              <div>
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Pending</h3>
                <p className="text-3xl font-extrabold text-amber-500">
                  {complaints.filter(c => c.status === 'Pending').length}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🔄</span>
              <div>
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">In Progress</h3>
                <p className="text-3xl font-extrabold text-sky-500">
                  {complaints.filter(c => c.status === 'In Progress').length}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-card p-5 border border-stone-200 hover:shadow-card-hover transition-shadow">
            <div className="flex items-center gap-3">
              <span className="text-2xl">✅</span>
              <div>
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Resolved</h3>
                <p className="text-3xl font-extrabold text-emerald-600">
                  {complaints.filter(c => c.status === 'Resolved').length}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-card p-6 border border-stone-200">
          <h3 className="text-xl font-bold text-slate-800 mb-4">Active Complaints</h3>
          
          {complaints.filter(c => c.status !== 'Resolved').length === 0 ? (
            <div className="text-center py-12">
              <span className="text-4xl mb-3 block">📭</span>
              <p className="text-slate-400">No active complaints.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="px-4 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Title</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Location</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Status</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Date</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {complaints.filter(c => c.status !== 'Resolved').map(complaint => (
                    <tr key={complaint._id} className="hover:bg-emerald-50/40 transition">
                      <td className="px-4 py-3 font-medium text-slate-700">{complaint.title}</td>
                      <td className="px-4 py-3 text-slate-600">{complaint.location}</td>
                      <td className="px-4 py-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusBadge(complaint.status)}`}>
                          {complaint.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-500">
                        {new Date(complaint.createdAt).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <button
                            onClick={() => {
                              setSelectedComplaint(complaint);
                              setShowViewModal(true);
                            }}
                            className="px-3 py-1.5 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition text-sm font-medium"
                          >
                            View
                          </button>
                          <button
                            onClick={() => openModal(complaint)}
                            className="px-3 py-1.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition text-sm font-medium"
                          >
                            Update
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

        {/* Resolved Complaints Section */}
        <div className="bg-white rounded-2xl shadow-card p-6 border border-stone-200 mt-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-2xl">✅</span>
            <h3 className="text-xl font-bold text-slate-800">Resolved Complaints</h3>
            <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-bold">
              {complaints.filter(c => c.status === 'Resolved').length}
            </span>
          </div>
          
          {complaints.filter(c => c.status === 'Resolved').length === 0 ? (
            <div className="text-center py-12">
              <span className="text-4xl mb-3 block">📋</span>
              <p className="text-slate-400">No resolved complaints yet.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-emerald-50">
                    <th className="px-4 py-3 text-left text-xs font-bold text-emerald-600 uppercase tracking-wider">Title</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-emerald-600 uppercase tracking-wider">Location</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-emerald-600 uppercase tracking-wider">Status</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-emerald-600 uppercase tracking-wider">Date</th>
                    <th className="px-4 py-3 text-left text-xs font-bold text-emerald-600 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {complaints.filter(c => c.status === 'Resolved').map(complaint => (
                    <tr key={complaint._id} className="hover:bg-emerald-50/40 transition">
                      <td className="px-4 py-3 font-medium text-slate-700">{complaint.title}</td>
                      <td className="px-4 py-3 text-slate-600">{complaint.location}</td>
                      <td className="px-4 py-3">
                        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
                          ✅ Resolved
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-500">
                        {new Date(complaint.createdAt).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => {
                            setSelectedComplaint(complaint);
                            setShowViewModal(true);
                          }}
                          className="px-3 py-1.5 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition text-sm font-medium"
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

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
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getStatusBadge(selectedComplaint.status)}`}>
                  {selectedComplaint.status}
                </span>
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
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Date Submitted</label>
                  <p className="text-slate-700 font-medium">{new Date(selectedComplaint.createdAt).toLocaleString()}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Submitted By</label>
                  <p className="text-slate-700 font-medium">{selectedComplaint.citizen?.name || 'N/A'}</p>
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5">Description</label>
                <p className="text-slate-700 bg-stone-50 p-4 rounded-xl">{selectedComplaint.description}</p>
              </div>

              {selectedComplaint.remarks && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5">Remarks</label>
                  <p className="text-slate-700 bg-emerald-50 p-4 rounded-xl border-l-4 border-emerald-500">{selectedComplaint.remarks}</p>
                </div>
              )}

              {selectedComplaint.imageData && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Complaint Image</label>
                  <div className="rounded-xl p-4 bg-stone-50 border border-stone-200">
                    <img 
                      src={getImageSrc(selectedComplaint)} 
                      alt="Complaint" 
                      className="w-64 h-64 object-contain rounded-lg cursor-pointer hover:opacity-90 transition"
                      onClick={() => openImageInNewTab(getImageSrc(selectedComplaint))}
                    />
                    <p className="text-xs text-slate-400 mt-2 text-center">Click image to view full size</p>
                  </div>
                </div>
              )}

              {selectedComplaint.resolvedImageData && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Resolved Image</label>
                  <div className="rounded-xl p-4 bg-emerald-50 border border-emerald-200">
                    <img 
                      src={selectedComplaint.resolvedImageData.startsWith('data:') 
                        ? selectedComplaint.resolvedImageData 
                        : `data:${selectedComplaint.resolvedImageType || 'image/jpeg'};base64,${selectedComplaint.resolvedImageData}`} 
                      alt="Resolved" 
                      className="w-64 h-64 object-contain rounded-lg cursor-pointer hover:opacity-90 transition border-2 border-emerald-400"
                      onClick={() => openImageInNewTab(selectedComplaint.resolvedImageData.startsWith('data:') 
                        ? selectedComplaint.resolvedImageData 
                        : `data:${selectedComplaint.resolvedImageType || 'image/jpeg'};base64,${selectedComplaint.resolvedImageData}`)}
                    />
                    <p className="text-xs text-emerald-600 mt-2 text-center font-medium">Photo of resolved issue</p>
                  </div>
                </div>
              )}
            </div>

            <div className="px-6 py-4 border-t border-stone-100 flex justify-end">
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

      {/* Update Status Modal */}
      {showModal && selectedComplaint && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-elevated border border-stone-200">
            <div className="px-6 py-4 border-b border-stone-100 flex justify-between items-center sticky top-0 bg-white rounded-t-2xl z-10">
              <h3 className="text-lg font-bold text-slate-800">Update Complaint</h3>
              <button onClick={() => { stopResolvedCamera(); setImageError(''); setShowModal(false); }} className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 text-slate-400 hover:text-slate-600 transition">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6">
              <div className="mb-6 p-4 bg-stone-50 rounded-xl border border-stone-200">
                <div className="flex gap-4">
                  <div className="flex-1">
                    <h4 className="font-bold text-lg text-slate-800">{selectedComplaint.title}</h4>
                    <p className="text-slate-600 mt-2 text-sm">{selectedComplaint.description}</p>
                    <div className="flex gap-4 mt-3 text-xs text-slate-400">
                      <span>📍 {selectedComplaint.location}</span>
                      <span>📅 {new Date(selectedComplaint.createdAt).toLocaleDateString()}</span>
                    </div>
                  </div>
                  {selectedComplaint.imageData && (
                    <div className="flex-shrink-0">
                      <img 
                        src={getImageSrc(selectedComplaint)} 
                        alt="Complaint" 
                        className="w-64 h-64 object-contain rounded-xl border border-stone-200 cursor-pointer hover:border-emerald-400 transition"
                        onClick={() => openImageInNewTab(getImageSrc(selectedComplaint))}
                      />
                    </div>
                  )}
                </div>
              </div>

              <form onSubmit={handleUpdateStatus} className="space-y-4">
                <div>
                  <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Status</label>
                  <select
                    required
                    value={status}
                    onChange={(e) => setStatus(e.target.value)}
                    className="w-full px-4 py-2.5 bg-stone-50 border border-stone-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all text-sm"
                  >
                    <option value="Pending">Pending</option>
                    <option value="In Progress">In Progress</option>
                    <option value="Resolved">Resolved</option>
                    <option value="Rejected">Rejected</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Remarks</label>
                  <textarea
                    rows="4"
                    value={remarks}
                    onChange={(e) => setRemarks(e.target.value)}
                    className="w-full px-4 py-2.5 bg-stone-50 border border-stone-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all text-sm"
                    placeholder="Enter remarks or progress update"
                  />
                </div>

                {status === 'Resolved' && (
                  <div>
                    <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Upload Resolved Image <span className="text-red-500">*</span></label>
                    
                    {imageError && (
                      <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-xl">
                        <div className="flex items-start gap-2">
                          <svg className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                          <p className="text-sm text-red-700 font-medium">{imageError}</p>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex gap-3 mb-3">
                      <label className="flex-1 flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-stone-300 rounded-xl cursor-pointer hover:border-emerald-400 hover:bg-emerald-50/50 transition text-sm text-slate-600">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                        Choose File
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files[0];
                            if (file) {
                              setResolvedImage(file);
                              setImageError('');
                              const reader = new FileReader();
                              reader.onloadend = () => setResolvedImagePreview(reader.result);
                              reader.readAsDataURL(file);
                            }
                          }}
                        />
                      </label>
                      <button
                        type="button"
                        onClick={startResolvedCamera}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-stone-300 rounded-xl hover:border-emerald-400 hover:bg-emerald-50/50 transition text-sm text-slate-600"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                        Take Photo
                      </button>
                    </div>

                    {showResolvedCamera && (
                      <div className="mb-3 rounded-xl overflow-hidden border-2 border-emerald-400">
                        <video
                          ref={resolvedVideoRef}
                          autoPlay
                          playsInline
                          className="w-full rounded-t-xl"
                        />
                        <div className="flex gap-2 p-2 bg-stone-50">
                          <button
                            type="button"
                            onClick={captureResolvedPhoto}
                            className="flex-1 py-2 bg-emerald-600 text-white rounded-lg font-semibold text-sm hover:bg-emerald-700 transition"
                          >
                            📸 Capture
                          </button>
                          <button
                            type="button"
                            onClick={stopResolvedCamera}
                            className="px-4 py-2 bg-red-500 text-white rounded-lg font-semibold text-sm hover:bg-red-600 transition"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}

                    {resolvedImagePreview && (
                      <div className="mt-3 relative inline-block">
                        <img
                          src={resolvedImagePreview}
                          alt="Resolved preview"
                          className="w-48 h-48 object-contain rounded-xl border-2 border-emerald-400"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            setResolvedImage(null);
                            setResolvedImagePreview(null);
                            setImageError('');
                          }}
                          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600"
                        >
                          ✕
                        </button>
                      </div>
                    )}
                  </div>
                )}

                <div className="flex gap-3 pt-4 border-t border-stone-100">
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex-1 bg-emerald-600 text-white font-bold py-3 rounded-xl hover:bg-emerald-700 transition-all disabled:opacity-50 text-sm shadow-md hover:shadow-lg"
                  >
                    {loading ? 'Updating...' : 'Update Status'}
                  </button>
                  <button
                    type="button"
                    onClick={() => { stopResolvedCamera(); setImageError(''); setShowModal(false); }}
                    className="px-6 py-3 bg-stone-100 text-slate-600 rounded-xl hover:bg-stone-200 transition font-semibold text-sm"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DepartmentDashboard;
