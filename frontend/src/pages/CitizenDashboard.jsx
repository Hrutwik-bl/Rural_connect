import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../api/axiosConfig';
import { useAuth } from '../context/AuthContext';

const CitizenDashboard = () => {
  const { user } = useAuth();
  const [complaints, setComplaints] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [showViewModal, setShowViewModal] = useState(false);
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [aiPredictions, setAiPredictions] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [toast, setToast] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  const [formData, setFormData] = useState({
    title: '',
    description: '',
    category: 'Water',
    location: '',
    locationCoords: null,
    imageData: null,
    imageType: null
  });

  useEffect(() => {
    fetchComplaints();
  }, []);

  const getAutomaticLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          const coords = { lat: latitude, lng: longitude };

          try {
            const response = await fetch(
              `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json&accept-language=en`
            );
            const data = await response.json();
            const placeName = data.address?.village || data.address?.town || data.address?.city ||
              data.address?.suburb || data.address?.county || data.address?.state_district ||
              data.address?.state || data.display_name?.split(',')[0] || 'Unknown Location';
            setFormData(prev => ({
              ...prev,
              location: placeName,
              locationCoords: coords
            }));
          } catch (err) {
            setFormData(prev => ({
              ...prev,
              location: prev.location || '',
              locationCoords: coords
            }));
            showToast('GPS captured but could not detect place name. Please enter location manually.', 'error');
          }
        },
        (geoError) => {
          console.log('Geolocation error:', geoError);
          showToast('Could not get GPS location. Location verification requires GPS.', 'error');
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
      );
    } else {
      showToast('Geolocation is not supported by this browser.', 'error');
    }
  };

  const fetchComplaints = async () => {
    try {
      const response = await apiClient.get('/api/complaints/my-complaints');
      setComplaints(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching complaints:', err);
      setError(`Failed to load complaints: ${err.message}`);
      setComplaints([]);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target.result.split(',')[1];
        setFormData({
          ...formData,
          imageData: base64,
          imageType: file.type
        });
        setImagePreview(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      streamRef.current = stream;
      setShowCamera(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }, 100);
    } catch (err) {
      console.error('Camera error:', err);
      showToast('Could not access camera. Please check permissions or use file upload instead.', 'error');
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    const base64 = dataUrl.split(',')[1];
    setFormData(prev => ({ ...prev, imageData: base64, imageType: 'image/jpeg' }));
    setImagePreview(dataUrl);
    stopCamera();
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setShowCamera(false);
  };

  const predictWithAI = async () => {
    if (!formData.description.trim()) {
      showToast('Please enter a description first', 'error');
      return;
    }

    if (!formData.imageData) {
      showToast('Please upload an image first', 'error');
      return;
    }

    setAiLoading(true);
    try {
      const response = await apiClient.post('/api/complaints/predict', {
        description: formData.description,
        imageData: formData.imageData,
        imageType: formData.imageType
      });

      const data = response.data;
      
      // Handle GPT-2 rejection (not a complaint)
      if (data.rejected) {
        setAiPredictions({
          ...data,
          rejected: true,
          is_valid: false
        });
      } else {
        setAiPredictions({
          ...data,
          is_valid: data.validity?.is_valid ?? true
        });
        // Auto-apply AI department prediction
        const aiDept = data.predicted_department;
        if (aiDept) {
          const normalized = aiDept === 'Road' ? 'Roads' : aiDept;
          setFormData(prev => ({ ...prev, category: normalized }));
        }
      }
    } catch (err) {
      console.error('AI Prediction Error:', err);
      showToast('Could not analyze complaint. Please try again.', 'error');
    } finally {
      setAiLoading(false);
    }
  };

  const normalizeDepartment = (dept) => {
    if (!dept) return '';
    if (dept === 'Road') return 'Roads';
    return dept;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (!formData.locationCoords) {
        showToast('GPS location is required. Please enable location services and try again.', 'error');
        setLoading(false);
        return;
      }

      const submitData = {
        title: formData.title,
        description: formData.description,
        category: formData.category,
        department: formData.category,
        location: formData.location,
        locationCoords: formData.locationCoords,
        imageData: formData.imageData,
        imageType: formData.imageType
      };
      console.log('Submitting complaint:', { ...submitData, imageData: submitData.imageData ? '[base64]' : null });
      await apiClient.post('/api/complaints', submitData);
      setShowModal(false);
      stopCamera();
      setFormData({
        title: '',
        description: '',
        category: 'Water',
        location: '',
        locationCoords: null,
        imageData: null,
        imageType: null
      });
      setImagePreview(null);
      setAiPredictions(null);
      fetchComplaints();
      showToast('Complaint submitted successfully!', 'success');
    } catch (err) {
      console.error('Submit error:', err.response?.data || err.message);
      const errorMsg = err.response?.data?.message || err.response?.data?.errors?.[0]?.msg || err.message || 'Unknown error';
      showToast(`Error submitting complaint: ${errorMsg}`, 'error');
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

  return (
    <div className="min-h-screen bg-stone-100 py-8">
      {/* Toast notification */}
      {toast && (
        <div className="fixed top-6 right-6 z-[9999] animate-slide-in">
          <div className={`flex items-center gap-3 px-5 py-3 rounded-xl shadow-elevated text-white font-medium text-sm max-w-md backdrop-blur-sm ${
            toast.type === 'success' ? 'bg-emerald-600' : 'bg-red-600'
          }`}>
            {toast.type === 'success' ? (
              <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            <span>{toast.message}</span>
            <button onClick={() => setToast(null)} className="ml-2 text-white/80 hover:text-white">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      <div className="container mx-auto px-4">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-4 text-sm">
            {error}
          </div>
        )}
        
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-card border border-stone-200 p-6 mb-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-extrabold text-slate-800">Welcome, {user?.name}!</h2>
              <p className="text-slate-400 text-sm mt-0.5">Citizen Dashboard</p>
            </div>
            <button
              onClick={() => {
                setShowModal(true);
                getAutomaticLocation();
              }}
              className="px-6 py-2.5 bg-emerald-600 text-white rounded-xl hover:bg-emerald-700 transition-all font-bold text-sm shadow-md hover:shadow-lg hover:shadow-emerald-600/20 hover:-translate-y-0.5 flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
              New Complaint
            </button>
          </div>
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-2xl shadow-card border border-stone-200 p-5 hover:shadow-card-hover hover:-translate-y-1 transition-all">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-lg">📋</div>
              <div>
                <p className="text-2xl font-extrabold text-slate-800">{complaints.length}</p>
                <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Total</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-card border border-stone-200 p-5 hover:shadow-card-hover hover:-translate-y-1 transition-all">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center text-lg">⏳</div>
              <div>
                <p className="text-2xl font-extrabold text-amber-600">{complaints.filter(c => c.status === 'Pending').length}</p>
                <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Pending</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-card border border-stone-200 p-5 hover:shadow-card-hover hover:-translate-y-1 transition-all">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-sky-50 flex items-center justify-center text-lg">🔄</div>
              <div>
                <p className="text-2xl font-extrabold text-sky-600">{complaints.filter(c => c.status === 'In Progress').length}</p>
                <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">In Progress</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-card border border-stone-200 p-5 hover:shadow-card-hover hover:-translate-y-1 transition-all">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-lg">✅</div>
              <div>
                <p className="text-2xl font-extrabold text-emerald-600">{complaints.filter(c => c.status === 'Resolved').length}</p>
                <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Resolved</p>
              </div>
            </div>
          </div>
        </div>

        {/* Complaints Table */}
        <div className="bg-white rounded-2xl shadow-card border border-stone-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-stone-100">
            <h3 className="text-lg font-bold text-slate-800">Your Complaints</h3>
          </div>

          {complaints.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-4xl mb-3">📭</div>
              <p className="text-slate-400 font-medium">No complaints filed yet.</p>
              <p className="text-slate-300 text-sm mt-1">Click "New Complaint" to get started</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-50 border-b border-stone-200">
                    <th className="px-5 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Title</th>
                    <th className="px-5 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Category</th>
                    <th className="px-5 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Location</th>
                    <th className="px-5 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Status</th>
                    <th className="px-5 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Date</th>
                    <th className="px-5 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {complaints.map(complaint => (
                    <tr key={complaint._id} className="hover:bg-emerald-50/50 transition-colors">
                      <td className="px-5 py-3.5 font-medium text-slate-800 text-sm">{complaint.title}</td>
                      <td className="px-5 py-3.5 text-sm text-slate-600">{complaint.category}</td>
                      <td className="px-5 py-3.5 text-sm text-slate-600">{complaint.location}</td>
                      <td className="px-5 py-3.5">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusBadge(complaint.status)}`}>
                          {complaint.status}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-sm text-slate-400">
                        {new Date(complaint.createdAt).toLocaleDateString()}
                      </td>
                      <td className="px-5 py-3.5">
                        <button
                          onClick={() => {
                            setSelectedComplaint(complaint);
                            setShowViewModal(true);
                          }}
                          className="text-emerald-600 hover:text-emerald-700 text-sm font-semibold hover:underline"
                        >
                          View Details
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

      {showViewModal && selectedComplaint && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-elevated border border-stone-200">
            <div className="px-6 py-4 border-b border-stone-100 flex justify-between items-center sticky top-0 bg-white rounded-t-2xl">
              <h3 className="text-lg font-bold text-slate-800">Complaint Details</h3>
              <button onClick={() => setShowViewModal(false)} className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 text-slate-400 hover:text-slate-600 transition">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-5">
              <div className="flex items-start justify-between">
                <h4 className="text-xl font-bold text-slate-800">{selectedComplaint.title}</h4>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusBadge(selectedComplaint.status)}`}>
                  {selectedComplaint.status}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Category</label>
                  <p className="text-slate-800 font-medium text-sm">{selectedComplaint.category}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Location</label>
                  <p className="text-slate-800 font-medium text-sm">{selectedComplaint.location}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Date Submitted</label>
                  <p className="text-slate-800 font-medium text-sm">{new Date(selectedComplaint.createdAt).toLocaleString()}</p>
                </div>
                <div className="bg-stone-50 rounded-xl p-3">
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Complaint ID</label>
                  <p className="text-slate-500 text-xs font-mono">{selectedComplaint._id}</p>
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Description</label>
                <p className="text-slate-700 bg-stone-50 p-4 rounded-xl text-sm leading-relaxed">{selectedComplaint.description}</p>
              </div>

              {selectedComplaint.imageData && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Attached Image</label>
                  <div className="bg-stone-50 rounded-xl p-4">
                    <img
                      src={getImageSrc(selectedComplaint)}
                      alt="Complaint"
                      className="w-64 h-64 object-contain rounded-xl cursor-pointer hover:opacity-90 transition"
                      onClick={() => {
                        const w = window.open('', '_blank');
                        if (w) {
                          w.document.write(`<html><head><title>Image Preview</title><style>body{margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#1a1a1a;}</style></head><body><img src="${getImageSrc(selectedComplaint)}" style="max-width:100%;max-height:100vh;object-fit:contain;"/></body></html>`);
                          w.document.close();
                        }
                      }}
                    />
                    <p className="text-xs text-slate-400 mt-2 text-center">Click image to view full size</p>
                  </div>
                </div>
              )}

              {/* Resolved Image */}
              {selectedComplaint.resolvedImageData && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Resolved Image</label>
                  <div className="rounded-xl p-4 bg-emerald-50 border border-emerald-200">
                    <img
                      src={selectedComplaint.resolvedImageData.startsWith('data:') 
                        ? selectedComplaint.resolvedImageData 
                        : `data:${selectedComplaint.resolvedImageType || 'image/jpeg'};base64,${selectedComplaint.resolvedImageData}`}
                      alt="Resolved"
                      className="w-64 h-64 object-contain rounded-xl cursor-pointer hover:opacity-90 transition border-2 border-emerald-400"
                      onClick={() => {
                        const src = selectedComplaint.resolvedImageData.startsWith('data:') 
                          ? selectedComplaint.resolvedImageData 
                          : `data:${selectedComplaint.resolvedImageType || 'image/jpeg'};base64,${selectedComplaint.resolvedImageData}`;
                        const w = window.open('', '_blank');
                        if (w) {
                          w.document.write(`<html><head><title>Resolved Image</title><style>body{margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#1a1a1a;}</style></head><body><img src="${src}" style="max-width:100%;max-height:100vh;object-fit:contain;"/></body></html>`);
                          w.document.close();
                        }
                      }}
                    />
                    <p className="text-xs text-emerald-600 mt-2 text-center font-medium">Photo of resolved issue</p>
                  </div>
                </div>
              )}

              {/* Location Verification Results */}
              {selectedComplaint.status === 'Resolved' && selectedComplaint.locationVerificationScore != null && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Location Verification</label>
                    <div className={`rounded-xl p-4 border-2 ${
                      selectedComplaint.locationVerificationScore > 0.5 
                        ? 'bg-emerald-50 border-emerald-300' 
                        : 'bg-red-50 border-red-300'
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        {selectedComplaint.locationVerificationScore > 0.5 ? (
                          <svg className="w-6 h-6 text-emerald-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
                        ) : (
                          <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                        )}
                        <span className={`text-lg font-bold ${
                          selectedComplaint.locationVerificationScore > 0.5 ? 'text-emerald-700' : 'text-red-700'
                        }`}>
                          {selectedComplaint.locationVerificationScore > 0.5 ? 'Location Verified' : 'Location Mismatch'}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 mt-3">
                        <div className="bg-white/70 rounded-lg p-2.5">
                          <p className="text-xs text-slate-500 font-semibold">Distance</p>
                          <p className="text-sm font-bold text-slate-800">
                            {selectedComplaint.locationVerificationDistance != null 
                              ? `${Math.round(selectedComplaint.locationVerificationDistance)} meters`
                              : 'N/A'}
                          </p>
                        </div>
                        <div className="bg-white/70 rounded-lg p-2.5">
                          <p className="text-xs text-slate-500 font-semibold">Verification Score</p>
                          <p className="text-sm font-bold text-slate-800">
                            {(selectedComplaint.locationVerificationScore * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                </div>
              )}

              {/* Remarks */}
              {selectedComplaint.remarks && (
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Remarks</label>
                  <p className="text-slate-700 bg-emerald-50 p-4 rounded-xl text-sm leading-relaxed border-l-4 border-emerald-500">{selectedComplaint.remarks}</p>
                </div>
              )}
            </div>

            <div className="px-6 py-4 border-t border-stone-100 flex justify-end">
              <button
                onClick={() => setShowViewModal(false)}
                className="px-5 py-2 bg-stone-100 text-slate-600 rounded-xl hover:bg-stone-200 transition font-medium text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-elevated border border-stone-200">
            <div className="px-6 py-4 border-b border-stone-100 flex justify-between items-center sticky top-0 bg-white rounded-t-2xl z-10">
              <div>
                <h3 className="text-lg font-bold text-slate-800">File New Complaint</h3>
                <p className="text-xs text-slate-400 mt-0.5">AI-Powered Analysis</p>
              </div>
              <button onClick={() => setShowModal(false)} className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 text-slate-400 hover:text-slate-600 transition">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <form className="space-y-4">
                    <div>
                      <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Description *</label>
                      <textarea
                        required
                        rows="3"
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        className="w-full px-4 py-2.5 bg-stone-50 border border-stone-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all text-sm"
                        placeholder="Detailed description of the issue"
                      />
                    </div>

                    {aiPredictions && (
                      <div>
                        <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Department (AI Assigned)</label>
                        <div className="w-full px-4 py-2.5 bg-emerald-50 border border-emerald-300 rounded-xl text-sm font-semibold text-emerald-700">
                          {formData.category}
                        </div>
                        <p className="text-xs text-emerald-500 mt-1">Automatically assigned by AI analysis.</p>
                      </div>
                    )}

                    <div>
                      <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Location *</label>
                      <input
                        type="text"
                        required
                        value={formData.location}
                        onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                        className="w-full px-4 py-2.5 bg-stone-50 border border-stone-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all text-sm"
                        placeholder="Enter location"
                      />
                      {formData.locationCoords ? (
                        <div className="flex items-center gap-1.5 mt-1.5">
                          <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
                          <span className="text-xs text-emerald-600 font-medium">GPS location captured successfully</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 mt-1.5">
                          <svg className="w-4 h-4 text-amber-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                          <span className="text-xs text-amber-600 font-medium">GPS required — enable location services</span>
                        </div>
                      )}
                    </div>

                    <div>
                      <label className="block text-slate-600 font-semibold mb-1.5 text-sm">Upload Image *</label>
                      <div className="flex gap-2 mb-2">
                        <label className="flex-1 cursor-pointer">
                          <div className="flex items-center justify-center px-4 py-2.5 border-2 border-dashed border-stone-300 rounded-xl hover:border-emerald-400 hover:bg-emerald-50 transition-all">
                            <svg className="w-5 h-5 mr-2 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <span className="text-sm text-slate-500">Choose File</span>
                          </div>
                          <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="hidden"
                          />
                        </label>
                        <button
                          type="button"
                          onClick={showCamera ? stopCamera : startCamera}
                          className={`flex-1 flex items-center justify-center px-4 py-2.5 border-2 rounded-xl transition-all ${
                            showCamera
                              ? 'border-red-300 bg-red-50 text-red-600 hover:bg-red-100'
                              : 'border-dashed border-stone-300 hover:border-emerald-400 hover:bg-emerald-50 text-slate-500'
                          }`}
                        >
                          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                          <span className="text-sm">{showCamera ? 'Close Camera' : 'Take Photo'}</span>
                        </button>
                      </div>
                      <p className="text-xs text-slate-400">Upload a photo or take a live photo of the issue for AI analysis</p>
                    </div>

                    {showCamera && (
                      <div className="border-2 border-emerald-400 rounded-xl p-3 bg-slate-900">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="w-full rounded-lg"
                        />
                        <canvas ref={canvasRef} className="hidden" />
                        <button
                          type="button"
                          onClick={capturePhoto}
                          className="w-full mt-2 px-4 py-2.5 bg-emerald-600 text-white rounded-xl hover:bg-emerald-700 transition font-bold flex items-center justify-center gap-2 text-sm"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                          Capture Photo
                        </button>
                      </div>
                    )}

                    {imagePreview && (
                      <div className="bg-stone-50 rounded-xl p-3 border border-stone-200">
                        <img
                          src={imagePreview}
                          alt="Preview"
                          className="w-full h-48 object-contain rounded-lg"
                        />
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={predictWithAI}
                      disabled={aiLoading || !formData.description || !formData.imageData}
                      className="w-full px-4 py-3 bg-slate-800 text-white rounded-xl hover:bg-slate-700 transition-all font-bold disabled:opacity-40 text-sm flex items-center justify-center gap-2"
                    >
                      {aiLoading ? (
                        <>
                          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
                          Analyze with AI
                        </>
                      )}
                    </button>
                  </form>
                </div>

                <div>
                  <div className="bg-stone-50 rounded-xl p-5 h-full border border-stone-200">
                    <h4 className="text-sm font-bold text-slate-800 mb-4 uppercase tracking-wider">AI Analysis</h4>

                    {!aiPredictions ? (
                      <div className="text-center py-10">
                        <div className="w-12 h-12 rounded-full bg-stone-200 flex items-center justify-center mx-auto mb-3">
                          <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
                        </div>
                        <p className="text-sm text-slate-400">Fill in the form and click "Analyze with AI"</p>
                        <p className="text-xs text-slate-300 mt-1">AI will analyze your complaint text and image</p>
                      </div>
                    ) : aiPredictions.rejected ? (
                      <div className="space-y-4">
                        <div className="bg-red-50 rounded-xl p-4 border border-red-200">
                          <div className="flex items-center gap-2 mb-2">
                            <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                            <p className="text-base font-bold text-red-700">Not a Valid Complaint</p>
                          </div>
                          <p className="text-sm text-red-600">
                            {aiPredictions.rejection_reason || 'Your description does not appear to be a complaint. Please describe the problem or issue you are facing.'}
                          </p>
                        </div>

                        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3">
                          <p className="text-sm text-amber-700">
                            Please write a description about an actual problem (e.g., broken road, water leakage, power outage) to submit a complaint.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="bg-white rounded-xl p-4 border border-stone-200">
                          <p className="text-xs text-slate-400 uppercase font-bold tracking-wider">Identified Department</p>
                          <p className="text-xl font-extrabold text-emerald-600 mt-1">{aiPredictions.predicted_department || 'N/A'}</p>
                          <div className="mt-2 h-1.5 bg-stone-100 rounded-full overflow-hidden">
                            <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${Math.round((aiPredictions.department_confidence || aiPredictions.confidence || 0.8) * 100)}%` }}></div>
                          </div>
                          <p className="text-xs text-slate-400 mt-1">
                            {Math.round((aiPredictions.department_confidence || aiPredictions.confidence || 0.8) * 100)}% Confidence
                          </p>
                        </div>

                        <div className="bg-white rounded-xl p-4 border border-stone-200">
                          <p className="text-xs text-slate-400 uppercase font-bold tracking-wider">Identified Severity</p>
                          <p className="text-xl font-extrabold text-amber-600 mt-1">{aiPredictions.predicted_severity || 'N/A'}</p>
                        </div>

                        <div className={`bg-white rounded-xl p-4 border ${aiPredictions.is_valid ? 'border-emerald-200' : 'border-red-200'}`}>
                          <p className="text-xs text-slate-400 uppercase font-bold tracking-wider">Validation</p>
                          <div className="flex items-center gap-2 mt-1">
                            {aiPredictions.is_valid ? (
                              <svg className="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
                            ) : (
                              <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                            )}
                            <p className={`text-lg font-bold ${aiPredictions.is_valid ? 'text-emerald-600' : 'text-red-600'}`}>
                              {aiPredictions.is_valid ? 'Valid' : 'Not Valid'}
                            </p>
                          </div>
                          {!aiPredictions.is_valid && aiPredictions.validity?.message && (
                            <p className="text-xs text-red-500 mt-2">{aiPredictions.validity.message}</p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="px-6 py-4 border-t border-stone-100 flex gap-3">
              {!aiPredictions ? (
                <div className="flex-1 bg-stone-100 text-slate-400 font-semibold py-3 rounded-xl text-center text-sm">
                  Please analyze with AI before submitting
                </div>
              ) : aiPredictions.rejected ? (
                <div className="flex-1 bg-red-50 text-red-600 font-semibold py-3 rounded-xl text-center text-sm border border-red-200">
                  Not a Valid Complaint — Please rewrite with a negative/problem description
                </div>
              ) : !aiPredictions.is_valid ? (
                <div className="flex-1 bg-red-50 text-red-600 font-semibold py-3 rounded-xl text-center text-sm border border-red-200">
                  Not Valid — Description does not match the uploaded image. Please correct your description or upload a matching image.
                </div>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="flex-1 bg-emerald-600 text-white font-bold py-3 rounded-xl hover:bg-emerald-700 transition-all disabled:opacity-50 text-sm shadow-md hover:shadow-lg"
                >
                  {loading ? 'Submitting...' : 'Submit Complaint'}
                </button>
              )}
              <button
                type="button"
                onClick={() => {
                  setShowModal(false);
                  setAiPredictions(null);
                  stopCamera();
                }}
                className="px-6 py-3 bg-stone-100 text-slate-600 rounded-xl hover:bg-stone-200 transition font-semibold text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CitizenDashboard;
