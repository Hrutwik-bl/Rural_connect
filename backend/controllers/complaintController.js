const { validationResult } = require('express-validator');
const axios = require('axios');
const Complaint = require('../models/Complaint');

// @desc    Predict complaint category and severity using AI
// @route   POST /api/complaints/predict
// @access  Private/Citizen
exports.predictComplaint = async (req, res) => {
	const errors = validationResult(req);
	if (!errors.isEmpty()) {
		return res.status(400).json({ errors: errors.array() });
	}

	try {
		const { description, imageData } = req.body;

		if (!description || !imageData) {
			return res.status(400).json({
				message: 'Description and image are required for prediction'
			});
		}

		const mlApiUrl = process.env.ML_API_URL || 'http://localhost:8003';

		try {
			const mlResponse = await axios.post(`${mlApiUrl}/predict-complaint`, {
				description,
				image_data: imageData
			});

			const mlData = mlResponse.data;

			// Handle GPT-2 rejection (not a valid complaint)
			if (mlData.rejected) {
				return res.json({
					predicted_department: null,
					predicted_severity: null,
					rejected: true,
					rejection_reason: mlData.rejection_reason,
					sentiment: mlData.sentiment,
					confidence: 0,
					is_valid: false,
					method: mlData.method,
					analysis: mlData.analysis
				});
			}

			return res.json({
				predicted_department: mlData.predicted_department,
				predicted_severity: mlData.predicted_severity,
				rejected: false,
				department_confidence: mlData.department_confidence,
				severity_confidence: mlData.severity_confidence,
				confidence: mlData.overall_confidence || mlData.confidence || 0.85,
				is_valid: mlData.validity?.is_valid ?? mlData.is_valid,
				valid_score: mlData.validity?.score ?? mlData.valid_score,
				message: mlData.validity?.message ?? mlData.message ?? null,
				sentiment: mlData.sentiment,
				cross_validation: mlData.cross_validation,
				validity: mlData.validity,
				method: mlData.method,
				has_image: mlData.has_image,
				image_analysis: mlData.image_analysis,
				text_analysis: mlData.text_analysis,
				analysis: mlData.analysis
			});
		} catch (mlError) {
			console.error('ML API Error:', mlError.message);
			return res.status(503).json({
				message: 'AI service temporarily unavailable. Please try again later.',
				error: mlError.message
			});
		}
	} catch (error) {
		console.error('Prediction error:', error);
		return res.status(500).json({
			message: 'Error processing prediction',
			error: error.message
		});
	}
};

// @desc    Create new complaint
// @route   POST /api/complaints
// @access  Private/Citizen
exports.createComplaint = async (req, res) => {
	const errors = validationResult(req);
	if (!errors.isEmpty()) {
		return res.status(400).json({ errors: errors.array() });
	}

	try {
		const { title, description, category, department, location, locationCoords, priority, imageData, imageType } = req.body;

		console.log('Creating complaint with data:', { title, description, category, department, location, hasImage: !!imageData });

		const normalizeDepartment = (dept) => {
			if (!dept) return null;
			if (dept === 'Road') return 'Roads';
			if (dept === 'Roads') return 'Roads';
			if (dept === 'Water') return 'Water';
			if (dept === 'Electricity') return 'Electricity';
			return dept;
		};

		// Use category or department from frontend (AI prediction already applied on frontend)
		const resolvedCategory = normalizeDepartment(category) || normalizeDepartment(department);
		if (!resolvedCategory) {
			return res.status(400).json({ message: 'Department/Category is required' });
		}

		const resolvedSeverity = priority || 'Medium';
		const resolvedPriority = resolvedSeverity || 'Medium';

		const resolvedTitle = title || (description
			? `${description.substring(0, 60)}${description.length > 60 ? '...' : ''}`
			: 'Complaint');

		const timeLimitHours = resolvedSeverity === 'Critical' ? 24 : resolvedSeverity === 'High' ? 48 : 72;
		const deadline = new Date(Date.now() + timeLimitHours * 60 * 60 * 1000);

		const complaint = await Complaint.create({
			title: resolvedTitle,
			description,
			category: resolvedCategory,
			department: resolvedCategory,
			location,
			locationCoords,
			imageData,
			imageType,
			severity: resolvedSeverity,
			priority: resolvedPriority,
			citizen: req.user.id,
			timeLimit: timeLimitHours,
			deadline: deadline
		});

		const populatedComplaint = await Complaint.findById(complaint._id)
			.populate('citizen', 'name email phone');

		return res.status(201).json(populatedComplaint);
	} catch (error) {
		console.error('Create complaint error:', error);
		return res.status(500).json({ message: 'Server error', error: error.message });
	}
};

// @desc    Get complaints for citizen
// @route   GET /api/complaints/my-complaints
// @access  Private/Citizen
exports.getMyComplaints = async (req, res) => {
	try {
		const complaints = await Complaint.find({ citizen: req.user.id })
			.sort({ createdAt: -1 })
			.populate('citizen', 'name email phone');
		return res.json(complaints);
	} catch (error) {
		console.error('Get my complaints error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Get all complaints
// @route   GET /api/complaints/all
// @access  Private/Admin,Department,Citizen
exports.getAllComplaints = async (req, res) => {
	try {
		let query = {};

		if (req.user.role === 'department') {
			query.category = req.user.department;
		}

		if (req.user.role === 'citizen') {
			query.citizen = req.user.id;
		}

		const complaints = await Complaint.find(query)
			.sort({ createdAt: -1 })
			.populate('citizen', 'name email phone')
			.populate('assignedTo', 'name email');

		return res.json(complaints);
	} catch (error) {
		console.error('Get all complaints error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Get complaints with time status
// @route   GET /api/complaints/time-status/all
// @access  Private/Admin,Department
exports.getComplaintsWithTimeStatus = async (req, res) => {
	try {
		const complaints = await Complaint.find({})
			.sort({ createdAt: -1 })
			.populate('citizen', 'name email phone');

		const enriched = complaints.map((complaint) => {
			const deadline = complaint.deadline ? new Date(complaint.deadline) : null;
			const now = new Date();
			let timeLeftHours = null;
			let isOverdue = false;

			if (deadline) {
				const diffMs = deadline.getTime() - now.getTime();
				timeLeftHours = Math.round(diffMs / (1000 * 60 * 60));
				isOverdue = diffMs < 0;
			}

			return {
				...complaint.toObject(),
				timeLeftHours,
				isOverdue
			};
		});

		return res.json(enriched);
	} catch (error) {
		console.error('Get complaints with time status error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Get complaints by department
// @route   GET /api/complaints/department/:department
// @access  Private/Department,Admin
exports.getComplaintsByDepartment = async (req, res) => {
	try {
		const complaints = await Complaint.find({ category: req.params.department })
			.sort({ createdAt: -1 })
			.populate('citizen', 'name email phone');

		return res.json(complaints);
	} catch (error) {
		console.error('Get complaints by department error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Add progress update
// @route   POST /api/complaints/:id/progress
// @access  Private/Department,Admin
exports.addProgressUpdate = async (req, res) => {
	try {
		const complaint = await Complaint.findById(req.params.id);
		if (!complaint) {
			return res.status(404).json({ message: 'Complaint not found' });
		}

		complaint.progressUpdates.push({
			message: req.body.message,
			updatedBy: req.user.id,
			updatedByName: req.user.name,
			role: req.user.role
		});

		await complaint.save();
		return res.json(complaint.progressUpdates);
	} catch (error) {
		console.error('Add progress update error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Get progress updates
// @route   GET /api/complaints/:id/progress
// @access  Private
exports.getProgressUpdates = async (req, res) => {
	try {
		const complaint = await Complaint.findById(req.params.id);
		if (!complaint) {
			return res.status(404).json({ message: 'Complaint not found' });
		}

		return res.json(complaint.progressUpdates || []);
	} catch (error) {
		console.error('Get progress updates error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Get complaint by id
// @route   GET /api/complaints/:id
// @access  Private
exports.getComplaintById = async (req, res) => {
	try {
		const complaint = await Complaint.findById(req.params.id)
			.populate('citizen', 'name email phone')
			.populate('assignedTo', 'name email');

		if (!complaint) {
			return res.status(404).json({ message: 'Complaint not found' });
		}

		return res.json(complaint);
	} catch (error) {
		console.error('Get complaint by id error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Update complaint status
// @route   PUT /api/complaints/:id/status
// @access  Private/Department
exports.updateComplaintStatus = async (req, res) => {
	try {
		const { status, remarks, resolvedLocationCoords, resolvedImageData, resolvedImageType } = req.body;
		const complaint = await Complaint.findById(req.params.id);

		if (!complaint) {
			return res.status(404).json({ message: 'Complaint not found' });
		}

		// Handle location verification BEFORE setting status to Resolved
		if (status === 'Resolved') {
			// Require resolved image for resolution
			if (!resolvedImageData) {
				return res.status(400).json({
					message: 'A photo of the resolved issue is required to mark a complaint as resolved.'
				});
			}

			// Require location coordinates for resolution
			if (!resolvedLocationCoords || !resolvedLocationCoords.lat || !resolvedLocationCoords.lng) {
				return res.status(400).json({
					message: 'Location coordinates are required to mark a complaint as resolved. Please enable location services.'
				});
			}

			// Require original complaint to have coordinates
			if (!complaint.locationCoords || !complaint.locationCoords.lat || !complaint.locationCoords.lng) {
				// If original complaint has no coordinates, allow resolution without verification
				console.log('Original complaint has no location coordinates, skipping verification');
			} else {
				// Verify location using DL model
				const mlApiUrl = process.env.ML_API_URL || 'http://localhost:8003';

				try {
					const verifyResponse = await axios.post(`${mlApiUrl}/verify-location`, {
						complaint_lat: complaint.locationCoords.lat,
						complaint_lon: complaint.locationCoords.lng,
						resolved_lat: resolvedLocationCoords.lat,
						resolved_lon: resolvedLocationCoords.lng
					});

					console.log(`Location verification: ${verifyResponse.data.distance_meters}m, score: ${verifyResponse.data.resolved_probability}`);

					// Store verification results
					complaint.resolvedLocationCoords = resolvedLocationCoords;
					complaint.locationVerificationScore = verifyResponse.data.resolved_probability;
					complaint.locationVerificationDistance = verifyResponse.data.distance_meters;

					// REJECT resolution if location doesn't match - DO NOT allow resolved status
					if (!verifyResponse.data.resolved) {
						return res.status(400).json({
							message: `Location verification failed. You are ${Math.round(verifyResponse.data.distance_meters)}m away from the complaint location. Please go to the actual location to resolve this complaint.`,
							distance: verifyResponse.data.distance_meters,
							verificationScore: verifyResponse.data.resolved_probability,
							verified: false
						});
					}
				} catch (mlError) {
					console.error('Location verification API error:', mlError.message);
					// BLOCK resolution if ML service is unavailable - don't allow bypass
					return res.status(503).json({
						message: 'Location verification service is unavailable. Please try again later.',
						verified: false
					});
				}
			}

			// Only set to Resolved AFTER verification passes
			complaint.status = 'Resolved';
			complaint.resolvedAt = new Date();
			complaint.resolvedLocationCoords = resolvedLocationCoords;
			complaint.resolvedImageData = resolvedImageData;
			complaint.resolvedImageType = resolvedImageType || 'image/jpeg';
		} else {
			// For non-Resolved status updates
			if (status) complaint.status = status;
		}

		if (remarks) complaint.remarks = remarks;

		await complaint.save();
		return res.json(complaint);
	} catch (error) {
		console.error('Update complaint status error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Escalate complaint
// @route   PUT /api/complaints/:id/escalate
// @access  Private/Admin
exports.escalateComplaint = async (req, res) => {
	try {
		const complaint = await Complaint.findById(req.params.id);

		if (!complaint) {
			return res.status(404).json({ message: 'Complaint not found' });
		}

		complaint.escalated = true;
		complaint.escalatedBy = req.user.id;
		complaint.escalatedAt = new Date();
		complaint.status = 'Escalated';

		await complaint.save();
		return res.json({ message: 'Complaint escalated', complaint });
	} catch (error) {
		console.error('Escalate complaint error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Check and auto-escalate overdue complaints
// @route   POST /api/complaints/check-escalate/overdue
// @access  Private/Admin
exports.checkAndAutoEscalate = async (req, res) => {
	try {
		const now = new Date();
		const overdue = await Complaint.find({
			deadline: { $lt: now },
			status: { $nin: ['Resolved', 'Rejected', 'Escalated'] }
		});

		const updates = await Promise.all(overdue.map(async (complaint) => {
			complaint.escalated = true;
			complaint.escalatedAt = new Date();
			complaint.status = 'Escalated';
			return complaint.save();
		}));

		return res.json({ escalated: updates.length });
	} catch (error) {
		console.error('Auto-escalate error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};

// @desc    Delete complaint
// @route   DELETE /api/complaints/:id
// @access  Private/Admin
exports.deleteComplaint = async (req, res) => {
	try {
		const complaint = await Complaint.findById(req.params.id);
		if (!complaint) {
			return res.status(404).json({ message: 'Complaint not found' });
		}

		await Complaint.findByIdAndDelete(req.params.id);
		return res.json({ message: 'Complaint deleted' });
	} catch (error) {
		console.error('Delete complaint error:', error);
		return res.status(500).json({ message: 'Server error' });
	}
};
