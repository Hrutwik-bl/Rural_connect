const mongoose = require('mongoose');

const progressUpdateSchema = new mongoose.Schema({
	message: {
		type: String,
		required: true
	},
	updatedBy: {
		type: mongoose.Schema.Types.ObjectId,
		ref: 'User'
	},
	updatedByName: {
		type: String
	},
	role: {
		type: String
	},
	timestamp: {
		type: Date,
		default: Date.now
	}
}, { _id: false });

const complaintSchema = new mongoose.Schema({
	title: {
		type: String,
		trim: true,
		default: null
	},
	description: {
		type: String,
		required: [true, 'Description is required']
	},
	category: {
		type: String,
		required: [true, 'Category is required'],
		enum: ['Water', 'Electricity', 'Roads']
	},
	department: {
		type: String,
		enum: ['Water', 'Electricity', 'Roads']
	},
	priority: {
		type: String,
		enum: ['Low', 'Medium', 'High', 'Critical'],
		default: 'Medium'
	},
	severity: {
		type: String,
		enum: ['Low', 'Medium', 'High', 'Critical']
	},
	status: {
		type: String,
		enum: ['Pending', 'In Progress', 'Resolved', 'Rejected', 'Escalated'],
		default: 'Pending'
	},
	citizen: {
		type: mongoose.Schema.Types.ObjectId,
		ref: 'User',
		required: true
	},
	assignedTo: {
		type: mongoose.Schema.Types.ObjectId,
		ref: 'User'
	},
	location: {
		type: String,
		trim: true
	},
	locationCoords: {
		lat: Number,
		lng: Number
	},
	resolvedLocationCoords: {
		lat: Number,
		lng: Number
	},
	locationVerificationScore: {
		type: Number
	},
	locationVerificationDistance: {
		type: Number
	},
	imageData: {
		type: String
	},
	imageType: {
		type: String
	},
	resolvedImageData: {
		type: String
	},
	resolvedImageType: {
		type: String
	},
	escalated: {
		type: Boolean,
		default: false
	},
	escalatedBy: {
		type: mongoose.Schema.Types.ObjectId,
		ref: 'User'
	},
	escalatedAt: {
		type: Date
	},
	resolvedAt: {
		type: Date
	},
	remarks: {
		type: String
	},
	timeLimit: {
		type: Number
	},
	deadline: {
		type: Date
	},
	progressUpdates: [progressUpdateSchema]
}, { timestamps: true });

module.exports = mongoose.model('Complaint', complaintSchema);
