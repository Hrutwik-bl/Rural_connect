const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const complaintController = require('../controllers/complaintController');
const { protect, authorize } = require('../middleware/auth');

// All routes require authentication
router.use(protect);

// AI Prediction endpoint (must be before POST / to avoid conflicts)
router.post('/predict', [
  body('description').trim().notEmpty().withMessage('Description is required'),
  body('imageData').notEmpty().withMessage('Image data is required')
], authorize('citizen'), complaintController.predictComplaint);

// Create a new complaint (Citizens only)
router.post('/', [
  body('title').optional().trim(),
  body('description').trim().notEmpty().withMessage('Description is required'),
  body('category').optional().isIn(['Water', 'Electricity', 'Roads'])
    .withMessage('Invalid category'),
  body('department').optional().isIn(['Water', 'Electricity', 'Roads'])
    .withMessage('Invalid department')
], authorize('citizen'), complaintController.createComplaint);

// Get complaints for citizen (they can only see their own)
router.get('/my-complaints', authorize('citizen'), complaintController.getMyComplaints);

// Get all complaints (Admin, Department, Citizen - MUST BE AFTER /my-complaints)
router.get('/all', authorize('admin', 'department', 'citizen'), complaintController.getAllComplaints);

// Get complaints with time status - BEFORE generic /:id routes
router.get('/time-status/all', authorize('admin', 'department'), complaintController.getComplaintsWithTimeStatus);

// Get complaints by department
router.get('/department/:department', authorize('department', 'admin'), complaintController.getComplaintsByDepartment);


// Add progress update (Department and Admin) - BEFORE /:id
router.post('/:id/progress', 
  [body('message').trim().notEmpty().withMessage('Progress message is required')],
  authorize('department', 'admin'),
  complaintController.addProgressUpdate
);

// Get progress updates for a complaint - BEFORE /:id
router.get('/:id/progress', complaintController.getProgressUpdates);

// Get single complaint
router.get('/:id', complaintController.getComplaintById);

// Update complaint status (Department only)
router.put('/:id/status', authorize('department'), complaintController.updateComplaintStatus);

// Escalate complaint (Admin only)
router.put('/:id/escalate', authorize('admin'), complaintController.escalateComplaint);

// Check and auto-escalate overdue complaints (Admin only)
router.post('/check-escalate/overdue', authorize('admin'), complaintController.checkAndAutoEscalate);

// Delete complaint (Admin only)
router.delete('/:id', authorize('admin'), complaintController.deleteComplaint);
module.exports = router;
