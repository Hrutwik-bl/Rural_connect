const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const { protect, authorize } = require('../middleware/auth');

// All routes require authentication and admin role
router.use(protect);
router.use(authorize('admin'));

// Get all users
router.get('/', userController.getAllUsers);

// Get pending department approvals
router.get('/pending-approvals', userController.getPendingApprovals);

// Get users by role
router.get('/role/:role', userController.getUsersByRole);

// Get single user
router.get('/:id', userController.getUserById);

// Approve department user
router.put('/:id/approve', userController.approveUser);

// Reject department user
router.put('/:id/reject', userController.rejectUser);

// Update user
router.put('/:id', userController.updateUser);

// Delete user
router.delete('/:id', userController.deleteUser);

module.exports = router;
