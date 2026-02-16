const jwt = require('jsonwebtoken');
const { validationResult } = require('express-validator');
const User = require('../models/User');

// Generate JWT token
const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET, {
    expiresIn: '30d'
  });
};

// @desc    Register user
// @route   POST /api/auth/register
// @access  Public
exports.register = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    const { name, email, password, role, department, phone, address } = req.body;

    // Block admin registration - admins can only be created by developers
    if (role === 'admin') {
      return res.status(403).json({ 
        message: 'Admin accounts cannot be registered through this portal. Please contact the system administrator.' 
      });
    }

    // Check if user exists
    const userExists = await User.findOne({ email });
    if (userExists) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Create user
    const user = await User.create({
      name,
      email,
      password,
      role,
      department: role === 'department' ? department : '',
      phone,
      address
    });

    // For department users, notify about approval requirement
    if (role === 'department') {
      return res.status(201).json({
        message: 'Registration successful! Your account is pending admin approval. You will be able to log in once approved.',
        requiresApproval: true,
        _id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        department: user.department
      });
    }

    // For citizens, return token for immediate login
    res.status(201).json({
      _id: user._id,
      name: user.name,
      email: user.email,
      role: user.role,
      department: user.department,
      token: generateToken(user._id)
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// @desc    Login user
// @route   POST /api/auth/login
// @access  Public
exports.login = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    console.log('Validation errors:', errors.array());
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    const { email, password } = req.body;
    console.log('Login attempt for email:', email);

    // Check if user exists
    const user = await User.findOne({ email });
    if (!user) {
      console.log('User not found for email:', email);
      return res.status(401).json({ message: 'User not found. Please register first.' });
    }

    console.log('User found:', user.email, 'Role:', user.role);

    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      console.log('Password mismatch for user:', email);
      return res.status(401).json({ message: 'Wrong password. Please try again.' });
    }

    console.log('Password matched for user:', email);

    // Check if department user is approved
    if (user.role === 'department' && !user.approved) {
      console.log('Department user not approved:', email);
      return res.status(403).json({ 
        message: 'Your account is pending admin approval. Please wait for approval before logging in.' 
      });
    }

    console.log('All checks passed for user:', email);

    // Return user data with token
    res.json({
      _id: user._id,
      name: user.name,
      email: user.email,
      role: user.role,
      department: user.department,
      token: generateToken(user._id)
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// @desc    Get current user
// @route   GET /api/auth/me
// @access  Private
exports.getMe = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-password');
    res.json(user);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};
