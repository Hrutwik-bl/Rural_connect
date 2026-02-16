const mongoose = require('mongoose');
const dotenv = require('dotenv');
const User = require('./models/User');

// Load environment variables
dotenv.config();

// Connect to database
mongoose.connect(process.env.MONGODB_URI)
.then(() => console.log('MongoDB connected'))
.catch((err) => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

// Admin user data - CHANGE THESE VALUES
const adminData = {
  name: 'System Administrator',
  email: 'admin@ruralportal.com',
  password: 'admin@123',
  role: 'admin',
  phone: '1234567890',
  address: 'System',
  approved: true
};

// Create admin user
const createAdmin = async () => {
  try {
    // Check if admin already exists
    const existingAdmin = await User.findOne({ email: adminData.email });
    
    if (existingAdmin) {
      console.log('❌ Admin user with this email already exists!');
      console.log(`Email: ${existingAdmin.email}`);
      process.exit(1);
    }

    // Create new admin
    const admin = await User.create(adminData);

    console.log('✅ Admin user created successfully!');
    console.log('==================================');
    console.log(`Name: ${admin.name}`);
    console.log(`Email: ${admin.email}`);
    console.log(`Role: ${admin.role}`);
    console.log(`Password: ${adminData.password}`);
    console.log('==================================');
    console.log('⚠️  Please change the password after first login!');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error creating admin:', error.message);
    process.exit(1);
  }
};

// Run the script
createAdmin();
