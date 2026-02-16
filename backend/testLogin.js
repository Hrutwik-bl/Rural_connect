const mongoose = require('mongoose');
const dotenv = require('dotenv');
const path = require('path');
const User = require('./models/User');

dotenv.config({ path: path.join(__dirname, '.env') });

mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB connected'))
  .catch((err) => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

const testLogin = async () => {
  try {
    // Check if admin exists
    const admin = await User.findOne({ email: 'admin@ruralportal.com' });
    
    if (!admin) {
      console.log('❌ Admin user not found in database!');
      console.log('Creating admin...');
      
      const newAdmin = await User.create({
        name: 'System Administrator',
        email: 'admin@ruralportal.com',
        password: 'admin@123',
        role: 'admin',
        phone: '1234567890',
        address: 'System',
        approved: true
      });
      
      console.log('✅ Admin created successfully');
      console.log('Admin ID:', newAdmin._id);
    } else {
      console.log('✅ Admin user found!');
      console.log('Name:', admin.name);
      console.log('Email:', admin.email);
      console.log('Role:', admin.role);
      console.log('Approved:', admin.approved);
      
      // Test password comparison
      const isPasswordCorrect = await admin.comparePassword('admin@123');
      console.log('Password match:', isPasswordCorrect ? '✅ YES' : '❌ NO');
    }
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

testLogin();
