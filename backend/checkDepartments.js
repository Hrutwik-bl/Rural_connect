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

// Check department users
const checkDepartments = async () => {
  try {
    const deptUsers = await User.find({ role: 'department' });
    
    console.log('\n🏢 DEPARTMENT USERS IN DATABASE:\n');
    console.log('='.repeat(60));
    
    if (deptUsers.length === 0) {
      console.log('❌ No department users found in database.');
      console.log('\nTo create department users, register through the portal');
      console.log('with role "department" and get admin approval.');
    } else {
      deptUsers.forEach((user, index) => {
        console.log(`\n${index + 1}. Department: ${user.department || 'Not specified'}`);
        console.log(`   Name: ${user.name}`);
        console.log(`   Email: ${user.email}`);
        console.log(`   Approved: ${user.approved ? '✅ Yes' : '⏳ Pending'}`);
        console.log(`   Phone: ${user.phone || 'N/A'}`);
      });
    }
    
    console.log('\n' + '='.repeat(60));
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

// Run the script
checkDepartments();
