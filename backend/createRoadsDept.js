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

// Roads Department user data
const roadsDeptData = {
  name: 'Roads Department Head',
  email: 'roads@dept.com',
  password: 'Roads@123',
  role: 'department',
  department: 'Roads',
  phone: '9876543210',
  address: 'Roads Department Office',
  approved: true
};

// Create Roads Department user
const createRoadsDept = async () => {
  try {
    // Check if already exists
    const existing = await User.findOne({ email: roadsDeptData.email });
    
    if (existing) {
      console.log('❌ Roads Department user already exists!');
      console.log(`Email: ${existing.email}`);
      console.log(`Department: ${existing.department}`);
      process.exit(1);
    }

    // Create new department user
    const dept = await User.create(roadsDeptData);

    console.log('\n✅ Roads Department user created successfully!');
    console.log('='.repeat(60));
    console.log(`Name: ${dept.name}`);
    console.log(`Email: ${dept.email}`);
    console.log(`Password: ${roadsDeptData.password}`);
    console.log(`Department: ${dept.department}`);
    console.log(`Role: ${dept.role}`);
    console.log(`Approved: ${dept.approved ? 'Yes' : 'No'}`);
    console.log('='.repeat(60));
    console.log('\n🔐 Login Credentials:');
    console.log(`   Email: ${dept.email}`);
    console.log(`   Password: ${roadsDeptData.password}`);
    console.log('\n⚠️  You can change the password after first login!');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error creating Roads Department user:', error.message);
    process.exit(1);
  }
};

// Run the script
createRoadsDept();
