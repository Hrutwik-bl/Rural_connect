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

const createTestCitizens = async () => {
  try {
    // Create test citizen accounts
    const testCitizens = [
      {
        name: 'Test Citizen 1',
        email: 'citizen1@test.com',
        password: 'password123',
        role: 'citizen',
        phone: '9876543210',
        address: 'Test Address 1'
      },
      {
        name: 'Test Citizen 2',
        email: 'citizen2@test.com',
        password: 'password123',
        role: 'citizen',
        phone: '9876543211',
        address: 'Test Address 2'
      }
    ];

    console.log('Creating test citizen accounts...\n');

    for (const citizenData of testCitizens) {
      const existingUser = await User.findOne({ email: citizenData.email });
      
      if (existingUser) {
        console.log(`⚠️  ${citizenData.email} already exists`);
      } else {
        const newCitizen = await User.create(citizenData);
        console.log(`✅ Created: ${citizenData.name}`);
        console.log(`   Email: ${citizenData.email}`);
        console.log(`   Password: ${citizenData.password}`);
        console.log(`   Role: ${citizenData.role}\n`);
      }
    }

    console.log('✅ All test citizen accounts ready!\n');
    console.log('LOGIN CREDENTIALS:');
    console.log('==================');
    console.log('Email: citizen1@test.com');
    console.log('Password: password123\n');
    console.log('Email: citizen2@test.com');
    console.log('Password: password123');

    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

createTestCitizens();
