const mongoose = require('mongoose');
const dotenv = require('dotenv');
const path = require('path');
const User = require('./models/User');
const Complaint = require('./models/Complaint');

dotenv.config({ path: path.join(__dirname, '.env') });

mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB connected'))
  .catch((err) => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

const getCitizenAccounts = async () => {
  try {
    // Get all complaints and find who filed them
    const complaints = await Complaint.find().populate('citizen', 'name email phone');
    
    const citizenSet = new Set();
    complaints.forEach(complaint => {
      if (complaint.citizen) {
        citizenSet.add(JSON.stringify({
          id: complaint.citizen._id,
          name: complaint.citizen.name,
          email: complaint.citizen.email,
          phone: complaint.citizen.phone
        }));
      }
    });

    console.log('\n👥 CITIZEN ACCOUNTS THAT FILED COMPLAINTS:\n');
    citizenSet.forEach(citizen => {
      const c = JSON.parse(citizen);
      console.log(`Name: ${c.name}`);
      console.log(`Email: ${c.email}`);
      console.log(`Phone: ${c.phone}`);
      console.log(`ID: ${c.id}\n`);
    });

    // Also get all citizen users
    console.log('\n👥 ALL CITIZEN USERS IN DATABASE:\n');
    const allCitizens = await User.find({ role: 'citizen' }).select('-password');
    
    if (allCitizens.length === 0) {
      console.log('No citizen users found');
    } else {
      allCitizens.forEach(citizen => {
        console.log(`Name: ${citizen.name}`);
        console.log(`Email: ${citizen.email}`);
        console.log(`Phone: ${citizen.phone || 'N/A'}`);
        console.log(`Approved: ${citizen.approved}\n`);
      });
    }

    console.log('\n⚠️  Note: You can try logging in with these emails, but we don\'t know the passwords.');
    console.log('To test the Citizen Dashboard, please register a new citizen account.');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

getCitizenAccounts();
