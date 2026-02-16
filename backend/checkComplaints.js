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

const checkComplaints = async () => {
  try {
    const complaints = await Complaint.find()
      .populate('citizen', 'name email')
      .sort({ createdAt: -1 });
    
    if (complaints.length === 0) {
      console.log('ℹ️  No complaints in database yet.');
      console.log('To create test complaints:');
      console.log('1. Login as citizen (register a citizen account)');
      console.log('2. Go to Citizen Dashboard');
      console.log('3. Create a complaint with title, description, and upload an image');
      console.log('4. The AI will analyze it and assign to a department');
    } else {
      console.log(`✅ Found ${complaints.length} complaint(s):\n`);
      complaints.forEach((complaint, index) => {
        console.log(`${index + 1}. ID: ${complaint._id}`);
        console.log(`   Title: ${complaint.title}`);
        console.log(`   Description: ${complaint.description.substring(0, 50)}...`);
        console.log(`   Category: ${complaint.category}`);
        console.log(`   Severity: ${complaint.severity}`);
        console.log(`   Status: ${complaint.status}`);
        console.log(`   Citizen: ${complaint.citizen ? complaint.citizen.name : 'Unknown'}`);
        console.log('');
      });
    }
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

checkComplaints();
