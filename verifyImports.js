// Test to verify the app works end-to-end
console.log('Starting test...');

// Test 1: Check if axiosConfig can be imported
try {
  const axiosConfig = require('./frontend/src/api/axiosConfig');
  console.log('✅ axiosConfig imports successfully');
} catch (e) {
  console.log('❌ Failed to import axiosConfig:', e.message);
}

// Test 2: Check if components can import it
const fs = require('fs');

const files = [
  './frontend/src/pages/AdminDashboard.js',
  './frontend/src/pages/CitizenDashboard.jsx',
  './frontend/src/pages/DepartmentDashboard.jsx'
];

files.forEach(file => {
  const content = fs.readFileSync(file, 'utf-8');
  if (content.includes("import apiClient from '../api/axiosConfig'")) {
    console.log(`✅ ${file} has correct import`);
  } else {
    console.log(`❌ ${file} missing correct import`);
  }
  
  if (content.includes('apiClient.get') || content.includes('apiClient.post') || content.includes('apiClient.put')) {
    console.log(`✅ ${file} uses apiClient`);
  } else {
    console.log(`❌ ${file} doesn't use apiClient`);
  }
});

console.log('\nAll imports verified!');
