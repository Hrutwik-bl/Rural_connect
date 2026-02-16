const http = require('http');

// First, get an admin token
const loginData = JSON.stringify({
  email: 'admin@ruralportal.com',
  password: 'admin@123'
});

const loginOptions = {
  hostname: 'localhost',
  port: 5000,
  path: '/api/auth/login',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(loginData)
  }
};

const loginReq = http.request(loginOptions, (res) => {
  let data = '';
  res.on('data', (chunk) => {
    data += chunk;
  });
  res.on('end', () => {
    const loginResponse = JSON.parse(data);
    console.log('✅ Login successful');
    console.log(`Token: ${loginResponse.token.substring(0, 30)}...`);
    console.log(`Role: ${loginResponse.role}\n`);

    // Now test getting all complaints
    testGetAllComplaints(loginResponse.token);
  });
});

loginReq.write(loginData);
loginReq.end();

function testGetAllComplaints(token) {
  const getAllOptions = {
    hostname: 'localhost',
    port: 5000,
    path: '/api/complaints/all',
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  const getAllReq = http.request(getAllOptions, (res) => {
    console.log(`Testing /api/complaints/all`);
    console.log(`Status: ${res.statusCode}`);
    let data = '';
    res.on('data', (chunk) => {
      data += chunk;
    });
    res.on('end', () => {
      const complaints = JSON.parse(data);
      console.log(`Found ${complaints.length} complaints`);
      if (complaints.length > 0) {
        console.log(`First complaint: ${complaints[0].title}\n`);
      }
    });
  });

  getAllReq.on('error', (e) => {
    console.error('Error:', e.message);
  });

  getAllReq.end();
}
