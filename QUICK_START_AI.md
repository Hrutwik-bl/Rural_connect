# 🚀 Quick Start Guide - AI Features

## Prerequisites
- Node.js installed
- Python 3.8+ installed
- MongoDB running
- Backend running on port 5000
- Frontend running on port 3000

---

## 📋 Step-by-Step Setup

### 1️⃣ Start Backend Server
```bash
cd backend
npm install  # First time only
npm start
```
✅ Should see: `Server running on port 5000`

### 2️⃣ Start Frontend Server
```bash
cd frontend
npm install  # First time only
npm run dev
```
✅ Should see: `Local: http://localhost:3000`

### 3️⃣ Start ML/AI Service (Optional but Recommended)
```bash
cd DL_models
pip install -r requirements.txt  # First time only
python api.py
```
✅ Should see: `Running on http://localhost:8000`

---

## 🎯 How to Test AI Features

### 1. Access Application
- Open browser: **http://localhost:3000**
- Register as a **Citizen**
- Login with your credentials

### 2. File Complaint with AI
1. Click **"+ New Complaint"** button
2. Fill in the form:
   - **Title**: "Pothole on Main Street"
   - **Description**: "There's a large pothole causing traffic accidents"
   - **Location**: "Main Street, Downtown"
3. **Upload Image**: Choose a photo of road damage
4. Click **"🤖 Analyze with AI"** button
5. Wait 2-3 seconds for predictions
6. See AI predictions on the right panel:
   - ✅ Predicted Department: **Roads**
   - ✅ Predicted Severity: **High**
   - ✅ Image Validation: **Valid** (92% match)
7. Click **"✓ Apply AI Predictions"**
8. Click **"Submit Complaint"**

### 3. View in Department Dashboard
- Login as **Department** user (Water/Electricity/Roads)
- See complaint auto-routed to correct department
- View the severity and deadline info

### 4. Admin Approval
- Login as **Admin**
- See all complaints with AI predictions
- View complaint details and progress

---

## 🧪 Test Cases

### Test 1: Road Damage
**Expected**: Routes to "Roads" department
- Title: "Big pothole"
- Description: "Large road damage"
- Image: Photo of pothole
- Expected Severity: High
- Expected Deadline: 48 hours

### Test 2: Power Issue
**Expected**: Routes to "Electricity" department
- Title: "No power"
- Description: "Complete blackout in area"
- Image: Photo of power lines
- Expected Severity: Critical
- Expected Deadline: 24 hours

### Test 3: Water Problem
**Expected**: Routes to "Water" department
- Title: "Water leak"
- Description: "Burst water main"
- Image: Photo of water leak
- Expected Severity: High
- Expected Deadline: 48 hours

---

## 🔍 Troubleshooting

### Issue: AI Prediction Button Disabled
**Solution**: 
- Ensure description is entered (minimum 10 characters)
- Ensure image is uploaded
- Both fields are required

### Issue: "AI service temporarily unavailable"
**Solution**:
```bash
# Check if ML service is running
curl http://localhost:8000/health

# If not running, start it:
cd DL_models
python api.py
```

### Issue: Image Validation Fails
**Solution**:
- Upload image matching the description
- Use clear, well-lit photos
- Ensure image is relevant to the issue described
- Supported formats: PNG, JPG, JPEG

### Issue: No Complaint Visible in Dashboard
**Solution**:
```bash
# Check MongoDB connection in backend
# Check backend logs for errors
cd backend
npm start  # Restart backend
```

---

## 📊 Live Testing URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000/api
- **ML API**: http://localhost:8000/predict-complaint
- **Admin Panel**: http://localhost:3000 (login as admin)

---

## 🧑‍💻 Test Credentials

### Admin User
- Email: `admin@test.com`
- Password: `Admin@123`

### Department User
- Email: `dept_water@test.com`
- Password: `Dept@123`

### Citizen Users
- Create your own through registration

---

## 📈 Monitoring

### Check AI Performance
```bash
# See ML API logs
cd DL_models
# Check console output for prediction accuracy
```

### Backend Logs
```bash
cd backend
# Check terminal for request logs
npm start
```

### Frontend Logs
```bash
# Open browser Developer Tools (F12)
# Console tab shows all API calls
```

---

## ✅ Verification Checklist

- [ ] Backend running on port 5000
- [ ] Frontend running on port 3000
- [ ] ML service running on port 8000
- [ ] MongoDB connected
- [ ] Can register as citizen
- [ ] Can login successfully
- [ ] Can file complaint
- [ ] AI button is clickable
- [ ] AI predictions appear
- [ ] Complaint submits successfully
- [ ] Complaint appears in department dashboard

---

## 🎓 AI Features Summary

| Feature | Status | Location |
|---------|--------|----------|
| Image Upload | ✅ Active | Citizen Dashboard |
| AI Prediction | ✅ Active | Modal Form |
| Department Detection | ✅ Active | Backend API |
| Severity Classification | ✅ Active | Backend API |
| Image Validation | ✅ Active | ML Service |
| Deadline Assignment | ✅ Active | Complaint Model |
| Manual Override | ✅ Active | Citizen Dashboard |
| Confidence Scores | ✅ Active | Prediction Panel |
| Fallback Mode | ✅ Active | No AI service |

---

## 🎯 Next Steps

1. Test AI features with sample complaints
2. Verify predictions are accurate
3. Check deadline calculations
4. Monitor ML service performance
5. Provide feedback for model improvement

---

## 💡 Tips

- Use high-quality images for better predictions
- Write descriptive complaint titles
- Provide detailed descriptions in the form
- Wait for AI analysis before submitting
- Review AI predictions before applying

---

## 🆘 Support

For issues:
1. Check troubleshooting section above
2. Review backend logs
3. Check ML service status
4. Verify all servers are running
5. Clear browser cache if needed

---

**Status**: ✅ AI Features Ready for Testing!
