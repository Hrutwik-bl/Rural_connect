# Rural Service Request Portal

A comprehensive MERN stack application that enables rural citizens to raise service complaints, departments to manage assigned complaints, and administrators to oversee all operations.

## 🚀 Features

### For Citizens
- Register and login with secure authentication
- Raise complaints about various service issues (Water, Electricity, Roads, Sanitation, Healthcare, Education, etc.)
- Track complaint status in real-time
- View complaint history and details

### For Departments
- View complaints specific to their department
- Update complaint status (Pending, In Progress, Resolved, Rejected)
- Add remarks and notes to complaints
- Track department performance metrics

### For Administrators
- View all complaints across all departments
- Escalate critical complaints
- Manage users and departments
- Access comprehensive analytics and statistics
- Delete complaints if necessary

## 🛠️ Technology Stack

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **MongoDB** - Database
- **Mongoose** - ODM for MongoDB
- **JWT** - Authentication
- **bcryptjs** - Password hashing
- **express-validator** - Input validation

### Frontend
- **React** - UI library
- **React Router** - Routing
- **Bootstrap & React-Bootstrap** - UI components
- **Axios** - HTTP client
- **Context API** - State management

## 📁 Project Structure

```
rural-service-portal/
├── backend/
│   ├── controllers/
│   │   ├── authController.js
│   │   ├── complaintController.js
│   │   └── userController.js
│   ├── models/
│   │   ├── User.js
│   │   └── Complaint.js
│   ├── routes/
│   │   ├── auth.js
│   │   ├── complaints.js
│   │   └── users.js
│   ├── middleware/
│   │   └── auth.js
│   ├── .env
│   ├── server.js
│   └── package.json
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navigation.js
│   │   │   └── PrivateRoute.js
│   │   ├── context/
│   │   │   └── AuthContext.js
│   │   ├── pages/
│   │   │   ├── LandingPage.js
│   │   │   ├── Login.js
│   │   │   ├── Register.js
│   │   │   ├── CitizenDashboard.js
│   │   │   ├── DepartmentDashboard.js
│   │   │   └── AdminDashboard.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- MongoDB (v4.4 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd new_ruralConnect
```

2. **Install Backend Dependencies**
```bash
cd backend
npm install
```

3. **Install Frontend Dependencies**
```bash
cd ../frontend
npm install
```

4. **Setup MongoDB**
- Make sure MongoDB is installed and running on your system
- Default connection: `mongodb://localhost:27017/ruralServicePortal`
- Or use MongoDB Atlas for cloud database

5. **Configure Environment Variables**
   
Create a `.env` file in the `backend` folder with the following:
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/ruralServicePortal
JWT_SECRET=your_jwt_secret_key_change_this_in_production
NODE_ENV=development
```

### Running the Application

#### Start Backend Server
```bash
cd backend
npm start
```
The backend server will run on `http://localhost:5000`

For development with auto-restart:
```bash
npm run dev
```

#### Start Frontend Application
```bash
cd frontend
npm start
```
The React app will run on `http://localhost:3000`

## 👥 User Roles & Access

### Citizen
- ✅ Can register freely without approval
- ✅ Get immediate access after registration
- ✅ Can raise new complaints
- ✅ View their own complaints
- ✅ Track complaint status

### Department Head
- 📝 Can register through the portal
- ⏳ Registration requires admin approval
- ❌ Cannot login until approved by admin
- ✅ After approval:
  - View complaints assigned to their department
  - Update complaint status
  - Add remarks to complaints

### Admin
- ❌ **Cannot register through the portal**
- 🔐 **Only created by developers** using scripts
- ✅ Have full system access:
  - View all complaints
  - Escalate complaints
  - Approve/reject department registrations
  - Manage users
  - Delete complaints
  - Access full analytics

## 🔐 Admin Creation (Developers Only)

Admins cannot be created through the registration portal. Use the provided script:

1. **Edit admin details** in `backend/createAdmin.js`
2. **Run the command:**
```bash
cd backend
npm run create-admin
```

See [backend/ADMIN_SETUP.md](backend/ADMIN_SETUP.md) for detailed instructions.

## 📋 User Registration & Approval Workflow

### For Citizens (Free Registration)
1. Go to Registration page
2. Fill in details and select "Citizen" role
3. Submit registration
4. ✅ Immediate access - can login right away

### For Department Heads (Requires Approval)
1. Go to Registration page
2. Fill in details and select "Department Head" role
3. Select department (Water, Electricity, Roads, etc.)
4. Submit registration
5. ⏳ See message: "Registration successful! Pending admin approval"
6. ❌ Cannot login yet
7. Wait for admin to approve
8. ✅ After approval, can login freely

### For Admins (Developer Created)
1. Only developers with database access can create admins
2. Use `npm run create-admin` script in backend folder
3. ✅ Can login immediately with created credentials

## 🧪 Demo Credentials

### Admin Account
Use the script to create admin (see Admin Creation section above), or use existing:
- Email: admin@ruralportal.com
- Password: admin@123
- Role: admin

### Testing Department Approval Workflow
1. Register a new department user through the portal
2. Login as admin
3. Go to "Pending Approvals" tab in admin dashboard
4. Approve the department registration
5. Logout and login as the approved department user

### Citizen Account
Register any citizen account through the portal - instant access!

## 📡 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user (Protected)

### Complaints
- `POST /api/complaints` - Create complaint (Citizen)
- `GET /api/complaints/my-complaints` - Get user's complaints (Citizen)
- `GET /api/complaints/all` - Get all complaints (Admin/Department)
- `GET /api/complaints/department/:department` - Get complaints by department
- `GET /api/complaints/:id` - Get single complaint
- `PUT /api/complaints/:id/status` - Update complaint status (Department/Admin)
- `PUT /api/complaints/:id/escalate` - Escalate complaint (Admin)
- `DELETE /api/complaints/:id` - Delete complaint (Admin)

### Users
- `GET /api/users` - Get all users (Admin)
- `GET /api/users/role/:role` - Get users by role (Admin)
- `GET /api/users/:id` - Get single user (Admin)
- `PUT /api/users/:id` - Update user (Admin)
- `DELETE /api/users/:id` - Delete user (Admin)

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcryptjs
- Role-based access control
- Protected API routes
- Input validation
- CORS enabled

## 🎨 Key Features Explained

### Complaint Management
- Multiple status tracking (Pending, In Progress, Resolved, Rejected, Escalated)
- Priority levels (Low, Medium, High, Critical)
- Department categorization
- Location tracking

### Real-time Updates
- Automatic complaint status updates
- Dashboard statistics
- Category-wise complaint distribution

### User Management
- Secure registration and login
- Profile management
- Role-based dashboards

## 🐛 Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running: `mongod` or check system services
- Verify connection string in `.env` file
- Check if port 27017 is not being used by another application

### Port Already in Use
- Backend: Change PORT in `.env` file
- Frontend: Set PORT environment variable before starting
  ```bash
  PORT=3001 npm start
  ```

### CORS Issues
- Backend is configured to accept requests from all origins during development
- For production, update CORS configuration in `server.js`

## 📝 Future Enhancements

- [ ] File upload for complaints (images/documents)
- [ ] Email notifications
- [ ] SMS alerts
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Mobile responsive improvements
- [ ] Real-time chat support
- [ ] Complaint rating system
- [ ] Export reports (PDF/Excel)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Development

### Code Structure
- Follow ES6+ standards
- Use async/await for asynchronous operations
- Implement proper error handling
- Add comments for complex logic

### Testing
```bash
# Backend tests
cd backend
npm test

# Frontend tests
cd frontend
npm test
```

## 📞 Support

For support, email support@ruralserviceportal.com or raise an issue in the repository.

---

**Built with ❤️ for Rural Communities**
