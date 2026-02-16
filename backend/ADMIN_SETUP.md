# Admin Setup Guide

## Creating Admin Accounts (Developers Only)

Admin accounts cannot be created through the registration portal for security reasons. Only developers with database access can create admin accounts.

### Method 1: Using the createAdmin.js Script

1. **Edit the admin details** in `createAdmin.js`:
   ```javascript
   const adminData = {
     name: 'System Administrator',
     email: 'admin@ruralportal.com',
     password: 'admin@123',
     role: 'admin',
     phone: '1234567890',
     address: 'System',
     approved: true
   };
   ```

2. **Run the script**:
   ```bash
   cd backend
   node createAdmin.js
   ```

3. **Login credentials** will be displayed in the console.

⚠️ **Important**: Change the password immediately after first login!

### Method 2: Using MongoDB Directly

If you prefer to create an admin user directly in MongoDB:

1. Connect to your MongoDB database
2. Use the following command (replace values as needed):

```javascript
db.users.insertOne({
  name: "Admin Name",
  email: "admin@example.com",
  password: "$2a$10$hashedPasswordHere", // Use bcrypt to hash
  role: "admin",
  approved: true,
  createdAt: new Date()
})
```

**Note**: You'll need to hash the password using bcrypt first.

## User Approval Workflow

### For Citizens:
- ✅ Can register freely without approval
- ✅ Get immediate access after registration

### For Department Heads:
- ⏳ Register through the portal
- ⏳ See message: "Registration successful! Pending admin approval"
- ❌ Cannot login until approved
- ✅ Admin reviews and approves/rejects
- ✅ Can login after approval

### For Admins:
- ❌ Cannot register through portal
- ✅ Only created by developers using scripts
- ✅ Have full system access

## Admin Approval Process

1. Department head registers through the portal
2. Admin receives notification in dashboard
3. Admin reviews the registration in "Pending Approvals" tab
4. Admin can:
   - **Approve**: User gets access and can login
   - **Reject**: Registration is deleted

## Security Notes

- Admin role is restricted to prevent unauthorized access
- Department accounts require manual approval
- All passwords are hashed using bcrypt
- JWT tokens are used for authentication
