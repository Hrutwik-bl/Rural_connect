import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const PrivateRoute = ({ children, role }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="text-center mt-5">Loading...</div>;
  }

  if (!user) {
    return <Navigate to="/login" />;
  }

  // Check if user role matches the required role
  if (role && user.role !== role) {
    // Redirect to appropriate dashboard based on actual user role
    if (user.role === 'admin') {
      return <Navigate to="/admin-dashboard" />;
    } else if (user.role === 'department') {
      return <Navigate to="/department-dashboard" />;
    } else if (user.role === 'citizen') {
      return <Navigate to="/citizen-dashboard" />;
    }
    return <Navigate to="/" />;
  }

  return children;
};

export default PrivateRoute;
