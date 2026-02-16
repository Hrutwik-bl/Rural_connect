import React, { createContext, useState, useContext, useEffect } from 'react';
import apiClient from '../api/axiosConfig';

// Configure axios to use backend URL directly (fallback for direct axios calls)
const API_BASE_URL = 'http://localhost:5000';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in and validate token (per-tab session)
    let token = sessionStorage.getItem('token');
    let userData = sessionStorage.getItem('user');

    // Migrate any legacy localStorage auth to sessionStorage (one-time)
    if ((!token || !userData) && localStorage.getItem('token') && localStorage.getItem('user')) {
      token = localStorage.getItem('token');
      userData = localStorage.getItem('user');
      sessionStorage.setItem('token', token);
      sessionStorage.setItem('user', userData);
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }

    if (token && userData) {
      try {
        const parsedUser = JSON.parse(userData);
        apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        
        // Validate token with backend
        apiClient.get('/api/auth/me')
          .then((response) => {
            // Token is valid, update with latest user data
            setUser(response.data);
            localStorage.setItem('user', JSON.stringify(response.data));
          })
          .catch((error) => {
            console.error('Token validation failed:', error);
            // Token is invalid, clear storage
            sessionStorage.removeItem('token');
            sessionStorage.removeItem('user');
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            delete apiClient.defaults.headers.common['Authorization'];
            setUser(null);
          })
          .finally(() => {
            setLoading(false);
          });
      } catch (error) {
        console.error('Failed to parse user data:', error);
        sessionStorage.removeItem('token');
        sessionStorage.removeItem('user');
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setLoading(false);
      }
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (email, password) => {
    try {
      // Clear any previous authentication data
      sessionStorage.removeItem('token');
      sessionStorage.removeItem('user');
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      delete apiClient.defaults.headers.common['Authorization'];
      setUser(null);
      
      const response = await apiClient.post('/api/auth/login', { email, password });
      const { token, ...userData } = response.data;
      
      // Set new user data
      sessionStorage.setItem('token', token);
      sessionStorage.setItem('user', JSON.stringify(userData));
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      setUser(userData);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        message: error.response?.data?.message || 'Login failed' 
      };
    }
  };

  const register = async (userData) => {
    try {
      // Clear any previous authentication data
      sessionStorage.removeItem('token');
      sessionStorage.removeItem('user');
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      delete apiClient.defaults.headers.common['Authorization'];
      setUser(null);
      
      const response = await apiClient.post('/api/auth/register', userData);
      
      // Check if department user needs approval
      if (response.data.requiresApproval) {
        return { 
          success: true, 
          requiresApproval: true,
          message: response.data.message 
        };
      }
      
      const { token, ...userInfo } = response.data;
      
      sessionStorage.setItem('token', token);
      sessionStorage.setItem('user', JSON.stringify(userInfo));
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      setUser(userInfo);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        message: error.response?.data?.message || 'Registration failed' 
      };
    }
  };

  const logout = () => {
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('user');
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete apiClient.defaults.headers.common['Authorization'];
    setUser(null);
  };

  const value = {
    user,
    login,
    register,
    logout,
    loading
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
