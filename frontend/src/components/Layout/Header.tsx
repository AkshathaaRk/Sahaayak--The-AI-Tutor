import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { User, Search, Bell, Settings, LogOut, Wifi, WifiOff } from 'lucide-react';
import { useApp } from '../../contexts/AppContext';

const Header: React.FC = () => {
  const { user, isOnline, setUser } = useApp();
  const navigate = useNavigate();

  const handleLogout = () => {
    setUser(null);
    navigate('/');
  };

  return (
    <header className="bg-white border-b border-grey-200 px-4 py-3 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link to="/dashboard" className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-lavender-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">S</span>
          </div>
          <span className="text-xl font-bold text-grey-800">Sahayak</span>
        </Link>

        <div className="flex-1 max-w-md mx-8">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-grey-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search your content..."
              className="w-full pl-10 pr-4 py-2 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
            />
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            {isOnline ? (
              <Wifi className="w-4 h-4 text-green-500" />
            ) : (
              <WifiOff className="w-4 h-4 text-grey-400" />
            )}
            <span className="text-sm text-grey-600">
              {isOnline ? 'Online' : 'Offline'}
            </span>
          </div>

          <button className="p-2 hover:bg-grey-100 rounded-lg transition-colors">
            <Bell className="w-5 h-5 text-grey-600" />
          </button>

          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-lavender-100 rounded-full flex items-center justify-center">
              <User className="w-4 h-4 text-lavender-600" />
            </div>
            <span className="text-sm font-medium text-grey-700">{user?.name}</span>
          </div>

          <Link
            to="/settings"
            className="p-2 hover:bg-grey-100 rounded-lg transition-colors"
          >
            <Settings className="w-5 h-5 text-grey-600" />
          </Link>

          <button
            onClick={handleLogout}
            className="p-2 hover:bg-grey-100 rounded-lg transition-colors"
          >
            <LogOut className="w-5 h-5 text-grey-600" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;