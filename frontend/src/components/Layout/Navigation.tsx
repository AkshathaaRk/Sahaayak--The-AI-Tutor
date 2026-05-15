import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Home, 
  FileText, 
  Upload, 
  MessageCircle, 
  Compass, 
  Settings,
  User
} from 'lucide-react';

const Navigation: React.FC = () => {
  const navItems = [
    { to: '/dashboard', icon: Home, label: 'Dashboard' },
    { to: '/generate', icon: FileText, label: 'Generate Content' },
    { to: '/upload', icon: Upload, label: 'Process Files' },
    { to: '/coach', icon: MessageCircle, label: 'AI Coach' },
    { to: '/pathways', icon: Compass, label: 'Cultural Pathways' },
    { to: '/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <nav className="bg-white border-r border-grey-200 w-64 min-h-screen p-4">
      <div className="space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-lavender-50 text-lavender-700 border-l-4 border-lavender-500'
                  : 'text-grey-600 hover:bg-grey-50 hover:text-grey-800'
              }`
            }
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium">{item.label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
};

export default Navigation;