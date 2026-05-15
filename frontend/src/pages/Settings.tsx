import React, { useState } from 'react';
import { User, Bell, Globe, Trash2, Download, Upload, Save } from 'lucide-react';
import Header from '../components/Layout/Header';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import Input from '../components/UI/Input';
import { useApp } from '../contexts/AppContext';

const Settings: React.FC = () => {
  const { user, setUser } = useApp();
  const [activeTab, setActiveTab] = useState('profile');
  const [profileData, setProfileData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    grade: user?.grade || '',
    subjects: user?.subjects?.join(', ') || '',
    region: user?.region || '',
    phone: '',
    school: '',
    experience: ''
  });
  const [settings, setSettings] = useState({
    language: 'English',
    notifications: {
      email: true,
      push: true,
      weekly: true,
      updates: false
    },
    offline: {
      autoSync: true,
      syncFrequency: 'daily',
      cacheSize: '500MB'
    }
  });

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'language', label: 'Language', icon: Globe },
    { id: 'offline', label: 'Offline Settings', icon: Download }
  ];

  const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setProfileData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSettingsChange = (section: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section as keyof typeof prev],
        [key]: value
      }
    }));
  };

  const handleSaveProfile = () => {
    if (user) {
      setUser({
        ...user,
        name: profileData.name,
        email: profileData.email,
        grade: profileData.grade,
        subjects: profileData.subjects.split(',').map(s => s.trim()),
        region: profileData.region
      });
    }
    alert('Profile updated successfully!');
  };

  const handleClearCache = () => {
    if (confirm('Are you sure you want to clear all downloaded content? This action cannot be undone.')) {
      alert('Cache cleared successfully!');
    }
  };

  const renderProfileTab = () => (
    <div className="space-y-6">
      <div className="text-center pb-6 border-b border-grey-200">
        <div className="w-20 h-20 bg-lavender-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <User className="w-10 h-10 text-lavender-600" />
        </div>
        <h3 className="text-lg font-semibold text-grey-800">Teacher Profile</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Input
          label="Full Name"
          type="text"
          name="name"
          value={profileData.name}
          onChange={handleProfileChange}
          required
        />
        
        <Input
          label="Email"
          type="email"
          name="email"
          value={profileData.email}
          onChange={handleProfileChange}
          required
        />
        
        <Input
          label="Phone"
          type="tel"
          name="phone"
          value={profileData.phone}
          onChange={handleProfileChange}
        />
        
        <Input
          label="School Name"
          type="text"
          name="school"
          value={profileData.school}
          onChange={handleProfileChange}
        />
        
        <div className="space-y-2">
          <label className="block text-sm font-medium text-grey-700">Grade Level</label>
          <select
            name="grade"
            value={profileData.grade}
            onChange={handleProfileChange}
            className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
          >
            <option value="">Select grade level</option>
            <option value="Primary">Primary (1-5)</option>
            <option value="Middle">Middle (6-8)</option>
            <option value="Secondary">Secondary (9-12)</option>
            <option value="Multi-grade">Multi-grade</option>
          </select>
        </div>
        
        <Input
          label="Teaching Experience"
          type="text"
          name="experience"
          placeholder="e.g., 5 years"
          value={profileData.experience}
          onChange={handleProfileChange}
        />
      </div>
      
      <Input
        label="Subjects"
        type="text"
        name="subjects"
        placeholder="Mathematics, Science, English (comma-separated)"
        value={profileData.subjects}
        onChange={handleProfileChange}
      />
      
      <Input
        label="Region"
        type="text"
        name="region"
        value={profileData.region}
        onChange={handleProfileChange}
      />
      
      <Button variant="primary" onClick={handleSaveProfile} icon={Save}>
        Save Profile
      </Button>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-grey-800">Notification Preferences</h3>
      
      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-grey-50 rounded-lg">
          <div>
            <h4 className="font-medium text-grey-800">Email Notifications</h4>
            <p className="text-sm text-grey-600">Receive updates and tips via email</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.notifications.email}
              onChange={(e) => handleSettingsChange('notifications', 'email', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-grey-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-lavender-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-grey-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-lavender-500"></div>
          </label>
        </div>

        <div className="flex items-center justify-between p-4 bg-grey-50 rounded-lg">
          <div>
            <h4 className="font-medium text-grey-800">Push Notifications</h4>
            <p className="text-sm text-grey-600">Instant notifications for important updates</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.notifications.push}
              onChange={(e) => handleSettingsChange('notifications', 'push', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-grey-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-lavender-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-grey-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-lavender-500"></div>
          </label>
        </div>

        <div className="flex items-center justify-between p-4 bg-grey-50 rounded-lg">
          <div>
            <h4 className="font-medium text-grey-800">Weekly Summary</h4>
            <p className="text-sm text-grey-600">Weekly digest of your teaching activities</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.notifications.weekly}
              onChange={(e) => handleSettingsChange('notifications', 'weekly', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-grey-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-lavender-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-grey-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-lavender-500"></div>
          </label>
        </div>

        <div className="flex items-center justify-between p-4 bg-grey-50 rounded-lg">
          <div>
            <h4 className="font-medium text-grey-800">Feature Updates</h4>
            <p className="text-sm text-grey-600">Notifications about new features and improvements</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.notifications.updates}
              onChange={(e) => handleSettingsChange('notifications', 'updates', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-grey-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-lavender-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-grey-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-lavender-500"></div>
          </label>
        </div>
      </div>
    </div>
  );

  const renderLanguageTab = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-grey-800">Language Settings</h3>
      
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-grey-700">App Language</label>
          <select
            value={settings.language}
            onChange={(e) => setSettings(prev => ({ ...prev, language: e.target.value }))}
            className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
          >
            <option value="English">English</option>
            <option value="Hindi">Hindi</option>
            <option value="Bengali">Bengali</option>
            <option value="Telugu">Telugu</option>
            <option value="Marathi">Marathi</option>
            <option value="Tamil">Tamil</option>
            <option value="Gujarati">Gujarati</option>
            <option value="Kannada">Kannada</option>
            <option value="Malayalam">Malayalam</option>
            <option value="Punjabi">Punjabi</option>
          </select>
        </div>
        
        <div className="p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-700">
            <strong>Note:</strong> Content generation will still primarily be in English or the language you specify in your prompts. 
            This setting only affects the user interface.
          </p>
        </div>
      </div>
    </div>
  );

  const renderOfflineTab = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-grey-800">Offline Settings</h3>
      
      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-grey-50 rounded-lg">
          <div>
            <h4 className="font-medium text-grey-800">Auto-sync when online</h4>
            <p className="text-sm text-grey-600">Automatically sync your content when connected to internet</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.offline.autoSync}
              onChange={(e) => handleSettingsChange('offline', 'autoSync', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-grey-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-lavender-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-grey-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-lavender-500"></div>
          </label>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-grey-700">Sync Frequency</label>
          <select
            value={settings.offline.syncFrequency}
            onChange={(e) => handleSettingsChange('offline', 'syncFrequency', e.target.value)}
            className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
          >
            <option value="realtime">Real-time</option>
            <option value="hourly">Every hour</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
          </select>
        </div>

        <div className="p-4 bg-grey-50 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-grey-800">Storage Used</h4>
            <span className="text-sm text-grey-600">156 MB of 500 MB</span>
          </div>
          <div className="w-full bg-grey-200 rounded-full h-2">
            <div className="bg-lavender-500 h-2 rounded-full" style={{ width: '31%' }}></div>
          </div>
        </div>

        <div className="flex space-x-4">
          <Button variant="outline" icon={Upload}>
            Export Data
          </Button>
          <Button variant="outline" icon={Trash2} onClick={handleClearCache}>
            Clear Cache
          </Button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-grey-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-grey-800 mb-2">Settings</h1>
          <p className="text-grey-600">Manage your profile and app preferences</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Tabs */}
          <Card padding="sm">
            <nav className="space-y-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-lavender-50 text-lavender-700 border-l-4 border-lavender-500'
                      : 'text-grey-600 hover:bg-grey-50'
                  }`}
                >
                  <tab.icon className="w-5 h-5" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              ))}
            </nav>
          </Card>

          {/* Content */}
          <Card className="lg:col-span-3">
            {activeTab === 'profile' && renderProfileTab()}
            {activeTab === 'notifications' && renderNotificationsTab()}
            {activeTab === 'language' && renderLanguageTab()}
            {activeTab === 'offline' && renderOfflineTab()}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Settings;