import React from 'react';
import { Link } from 'react-router-dom';
import { 
  PlusCircle, 
  Upload, 
  MessageCircle, 
  Compass, 
  Clock,
  TrendingUp,
  Users,
  BookOpen,
  Download,
  Eye,
  Trash2
} from 'lucide-react';
import Header from '../components/Layout/Header';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import { useApp } from '../contexts/AppContext';

const Dashboard: React.FC = () => {
  const { user, recentContent } = useApp();

  const quickActions = [
    {
      title: 'Generate New Content',
      description: 'Create lesson plans, stories, and activities',
      icon: PlusCircle,
      link: '/generate',
      color: 'bg-lavender-500'
    },
    {
      title: 'Upload Documents',
      description: 'Process files into study materials',
      icon: Upload,
      link: '/upload',
      color: 'bg-blue-500'
    },
    {
      title: 'AI Teacher Coach',
      description: 'Get personalized teaching advice',
      icon: MessageCircle,
      link: '/coach',
      color: 'bg-green-500'
    },
    {
      title: 'Cultural Pathways',
      description: 'Explore culturally responsive activities',
      icon: Compass,
      link: '/pathways',
      color: 'bg-orange-500'
    },
  ];

  const stats = [
    { label: 'Content Created', value: '47', icon: BookOpen },
    { label: 'Students Impacted', value: '120', icon: Users },
    { label: 'Weekly Growth', value: '+15%', icon: TrendingUp },
    { label: 'Time Saved', value: '12h', icon: Clock },
  ];

  const mockRecentContent = [
    {
      id: '1',
      title: 'Mathematics - Fractions for Grade 4',
      type: 'lesson',
      createdAt: '2 hours ago',
      subject: 'Mathematics'
    },
    {
      id: '2',
      title: 'Story: The Clever Farmer',
      type: 'story',
      createdAt: '1 day ago',
      subject: 'Language Arts'
    },
    {
      id: '3',
      title: 'Science Quiz - Water Cycle',
      type: 'quiz',
      createdAt: '3 days ago',
      subject: 'Science'
    },
  ];

  return (
    <div className="min-h-screen bg-grey-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-grey-800 mb-2">
            Welcome back, {user?.name}!
          </h1>
          <p className="text-grey-600">
            Ready to create amazing learning experiences for your students?
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {stats.map((stat) => (
            <Card key={stat.label} className="text-center">
              <div className="flex items-center justify-center mb-2">
                <stat.icon className="w-8 h-8 text-lavender-600" />
              </div>
              <div className="text-2xl font-bold text-grey-800 mb-1">{stat.value}</div>
              <div className="text-sm text-grey-600">{stat.label}</div>
            </Card>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-grey-800 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {quickActions.map((action) => (
              <Link key={action.title} to={action.link}>
                <Card hover className="text-center h-full">
                  <div className={`w-12 h-12 ${action.color} rounded-lg flex items-center justify-center mx-auto mb-4`}>
                    <action.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-grey-800 mb-2">{action.title}</h3>
                  <p className="text-grey-600 text-sm">{action.description}</p>
                </Card>
              </Link>
            ))}
          </div>
        </div>

        {/* Recent Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-grey-800">Recent Content</h2>
              <Button variant="outline" size="sm">View All</Button>
            </div>
            <Card>
              <div className="space-y-4">
                {mockRecentContent.map((item) => (
                  <div key={item.id} className="flex items-center justify-between p-4 bg-grey-50 rounded-lg">
                    <div className="flex-1">
                      <h3 className="font-semibold text-grey-800">{item.title}</h3>
                      <p className="text-sm text-grey-600">{item.subject} • {item.createdAt}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="p-2 text-grey-500 hover:text-grey-700">
                        <Eye className="w-4 h-4" />
                      </button>
                      <button className="p-2 text-grey-500 hover:text-grey-700">
                        <Download className="w-4 h-4" />
                      </button>
                      <button className="p-2 text-grey-500 hover:text-red-500">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Tips & Announcements */}
          <div>
            <h2 className="text-xl font-semibold text-grey-800 mb-4">Tips & Updates</h2>
            <Card>
              <div className="space-y-6">
                <div className="p-4 bg-lavender-50 rounded-lg">
                  <h3 className="font-semibold text-lavender-800 mb-2">New Feature!</h3>
                  <p className="text-lavender-700 text-sm">
                    Try our new Cultural Pathways feature to create locally relevant learning experiences.
                  </p>
                </div>
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-semibold text-blue-800 mb-2">Teaching Tip</h3>
                  <p className="text-blue-700 text-sm">
                    Use the AI Coach to get strategies for managing multi-grade classrooms effectively.
                  </p>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <h3 className="font-semibold text-green-800 mb-2">Offline Mode</h3>
                  <p className="text-green-700 text-sm">
                    Download content to access it offline. Perfect for areas with limited connectivity.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;