import React from 'react';
import { Link } from 'react-router-dom';
import { 
  BookOpen, 
  Users, 
  Globe, 
  Sparkles, 
  CheckCircle, 
  ArrowRight,
  Play,
  Star
} from 'lucide-react';
import Button from '../components/UI/Button';
import Card from '../components/UI/Card';

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-lavender-50 to-white">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-grey-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-lavender-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <span className="text-xl font-bold text-grey-800">Sahayak</span>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/login" className="text-grey-600 hover:text-grey-800 transition-colors">
              Login
            </Link>
            <Link to="/register">
              <Button variant="primary" size="sm">Get Started</Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 py-20 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold text-grey-800 mb-6 animate-fade-in">
            Your AI Teaching Assistant for 
            <span className="text-lavender-600"> Multi-Grade Classrooms</span>
          </h1>
          <p className="text-xl text-grey-600 mb-8 animate-slide-up">
            Empower your teaching with culturally responsive AI that understands your local context. 
            Generate engaging content, get personalized coaching, and create meaningful learning experiences.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-slide-up">
            <Link to="/register">
              <Button variant="primary" size="lg" icon={ArrowRight}>
                Start Teaching Smarter
              </Button>
            </Link>
            <Button variant="outline" size="lg" icon={Play}>
              Watch Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-grey-800 mb-4">
            Everything You Need to Excel in Multi-Grade Teaching
          </h2>
          <p className="text-xl text-grey-600 max-w-2xl mx-auto">
            Designed specifically for teachers managing multiple grade levels with limited resources.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <Card hover className="text-center">
            <div className="w-12 h-12 bg-lavender-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <BookOpen className="w-6 h-6 text-lavender-600" />
            </div>
            <h3 className="text-xl font-semibold text-grey-800 mb-2">Smart Content Generation</h3>
            <p className="text-grey-600">
              Create lesson plans, stories, and activities tailored to your students' grade levels and local context.
            </p>
          </Card>

          <Card hover className="text-center">
            <div className="w-12 h-12 bg-lavender-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Users className="w-6 h-6 text-lavender-600" />
            </div>
            <h3 className="text-xl font-semibold text-grey-800 mb-2">AI Teacher Coach</h3>
            <p className="text-grey-600">
              Get personalized advice for classroom management, student engagement, and teaching strategies.
            </p>
          </Card>

          <Card hover className="text-center">
            <div className="w-12 h-12 bg-lavender-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Globe className="w-6 h-6 text-lavender-600" />
            </div>
            <h3 className="text-xl font-semibold text-grey-800 mb-2">Cultural Integration</h3>
            <p className="text-grey-600">
              Connect learning to local culture, traditions, and community knowledge for meaningful education.
            </p>
          </Card>

          <Card hover className="text-center">
            <div className="w-12 h-12 bg-lavender-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-6 h-6 text-lavender-600" />
            </div>
            <h3 className="text-xl font-semibold text-grey-800 mb-2">File Processing</h3>
            <p className="text-grey-600">
              Transform PDFs, videos, and presentations into student-friendly summaries and study materials.
            </p>
          </Card>

          <Card hover className="text-center">
            <div className="w-12 h-12 bg-lavender-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-6 h-6 text-lavender-600" />
            </div>
            <h3 className="text-xl font-semibold text-grey-800 mb-2">Offline Support</h3>
            <p className="text-grey-600">
              Access your content and tools even without internet connection. Sync when you're back online.
            </p>
          </Card>

          <Card hover className="text-center">
            <div className="w-12 h-12 bg-lavender-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Star className="w-6 h-6 text-lavender-600" />
            </div>
            <h3 className="text-xl font-semibold text-grey-800 mb-2">Personalized Learning</h3>
            <p className="text-grey-600">
              Adapt content for different learning levels within the same classroom effectively.
            </p>
          </Card>
        </div>
      </section>

      {/* Testimonials */}
      <section className="bg-grey-50 py-20">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-grey-800 mb-4">
              Trusted by Teachers Across Communities
            </h2>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <Card>
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-lavender-100 rounded-full flex items-center justify-center mr-3">
                  <span className="text-lavender-600 font-semibold">P</span>
                </div>
                <div>
                  <h4 className="font-semibold text-grey-800">Priya Sharma</h4>
                  <p className="text-sm text-grey-600">Primary Teacher, Karnataka</p>
                </div>
              </div>
              <p className="text-grey-600">
                "Sahayak has transformed how I teach my multi-grade classroom. The cultural content helps my students connect with their heritage while learning."
              </p>
            </Card>

            <Card>
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-lavender-100 rounded-full flex items-center justify-center mr-3">
                  <span className="text-lavender-600 font-semibold">R</span>
                </div>
                <div>
                  <h4 className="font-semibold text-grey-800">Rajesh Kumar</h4>
                  <p className="text-sm text-grey-600">Secondary Teacher, Rajasthan</p>
                </div>
              </div>
              <p className="text-grey-600">
                "The AI coach feature is like having a mentor available 24/7. It's helped me become a more confident teacher."
              </p>
            </Card>

            <Card>
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-lavender-100 rounded-full flex items-center justify-center mr-3">
                  <span className="text-lavender-600 font-semibold">A</span>
                </div>
                <div>
                  <h4 className="font-semibold text-grey-800">Anita Patel</h4>
                  <p className="text-sm text-grey-600">Rural School Teacher, Gujarat</p>
                </div>
              </div>
              <p className="text-grey-600">
                "Even with limited internet, I can access all my materials offline. This has been a game-changer for our remote school."
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-4 py-20 text-center">
        <div className="bg-lavender-600 rounded-2xl p-12 text-white">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Transform Your Teaching?
          </h2>
          <p className="text-xl mb-8 text-lavender-100">
            Join thousands of teachers already using Sahayak to create engaging, culturally responsive learning experiences.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/register">
              <Button variant="secondary" size="lg">
                Get Started Free
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="border-white text-white hover:bg-white hover:text-lavender-600">
              Schedule Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-grey-800 text-white py-12">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-lavender-500 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">S</span>
                </div>
                <span className="text-xl font-bold">Sahayak</span>
              </div>
              <p className="text-grey-400">
                Empowering teachers with AI to create culturally responsive education for all.
              </p>
            </div>
            <div>
              <h5 className="font-semibold mb-4">Product</h5>
              <ul className="space-y-2 text-grey-400">
                <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Demo</a></li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold mb-4">Support</h5>
              <ul className="space-y-2 text-grey-400">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Training</a></li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold mb-4">Company</h5>
              <ul className="space-y-2 text-grey-400">
                <li><a href="#" className="hover:text-white transition-colors">About</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-grey-700 mt-8 pt-8 text-center text-grey-400">
            <p>&copy; 2024 Sahayak. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;