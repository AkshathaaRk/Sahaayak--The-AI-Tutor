import React, { useState } from 'react';
import { Sparkles, Save, Edit, Download, ThumbsUp, ThumbsDown, AlertCircle } from 'lucide-react';
import Header from '../components/Layout/Header';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import Input from '../components/UI/Input';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import { useApp } from '../contexts/AppContext';
import { contentAPI, ContentItem, healthAPI } from '../services/api';

const Generate: React.FC = () => {
  const { loading, setLoading, user } = useApp();
  const [formData, setFormData] = useState({
    topic: '',
    grade: '',
    subject: '',
    contentType: 'lesson',
    context: ''
  });
  const [generatedContent, setGeneratedContent] = useState<ContentItem | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Test function to check backend connection
  const testConnection = async () => {
    console.log('🧪 Testing backend connection...');
    try {
      const response = await healthAPI.check();
      if (response.success) {
        alert('✅ Backend connection successful!');
        console.log('✅ Health check passed:', response.data);
      } else {
        alert('❌ Backend health check failed: ' + response.error);
        console.error('❌ Health check failed:', response.error);
      }
    } catch (err) {
      alert('💥 Connection test failed: ' + err);
      console.error('💥 Connection test error:', err);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleGenerate = async () => {
    if (!formData.topic || !formData.grade || !formData.subject) {
      setError('Please fill in all required fields');
      return;
    }

    setIsGenerating(true);
    setLoading(true);
    setError(null);

    try {
      // Call the real AI backend API
      const response = await contentAPI.generate({
        topic: formData.topic,
        grade: formData.grade,
        subject: formData.subject,
        contentType: formData.contentType,
        context: formData.context,
        user_id: user?.id || 'sahayak_user',
        session_id: `content_${Date.now()}`
      });

      if (response.success && response.data) {
        setGeneratedContent(response.data.content);
        setError(null);
      } else {
        setError(response.error || 'Failed to generate content');
        setGeneratedContent(null);
      }
    } catch (err) {
      setError('Network error. Please check if the AI backend is running.');
      setGeneratedContent(null);
    } finally {
      setIsGenerating(false);
      setLoading(false);
    }
  };

  const contentTypes = [
    { value: 'lesson', label: 'Lesson Plan' },
    { value: 'story', label: 'Educational Story' },
    { value: 'quiz', label: 'Quiz/Assessment' },
    { value: 'activity', label: 'Learning Activity' },
    { value: 'summary', label: 'Topic Summary' }
  ];

  return (
    <div className="min-h-screen bg-grey-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-grey-800 mb-2">Generate New Content</h1>
          <p className="text-grey-600">
            Create engaging, culturally responsive educational content tailored to your students.
          </p>
          {/* Temporary test button */}
          <div className="mt-4">
            <Button variant="outline" size="sm" onClick={testConnection}>
              🧪 Test Backend Connection
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <Card>
            <h2 className="text-xl font-semibold text-grey-800 mb-6">Content Details</h2>
            
            <div className="space-y-6">
              <Input
                label="Topic"
                type="text"
                name="topic"
                placeholder="e.g., Fractions, Water Cycle, Local History"
                value={formData.topic}
                onChange={handleChange}
                required
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-grey-700">
                    Grade Level <span className="text-red-500">*</span>
                  </label>
                  <select
                    name="grade"
                    value={formData.grade}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
                  >
                    <option value="">Select grade</option>
                    <option value="Grade 1">Grade 1</option>
                    <option value="Grade 2">Grade 2</option>
                    <option value="Grade 3">Grade 3</option>
                    <option value="Grade 4">Grade 4</option>
                    <option value="Grade 5">Grade 5</option>
                    <option value="Grade 6">Grade 6</option>
                    <option value="Grade 7">Grade 7</option>
                    <option value="Grade 8">Grade 8</option>
                    <option value="Multi-grade">Multi-grade</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-grey-700">
                    Subject <span className="text-red-500">*</span>
                  </label>
                  <select
                    name="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
                  >
                    <option value="">Select subject</option>
                    <option value="Mathematics">Mathematics</option>
                    <option value="Science">Science</option>
                    <option value="English">English</option>
                    <option value="Social Studies">Social Studies</option>
                    <option value="Hindi">Hindi</option>
                    <option value="Art">Art</option>
                    <option value="Physical Education">Physical Education</option>
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-grey-700">Content Type</label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {contentTypes.map((type) => (
                    <label key={type.value} className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="contentType"
                        value={type.value}
                        checked={formData.contentType === type.value}
                        onChange={handleChange}
                        className="w-4 h-4 text-lavender-600 border-grey-300 focus:ring-lavender-500"
                      />
                      <span className="text-sm text-grey-700">{type.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-grey-700">
                  Local Context (Optional)
                </label>
                <textarea
                  name="context"
                  value={formData.context}
                  onChange={handleChange}
                  placeholder="Add local village names, cultural events, traditions, or regional examples..."
                  rows={3}
                  className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent"
                />
              </div>

              <Button
                variant="primary"
                size="lg"
                className="w-full"
                onClick={handleGenerate}
                loading={isGenerating}
                icon={Sparkles}
              >
                Generate Content
              </Button>
            </div>
          </Card>

          {/* Generated Content */}
          <Card>
            <h2 className="text-xl font-semibold text-grey-800 mb-6">Generated Content</h2>

            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-red-800 font-medium">Error</p>
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              </div>
            )}

            {isGenerating ? (
              <div className="flex flex-col items-center justify-center py-12">
                <LoadingSpinner size="lg" />
                <p className="text-grey-600 mt-4">🤖 AI is generating your content...</p>
                <p className="text-grey-500 text-sm mt-2">This may take 10-30 seconds</p>
              </div>
            ) : generatedContent ? (
              <div className="space-y-6">
                {/* Content Metadata */}
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-blue-900">{generatedContent.title}</h3>
                    {generatedContent.model_used && (
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                        🤖 {generatedContent.model_used}
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2 text-sm text-blue-700">
                    <span>📚 {generatedContent.subject}</span>
                    <span>🎓 {generatedContent.grade}</span>
                    <span>📝 {generatedContent.type}</span>
                    <span>⏰ {new Date(generatedContent.createdAt).toLocaleTimeString()}</span>
                  </div>
                </div>

                {/* Generated Content */}
                <div className="bg-grey-50 p-6 rounded-lg max-h-96 overflow-y-auto">
                  <pre className="whitespace-pre-wrap text-sm text-grey-700 font-sans leading-relaxed">
                    {generatedContent.content}
                  </pre>
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button variant="secondary" icon={Save}>
                    Save to Library
                  </Button>
                  <Button variant="outline" icon={Edit}>
                    Edit Content
                  </Button>
                  <Button variant="outline" icon={Download}>
                    Download PDF
                  </Button>
                </div>

                <div className="border-t pt-6">
                  <p className="text-sm text-grey-600 mb-3">Was this content helpful?</p>
                  <div className="flex items-center space-x-4">
                    <Button variant="ghost" size="sm" icon={ThumbsUp}>
                      Yes, very helpful!
                    </Button>
                    <Button variant="ghost" size="sm" icon={ThumbsDown}>
                      Needs improvement
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-grey-500">
                <Sparkles className="w-12 h-12 mb-4" />
                <p className="text-center">Fill in the details and click "Generate Content" to get started!</p>
                <p className="text-sm text-grey-400 mt-2">AI-powered educational content generation</p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Generate;