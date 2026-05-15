import React, { useState } from 'react';
import { Compass, Globe, Users, Heart, Save, Share2 } from 'lucide-react';
import Header from '../components/Layout/Header';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import Input from '../components/UI/Input';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import { useApp } from '../contexts/AppContext';

const Pathways: React.FC = () => {
  const { loading, setLoading } = useApp();
  const [formData, setFormData] = useState({
    topic: '',
    grade: '',
    region: '',
    culturalFocus: ''
  });
  const [pathwayIdeas, setPathwayIdeas] = useState<Array<{
    id: string;
    title: string;
    description: string;
    activities: string[];
    culturalElements: string[];
    subject: string;
    grade: string;
  }>>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleGenerate = async () => {
    if (!formData.topic || !formData.grade) {
      alert('Please fill in the required fields');
      return;
    }

    setIsGenerating(true);
    setLoading(true);

    // Simulate AI generation
    await new Promise(resolve => setTimeout(resolve, 3000));

    const mockPathways = [
      {
        id: '1',
        title: `Cultural Mathematics: ${formData.topic} in Local Context`,
        description: `Explore ${formData.topic} through traditional practices and cultural artifacts from ${formData.region || 'your region'}. Students will discover how mathematical concepts are embedded in their cultural heritage.`,
        activities: [
          'Create geometric patterns using traditional rangoli designs',
          'Measure and calculate using local units of measurement',
          'Analyze mathematical patterns in traditional crafts',
          'Interview elders about traditional counting systems',
          'Design a cultural festival using mathematical planning'
        ],
        culturalElements: [
          'Traditional art forms and patterns',
          'Local festivals and celebrations',
          'Traditional games and sports',
          'Cultural stories and folklore',
          'Community practices and wisdom'
        ],
        subject: 'Mathematics',
        grade: formData.grade
      },
      {
        id: '2',
        title: `Community Science: ${formData.topic} Through Local Eyes`,
        description: `Connect scientific concepts to local environment and traditional knowledge. Students will learn how their community has been using science for generations.`,
        activities: [
          'Field trip to local natural areas',
          'Interview community experts and elders',
          'Document traditional ecological knowledge',
          'Create a community science fair',
          'Develop local solutions to environmental challenges'
        ],
        culturalElements: [
          'Traditional farming practices',
          'Local medicinal plants and herbs',
          'Weather prediction methods',
          'Traditional food preservation',
          'Community environmental practices'
        ],
        subject: 'Science',
        grade: formData.grade
      },
      {
        id: '3',
        title: `Heritage Stories: ${formData.topic} Narratives`,
        description: `Weave educational content into the rich tapestry of local stories, legends, and oral traditions. Students will learn while connecting to their cultural roots.`,
        activities: [
          'Collect and document local stories',
          'Create multimedia presentations',
          'Perform traditional storytelling',
          'Write modern adaptations of classic tales',
          'Interview community storytellers'
        ],
        culturalElements: [
          'Local legends and myths',
          'Historical events and heroes',
          'Traditional proverbs and sayings',
          'Cultural values and morals',
          'Community celebrations and rituals'
        ],
        subject: 'Language Arts',
        grade: formData.grade
      }
    ];

    setPathwayIdeas(mockPathways);
    setIsGenerating(false);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-grey-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-grey-800 mb-2">Culturally Responsive Learning Pathways</h1>
          <p className="text-grey-600">
            Discover how to integrate local culture, traditions, and community knowledge into your curriculum for meaningful, relevant learning experiences.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Form */}
          <Card>
            <h2 className="text-xl font-semibold text-grey-800 mb-6">Create Cultural Pathways</h2>
            
            <div className="space-y-6">
              <Input
                label="Topic/Theme"
                type="text"
                name="topic"
                placeholder="e.g., Water Conservation, Geometry, History"
                value={formData.topic}
                onChange={handleChange}
                required
              />

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
                  <option value="Primary (1-3)">Primary (1-3)</option>
                  <option value="Upper Primary (4-5)">Upper Primary (4-5)</option>
                  <option value="Middle (6-8)">Middle (6-8)</option>
                  <option value="Secondary (9-10)">Secondary (9-10)</option>
                  <option value="Multi-grade">Multi-grade</option>
                </select>
              </div>

              <Input
                label="Region/Community"
                type="text"
                name="region"
                placeholder="e.g., Karnataka, Rajasthan, Your village name"
                value={formData.region}
                onChange={handleChange}
              />

              <div className="space-y-2">
                <label className="block text-sm font-medium text-grey-700">
                  Cultural Focus (Optional)
                </label>
                <textarea
                  name="culturalFocus"
                  value={formData.culturalFocus}
                  onChange={handleChange}
                  placeholder="Specific cultural elements you'd like to include: festivals, traditions, crafts, stories, etc."
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
                icon={Compass}
              >
                Generate Pathway Ideas
              </Button>
            </div>
          </Card>

          {/* Generated Pathways */}
          <div className="lg:col-span-2">
            <h2 className="text-xl font-semibold text-grey-800 mb-6">Cultural Learning Pathways</h2>
            
            {isGenerating ? (
              <Card>
                <div className="flex flex-col items-center justify-center py-12">
                  <LoadingSpinner size="lg" />
                  <p className="text-grey-600 mt-4">Creating culturally responsive pathways...</p>
                </div>
              </Card>
            ) : pathwayIdeas.length > 0 ? (
              <div className="space-y-6">
                {pathwayIdeas.map((pathway) => (
                  <Card key={pathway.id} className="hover:shadow-lg transition-shadow">
                    <div className="space-y-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-grey-800">{pathway.title}</h3>
                          <p className="text-sm text-grey-600">{pathway.subject} • {pathway.grade}</p>
                        </div>
                        <div className="flex space-x-2">
                          <Button variant="ghost" size="sm" icon={Save}>
                            Save
                          </Button>
                          <Button variant="ghost" size="sm" icon={Share2}>
                            Share
                          </Button>
                        </div>
                      </div>
                      
                      <p className="text-grey-700">{pathway.description}</p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold text-grey-800 mb-2 flex items-center">
                            <Users className="w-4 h-4 mr-2" />
                            Learning Activities
                          </h4>
                          <ul className="space-y-1 text-sm text-grey-600">
                            {pathway.activities.map((activity, index) => (
                              <li key={index} className="flex items-start">
                                <span className="w-2 h-2 bg-lavender-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {activity}
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-grey-800 mb-2 flex items-center">
                            <Heart className="w-4 h-4 mr-2" />
                            Cultural Elements
                          </h4>
                          <ul className="space-y-1 text-sm text-grey-600">
                            {pathway.culturalElements.map((element, index) => (
                              <li key={index} className="flex items-start">
                                <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {element}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <div className="flex flex-col items-center justify-center py-12 text-grey-500">
                  <Globe className="w-12 h-12 mb-4" />
                  <p className="text-center">Enter a topic and grade level to generate culturally responsive learning pathways!</p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Pathways;