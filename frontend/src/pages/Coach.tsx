import React, { useState } from 'react';
import { MessageCircle, Send, ThumbsUp, ThumbsDown, Lightbulb, AlertCircle } from 'lucide-react';
import Header from '../components/Layout/Header';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import { useApp } from '../contexts/AppContext';
import { coachAPI, CoachingResponse } from '../services/api';

const Coach: React.FC = () => {
  const { loading, setLoading, user } = useApp();
  const [query, setQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [conversation, setConversation] = useState<Array<{
    id: string;
    type: 'user' | 'ai';
    message: string;
    category?: string;
    timestamp: Date;
    model_used?: string;
  }>>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const categories = [
    { value: 'classroom-management', label: 'Classroom Management' },
    { value: 'student-engagement', label: 'Student Engagement' },
    { value: 'multi-grade', label: 'Multi-grade Teaching' },
    { value: 'cultural-integration', label: 'Cultural Integration' },
    { value: 'technology', label: 'Technology Integration' },
    { value: 'assessment', label: 'Assessment & Evaluation' },
    { value: 'parent-communication', label: 'Parent Communication' },
    { value: 'special-needs', label: 'Special Needs Students' }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      type: 'user' as const,
      message: query,
      category: selectedCategory,
      timestamp: new Date()
    };

    setConversation(prev => [...prev, userMessage]);
    setIsGenerating(true);
    setLoading(true);
    setError(null);

    try {
      // Call the real AI coaching API
      const response = await coachAPI.query({
        query: query,
        category: selectedCategory,
        user_id: user?.id || 'sahayak_user',
        session_id: `coaching_${Date.now()}`
      });

      if (response.success && response.data) {
        const aiMessage = {
          id: response.data.coaching.id,
          type: 'ai' as const,
          message: response.data.coaching.response,
          timestamp: new Date(response.data.coaching.createdAt),
          model_used: response.data.coaching.model_used
        };

        setConversation(prev => [...prev, aiMessage]);
        setQuery('');
        setSelectedCategory('');
        setError(null);
      } else {
        setError(response.error || 'Failed to get coaching response');
      }
    } catch (err) {
      setError('Network error. Please check if the AI backend is running.');
    } finally {
      setIsGenerating(false);
      setLoading(false);
    }
  };

  const generateMockResponse = (query: string, category: string) => {
    const responses = {
      'classroom-management': `Here are some effective strategies for classroom management in multi-grade settings:

1. **Establish Clear Routines**: Create consistent daily routines that all grade levels can follow. This helps students know what to expect and reduces disruptions.

2. **Use Visual Cues**: Implement hand signals, colored cards, or other visual indicators to manage transitions and behavior without disrupting ongoing instruction.

3. **Peer Mentoring**: Pair older students with younger ones. This helps with classroom management while providing leadership opportunities for older students.

4. **Flexible Seating**: Arrange desks in clusters or use moveable furniture to easily group students by grade level or ability when needed.

5. **Independent Work Stations**: Create learning centers with grade-appropriate activities that students can work on independently while you focus on direct instruction with other groups.

**Cultural Consideration**: Incorporate traditional values of respect and community cooperation that are common in many cultures to build a positive classroom environment.

Would you like specific strategies for any particular challenge you're facing?`,
      
      'student-engagement': `Here are proven strategies to boost student engagement in your classroom:

1. **Connect to Local Context**: Use examples from students' daily lives, local culture, and community experiences to make lessons relevant and engaging.

2. **Interactive Learning**: Incorporate games, role-plays, and hands-on activities that get students physically and mentally involved.

3. **Choice and Voice**: Give students options in how they learn or demonstrate their understanding. This increases ownership and engagement.

4. **Storytelling**: Use local stories, legends, and cultural narratives to teach concepts across subjects.

5. **Real-World Applications**: Show students how what they're learning applies to their lives and future goals.

6. **Celebrate Diversity**: Acknowledge and celebrate the different backgrounds, languages, and experiences students bring to class.

**Quick Engagement Boosters**:
- Start lessons with a question or challenge
- Use movement and music
- Incorporate technology when available
- Create opportunities for peer interaction

What specific subject or grade level are you looking to make more engaging?`,
      
      'multi-grade': `Teaching multiple grades simultaneously can be challenging but very rewarding. Here are key strategies:

1. **Thematic Units**: Plan lessons around themes that can be adapted for different grade levels. For example, "Our Community" can work for all grades with varying complexity.

2. **Differentiated Instruction**: 
   - Same content, different complexity levels
   - Visual aids for younger students, written instructions for older ones
   - Varied assessment methods

3. **Collaborative Learning**: 
   - Mixed-age group projects
   - Peer tutoring opportunities
   - Older students helping younger ones

4. **Station Rotation**: Set up learning stations with grade-appropriate activities while you provide direct instruction to one group at a time.

5. **Technology Integration**: Use devices or apps that can adapt to different skill levels automatically.

6. **Time Management**: 
   - Plan 20-30 minute focused sessions per grade
   - Use independent work time effectively
   - Overlap subjects when possible

**Sample Daily Structure**:
- Morning: Whole group activity (story, news, calendar)
- Mid-morning: Grade-specific math instruction
- Late morning: Integrated science/social studies
- Afternoon: Language arts with differentiated groups

Would you like help planning a specific multi-grade lesson or daily schedule?`,
      
      default: `Thank you for your question about "${query}". Here's some personalized advice:

Based on your query, I understand you're looking for practical strategies to improve your teaching effectiveness. Here are some key recommendations:

1. **Start Small**: Choose one or two strategies to implement rather than trying to change everything at once.

2. **Observe and Adapt**: Pay attention to how your students respond and adjust your approach accordingly.

3. **Build on Strengths**: Use what's already working well in your classroom as a foundation for new strategies.

4. **Community Integration**: Connect your teaching to local knowledge, traditions, and community resources.

5. **Reflect Regularly**: Take time to think about what worked well and what could be improved.

6. **Collaborate**: Share experiences with other teachers and learn from their successes and challenges.

**Remember**: Every classroom is unique, and what works for one teacher may need to be adapted for another. The key is to be patient with yourself and your students as you try new approaches.

Would you like me to elaborate on any of these points or help you with a specific aspect of your teaching practice?`
    };

    return responses[category as keyof typeof responses] || responses.default;
  };

  return (
    <div className="min-h-screen bg-grey-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-grey-800 mb-2">AI Teacher Coach</h1>
          <p className="text-grey-600">
            Get personalized advice and strategies for your teaching challenges. I'm here to help you become an even better educator.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Categories Sidebar */}
          <Card className="lg:col-span-1">
            <h2 className="text-lg font-semibold text-grey-800 mb-4">Choose a Category</h2>
            <div className="space-y-2">
              {categories.map((category) => (
                <button
                  key={category.value}
                  onClick={() => setSelectedCategory(category.value)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                    selectedCategory === category.value
                      ? 'bg-lavender-100 text-lavender-700 border-l-4 border-lavender-500'
                      : 'text-grey-600 hover:bg-grey-50'
                  }`}
                >
                  {category.label}
                </button>
              ))}
            </div>
          </Card>

          {/* Chat Area */}
          <Card className="lg:col-span-3">
            <div className="flex flex-col h-96">
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {conversation.length === 0 ? (
                  <div className="text-center py-8">
                    <MessageCircle className="w-12 h-12 text-grey-400 mx-auto mb-4" />
                    <p className="text-grey-500">Ask me anything about teaching! I'm here to help.</p>
                    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-lavender-50 p-4 rounded-lg">
                        <Lightbulb className="w-6 h-6 text-lavender-600 mb-2" />
                        <p className="text-sm text-lavender-700">
                          "How can I manage a multi-grade classroom effectively?"
                        </p>
                      </div>
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <Lightbulb className="w-6 h-6 text-blue-600 mb-2" />
                        <p className="text-sm text-blue-700">
                          "What are some engaging activities for mixed-age groups?"
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  conversation.map((message) => (
                    <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-3xl px-4 py-3 rounded-lg ${
                        message.type === 'user' 
                          ? 'bg-lavender-500 text-white' 
                          : 'bg-grey-100 text-grey-800'
                      }`}>
                        <pre className="whitespace-pre-wrap text-sm font-sans">{message.message}</pre>
                        {message.type === 'ai' && (
                          <div className="flex items-center space-x-2 mt-3 pt-3 border-t border-grey-200">
                            <span className="text-xs text-grey-600">Was this helpful?</span>
                            <Button variant="ghost" size="sm" className="text-green-600 hover:text-green-700">
                              <ThumbsUp className="w-3 h-3" />
                            </Button>
                            <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-700">
                              <ThumbsDown className="w-3 h-3" />
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                
                {isGenerating && (
                  <div className="flex justify-start">
                    <div className="bg-grey-100 px-4 py-3 rounded-lg">
                      <LoadingSpinner size="sm" />
                      <span className="ml-2 text-sm text-grey-600">Thinking...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Input Form */}
              <form onSubmit={handleSubmit} className="border-t p-4">
                <div className="flex space-x-4">
                  <div className="flex-1">
                    <textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Ask me about classroom management, student engagement, or any teaching challenge..."
                      rows={3}
                      className="w-full px-4 py-3 border border-grey-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lavender-500 focus:border-transparent resize-none"
                    />
                  </div>
                  <Button
                    type="submit"
                    variant="primary"
                    disabled={!query.trim() || isGenerating}
                    icon={Send}
                  >
                    Ask
                  </Button>
                </div>
                {selectedCategory && (
                  <p className="text-sm text-grey-600 mt-2">
                    Category: {categories.find(c => c.value === selectedCategory)?.label}
                  </p>
                )}
              </form>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Coach;