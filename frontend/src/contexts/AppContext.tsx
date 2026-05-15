import React, { createContext, useContext, useState, ReactNode } from 'react';
import { User, ContentItem, CoachingQuery, PathwayIdea } from '../types';

interface AppContextType {
  user: User | null;
  setUser: (user: User | null) => void;
  isOnline: boolean;
  setIsOnline: (online: boolean) => void;
  recentContent: ContentItem[];
  setRecentContent: (content: ContentItem[]) => void;
  coachingHistory: CoachingQuery[];
  setCoachingHistory: (history: CoachingQuery[]) => void;
  pathwayIdeas: PathwayIdea[];
  setPathwayIdeas: (ideas: PathwayIdea[]) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  // Temporary default user for testing - remove this in production
  const [user, setUser] = useState<User | null>({
    id: 'test_user_123',
    name: 'Test Teacher',
    email: 'teacher@sahayak.com',
    grade: 'Multi-grade',
    subjects: ['Mathematics', 'Science', 'English'],
    region: 'India'
  });
  const [isOnline, setIsOnline] = useState(true);
  const [recentContent, setRecentContent] = useState<ContentItem[]>([]);
  const [coachingHistory, setCoachingHistory] = useState<CoachingQuery[]>([]);
  const [pathwayIdeas, setPathwayIdeas] = useState<PathwayIdea[]>([]);
  const [loading, setLoading] = useState(false);

  return (
    <AppContext.Provider
      value={{
        user,
        setUser,
        isOnline,
        setIsOnline,
        recentContent,
        setRecentContent,
        coachingHistory,
        setCoachingHistory,
        pathwayIdeas,
        setPathwayIdeas,
        loading,
        setLoading,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};