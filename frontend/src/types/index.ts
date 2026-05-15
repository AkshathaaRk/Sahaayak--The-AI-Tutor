export interface User {
  id: string;
  name: string;
  email: string;
  grade?: string;
  subjects?: string[];
  region?: string;
  avatar?: string;
}

export interface ContentItem {
  id: string;
  title: string;
  type: 'lesson' | 'story' | 'quiz' | 'summary';
  subject: string;
  grade: string;
  content: string;
  createdAt: Date;
  tags?: string[];
}

export interface GenerationRequest {
  topic: string;
  grade: string;
  subject: string;
  contentType: string;
  context?: string;
}

export interface FileUpload {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
}

export interface CoachingQuery {
  id: string;
  query: string;
  category?: string;
  response?: string;
  helpful?: boolean;
  createdAt: Date;
}

export interface PathwayIdea {
  id: string;
  title: string;
  description: string;
  activities: string[];
  culturalElements: string[];
  subject: string;
  grade: string;
}