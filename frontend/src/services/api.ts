// API service for connecting Sahayak frontend to AI backend
const API_BASE_URL = 'http://127.0.0.1:5000';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  grade?: string;
  subjects?: string[];
  region?: string;
}

export interface ContentItem {
  id: string;
  title: string;
  type: 'lesson' | 'story' | 'quiz' | 'summary';
  subject: string;
  grade: string;
  content: string;
  createdAt: number;
  tags?: string[];
  model_used?: string;
}

export interface CoachingResponse {
  id: string;
  query: string;
  category?: string;
  response: string;
  helpful?: boolean;
  createdAt: number;
  model_used?: string;
}

export interface PathwayIdea {
  id: string;
  title: string;
  description: string;
  activities: string[];
  culturalElements: string[];
  subject: string;
  grade: string;
  fullContent?: string;
  model_used?: string;
}

export interface FileAnalysis {
  id: string;
  filename: string;
  type: 'pdf' | 'image' | 'text';
  summary?: string;
  analysis?: string;
  extractionMethod?: string;
  textLength?: number;
  model_used?: string;
}

// Helper function to make API calls
async function apiCall<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  try {
    console.log(`🔗 API Call: ${API_BASE_URL}${endpoint}`);

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    console.log(`📡 Response Status: ${response.status} ${response.statusText}`);

    const data = await response.json();
    console.log('📦 Response Data:', data);

    if (response.ok) {
      return { success: true, data };
    } else {
      console.error('❌ API Error:', data);
      return { success: false, error: data.error || 'Unknown error' };
    }
  } catch (error) {
    console.error('💥 Network Error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error'
    };
  }
}

// Authentication APIs
export const authAPI = {
  login: async (email: string, password: string): Promise<ApiResponse<{ user: User; token: string }>> => {
    return apiCall('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  register: async (userData: {
    name: string;
    email: string;
    password: string;
    grade?: string;
    subjects?: string[];
    region?: string;
  }): Promise<ApiResponse<{ user: User; token: string }>> => {
    return apiCall('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },
};

// Content Generation API
export const contentAPI = {
  generate: async (params: {
    topic: string;
    grade: string;
    subject: string;
    contentType: string;
    context?: string;
    user_id?: string;
    session_id?: string;
  }): Promise<ApiResponse<{ content: ContentItem }>> => {
    return apiCall('/api/generate/content', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },
};

// Coaching API
export const coachAPI = {
  query: async (params: {
    query: string;
    category?: string;
    user_id?: string;
    session_id?: string;
  }): Promise<ApiResponse<{ coaching: CoachingResponse }>> => {
    return apiCall('/api/coach/query', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },
};

// Pathways API
export const pathwaysAPI = {
  generate: async (params: {
    subject: string;
    grade: string;
    interests?: string[];
    user_id?: string;
    session_id?: string;
  }): Promise<ApiResponse<{ pathway: PathwayIdea }>> => {
    return apiCall('/api/pathways/generate', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },
};

// File Upload API
export const uploadAPI = {
  analyze: async (
    file: File,
    params: {
      analysis_type?: string;
      question?: string;
      language?: string;
      user_id?: string;
      session_id?: string;
    } = {}
  ): Promise<ApiResponse<{ analysis: FileAnalysis }>> => {
    const formData = new FormData();
    formData.append('file', file);
    
    Object.entries(params).forEach(([key, value]) => {
      if (value) formData.append(key, value);
    });

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true, data };
      } else {
        return { success: false, error: data.error || 'Upload failed' };
      }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Upload error' 
      };
    }
  },
};

// Health check API
export const healthAPI = {
  check: async (): Promise<ApiResponse<any>> => {
    return apiCall('/health-check');
  },
};

export default {
  auth: authAPI,
  content: contentAPI,
  coach: coachAPI,
  pathways: pathwaysAPI,
  upload: uploadAPI,
  health: healthAPI,
};
