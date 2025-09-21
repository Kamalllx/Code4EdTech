// Backend API Configuration
// This file contains the API endpoints that your friend's backend should implement

export const BACKEND_ENDPOINTS = {
  // Base URL - Update this to match your friend's backend
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
  
  // Health and Status
  HEALTH: '/api/health',
  
  // Student Management
  STUDENTS: {
    LIST: '/api/students',
    CREATE: '/api/students',
    GET: (id: string) => `/api/students/${id}`,
    UPDATE: (id: string) => `/api/students/${id}`,
    DELETE: (id: string) => `/api/students/${id}`,
  },
  
  // Exam Management
  EXAMS: {
    LIST: '/api/exams',
    CREATE: '/api/exams',
    GET: (id: string) => `/api/exams/${id}`,
    UPDATE: (id: string) => `/api/exams/${id}`,
    DELETE: (id: string) => `/api/exams/${id}`,
  },
  
  // OMR Processing
  OMR: {
    UPLOAD: '/api/upload',
    BATCH_UPLOAD: '/api/batch/upload',
    STATUS: (id: string) => `/api/upload/${id}`,
    PROCESS: (id: string) => `/api/process/${id}`,
  },
  
  // Results and Analytics
  RESULTS: {
    STUDENT: (id: string) => `/api/results/${id}`,
    EXAM: (id: string) => `/api/results/exam/${id}`,
    EXPORT: (id: string) => `/api/export/${id}`,
    STATISTICS: (id: string) => `/api/statistics/${id}`,
  },
  
  // AR Integration (for your friend's AR system)
  AR: {
    CAPTURE: '/api/ar/capture',
    PROCESS: '/api/ar/process',
    STATUS: '/api/ar/status',
  }
}

// Expected API Response Types
export interface APIResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface Student {
  id: number
  student_id: string
  name: string
  email?: string
  phone?: string
  created_at: string
  updated_at: string
}

export interface Exam {
  id: number
  exam_name: string
  exam_date: string
  total_questions: number
  subjects: string[]
  answer_key: Record<string, string>
  created_at: string
}

export interface OMRUploadResponse {
  success: boolean
  omr_sheet_id: number
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'
  result?: {
    total_score: number
    percentage: number
    subject_scores: Record<string, number>
    answers: Record<string, string>
  }
  error?: string
}

export interface ARCaptureRequest {
  image_data: string // base64 encoded image
  metadata: {
    timestamp: string
    device_info?: string
    location?: {
      lat: number
      lng: number
    }
  }
}

// API Helper Functions
export class BackendAPI {
  private baseURL: string

  constructor(baseURL?: string) {
    this.baseURL = baseURL || BACKEND_ENDPOINTS.BASE_URL
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`)
      }

      return data
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    }
  }

  // Health check
  async checkHealth(): Promise<APIResponse> {
    return this.request(BACKEND_ENDPOINTS.HEALTH)
  }

  // Students
  async getStudents(): Promise<APIResponse<Student[]>> {
    return this.request(BACKEND_ENDPOINTS.STUDENTS.LIST)
  }

  async createStudent(student: Omit<Student, 'id' | 'created_at' | 'updated_at'>): Promise<APIResponse<Student>> {
    return this.request(BACKEND_ENDPOINTS.STUDENTS.CREATE, {
      method: 'POST',
      body: JSON.stringify(student)
    })
  }

  // Exams
  async getExams(): Promise<APIResponse<Exam[]>> {
    return this.request(BACKEND_ENDPOINTS.EXAMS.LIST)
  }

  async createExam(exam: Omit<Exam, 'id' | 'created_at'>): Promise<APIResponse<Exam>> {
    return this.request(BACKEND_ENDPOINTS.EXAMS.CREATE, {
      method: 'POST',
      body: JSON.stringify(exam)
    })
  }

  // OMR Upload
  async uploadOMR(formData: FormData): Promise<APIResponse<OMRUploadResponse>> {
    return this.request(BACKEND_ENDPOINTS.OMR.UPLOAD, {
      method: 'POST',
      body: formData,
      headers: {} // Let browser set Content-Type for FormData
    })
  }

  // AR Integration
  async processARCapture(captureData: ARCaptureRequest): Promise<APIResponse<OMRUploadResponse>> {
    return this.request(BACKEND_ENDPOINTS.AR.PROCESS, {
      method: 'POST',
      body: JSON.stringify(captureData)
    })
  }

  // Results
  async getStudentResults(studentId: string, examId?: string): Promise<APIResponse> {
    const endpoint = examId 
      ? `${BACKEND_ENDPOINTS.RESULTS.STUDENT(studentId)}?exam_id=${examId}`
      : BACKEND_ENDPOINTS.RESULTS.STUDENT(studentId)
    return this.request(endpoint)
  }

  async getExamResults(examId: string): Promise<APIResponse> {
    return this.request(BACKEND_ENDPOINTS.RESULTS.EXAM(examId))
  }

  // Statistics
  async getStatistics(examId: string): Promise<APIResponse> {
    return this.request(BACKEND_ENDPOINTS.RESULTS.STATISTICS(examId))
  }
}

// Export a default instance
export const backendAPI = new BackendAPI()
