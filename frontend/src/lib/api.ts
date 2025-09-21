import axios from 'axios'
import { backendAPI, type APIResponse } from './backend-config'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// API endpoints
export const endpoints = {
  // Health check
  health: '/api/health',
  
  // Students
  students: '/api/students',
  student: (id: string) => `/api/students/${id}`,
  
  // Exams
  exams: '/api/exams',
  exam: (id: string) => `/api/exams/${id}`,
  examSummary: (id: string) => `/api/statistics/${id}`,
  
  // OMR Processing
  uploadOMR: '/api/upload',
  uploadOMREnhanced: '/api/upload/enhanced',
  omrStatus: (id: string) => `/api/upload/${id}`,
  processingDetails: (id: string) => `/api/processing/details/${id}`,
  detectionPreview: (id: string) => `/api/detection/preview/${id}`,
  
  // Results
  studentResults: (id: string) => `/api/results/${id}`,
  examResults: (id: string) => `/api/results`,
  exportResults: (id: string) => `/api/export/${id}`,
  
  // Statistics
  overview: '/api/statistics/overview',
  recentActivity: '/api/recent-activity',
  
  // Flagged sheets
  flaggedSheets: '/api/flagged-sheets',
  reviewFlagged: (id: string) => `/api/flagged-sheets/${id}/review`,
}

// Types
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
  answer_key: Record<string, string[]>
  created_at: string
  created_by?: string
}

export interface OMRSheet {
  id: number
  student_id: number
  exam_id: number
  sheet_version: string
  image_path: string
  processed_image_path?: string
  upload_timestamp: string
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'
  processing_timestamp?: string
  error_message?: string
  confidence_score?: number
  needs_review: boolean
}

export interface Result {
  id: number
  omr_sheet_id: number
  student_id: number
  exam_id: number
  subject_scores: Record<string, number>
  total_score: number
  percentage: number
  answers: Record<string, any>
  processing_metadata?: Record<string, any>
  created_at: string
}

export interface SystemStats {
  total_students: number
  total_exams: number
  total_sheets: number
  processed_sheets: number
  pending_reviews: number
  processing_rate: number
}

export interface FlaggedSheet {
  id: number
  omr_sheet_id: number
  flag_type: string
  flag_reason: string
  confidence_threshold?: number
  reviewer_notes?: string
  status: 'pending' | 'approved' | 'rejected'
  reviewed_by?: string
  reviewed_at?: string
  created_at: string
  student_id?: string
  name?: string
  exam_name?: string
  image_path?: string
}

// API functions - Updated to use backend API
export const apiClient = {
  // Health check
  async checkHealth() {
    return await backendAPI.checkHealth()
  },

  // Students
  async getStudents(): Promise<Student[]> {
    const response = await backendAPI.getStudents()
    return response.data || []
  },

  async createStudent(student: Omit<Student, 'id' | 'created_at' | 'updated_at'>) {
    return await backendAPI.createStudent(student)
  },

  // Exams
  async getExams(): Promise<Exam[]> {
    const response = await backendAPI.getExams()
    return response.data || []
  },

  async createExam(exam: Omit<Exam, 'id' | 'created_at'>) {
    return await backendAPI.createExam(exam)
  },

  async getExamSummary(examId: string) {
    return await backendAPI.getStatistics(examId)
  },

  // OMR Processing
  async uploadOMRSheet(formData: FormData) {
    return await backendAPI.uploadOMR(formData)
  },

  async uploadOMRSheetEnhanced(formData: FormData) {
    const response = await api.post(endpoints.uploadOMREnhanced, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async getOMRStatus(sheetId: string) {
    const response = await api.get(endpoints.omrStatus(sheetId))
    return response.data
  },

  async getProcessingDetails(sheetId: string) {
    const response = await api.get(endpoints.processingDetails(sheetId))
    return response.data
  },

  async getDetectionPreview(sheetId: string) {
    const response = await api.get(endpoints.detectionPreview(sheetId))
    return response.data
  },

  // Results
  async getStudentResults(studentId: string, examId?: string) {
    return await backendAPI.getStudentResults(studentId, examId)
  },

  async getExamResults(examId: string): Promise<Result[]> {
    const response = await backendAPI.getExamResults(examId)
    return response.data || []
  },

  async exportResults(examId: string, format: 'csv' | 'excel' = 'csv') {
    const response = await api.get(`/api/export/${examId}?format=${format}`, {
      responseType: 'blob',
    })
    return response.data
  },

  // Statistics
  async getOverviewStats(): Promise<SystemStats> {
    const response = await backendAPI.getStatistics('1') // Use exam ID 1 for now
    return response.data || {
      total_students: 0,
      total_exams: 0,
      total_sheets: 0,
      processed_sheets: 0,
      pending_reviews: 0,
      processing_rate: 0
    }
  },

  // AR Integration
  async processARCapture(imageData: string, metadata?: any) {
    return await backendAPI.processARCapture({
      image_data: imageData,
      metadata: {
        timestamp: new Date().toISOString(),
        ...metadata
      }
    })
  },

  // Flagged sheets
  async getFlaggedSheets(status?: string): Promise<FlaggedSheet[]> {
    const url = status ? `/api/flagged-sheets?status=${status}` : '/api/flagged-sheets'
    const response = await api.get(url)
    return response.data || []
  },

  async reviewFlaggedSheet(flagId: string, status: string, reviewerNotes?: string, reviewedBy?: string) {
    const response = await api.post(`/api/flagged-sheets/${flagId}/review`, {
      status,
      reviewer_notes: reviewerNotes,
      reviewed_by: reviewedBy,
    })
    return response.data
  },
}
