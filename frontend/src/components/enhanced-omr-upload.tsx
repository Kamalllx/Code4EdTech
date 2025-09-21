'use client'

import React, { useState } from 'react'
import { apiClient } from '@/lib/api'

interface OMRResult {
  total_questions: number
  total_score: number
  percentage: number
  model_confidence: number
  question_numbers_detected: number
  bubbles_detected: number
  questions_mapped: number
  subject_scores?: Record<string, number>
  student_answers?: Record<string, string>
  database_save?: {
    omr_sheet_id: number
  }
}

interface DetectionPreview {
  question_numbers?: number[]
  bubbles?: Array<{ x: number; y: number; width: number; height: number }>
  question_mapping?: Record<string, string>
}

interface EnhancedOMRUploadProps {
  onUploadSuccess?: (result: OMRResult) => void
  onUploadError?: (error: string) => void
}

export default function EnhancedOMRUpload({ 
  onUploadSuccess, 
  onUploadError 
}: EnhancedOMRUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [studentId, setStudentId] = useState('')
  const [examId, setExamId] = useState('')
  const [sheetVersion, setSheetVersion] = useState('Set A')
  const [result, setResult] = useState<OMRResult | null>(null)
  const [detectionPreview, setDetectionPreview] = useState<DetectionPreview | null>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setResult(null)
      setDetectionPreview(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile || !studentId || !examId) {
      onUploadError?.('Please select a file and provide student ID and exam ID')
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('student_id', studentId)
      formData.append('exam_id', examId)
      formData.append('sheet_version', sheetVersion)

      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      const response = await apiClient.uploadOMRSheetEnhanced(formData)
      
      clearInterval(progressInterval)
      setUploadProgress(100)

      if (response.success) {
        setResult(response.result)
        onUploadSuccess?.(response.result)
        
        // Get detection preview
        if (response.result.database_save?.omr_sheet_id) {
          try {
            const preview = await apiClient.getDetectionPreview(
              response.result.database_save.omr_sheet_id.toString()
            )
            setDetectionPreview(preview)
          } catch (error) {
            console.warn('Could not fetch detection preview:', error)
          }
        }
      } else {
        throw new Error(response.error || 'Upload failed')
      }

    } catch (error: unknown) {
      console.error('Upload error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Upload failed'
      onUploadError?.(errorMessage)
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        Enhanced OMR Sheet Upload
      </h2>
      
      <div className="space-y-6">
        {/* File Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select OMR Sheet Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          {selectedFile && (
            <p className="mt-2 text-sm text-gray-600">
              Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        {/* Form Fields */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Student ID
            </label>
            <input
              type="text"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter student ID"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Exam ID
            </label>
            <input
              type="text"
              value={examId}
              onChange={(e) => setExamId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter exam ID"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sheet Version
            </label>
            <select
              value={sheetVersion}
              onChange={(e) => setSheetVersion(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="Set A">Set A</option>
              <option value="Set B">Set B</option>
              <option value="Set C">Set C</option>
            </select>
          </div>
        </div>

        {/* Upload Button */}
        <div className="flex justify-center">
          <button
            onClick={handleUpload}
            disabled={!selectedFile || !studentId || !examId || isUploading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isUploading ? 'Processing...' : 'Upload & Process'}
          </button>
        </div>

        {/* Progress Bar */}
        {isUploading && (
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-8 space-y-6">
            <h3 className="text-xl font-semibold text-gray-800">Processing Results</h3>
            
            {/* Summary */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-gray-800 mb-3">Summary</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Total Questions:</span>
                  <span className="ml-2 font-medium">{result.total_questions}</span>
                </div>
                <div>
                  <span className="text-gray-600">Correct Answers:</span>
                  <span className="ml-2 font-medium">{result.total_score}</span>
                </div>
                <div>
                  <span className="text-gray-600">Percentage:</span>
                  <span className="ml-2 font-medium">{result.percentage}%</span>
                </div>
                <div>
                  <span className="text-gray-600">Confidence:</span>
                  <span className="ml-2 font-medium">{(result.model_confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Detection Statistics */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-gray-800 mb-3">Detection Statistics</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Question Numbers:</span>
                  <span className="ml-2 font-medium">{result.question_numbers_detected}</span>
                </div>
                <div>
                  <span className="text-gray-600">Bubbles Detected:</span>
                  <span className="ml-2 font-medium">{result.bubbles_detected}</span>
                </div>
                <div>
                  <span className="text-gray-600">Questions Mapped:</span>
                  <span className="ml-2 font-medium">{result.questions_mapped}</span>
                </div>
              </div>
            </div>

            {/* Subject Scores */}
            {result.subject_scores && (
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-800 mb-3">Subject Scores</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  {Object.entries(result.subject_scores).map(([subject, score]) => (
                    <div key={subject}>
                      <span className="text-gray-600">{subject}:</span>
                      <span className="ml-2 font-medium">{String(score)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Sample Answers */}
            {result.student_answers && (
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-800 mb-3">Sample Student Answers</h4>
                <div className="grid grid-cols-5 gap-2 text-sm">
                  {Object.entries(result.student_answers).slice(0, 20).map(([question, answer]) => (
                    <div key={question} className="text-center">
                      <div className="font-medium">{question}</div>
                      <div className="text-gray-600">{String(answer)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Detection Preview */}
        {detectionPreview && (
          <div className="mt-8">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Detection Preview</h3>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-800 mb-2">Question Numbers Detected</h4>
                  <div className="text-sm text-gray-600">
                    {detectionPreview.question_numbers?.length || 0} question numbers found
                  </div>
                </div>
                <div>
                  <h4 className="font-medium text-gray-800 mb-2">Bubbles Detected</h4>
                  <div className="text-sm text-gray-600">
                    {detectionPreview.bubbles?.length || 0} bubbles found
                  </div>
                </div>
              </div>
              
              {detectionPreview.question_mapping && (
                <div className="mt-4">
                  <h4 className="font-medium text-gray-800 mb-2">Question Mapping</h4>
                  <div className="text-sm text-gray-600">
                    {Object.keys(detectionPreview.question_mapping).length} questions mapped to bubbles
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
