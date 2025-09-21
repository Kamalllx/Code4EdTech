'use client'

import { useState, useEffect, useRef } from 'react'
import { Upload, FileText, User, Calendar, AlertCircle, CheckCircle, Clock, Camera } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient, type Student, type Exam } from '@/lib/api'
import ARIntegration from '@/components/ar-integration'

export default function UploadPage() {
  const [students, setStudents] = useState<Student[]>([])
  const [exams, setExams] = useState<Exam[]>([])
  const [selectedStudent, setSelectedStudent] = useState<string>('')
  const [selectedExam, setSelectedExam] = useState<string>('')
  const [sheetVersion, setSheetVersion] = useState<string>('A')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [uploadResult, setUploadResult] = useState<any>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [studentsData, examsData] = await Promise.all([
        apiClient.getStudents(),
        apiClient.getExams()
      ])
      setStudents(studentsData)
      setExams(examsData)
    } catch (err) {
      setError('Failed to load data')
      console.error('Error fetching data:', err)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      // Validate file type
      const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png']
      if (!allowedTypes.includes(file.type)) {
        setError('Please select a valid image file (JPEG, JPG, or PNG)')
        return
      }

      // Validate file size (10MB max)
      const maxSize = 10 * 1024 * 1024 // 10MB
      if (file.size > maxSize) {
        setError('File size must be less than 10MB')
        return
      }

      setSelectedFile(file)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile || !selectedStudent || !selectedExam) {
      setError('Please fill in all required fields')
      return
    }

    setUploading(true)
    setUploadStatus('uploading')
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('student_id', selectedStudent)
      formData.append('exam_id', selectedExam)
      formData.append('sheet_version', sheetVersion)

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 200)

      const result = await apiClient.uploadOMRSheet(formData)
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      setUploadStatus('processing')
      setUploadResult(result)

      // Simulate processing time
      setTimeout(() => {
        setUploadStatus('completed')
        setUploading(false)
      }, 3000)

    } catch (err) {
      setError('Upload failed. Please try again.')
      setUploadStatus('error')
      setUploading(false)
      console.error('Upload error:', err)
    }
  }

  const resetForm = () => {
    setSelectedFile(null)
    setSelectedStudent('')
    setSelectedExam('')
    setSheetVersion('A')
    setUploadProgress(0)
    setUploadStatus('idle')
    setError(null)
    setUploadResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Upload OMR Sheet</h1>
              <p className="text-gray-600 mt-1">Upload and process OMR sheets for evaluation</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="manual" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="manual" className="flex items-center space-x-2">
              <Upload className="h-4 w-4" />
              <span>Manual Upload</span>
            </TabsTrigger>
            <TabsTrigger value="ar" className="flex items-center space-x-2">
              <Camera className="h-4 w-4" />
              <span>AR Camera</span>
            </TabsTrigger>
            <TabsTrigger value="batch" className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span>Batch Upload</span>
            </TabsTrigger>
          </TabsList>

          {/* Manual Upload Tab */}
          <TabsContent value="manual">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Form */}
          <Card>
            <CardHeader>
              <CardTitle>Upload OMR Sheet</CardTitle>
              <CardDescription>Select student, exam, and upload the OMR sheet image</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Student Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Student *
                </label>
                <select
                  value={selectedStudent}
                  onChange={(e) => setSelectedStudent(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={uploading}
                >
                  <option value="">Choose a student...</option>
                  {students.map((student) => (
                    <option key={student.id} value={student.id}>
                      {student.student_id} - {student.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Exam Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Exam *
                </label>
                <select
                  value={selectedExam}
                  onChange={(e) => setSelectedExam(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={uploading}
                >
                  <option value="">Choose an exam...</option>
                  {exams.map((exam) => (
                    <option key={exam.id} value={exam.id}>
                      {exam.exam_name} - {exam.exam_date}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sheet Version */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sheet Version *
                </label>
                <select
                  value={sheetVersion}
                  onChange={(e) => setSheetVersion(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={uploading}
                >
                  <option value="A">Version A</option>
                  <option value="B">Version B</option>
                  <option value="C">Version C</option>
                  <option value="D">Version D</option>
                </select>
              </div>

              {/* File Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload OMR Sheet Image *
                </label>
                <div
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    selectedFile
                      ? 'border-green-300 bg-green-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/jpg,image/png"
                    onChange={handleFileSelect}
                    className="hidden"
                    disabled={uploading}
                  />
                  {selectedFile ? (
                    <div>
                      <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                      <p className="text-sm font-medium text-green-700">{selectedFile.name}</p>
                      <p className="text-xs text-green-600">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  ) : (
                    <div>
                      <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm font-medium text-gray-700">Click to upload</p>
                      <p className="text-xs text-gray-500">JPEG, JPG, or PNG (max 10MB)</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="flex items-center space-x-2 text-red-600 text-sm">
                  <AlertCircle className="h-4 w-4" />
                  <span>{error}</span>
                </div>
              )}

              {/* Upload Button */}
              <Button
                onClick={handleUpload}
                disabled={!selectedFile || !selectedStudent || !selectedExam || uploading}
                className="w-full"
              >
                {uploading ? (
                  <>
                    <Clock className="h-4 w-4 mr-2 animate-spin" />
                    {uploadStatus === 'uploading' ? 'Uploading...' : 'Processing...'}
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload & Process
                  </>
                )}
              </Button>

              {/* Progress Bar */}
              {uploading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Upload Progress</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Upload Status */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Status</CardTitle>
              <CardDescription>Current upload and processing status</CardDescription>
            </CardHeader>
            <CardContent>
              {uploadStatus === 'idle' && (
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No upload in progress</p>
                </div>
              )}

              {uploadStatus === 'uploading' && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-100 rounded-full">
                      <Upload className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium">Uploading file...</p>
                      <p className="text-sm text-gray-500">{selectedFile?.name}</p>
                    </div>
                  </div>
                  <Progress value={uploadProgress} />
                </div>
              )}

              {uploadStatus === 'processing' && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-yellow-100 rounded-full">
                      <Clock className="h-5 w-5 text-yellow-600 animate-spin" />
                    </div>
                    <div>
                      <p className="font-medium">Processing OMR sheet...</p>
                      <p className="text-sm text-gray-500">Detecting bubbles and extracting answers</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Processing</span>
                      <span>In Progress</span>
                    </div>
                    <Progress value={100} />
                  </div>
                </div>
              )}

              {uploadStatus === 'completed' && uploadResult && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-green-100 rounded-full">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium text-green-700">Processing Complete!</p>
                      <p className="text-sm text-gray-500">Sheet ID: {uploadResult.sheet_id}</p>
                    </div>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <p className="text-sm text-green-700">
                      Your OMR sheet has been successfully processed. 
                      You can view the results in the Results section.
                    </p>
                  </div>

                  <Button onClick={resetForm} variant="outline" className="w-full">
                    Upload Another Sheet
                  </Button>
                </div>
              )}

              {uploadStatus === 'error' && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-red-100 rounded-full">
                      <AlertCircle className="h-5 w-5 text-red-600" />
                    </div>
                    <div>
                      <p className="font-medium text-red-700">Upload Failed</p>
                      <p className="text-sm text-gray-500">Please try again</p>
                    </div>
                  </div>
                  
                  <div className="bg-red-50 p-4 rounded-lg">
                    <p className="text-sm text-red-700">
                      There was an error processing your OMR sheet. 
                      Please check the image quality and try again.
                    </p>
                  </div>

                  <Button onClick={resetForm} variant="outline" className="w-full">
                    Try Again
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
            </div>
          </TabsContent>

          {/* AR Camera Tab */}
          <TabsContent value="ar">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>AR Camera Upload</CardTitle>
                  <CardDescription>
                    Use your device's camera to automatically detect and capture OMR sheets
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ARIntegration 
                    onImageCapture={(imageData) => {
                      console.log('AR Image captured:', imageData)
                    }}
                    onProcessingComplete={(result) => {
                      console.log('AR Processing complete:', result)
                      setUploadResult(result)
                      setUploadStatus('completed')
                    }}
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Batch Upload Tab */}
          <TabsContent value="batch">
            <Card>
              <CardHeader>
                <CardTitle>Batch Upload</CardTitle>
                <CardDescription>Upload multiple OMR sheets at once</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Batch upload feature coming soon</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Instructions */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Upload Instructions</CardTitle>
            <CardDescription>Guidelines for uploading OMR sheets</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Image Requirements</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Use clear, well-lit images</li>
                  <li>• Ensure the entire OMR sheet is visible</li>
                  <li>• Avoid shadows and glare</li>
                  <li>• Supported formats: JPEG, JPG, PNG</li>
                  <li>• Maximum file size: 10MB</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Processing Tips</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Make sure bubbles are clearly marked</li>
                  <li>• Use dark pencil or pen for marking</li>
                  <li>• Avoid smudges or erasures</li>
                  <li>• Keep the sheet flat and straight</li>
                  <li>• Ensure good contrast between marks and paper</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
