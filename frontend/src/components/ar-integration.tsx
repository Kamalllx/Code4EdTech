'use client'

import { useState, useEffect, useRef } from 'react'
import Image from 'next/image'
import { Camera, Upload, AlertCircle, CheckCircle } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

interface ARIntegrationProps {
  onImageCapture: (imageData: string) => void
  onProcessingComplete: (result: Record<string, unknown>) => void
}

export default function ARIntegration({ onImageCapture, onProcessingComplete }: ARIntegrationProps) {
  const [isARActive, setIsARActive] = useState(false)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [processing, setProcessing] = useState(false)
  const [arStatus, setArStatus] = useState<'idle' | 'detecting' | 'captured' | 'processing' | 'completed'>('idle')
  const [error, setError] = useState<string | null>(null)
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // AR Camera setup
  const startARCamera = async () => {
    try {
      setArStatus('detecting')
      setError(null)
      
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment', // Use back camera for AR
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
        setIsARActive(true)
        setArStatus('detecting')
      }
    } catch (err) {
      setError('Camera access denied or not available')
      console.error('Camera error:', err)
    }
  }

  // Capture image from AR camera
  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      const context = canvas.getContext('2d')
      
      if (context) {
        // Set canvas size to match video
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        
        // Draw current video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height)
        
        // Convert to base64 image
        const imageData = canvas.toDataURL('image/jpeg', 0.8)
        setCapturedImage(imageData)
        setArStatus('captured')
        onImageCapture(imageData)
      }
    }
  }

  // Process captured image
  const processImage = async () => {
    if (!capturedImage) return
    
    setProcessing(true)
    setArStatus('processing')
    
    try {
      // Convert base64 to blob
      const response = await fetch(capturedImage)
      const blob = await response.blob()
      
      // Create FormData for API
      const formData = new FormData()
      formData.append('file', blob, 'ar-captured-image.jpg')
      formData.append('source', 'ar_camera')
      formData.append('timestamp', new Date().toISOString())
      
      // Send to your friend's backend
      const apiResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })
      
      if (apiResponse.ok) {
        const result = await apiResponse.json()
        setArStatus('completed')
        onProcessingComplete(result)
      } else {
        throw new Error('Processing failed')
      }
    } catch (err) {
      setError('Failed to process image')
      console.error('Processing error:', err)
    } finally {
      setProcessing(false)
    }
  }

  // Stop AR camera
  const stopARCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
    }
    setIsARActive(false)
    setArStatus('idle')
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopARCamera()
    }
  }, [])

  return (
    <div className="space-y-6">
      {/* AR Camera View */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Camera className="h-5 w-5" />
            <span>AR Camera</span>
            <Badge variant={isARActive ? "default" : "secondary"}>
              {isARActive ? "Active" : "Inactive"}
            </Badge>
          </CardTitle>
          <CardDescription>
            Point your camera at the OMR sheet for automatic detection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Video Feed */}
            <div className="relative bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                className="w-full h-64 object-cover"
                playsInline
                muted
              />
              
              {/* AR Overlay */}
              {isARActive && (
                <div className="absolute inset-0 pointer-events-none">
                  {/* Detection overlay */}
                  <div className="absolute top-4 left-4 right-4 h-8 bg-blue-500/20 border-2 border-blue-500 rounded flex items-center justify-center">
                    <span className="text-white text-sm font-medium">
                      {arStatus === 'detecting' ? 'Detecting OMR sheet...' : 'OMR sheet detected'}
                    </span>
                  </div>
                  
                  {/* Corner markers for sheet detection */}
                  <div className="absolute top-8 left-8 w-4 h-4 border-2 border-green-400 rounded"></div>
                  <div className="absolute top-8 right-8 w-4 h-4 border-2 border-green-400 rounded"></div>
                  <div className="absolute bottom-8 left-8 w-4 h-4 border-2 border-green-400 rounded"></div>
                  <div className="absolute bottom-8 right-8 w-4 h-4 border-2 border-green-400 rounded"></div>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex space-x-4">
              {!isARActive ? (
                <Button onClick={startARCamera} className="flex-1">
                  <Camera className="h-4 w-4 mr-2" />
                  Start AR Camera
                </Button>
              ) : (
                <>
                  <Button onClick={captureImage} variant="outline" className="flex-1">
                    <Upload className="h-4 w-4 mr-2" />
                    Capture Image
                  </Button>
                  <Button onClick={stopARCamera} variant="outline">
                    Stop Camera
                  </Button>
                </>
              )}
            </div>

            {/* Error Display */}
            {error && (
              <div className="flex items-center space-x-2 text-red-600 text-sm">
                <AlertCircle className="h-4 w-4" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Captured Image Preview */}
      {capturedImage && (
        <Card>
          <CardHeader>
            <CardTitle>Captured Image</CardTitle>
            <CardDescription>Preview of the captured OMR sheet</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Image
                src={capturedImage}
                alt="Captured OMR sheet"
                width={400}
                height={256}
                className="w-full h-64 object-contain border rounded-lg"
              />
              
              <div className="flex space-x-4">
                <Button onClick={processImage} disabled={processing} className="flex-1">
                  {processing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Process Image
                    </>
                  )}
                </Button>
                <Button 
                  onClick={() => {
                    setCapturedImage(null)
                    setArStatus('detecting')
                  }} 
                  variant="outline"
                >
                  Retake
                </Button>
              </div>

              {/* Processing Progress */}
              {processing && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Processing OMR sheet...</span>
                    <span>Analyzing bubbles</span>
                  </div>
                  <Progress value={100} />
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
