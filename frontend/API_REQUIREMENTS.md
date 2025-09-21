# API Requirements for OMR Evaluation System

This document outlines the API endpoints that your backend needs to implement to work with the Next.js frontend.

## Base URL
```
http://localhost:5000
```

## Required Endpoints

### 1. Health Check
```
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-21T13:30:00Z",
  "model_loaded": true
}
```

### 2. Student Management

#### Get All Students
```
GET /api/students
```
**Response:**
```json
[
  {
    "id": 1,
    "student_id": "STU001",
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-09-21T13:30:00Z"
  }
]
```

#### Create Student
```
POST /api/students
```
**Request Body:**
```json
{
  "student_id": "STU002",
  "name": "Jane Smith",
  "email": "jane@example.com"
}
```

### 3. Exam Management

#### Get All Exams
```
GET /api/exams
```
**Response:**
```json
[
  {
    "id": 1,
    "exam_name": "Midterm Exam",
    "exam_date": "2025-09-21",
    "total_questions": 100,
    "subjects": ["Math", "Physics", "Chemistry"],
    "answer_key": {
      "Q1": "A",
      "Q2": "B",
      "Q3": "C"
    },
    "created_at": "2025-09-21T13:30:00Z"
  }
]
```

#### Create Exam
```
POST /api/exams
```
**Request Body:**
```json
{
  "exam_name": "Final Exam",
  "exam_date": "2025-12-15",
  "total_questions": 100,
  "subjects": ["Math", "Physics", "Chemistry", "Biology", "English"],
  "answer_key": {
    "Q1": "A",
    "Q2": "B",
    "Q3": "C"
  }
}
```

### 4. OMR Processing

#### Upload OMR Sheet
```
POST /api/upload
```
**Request:** `multipart/form-data`
- `file`: Image file (JPEG, PNG)
- `student_id`: Student ID
- `exam_id`: Exam ID
- `sheet_version`: Sheet version (A, B, C, D)

**Response:**
```json
{
  "success": true,
  "omr_sheet_id": 123,
  "processing_status": "completed",
  "result": {
    "total_score": 85,
    "percentage": 85.0,
    "subject_scores": {
      "Math": 18,
      "Physics": 17,
      "Chemistry": 16,
      "Biology": 17,
      "English": 17
    },
    "answers": {
      "Q1": "A",
      "Q2": "B",
      "Q3": "C"
    }
  }
}
```

#### Batch Upload
```
POST /api/batch/upload
```
**Request:** `multipart/form-data`
- `files[]`: Multiple image files
- `exam_id`: Exam ID
- `sheet_version`: Sheet version

### 5. AR Integration

#### Process AR Capture
```
POST /api/ar/process
```
**Request Body:**
```json
{
  "image_data": "base64_encoded_image",
  "metadata": {
    "timestamp": "2025-09-21T13:30:00Z",
    "device_info": "iPhone 15",
    "location": {
      "lat": 40.7128,
      "lng": -74.0060
    }
  }
}
```

### 6. Results and Analytics

#### Get Student Results
```
GET /api/results/{student_id}?exam_id={exam_id}
```

#### Get Exam Results
```
GET /api/results/exam/{exam_id}
```

#### Get Statistics
```
GET /api/statistics/{exam_id}
```
**Response:**
```json
{
  "total_students": 150,
  "average_score": 78.5,
  "score_distribution": [10, 25, 45, 35, 20],
  "highest_score": 95,
  "lowest_score": 45
}
```

#### Export Results
```
GET /api/export/{exam_id}?format=csv
```

## Error Handling

All endpoints should return consistent error responses:

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

## Common Error Codes
- `VALIDATION_ERROR`: Invalid input data
- `FILE_TOO_LARGE`: Uploaded file exceeds size limit
- `INVALID_FORMAT`: Unsupported file format
- `PROCESSING_FAILED`: OMR processing failed
- `STUDENT_NOT_FOUND`: Student ID not found
- `EXAM_NOT_FOUND`: Exam ID not found

## File Upload Limits
- Maximum file size: 10MB
- Supported formats: JPEG, JPG, PNG
- Maximum batch size: 50 files

## Authentication (Optional)
If you want to add authentication, use JWT tokens:

```
Authorization: Bearer <jwt_token>
```

## CORS Configuration
Make sure your backend allows CORS from the frontend:
```
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
```

## Testing
You can test the API endpoints using:
- Postman
- curl commands
- The frontend application

## Frontend Integration
The frontend is configured to work with these endpoints. Make sure your backend implements all the required endpoints for full functionality.

## AR Integration Notes
For AR functionality, the frontend will:
1. Capture images using device camera
2. Send base64 encoded images to `/api/ar/process`
3. Handle the processing response
4. Display results in the UI

Your backend should be able to:
1. Accept base64 image data
2. Process OMR sheets from AR captures
3. Return the same response format as regular uploads
