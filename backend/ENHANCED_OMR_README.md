# Enhanced OMR Processing System

## Overview

The Enhanced OMR Processing System is a comprehensive solution for detecting question numbers and bubbles, mapping bubbles to questions, saving results to a database, and sending results to the frontend. This system provides advanced computer vision capabilities for automated OMR sheet evaluation.

## Features

### 🔍 Question Number Detection
- **OCR Integration**: Uses Tesseract OCR to detect question numbers in OMR sheets
- **Intelligent Filtering**: Filters detected text to focus on question numbers with high confidence
- **Spatial Analysis**: Analyzes position and confidence of detected numbers

### ⭕ Enhanced Bubble Detection
- **Advanced Contour Analysis**: Improved bubble detection using contour analysis
- **Flexible Filtering**: More lenient area and circularity filters for better detection
- **Position Sorting**: Automatically sorts bubbles by position (top to bottom, left to right)

### 🔗 Smart Bubble-to-Question Mapping
- **Spatial Relationship Analysis**: Maps bubbles to questions based on spatial proximity
- **Row-based Grouping**: Groups question numbers by rows for better organization
- **Option Mapping**: Maps detected bubbles to answer options (A, B, C, D)

### 💾 Database Integration
- **Complete Data Storage**: Saves all processing results to SQLite database
- **Audit Trail**: Maintains detailed audit trail of all processing steps
- **Performance Metrics**: Tracks processing time and model confidence

### 🌐 Frontend Integration
- **REST API**: Comprehensive REST API for frontend communication
- **Real-time Processing**: Live processing status updates
- **Enhanced Upload Component**: React component with advanced detection preview

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Database     │
│   (React/Next)  │◄──►│   (Flask)        │◄──►│   (SQLite)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Enhanced OMR    │
                    │ Processor       │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ YOLO Model      │
                    │ (Bubble Class.) │
                    └──────────────────┘
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Tesseract OCR
- Trained YOLO model

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# Run the enhanced processor test
python test_enhanced_pipeline.py

# Start the backend API
python web_backend.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### Enhanced OMR Processing
- `POST /api/upload/enhanced` - Upload and process OMR sheet with enhanced detection
- `GET /api/processing/details/{id}` - Get detailed processing information
- `GET /api/detection/preview/{id}` - Get detection preview with mappings

### Standard Endpoints
- `GET /api/health` - Health check
- `GET /api/students` - Get all students
- `POST /api/students` - Add new student
- `GET /api/exams` - Get all exams
- `POST /api/exams` - Add new exam
- `GET /api/results/{student_id}` - Get student results
- `GET /api/statistics/{exam_id}` - Get exam statistics

## Usage Examples

### 1. Enhanced OMR Processing

```python
from enhanced_omr_processor import EnhancedOMRProcessor

# Initialize processor
processor = EnhancedOMRProcessor("path/to/model.pt")

# Process OMR sheet with enhanced detection
result = processor.process_omr_sheet_enhanced(
    image_path="omr_sheet.jpg",
    answer_key={"Q1": "A", "Q2": "B", ...},
    sheet_version="Set A"
)

print(f"Questions detected: {result['question_numbers_detected']}")
print(f"Bubbles detected: {result['bubbles_detected']}")
print(f"Questions mapped: {result['questions_mapped']}")
print(f"Total score: {result['total_score']}/{result['total_questions']}")
```

### 2. Complete Pipeline with Database

```python
# Process and save to database
result = processor.process_and_save(
    image_path="omr_sheet.jpg",
    student_id=1,
    exam_id=1,
    answer_key={"Q1": "A", "Q2": "B", ...},
    sheet_version="Set A"
)

if result.get('success', True):
    print("Processing completed successfully!")
    print(f"OMR Sheet ID: {result['database_save']['omr_sheet_id']}")
    print(f"Evaluation ID: {result['database_save']['evaluation_id']}")
```

### 3. Frontend Integration

```typescript
// Upload OMR sheet with enhanced detection
const formData = new FormData()
formData.append('file', file)
formData.append('student_id', '123')
formData.append('exam_id', '1')
formData.append('sheet_version', 'Set A')

const response = await apiClient.uploadOMRSheetEnhanced(formData)

// Get detection preview
const preview = await apiClient.getDetectionPreview(response.result.database_save.omr_sheet_id)
console.log('Question numbers:', preview.question_numbers)
console.log('Bubbles:', preview.bubbles)
console.log('Question mapping:', preview.question_mapping)
```

## Processing Pipeline

### 1. Image Preprocessing
- Convert to grayscale
- Apply Gaussian blur for noise reduction
- Adaptive thresholding for better contrast

### 2. Question Number Detection
- OCR text detection with Tesseract
- Filter for numeric text with high confidence
- Sort by position (top to bottom, left to right)

### 3. Bubble Detection
- Contour detection and analysis
- Area and circularity filtering
- Position-based sorting

### 4. Spatial Mapping
- Group question numbers by rows
- Map bubbles to questions based on proximity
- Associate bubbles with answer options

### 5. Classification
- Extract individual bubble images
- Classify using trained YOLO model
- Determine marked/unmarked status

### 6. Result Processing
- Map student answers to questions
- Calculate scores and percentages
- Generate subject-wise scores

## Database Schema

### Tables
- `students` - Student information
- `exams` - Exam details and answer keys
- `omr_sheets` - OMR sheet metadata
- `evaluation_results` - Processing results
- `audit_trail` - Processing audit log
- `model_performance` - Model performance metrics

### Enhanced Fields
- `question_numbers_detected` - Count of detected question numbers
- `bubbles_detected` - Count of detected bubbles
- `questions_mapped` - Count of successfully mapped questions
- `question_mapping` - Detailed mapping data

## Performance Metrics

### Detection Accuracy
- Question number detection rate
- Bubble detection accuracy
- Mapping success rate
- Classification confidence

### Processing Performance
- Total processing time
- Individual step timing
- Memory usage
- Model inference time

## Troubleshooting

### Common Issues

1. **OCR Not Working**
   - Ensure Tesseract is installed and in PATH
   - Check image quality and contrast
   - Verify language settings

2. **Model Loading Errors**
   - Verify model file exists and is accessible
   - Check model compatibility
   - Ensure sufficient memory

3. **Database Connection Issues**
   - Check database file permissions
   - Verify database initialization
   - Check for concurrent access

4. **Frontend API Errors**
   - Verify backend is running
   - Check CORS settings
   - Validate API endpoints

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
processor = EnhancedOMRProcessor(model_path)
result = processor.process_omr_sheet_enhanced(image_path, answer_key)
```

## Testing

### Run Test Suite
```bash
python test_enhanced_pipeline.py
```

### Test Components
- Database connection and operations
- Enhanced processor initialization
- Image processing capabilities
- Complete pipeline execution
- API endpoint functionality

## Future Enhancements

### Planned Features
- Multi-language OCR support
- Advanced image preprocessing
- Real-time processing optimization
- Machine learning model improvements
- Enhanced error handling and recovery

### Performance Optimizations
- Parallel processing support
- GPU acceleration
- Caching mechanisms
- Batch processing improvements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test suite for examples
