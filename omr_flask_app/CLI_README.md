# OMR Flask Application - CLI Interface

This CLI interface provides command-line access to the OMR (Optical Mark Recognition) Flask application, allowing users to interact with the system without needing a web browser.

## Features

- **Image Capture**: Capture images from PC, phone, or AR cameras
- **Single Image Processing**: Process individual OMR sheets with optional answer keys
- **Batch Processing**: Process multiple images from a directory
- **Offline Mode**: Use local services without Flask app running
- **System Status**: Check health and configuration of services
- **Camera Management**: List and configure available cameras

## Installation

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Commands

```bash
# Show help
python cli.py --help

# Check system status
python cli.py status

# List available cameras
python cli.py cameras
```

### Image Capture

```bash
# Capture from PC camera
python cli.py capture

# Capture from phone camera
python cli.py capture --camera phone

# Capture to specific path
python cli.py capture --output my_capture.jpg
```

### Single Image Processing

```bash
# Process image without answer key
python cli.py process image.jpg

# Process image with answer key
python cli.py process image.jpg --answer-key sample_answer_key.json
```

### Batch Processing

```bash
# Process all JPG images in directory
python cli.py batch images_folder/

# Process with answer key
python cli.py batch images_folder/ --answer-key sample_answer_key.json

# Process with custom file pattern
python cli.py batch images_folder/ --pattern "*.png" --answer-key key.json
```

### Offline Mode

Use local services without Flask app:

```bash
# Any command with --offline flag
python cli.py --offline process image.jpg
python cli.py --offline batch images_folder/ --answer-key key.json
```

### Custom Configuration

```bash
# Use different Flask app URL
python cli.py --url http://localhost:8080 status

# Use custom output directory
python cli.py --output-dir my_results process image.jpg
```

## Answer Key Format

The answer key should be a JSON file with the following structure:

```json
{
  "exam_id": "exam_001",
  "exam_title": "Sample Test",
  "total_questions": 10,
  "questions_per_row": 5,
  "options_per_question": 4,
  "correct_answers": {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "A",
    "6": "B",
    "7": "C",
    "8": "D",
    "9": "A",
    "10": "B"
  },
  "scoring": {
    "correct_marks": 1,
    "incorrect_marks": -0.25,
    "unanswered_marks": 0,
    "grading_scale": {
      "A+": 90,
      "A": 80,
      "B": 60,
      "C": 50,
      "F": 0
    }
  }
}
```

See `sample_answer_key.json` for a complete example.

## Output Structure

The CLI creates the following output structure:

```
cli_output/
├── captures/          # Captured images
├── processed/         # Preprocessed images
├── results/           # Final results and reports
│   ├── annotated_*.jpg    # Images with detected bubbles
│   ├── report_*.json      # Processing reports
│   └── batch_report_*.json # Batch processing summaries
└── audit/             # Audit trails (offline mode)
```

## Processing Pipeline

1. **Image Capture/Loading**: Load image from camera or file
2. **Preprocessing**: Denoise, enhance contrast, correct perspective
3. **Bubble Detection**: Use computer vision to find bubble regions
4. **Classification**: Use YOLO model to classify bubble states
5. **Answer Extraction**: Organize bubbles by questions and extract answers
6. **Scoring**: Compare with answer key and calculate scores
7. **Report Generation**: Create detailed reports and annotated images

## Examples

### Quick Start

```bash
# Process a single image with built-in sample answer key
python cli.py process sample_omr.jpg --answer-key sample_answer_key.json
```

### Complete Workflow

```bash
# 1. Check system status
python cli.py status

# 2. List cameras
python cli.py cameras

# 3. Capture image
python cli.py capture --camera pc --output test_sheet.jpg

# 4. Process captured image
python cli.py process test_sheet.jpg --answer-key sample_answer_key.json

# 5. Check results in cli_output/results/
```

### Batch Processing Workflow

```bash
# Process entire Set A from sample data
python cli.py batch "../Theme 1 - Sample Data/Set A/" --answer-key sample_answer_key.json --pattern "*.jpeg"

# Process with offline mode
python cli.py --offline batch "../Theme 1 - Sample Data/Set B/" --answer-key sample_answer_key.json --pattern "*.jpeg"
```

## Error Handling

- The CLI provides detailed error messages and suggestions
- Use `--offline` mode if Flask app is not running
- Check system status to troubleshoot configuration issues
- Ensure YOLO model is available for OMR evaluation

## Advanced Usage

### Integration with Scripts

```python
from cli import OMRCLIClient

# Create client
client = OMRCLIClient(offline_mode=True)

# Process image programmatically
result = client.process_file("image.jpg", "answer_key.json")
if result['success']:
    print("Processing successful!")
```

### Custom Processing Pipeline

The CLI can be extended with custom processing steps by modifying the `_process_image` method or adding new commands to the argument parser.

## Troubleshooting

1. **YOLO model not found**: Ensure the model path in config is correct
2. **Camera not detected**: Check camera permissions and drivers
3. **Processing fails**: Use `python cli.py status` to check service availability
4. **No output**: Check `cli_output/` directory permissions

## Support

For issues or questions:
1. Check the system status: `python cli.py status`
2. Review the processing logs in output files
3. Try offline mode if Flask app issues occur
4. Ensure all dependencies are correctly installed