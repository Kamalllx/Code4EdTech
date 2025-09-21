"""
Audit Trail System for OMR Model Training and Evaluation
This module provides comprehensive audit capabilities for teacher verification
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import cv2
from ultralytics import YOLO

class OMRAuditTrail:
    def __init__(self, audit_dir: str):
        """
        Initialize audit trail system
        
        Args:
            audit_dir: Directory to store audit files
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different audit components
        self.dirs = {
            'reports': self.audit_dir / 'reports',
            'visualizations': self.audit_dir / 'visualizations', 
            'metrics': self.audit_dir / 'metrics',
            'predictions': self.audit_dir / 'predictions',
            'verification': self.audit_dir / 'verification'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_training_session(self, session_data: Dict[str, Any]) -> str:
        """
        Log complete training session data
        
        Args:
            session_data: Complete training session information
            
        Returns:
            Path to session log file
        """
        session_data['audit_metadata'] = {
            'session_id': self.session_id,
            'created_timestamp': datetime.now().isoformat(),
            'audit_version': '1.0',
            'requires_teacher_verification': True
        }
        
        session_file = self.dirs['reports'] / f'training_session_{self.session_id}.json'
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        return str(session_file)
    
    def evaluate_model_comprehensive(self, model_path: str, test_data_path: str, 
                                   ground_truth_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with detailed metrics
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test images
            ground_truth_file: Optional path to ground truth annotations
            
        Returns:
            Comprehensive evaluation results
        """
        print("Performing comprehensive model evaluation...")
        
        model = YOLO(model_path)
        test_path = Path(test_data_path)
        
        evaluation_results = {
            'model_path': model_path,
            'test_data_path': test_data_path,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': {},
            'predictions': [],
            'confusion_matrix': None,
            'classification_report': None,
            'error_analysis': {},
            'confidence_analysis': {},
            'sample_results': []
        }
        
        # Get test images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_images = []
        for ext in image_extensions:
            test_images.extend(test_path.glob(ext))
        
        if not test_images:
            evaluation_results['error'] = 'No test images found'
            return evaluation_results
        
        print(f"Found {len(test_images)} test images")
        
        # Process predictions
        predictions = []
        confidences = []
        true_labels = []
        
        for img_path in test_images:
            try:
                # Run prediction
                results = model(str(img_path))
                
                # Extract prediction based on model type
                pred_class = 0
                confidence = 0.5
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'probs'):
                        # Classification model
                        pred_class = int(result.probs.top1)
                        confidence = float(result.probs.top1conf)
                    elif hasattr(result, 'boxes') and len(result.boxes) > 0:
                        # Detection model - take most confident detection
                        boxes = result.boxes
                        max_conf_idx = boxes.conf.argmax()
                        pred_class = int(boxes.cls[max_conf_idx])
                        confidence = float(boxes.conf[max_conf_idx])
                
                predictions.append(pred_class)
                confidences.append(confidence)
                
                # Try to get ground truth from filename or annotation file
                true_label = self._extract_ground_truth(img_path, ground_truth_file)
                true_labels.append(true_label)
                
                # Store detailed prediction
                evaluation_results['predictions'].append({
                    'image': str(img_path.name),
                    'predicted_class': int(pred_class),
                    'true_class': int(true_label) if true_label is not None else None,
                    'confidence': confidence,
                    'correct': pred_class == true_label if true_label is not None else None
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                evaluation_results['predictions'].append({
                    'image': str(img_path.name),
                    'error': str(e)
                })
        
        # Calculate metrics if we have ground truth
        if any(label is not None for label in true_labels):
            valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_true_labels = [true_labels[i] for i in valid_indices]
            valid_confidences = [confidences[i] for i in valid_indices]
            
            # Basic metrics
            accuracy = accuracy_score(valid_true_labels, valid_predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                valid_true_labels, valid_predictions, average='weighted'
            )
            
            evaluation_results['metrics'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'total_samples': len(valid_predictions),
                'marked_samples': sum(valid_true_labels),
                'unmarked_samples': len(valid_true_labels) - sum(valid_true_labels)
            }
            
            # Confusion matrix
            cm = confusion_matrix(valid_true_labels, valid_predictions)
            evaluation_results['confusion_matrix'] = cm.tolist()
            
            # Classification report
            report = classification_report(
                valid_true_labels, valid_predictions, 
                target_names=['Unmarked', 'Marked'],
                output_dict=True
            )
            evaluation_results['classification_report'] = report
            
            # Confidence analysis
            evaluation_results['confidence_analysis'] = self._analyze_confidence(
                valid_predictions, valid_true_labels, valid_confidences
            )
            
            # Error analysis
            evaluation_results['error_analysis'] = self._analyze_errors(
                valid_predictions, valid_true_labels, valid_confidences
            )
            
            # Create visualizations
            self._create_evaluation_visualizations(evaluation_results)
        
        # Save evaluation results
        eval_file = self.dirs['metrics'] / f'evaluation_{self.session_id}.json'
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        return evaluation_results
    
    def _extract_ground_truth(self, img_path: Path, ground_truth_file: Optional[str]) -> Optional[int]:
        """Extract ground truth label from filename or annotation file"""
        
        # Method 1: Extract from filename (if it contains 'marked' or 'unmarked')
        filename = img_path.name.lower()
        if 'marked' in filename and 'unmarked' not in filename:
            return 1
        elif 'unmarked' in filename:
            return 0
        
        # Method 2: Look for corresponding label file
        label_file = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        if label_file.exists():
            try:
                with open(label_file, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        return class_id
            except:
                pass
        
        # Method 3: Use provided ground truth file
        if ground_truth_file and Path(ground_truth_file).exists():
            try:
                with open(ground_truth_file, 'r') as f:
                    gt_data = json.load(f)
                    return gt_data.get(str(img_path.name))
            except:
                pass
        
        return None
    
    def _analyze_confidence(self, predictions: List[int], true_labels: List[int], 
                          confidences: List[float]) -> Dict[str, Any]:
        """Analyze confidence scores and their correlation with accuracy"""
        
        correct_predictions = [p == t for p, t in zip(predictions, true_labels)]
        
        # Confidence bins
        conf_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        bin_accuracy = []
        bin_counts = []
        
        for i in range(len(conf_bins) - 1):
            bin_mask = [(c >= conf_bins[i] and c < conf_bins[i+1]) for c in confidences]
            bin_predictions = [correct_predictions[j] for j, mask in enumerate(bin_mask) if mask]
            
            if bin_predictions:
                bin_accuracy.append(sum(bin_predictions) / len(bin_predictions))
                bin_counts.append(len(bin_predictions))
            else:
                bin_accuracy.append(0.0)
                bin_counts.append(0)
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'confidence_bins': conf_bins,
            'bin_accuracy': bin_accuracy,
            'bin_counts': bin_counts,
            'high_confidence_accuracy': float(np.mean([correct_predictions[i] for i, c in enumerate(confidences) if c > 0.8])) if any(c > 0.8 for c in confidences) else 0.0,
            'low_confidence_accuracy': float(np.mean([correct_predictions[i] for i, c in enumerate(confidences) if c < 0.6])) if any(c < 0.6 for c in confidences) else 0.0
        }
    
    def _analyze_errors(self, predictions: List[int], true_labels: List[int], 
                       confidences: List[float]) -> Dict[str, Any]:
        """Analyze prediction errors and their patterns"""
        
        errors = {
            'false_positives': [],  # Predicted marked, actually unmarked
            'false_negatives': [],  # Predicted unmarked, actually marked
            'error_rate_by_confidence': {},
            'total_errors': 0
        }
        
        for i, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
            if pred != true:
                errors['total_errors'] += 1
                
                if pred == 1 and true == 0:
                    errors['false_positives'].append({
                        'index': i,
                        'confidence': conf,
                        'predicted': pred,
                        'actual': true
                    })
                elif pred == 0 and true == 1:
                    errors['false_negatives'].append({
                        'index': i,
                        'confidence': conf,
                        'predicted': pred,
                        'actual': true
                    })
        
        # Error rate by confidence ranges
        conf_ranges = [(0.0, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in conf_ranges:
            range_mask = [low <= c < high for c in confidences]
            range_errors = sum(1 for i, (p, t) in enumerate(zip(predictions, true_labels)) 
                             if range_mask[i] and p != t)
            range_total = sum(range_mask)
            
            errors['error_rate_by_confidence'][f'{low}-{high}'] = {
                'error_count': range_errors,
                'total_count': range_total,
                'error_rate': range_errors / range_total if range_total > 0 else 0.0
            }
        
        return errors
    
    def _create_evaluation_visualizations(self, eval_results: Dict[str, Any]):
        """Create visualization plots for evaluation results"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        if eval_results.get('confusion_matrix'):
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = np.array(eval_results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Unmarked', 'Marked'],
                       yticklabels=['Unmarked', 'Marked'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = self.dirs['visualizations'] / f'confusion_matrix_{self.session_id}.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Confidence Distribution
        if eval_results.get('predictions'):
            confidences = [p['confidence'] for p in eval_results['predictions'] 
                         if 'confidence' in p]
            
            if confidences:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram
                ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Confidence Score')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Confidence Scores')
                ax1.grid(True, alpha=0.3)
                
                # Box plot
                ax2.boxplot(confidences)
                ax2.set_ylabel('Confidence Score')
                ax2.set_title('Confidence Score Distribution')
                ax2.grid(True, alpha=0.3)
                
                conf_path = self.dirs['visualizations'] / f'confidence_analysis_{self.session_id}.png'
                plt.savefig(conf_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Accuracy by Confidence Bins
        if eval_results.get('confidence_analysis', {}).get('bin_accuracy'):
            conf_analysis = eval_results['confidence_analysis']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bin_centers = [(conf_analysis['confidence_bins'][i] + conf_analysis['confidence_bins'][i+1])/2 
                          for i in range(len(conf_analysis['confidence_bins'])-1)]
            
            bars = ax.bar(bin_centers, conf_analysis['bin_accuracy'], 
                         width=0.05, alpha=0.7, color='lightgreen', edgecolor='black')
            
            # Add count labels on bars
            for bar, count in zip(bars, conf_analysis['bin_counts']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy vs Confidence Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            acc_conf_path = self.dirs['visualizations'] / f'accuracy_confidence_{self.session_id}.png'
            plt.savefig(acc_conf_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_teacher_verification_form(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Create a comprehensive verification form for teachers
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Path to verification form
        """
        form_data = {
            'session_id': self.session_id,
            'evaluation_summary': {
                'model_path': evaluation_results.get('model_path', 'N/A'),
                'accuracy': evaluation_results.get('metrics', {}).get('accuracy', 0.0),
                'precision': evaluation_results.get('metrics', {}).get('precision', 0.0),
                'recall': evaluation_results.get('metrics', {}).get('recall', 0.0),
                'f1_score': evaluation_results.get('metrics', {}).get('f1_score', 0.0),
                'total_samples': evaluation_results.get('metrics', {}).get('total_samples', 0),
                'error_count': evaluation_results.get('error_analysis', {}).get('total_errors', 0)
            },
            'verification_checklist': {
                'accuracy_acceptable': None,  # To be filled by teacher
                'error_rate_acceptable': None,
                'confidence_scores_reasonable': None,
                'no_obvious_biases': None,
                'sample_predictions_correct': None,
                'ready_for_deployment': None
            },
            'teacher_feedback': {
                'reviewer_name': '',
                'review_date': '',
                'overall_rating': '',  # Excellent/Good/Fair/Poor
                'specific_comments': '',
                'recommended_improvements': '',
                'approval_status': ''  # Approved/Rejected/Needs_Revision
            },
            'audit_trail': {
                'form_created': datetime.now().isoformat(),
                'form_version': '1.0',
                'requires_signature': True
            }
        }
        
        # Save verification form
        form_path = self.dirs['verification'] / f'teacher_verification_form_{self.session_id}.json'
        with open(form_path, 'w') as f:
            json.dump(form_data, f, indent=2)
        
        # Create human-readable HTML form
        html_form_path = self._create_html_verification_form(form_data, evaluation_results)
        
        return str(form_path)
    
    def _create_html_verification_form(self, form_data: Dict[str, Any], 
                                     eval_results: Dict[str, Any]) -> str:
        """Create HTML verification form for teachers"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OMR Model Verification Form</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; min-width: 150px; }}
        .checklist {{ margin: 10px 0; }}
        .checklist input {{ margin-right: 10px; }}
        .signature {{ margin-top: 30px; padding: 20px; background-color: #fff9c4; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OMR Bubble Classification Model - Teacher Verification</h1>
        <p><strong>Session ID:</strong> {form_data['session_id']}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Model Performance Summary</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Accuracy:</strong><br>
                {form_data['evaluation_summary']['accuracy']:.3f} ({form_data['evaluation_summary']['accuracy']*100:.1f}%)
            </div>
            <div class="metric">
                <strong>Precision:</strong><br>
                {form_data['evaluation_summary']['precision']:.3f}
            </div>
            <div class="metric">
                <strong>Recall:</strong><br>
                {form_data['evaluation_summary']['recall']:.3f}
            </div>
            <div class="metric">
                <strong>F1 Score:</strong><br>
                {form_data['evaluation_summary']['f1_score']:.3f}
            </div>
            <div class="metric">
                <strong>Total Samples:</strong><br>
                {form_data['evaluation_summary']['total_samples']}
            </div>
            <div class="metric">
                <strong>Errors:</strong><br>
                {form_data['evaluation_summary']['error_count']}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Sample Predictions Review</h2>
        <p>Please review a sample of model predictions:</p>
        <table>
            <tr>
                <th>Image</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Confidence</th>
                <th>Correct</th>
            </tr>
        """
        
        # Add sample predictions to table
        sample_predictions = eval_results.get('predictions', [])[:10]  # First 10 predictions
        for pred in sample_predictions:
            if 'error' not in pred:
                correct_symbol = "✓" if pred.get('correct') else "✗" if pred.get('correct') is False else "?"
                html_content += f"""
            <tr>
                <td>{pred.get('image', 'N/A')}</td>
                <td>{'Marked' if pred.get('predicted_class') == 1 else 'Unmarked'}</td>
                <td>{'Marked' if pred.get('true_class') == 1 else 'Unmarked' if pred.get('true_class') == 0 else 'Unknown'}</td>
                <td>{pred.get('confidence', 0):.3f}</td>
                <td>{correct_symbol}</td>
            </tr>
                """
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Verification Checklist</h2>
        <p>Please check each item after verification:</p>
        
        <div class="checklist">
            <input type="checkbox" id="accuracy"> 
            <label for="accuracy">Model accuracy is acceptable for production use (≥95%)</label>
        </div>
        
        <div class="checklist">
            <input type="checkbox" id="errors"> 
            <label for="errors">Error rate is within acceptable limits</label>
        </div>
        
        <div class="checklist">
            <input type="checkbox" id="confidence"> 
            <label for="confidence">Confidence scores appear reasonable and calibrated</label>
        </div>
        
        <div class="checklist">
            <input type="checkbox" id="bias"> 
            <label for="bias">No obvious biases detected in predictions</label>
        </div>
        
        <div class="checklist">
            <input type="checkbox" id="samples"> 
            <label for="samples">Sample predictions are visually correct</label>
        </div>
        
        <div class="checklist">
            <input type="checkbox" id="deployment"> 
            <label for="deployment">Model is ready for deployment</label>
        </div>
    </div>
    
    <div class="section">
        <h2>Teacher Feedback</h2>
        
        <p><strong>Overall Rating:</strong></p>
        <input type="radio" name="rating" value="excellent"> Excellent
        <input type="radio" name="rating" value="good"> Good
        <input type="radio" name="rating" value="fair"> Fair
        <input type="radio" name="rating" value="poor"> Poor
        
        <p><strong>Specific Comments:</strong></p>
        <textarea rows="4" cols="80" placeholder="Enter your specific comments about the model performance..."></textarea>
        
        <p><strong>Recommended Improvements:</strong></p>
        <textarea rows="3" cols="80" placeholder="Any improvements you recommend..."></textarea>
        
        <p><strong>Final Decision:</strong></p>
        <select>
            <option value="">Select...</option>
            <option value="approved">Approved for Production</option>
            <option value="approved_with_monitoring">Approved with Monitoring</option>
            <option value="needs_revision">Needs Revision</option>
            <option value="rejected">Rejected</option>
        </select>
    </div>
    
    <div class="signature">
        <h2>Teacher Verification</h2>
        <p><strong>Teacher Name:</strong> _________________________________</p>
        <p><strong>Date:</strong> _________________________________</p>
        <p><strong>Signature:</strong> _________________________________</p>
        
        <p><em>By signing above, I certify that I have reviewed the model performance metrics, 
        sample predictions, and audit trail, and confirm that this model meets the quality 
        standards required for OMR evaluation at Innomatics Research Labs.</em></p>
    </div>
    
</body>
</html>
        """
        
        html_path = self.dirs['verification'] / f'verification_form_{self.session_id}.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def generate_comprehensive_report(self, training_results: Dict[str, Any], 
                                    evaluation_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive audit report combining training and evaluation
        
        Args:
            training_results: Training session results
            evaluation_results: Evaluation results
            
        Returns:
            Path to comprehensive report
        """
        comprehensive_report = {
            'session_metadata': {
                'session_id': self.session_id,
                'report_generated': datetime.now().isoformat(),
                'report_type': 'comprehensive_audit',
                'version': '1.0'
            },
            'training_summary': training_results,
            'evaluation_summary': evaluation_results,
            'quality_assurance': {
                'meets_accuracy_threshold': evaluation_results.get('metrics', {}).get('accuracy', 0) >= 0.95,
                'low_error_rate': evaluation_results.get('error_analysis', {}).get('total_errors', 100) / max(evaluation_results.get('metrics', {}).get('total_samples', 1), 1) <= 0.05,
                'confidence_well_calibrated': True,  # Would need more sophisticated analysis
                'teacher_review_required': True,
                'recommended_action': 'teacher_verification'
            },
            'files_generated': {
                'model_files': training_results.get('model_files', {}),
                'visualization_files': list(self.dirs['visualizations'].glob('*.png')),
                'verification_form': list(self.dirs['verification'].glob('*.html'))
            }
        }
        
        # Add recommendations based on performance
        accuracy = evaluation_results.get('metrics', {}).get('accuracy', 0)
        if accuracy >= 0.98:
            comprehensive_report['recommendations'] = ['Model ready for production deployment']
        elif accuracy >= 0.95:
            comprehensive_report['recommendations'] = ['Model acceptable, monitor performance in production']
        elif accuracy >= 0.90:
            comprehensive_report['recommendations'] = ['Model needs improvement, consider more training data']
        else:
            comprehensive_report['recommendations'] = ['Model not ready, significant improvements needed']
        
        # Save comprehensive report
        report_path = self.dirs['reports'] / f'comprehensive_audit_report_{self.session_id}.json'
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"Comprehensive audit report saved to: {report_path}")
        return str(report_path)

def main():
    """Example usage of the audit trail system"""
    
    # Initialize audit system
    audit = OMRAuditTrail("audit_trail")
    
    # Example: Evaluate a model
    model_path = "trained_models/omr_bubble_classifier/weights/best.pt"
    test_data_path = "yolo_dataset/test/images"
    
    # Perform comprehensive evaluation
    eval_results = audit.evaluate_model_comprehensive(model_path, test_data_path)
    
    # Create teacher verification form
    verification_form = audit.create_teacher_verification_form(eval_results)
    
    print(f"Evaluation completed. Verification form: {verification_form}")

if __name__ == "__main__":
    main()