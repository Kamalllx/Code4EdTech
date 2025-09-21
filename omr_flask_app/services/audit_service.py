"""
Audit Service for OMR Flask Application
Handles audit trail, logging, and compliance tracking
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import logging


class AuditService:
    def __init__(self, audit_dir: str):
        """
        Initialize audit service
        
        Args:
            audit_dir: Directory to store audit logs
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.audit_dir / 'sessions').mkdir(exist_ok=True)
        (self.audit_dir / 'daily_logs').mkdir(exist_ok=True)
        (self.audit_dir / 'system_logs').mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup file logging
        log_file = self.audit_dir / 'system_logs' / f'audit_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def create_capture_entry(self, session_id: str, camera_type: str, 
                           image_path: str, image_shape: tuple) -> Dict[str, Any]:
        """Create audit entry for image capture"""
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'session_id': session_id,
            'event_type': 'image_capture',
            'timestamp': datetime.now().isoformat(),
            'camera_type': camera_type,
            'image_path': str(image_path),
            'image_info': {
                'width': image_shape[1],
                'height': image_shape[0],
                'channels': image_shape[2] if len(image_shape) > 2 else 1,
                'size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
            },
            'compliance': {
                'data_retention_required': True,
                'encryption_required': False,
                'anonymization_required': True
            }
        }
        
        # Save session audit
        self._save_session_audit(session_id, audit_entry)
        
        # Log event
        self.logger.info(f"Image captured - Session: {session_id}, Camera: {camera_type}")
        
        return audit_entry
    
    def update_preprocessing(self, session_id: str, preprocessing_info: Dict[str, Any], 
                           processed_image_path: str) -> None:
        """Update audit with preprocessing information"""
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'session_id': session_id,
            'event_type': 'preprocessing',
            'timestamp': datetime.now().isoformat(),
            'preprocessing_info': preprocessing_info,
            'processed_image_path': str(processed_image_path),
            'processing_quality': self._assess_preprocessing_quality(preprocessing_info)
        }
        
        # Save session audit
        self._save_session_audit(session_id, audit_entry)
        
        # Log event
        self.logger.info(f"Image preprocessed - Session: {session_id}")
    
    def update_evaluation(self, session_id: str, evaluation_results: Dict[str, Any], 
                         report_path: str) -> None:
        """Update audit with evaluation results"""
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'session_id': session_id,
            'event_type': 'omr_evaluation',
            'timestamp': datetime.now().isoformat(),
            'evaluation_summary': {
                'success': evaluation_results.get('success', False),
                'total_bubbles': evaluation_results.get('total_bubbles_detected', 0),
                'marked_bubbles': evaluation_results.get('marked_bubbles', 0),
                'questions_answered': len(evaluation_results.get('student_answers', {})),
                'score_percentage': evaluation_results.get('scoring_results', {}).get('score_percentage', 0)
            },
            'report_path': str(report_path),
            'model_version': evaluation_results.get('model_info', {}),
            'quality_flags': self._assess_evaluation_quality(evaluation_results)
        }
        
        # Save session audit
        self._save_session_audit(session_id, audit_entry)
        
        # Log event
        score = audit_entry['evaluation_summary']['score_percentage']
        self.logger.info(f"OMR evaluation completed - Session: {session_id}, Score: {score:.1f}%")
    
    def get_session_audit(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get complete audit trail for a session"""
        session_file = self.audit_dir / 'sessions' / f'{session_id}.json'
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read session audit {session_id}: {str(e)}")
            return None
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        try:
            stats = {
                'total_sessions': 0,
                'sessions_today': 0,
                'total_evaluations': 0,
                'average_score': 0.0,
                'success_rate': 0.0,
                'system_uptime': self._get_system_uptime(),
                'storage_usage': self._get_storage_usage(),
                'recent_activity': []
            }
            
            # Count session files
            session_files = list((self.audit_dir / 'sessions').glob('*.json'))
            stats['total_sessions'] = len(session_files)
            
            # Analyze recent sessions
            today = datetime.now().date()
            scores = []
            successful_evaluations = 0
            recent_sessions = []
            
            for session_file in session_files[-50:]:  # Last 50 sessions
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    session_date = None
                    has_evaluation = False
                    session_score = None
                    
                    for event in session_data.get('events', []):
                        event_time = datetime.fromisoformat(event['timestamp'])
                        
                        if session_date is None:
                            session_date = event_time.date()
                        
                        if event['event_type'] == 'omr_evaluation':
                            has_evaluation = True
                            stats['total_evaluations'] += 1
                            
                            if event.get('evaluation_summary', {}).get('success', False):
                                successful_evaluations += 1
                                session_score = event['evaluation_summary'].get('score_percentage', 0)
                                scores.append(session_score)
                    
                    if session_date == today:
                        stats['sessions_today'] += 1
                    
                    # Add to recent activity
                    if len(recent_sessions) < 10:
                        recent_sessions.append({
                            'session_id': session_file.stem,
                            'date': session_date.isoformat() if session_date else None,
                            'has_evaluation': has_evaluation,
                            'score': session_score
                        })
                
                except Exception as e:
                    self.logger.error(f"Error processing session file {session_file}: {str(e)}")
            
            # Calculate statistics
            if scores:
                stats['average_score'] = sum(scores) / len(scores)
            
            if stats['total_evaluations'] > 0:
                stats['success_rate'] = (successful_evaluations / stats['total_evaluations']) * 100
            
            stats['recent_activity'] = recent_sessions
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating system statistics: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_old_audits(self, retention_days: int = 30) -> Dict[str, Any]:
        """Clean up old audit files based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleanup_results = {
            'files_deleted': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        try:
            # Clean up session files
            for session_file in (self.audit_dir / 'sessions').glob('*.json'):
                try:
                    file_time = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_size = session_file.stat().st_size
                        session_file.unlink()
                        cleanup_results['files_deleted'] += 1
                        cleanup_results['space_freed_mb'] += file_size / (1024 * 1024)
                except Exception as e:
                    cleanup_results['errors'].append(f"Error deleting {session_file}: {str(e)}")
            
            # Clean up daily logs
            for log_file in (self.audit_dir / 'daily_logs').glob('*.json'):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        cleanup_results['files_deleted'] += 1
                        cleanup_results['space_freed_mb'] += file_size / (1024 * 1024)
                except Exception as e:
                    cleanup_results['errors'].append(f"Error deleting {log_file}: {str(e)}")
            
            self.logger.info(f"Audit cleanup completed: {cleanup_results['files_deleted']} files deleted, "
                           f"{cleanup_results['space_freed_mb']:.2f} MB freed")
            
        except Exception as e:
            cleanup_results['errors'].append(f"General cleanup error: {str(e)}")
            self.logger.error(f"Audit cleanup failed: {str(e)}")
        
        return cleanup_results
    
    def export_audit_report(self, start_date: datetime, end_date: datetime, 
                          format_type: str = 'json') -> str:
        """Export audit report for a date range"""
        try:
            report_data = {
                'report_generated': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'sessions': [],
                'summary': {
                    'total_sessions': 0,
                    'successful_evaluations': 0,
                    'failed_evaluations': 0,
                    'average_score': 0.0
                }
            }
            
            # Collect sessions in date range
            scores = []
            successful_count = 0
            failed_count = 0
            
            for session_file in (self.audit_dir / 'sessions').glob('*.json'):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    # Check if session is in date range
                    session_events = session_data.get('events', [])
                    if not session_events:
                        continue
                    
                    first_event_time = datetime.fromisoformat(session_events[0]['timestamp'])
                    
                    if start_date <= first_event_time <= end_date:
                        report_data['sessions'].append(session_data)
                        report_data['summary']['total_sessions'] += 1
                        
                        # Check for evaluation results
                        for event in session_events:
                            if event['event_type'] == 'omr_evaluation':
                                if event.get('evaluation_summary', {}).get('success', False):
                                    successful_count += 1
                                    scores.append(event['evaluation_summary'].get('score_percentage', 0))
                                else:
                                    failed_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing session {session_file} for report: {str(e)}")
            
            # Calculate summary statistics
            report_data['summary']['successful_evaluations'] = successful_count
            report_data['summary']['failed_evaluations'] = failed_count
            
            if scores:
                report_data['summary']['average_score'] = sum(scores) / len(scores)
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'audit_report_{timestamp}.{format_type}'
            report_path = self.audit_dir / report_filename
            
            if format_type == 'json':
                with open(report_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            else:
                # Could add CSV, Excel export here
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"Audit report exported: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export audit report: {str(e)}")
            raise
    
    def _save_session_audit(self, session_id: str, audit_entry: Dict[str, Any]) -> None:
        """Save audit entry to session file"""
        session_file = self.audit_dir / 'sessions' / f'{session_id}.json'
        
        # Load existing session data or create new
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            except Exception:
                session_data = {'session_id': session_id, 'events': []}
        else:
            session_data = {'session_id': session_id, 'events': []}
        
        # Add new event
        session_data['events'].append(audit_entry)
        session_data['last_updated'] = datetime.now().isoformat()
        
        # Save updated session data
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save session audit {session_id}: {str(e)}")
    
    def _assess_preprocessing_quality(self, preprocessing_info: Dict[str, Any]) -> List[str]:
        """Assess preprocessing quality and return any flags"""
        flags = []
        
        quality_metrics = preprocessing_info.get('quality_metrics', {})
        
        # Check sharpness
        sharpness = quality_metrics.get('sharpness', 0)
        if sharpness < 100:
            flags.append('low_sharpness')
        
        # Check contrast
        contrast = quality_metrics.get('contrast', 0)
        if contrast < 30:
            flags.append('low_contrast')
        
        # Check brightness
        brightness = quality_metrics.get('brightness', 0)
        if brightness < 50 or brightness > 200:
            flags.append('brightness_issues')
        
        # Check similarity to original
        similarity = quality_metrics.get('similarity_to_original', 1.0)
        if similarity < 0.7:
            flags.append('significant_changes')
        
        return flags
    
    def _assess_evaluation_quality(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Assess evaluation quality and return any flags"""
        flags = []
        
        # Check if evaluation was successful
        if not evaluation_results.get('success', False):
            flags.append('evaluation_failed')
            return flags
        
        # Check detection counts
        total_bubbles = evaluation_results.get('total_bubbles_detected', 0)
        if total_bubbles == 0:
            flags.append('no_bubbles_detected')
        elif total_bubbles < 10:
            flags.append('few_bubbles_detected')
        
        # Check answer quality
        student_answers = evaluation_results.get('student_answers', {})
        multiple_marks = sum(1 for answer in student_answers.values() 
                           if answer.get('multiple_marks', False))
        
        if multiple_marks > 0:
            flags.append('multiple_marks_detected')
        
        # Check confidence levels
        low_confidence_answers = sum(1 for answer in student_answers.values() 
                                   if answer.get('confidence', 1.0) < 0.7)
        
        if low_confidence_answers > len(student_answers) * 0.3:
            flags.append('low_confidence_answers')
        
        return flags
    
    def _get_system_uptime(self) -> str:
        """Get system uptime information"""
        try:
            # This is a simplified version - in production, you'd track actual application uptime
            return "System statistics tracking active"
        except Exception:
            return "Unknown"
    
    def _get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage information"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.audit_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'directory': str(self.audit_dir)
            }
        except Exception as e:
            return {'error': str(e)}