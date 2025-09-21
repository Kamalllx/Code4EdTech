 #!/usr/bin/env python3
"""
OMR Database System
Handles database operations for OMR evaluation results, student data, and audit trails.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

class OMRDatabase:
    """Database manager for OMR evaluation system"""
    
    def __init__(self, db_path: str = "omr_evaluation.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Exams table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_name TEXT NOT NULL,
                    exam_date DATE NOT NULL,
                    total_questions INTEGER NOT NULL,
                    subjects TEXT NOT NULL,  -- JSON array of subjects
                    answer_key TEXT NOT NULL,  -- JSON object with answers
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # OMR Sheets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS omr_sheets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    exam_id INTEGER NOT NULL,
                    sheet_image_path TEXT NOT NULL,
                    processed_image_path TEXT,
                    sheet_version TEXT,  -- Set A, Set B, etc.
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    FOREIGN KEY (student_id) REFERENCES students (id),
                    FOREIGN KEY (exam_id) REFERENCES exams (id)
                )
            ''')
            
            # Evaluation Results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    omr_sheet_id INTEGER NOT NULL,
                    subject_scores TEXT NOT NULL,  -- JSON object with subject-wise scores
                    total_score INTEGER NOT NULL,
                    percentage REAL NOT NULL,
                    answers TEXT NOT NULL,  -- JSON array of student answers
                    processing_time REAL,
                    model_confidence REAL,
                    evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (omr_sheet_id) REFERENCES omr_sheets (id)
                )
            ''')
            
            # Audit Trail table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    omr_sheet_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,  -- JSON object with action details
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    FOREIGN KEY (omr_sheet_id) REFERENCES omr_sheets (id)
                )
            ''')
            
            # Model Performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    training_date TIMESTAMP,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_path TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            logging.info("Database initialized successfully")
    
    def add_student(self, student_id: str, name: str, email: str = None) -> int:
        """Add a new student to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO students (student_id, name, email) VALUES (?, ?, ?)",
                    (student_id, name, email)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Student already exists, return existing ID
                cursor.execute(
                    "SELECT id FROM students WHERE student_id = ?",
                    (student_id,)
                )
                return cursor.fetchone()[0]
    
    def add_exam(self, exam_name: str, exam_date: str, total_questions: int, 
                 subjects: List[str], answer_key: Dict) -> int:
        """Add a new exam to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO exams (exam_name, exam_date, total_questions, subjects, answer_key) VALUES (?, ?, ?, ?, ?)",
                (exam_name, exam_date, total_questions, json.dumps(subjects), json.dumps(answer_key))
            )
            return cursor.lastrowid
    
    def add_omr_sheet(self, student_id: int, exam_id: int, sheet_image_path: str, 
                     sheet_version: str = None) -> int:
        """Add a new OMR sheet to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO omr_sheets (student_id, exam_id, sheet_image_path, sheet_version) VALUES (?, ?, ?, ?)",
                (student_id, exam_id, sheet_image_path, sheet_version)
            )
            return cursor.lastrowid
    
    def update_omr_processing(self, omr_sheet_id: int, processed_image_path: str, 
                            status: str = 'completed'):
        """Update OMR sheet processing status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE omr_sheets SET processed_image_path = ?, processing_status = ? WHERE id = ?",
                (processed_image_path, status, omr_sheet_id)
            )
    
    def add_evaluation_result(self, omr_sheet_id: int, subject_scores: Dict, 
                            total_score: int, percentage: float, answers: List,
                            processing_time: float = None, model_confidence: float = None) -> int:
        """Add evaluation results for an OMR sheet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO evaluation_results (omr_sheet_id, subject_scores, total_score, percentage, answers, processing_time, model_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (omr_sheet_id, json.dumps(subject_scores), total_score, percentage, 
                 json.dumps(answers), processing_time, model_confidence)
            )
            return cursor.lastrowid
    
    def add_audit_entry(self, omr_sheet_id: int, action: str, details: Dict = None, 
                       user_id: str = None):
        """Add an audit trail entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO audit_trail (omr_sheet_id, action, details, user_id) VALUES (?, ?, ?, ?)",
                (omr_sheet_id, action, json.dumps(details) if details else None, user_id)
            )
    
    def add_model_performance(self, model_name: str, model_version: str, 
                            accuracy: float, precision: float, recall: float, 
                            f1_score: float, model_path: str, training_date: str = None):
        """Add model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO model_performance (model_name, model_version, accuracy, precision_score, recall_score, f1_score, model_path, training_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (model_name, model_version, accuracy, precision, recall, f1_score, model_path, training_date)
            )
    
    def get_student_results(self, student_id: str, exam_id: int = None) -> List[Dict]:
        """Get evaluation results for a student"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT er.*, s.student_id, s.name, e.exam_name, e.exam_date
                FROM evaluation_results er
                JOIN omr_sheets os ON er.omr_sheet_id = os.id
                JOIN students s ON os.student_id = s.id
                JOIN exams e ON os.exam_id = e.id
                WHERE s.student_id = ?
            """
            params = [student_id]
            
            if exam_id:
                query += " AND e.id = ?"
                params.append(exam_id)
            
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['subject_scores'] = json.loads(result['subject_scores'])
                result['answers'] = json.loads(result['answers'])
                results.append(result)
            
            return results
    
    def get_exam_statistics(self, exam_id: int) -> Dict:
        """Get statistics for an exam"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total students
            cursor.execute(
                "SELECT COUNT(DISTINCT student_id) FROM omr_sheets WHERE exam_id = ?",
                (exam_id,)
            )
            total_students = cursor.fetchone()[0]
            
            # Get average score
            cursor.execute(
                "SELECT AVG(percentage) FROM evaluation_results er JOIN omr_sheets os ON er.omr_sheet_id = os.id WHERE os.exam_id = ?",
                (exam_id,)
            )
            avg_score = cursor.fetchone()[0] or 0
            
            # Get score distribution
            cursor.execute(
                "SELECT percentage FROM evaluation_results er JOIN omr_sheets os ON er.omr_sheet_id = os.id WHERE os.exam_id = ?",
                (exam_id,)
            )
            scores = [row[0] for row in cursor.fetchall()]
            
            return {
                'total_students': total_students,
                'average_score': round(avg_score, 2),
                'score_distribution': scores,
                'highest_score': max(scores) if scores else 0,
                'lowest_score': min(scores) if scores else 0
            }
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluation results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT er.*, s.student_id, s.name, e.exam_name, os.sheet_image_path
                FROM evaluation_results er
                JOIN omr_sheets os ON er.omr_sheet_id = os.id
                JOIN students s ON os.student_id = s.id
                JOIN exams e ON os.exam_id = e.id
                ORDER BY er.evaluation_timestamp DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['subject_scores'] = json.loads(result['subject_scores'])
                result['answers'] = json.loads(result['answers'])
                results.append(result)
            
            return results
    
    def export_results_csv(self, exam_id: int, output_path: str):
        """Export exam results to CSV"""
        import csv
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT s.student_id, s.name, er.total_score, er.percentage, 
                       er.subject_scores, er.evaluation_timestamp
                FROM evaluation_results er
                JOIN omr_sheets os ON er.omr_sheet_id = os.id
                JOIN students s ON os.student_id = s.id
                WHERE os.exam_id = ?
                ORDER BY er.total_score DESC
            """, (exam_id,))
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['student_id', 'name', 'total_score', 'percentage', 'subject_scores', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in cursor.fetchall():
                    writer.writerow(dict(row))
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data (optional maintenance function)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old audit entries
            cursor.execute(
                "DELETE FROM audit_trail WHERE timestamp < datetime('now', '-{} days')".format(days_old)
            )
            
            conn.commit()
            logging.info(f"Cleaned up data older than {days_old} days")

# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = OMRDatabase()
    
    # Example: Add a student
    student_id = db.add_student("STU001", "John Doe", "john@example.com")
    print(f"Added student with ID: {student_id}")
    
    # Example: Add an exam
    subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "English"]
    answer_key = {
        "Q1": "A", "Q2": "B", "Q3": "C", "Q4": "D", "Q5": "A",
        # ... more answers
    }
    exam_id = db.add_exam("Midterm Exam", "2024-01-15", 100, subjects, answer_key)
    print(f"Added exam with ID: {exam_id}")
    
    print("Database setup completed successfully!")