#!/usr/bin/env python3
"""
OMR Evaluation Web Interface
Streamlit-based web application for OMR sheet evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Import our modules
from database import OMRDatabase
from omr_processor import OMRProcessor

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = OMRDatabase()
if 'processor' not in st.session_state:
    st.session_state.processor = None

def load_model():
    """Load the trained OMR model"""
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    if not Path(model_path).exists():
        st.error(f"Trained model not found at: {model_path}")
        st.info("Please train the model first using: `python train.py`")
        return None
    
    try:
        processor = OMRProcessor(model_path)
        st.session_state.processor = processor
        return processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    """Main application"""
    st.title("📝 OMR Evaluation System")
    st.markdown("Automated Optical Mark Recognition for Educational Assessments")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📤 Upload Sheets", "📊 Results", "⚙️ Settings", "📈 Analytics"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload Sheets":
        show_upload_page()
    elif page == "📊 Results":
        show_results_page()
    elif page == "⚙️ Settings":
        show_settings_page()
    elif page == "📈 Analytics":
        show_analytics_page()

def show_dashboard():
    """Dashboard page"""
    st.header("Dashboard")
    
    # Load model status
    if st.session_state.processor is None:
        if st.button("Load OMR Model"):
            with st.spinner("Loading model..."):
                processor = load_model()
                if processor:
                    st.success("Model loaded successfully!")
                    st.rerun()
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Status", "✅ Ready" if st.session_state.processor else "❌ Not Loaded")
    
    with col2:
        # Get total students from database
        try:
            with st.session_state.db.db_path as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM students")
                total_students = cursor.fetchone()[0]
        except:
            total_students = 0
        st.metric("Total Students", total_students)
    
    with col3:
        try:
            with st.session_state.db.db_path as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM evaluation_results")
                total_evaluations = cursor.fetchone()[0]
        except:
            total_evaluations = 0
        st.metric("Evaluations Done", total_evaluations)
    
    with col4:
        st.metric("System Status", "🟢 Online")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    try:
        recent_results = st.session_state.db.get_recent_evaluations(5)
        if recent_results:
            df = pd.DataFrame(recent_results)
            st.dataframe(df[['student_id', 'name', 'total_score', 'percentage', 'evaluation_timestamp']], 
                        use_container_width=True)
        else:
            st.info("No recent evaluations found.")
    except Exception as e:
        st.error(f"Error loading recent activity: {e}")

def show_upload_page():
    """Upload and process OMR sheets"""
    st.header("Upload OMR Sheets")
    
    # Check if model is loaded
    if st.session_state.processor is None:
        st.warning("Please load the OMR model first from the Dashboard.")
        return
    
    # Upload files
    uploaded_files = st.file_uploader(
        "Choose OMR sheet images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload one or more OMR sheet images"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Exam configuration
        st.subheader("Exam Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            exam_name = st.text_input("Exam Name", value="Midterm Exam")
            exam_date = st.date_input("Exam Date", value=datetime.now().date())
        
        with col2:
            total_questions = st.number_input("Total Questions", min_value=1, max_value=200, value=100)
            sheet_version = st.selectbox("Sheet Version", ["Set A", "Set B", "Set C", "Set D"])
        
        # Answer key input
        st.subheader("Answer Key")
        answer_key_method = st.radio(
            "How would you like to provide the answer key?",
            ["Upload JSON file", "Manual input", "Use existing exam"]
        )
        
        answer_key = {}
        
        if answer_key_method == "Upload JSON file":
            answer_key_file = st.file_uploader("Upload answer key JSON", type=['json'])
            if answer_key_file:
                try:
                    answer_key = json.load(answer_key_file)
                    st.success(f"Loaded answer key with {len(answer_key)} questions")
                except Exception as e:
                    st.error(f"Error loading answer key: {e}")
        
        elif answer_key_method == "Manual input":
            st.info("Enter answers for each question (A, B, C, D)")
            for i in range(1, min(total_questions + 1, 11)):  # Show first 10 for demo
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"Q{i}")
                with col2:
                    answer = st.selectbox(f"Answer for Q{i}", ["A", "B", "C", "D"], key=f"q{i}")
                    answer_key[f"Q{i}"] = answer
        
        elif answer_key_method == "Use existing exam":
            # Load existing exams from database
            try:
                with st.session_state.db.db_path as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, exam_name, answer_key FROM exams")
                    exams = cursor.fetchall()
                
                if exams:
                    exam_options = {f"{exam[1]} (ID: {exam[0]})": exam[0] for exam in exams}
                    selected_exam = st.selectbox("Select existing exam", list(exam_options.keys()))
                    
                    if selected_exam:
                        exam_id = exam_options[selected_exam]
                        # Load answer key
                        cursor.execute("SELECT answer_key FROM exams WHERE id = ?", (exam_id,))
                        answer_key = json.loads(cursor.fetchone()[0])
                        st.success(f"Loaded answer key from {selected_exam}")
                else:
                    st.info("No existing exams found.")
            except Exception as e:
                st.error(f"Error loading exams: {e}")
        
        # Process button
        if st.button("Process OMR Sheets", type="primary"):
            if not answer_key:
                st.error("Please provide an answer key first.")
                return
            
            # Process each uploaded file
            progress_bar = st.progress(0)
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the sheet
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = st.session_state.processor.process_omr_sheet(
                            temp_path, answer_key, sheet_version
                        )
                        result['filename'] = uploaded_file.name
                        results.append(result)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    results.append({
                        'filename': uploaded_file.name,
                        'error': str(e)
                    })
            
            # Display results
            st.subheader("Processing Results")
            
            if results:
                # Create results DataFrame
                df_data = []
                for result in results:
                    if 'error' not in result:
                        df_data.append({
                            'Filename': result['filename'],
                            'Total Score': result['total_score'],
                            'Percentage': f"{result['percentage']:.1f}%",
                            'Processing Time': f"{result['processing_time']:.2f}s",
                            'Model Confidence': f"{result['model_confidence']:.3f}"
                        })
                    else:
                        df_data.append({
                            'Filename': result['filename'],
                            'Total Score': 'Error',
                            'Percentage': 'Error',
                            'Processing Time': 'Error',
                            'Model Confidence': 'Error'
                        })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_results_page():
    """Results and evaluation page"""
    st.header("Evaluation Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        exam_filter = st.selectbox("Filter by Exam", ["All", "Recent", "Specific Exam"])
    
    with col2:
        score_filter = st.selectbox("Filter by Score Range", ["All", "High (80%+)", "Medium (60-80%)", "Low (<60%)"])
    
    with col3:
        date_filter = st.date_input("Filter by Date", value=None)
    
    # Load and display results
    try:
        recent_results = st.session_state.db.get_recent_evaluations(50)
        
        if recent_results:
            df = pd.DataFrame(recent_results)
            
            # Apply filters
            if exam_filter != "All":
                # Add exam filtering logic here
                pass
            
            if score_filter != "All":
                if score_filter == "High (80%+)":
                    df = df[df['percentage'] >= 80]
                elif score_filter == "Medium (60-80%)":
                    df = df[(df['percentage'] >= 60) & (df['percentage'] < 80)]
                elif score_filter == "Low (<60%)":
                    df = df[df['percentage'] < 60]
            
            # Display results
            st.dataframe(
                df[['student_id', 'name', 'total_score', 'percentage', 'evaluation_timestamp']],
                use_container_width=True
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Score", f"{df['percentage'].mean():.1f}%")
            with col2:
                st.metric("Highest Score", f"{df['percentage'].max():.1f}%")
            with col3:
                st.metric("Lowest Score", f"{df['percentage'].min():.1f}%")
            with col4:
                st.metric("Total Students", len(df))
        
        else:
            st.info("No evaluation results found.")
    
    except Exception as e:
        st.error(f"Error loading results: {e}")

def show_settings_page():
    """Settings and configuration page"""
    st.header("System Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    
    model_path = st.text_input(
        "Model Path",
        value="omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt",
        help="Path to the trained OMR model"
    )
    
    if st.button("Test Model"):
        if Path(model_path).exists():
            try:
                processor = OMRProcessor(model_path)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
        else:
            st.error("Model file not found!")
    
    # Database settings
    st.subheader("Database Configuration")
    
    db_path = st.text_input("Database Path", value="omr_evaluation.db")
    
    if st.button("Initialize Database"):
        try:
            db = OMRDatabase(db_path)
            st.success("Database initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing database: {e}")
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Python Version: {st.__version__}")
        st.info(f"Streamlit Version: {st.__version__}")
    
    with col2:
        st.info(f"Database Path: {db_path}")
        st.info(f"Model Path: {model_path}")

def show_analytics_page():
    """Analytics and reporting page"""
    st.header("Analytics & Reporting")
    
    # Load data
    try:
        recent_results = st.session_state.db.get_recent_evaluations(100)
        
        if not recent_results:
            st.info("No data available for analytics.")
            return
        
        df = pd.DataFrame(recent_results)
        
        # Score distribution
        st.subheader("Score Distribution")
        
        fig = px.histogram(
            df, 
            x='percentage', 
            nbins=20,
            title="Score Distribution",
            labels={'percentage': 'Score (%)', 'count': 'Number of Students'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance over time
        st.subheader("Performance Trends")
        
        df['date'] = pd.to_datetime(df['evaluation_timestamp']).dt.date
        daily_avg = df.groupby('date')['percentage'].mean().reset_index()
        
        fig = px.line(
            daily_avg, 
            x='date', 
            y='percentage',
            title="Average Score Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Subject-wise performance (if available)
        st.subheader("Subject-wise Performance")
        
        # This would need to be implemented based on your data structure
        st.info("Subject-wise analysis requires additional data structure.")
        
        # Export options
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"omr_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export to Excel"):
                # This would require openpyxl
                st.info("Excel export requires additional dependencies.")
    
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

if __name__ == "__main__":
    main()