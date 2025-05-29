import streamlit as st
import pandas as pd
import numpy as np
from model import FraudDetectionModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import io
from PIL import Image
import os
from typing import Optional, Tuple, Dict, Any
import logging
from datetime import datetime
import json
import joblib
import time
from pathlib import Path

# Configure logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'fraud_detection_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_metadata' not in st.session_state:
    st.session_state.model_metadata = None

# Load the model
@st.cache_resource
def load_model() -> Optional[FraudDetectionModel]:
    """
    Cache the model loading to prevent reloading on every interaction.
    
    Returns:
        Optional[FraudDetectionModel]: Loaded model or None if loading fails
    """
    try:
        model = FraudDetectionModel()
        model_path = 'bilstm_fraud_detection_v1.0.0'
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            st.error(f"Model directory not found at {model_path}")
            return None
            
        if model.load_model(model_path):
            st.success("Model loaded successfully!")
            return model
        return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the scaler
@st.cache_resource
def load_scaler():
    scaler_path = Path("models/scaler.joblib")
    if not scaler_path.exists():
        st.error("Scaler file not found! Please ensure the scaler is saved in the models directory.")
        return None
    return joblib.load(scaler_path)

def process_data_in_batches(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Process data in batches to prevent memory issues.
    
    Args:
        df: Input DataFrame
        batch_size: Size of each batch
        
    Yields:
        DataFrame: Batch of data
    """
    total_rows = len(df)
    for i in range(0, total_rows, batch_size):
        yield df.iloc[i:min(i + batch_size, total_rows)]

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        plt.Figure: Confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def display_model_info() -> None:
    """Display model information in the sidebar."""
    with st.sidebar:
        st.title("Model Information")
        
        # Display model metadata if available
        if st.session_state.model_metadata:
            st.subheader("Model Version")
            st.write(f"Version: {st.session_state.model_metadata['version']}")
            st.write(f"Last Updated: {st.session_state.model_metadata['last_updated']}")
            
            if 'performance_metrics' in st.session_state.model_metadata:
                st.subheader("Performance Metrics")
                metrics = st.session_state.model_metadata['performance_metrics']
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value:.4f}")
        
        st.markdown("""
        ### Model Architecture
        - BiLSTM with 64 and 32 units
        - Dropout Regularization
        - Dense layers with ReLU activation
        - Sigmoid Output
        """)
        
        try:
            st.image("model_architecture.png", caption="BiLSTM Model Architecture", use_column_width=True)
        except Exception as e:
            logger.warning(f"Model architecture image not found: {str(e)}")
            st.warning("Model architecture image not found")

def process_uploaded_file(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Process the uploaded CSV file.
    
    Args:
        uploaded_file: Uploaded file object
        
    Returns:
        Optional[pd.DataFrame]: Processed DataFrame or None if processing fails
    """
    try:
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            
            # Basic data validation
            if len(df) == 0:
                st.error("Uploaded file is empty")
                return None
                
            st.session_state.data = df
            return df
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the input data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        expected_features = 30  # V1-V28, Time, Amount
        if len(df.columns) != expected_features and 'Class' not in df.columns:
            st.error(f"Input data must have exactly {expected_features} features (V1-V28, Time, Amount)")
            return False
            
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"Found {missing_values} missing values in the dataset")
            
        # Check for infinite values
        inf_values = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        if inf_values > 0:
            st.warning(f"Found {inf_values} infinite values in the dataset")
            
        return True
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        st.error(f"Error validating data: {str(e)}")
        return False

def display_results(results_df: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> None:
    """
    Display results in tabs.
    
    Args:
        results_df: DataFrame containing results
        y_true: Optional true labels for evaluation
    """
    tab1, tab2, tab3 = st.tabs(["Results", "Evaluation", "Download"])
    
    with tab1:
        st.subheader("Detection Results")
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(results_df))
        with col2:
            fraud_count = results_df['Fraud_Prediction'].sum()
            st.metric("Fraudulent Transactions", fraud_count)
        with col3:
            fraud_percentage = (fraud_count / len(results_df)) * 100
            st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
        
        # Display results table
        st.dataframe(results_df)
        
        # Add a pie chart of fraud vs non-fraud predictions
        fraud_counts = results_df['Fraud_Prediction'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(fraud_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'])
        ax.set_title('Distribution of Predictions')
        st.pyplot(fig)
    
    with tab2:
        if y_true is not None:
            st.subheader("Model Evaluation")
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(y_true, results_df['Fraud_Prediction'])
            st.pyplot(fig)
            
            # Display classification report
            report = classification_report(y_true, results_df['Fraud_Prediction'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Add ROC curve if probabilities are available
            if 'Fraud_Probability' in results_df.columns:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, results_df['Fraud_Probability'])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
        else:
            st.info("No ground truth labels available for evaluation")
    
    with tab3:
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

def main() -> None:
    """Main function to run the Streamlit app."""
    # Display model information
    display_model_info()

    # Main content
    st.title("Credit Card Fraud Detection System")
    st.write("Upload a CSV file containing credit card transaction data for fraud detection.")
    
    # Initialize model if not already loaded
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
            if st.session_state.model is None:
                st.error("""
                Model not found. Please ensure the following directory structure is present:
                bilstm_fraud_detection_v1.0.0/
                ├── model/
                │   ├── saved_model.pb
                │   ├── keras_metadata.pb
                │   └── variables/
                ├── scaler.npy
                └── metadata.json
                
                If you're deploying on Streamlit Cloud, make sure these files are included in your GitHub repository.
                """)
                return
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Process uploaded file
        df = process_uploaded_file(uploaded_file)
        if df is None:
            return
            
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Validate data
        if not validate_data(df):
            return
        
        # Remove Class column if it exists
        y_true = df['Class'] if 'Class' in df.columns else None
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        
        # Create two columns for threshold and visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Threshold slider
            threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)
            st.write("""
            - Lower threshold: More sensitive to fraud (higher false positives)
            - Higher threshold: More strict fraud detection (higher false negatives)
            """)
        
        with col2:
            st.write("Real-time Fraud Detection Visualization")
            st.write("""
            The model will process your data in batches to prevent memory issues.
            You can monitor the progress in the progress bar below.
            """)
        
        # Make predictions
        if st.button("Detect Fraud"):
            with st.spinner("Processing..."):
                try:
                    # Process data in batches
                    all_predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, batch_df in enumerate(process_data_in_batches(df)):
                        batch_predictions = st.session_state.model.predict(batch_df.values)
                        all_predictions.extend(batch_predictions)
                        # Calculate progress as a fraction between 0 and 1
                        progress = min(1.0, (i + 1) * 1000 / len(df))
                        progress_bar.progress(progress)
                    
                    predictions_prob = np.array(all_predictions)
                    predictions = (predictions_prob > threshold).astype(int)
                    st.session_state.predictions = predictions
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Fraud_Probability'] = predictions_prob
                    results_df['Fraud_Prediction'] = predictions
                    
                    # Display results
                    display_results(results_df, y_true)
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 
