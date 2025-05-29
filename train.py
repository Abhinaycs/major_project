import logging
from model import FraudDetectionModel
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

# Configure logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_model():
    try:
        logger.info("Starting model training process...")
        
        # Check if dataset exists
        if not os.path.exists('creditcard.csv'):
            raise FileNotFoundError("Dataset file 'creditcard.csv' not found")
        
        # Load dataset
        logger.info("Loading dataset...")
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Initialize model
        logger.info("Initializing model...")
        model = FraudDetectionModel(model_version="1.0.0")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X, y = model.preprocess_data(df)
        logger.info(f"Data preprocessed. Features shape: {X.shape}")
        
        # Balance data
        logger.info("Balancing dataset...")
        X_balanced, y_balanced = model.balance_data(X, y)
        logger.info(f"Data balanced. New shape: {X_balanced.shape}")
        
        # Split data
        logger.info("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.2, 
            random_state=42,
            stratify=y_balanced
        )
        logger.info(f"Data split complete. Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
        
        # Build model
        logger.info("Building model architecture...")
        model.build_model(input_shape=(1, 30))
        
        # Train model
        logger.info("Starting model training...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=32
        )
        
        # Create model directory if it doesn't exist
        model_dir = 'bilstm_fraud_detection_v1.0.0'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        logger.info("Saving model and artifacts...")
        model.save_model(model_dir)
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()