import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import logging 
from typing import Tuple, Optional, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self, model_version: str = "1.0.0"):
        self.scaler = StandardScaler()
        self.model = None
        self.model_version = model_version
        self.model_metadata = {
            "version": model_version,
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "performance_metrics": {}
        }
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input dataframe."""
        try:
            # Validate input
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            # Drop duplicates and null values
            df = df.drop_duplicates()
            df = df.dropna()
            
            if len(df) == 0:
                raise ValueError("No valid data after preprocessing")
            
            # Separate features and target
            if 'Class' not in df.columns:
                raise ValueError("Target column 'Class' not found in data")
                
            X = df.drop('Class', axis=1)
            y = df['Class']
            
            # Validate feature count
            expected_features = 30  # V1-V28, Time, Amount
            if X.shape[1] != expected_features:
                raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Reshape for BiLSTM (samples, time_steps, features)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            
            return X_reshaped, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance the dataset."""
        try:
            smote = SMOTE(random_state=42)
            X_reshaped = X.reshape(X.shape[0], X.shape[2])
            X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
            return X_balanced.reshape(X_balanced.shape[0], 1, X_balanced.shape[1]), y_balanced
        except Exception as e:
            logger.error(f"Error in data balancing: {str(e)}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build the BiLSTM model."""
        try:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.AUC()]
            )
            
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Error in model building: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray, 
             epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train the model with callbacks."""
        try:
            callbacks = [
                ModelCheckpoint(
                    f'bilstm_fraud_detection_v{self.model_version}',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    save_format='tf'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Update metadata
            self.model_metadata["last_updated"] = datetime.now().isoformat()
            self.model_metadata["performance_metrics"] = {
                "final_val_loss": float(history.history['val_loss'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1])
            }
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Validate input shape
            if len(X.shape) != 2:
                raise ValueError(f"Expected 2D input array, got shape {X.shape}")
            
            # Ensure input is properly shaped
            X_scaled = self.scaler.transform(X)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            
            # Convert input to tensor and make prediction
            input_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)
            predictions = self.model(input_tensor)
            
            # Extract predictions from the output dictionary
            return predictions['dense_1'].numpy()
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def save_model(self, path: str = None) -> None:
        """Save the model, scaler, and metadata."""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            if path is None:
                path = f'bilstm_fraud_detection_v{self.model_version}'
            
            # Create model directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model using SavedModel format
            tf.saved_model.save(self.model, os.path.join(path, 'model'))
            
            # Save scaler
            scaler_path = os.path.join(path, 'scaler.npy')
            np.save(scaler_path, self.scaler)
            
            # Save metadata
            metadata_path = os.path.join(path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=4)
                
            logger.info(f"Model saved successfully at {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str = None) -> bool:
        """Load the model, scaler, and metadata."""
        try:
            if path is None:
                path = f'bilstm_fraud_detection_v{self.model_version}'
            
            model_path = os.path.join(path, 'model')
            if not os.path.exists(model_path):
                logger.error(f"Model directory not found at {model_path}")
                return False
            
            # Load model using SavedModel format and get the concrete function
            loaded_model = tf.saved_model.load(model_path)
            self.model = loaded_model.signatures['serving_default']
            
            # Load scaler
            scaler_path = os.path.join(path, 'scaler.npy')
            if os.path.exists(scaler_path):
                self.scaler = np.load(scaler_path, allow_pickle=True).item()
            else:
                logger.warning(f"Scaler file not found at {scaler_path}")
            
            # Load metadata
            metadata_path = os.path.join(path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            else:
                logger.warning(f"Metadata file not found at {metadata_path}")
            
            logger.info(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False 