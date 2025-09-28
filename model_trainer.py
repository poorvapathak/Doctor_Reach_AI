"""
Simplified model training and evaluation utilities
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, classification_report
from xgboost import XGBClassifier

from config import (
    RANDOM_STATE, TEST_SIZE, 
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_PATH, PRECISION_K
)


class ModelTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.pipeline = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Create models directory if it doesn't exist
        model_dir = os.path.dirname(MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERIC_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
            ]
        )
        return self.preprocessor
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set class distribution:")
        print(self.y_train.value_counts(normalize=True))
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_model_pipeline(self):
        """Create model pipeline with preprocessor and classifier"""
        if self.preprocessor is None:
            self.create_preprocessor()
        
        # Simple XGBoost without early stopping for GridSearchCV
        self.model = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='auc',
            verbosity=0  # Reduce verbosity
        )
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        return self.pipeline
    
    def train_simple_model(self, X_train=None, y_train=None):
        """Train a simple model without hyperparameter tuning"""
        if X_train is None or y_train is None:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Training data not available. Run split_data first.")
            X_train, y_train = self.X_train, self.y_train
        
        if self.pipeline is None:
            self.create_model_pipeline()
        
        print("Training XGBoost model...")
        self.pipeline.fit(X_train, y_train)
        self.best_model = self.pipeline
        
        print("Model training completed!")
        return self.best_model
    
    def evaluate_model(self, X_test=None, y_test=None):
        """Evaluate the trained model"""
        if X_test is None or y_test is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Test data not available. Run split_data first.")
            X_test, y_test = self.X_test, self.y_test
        
        if self.best_model is None:
            raise ValueError("Model not trained. Run train_simple_model first.")
        
        # Predictions
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        # Precision@K
        top_k_indices = np.argsort(y_pred_proba)[-PRECISION_K:]
        precision_at_k = precision_score(y_test.iloc[top_k_indices], y_pred[top_k_indices])
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"AUC Score: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Precision@{PRECISION_K}: {precision_at_k:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'precision_at_k': precision_at_k,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_model(self, filepath=MODEL_PATH):
        """Save the trained model to disk"""
        if self.best_model is None:
            raise ValueError("Model not trained. Run train_simple_model first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"Model saved successfully to: {filepath}")
    
    def load_model(self, filepath=MODEL_PATH):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.best_model = pickle.load(f)
        
        print(f"Model loaded successfully from: {filepath}")
        return self.best_model
    
    def train_full_pipeline(self, X, y):
        """Train the complete pipeline from scratch"""
        print("Starting simplified training pipeline...")
        
        # Split data
        self.split_data(X, y)
        
        # Create and train model (simplified - no grid search)
        self.train_simple_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Save model
        self.save_model()
        
        print("Training pipeline completed successfully!")
        return metrics


def main():
    """Test model training functionality"""
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    df = loader.convert_datetime()
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    X, y, _ = engineer.process_all_features(df)
    
    # Train model
    trainer = ModelTrainer()
    metrics = trainer.train_full_pipeline(X, y)
    
    print(f"\nFinal model metrics: {metrics}")


if __name__ == "__main__":
    main()