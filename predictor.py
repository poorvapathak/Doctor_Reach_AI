"""
Prediction and recommendation system for doctor survey targeting
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from config import (
    MODEL_PATH, OUTPUT_PATH, TOP_K_DOCTORS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)


class DoctorPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.original_df = None
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(OUTPUT_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded successfully from: {self.model_path}")
        return self.model
    
    def prepare_original_data(self, df):
        """Store the original processed dataframe for predictions"""
        # Ensure all required features are present
        required_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        available_features = set(df.columns)
        
        missing_features = required_features - available_features
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        self.original_df = df.copy()
        print("Original data prepared for predictions.")
    
    def predict_for_hour(self, hour, top_k=TOP_K_DOCTORS):
        """
        Predict top doctors likely to be active at a specific hour
        
        Args:
            hour (int): Target hour (0-23)
            top_k (int): Number of top doctors to return
            
        Returns:
            pandas.DataFrame: Top doctors with their prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.original_df is None:
            raise ValueError("Original data not prepared. Call prepare_original_data() first.")
        
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23")
        
        # Create prediction data for the specific hour
        df_pred = self.original_df.copy()
        df_pred['Target_Hour'] = hour
        df_pred['Target_Sin_Hour'] = np.sin(2 * np.pi * hour / 24)
        df_pred['Target_Cos_Hour'] = np.cos(2 * np.pi * hour / 24)
        
        # Select features for prediction
        X_pred = df_pred[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        
        # Make predictions
        probabilities = self.model.predict_proba(X_pred)[:, 1]
        
        # Add probabilities to dataframe
        df_pred['Probability'] = probabilities
        
        # Sort by probability and get top K
        top_doctors = df_pred.sort_values('Probability', ascending=False).head(top_k)
        
        # Select relevant columns for output
        result_columns = ['NPI', 'State', 'Region', 'Speciality', 'Probability']
        top_doctors_result = top_doctors[result_columns].copy()
        
        print(f"Predicted top {top_k} doctors for hour {hour}:00")
        print(f"Probability range: {top_doctors_result['Probability'].min():.3f} - {top_doctors_result['Probability'].max():.3f}")
        
        return top_doctors_result
    
    def predict_for_multiple_hours(self, hours, top_k=TOP_K_DOCTORS):
        """
        Predict top doctors for multiple hours
        
        Args:
            hours (list): List of target hours
            top_k (int): Number of top doctors to return per hour
            
        Returns:
            dict: Dictionary with hour as key and top doctors DataFrame as value
        """
        results = {}
        
        for hour in hours:
            try:
                results[hour] = self.predict_for_hour(hour, top_k)
            except Exception as e:
                print(f"Error predicting for hour {hour}: {str(e)}")
                continue
        
        return results
    
    def get_best_hours_for_doctor(self, npi, hours_to_check=None):
        """
        Find the best hours to target a specific doctor
        
        Args:
            npi (str/int): Doctor's NPI identifier
            hours_to_check (list): Hours to check (default: all 24 hours)
            
        Returns:
            pandas.DataFrame: Hours sorted by prediction probability for the doctor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.original_df is None:
            raise ValueError("Original data not prepared. Call prepare_original_data() first.")
        
        if hours_to_check is None:
            hours_to_check = list(range(24))
        
        # Check if doctor exists in original data
        doctor_data = self.original_df[self.original_df['NPI'] == npi]
        if doctor_data.empty:
            raise ValueError(f"Doctor with NPI {npi} not found in the dataset")
        
        results = []
        
        for hour in hours_to_check:
            # Create prediction data for this doctor at this hour
            df_pred = doctor_data.copy()
            df_pred['Target_Hour'] = hour
            df_pred['Target_Sin_Hour'] = np.sin(2 * np.pi * hour / 24)
            df_pred['Target_Cos_Hour'] = np.cos(2 * np.pi * hour / 24)
            
            # Select features for prediction
            X_pred = df_pred[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            
            # Make prediction
            probability = self.model.predict_proba(X_pred)[:, 1][0]
            
            results.append({
                'Hour': hour,
                'Probability': probability
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Probability', ascending=False)
        
        return results_df
    
    def save_predictions(self, predictions_df, filepath=OUTPUT_PATH):
        """Save predictions to CSV file"""
        predictions_df.to_csv(filepath, index=False)
        print(f"Predictions saved to: {filepath}")
    
    def get_activity_insights(self, hour):
        """Get insights about doctor activity patterns for a specific hour"""
        if self.original_df is None:
            raise ValueError("Original data not prepared. Call prepare_original_data() first.")
        
        # Get predictions for this hour
        top_doctors = self.predict_for_hour(hour, top_k=100)  # Get more for analysis
        
        # Analyze patterns
        region_dist = top_doctors['Region'].value_counts()
        specialty_dist = top_doctors['Speciality'].value_counts()
        
        insights = {
            'hour': hour,
            'total_predicted_active': len(top_doctors),
            'avg_probability': top_doctors['Probability'].mean(),
            'top_regions': region_dist.head(3).to_dict(),
            'top_specialties': specialty_dist.head(3).to_dict(),
            'probability_range': {
                'min': top_doctors['Probability'].min(),
                'max': top_doctors['Probability'].max()
            }
        }
        
        return insights


class PredictionScheduler:
    """Helper class for scheduling predictions and recommendations"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def get_current_hour_recommendations(self):
        """Get recommendations for the current hour"""
        current_hour = datetime.now().hour
        return self.predictor.predict_for_hour(current_hour)
    
    def get_next_hour_recommendations(self):
        """Get recommendations for the next hour"""
        next_hour = (datetime.now().hour + 1) % 24
        return self.predictor.predict_for_hour(next_hour)
    
    def get_peak_hours_analysis(self):
        """Analyze peak hours for doctor activity"""
        all_hours = list(range(24))
        hour_scores = []
        
        for hour in all_hours:
            top_doctors = self.predictor.predict_for_hour(hour, top_k=50)
            avg_probability = top_doctors['Probability'].mean()
            hour_scores.append({
                'hour': hour,
                'avg_probability': avg_probability,
                'top_probability': top_doctors['Probability'].max()
            })
        
        peak_hours_df = pd.DataFrame(hour_scores)
        peak_hours_df = peak_hours_df.sort_values('avg_probability', ascending=False)
        
        return peak_hours_df
    
    def create_campaign_schedule(self, campaign_hours=5):
        """
        Create an optimal schedule for survey campaigns
        
        Args:
            campaign_hours (int): Number of hours to schedule campaigns
            
        Returns:
            dict: Campaign schedule with recommendations
        """
        peak_hours = self.get_peak_hours_analysis()
        top_hours = peak_hours.head(campaign_hours)['hour'].tolist()
        
        campaign_schedule = {}
        
        for hour in top_hours:
            doctors = self.predictor.predict_for_hour(hour)
            insights = self.predictor.get_activity_insights(hour)
            
            campaign_schedule[hour] = {
                'recommended_doctors': doctors,
                'insights': insights,
                'campaign_priority': 'High' if insights['avg_probability'] > 0.7 else 'Medium'
            }
        
        return campaign_schedule


def main():
    """Test prediction functionality"""
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    df = loader.convert_datetime()
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    df_processed = engineer.create_cyclic_features(df)
    df_processed = engineer.create_aggregate_features(df_processed)
    
    # Initialize predictor
    predictor = DoctorPredictor()
    
    try:
        # Try to load existing model
        predictor.load_model()
    except FileNotFoundError:
        print("No trained model found. Training new model...")
        # Train new model if not exists
        X, y, _ = engineer.process_all_features(df)
        trainer = ModelTrainer()
        trainer.train_full_pipeline(X, y)
        predictor.load_model()
    
    # Prepare data for predictions
    predictor.prepare_original_data(df_processed)
    
    # Test predictions for different hours
    test_hours = [9, 13, 17]
    
    for hour in test_hours:
        print(f"\n{'='*50}")
        print(f"PREDICTIONS FOR HOUR {hour}:00")
        print('='*50)
        
        top_doctors = predictor.predict_for_hour(hour)
        print(top_doctors.head(10))
        
        # Save predictions
        output_file = f'output/top_doctors_hour_{hour}.csv'
        predictor.save_predictions(top_doctors, output_file)
        
        # Get insights
        insights = predictor.get_activity_insights(hour)
        print(f"\nInsights for hour {hour}:")
        print(f"Average probability: {insights['avg_probability']:.3f}")
        print(f"Top regions: {insights['top_regions']}")
        print(f"Top specialties: {insights['top_specialties']}")


if __name__ == "__main__":
    main()