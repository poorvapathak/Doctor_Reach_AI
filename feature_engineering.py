"""
Feature engineering utilities for doctor survey targeting
"""

import pandas as pd
import numpy as np
import datetime
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


class FeatureEngineer:
    def __init__(self):
        self.expanded_df = None
    
    def create_cyclic_features(self, df):
        """Create cyclic features for time-based columns"""
        df_copy = df.copy()
        
        # Create cyclic features for login and logout hours
        df_copy['Login_Sin_Hour'] = np.sin(2 * np.pi * df_copy['Login Hour'] / 24)
        df_copy['Login_Cos_Hour'] = np.cos(2 * np.pi * df_copy['Login Hour'] / 24)
        df_copy['Logout_Sin_Hour'] = np.sin(2 * np.pi * df_copy['Logout Hour'] / 24)
        df_copy['Logout_Cos_Hour'] = np.cos(2 * np.pi * df_copy['Logout Hour'] / 24)
        
        return df_copy
    
    def create_aggregate_features(self, df):
        """Create doctor-level aggregate features"""
        df_copy = df.copy()
        
        # Specialty-based aggregates
        df_copy['Avg_Usage_Time'] = df_copy.groupby('Speciality')['Usage Time (mins)'].transform('mean')
        df_copy['Std_Login_Hour'] = df_copy.groupby('Speciality')['Login Hour'].transform('std')
        
        # Region-based aggregates
        df_copy['Avg_Attempts_By_Region'] = df_copy.groupby('Region')['Count of Survey Attempts'].transform('mean')
        
        # Historical attendance rate (proxy)
        mean_usage = df_copy['Usage Time (mins)'].mean()
        df_copy['Historical_Attendance_Rate'] = (
            df_copy.groupby('Region')['Count of Survey Attempts'].transform('mean') / mean_usage
        )
        
        return df_copy
    
    def expand_for_hourly_prediction(self, df):
        """
        Expand dataset for binary classification - each doctor for each hour
        Target: 1 if doctor session overlaps with hour window (±30 mins)
        """
        expanded_rows = []
        
        for _, row in df.iterrows():
            for hour in range(24):
                # Define hour window (±30 minutes around the target hour)
                window_start = row['Login Time'].replace(
                    hour=hour, minute=0, second=0
                ) - datetime.timedelta(minutes=30)
                window_end = window_start + datetime.timedelta(minutes=60)
                
                # Check if session overlaps with window
                active = (row['Login Time'] < window_end) and (row['Logout Time'] > window_start)
                
                # Create new row
                new_row = row.copy()
                new_row['Target_Hour'] = hour
                new_row['Active'] = 1 if active else 0
                new_row['Target_Sin_Hour'] = np.sin(2 * np.pi * hour / 24)
                new_row['Target_Cos_Hour'] = np.cos(2 * np.pi * hour / 24)
                
                expanded_rows.append(new_row)
        
        self.expanded_df = pd.DataFrame(expanded_rows)
        
        print(f"Expanded dataset shape: {self.expanded_df.shape}")
        print("Class distribution (Active=1):")
        print(self.expanded_df['Active'].value_counts(normalize=True))
        
        return self.expanded_df
    
    def get_features_and_target(self, df=None):
        """Extract features and target variable"""
        if df is None:
            if self.expanded_df is None:
                raise ValueError("No expanded dataset available. Run expand_for_hourly_prediction first.")
            df = self.expanded_df
        
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df['Active']
        
        return X, y
    
    def process_all_features(self, df):
        """Apply all feature engineering steps"""
        print("Creating cyclic features...")
        df = self.create_cyclic_features(df)
        
        print("Creating aggregate features...")
        df = self.create_aggregate_features(df)
        
        print("Expanding dataset for hourly prediction...")
        expanded_df = self.expand_for_hourly_prediction(df)
        
        print("Extracting features and target...")
        X, y = self.get_features_and_target()
        
        return X, y, expanded_df


def main():
    """Test feature engineering functionality"""
    from data_loader import DataLoader
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    df = loader.convert_datetime()
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    X, y, expanded_df = engineer.process_all_features(df)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")


if __name__ == "__main__":
    main()