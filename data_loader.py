"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
import datetime
from config import DATA_PATH


class DataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.df = None
    
    def load_data(self):
        """Load the dataset from CSV"""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst 5 Rows:")
        print(self.df.head())
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        print(f"Duplicate rows: {self.df.duplicated().sum()}")
    
    def convert_datetime(self):
        """Convert time columns to datetime and extract hours"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert to datetime
        self.df['Login Time'] = pd.to_datetime(self.df['Login Time'])
        self.df['Logout Time'] = pd.to_datetime(self.df['Logout Time'])
        
        # Extract hours
        self.df['Login Hour'] = self.df['Login Time'].dt.hour
        self.df['Logout Hour'] = self.df['Logout Time'].dt.hour
        
        print("Datetime conversion completed.")
        return self.df
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.describe()
    
    def get_login_logout_patterns(self):
        """Analyze login/logout hour patterns"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        login_counts = self.df['Login Hour'].value_counts().sort_index()
        logout_counts = self.df['Logout Hour'].value_counts().sort_index()
        
        return login_counts, logout_counts


def main():
    """Test the data loader functionality"""
    loader = DataLoader()
    df = loader.load_data()
    loader.basic_info()
    df_processed = loader.convert_datetime()
    print("\nSummary Statistics:")
    print(loader.get_summary_stats())
    
    login_counts, logout_counts = loader.get_login_logout_patterns()
    print("\nMost common login hours:")
    print(login_counts)
    print("\nMost common logout hours:")
    print(logout_counts)


if __name__ == "__main__":
    main()