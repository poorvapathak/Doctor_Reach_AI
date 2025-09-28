"""
Data visualization utilities for EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DataVisualizer:
    def __init__(self, save_plots=True, output_dir='plots'):
        self.save_plots = save_plots
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if self.save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_style("whitegrid")
    
    def plot_region_distribution(self, df):
        """Plot region distribution"""
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Region', hue='Region', data=df, palette='viridis', legend=False)
        plt.title('Region Distribution')
        plt.xlabel('Region')
        plt.ylabel('Count')
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/region_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_specialty_distribution(self, df):
        """Plot specialty distribution"""
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Speciality', hue='Speciality', data=df, palette='viridis', legend=False)
        plt.title('Speciality Distribution')
        plt.xlabel('Speciality')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/specialty_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df):
        """Plot correlation heatmap for numerical features"""
        numerical_cols = ['Usage Time (mins)', 'Count of Survey Attempts', 'Login Hour', 'Logout Hour']
        plt.figure(figsize=(12, 10))
        correlation = df[numerical_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_activity_by_hour(self, df):
        """Plot activity patterns by hour"""
        activity_by_hour = pd.DataFrame({
            'Login': df['Login Hour'].value_counts().sort_index(),
            'Logout': df['Logout Hour'].value_counts().sort_index()
        })
        activity_by_hour.fillna(0, inplace=True)
        
        plt.figure(figsize=(12, 6))
        activity_by_hour.plot(kind='bar')
        plt.title('Doctor Activity by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Doctors')
        plt.legend(['Login', 'Logout'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/activity_by_hour.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_usage_time_distribution(self, df):
        """Plot usage time distribution"""
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df['Usage Time (mins)'], bins=30, kde=True)
        plt.title('Usage Time Distribution')
        plt.xlabel('Usage Time (minutes)')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df['Usage Time (mins)'])
        plt.title('Usage Time Box Plot')
        plt.ylabel('Usage Time (minutes)')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/usage_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_survey_attempts_by_specialty(self, df):
        """Plot survey attempts by specialty"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Speciality', y='Count of Survey Attempts', data=df)
        plt.title('Survey Attempts by Specialty')
        plt.xlabel('Specialty')
        plt.ylabel('Count of Survey Attempts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/survey_attempts_by_specialty.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_survey_attempts_by_region(self, df):
        """Plot survey attempts by region"""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Region', y='Count of Survey Attempts', data=df)
        plt.title('Survey Attempts by Region')
        plt.xlabel('Region')
        plt.ylabel('Count of Survey Attempts')
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/survey_attempts_by_region.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_eda(self, df):
        """Create all visualization plots"""
        print("Creating comprehensive EDA visualizations...")
        
        self.plot_region_distribution(df)
        self.plot_specialty_distribution(df)
        self.plot_correlation_heatmap(df)
        self.plot_activity_by_hour(df)
        self.plot_usage_time_distribution(df)
        self.plot_survey_attempts_by_specialty(df)
        self.plot_survey_attempts_by_region(df)
        
        if self.save_plots:
            print(f"All plots saved in '{self.output_dir}' directory.")


def main():
    """Test visualization functionality"""
    from data_loader import DataLoader
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    df = loader.convert_datetime()
    
    # Create visualizations
    visualizer = DataVisualizer(save_plots=True)
    visualizer.create_comprehensive_eda(df)


if __name__ == "__main__":
    main()