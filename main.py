"""
Main script to run the complete Doctor Survey Targeting pipeline
"""

import argparse
import os
import sys
from datetime import datetime

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from visualizer import DataVisualizer
from model_trainer import ModelTrainer
from predictor import DoctorPredictor, PredictionScheduler


def run_eda(data_path='data/dummy_npi_data.csv', create_plots=True):
    """Run exploratory data analysis"""
    print("="*60)
    print("RUNNING EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    loader = DataLoader(data_path)
    df = loader.load_data()
    loader.basic_info()
    df = loader.convert_datetime()
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print(loader.get_summary_stats())
    
    # Show login/logout patterns
    login_counts, logout_counts = loader.get_login_logout_patterns()
    print("\nLogin/Logout Patterns:")
    print("Most common login hours:")
    print(login_counts.head())
    print("Most common logout hours:")
    print(logout_counts.head())
    
    # Create visualizations
    if create_plots:
        visualizer = DataVisualizer(save_plots=True)
        visualizer.create_comprehensive_eda(df)
    
    return df


def run_training(df, retrain=False):
    """Run model training pipeline"""
    print("="*60)
    print("RUNNING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Check if model already exists
    from config import MODEL_PATH
    
    if os.path.exists(MODEL_PATH) and not retrain:
        print(f"Model already exists at {MODEL_PATH}")
        print("Use --retrain flag to retrain the model")
        return
    
    # Feature engineering
    engineer = FeatureEngineer()
    X, y, expanded_df = engineer.process_all_features(df)
    
    # Train model
    trainer = ModelTrainer()
    metrics = trainer.train_full_pipeline(X, y)
    
    print("\nTraining completed successfully!")
    print(f"Final metrics: {metrics}")
    
    return trainer


def run_predictions(df, target_hour=None, top_k=20):
    """Run predictions for doctor targeting"""
    print("="*60)
    print("RUNNING PREDICTION PIPELINE")
    print("="*60)
    
    # Feature engineering (without expanding dataset)
    engineer = FeatureEngineer()
    df_processed = engineer.create_cyclic_features(df)
    df_processed = engineer.create_aggregate_features(df_processed)
    
    # Add placeholder target features (will be updated during prediction)
    df_processed['Target_Sin_Hour'] = 0.0
    df_processed['Target_Cos_Hour'] = 0.0
    
    # Initialize predictor
    predictor = DoctorPredictor()
    predictor.load_model()
    predictor.prepare_original_data(df_processed)
    
    # If no specific hour provided, use current hour
    if target_hour is None:
        target_hour = datetime.now().hour
        print(f"No target hour specified. Using current hour: {target_hour}")
    
    # Make predictions
    print(f"\nPredicting top {top_k} doctors for hour {target_hour}:00")
    top_doctors = predictor.predict_for_hour(target_hour, top_k)
    
    # Display results
    print(top_doctors)
    
    # Save predictions
    from config import OUTPUT_PATH
    predictor.save_predictions(top_doctors, OUTPUT_PATH)
    
    # Get insights
    insights = predictor.get_activity_insights(target_hour)
    print(f"\nActivity Insights for hour {target_hour}:")
    print(f"Average probability: {insights['avg_probability']:.3f}")
    print(f"Probability range: {insights['probability_range']['min']:.3f} - {insights['probability_range']['max']:.3f}")
    print(f"Top regions: {insights['top_regions']}")
    print(f"Top specialties: {insights['top_specialties']}")
    
    return predictor, top_doctors


def run_campaign_analysis(df):
    """Run campaign scheduling analysis"""
    print("="*60)
    print("RUNNING CAMPAIGN ANALYSIS")
    print("="*60)
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_processed = engineer.create_cyclic_features(df)
    df_processed = engineer.create_aggregate_features(df_processed)
    
    # Add placeholder target features
    df_processed['Target_Sin_Hour'] = 0.0
    df_processed['Target_Cos_Hour'] = 0.0
    
    # Initialize predictor and scheduler
    predictor = DoctorPredictor()
    predictor.load_model()
    predictor.prepare_original_data(df_processed)
    
    scheduler = PredictionScheduler(predictor)
    
    # Peak hours analysis
    print("Analyzing peak hours...")
    peak_hours = scheduler.get_peak_hours_analysis()
    print("\nTop 10 Peak Hours:")
    print(peak_hours.head(10))
    
    # Campaign schedule
    print("\nCreating optimal campaign schedule...")
    campaign_schedule = scheduler.create_campaign_schedule(campaign_hours=5)
    
    print("\nOptimal Campaign Schedule:")
    for hour, details in campaign_schedule.items():
        insights = details['insights']
        priority = details['campaign_priority']
        print(f"Hour {hour:2d}:00 | Priority: {priority:6s} | Avg Prob: {insights['avg_probability']:.3f} | Top Doctors: {len(details['recommended_doctors'])}")
    
    # Save campaign schedule
    campaign_output = 'output/campaign_schedule.csv'
    all_recommendations = []
    
    for hour, details in campaign_schedule.items():
        doctors = details['recommended_doctors'].copy()
        doctors['Campaign_Hour'] = hour
        doctors['Priority'] = details['campaign_priority']
        all_recommendations.append(doctors)
    
    if all_recommendations:
        import pandas as pd
        campaign_df = pd.concat(all_recommendations, ignore_index=True)
        campaign_df.to_csv(campaign_output, index=False)
        print(f"\nCampaign schedule saved to: {campaign_output}")
    
    return scheduler, campaign_schedule


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Doctor Survey Targeting ML Pipeline')
    
    # Add arguments
    parser.add_argument('--data', default='data/dummy_npi_data.csv', 
                       help='Path to the data file')
    parser.add_argument('--mode', choices=['eda', 'train', 'predict', 'campaign', 'full'], 
                       default='full', help='Mode to run the pipeline')
    parser.add_argument('--retrain', action='store_true', 
                       help='Force retrain the model even if it exists')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip creating plots in EDA mode')
    parser.add_argument('--hour', type=int, choices=range(24), 
                       help='Target hour for predictions (0-23)')
    parser.add_argument('--top-k', type=int, default=20, 
                       help='Number of top doctors to predict')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        print("Please ensure the dummy_npi_data.csv file is in the data/ directory")
        return
    
    try:
        if args.mode == 'eda':
            run_eda(args.data, create_plots=not args.no_plots)
            
        elif args.mode == 'train':
            df = run_eda(args.data, create_plots=False)
            run_training(df, retrain=args.retrain)
            
        elif args.mode == 'predict':
            df = run_eda(args.data, create_plots=False)
            run_predictions(df, target_hour=args.hour, top_k=args.top_k)
            
        elif args.mode == 'campaign':
            df = run_eda(args.data, create_plots=False)
            run_campaign_analysis(df)
            
        elif args.mode == 'full':
            # Run complete pipeline
            print("Running complete Doctor Survey Targeting pipeline...")
            
            # 1. EDA
            df = run_eda(args.data, create_plots=not args.no_plots)
            
            # 2. Training
            run_training(df, retrain=args.retrain)
            
            # 3. Predictions
            run_predictions(df, target_hour=args.hour, top_k=args.top_k)
            
            # 4. Campaign Analysis
            run_campaign_analysis(df)
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Check the following directories for outputs:")
            print("- plots/: EDA visualizations")
            print("- models/: Trained model")
            print("- output/: Predictions and campaign schedule")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)