"""
Simplified main script for testing the pipeline
"""

import os
import sys

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'output', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Directory created: {dir_name}")

def test_basic_imports():
    """Test basic imports step by step"""
    print("Testing imports...")
    
    try:
        print("1. Testing config...")
        import config
        print("   ✓ config imported")
        
        print("2. Testing data_loader...")
        from data_loader import DataLoader
        print("   ✓ DataLoader imported")
        
        print("3. Testing feature_engineering...")
        from feature_engineering import FeatureEngineer  
        print("   ✓ FeatureEngineer imported")
        
        print("4. Testing visualizer...")
        from visualizer import DataVisualizer
        print("   ✓ DataVisualizer imported")
        
        print("5. Testing model_trainer...")
        from model_trainer import ModelTrainer
        print("   ✓ ModelTrainer imported")
        
        print("6. Testing predictor...")
        from predictor import DoctorPredictor, PredictionScheduler
        print("   ✓ DoctorPredictor and PredictionScheduler imported")
        
        return True
        
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Other error: {e}")
        return False

def run_basic_pipeline():
    """Run a basic version of the pipeline"""
    print("\n" + "="*50)
    print("RUNNING BASIC PIPELINE")
    print("="*50)
    
    try:
        # Import modules
        from data_loader import DataLoader
        from feature_engineering import FeatureEngineer
        
        # Check if data file exists
        data_file = 'data/dummy_npi_data.csv'
        if not os.path.exists(data_file):
            print(f"✗ Data file not found: {data_file}")
            print("Please place your dummy_npi_data.csv file in the data/ directory")
            return False
        
        # Load data
        print("Loading data...")
        loader = DataLoader(data_file)
        df = loader.load_data()
        loader.basic_info()
        
        print("Converting datetime...")
        df = loader.convert_datetime()
        
        print("Creating basic features...")
        engineer = FeatureEngineer()
        df_processed = engineer.create_cyclic_features(df)
        df_processed = engineer.create_aggregate_features(df_processed)
        
        print("✓ Basic pipeline completed successfully!")
        print(f"Final dataset shape: {df_processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("DOCTOR SURVEY TARGETING - SIMPLIFIED TEST")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_basic_imports():
        print("✗ Import test failed. Please check your files.")
        return 1
    
    print("✓ All imports successful!")
    
    # Run basic pipeline
    if not run_basic_pipeline():
        print("✗ Basic pipeline failed.")
        return 1
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("You can now run: python main.py --mode full")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)