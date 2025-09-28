"""
Debug script to test all imports and identify issues
"""

import os
import sys

def test_file_existence():
    """Check if all required files exist"""
    required_files = [
        'config.py',
        'data_loader.py', 
        'feature_engineering.py',
        'visualizer.py',
        'model_trainer.py',
        'predictor.py',
        'main.py',
        'requirements.txt'
    ]
    
    print("=== FILE EXISTENCE CHECK ===")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} - EXISTS")
        else:
            print(f"✗ {file} - MISSING")
    print()

def test_imports():
    """Test importing all modules"""
    modules = [
        ('config', ['NUMERIC_FEATURES', 'CATEGORICAL_FEATURES']),
        ('data_loader', ['DataLoader']),
        ('feature_engineering', ['FeatureEngineer']),
        ('visualizer', ['DataVisualizer']),
        ('model_trainer', ['ModelTrainer']),
        ('predictor', ['DoctorPredictor', 'PredictionScheduler'])
    ]
    
    print("=== IMPORT TEST ===")
    for module_name, classes in modules:
        try:
            module = __import__(module_name)
            print(f"✓ {module_name} - IMPORTED")
            
            # Check if classes exist
            for class_name in classes:
                if hasattr(module, class_name):
                    print(f"  ✓ {class_name} - FOUND")
                else:
                    print(f"  ✗ {class_name} - NOT FOUND")
                    
        except ImportError as e:
            print(f"✗ {module_name} - IMPORT ERROR: {str(e)}")
        except Exception as e:
            print(f"✗ {module_name} - OTHER ERROR: {str(e)}")
    print()

def test_dependencies():
    """Test required dependencies"""
    dependencies = [
        'pandas',
        'numpy', 
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn'
    ]
    
    print("=== DEPENDENCY CHECK ===")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} - INSTALLED")
        except ImportError:
            print(f"✗ {dep} - NOT INSTALLED")
    print()

def check_data_file():
    """Check if data file exists"""
    data_paths = [
        'data/dummy_npi_data.csv',
        'dummy_npi_data.csv'
    ]
    
    print("=== DATA FILE CHECK ===")
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ Data file found at: {path}")
            return
    
    print("✗ Data file not found in expected locations:")
    for path in data_paths:
        print(f"  - {path}")
    print()

def main():
    """Run all diagnostic tests"""
    print("DIAGNOSTIC SCRIPT FOR DOCTOR SURVEY TARGETING PROJECT")
    print("="*60)
    
    test_file_existence()
    test_dependencies() 
    test_imports()
    check_data_file()
    
    print("=== SUMMARY ===")
    print("If you see any ✗ marks above, please fix those issues first.")
    print("All ✓ marks indicate everything is working correctly.")

if __name__ == "__main__":
    main()