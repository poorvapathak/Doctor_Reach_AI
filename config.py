"""
Configuration file for Doctor Survey Targeting project
"""

# File paths
DATA_PATH = 'data/dummy_npi_data.csv'
MODEL_PATH = 'models/doctor_survey_model.pkl'
OUTPUT_PATH = 'output/top_doctors.csv'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# Feature definitions
NUMERIC_FEATURES = [
    'Usage Time (mins)', 
    'Login_Sin_Hour', 
    'Login_Cos_Hour', 
    'Logout_Sin_Hour', 
    'Logout_Cos_Hour',
    'Target_Sin_Hour', 
    'Target_Cos_Hour', 
    'Avg_Usage_Time', 
    'Std_Login_Hour', 
    'Avg_Attempts_By_Region',
    'Historical_Attendance_Rate'
]

CATEGORICAL_FEATURES = ['State', 'Region', 'Speciality']

# Hyperparameter grid for tuning
PARAM_GRID = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__min_child_weight': [1, 3],
    'classifier__subsample': [0.8, 1.0]
}

# Prediction settings
TOP_K_DOCTORS = 20
PRECISION_K = 20