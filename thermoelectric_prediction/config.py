
"""
Configuration file for thermoelectric_prediction.
Users can directly modify this file to change parameters.
"""

# ============================================================================
# GENERAL CONFIGURATION
# ============================================================================

OUTPUT_DIR = './output'
DATA_FORMATS = ['.csv', '.xlsx', '.xls']
CSV_SEPARATOR = ';'

# ============================================================================
# DBSCAN CONFIGURATION
# ============================================================================

DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 10
DBSCAN_METRIC = 'euclidean'
DBSCAN_ALGORITHM = 'auto'
DBSCAN_COLUMNS = ['s_p', 's_n']
DBSCAN_N_JOBS = None
DBSCAN_FIGURE_SIZE = (12, 5)
DBSCAN_DPI = 300
DBSCAN_SAVE_FORMAT = 'pdf'
DBSCAN_COLORS = {
    -1: 'green', 0: 'blue', 1: 'red', 2: 'orange', 3: 'purple'
}


DBSCAN_LEAF_SIZE = 30
DBSCAN_P = 2
DBSCAN_MARKERS = {
    -1: 'x', 0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'v'
}
DBSCAN_LABELS = {
    -1: 'Bruit', 0: 'Cluster 0', 1: 'Cluster 1', 
    2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'
}
DBSCAN_GRID_PARAMS = {
    'alpha': 0.3, 'linestyle': '--', 'linewidth': 0.5
}

# ============================================================================
# LINEAR REGRESSION CONFIGURATION
# ============================================================================
# Pair of properties for linear regression
LINEAR_FEATURE = 's_p'           # Independent variable (X)
LINEAR_TARGET = 's_n'            # Dependent variable (Y)
LINEAR_TOLERANCE = 150           # Filtering tolerance (None for no filtering)
LINEAR_X_LABEL = '$s_p$ ($\mu V / K$)'
LINEAR_Y_LABEL = '$s_n$ ($\mu V / K$)'
LINEAR_X_UNIT = 'μV/K'
LINEAR_Y_UNIT = 'μV/K'

# Model parameters
LINEAR_FIT_INTERCEPT = True
LINEAR_COPY_X = True
LINEAR_N_JOBS = None
LINEAR_POSITIVE = False

# Visualization
LINEAR_FIGURE_SIZE = (8, 6)
LINEAR_SCATTER_COLOR = 'blue'
LINEAR_SCATTER_MARKER = 'o'
LINEAR_SCATTER_SIZE = 50
LINEAR_LINE_COLOR = 'black'
LINEAR_LINE_STYLE = '--'
LINEAR_LINE_WIDTH = 2
LINEAR_DPI = 300
LINEAR_SAVE_FORMAT = 'pdf'

# ============================================================================
# OTHER MODELS CONFIGURATION (Random Forest, etc.)
# ============================================================================

# Default columns to drop
OTHER_DROP_COLUMNS = ['mpid', 'formula', 'composition']

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = None
RF_BOOTSTRAP = True
RF_OOB_SCORE = False
RF_N_JOBS = -1
RF_VERBOSE = 0
RF_WARM_START = False

# Gradient Boosting
GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 3
GB_MIN_SAMPLES_SPLIT = 5
GB_MIN_SAMPLES_LEAF = 1
GB_SUBSAMPLE = 1.0

# SVR
SVR_KERNEL = 'rbf'
SVR_C = 1.0
SVR_EPSILON = 0.1
SVR_GAMMA = 'scale'
SVR_TOL = 0.001

# KNN
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = 'uniform'
KNN_ALGORITHM = 'auto'
KNN_LEAF_SIZE = 30
KNN_P = 2

# Decision Tree
DT_MAX_DEPTH = None
DT_MIN_SAMPLES_SPLIT = 2
DT_MIN_SAMPLES_LEAF = 1
DT_MAX_FEATURES = None

# Common parameters
RANDOM_STATE = 42

# Data splitting
OTHER_TEST_SIZE = 0.1
OTHER_TRAIN_SIZE = None
OTHER_SHUFFLE = True

# Visualization
OTHER_FIGURE_SIZE = (6, 5)
OTHER_SCATTER_COLOR = 'red'
OTHER_SCATTER_SIZE = 20
OTHER_SCATTER_ALPHA = 0.6
OTHER_LINE_COLOR = 'black'
OTHER_LINE_STYLE = '--'
OTHER_LINE_WIDTH = 1
OTHER_IMPORTANCE_TOP_N = 20
OTHER_DPI = 300
OTHER_SAVE_FORMAT = 'pdf'

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def get_config(module_name=None):
    """
    Get the configuration for a specific module.
    """
    if module_name == 'DBSCAN':
        return {
            'EPS': DBSCAN_EPS,
            'MIN_SAMPLES': DBSCAN_MIN_SAMPLES,
            'METRIC': DBSCAN_METRIC,
            'ALGORITHM': DBSCAN_ALGORITHM,
            'COLUMNS': DBSCAN_COLUMNS,
            'N_JOBS': DBSCAN_N_JOBS,
            'FIGURE_SIZE': DBSCAN_FIGURE_SIZE,
            'DPI': DBSCAN_DPI,
            'SAVE_FORMAT': DBSCAN_SAVE_FORMAT,
            'COLORS': DBSCAN_COLORS,
            'OUTPUT_DIR': OUTPUT_DIR,
            'DATA_FORMATS': DATA_FORMATS,
            'CSV_SEPARATOR': CSV_SEPARATOR,
            'LEAF_SIZE': DBSCAN_LEAF_SIZE,      
            'P': DBSCAN_P,                    
            'MARKERS': DBSCAN_MARKERS,          
            'LABELS': DBSCAN_LABELS,           
            'GRID_PARAMS': DBSCAN_GRID_PARAMS,  
        }
        
     
    elif module_name == 'LINEAR_REGRESSION':
        return {
            'FEATURE': LINEAR_FEATURE,
            'TARGET': LINEAR_TARGET,
            'TOLERANCE': LINEAR_TOLERANCE,
            'X_LABEL': LINEAR_X_LABEL,
            'Y_LABEL': LINEAR_Y_LABEL,
            'X_UNIT': LINEAR_X_UNIT,
            'Y_UNIT': LINEAR_Y_UNIT,
            'FIT_INTERCEPT': LINEAR_FIT_INTERCEPT,
            'COPY_X': LINEAR_COPY_X,
            'N_JOBS': LINEAR_N_JOBS,
            'POSITIVE': LINEAR_POSITIVE,
            'FIGURE_SIZE': LINEAR_FIGURE_SIZE,
            'SCATTER_COLOR': LINEAR_SCATTER_COLOR,
            'SCATTER_MARKER': LINEAR_SCATTER_MARKER,
            'SCATTER_SIZE': LINEAR_SCATTER_SIZE,
            'LINE_COLOR': LINEAR_LINE_COLOR,
            'LINE_STYLE': LINEAR_LINE_STYLE,
            'LINE_WIDTH': LINEAR_LINE_WIDTH,
            'DPI': LINEAR_DPI,
            'SAVE_FORMAT': LINEAR_SAVE_FORMAT,
            'OUTPUT_DIR': OUTPUT_DIR,
            'DATA_FORMATS': DATA_FORMATS,
            'CSV_SEPARATOR': CSV_SEPARATOR,
        }
    
    elif module_name == 'OTHER_MODELS':
        return {
            'DROP_COLUMNS': OTHER_DROP_COLUMNS,
            
            # Random Forest
            'RF_N_ESTIMATORS': RF_N_ESTIMATORS,
            'RF_MAX_DEPTH': RF_MAX_DEPTH,
            'RF_MIN_SAMPLES_SPLIT': RF_MIN_SAMPLES_SPLIT,
            'RF_MIN_SAMPLES_LEAF': RF_MIN_SAMPLES_LEAF,
            'RF_MAX_FEATURES': RF_MAX_FEATURES,
            'RF_BOOTSTRAP': RF_BOOTSTRAP,
            'RF_OOB_SCORE': RF_OOB_SCORE,
            'RF_N_JOBS': RF_N_JOBS,
            'RF_VERBOSE': RF_VERBOSE,
            'RF_WARM_START': RF_WARM_START,
            
            # Gradient Boosting
            'GB_N_ESTIMATORS': GB_N_ESTIMATORS,
            'GB_LEARNING_RATE': GB_LEARNING_RATE,
            'GB_MAX_DEPTH': GB_MAX_DEPTH,
            'GB_MIN_SAMPLES_SPLIT': GB_MIN_SAMPLES_SPLIT,
            'GB_MIN_SAMPLES_LEAF': GB_MIN_SAMPLES_LEAF,
            'GB_SUBSAMPLE': GB_SUBSAMPLE,
            
            # SVR
            'SVR_KERNEL': SVR_KERNEL,
            'SVR_C': SVR_C,
            'SVR_EPSILON': SVR_EPSILON,
            'SVR_GAMMA': SVR_GAMMA,
            'SVR_TOL': SVR_TOL,
            
            # KNN
            'KNN_N_NEIGHBORS': KNN_N_NEIGHBORS,
            'KNN_WEIGHTS': KNN_WEIGHTS,
            'KNN_ALGORITHM': KNN_ALGORITHM,
            'KNN_LEAF_SIZE': KNN_LEAF_SIZE,
            'KNN_P': KNN_P,
            
            # Decision Tree
            'DT_MAX_DEPTH': DT_MAX_DEPTH,
            'DT_MIN_SAMPLES_SPLIT': DT_MIN_SAMPLES_SPLIT,
            'DT_MIN_SAMPLES_LEAF': DT_MIN_SAMPLES_LEAF,
            'DT_MAX_FEATURES': DT_MAX_FEATURES,
            
            # Common parameters
            'RANDOM_STATE': RANDOM_STATE,
            
            # Data splitting
            'TEST_SIZE': OTHER_TEST_SIZE,
            'TRAIN_SIZE': OTHER_TRAIN_SIZE,
            'SHUFFLE': OTHER_SHUFFLE,
            
            # Visualization
            'FIGURE_SIZE': OTHER_FIGURE_SIZE,
            'SCATTER_COLOR': OTHER_SCATTER_COLOR,
            'SCATTER_SIZE': OTHER_SCATTER_SIZE,
            'SCATTER_ALPHA': OTHER_SCATTER_ALPHA,
            'LINE_COLOR': OTHER_LINE_COLOR,
            'LINE_STYLE': OTHER_LINE_STYLE,
            'LINE_WIDTH': OTHER_LINE_WIDTH,
            'IMPORTANCE_TOP_N': OTHER_IMPORTANCE_TOP_N,
            'DPI': OTHER_DPI,
            'SAVE_FORMAT': OTHER_SAVE_FORMAT,
            'OUTPUT_DIR': OUTPUT_DIR,
            'DATA_FORMATS': DATA_FORMATS,
            'CSV_SEPARATOR': CSV_SEPARATOR
        }
    
    elif module_name is None:
        return {
            'DBSCAN': get_config('DBSCAN'),
            'LINEAR_REGRESSION': get_config('LINEAR_REGRESSION'),
            'OTHER_MODELS': get_config('OTHER_MODELS')
        }
    
    else:
        raise ValueError(f"Module {module_name} not recognized.")

def update_config(module_name, **kwargs):
    """
    Update the configuration of a module.
    """
    if module_name == 'DBSCAN':
        for key, value in kwargs.items():
            if key.startswith('DBSCAN_'):
                globals()[key] = value
                print(f"  {key} = {value}")
    
    elif module_name == 'LINEAR_REGRESSION':
        for key, value in kwargs.items():
            if key.startswith('LINEAR_'):
                globals()[key] = value
                print(f"  {key} = {value}")
    
    elif module_name == 'OTHER_MODELS':
        for key, value in kwargs.items():
            if key.startswith(('RF_', 'GB_', 'SVR_', 'KNN_', 'DT_', 'OTHER_')):
                globals()[key] = value
                print(f"  {key} = {value}")
    
    else:
        raise ValueError(f"Module {module_name} not recognized.")

def print_config(module_name=None):
    """
    Display the configuration.
    """
    if module_name:
        config = get_config(module_name)
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {module_name}")
        print('='*60)
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        for name in ['DBSCAN', 'LINEAR_REGRESSION', 'OTHER_MODELS']:
            print_config(name)

