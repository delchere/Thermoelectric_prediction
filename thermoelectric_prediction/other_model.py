"""
Module for various thermoelectric prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Optional, Dict, Any

from .config import get_config


class ModelPredictor:
    """
    Module for various thermoelectric prediction models.
    
    Available properties for prediction:
    - 's_n': n-type Seebeck coefficient (μV/K)
    - 's_p': p-type Seebeck coefficient (μV/K)
    - 'pf_n': n-type Power factor (μW/(cm·K²))
    - 'pf_p': p-type Power factor (μW/(cm·K²))
    - 'sigma_n': n-type Conductivity (Ω⁻¹·cm⁻¹)
    - 'sigma_p': p-type Conductivity (Ω⁻¹·cm⁻¹)
    
    Available models:
    - 'random_forest': Random Forest Regressor
    - 'gradient_boosting': Gradient Boosting Regressor
    - 'svr': Support Vector Regression
    - 'knn': K-Nearest Neighbors Regressor
    - 'decision_tree': Decision Tree Regressor
    """
    
    # Available thermoelectric properties for prediction
    THERMOELECTRIC_PROPERTIES = [
        's_n', 's_p', 'pf_n', 'pf_p', 'sigma_n', 'sigma_p'
    ]
    
    # Dictionary of available models
    MODEL_TYPES = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'svr': SVR,
        'knn': KNeighborsRegressor,
        'decision_tree': DecisionTreeRegressor
    }
    
    def __init__(self, target_column: str = "s_n", model_type: str = "random_forest"):
        """
        Initialize the model predictor.
        
        Args:
            target_column (str): Property to predict (see THERMOELECTRIC_PROPERTIES)
            model_type (str): Type of model (see MODEL_TYPES)
        
        Raises:
            ValueError: If property or model is not available
        """
        self.target_column = target_column.lower()
        self.model_type = model_type.lower()
        
        # Check if target property is available
        if self.target_column not in self.THERMOELECTRIC_PROPERTIES:
            raise ValueError(
                f"Property '{target_column}' not available. "
                f"Available properties: {self.THERMOELECTRIC_PROPERTIES}"
            )
        
        # Check if model is available
        if self.model_type not in self.MODEL_TYPES:
            available_models = list(self.MODEL_TYPES.keys())
            raise ValueError(
                f"Model '{model_type}' not available. "
                f"Available models: {available_models}"
            )
        
        # Load configuration
        self.config = get_config('OTHER_MODELS')
        
        # Initialize attributes
        self.model = None
        self.feature_importance = None
        self.results = {}
        self.X_columns = None
        
        # Create output directory
        os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)
        
        print(f"ModelPredictor initialized:")
        print(f"  Model type: {self.model_type}")
        print(f"  Property to predict: {self.target_column}")
        print(f"  Output directory: {self.config['OUTPUT_DIR']}")
    
    def get_available_properties(self) -> List[str]:
        """
        Returns the list of available properties for prediction.
        
        Returns:
            list: Names of available thermoelectric properties
        """
        return self.THERMOELECTRIC_PROPERTIES.copy()
    
    def get_available_models(self) -> List[str]:
        """
        Returns the list of available models.
        
        Returns:
            list: Names of available models
        """
        return list(self.MODEL_TYPES.keys())
    
    def _get_columns_to_drop(self, df_columns: List[str]) -> List[str]:
        """
        Determines which columns to drop from the dataset.
        
        Args:
            df_columns (list): List of DataFrame columns
        
        Returns:
            list: Columns to drop
        """
        # Metadata columns to always drop
        to_drop = [
            col for col in self.config['DROP_COLUMNS'] 
            if col in df_columns
        ]
        
        # Drop all thermoelectric properties EXCEPT the target
        for prop in self.THERMOELECTRIC_PROPERTIES:
            if prop != self.target_column and prop in df_columns:
                to_drop.append(prop)
        
        return list(set(to_drop))
    
    def load_data(self, filepath: str, **kwargs) -> tuple:
        """
        Load data from a file.
        
        Args:
            filepath (str): Path to data file
            **kwargs: Additional arguments for pandas.read_csv or pandas.read_excel
        
        Returns:
            tuple: (X, y, df_original) where:
                - X: Features (DataFrame)
                - y: Target (Series)
                - df_original: Original DataFrame
        
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine format and load
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.csv':
            # Try multiple separators
            try:
                df = pd.read_csv(filepath, sep=self.config['CSV_SEPARATOR'], **kwargs)
            except:
                try:
                    df = pd.read_csv(filepath, sep=',', **kwargs)
                except:
                    df = pd.read_csv(filepath, sep=';', **kwargs)
        
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, **kwargs)
        
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Accepted formats: {self.config['DATA_FORMATS']}"
            )
        
        print(f"File loaded: {filepath}")
        print(f"  Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Check if target column exists
        if self.target_column not in df.columns:
            available_in_file = [col for col in self.THERMOELECTRIC_PROPERTIES if col in df.columns]
            raise ValueError(
                f"Target column '{self.target_column}' not found.\n"
                f"Properties available in file: {available_in_file}"
            )
        
        # Create a copy for manipulation
        df_processed = df.copy()
        
        # Determine columns to drop
        columns_to_drop = self._get_columns_to_drop(df_processed.columns)
        
        if columns_to_drop:
            print(f"  Columns to drop: {columns_to_drop}")
            df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
        
        # Separate features/target
        y = df_processed.pop(self.target_column)
        X = df_processed
        
        # Save column names
        self.X_columns = X.columns.tolist()
        
        print(f"  Features after processing: {X.shape[1]} columns")
        print(f"  Target: {self.target_column}")
        print(f"  Samples: {len(X)}")
        
        return X, y, df
    
    def _get_model_params(self) -> Dict[str, Any]:
        """
        Returns model parameters based on model type.
        
        Returns:
            dict: Model parameters
        """
        if self.model_type == 'random_forest':
            return {
                'n_estimators': self.config['RF_N_ESTIMATORS'],
                'max_depth': self.config['RF_MAX_DEPTH'],
                'min_samples_split': self.config['RF_MIN_SAMPLES_SPLIT'],
                'min_samples_leaf': self.config['RF_MIN_SAMPLES_LEAF'],
                'max_features': self.config['RF_MAX_FEATURES'],
                'bootstrap': self.config['RF_BOOTSTRAP'],
                'oob_score': self.config['RF_OOB_SCORE'],
                'random_state': self.config['RANDOM_STATE'],
                'n_jobs': self.config['RF_N_JOBS'],
                'verbose': self.config['RF_VERBOSE'],
                'warm_start': self.config['RF_WARM_START']
            }
        
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': self.config['GB_N_ESTIMATORS'],
                'learning_rate': self.config['GB_LEARNING_RATE'],
                'max_depth': self.config['GB_MAX_DEPTH'],
                'min_samples_split': self.config['GB_MIN_SAMPLES_SPLIT'],
                'min_samples_leaf': self.config['GB_MIN_SAMPLES_LEAF'],
                'subsample': self.config['GB_SUBSAMPLE'],
                'random_state': self.config['RANDOM_STATE']
            }
        
        elif self.model_type == 'svr':
            return {
                'kernel': self.config['SVR_KERNEL'],
                'C': self.config['SVR_C'],
                'epsilon': self.config['SVR_EPSILON'],
                'gamma': self.config['SVR_GAMMA'],
                'tol': self.config['SVR_TOL']
            }
        
        elif self.model_type == 'knn':
            return {
                'n_neighbors': self.config['KNN_N_NEIGHBORS'],
                'weights': self.config['KNN_WEIGHTS'],
                'algorithm': self.config['KNN_ALGORITHM'],
                'leaf_size': self.config['KNN_LEAF_SIZE'],
                'p': self.config['KNN_P']
            }
        
        elif self.model_type == 'decision_tree':
            return {
                'max_depth': self.config['DT_MAX_DEPTH'],
                'min_samples_split': self.config['DT_MIN_SAMPLES_SPLIT'],
                'min_samples_leaf': self.config['DT_MIN_SAMPLES_LEAF'],
                'random_state': self.config['RANDOM_STATE'],
                'max_features': self.config['DT_MAX_FEATURES']
            }
        
        else:
            return {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            **kwargs: Additional parameters for the model
        
        Returns:
            dict: Training results
        
        Raises:
            ValueError: If data is insufficient
        """
        # Check data
        if len(X) < 10:
            raise ValueError("Insufficient number of samples for training.")
        
        if len(X.columns) == 0:
            raise ValueError("No features available for training.")
        
        # Parameters for train_test_split
        test_size = kwargs.pop('test_size', self.config['TEST_SIZE'])
        random_state = kwargs.pop('random_state', self.config['RANDOM_STATE'])
        shuffle = kwargs.pop('shuffle', self.config['SHUFFLE'])
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=shuffle
        )
        
        # Get model parameters
        model_params = self._get_model_params()
        
        # Update with provided kwargs
        model_params.update(kwargs)
        
        # Create the model
        model_class = self.MODEL_TYPES[self.model_type]
        self.model = model_class(**model_params)
        
        print(f"\nTraining {self.model_type} model:")
        print(f"  Property to predict: {self.target_column}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Number of features: {X_train.shape[1]}")
        
        # Training
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.results = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_names': self.X_columns,
            'model_type': self.model_type,
            'target': self.target_column,
            'model_params': model_params
        }
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"  Feature importance: available")
        
        print(f"  Training completed")
        
        return self.results
    
    def evaluate(self, print_results: bool = True) -> Dict[str, Any]:
        """
        Display model evaluation results.
        
        Args:
            print_results (bool): Display results on screen
        
        Returns:
            dict: Evaluation results
        
        Raises:
            ValueError: If model has not been trained
        """
        if not self.results:
            raise ValueError("Model not trained. Run train() first.")
        
        if print_results:
            print('\n' + '='*60)
            print(f'MODEL EVALUATION - {self.model_type.upper()}')
            print('='*60)
            print(f"Model type: {self.results['model_type']}")
            print(f"Predicted property: {self.results['target']}")
            print(f"R² score: {self.results['r2']:.3f}")
            print(f"RMSE: {self.results['rmse']:.3f}")
            print(f"MAE: {self.results['mae']:.3f}")
            print(f"Training samples: {self.results['train_size']}")
            print(f"Test samples: {self.results['test_size']}")
            print(f"Number of features: {len(self.results['feature_names'])}")
            print('='*60)
        
        return self.results
    
    def plot_predictions(self, save_figure: bool = True, filename: Optional[str] = None):
        """
        Display predictions vs actual values.
        
        Args:
            save_figure (bool): Save the figure
            filename (str): Save filename
        
        Raises:
            ValueError: If model has not been trained
        """
        if not self.results:
            raise ValueError("Model not trained. Run train() first.")
        
        y_test = self.results['y_test']
        y_pred = self.results['y_pred']
        
        plt.figure(figsize=self.config['FIGURE_SIZE'])
        
        # Determine graph limits
        max_val = max(np.max(y_test), np.max(y_pred))
        min_val = min(np.min(y_test), np.min(y_pred))
        margin = (max_val - min_val) * 0.05
        
        # Perfect prediction line
        plt.plot(
            [min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            color=self.config['LINE_COLOR'], 
            linestyle=self.config['LINE_STYLE'],
            linewidth=self.config['LINE_WIDTH'],
            label='Perfect prediction'
        )
        
        # Prediction points
        plt.scatter(
            y_test, y_pred, 
            s=self.config['SCATTER_SIZE'], 
            color=self.config['SCATTER_COLOR'], 
            alpha=self.config['SCATTER_ALPHA'], 
            label='Predictions'
        )
        
        # Labels based on property
        if self.target_column.startswith('s_'):
            unit = 'μV/K'
        elif self.target_column.startswith('pf_'):
            unit = 'μW/(cm·K²)'
        elif self.target_column.startswith('sigma_'):
            unit = 'Ω⁻¹·cm⁻¹'
        else:
            unit = ''
        
        plt.xlabel(f"Actual {self.target_column} ({unit})", fontsize=12)
        plt.ylabel(f"Predicted {self.target_column} ({unit})", fontsize=12)
        plt.title(
            f"{self.model_type} - {self.target_column} "
            f"(R² = {self.results['r2']:.3f})", 
            fontsize=14
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        if save_figure:
            if filename is None:
                filename = (
                    f"{self.model_type}_predictions_"
                    f"{self.target_column}.{self.config['SAVE_FORMAT']}"
                )
            
            save_path = os.path.join(self.config['OUTPUT_DIR'], filename)
            plt.savefig(save_path, dpi=self.config['DPI'], bbox_inches='tight')
            print(f"\nFigure saved: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n: Optional[int] = None, 
                                save_figure: bool = True):
        """
        Display feature importance.
        
        Args:
            top_n (int): Number of features to display
            save_figure (bool): Save the figure
        
        Raises:
            ValueError: If feature importance is not available
        """
        if self.feature_importance is None:
            print(f"Feature importance not available for {self.model_type}")
            return
        
        if top_n is None:
            top_n = self.config['IMPORTANCE_TOP_N']
        
        # Limit to number of available features
        top_n = min(top_n, len(self.feature_importance))
        
        # Take top_n features
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = plt.barh(
            range(len(top_features)), 
            top_features['importance'].values
        )
        
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.title(
            f'Top {top_n} Features - {self.model_type} ({self.target_column})', 
            fontsize=14
        )
        plt.gca().invert_yaxis()  # Invert Y axis to have most important on top
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save figure
        if save_figure:
            filename = (
                f"{self.model_type}_importance_"
                f"{self.target_column}.{self.config['SAVE_FORMAT']}"
            )
            save_path = os.path.join(self.config['OUTPUT_DIR'], filename)
            plt.savefig(save_path, dpi=self.config['DPI'], bbox_inches='tight')
            print(f"Feature importance figure saved: {save_path}")
        
        plt.show()
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_new (pd.DataFrame): New data
        
        Returns:
            np.ndarray: Predictions
        
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train() first.")
        
        return self.model.predict(X_new)
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filename (str): Save filename
        
        Returns:
            str: Path to saved file
        
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            filename = f"{self.model_type}_model_{self.target_column}.pkl"
        
        save_path = os.path.join(self.config['OUTPUT_DIR'], filename)
        joblib.dump(self.model, save_path)
        
        print(f"Model saved: {save_path}")
        return save_path
    
    def load_model(self, filename: str):
        """
        Load a saved model.
        
        Args:
            filename (str): Path to model file
        
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        self.model = joblib.load(filename)
        print(f"Model loaded from: {filename}")
    
    def summary(self):
        """
        Display a complete analysis summary.
        """
        if not self.results:
            print("No model trained. Run train() first.")
            return
        
        print('\n' + '='*60)
        print('MODEL SUMMARY')
        print('='*60)
        print(f"Model type: {self.model_type}")
        print(f"Predicted property: {self.target_column}")
        
        # Units based on property
        if self.target_column.startswith('s_'):
            unit = 'μV/K'
        elif self.target_column.startswith('pf_'):
            unit = 'μW/(cm·K²)'
        elif self.target_column.startswith('sigma_'):
            unit = 'Ω⁻¹·cm⁻¹'
        else:
            unit = ''
        
        print(f"Unit: {unit}")
        print(f"\nPerformance:")
        print(f"  R²: {self.results['r2']:.4f}")
        print(f"  RMSE: {self.results['rmse']:.4f} {unit}")
        print(f"  MAE: {self.results['mae']:.4f} {unit}")
        print(f"\nSamples:")
        print(f"  Training: {self.results['train_size']}")
        print(f"  Test: {self.results['test_size']}")
        print(f"Number of features: {len(self.results['feature_names'])}")
        
        if self.feature_importance is not None:
            print(f"\nTop 5 most important features:")
            for i, row in self.feature_importance.head(5).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        print('='*60)

