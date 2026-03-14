import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from .config import get_config

class LinearRegressionAnalyzer:
    """
    Simple linear regression module for thermoelectric analysis.
    Single plot: predicted vs actual with y = x line.
    """
    
    def __init__(self):
        """Initialize the analyzer with configuration."""
        self.config = get_config('LINEAR_REGRESSION')
        
        self.data = None
        self.model = None
        self.results = {}
        
        # Create output directory
        os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)
        
        print(f"Analysis: {self.config['FEATURE']} → {self.config['TARGET']}")
    
    def load_data(self, filepath):
        """Load data from a file."""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.csv':
            try:
                df = pd.read_csv(filepath, sep=self.config['CSV_SEPARATOR'])
            except:
                df = pd.read_csv(filepath, sep=';')
        
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        print(f"File loaded: {filepath} ({df.shape[0]} points)")
        
        # Check required columns
        required = [self.config['FEATURE'], self.config['TARGET']]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self.data = df
        return self.data
    
    def fit(self):
        """Perform linear regression."""
        if self.data is None:
            raise ValueError("Load the data first.")
        
        feature = self.config['FEATURE']
        target = self.config['TARGET']
        tolerance = self.config['TOLERANCE']
        
        print(f"\nRegression: {feature} → {target}")
        
        # Filter if tolerance specified
        if tolerance is not None:
            mask = np.abs(self.data[feature] - self.data[target]) <= tolerance
            df = self.data[mask].copy()
            print(f"Points after filtering: {len(df)}/{len(self.data)}")
        else:
            df = self.data.copy()
        
        if len(df) < 2:
            raise ValueError("Not enough points for analysis.")
        
        # Extract X and y
        X = df[[feature]].values
        y = df[target].values
        
        # Create and train the model
        self.model = LinearRegression()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        
        # Compute metrics
        slope = self.model.coef_[0]
        intercept = self.model.intercept_
        
        if intercept >= 0:
            equation = f"{target} = {slope:.3f}·{feature} + {intercept:.3f}"
        else:
            equation = f"{target} = {slope:.3f}·{feature} - {abs(intercept):.3f}"
        
        # Store results
        self.results = {
            'feature': feature,
            'target': target,
            'equation': equation,
            'slope': slope,
            'intercept': intercept,
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'n_points': len(df),
            'X': X.flatten(),
            'y_actual': y,
            'y_pred': y_pred
        }
        
        # Display results
        print(f"Equation: {equation}")
        print(f"R² = {self.results['r2']:.4f}")
        print(f"MAE = {self.results['mae']:.4f}")
        
        return self.results
    
    def plot(self, save=True):
        """Single plot: predicted vs actual with y = x line."""
        if not self.results:
            raise ValueError("Run fit() first.")
        
        # Create figure
        plt.figure(figsize=(6, 5))
        
        # Data
        y_actual = self.results['y_actual']
        y_pred = self.results['y_pred']
        
        # Scatter
        plt.scatter(y_actual, y_pred, 
                   color=self.config['SCATTER_COLOR'],
                   s=self.config['SCATTER_SIZE'],
                   alpha=0.6,
                   label='Data')
        
        # y = x line (perfect prediction)
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        
        plt.plot([min_val - margin, max_val + margin],
                 [min_val - margin, max_val + margin],
                 color='black',
                 linestyle='--',
                 linewidth=1,
                 label='y = x')
        
        # Labels and title
        plt.xlabel(f"Actual {self.config['Y_LABEL']}", fontsize=12)
        plt.ylabel(f"Predicted {self.config['Y_LABEL']}", fontsize=12)
        plt.title(f"{self.results['equation']}\nR² = {self.results['r2']:.3f}", fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save
        if save:
            filename = f"regression_{self.config['FEATURE']}_{self.config['TARGET']}.pdf"
            path = os.path.join(self.config['OUTPUT_DIR'], filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved: {path}")
        
        plt.show()
    
    def summary(self):
        """Display a simple summary."""
        if not self.results:
            print("No results to display.")
            return
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Equation: {self.results['equation']}")
        print(f"R²: {self.results['r2']:.4f}")
        print(f"MAE: {self.results['mae']:.4f}")
        print(f"Points: {self.results['n_points']}")
        print("="*50)
    
    def predict(self, X):
        """Predict a value."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if isinstance(X, (int, float)):
            X = np.array([[X]])
        
        return self.model.predict(X)[0]

