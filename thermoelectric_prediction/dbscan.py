import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Import configuration
from .config import get_config

class DBSCANAnalyzer:
    """
    DBSCAN module for thermoelectric data clustering analysis.
    
    This class allows applying the DBSCAN algorithm on thermoelectric
    data, visualizing clusters, and analyzing results.
    
    Usage example:
        >>> from thermoelectric_prediction import DBSCANAnalyzer
        >>> analyzer = DBSCANAnalyzer()
        >>> data = analyzer.load_data('my_data.xlsx')
        >>> labels = analyzer.fit(data.values)
        >>> results = analyzer.analyze()
        >>> analyzer.plot_clusters()
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the DBSCAN analyzer.
        
        Args:
            config_file (str): Path to a custom configuration file.
                              If None, uses default configuration.
        """
        # Load configuration
        if config_file is None:
            self.config = get_config('DBSCAN')
        else:
            # For now, use default configuration
            # Could implement loading from file later
            self.config = get_config()
            print(f"NOTE: Loading from config file not implemented, "
                  f"using default configuration.")
        
        # Initialize attributes
        self.data = None
        self.X_scaled = None
        self.labels = None
        self.db = None
        
        # Create output directory if necessary
        os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)
        
        print(f"DBSCANAnalyzer initialized with:")
        print(f"  EPS = {self.config['EPS']}")
        print(f"  MIN_SAMPLES = {self.config['MIN_SAMPLES']}")
        print(f"  Columns: {self.config['COLUMNS']}")
    
    def load_data(self, filepath, columns=None):
        """
        Load data from a file.
        
        Args:
            filepath (str): Path to data file
            columns (list): List of columns to use (None to use configuration)
            
        Returns:
            pd.DataFrame: Loaded data (only specified columns)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine columns to use
        if columns is None:
            columns = self.config['COLUMNS']
        
        # Determine format and load
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.csv':
            try:
                df = pd.read_csv(filepath, sep=self.config['CSV_SEPARATOR'])
            except Exception as e:
                # Try other common separators
                try:
                    df = pd.read_csv(filepath, sep=',')
                except:
                    try:
                        df = pd.read_csv(filepath, sep='\t')
                    except:
                        raise ValueError(f"Unable to read CSV file: {e}")
        
        elif file_ext in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(filepath)
            except Exception as e:
                raise ValueError(f"Unable to read Excel file: {e}")
        
        else:
            raise ValueError(f"File format not supported: {file_ext}. "
                            f"Accepted formats: {self.config['DATA_FORMATS']}")
        
        print(f"File loaded: {filepath}")
        print(f"  Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check that requested columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in file: {missing_cols}")
        
        # Select only requested columns
        self.data = df[columns].copy()
        
        # Check for missing values
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            print(f"WARNING: {missing_values} missing values detected.")
            # Option: could fill or remove here
            # For now, remove rows with NaN
            self.data = self.data.dropna()
            print(f"  {self.data.shape[0]} rows retained after removing NaN")
        
        print(f"  Columns used: {columns}")
        print(f"  Final data: {self.data.shape[0]} points")
        
        return self.data
    
    def fit(self, X=None):
        """
        Apply DBSCAN on the data.
        
        Args:
            X (array-like): Input data (None to use loaded data)
            
        Returns:
            array: Cluster labels (-1 for noise)
            
        Raises:
            ValueError: If no data is available
        """
        # Use loaded data if X is not provided
        if X is None:
            if self.data is None:
                raise ValueError("No data available. First load data with load_data().")
            X = self.data.values
        else:
            # Convert to numpy array if necessary
            if isinstance(X, pd.DataFrame):
                X = X.values
        
        print(f"\nApplying DBSCAN on {X.shape[0]} points...")
        print(f"  Parameters: eps={self.config['EPS']}, min_samples={self.config['MIN_SAMPLES']}")
        
        # Data standardization
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        
        # Apply DBSCAN
        self.db = DBSCAN(
            eps=self.config['EPS'],
            min_samples=self.config['MIN_SAMPLES'],
            metric=self.config['METRIC'],
            algorithm=self.config['ALGORITHM'],
            leaf_size=self.config['LEAF_SIZE'],
            p=self.config['P'],
            n_jobs=self.config['N_JOBS']
        )
        
        self.labels = self.db.fit_predict(self.X_scaled)
        
        print(f"  DBSCAN completed.")
        
        return self.labels
    
    def analyze(self):
        """
        Analyze and display clustering results.
        
        Returns:
            dict: Analysis results
            
        Raises:
            ValueError: If DBSCAN has not been run
        """
        if self.labels is None:
            raise ValueError("First run fit() on the data.")
        
        # Calculate metrics
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        # Separate noise points
        noise_mask = (self.labels == -1)
        core_mask = (self.labels != -1)
        
        # Descriptive statistics
        results = {
            'n_points_total': len(self.labels),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': (n_noise / len(self.labels)) * 100,
            'labels': self.labels,
            'noise_indices': np.where(noise_mask)[0].tolist(),
            'core_indices': np.where(core_mask)[0].tolist()
        }
        
        # Calculate statistics per cluster
        cluster_stats = {}
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            if label == -1:
                cluster_name = 'noise'
            else:
                cluster_name = f'cluster_{label}'
            
            mask = (self.labels == label)
            cluster_data = self.data.iloc[mask] if self.data is not None else None
            
            cluster_stats[label] = {
                'name': cluster_name,
                'size': mask.sum(),
                'percentage': (mask.sum() / len(self.labels)) * 100
            }
        
        results['cluster_stats'] = cluster_stats
        
        # Calculate silhouette coefficient (only if at least 2 clusters)
        if len(set(self.labels[core_mask])) > 1:
            silhouette = metrics.silhouette_score(
                self.X_scaled[core_mask], 
                self.labels[core_mask]
            )
            results['silhouette_score'] = silhouette
        else:
            results['silhouette_score'] = None
        
        # Display results
        print("\n" + "="*60)
        print("DBSCAN RESULTS ANALYSIS")
        print("="*60)
        
        print(f"\n1. GENERAL STATISTICS:")
        print(f"   Total number of points: {results['n_points_total']}")
        print(f"   Number of clusters: {results['n_clusters']}")
        print(f"   Noise points: {results['n_noise']} ({results['noise_percentage']:.1f}%)")
        
        if results['silhouette_score'] is not None:
            print(f"   Silhouette coefficient: {results['silhouette_score']:.3f}")
        
        print(f"\n2. DISTRIBUTION BY CLUSTER:")
        for label, stats in cluster_stats.items():
            if label == -1:
                print(f"   {stats['name']}: {stats['size']} points ({stats['percentage']:.1f}%)")
            else:
                print(f"   {stats['name']}: {stats['size']} points ({stats['percentage']:.1f}%)")
        
        print("\n" + "="*60)
        
        return results
    
    def get_cluster_data(self, cluster_label):
        """
        Retrieve data from a specific cluster.
        
        Args:
            cluster_label (int): Cluster label (-1 for noise)
            
        Returns:
            pd.DataFrame: Cluster data
            
        Raises:
            ValueError: If cluster doesn't exist or no data available
        """
        if self.data is None:
            raise ValueError("No data available.")
        
        if self.labels is None:
            raise ValueError("DBSCAN has not been run. Run fit() first.")
        
        if cluster_label not in self.labels:
            raise ValueError(f"Cluster {cluster_label} does not exist.")
        
        mask = (self.labels == cluster_label)
        cluster_data = self.data.iloc[mask].copy()
        
        if cluster_label == -1:
            print(f"Noise data: {len(cluster_data)} points")
        else:
            print(f"Cluster {cluster_label} data: {len(cluster_data)} points")
        
        return cluster_data
    
    def plot_clusters(self, save_figure=True, filename=None):
        """
        Display clusters with and without noise.
        
        Args:
            save_figure (bool): Save the figure
            filename (str): Name of save file
            
        Raises:
            ValueError: If DBSCAN has not been run
        """
        if self.labels is None:
            raise ValueError("First run fit() on the data.")
        
        if self.data is None:
            raise ValueError("No data available.")
        
        # Figure preparation
        fig, axs = plt.subplots(1, 2, figsize=self.config['FIGURE_SIZE'])
        
        # Original data (non-standardized)
        X_original = self.data.values
        
        # Plot 1: With noise (standardized data)
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            # Mask for this label
            mask = (self.labels == label)
            
            # Data for this cluster
            xy = self.X_scaled[mask]
            
            # Configuration according to label
            color = self.config['COLORS'].get(label, 'gray')
            marker = self.config['MARKERS'].get(label, 'o')
            
            if label == -1:
                label_name = self.config['LABELS'].get(label, 'Noise')
            else:
                label_name = self.config['LABELS'].get(label, f'Cluster {label}')
            
            # Plot points
            axs[0].scatter(xy[:, 0], xy[:, 1], 
                          c=color, s=30, 
                          marker=marker, 
                          label=label_name,
                          alpha=0.8)
        
        # First plot configuration
        axs[0].set_xlabel(f"{self.config['COLUMNS'][0]} (standardized)", fontsize=12)
        axs[0].set_ylabel(f"{self.config['COLUMNS'][1]} (standardized)", fontsize=12)
        axs[0].set_title('Clusters with noise points', fontsize=14)
        axs[0].grid(**self.config['GRID_PARAMS'])
        axs[0].legend(loc='best')
        
        # Plot 2: Without noise (original data)
        # Filter to remove noise
        core_mask = (self.labels != -1)
        X_core = X_original[core_mask]
        labels_core = self.labels[core_mask]
        
        unique_labels_core = set(labels_core)
        
        for label in unique_labels_core:
            # Mask for this label
            mask = (labels_core == label)
            
            # Data for this cluster
            xy = X_core[mask]
            
            # Configuration according to label
            color = self.config['COLORS'].get(label, 'gray')
            marker = self.config['MARKERS'].get(label, 'o')
            label_name = self.config['LABELS'].get(label, f'Cluster {label}')
            
            # Plot points
            axs[1].scatter(xy[:, 0], xy[:, 1], 
                          c=color, s=30, 
                          marker=marker, 
                          label=label_name,
                          alpha=0.8)
        
        # Second plot configuration
        axs[1].set_xlabel(f"{self.config['COLUMNS'][0]}", fontsize=12)
        axs[1].set_ylabel(f"{self.config['COLUMNS'][1]}", fontsize=12)
        axs[1].set_title('Clusters without noise points', fontsize=14)
        axs[1].grid(**self.config['GRID_PARAMS'])
        axs[1].legend(loc='best')
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save figure
        if save_figure:
            if filename is None:
                # Default filename
                filename = f"dbscan_clusters_eps{self.config['EPS']}_min{self.config['MIN_SAMPLES']}.{self.config['SAVE_FORMAT']}"
            
            save_path = os.path.join(self.config['OUTPUT_DIR'], filename)
            plt.savefig(save_path, dpi=self.config['DPI'], bbox_inches='tight')
            print(f"\nFigure saved: {save_path}")
        
        # Display
        plt.show()
    
    def save_results(self, filename=None):
        """
        Save results to a CSV file.
        
        Args:
            filename (str): Output filename
            
        Raises:
            ValueError: If no analysis has been performed
        """
        if self.labels is None:
            raise ValueError("No results to save. Run fit() and analyze() first.")
        
        if self.data is None:
            raise ValueError("No data available.")
        
        # Create DataFrame with results
        results_df = self.data.copy()
        results_df['cluster_label'] = self.labels
        
        # Add column for type (noise or cluster)
        results_df['cluster_type'] = results_df['cluster_label'].apply(
            lambda x: 'noise' if x == -1 else f'cluster_{x}'
        )
        
        # Default filename
        if filename is None:
            filename = f"dbscan_results_eps{self.config['EPS']}_min{self.config['MIN_SAMPLES']}.csv"
        
        save_path = os.path.join(self.config['OUTPUT_DIR'], filename)
        
        # Save
        results_df.to_csv(save_path, index=False)
        print(f"Results saved in: {save_path}")
        
        return save_path
    
    def summary(self):
        """
        Display a complete summary of the analysis.
        
        Returns:
            dict: Dictionary with all results
        """
        if self.labels is None:
            print("No analysis performed. Run fit() and analyze() first.")
            return None
        
        # Get analysis results
        results = self.analyze()
        
        # Display compact summary
        print("\n" + "="*60)
        print("DBSCAN SUMMARY")
        print("="*60)
        print(f"Parameters: eps={self.config['EPS']}, min_samples={self.config['MIN_SAMPLES']}")
        print(f"Identified clusters: {results['n_clusters']}")
        print(f"Noise points: {results['n_noise']} ({results['noise_percentage']:.1f}%)")
        
        if results.get('silhouette_score') is not None:
            print(f"Silhouette score: {results['silhouette_score']:.3f}")
        
        print("\nTo save results: analyzer.save_results()")
        print("To visualize: analyzer.plot_clusters()")
        print("To get cluster data: analyzer.get_cluster_data(label)")
        print("="*60)
        
        return results
    
    def save_clusters(self, original_filepath, output_dir=None):
        """
        Save each cluster to a separate file.
        
        Args:
            original_filepath (str): Path to original file
            output_dir (str): Output directory
            
        Returns:
            list: Paths of created files
        """
        if self.labels is None:
            raise ValueError("First run fit() on the data.")
        
        # Determine output directory
        if output_dir is None:
            output_dir = self.config['OUTPUT_DIR']
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load complete original data
        file_ext = os.path.splitext(original_filepath)[1].lower()
        
        if file_ext == '.csv':
            # Try multiple separators
            try:
                df_full = pd.read_csv(original_filepath, sep=self.config['CSV_SEPARATOR'])
            except:
                try:
                    df_full = pd.read_csv(original_filepath, sep=',')
                except:
                    df_full = pd.read_csv(original_filepath, sep=';')
        elif file_ext in ['.xlsx', '.xls']:
            df_full = pd.read_excel(original_filepath)
        else:
            raise ValueError(f"Format not supported: {file_ext}")
        
        # Get unique labels
        unique_labels = set(self.labels)
        created_files = []
        
        print("\n" + "="*60)
        print("SAVING INDIVIDUAL CLUSTERS")
        print("="*60)
        
        for label in unique_labels:
            # Mask for this cluster
            mask = (self.labels == label)
            
            # Filter data
            cluster_data = df_full.iloc[mask].copy()
            
            # Filename
            if label == -1:
                cluster_name = 'noise'
            else:
                cluster_name = f'cluster_{label}'
            
            # Output filename
            base_name = os.path.splitext(os.path.basename(original_filepath))[0]
            output_filename = f"{base_name}_{cluster_name}{file_ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save according to format
            if file_ext == '.csv':
                cluster_data.to_csv(output_path, index=False)
            elif file_ext in ['.xlsx', '.xls']:
                cluster_data.to_excel(output_path, index=False)
            
            # Statistics
            n_points = len(cluster_data)
            percentage = (n_points / len(df_full)) * 100
            
            print(f"  {cluster_name}: {n_points} points ({percentage:.1f}%) → {output_filename}")
            created_files.append(output_path)
        
        print(f"\nTotal: {len(created_files)} files created in '{output_dir}/'")
        print("="*60)
        
        return created_files
    
    def get_cluster_statistics(self):
        """
        Return detailed statistics for each cluster.
        
        Returns:
            dict: Statistics per cluster
        """
        if self.labels is None:
            raise ValueError("First run fit() on the data.")
        
        if self.data is None:
            raise ValueError("No data available.")
        
        stats = {}
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            mask = (self.labels == label)
            cluster_data = self.data.iloc[mask]
            
            if len(cluster_data) > 0:
                cluster_stats = {
                    'n_points': len(cluster_data),
                    'percentage': (len(cluster_data) / len(self.labels)) * 100
                }
                
                # Add statistics if enough points
                if len(cluster_data) > 1:
                    cluster_stats['mean'] = cluster_data.mean().to_dict()
                    cluster_stats['std'] = cluster_data.std().to_dict()
                
                cluster_stats['min'] = cluster_data.min().to_dict()
                cluster_stats['max'] = cluster_data.max().to_dict()
                
                if label == -1:
                    stats['noise'] = cluster_stats
                else:
                    stats[f'cluster_{label}'] = cluster_stats
        
        return stats
