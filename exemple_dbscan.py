from thermoelectric_prediction import DBSCANAnalyzer, print_config

# Display the current configuration
print_config()

# Initialize with custom parameters
analyzer = DBSCANAnalyzer()

# Load data from CSV
data = analyzer.load_data('data_dbscan.xlsx')

# Apply DBSCAN
labels = analyzer.fit()

# Display a summary
analyzer.summary()

# SAVE ALL CLUSTERS (NEW!)
print("\nSaving individual clusters...")
saved_files = analyzer.save_clusters('data_dbscan.xlsx')
print(f"Files created: {saved_files}")

# Visualize clusters
analyzer.plot_clusters(save_figure=True, filename='my_clusters.pdf')

# Save detailed results
analyzer.save_results('detailed_results.csv')

