# simple_example.py
from thermoelectric_prediction import LinearRegressionAnalyzer

# Initialize (automatically uses config.py)
analyzer = LinearRegressionAnalyzer()

# Load data
analyzer.load_data('data_linear.xlsx')

# Perform the regression
analyzer.fit()

# Visualize
analyzer.plot()

# Prediction (optional)
pred = analyzer.model.predict([[100]])[0]
print(f"\nPrediction for {analyzer.config['FEATURE']} = 100: {pred:.2f}")

