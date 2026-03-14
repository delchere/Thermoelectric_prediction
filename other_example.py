#!/usr/bin/env python3
"""
Simple example of using the other_model.py module
"""

from thermoelectric_prediction import ModelPredictor, print_config

# 1. Display the configuration
print_config('OTHER_MODELS')

# 2. Create a predictor for s_p using Random Forest
#the target column and the model type can be change according to your desire
print("\nExample: Predicting s_p with Random Forest")
predictor = ModelPredictor(target_column='s_p', model_type='random_forest')

# 3. Load the data
X, y, df = predictor.load_data('other_data.xlsx')

# 4. Train the model
results = predictor.train(X, y)

# 5. Evaluate
predictor.evaluate()

# 6. Visualize predictions
predictor.plot_predictions()

# 7. Feature importance
predictor.plot_feature_importance()

# 8. Summary
predictor.summary()

# 9. Save the model
predictor.save_model()

