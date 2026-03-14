from thermoelectric_prediction import LinearRegressionAnalyzer, print_config

# Afficher configuration
print_config('LINEAR_REGRESSION')

# Initialiser
analyzer = LinearRegressionAnalyzer()

# Charger données
data = analyzer.load_data('data_linear.xlsx')

# Préparer données (calcule sigma si nécessaire)
analyzer.prepare_data()

# Effectuer les régressions
results_sn = analyzer.fit_s_n()
results_pf = analyzer.fit_pf_n()
results_sigma = analyzer.fit_sigma_n()

# Afficher résumé
analyzer.summary()

# Visualiser
analyzer.plot_results(save_figure=True, filename='linear_regression_results.pdf')

# Faire des prédictions
s_p_value = 100
predicted_s_n = analyzer.predict('s_n', s_p_value)
print(f"\nPrédiction pour s_p = {s_p_value}: s_n = {predicted_s_n[0]:.2f}")
