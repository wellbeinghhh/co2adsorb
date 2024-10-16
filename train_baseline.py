from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data for the analysis
data = {
    'Specific surface area before amine dispersing': [643.03, 643.03, 643.03, 643.03, 643.03, 643.03, 185, 185, 185, 526, 526, 526, 999, 999, 999, 999, 700, 700, 700],
    'Specific surface area after amine dispersing': [205.98, 13.089, 1.87, 250.38, 174.1, 0.83, 110, 31, 20, 116, 17, 8, 153, 104, 71, 46, 389, 332, 147],
    'Pore volume before amine dispersing': [1.1441, 1.1441, 1.1441, 1.1441, 1.1441, 1.1441, 1.15, 1.15, 1.15, 0.959, 0.959, 0.959, 3.1, 3.1, 3.1, 3.1, 0.96, 0.96, 0.96],
    'Pore volume after amine dispersing': [0.4717, 0.0505, 0.0027, 0.6024, 0.3533, 0.0018, 0.642, 0.258, 0.02, 0.36, 0.033, 0.01, 1.11, 0.69, 0.46, 0.34, 0.68, 0.6, 0.27],
    'Type of amine': ['PEI(linear)', 'PEI(linear)', 'PEI(linear)', 'TEPA', 'TEPA', 'TEPA', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PGA', 'PGA', 'PGA'],
    'Molecular weight': [800, 800, 800, 189.3, 189.3, 189.3, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 1813, 1813, 1813],
    'Nitrogen atom content': [0.3256, 0.3256, 0.3256, 0.37, 0.37, 0.37, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.1916, 0.1916, 0.1916],
    'Primary amine proportion': [0, 0, 0, 0.4, 0.4, 0.4, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 1, 1, 1],
    'Secondary amine proportion': [1, 1, 1, 0.6, 0.6, 0.6, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0, 0, 0],
    'Tertiary amine proportion': [0, 0, 0, 0, 0, 0, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0, 0, 0],
    'Loading amount of amine': [0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.33, 0.5, 0.67, 0.33, 0.5, 0.67, 0.55, 0.6, 0.65, 0.7, 0.26, 0.31, 0.41],
    'Temperature': [25] * 16 + [35] * 3,
    'Partial pressure of CO2': [400] * 15 + [5000] * 4,
    'Relative humidity': [0] * 19,
    'CO2 adsorbed amount': [0.72, 1.34, 1.9, 0.89, 2.3, 3.44, 1.08, 1.66, 2.27, 1.08, 1.46, 1.92, 2.56, 2.83, 2.75, 2.12, 0.14, 0.3, 0.66]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Calculate performance metrics for the models
def compute_metrics(gt, predictions):
    mae = mean_absolute_error(gt, predictions)
    mse = mean_squared_error(gt, predictions)
    r2 = r2_score(gt, predictions)
    correlation_matrix = np.corrcoef(gt, predictions)
    correlation = correlation_matrix[0, 1]
    return mae, mse, r2, correlation

# Extract GT (Ground Truth) and predictions for various models
gt = df['CO2 adsorbed amount']
only_data = df['CO2 adsorbed amount'] * 0.9  # Replace with actual model predictions
impact_condition = df['CO2 adsorbed amount'] * 1.1  # Replace with actual model predictions
meaning_input_parameters = df['CO2 adsorbed amount'] * 1.05  # Replace with actual model predictions
full_prompt = df['CO2 adsorbed amount'] * 0.95  # Replace with actual model predictions

# Compute metrics for each method
mae_only_data, mse_only_data, r2_only_data, correlation_only_data = compute_metrics(gt, only_data)
mae_impact_condition, mse_impact_condition, r2_impact_condition, correlation_impact_condition = compute_metrics(gt, impact_condition)
mae_meaning_input_parameters, mse_meaning_input_parameters, r2_meaning_input_parameters, correlation_meaning_input_parameters = compute_metrics(gt, meaning_input_parameters)
mae_full_prompt, mse_full_prompt, r2_full_prompt, correlation_full_prompt = compute_metrics(gt, full_prompt)

# Summarize results into a DataFrame
results = {
    'Method': ['Only data', 'Impact condition', 'Meaning of input parameters', 'Full prompt'],
    'MAE': [mae_only_data, mae_impact_condition, mae_meaning_input_parameters, mae_full_prompt],
    'MSE': [mse_only_data, mse_impact_condition, mse_meaning_input_parameters, mse_full_prompt],
    'R2': [r2_only_data, r2_impact_condition, r2_meaning_input_parameters, r2_full_prompt],
    'Correlation': [correlation_only_data, correlation_impact_condition, correlation_meaning_input_parameters, correlation_full_prompt]
}

results_df = pd.DataFrame(results)

# Plot performance metrics for different methods
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Set colors for better visual differentiation
colors = ['skyblue', 'lightgreen', 'salmon', 'orchid']

# Plot MAE
axs[0, 0].bar(results_df['Method'], results_df['MAE'], color=colors[0])
axs[0, 0].set_title('Mean Absolute Error (MAE)')
axs[0, 0].set_ylabel('MAE')
axs[0, 0].tick_params(axis='x', rotation=45)

# Add value labels to the bars
for i, v in enumerate(results_df['MAE']):
    axs[0, 0].text(i, v + 0.01 if v < 0.4 else v - 0.05, f"{v:.2f}", ha='center', va='bottom' if v < 0.4 else 'top')

# Plot MSE
axs[0, 1].bar(results_df['Method'], results_df['MSE'], color=colors[1])
axs[0, 1].set_title('Mean Squared Error (MSE)')
axs[0, 1].set_ylabel('MSE')
axs[0, 1].tick_params(axis='x', rotation=45)

# Add value labels to the bars
for i, v in enumerate(results_df['MSE']):
    axs[0, 1].text(i, v + 0.01 if v < 0.35 else v - 0.05, f"{v:.2f}", ha='center', va='bottom' if v < 0.35 else 'top')

# Plot R2
axs[1, 0].bar(results_df['Method'], results_df['R2'], color=colors[2])
axs[1, 0].set_title('R-squared (R2)')
axs[1, 0].set_ylabel('R2')
axs[1, 0].tick_params(axis='x', rotation=45)

# Add value labels to the bars
for i, v in enumerate(results_df['R2']):
    axs[1, 0].text(i, v + 0.01 if v < 0.7 else v - 0.05, f"{v:.2f}", ha='center', va='bottom' if v < 0.7 else 'top')

# Plot Correlation
axs[1, 1].bar(results_df['Method'], results_df['Correlation'], color=colors[3])
axs[1, 1].set_title('Correlation')
axs[1, 1].set_ylabel('Correlation')
axs[1, 1].tick_params(axis='x', rotation=45)

# Add value labels to the bars
for i, v in enumerate(results_df['Correlation']):
    axs[1, 1].text(i, v + 0.01 if v < 0.9 else v - 0.05, f"{v:.2f}", ha='center', va='bottom' if v < 0.9 else 'top')

plt.tight_layout()
plt.show()
