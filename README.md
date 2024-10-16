# Methodology for Predicting Material Performance by Context-based Modeling: A Case Study on Solid Amine CO2 Adsorbents

## Overview
This repository contains code and datasets for predicting the CO₂ adsorption performance of solid amine adsorbents using both large language models (LLMs) and traditional machine learning (ML) models. 

## System Requirements

### Software Requirements
- Python version 3.6 or above
- Required Python libraries:
    - pandas==2.0.3
    - numpy==1.24.3
    - scikit-learn==1.3.0
    - shap==0.40.0
    - xgboost==1.5.0

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/materials-llm-prediction.git
    cd materials-llm-prediction
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation for LLMs

The dataset consists of numerical and textual data. To format the data for LLMs:

1. Use the script `prepare_llm_prompts.py` to prepare input-output pairs for the LLM model:

    ```python
    import pandas as pd

    # Load dataset
    df = pd.read_csv('data/material_performance.csv')

    # Example of preparing data for LLM prompts
    def generate_prompt(row):
        return f"Surface area: {row['surface_area']} m²/g, Pore volume: {row['pore_volume']} cm³/g, Amine type: {row['amine_type']}. CO2 adsorption uptake: {row['co2_adsorption']} mmol/g."

    df['prompt'] = df.apply(generate_prompt, axis=1)

    # Save the prompts for LLMs
    df[['prompt', 'co2_adsorption']].to_csv('llm_prompts.csv', index=False)
    ```

2. Feed these prepared prompts into the LLM model (e.g., ChatGPT API).

## Machine Learning Models

To train and evaluate traditional ML models, follow these steps:

1. Preprocess the dataset by extracting features (e.g., specific surface area, pore volume):

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load the dataset
    df = pd.read_csv('data/material_performance.csv')

    # Extract input features and target
    X = df[['surface_area', 'pore_volume', 'amine_type', 'temperature', 'humidity']]
    y = df['co2_adsorption']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```

2. Train the ML models (e.g., Random Forest):

    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}, R²: {r2}")
    ```

## SHAP Analysis

To perform feature importance analysis using SHAP:

1. Install SHAP if you haven't:
    ```bash
    pip install shap
    ```

2. Run SHAP analysis on the trained model:

    ```python
    import shap

    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Visualize the SHAP summary plot
    shap.summary_plot(shap_values, X_test)
    ```

## Results

All evaluation results, including metrics (MSE, R²) and SHAP importance values, will be saved and visualized in the `results/` directory.

---

This setup allows you to:
- Prepare prompts for LLMs
- Train traditional ML models (Random Forest, XGBoost, etc.)
- Perform SHAP analysis for feature importance

Feel free to modify or extend the code for your own research purposes.
