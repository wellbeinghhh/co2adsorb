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

The `prompt_generation.py` script prepares data prompts for training large language models (LLMs), specifically for predicting CO₂ adsorption performance based on various material properties.

### Steps for Prompt Generation:

1. **Build and Format the Prompts**:
   The script generates prompts based on material properties such as surface area, pore volume, amine type, and experimental conditions. You can control the level of detail included in the prompt using different parameters such as whether to include chemical properties, research background, or physical data description.

2. **Generate Prompts for Training and Testing**:

    Example usage:
    
    ```python
    import pandas as pd
    from prompt_generation import build_prompt, generate_test_data

    # Load the dataset
    df = pd.read_csv('data/material_performance.csv')

    # Specify training data indices
    train_indices = [0, 1, 2, 3, 4, 5]

    # Generate training prompts
    prompt = build_prompt(df, train_indices, include_research_background=True,
                          include_chemical=True, include_data_description=True,
                          include_impact_condition=True, include_key_features=True)

    # Print the generated prompt for review
    print(prompt)
    ```

3. **Generate Test Prompts**:
   To generate test prompts based on the same dataset for LLM evaluation:

    ```python
    # Specify test data indices
    test_indices = list(set(range(len(df))) - set(train_indices))

    # Generate test prompts and save them to a file
    for idx in range(0, len(test_indices), 5):
        test_prompt = generate_test_data(df, test_indices[idx:idx + 5])
        with open(f"test_sample_{idx}.txt", "w") as f:
            f.write(test_prompt)
    ```

4. **Save the Prompts**:
   The generated prompts will be saved as text files for input into the LLM (e.g., ChatGPT API or any other LLM model).


## Machine Learning Models

The `train_baseline.py` script is used to train and evaluate baseline machine learning models, such as Random Forest, for predicting CO₂ adsorption performance. It includes functionality for:

- Loading and preprocessing the dataset
- Training machine learning models
- Evaluating models using metrics like MAE, MSE, R², and correlation
- Visualizing the performance of different models and feature combinations

You can modify the script to use other machine learning models or customize the dataset as needed.

## SHAP Analysis

The `SHAP_analysis.py` script performs feature importance analysis using SHAP (SHapley Additive exPlanations). It supports multiple types of SHAP visualizations, which can be selected using command-line arguments.

### Available SHAP Analysis Options:
- **Summary Plot**: Displays an overview of feature importance across all samples.
- **Dependence Plot**: Shows the effect of a specific feature on the model's output.
- **Interaction Plot**: Visualizes interactions between features.
- **Force Plot**: Explains individual predictions by showing the contribution of each feature.
- **Waterfall Plot**: Breaks down a single prediction into its feature contributions.
- **Partial Dependence Plot**: Shows the marginal effect of individual features on predictions.

### Example Usage:

1. **Run SHAP Analysis by specifying the type of plot**:
    ```bash
    python SHAP_analysis.py --shap_type summary --output_dir ./SHAP_results
    ```

---

This setup allows you to:
- Prepare prompts for LLMs
- Train traditional ML models (Random Forest, XGBoost, etc.)
- Perform SHAP analysis for feature importance

Feel free to modify or extend the code for your own research purposes.
