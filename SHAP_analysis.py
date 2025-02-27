import os
import argparse
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay


# Function to create directory for saving plots
def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
哈哈哈哈

# Function to preprocess the dataset
def preprocess_data(df, feature_mapping):
    df = df.rename(columns=feature_mapping)
    X = df.drop(columns=['CO2 adsorbed amount', 'Type of amine'])
    y = df['CO2 adsorbed amount']

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X, y


# Function to perform SHAP summary plot
def shap_summary_plot(explainer, shap_values, X_test, feature_mapping, output_dir):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=list(feature_mapping.values()), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_plot.png'))
    plt.close()


# Function to perform SHAP dependence plot
def shap_dependence_plot(shap_values, X_test, feature_mapping, output_dir):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot("PVBD", shap_values, X_test, feature_names=list(feature_mapping.values()), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dependence_plot_PVBD.png'))
    plt.close()


# Function to generate SHAP interaction plots
def shap_interaction_plot(explainer, X_test, feature_mapping, output_dir, unit_mapping):
    interaction_values = explainer.shap_interaction_values(X_test)

    for i in range(len(feature_mapping)):
        for j in range(i + 1, len(feature_mapping)):
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                (i, j), interaction_values, X_test,
                feature_names=list(feature_mapping.values()),
                interaction_index="auto", show=False
            )
            xlabel_text = list(feature_mapping.values())[i]
            if unit_mapping[xlabel_text]:
                xlabel_text += f' ({unit_mapping[xlabel_text]})'
            plt.xlabel(xlabel_text)

            colorbar = plt.gcf().axes[-1]
            ylabel = list(feature_mapping.values())[j]
            if unit_mapping[ylabel]:
                ylabel += f' ({unit_mapping[ylabel]})'
            colorbar.set_ylabel(ylabel)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'interaction_plot_{i}_{j}.png'))
            plt.close()


# Function to create SHAP force plot
def shap_force_plot(explainer, shap_values, X_test, output_dir):
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
    shap.save_html(os.path.join(output_dir, 'force_plot.html'), force_plot)


# Function to create SHAP waterfall plot
def shap_waterfall_plot(shap_values, output_dir):
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waterfall_plot.png'))
    plt.close()


# Function to perform partial dependence plot
def partial_dependence_plot(model, X_test, feature_mapping, output_dir, unit_mapping):
    for feature_name in feature_mapping.values():
        plt.figure(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            model, X_test, [list(feature_mapping.values()).index(feature_name)],
            feature_names=list(feature_mapping.values())
        )
        xlabel_text = feature_name
        if unit_mapping[feature_name]:
            xlabel_text += f' ({unit_mapping[feature_name]})'
        plt.xlabel(xlabel_text)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'partial_dependence_plot_{feature_name}.png'))
        plt.close()


# Main function to handle arguments and call corresponding SHAP analysis functions
def main(args):
    # Create output directory
    create_output_dir(args.output_dir)

    # Feature mapping and units
    feature_mapping = {
        'Specific surface area before amine dispersing': 'SSABD',
        'Specific surface area after amine dispersing': 'SSAAD',
        'Pore volume before amine dispersing': 'PVBD',
        'Pore volume after amine dispersing': 'PVAD',
        'Molecular weight': 'MW',
        'Nitrogen atom content': 'NAC',
        'Primary amine proportion': 'PAP',
        'Secondary amine proportion': 'SAP',
        'Tertiary amine proportion': 'TAP',
        'Loading amount of amine': 'LA',
        'Temperature': 'Temp',
        'Partial pressure of CO2': 'PPCO$_2$',
        'Relative humidity': 'RH'
    }

    unit_mapping = {
        'SSABD': 'm²/g',
        'SSAAD': 'm²/g',
        'PVBD': 'cm³/g',
        'PVAD': 'cm³/g',
        'MW': 'g/mol',
        'NAC': '-',
        'PAP': '-',
        'SAP': '-',
        'TAP': '-',
        'LA': 'g/g',
        'Temp': '°C',
        'PPCO$_2$': 'ppm',
        'RH': '-'
    }

    # Sample data (replace with actual data loading process)
    data = {
        'Specific surface area before amine dispersing': [643.03, 643.03, 643.03, 643.03, 643.03, 643.03, 185, 185, 185,
                                                          526, 526, 526, 999, 999, 999, 999, 700, 700, 700],
        'Specific surface area after amine dispersing': [205.98, 13.089, 1.87, 250.38, 174.1, 0.83, 110, 31, 20, 116,
                                                         17, 8, 153, 104, 71, 46, 389, 332, 147],
        'Pore volume before amine dispersing': [1.1441, 1.1441, 1.1441, 1.1441, 1.1441, 1.1441, 1.15, 1.15, 1.15, 0.959,
                                                0.959, 0.959, 3.1, 3.1, 3.1, 3.1, 0.96, 0.96, 0.96],
        'Pore volume after amine dispersing': [0.4717, 0.0505, 0.0027, 0.6024, 0.3533, 0.0018, 0.642, 0.258, 0.02, 0.36,
                                               0.033, 0.01, 1.11, 0.69, 0.46, 0.34, 0.68, 0.6, 0.27],
        'Type of amine': ['PEI(linear)', 'PEI(linear)', 'PEI(linear)', 'TEPA', 'TEPA', 'TEPA', 'PEI(branced)',
                          'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)',
                          'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PEI(branced)', 'PGA', 'PGA', 'PGA'],
        'Molecular weight': [800, 800, 800, 189.3, 189.3, 189.3, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 1813,
                             1813, 1813],
        'Nitrogen atom content': [0.3256, 0.3256, 0.3256, 0.37, 0.37, 0.37, 0.3256, 0.3256, 0.3256, 0.3256, 0.3256,
                                  0.3256, 0.3256, 0.3256, 0.3256, 0.3256, 0.1916, 0.1916, 0.1916],
        'Primary amine proportion': [0, 0, 0, 0.4, 0.4, 0.4, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364,
                                     0.364, 0.364, 1, 1, 1],
        'Secondary amine proportion': [1, 1, 1, 0.6, 0.6, 0.6, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273,
                                       0.273, 0.273, 0, 0, 0],
        'Tertiary amine proportion': [0, 0, 0, 0, 0, 0, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364,
                                      0.364, 0, 0, 0],
        'Loading amount of amine': [0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.33, 0.5, 0.67, 0.33, 0.5, 0.67, 0.55, 0.6, 0.65,
                                    0.7, 0.26, 0.31, 0.41],
        'Temperature': [25] * 16 + [35] * 3,
        'Partial pressure of CO2': [400] * 15 + [5000] * 4,
        'Relative humidity': [0] * 19,
        'CO2 adsorbed amount': [0.72, 1.34, 1.9, 0.89, 2.3, 3.44, 1.08, 1.66, 2.27, 1.08, 1.46, 1.92, 2.56, 2.83, 2.75,
                                2.12, 0.14, 0.3, 0.66]
    }

    # Convert data into a DataFrame
    df = pd.DataFrame(data)

    # Preprocess data
    X, y = preprocess_data(df, feature_mapping)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Run selected SHAP analyses
    if args.shap_type == 'summary':
        shap_summary_plot(explainer, shap_values, X_test, feature_mapping, args.output_dir)
    elif args.shap_type == 'dependence':
        shap_dependence_plot(shap_values, X_test, feature_mapping, args.output_dir)
    elif args.shap_type == 'interaction':
        shap_interaction_plot(explainer, X_test, feature_mapping, args.output_dir, unit_mapping)
    elif args.shap_type == 'force':
        shap_force_plot(explainer, shap_values, X_test, args.output_dir)
    elif args.shap_type == 'waterfall':
        shap_waterfall_plot(shap_values, args.output_dir)
    elif args.shap_type == 'partial_dependence':
        partial_dependence_plot(model, X_test, feature_mapping, args.output_dir, unit_mapping)
    else:
        print(
            "Invalid SHAP analysis type. Please choose from 'summary', 'dependence', 'interaction', 'force', 'waterfall', 'partial_dependence'.")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Run SHAP analysis on CO2 adsorption data.')
    parser.add_argument('--shap_type', type=str, required=True,
                        help="Type of SHAP analysis: 'summary', 'dependence', 'interaction', 'force', 'waterfall', 'partial_dependence'.")
    parser.add_argument('--output_dir', type=str, default='./SHAP_results', help="Directory to save SHAP plots.")
    args = parser.parse_args()

    # Run main function
    main(args)
