import os
import pandas as pd
import numpy as np


# Function to build the LLM prompt based on selected features
def build_prompt(df, train_indices, include_research_background=True, include_chemical=True,
                 include_data_description=True,
                 include_impact_condition=True, include_key_features=True):
    """
    Build a structured prompt to input into a large language model (LLM) for predicting CO2 adsorption.

    Parameters:
    - df: DataFrame containing material properties and experimental results.
    - train_indices: List of row indices to be used for training examples.
    - include_research_background: Boolean, whether to include research background in the prompt.
    - include_chemical: Boolean, whether to include chemical data in the prompt.
    - include_data_description: Boolean, whether to include physical data description in the prompt.
    - include_impact_condition: Boolean, whether to include the impact condition information in the prompt.
    - include_key_features: Boolean, whether to highlight key features affecting CO2 adsorption.

    Returns:
    - A string containing the formatted prompt.
    """
    prompt = (
        "Instruction: You are a model that predicts the CO2 adsorption capacity of a material based on its physical and chemical properties.\n\n"
    )

    if include_research_background:
        prompt += (
            "## Research Background\n"
            "A novel approach is proposed by dispersing amine on a solid support surface to form a promising adsorbent for direct air capture with improved adsorption performance."
            " Dispersing amine on the solid support surface increases the probability of carbon dioxide binding by active amine molecules due to better exposure and minimal hindrance.\n\n"
        )

    if include_data_description:
        prompt += (
            "## Solid Support Design\n"
            # "### Group One Data Input\n"
            "- **Specific surface area before amine dispersing**: Total surface area per unit mass of the support solid material. Higher values indicate better exposure for amine.\n"
            "- **Pore volume before amine dispersing**: Total volume of void spaces or pores within a material per unit mass or volume before the amine is dispersed. High pore volume supports higher amine loading and better dispersion, thus improving CO2 adsorption capacity.\n"
            "- **Loading amount of amine**: Mass of amine per unit mass of the solid support. Higher loading amounts tend to increase CO2 adsorption but excessive amounts may hinder diffusion and resulted in reduction of CO2 adsorbed amount.\n"
            "- **Pore volume after amine dispersing**: Total volume of void spaces or pores within a material per unit mass or volume after the amine is dispersed. Reduced pore volume indicates amine has filled some pores.\n"
            "- **Specific surface area after amine dispersing**: The surface area per unit mass of a material after amine dispersion. Reduced surface area indicates amine cover the surface. Higher values indicate better exposure of amine sites for CO2 interaction.\n\n"
        )

    if include_chemical:
        prompt += (
            "## Chemical Modification by Amine\n"
            # "### Group Two Data Input\n"
            "- **Type of amine**: Different types of amines have varying properties that can affect CO2 adsorption.\n"
            "- **Nitrogen atom content in the amine**: The amount of nitrogen atoms presents in the amine molecule; it is a percentage of the total mass or moles of the compound.\n"
            "- **Molecular weight**: The molecular mass of the amine. Higher molecular weight amines adhere firmly but may block pores.\n"
            "- **Primary amine proportion**: Percentage of primary amine groups, which react with CO2 via a zwitterion mechanism to form a carbamate in dry conditions.\n"
            "- **Secondary amine proportion**: Percentage of secondary amine groups, which react with CO2 via a zwitterion mechanism to form a carbamate in dry conditions.\n"
            "- **Tertiary amine proportion**: Percentage of tertiary amine groups, which react with CO2 in the presence of water to form bicarbonate.\n\n"
        )

    if include_impact_condition:
        prompt += (
            "### Impact on CO2 Adsorption\n"
            "The dispersion of amines on the surface of solid materials leads to changes in specific surface area and pore volume. "
            "Amine can be regarded as existing in the form of a thin film on the surface of a solid material. "
            "The difference in surface area before and after dispersing of amine can be regarded as the area of the film. "
            "The difference in pore volume before and after dispersing of amine can be regarded as the volume of the film, so the thickness of the film can be estimated. "
            "It is generally believed that the larger area and thinner of the film, the CO2 adsorption performance is better. "
            "A larger spread area will have more adsorption sites, facilitating the contact between CO2 and amine substances.\n\n"

            'The final nitrogen content on the surface of the solid material is obtained based on the amine loading and the nitrogen atom content in the amine. '
            'These nitrogens exist on the solid surface in the form of different nitrogen-containing functional groups (Primary, secondary, and tertiary amine groups),'
            ' and ultimately achieve CO2 adsorption under the corresponding reaction mechanism with CO2.\n\n'
        )

    if include_key_features:
        prompt += (
            "## Key Features Influencing CO2 Adsorption Capacity\n"
            "- Pore volume before amine dispersing\n"
            "- Loading amount of amine\n"
            "- Specific surface area after amine dispersing\n"
            "- Pore volume after amine dispersing\n"
            "- Temperature\n"
            "- Relative humidity\n\n"
        )
    prompt += 'All features should be considered in the prediction model.\n'

    # Adding examples from the dataset
    examples = df.iloc[train_indices]
    for idx, row in examples.iterrows():
        prompt += "### Example Material:\n"
        prompt += (
            f"- Specific surface area before amine dispersing: {row['Specific surface area before amine dispersing']} m²/g\n"
            f"- Specific surface area after amine dispersing: {row['Specific surface area after amine dispersing']} m²/g\n"
            f"- Pore volume before amine dispersing: {row['Pore volume before amine dispersing']} cm³/g\n"
            f"- Pore volume after amine dispersing: {row['Pore volume after amine dispersing']} cm³/g\n"
            f"- Amine type: {row['Type of amine']}\n"
            f"- Molecular weight: {row['Molecular weight']} g/mol\n"
            f"- Nitrogen atom content: {row['Nitrogen atom content']} %\n"
            f"- Primary amine proportion: {row['Primary amine proportion'] * 100}%\n"
            f"- Secondary amine proportion: {row['Secondary amine proportion'] * 100}%\n"
            f"- Tertiary amine proportion: {row['Tertiary amine proportion'] * 100}%\n"
            f"- Loading amount of amine: {row['Loading amount of amine']} g/g\n"
            f"- Temperature: {row['Temperature']} °C\n"
            f"- Partial pressure of CO2: {row['Partial pressure of CO2']} atm\n"
            f"- Relative humidity: {row['Relative humidity']} %\n"
            f"- CO2 adsorbed amount: {row['CO2 adsorbed amount']} mmol/g\n\n"
        )
    prompt += 'The CO2 adsorbed amount that needs to be predicted are as follows:\n'

    return prompt


# Function to generate test data for prediction
def generate_test_data(df, test_indices):
    """
    Generate test data prompt for unseen data points.

    Parameters:
    - df: DataFrame containing material properties.
    - test_indices: List of row indices to be used for testing.

    Returns:
    - A string containing the test data formatted for prediction.
    """
    global results
    prompt = ''
    for idx, row in df.iloc[test_indices].iterrows():
        prompt += "Test Material:\n"
        prompt += f"  - Specific surface area before amine dispersing: {row['Specific surface area before amine dispersing']} m²/g\n"
        prompt += f"  - Specific surface area after amine dispersing: {row['Specific surface area after amine dispersing']} m²/g\n"
        prompt += f"  - Pore volume before amine dispersing: {row['Pore volume before amine dispersing']} cm³/g\n"
        prompt += f"  - Pore volume after amine dispersing: {row['Pore volume after amine dispersing']} cm³/g\n"
        prompt += f"  - Molecular weight: {row['Molecular weight']} g/mol\n"
        prompt += f"  - Nitrogen atom content: {row['Nitrogen atom content']}\n"
        prompt += f"  - Primary amine proportion: {row['Primary amine proportion'] * 100}%\n"
        prompt += f"  - Secondary amine proportion: {row['Secondary amine proportion'] * 100}%\n"
        prompt += f"  - Tertiary amine proportion: {row['Tertiary amine proportion'] * 100}%\n"
        prompt += f"  - Loading amount of amine: {row['Loading amount of amine']} g/g\n"
        prompt += f"  - Temperature: {row['Temperature']} °C\n"
        prompt += f"  - Partial pressure of CO2: {row['Partial pressure of CO2']} atm\n"
        prompt += f"  - Relative humidity: {row['Relative humidity']} %\n"
        prompt += f"  - CO2 adsorbed amount: \n\n"
      #  results.append(row['CO2 adsorbed amount'])
    return prompt


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

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# Training data indices
train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Generate prompt for training data
prompt = build_prompt(df, train_indices, include_research_background=True, include_data_description=True,
                      include_chemical=True, include_impact_condition=True, include_key_features=True)
# Test data indices
test_indices = list(set(range(len(df))) - set(train_indices))

# Create directory for saving the output files
save_dir = './11/'
os.makedirs(save_dir, exist_ok=True)

# Generate test prompts and save them to files
for idx in range(0, len(test_indices), 5):
    prompt += generate_test_data(df, test_indices[idx:idx + 5])
    with open(f"{save_dir}test_sample_{idx}.txt", "w") as f:
        f.write(prompt)
