# -*- coding: utf-8 -*-
"""
Professional Data Preprocessing Script for the Titanic Dataset.

This script loads the Titanic dataset, applies a robust preprocessing pipeline
with feature engineering, visualizes and removes outliers, and saves the 
final cleaned data.
"""

import logging
from pathlib import Path
import yaml

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. CONFIGURATION & SETUP ---

# Load configuration from YAML file
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    INPUT_FILE = Path(config["input_file"])
    OUTPUT_FILE = Path(config["output_file"])
    PLOT_FILE = Path(config["plot_file"])
    LOG_FILE = Path(config["log_file"])
except (FileNotFoundError, KeyError) as e:
    print(f"Error: Could not read configuration from config.yaml. Details: {e}")
    exit()

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

# Use a non-interactive backend for Matplotlib to prevent GUI errors
matplotlib.use('Agg')


# --- 2. CORE FUNCTIONS ---

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads data from a specified CSV file path."""
    try:
        logging.info(f"Loading data from '{filepath}'...")
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file '{filepath}' was not found.")
        raise

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers new features to enhance the dataset."""
    logging.info("Starting feature engineering...")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')
    logging.info("Feature engineering complete.")
    return df

def build_pipeline() -> ColumnTransformer:
    """Builds a scikit-learn pipeline for preprocessing the data."""
    numeric_features = ['Age', 'Fare', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Embarked', 'Sex', 'Title']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', ['IsAlone'])
        ],
        remainder='drop'
    )
    return preprocessor

def visualize_outliers(df: pd.DataFrame, filepath: Path) -> None:
    """Generates and saves a boxplot to visualize outliers before removal."""
    logging.info("Generating outlier visualization...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=df[['Age', 'Fare']], palette="Set2")
    plt.title('Boxplots of Standardized Age and Fare (Before Outlier Removal)', fontsize=16)
    plt.ylabel('Standardized Value', fontsize=12)
    plt.savefig(filepath)
    logging.info(f"Outlier plot saved to '{filepath}'.")

def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Removes outliers from a specified column using the IQR method."""
    logging.info(f"Removing outliers from column: {column}...")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    original_rows = len(df)
    df_filtered = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    rows_removed = original_rows - len(df_filtered)
    logging.info(f"Removed {rows_removed} outlier rows based on '{column}'.")

    return df_filtered

def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """Saves the processed DataFrame to a CSV file."""
    logging.info(f"Saving cleaned data to '{filepath}'...")
    df.to_csv(filepath, index=False)
    logging.info("Cleaned data saved successfully.")


# --- 3. MAIN EXECUTION BLOCK ---

def main():
    """Orchestrates the entire data loading, processing, and saving workflow."""
    logging.info("--- Starting Data Preprocessing Workflow ---")

    raw_df = load_data(INPUT_FILE)
    df_featured = engineer_features(raw_df.copy())

    features_to_process = ['Age', 'Fare', 'FamilySize', 'Embarked', 'Sex', 'Title', 'IsAlone']
    columns_to_keep = ['Survived', 'Pclass']
    X = df_featured[features_to_process]
    y = df_featured[columns_to_keep]

    preprocessor = build_pipeline()
    logging.info("Applying preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X)

    new_num_features = ['Age', 'Fare', 'FamilySize']
    new_cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(['Embarked', 'Sex', 'Title'])
    new_bin_features = ['IsAlone']
    all_feature_names = new_num_features + list(new_cat_features) + new_bin_features

    processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    df_transformed = pd.concat([y.reset_index(drop=True), processed_df], axis=1)

    logging.info("Preprocessing pipeline applied successfully.")

    # Visualize outliers BEFORE removing them
    visualize_outliers(df_transformed, PLOT_FILE)

    # Remove outliers from the transformed data
    logging.info(f"Shape before removing outliers: {df_transformed.shape}")
    df_final = remove_outliers(df_transformed, 'Fare')
    df_final = remove_outliers(df_final, 'Age')
    logging.info(f"Shape after removing outliers: {df_final.shape}")

    # Save the final, cleaned data
    save_data(df_final, OUTPUT_FILE)

    logging.info("--- Data Preprocessing Workflow Finished Successfully ---")


if __name__ == "__main__":
    main()

