import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns


# File path for Kaggle dataset

file_path = "/kaggle/input/brats2020-training-data/BraTS2020_training_data/content/data/name_mapping.csv"


# Step 1: Load the original dataset

original_name_mapping = pd.read_csv(file_path)


# Step 2: Show missing values before cleaning

print("Missing values before cleaning:\n", original_name_mapping.isnull().sum())


# Step 3: Remove duplicates

name_mapping = original_name_mapping.copy()  # Make a copy for cleaning

name_mapping.drop_duplicates(inplace=True)


# Step 4: Handle missing values (Drop rows with missing values for simplicity)

name_mapping.dropna(inplace=True)


# Step 5: Ensure correct formatting for columns (e.g., `BraTS_2017_subject_ID` etc.)

# If any columns are meant to be numeric, you can convert them as follows (assuming ID columns are strings):

name_mapping['BraTS_2017_subject_ID'] = name_mapping['BraTS_2017_subject_ID'].astype(str)

name_mapping['BraTS_2018_subject_ID'] = name_mapping['BraTS_2018_subject_ID'].astype(str)

name_mapping['BraTS_2019_subject_ID'] = name_mapping['BraTS_2019_subject_ID'].astype(str)

name_mapping['BraTS_2020_subject_ID'] = name_mapping['BraTS_2020_subject_ID'].astype(str)


# Step 6: Validate file paths (optional) - Check if the paths in 'BraTS_2017_subject_ID' column exist

invalid_paths = [path for path in name_mapping['BraTS_2017_subject_ID'] if not os.path.exists(path)]

if invalid_paths:

    print(f"Invalid paths found: {len(invalid_paths)}")

else:

    print("All paths are valid.")


# Step 7: Missing values after cleaning

print("\nMissing values after cleaning:\n", name_mapping.isnull().sum())


# Step 8: Compare row count before and after cleaning

print(f"\nOriginal dataset row count: {original_name_mapping.shape[0]}")

print(f"Cleaned dataset row count: {name_mapping.shape[0]}")


# Step 9: Visualize distribution of 'Grade' (or other categorical features)

plt.figure(figsize=(8, 6))

sns.countplot(x='Grade', data=name_mapping)

plt.title("Distribution of Grades in the Cleaned Data")

plt.xlabel('Grade')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()


# Step 10: Save the cleaned data

cleaned_file_path = "/kaggle/working/name_mapping_cleaned.csv"

name_mapping.to_csv(cleaned_file_path, index=False)

print(f"\nPreprocessed data saved to: {cleaned_file_path}")


