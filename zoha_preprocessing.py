import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# File path for Kaggle dataset
file_path = "/kaggle/input/brats2020-training-data/BraTS2020_training_data/content/data/meta_data.csv"

# Step 1: Load the original dataset
original_meta_data = pd.read_csv(file_path)

# Step 2: Show missing values before cleaning
print("Missing values before cleaning:\n", original_meta_data.isnull().sum())

# Step 3: Clean the data (drop duplicates, handle missing values, ensure numeric columns)
meta_data = original_meta_data.copy()  # Make a copy for cleaning
meta_data.drop_duplicates(inplace=True)
meta_data.dropna(inplace=True)
meta_data['target'] = pd.to_numeric(meta_data['target'], errors='coerce')
meta_data['volume'] = pd.to_numeric(meta_data['volume'], errors='coerce')
meta_data['slice'] = pd.to_numeric(meta_data['slice'], errors='coerce')

# Step 4: Missing values after cleaning
print("\nMissing values after cleaning:\n", meta_data.isnull().sum())

# Step 5: Compare row count before and after cleaning
print(f"\nOriginal dataset row count: {original_meta_data.shape[0]}")
print(f"Cleaned dataset row count: {meta_data.shape[0]}")

# Step 6: Plot distributions of numerical columns for comparison (before vs after cleaning)

# Distribution for 'target', 'volume', 'slice' in the original data
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# 'target' distribution
sns.countplot(x='target', data=original_meta_data, ax=axes[0, 0])
axes[0, 0].set_title('Original Target Distribution')
sns.countplot(x='target', data=meta_data, ax=axes[0, 1])
axes[0, 1].set_title('Cleaned Target Distribution')

# 'volume' distribution
sns.histplot(original_meta_data['volume'], kde=True, ax=axes[1, 0], color='blue')
axes[1, 0].set_title('Original Volume Distribution')
sns.histplot(meta_data['volume'], kde=True, ax=axes[1, 1], color='blue')
axes[1, 1].set_title('Cleaned Volume Distribution')

# 'slice' distribution
sns.histplot(original_meta_data['slice'], kde=True, ax=axes[2, 0], color='green')
axes[2, 0].set_title('Original Slice Distribution')
sns.histplot(meta_data['slice'], kde=True, ax=axes[2, 1], color='green')
axes[2, 1].set_title('Cleaned Slice Distribution')

plt.tight_layout()
plt.show()

# Step 7: Save the cleaned data (optional)
cleaned_file_path = "/kaggle/working/meta_data_cleaned.csv"
meta_data.to_csv(cleaned_file_path, index=False)
print(f"\nPreprocessed data saved to: {cleaned_file_path}")
