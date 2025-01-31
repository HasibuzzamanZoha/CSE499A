import random
import matplotlib.pyplot as plt

# Define the path to your data folder
data_folder = '/kaggle/input/brats2020-training-data/BraTS2020_training_data/content/data/'

# List all .h5 files
all_files = [file for file in os.listdir(data_folder) if file.endswith('.h5')]

# Shuffle the files for random splitting
random.shuffle(all_files)

# Split into training (70%), validation (15%), and test (15%)
train_size = int(0.7 * len(all_files))
val_size = int(0.15 * len(all_files))
test_size = len(all_files) - train_size - val_size

# Define the splits
train_files = all_files[:train_size]
val_files = all_files[train_size:train_size + val_size]
test_files = all_files[train_size + val_size:]

# Plot the distribution in a pie chart
sizes = [len(train_files), len(val_files), len(test_files)]
labels = ['Training Set', 'Validation Set', 'Test Set']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title("Dataset Splitting (Train-Val-Test)")
plt.axis('equal')
plt.show()

# Print the counts for verification
print(f"Training Set: {len(train_files)} files")
print(f"Validation Set: {len(val_files)} files")
print(f"Test Set: {len(test_files)} files")

