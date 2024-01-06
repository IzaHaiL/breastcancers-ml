import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm

# Load data from CSV files
print("Loading data...")
data1 = np.asarray(pd.read_csv("data/vgg19_output/vgg19_breast_cancer.csv", header=None))
data2 = np.asarray(pd.read_csv("data/resnet50_output/resnet50_breast_cancer.csv", header=None))

# Extract class labels from data1
class_data1 = data1[:, -1][np.newaxis].T
class_data1 = class_data1.astype(int)

# Extract features from both datasets
features_data1 = data1[:, 0:-1]
features_data2 = data2[:, 0:-1]

# Concatenate features and class labels
concatenated_data = np.concatenate((features_data1, features_data2), axis=1)
concatenated_data = np.concatenate((concatenated_data, class_data1), axis=1)

# Get the number of features
num_features = concatenated_data.shape[1] - 1
print("Number of features:", num_features)

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(concatenated_data)

# Apply Linear Discriminant Analysis (LDA)
lda = LDA(n_components=1)
print("Applying Linear Discriminant Analysis...")

# Use tqdm to create a progress bar
for _ in tqdm(range(100), desc="LDA Progress", unit="iteration"):
    features_after_lda = lda.fit_transform(scaled_features, np.ravel(class_data1))

# Save the concatenated and processed data to a new CSV file
output_file = "data/concat_output/breast_cancer.csv"
print("Saving data to", output_file)
np.savetxt(output_file, concatenated_data, delimiter=",", fmt='%.10f,' * num_features + '%i')

print("Processing complete.")
