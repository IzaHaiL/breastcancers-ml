# Import libraries
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = np.asarray(pd.read_csv("data/concat_output/breast_cancer.csv", header=None))

# Separate features and target from the dataset
X = df[:, 0:-1]

# Normalize data using Min-Max scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_normalized)

y = df[:, -1][np.newaxis].T
y = y.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3)

# Create MLP classifier object
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)

# Train the model using the training data with tqdm
print("Training the model...")
for epoch in tqdm(range(100), desc="Training progress", unit="epoch"):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y))

# Make predictions using the test data
y_pred = mlp.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
mse = mean_squared_error(y_test, y_pred)

# Print the results
print("Training completed.")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Mean Squared Error:", mse)
