import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from models.CNNModel import CNNModel


# Function to plot the confusion matrix.
def plotConfusionMatrix(confusionMatrix, classNames):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusionMatrix, annot=True, fmt='g', xticklabels=classNames, yticklabels=classNames)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# Load the trained model.
model = CNNModel()
ROOT = os.path.join(os.path.dirname(__file__), '..')
model.load_state_dict(torch.load(os.path.join(ROOT, "models/CNN_model_combined_SIBI.pth")))
model.eval()

# Load the alphabet testing dataset.
alphabet_data = pd.read_excel(os.path.join(ROOT, "data/alphabet_testing_data (1).xlsx"), header=0)
alphabet_data.pop("CHARACTER")
alphabet_groupValue, alphabet_coordinates = alphabet_data.pop("GROUPVALUE"), alphabet_data.copy()
alphabet_coordinates = np.reshape(alphabet_coordinates.values, (alphabet_coordinates.shape[0], 63, 1))
alphabet_coordinates = torch.from_numpy(alphabet_coordinates).float()
alphabet_groupValue = alphabet_groupValue.to_numpy()

# Load the number testing dataset.
number_data = pd.read_excel(os.path.join(ROOT, "data/numbers_testing_data (1).xlsx"), header=0)
number_data.pop("CHARACTER")
number_groupValue, number_coordinates = number_data.pop("GROUPVALUE"), number_data.copy()
number_coordinates = np.reshape(number_coordinates.values, (number_coordinates.shape[0], 63, 1))
number_coordinates = torch.from_numpy(number_coordinates).float()
number_groupValue = number_groupValue.to_numpy() + 26  # Shift labels to 26-35

# Combine testing datasets.
coordinates = torch.cat((alphabet_coordinates, number_coordinates), dim=0)
coordinates = [coordinates]
groupValue = np.concatenate((alphabet_groupValue, number_groupValue))

# Make predictions.
predictions = []
with torch.no_grad():
    for inputs in coordinates:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

# Calculate metrics.
accuracy = accuracy_score(predictions, groupValue)
precision = precision_score(groupValue, predictions, average='weighted', zero_division=0)
recall = recall_score(groupValue, predictions, average='weighted', zero_division=0)
f1 = f1_score(groupValue, predictions, average='weighted', zero_division=0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}")
print(f"Recall: {recall * 100:.2f}")
print(f"F1-Score: {f1 * 100:.2f}")

# Define class names for the confusion matrix.
classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# Compute the confusion matrix.
confusionMatrix = confusion_matrix(groupValue, predictions)

# Plot the confusion matrix.
plotConfusionMatrix(confusionMatrix, classNames)
