pip install pandas numpy
# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Datasets/Survey Lung Cancer/survey lung cancer.csv')

print("First 5 rows of the dataset:")
data.head()

print("\nDataset Information:")
print(data.info())

missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Preprocessing - Encoding categorical variables
label_encoder = LabelEncoder()
data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])

# Print the entire DataFrame to see the encoded columns
print(data)

# Alternatively, print specific columns
print("Encoded GENDER values:\n", data['GENDER'])
print("Encoded LUNG_CANCER values:\n", data['LUNG_CANCER'])

# Print unique encoded values for clarity
print("Unique values for GENDER (encoded):", data['GENDER'].unique())
print("Unique values for LUNG_CANCER (encoded):", data['LUNG_CANCER'].unique())

# Print unique values for GENDER
print("Unique values for GENDER (encoded):", data['GENDER'].unique())

# Map back to original labels for clarity
gender_mapping = {0: 'Female', 1: 'Male'}
print("Mapping of encoded values to original labels:", gender_mapping)

# Define features and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Calculate and plot the correlation matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Class distribution check
class_distribution = data['LUNG_CANCER'].value_counts()
print(class_distribution)

import matplotlib.pyplot as plt
import seaborn as sns

# Class distribution plot
sns.countplot(x='LUNG_CANCER', data=data)
plt.title('Class Distribution of Lung Cancer')
plt.xlabel('Lung Cancer (0: No, 1: Yes)')
plt.ylabel('Number of Samples')
plt.show()

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features for models like SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Dictionary to store results
results = {}

# Function to train, predict, and evaluate each model
for model_name, model in models.items():
    print(f"\n--- {model_name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store results
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    # Print the classification report for detailed metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')  # You can change the color map if you like
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Display all model comparison results
print("\nModel Comparison:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.2f}")

import matplotlib.pyplot as plt
import numpy as np

# Extract metrics for each model from the results dictionary
model_names = list(results.keys())
accuracies = [results[model]["Accuracy"] for model in model_names]
precisions = [results[model]["Precision"] for model in model_names]
recalls = [results[model]["Recall"] for model in model_names]
f1_scores = [results[model]["F1 Score"] for model in model_names]

# Define the bar width and positions for each metric group
bar_width = 0.2
r1 = np.arange(len(model_names))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plot each metric
plt.figure(figsize=(12, 6))

# Bar plots for each metric with updated colors
plt.bar(r1, accuracies, color='orange', width=bar_width, edgecolor='grey', label='Accuracy')
plt.bar(r2, precisions, color='blue', width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r3, recalls, color='purple', width=bar_width, edgecolor='grey', label='Recall')
plt.bar(r4, f1_scores, color='cyan', width=bar_width, edgecolor='grey', label='F1 Score')

# Adding labels and title
plt.xlabel('Models', fontweight='bold')
plt.xticks([r + bar_width * 1.5 for r in range(len(model_names))], model_names)
plt.ylabel('Scores', fontweight='bold')
plt.title('Model Performance Comparison')

# Add legend
plt.legend()

# Show plot
plt.show()

# Class distribution after SMOTE
new_class_distribution = pd.Series(y_resampled).value_counts()
print(new_class_distribution)

import matplotlib.pyplot as plt
import seaborn as sns

# Class distribution plot after SMOTE
sns.countplot(x=y_resampled)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Lung Cancer (0: No, 1: Yes)')
plt.ylabel('Number of Samples')
plt.show()

# Function to predict lung cancer for all models
def predict_lung_cancer_all_models(models, input_features):

    input_features_scaled = scaler.transform([input_features])  # Scale the features
    predictions = {}

    for model_name, model in models.items():
        prediction = model.predict(input_features_scaled)
        predictions[model_name] = 'Lung Cancer' if prediction[0] == 1 else 'No Lung Cancer'

    return predictions

# Example input features for a new patient
new_patient = [1,  # GENDER (1 for Male, 0 for Female)
               40,  # AGE
               2,   # OTHER FEATURES (adjust as per your dataset)
               1,
               0,
               1,
               2,
               0,
               1,
               0,
               0,
               1,
               1,
               0,
               0]  # Adjust these values according to your dataset

# Get predictions from all models
predictions = predict_lung_cancer_all_models(models, new_patient)

# Check if any model predicts lung cancer and print Yes or No
has_cancer = any(result == 'Lung Cancer' for result in predictions.values())

if has_cancer:
    print("Lung Cancer")
else:
    print("No Lung cancer")
