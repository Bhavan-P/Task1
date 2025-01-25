import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# Step 1: Load the dataset from a local file
data = pd.read_csv('Titanic-Dataset.csv')  # Replace with the path to your local dataset
# Step 2: Data Exploration and Cleaning
# Drop irrelevant columns
data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
# Step 3: Encode categorical variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
# Step 4: Separate features and target
X = data.drop(columns=['Survived'])  # Features
y = data['Survived']  # Target
# Step 5: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 6: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)
# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Step 9: Output the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
# Step 10: Show whether each passenger survived or not in the test set
survival = ['Survived' if pred == 1 else 'Did not survive' for pred in y_pred]
prediction_df = pd.DataFrame({
    'PassengerId': X_test.index,
    'Survival Prediction': survival
})
print("\nSurvival Predictions for Test Set Passengers:")
print(prediction_df)