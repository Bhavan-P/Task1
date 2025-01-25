# Titanic Survival Prediction 🚢

This project uses a **Random Forest Classifier** to predict the survival of passengers on the Titanic based on various features like age, gender, class, and more. The dataset used is the popular [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

---

## Features of the Project 📋

1. **Data Cleaning and Preprocessing**:
   - Handles missing values for key features (`Age`, `Embarked`).
   - Drops irrelevant columns (`Name`, `Ticket`, `Cabin`).
   - Encodes categorical variables (`Sex`, `Embarked`) for machine learning.

2. **Machine Learning Model**:
   - Implements a **Random Forest Classifier** for prediction.
   - Splits data into training (80%) and testing (20%) sets.
   - Evaluates performance using metrics like accuracy and a confusion matrix.

3. **Prediction Results**:
   - Outputs overall model accuracy.
   - Displays a confusion matrix for performance analysis.
   - Lists survival predictions for individual passengers in the test set.

## Installation and Dependencies🚀
       pip install -r requirements.txt
## How to Run the Script
To execute the script, run the following command in your terminal or command prompt:
   ```bash
     python Titan.py
