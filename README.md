
# Multiple Disease Prediction System Using Machine Learning

## Overview
This project implements a machine learning-based prediction system capable of detecting multiple diseases such as heart disease, kidney disease, Parkinson’s, and diabetes. The system takes disease-specific parameters and predicts the probability of the user having a particular disease using several machine learning algorithms.

## Key Features
- **Multiple Disease Prediction**: Predicts several diseases (Heart, Kidney, Parkinson’s, and Diabetes) using a single system.
- **High Accuracy**: Uses algorithms like Random Forest, Logistic Regression, and Decision Tree to achieve high prediction accuracy.
- **User-friendly Interface**: Built using Streamlit for ease of use.

## Tools Used
- **Random Forest**: Best performing machine learning model for disease prediction.
- **Streamlit**: A web-based interface for user interaction.
- **Pandas and Scikit-learn**: Libraries used for data manipulation and model building.

## Compilation Instructions
1. Install the required libraries:
   ```
   pip install scikit-learn pandas streamlit
   ```
2. Load the pre-processed datasets for each disease.
3. Train the machine learning models for Heart, Kidney, Parkinson’s, and Diabetes predictions.
4. Use the Streamlit interface to enter patient details and get disease prediction results.

## How It Works
1. **Data Input**: The user provides the system with necessary parameters such as blood pressure, cholesterol levels, and other symptoms.
2. **Model Training**: Multiple machine learning algorithms (Random Forest, Logistic Regression) are trained to predict each disease.
3. **Prediction**: The system outputs whether the patient is at risk for each disease based on the input parameters.

## Example Code (Disease Prediction Snippet)
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Authors
- Gudapareddy Mani Prakash Reddy
- Konda Reddy Balaji Reddy
- Bobburi Sai Kowshik

## Acknowledgments
- Project guided by Amrita Vishwa Vidyapeetham.
