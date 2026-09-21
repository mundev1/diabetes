# Diabetes Prediction Using Machine Learning

## Overview

This project predicts whether a person is diabetic or not based on medical attributes using a Support Vector Machine (SVM) classifier. The model is trained on the Pima Indians Diabetes Dataset and uses data preprocessing techniques to improve prediction accuracy.

## Features

* Data Analysis and Exploration
* Data Standardization
* Train-Test Split
* Support Vector Machine (SVM) Classification
* Model Evaluation using Accuracy Score
* Diabetes Prediction for New Patient Data

## Dataset

The project uses the `diabetes.csv` dataset containing the following features:

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| Pregnancies              | Number of pregnancies                            |
| Glucose                  | Plasma glucose concentration                     |
| BloodPressure            | Diastolic blood pressure                         |
| SkinThickness            | Triceps skinfold thickness                       |
| Insulin                  | 2-Hour serum insulin                             |
| BMI                      | Body Mass Index                                  |
| DiabetesPedigreeFunction | Diabetes pedigree function                       |
| Age                      | Age of the patient                               |
| Outcome                  | Diabetes status (0 = Non-Diabetic, 1 = Diabetic) |

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas:

```python
diabetes_dataset = pd.read_csv("diabetes.csv")
```

### 2. Data Exploration

* Display dataset information
* Check dataset dimensions
* Generate descriptive statistics
* Analyze class distribution
* Calculate grouped means

### 3. Data Preprocessing

Features and target variables are separated:

```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

### 4. Feature Scaling

StandardScaler is applied to normalize feature values:

```python
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
```

### 5. Data Splitting

The dataset is divided into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

### 6. Model Training

A Support Vector Machine (SVM) classifier with a linear kernel is trained:

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 7. Model Evaluation

Accuracy is calculated on both training and testing datasets:

```python
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

### 8. Prediction System

The model can predict diabetes status for new patient data.

Example:

```python
input_data = (5,166,72,19,175,25.8,0.587,51)
```

Output:

```text
The person is diabetic
```

or

```text
The person is not diabetic
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd diabetes-prediction
```

Install required libraries:

```bash
pip install numpy pandas scikit-learn
```

## Run the Project

```bash
python diabetes_prediction.py
```

## Model Performance

The model performance is evaluated using:

* Training Accuracy
* Testing Accuracy

Accuracy Score Formula:

```python
accuracy_score(actual_values, predicted_values)
```

## Project Structure

```text
Diabetes-Prediction/
│
├── diabetes.csv
├── diabetes_prediction.py
├── README.md
└── requirements.txt
```

## Future Improvements

* Hyperparameter tuning for improved accuracy.
* Use advanced models such as Random Forest and XGBoost.
* Build a Flask or Streamlit web application.
* Add model persistence using Pickle.
* Deploy the model on cloud platforms.

## Applications

* Healthcare Decision Support Systems
* Early Diabetes Detection
* Clinical Data Analysis
* Medical Research

## Author

Mun Dev Anuj Tomar
