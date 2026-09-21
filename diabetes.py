import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import svm

dataset_path = Path(__file__).with_name("diabetes_prediction_dataset.csv")
diabetes_dataset = pd.read_csv(dataset_path)

print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['diabetes'].value_counts())
print(diabetes_dataset.groupby('diabetes').mean(numeric_only=True))

X = diabetes_dataset.drop(columns='diabetes')
Y = diabetes_dataset['diabetes']

print(X)
print(Y)

numeric_features = X.select_dtypes(include='number').columns
categorical_features = X.select_dtypes(exclude='number').columns
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

classifier = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', svm.SVC(kernel='linear')),
])

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data : ', test_data_accuracy)

input_data = pd.DataFrame([{
    'gender': 'Male',
    'age': 51,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'never',
    'bmi': 25.8,
    'HbA1c_level': 6.0,
    'blood_glucose_level': 166,
}])

prediction = classifier.predict(input_data)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
