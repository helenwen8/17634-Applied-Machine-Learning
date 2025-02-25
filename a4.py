import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt 

data = pd.read_csv('music_education_dataset_new.csv')
# drop columns that do not matter
data = data.drop(columns=["Student_ID", "Timestamp"])

# preprocessing: one hot encoding for all categorical data
categoricalFeatures = ["Gender", "Class_Level", "Instrument", "Lesson_Type"]
numericalFeatures = ['Age', 'Accuracy (%)', 'Rhythm (%)', 'Tempo (BPM)', 'Pitch_Accuracy', 
                      'Volume (dB)', 'Heart_Rate (BPM)', 'Stress_Level', 'Engagement_Level', 
                      'Focus_Time (min)', 'Behavior_Score', 'Skill_Development', 'Engagement_Score']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categoricalFeatures),
    ('num', StandardScaler(), numericalFeatures)  # Normalize continuous features
])

# get test data 
X = data.drop(columns=["Performance_Score"])
y = data["Performance_Score"]
# split test into 70% train
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# split test into 15% test and 15% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define models

def decisionTree():
    decision_tree = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    decision_tree.fit(X_train, y_train)

    testModel(decision_tree, "Decision Tree Regressor")

    plt.figure(figsize=(20, 10)) 
    plot_tree(
        decision_tree["regressor"],
        filled=True,
        rounded=True
    )
    plt.show() 


def randomForest():
    random_forest = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    random_forest.fit(X_train, y_train)

    testModel(random_forest, "Random Forest Regressor")

def neuralNetwork():
    neural_network = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42))
    ])

    neural_network.fit(X_train, y_train)

    testModel(neural_network, "Multi-layer Perceptron regressor")


def testModel(model, name):
    y_val_pred = model.predict(X_val)
    print(name + " Performance on Validation Set:")
    print(f"MAE: {mean_absolute_error(y_val, y_val_pred)}")
    print(f"MSE: {mean_squared_error(y_val, y_val_pred)}")
    print(f"R² Score: {r2_score(y_val, y_val_pred)}")

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    print(name + " Performance on Test Set:")
    print(f"MAE: {mean_absolute_error(y_test, y_test_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_test_pred)}")
    print(f"R² Score: {r2_score(y_test, y_test_pred)}")


decisionTree()
randomForest()
neuralNetwork()
