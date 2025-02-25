import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error


data = pd.read_csv('music_education_dataset_new.csv')

# some EDA

with pd.option_context('display.max_columns', 40):
    print(data.describe(include="all"))
# data.info()

# Apply the default theme
sns.set_theme()

def EDA():
    '''
    commented out codes are explorations that didn't yield interesting results
    '''
    # heart reate vs performance score, no specific prediction
    # sns.lmplot(data=data, x="Heart_Rate (BPM)", y="Performance_Score", hue="Gender")


    # what if its diff than ages/ gender?
    # sns.lmplot(data=data, x="Focus_Time (min)", y="Performance_Score", hue="Gender")

    # sns.catplot(data=data, kind="swarm", x="Class_Level", y="Performance_Score", hue="Gender")

    # sns.lmplot(data=data, x="Age", y="Performance_Score", hue="Gender")

    # sns.lmplot(data=data, x="Stress_Level", y="Performance_Score", hue="Gender")

    # sns.catplot(data=data, kind="swarm", x="Instrument", y="Performance_Score", hue="Gender")

    # engagement level vs performance score, no necessarily changes
    a = sns.relplot(data=data, x="Engagement_Level", y="Performance_Score", hue="Gender")
    a.set(xlabel="Engagement Level (out of 10)", ylabel="Performance Score (out of 100)", title="Engagement Level vs Performance Score")

    # as people become more advanced, they engage less
    b = sns.catplot(data=data, kind="violin", x="Class_Level", y="Engagement_Level", order=["Beginner", "Intermediate", "Advanced"] )
    b.set(yticks=np.arange(1,10,1))
    b.set(xlabel="Class Level", ylabel="Engagement Level (out of 10)", title="Class Level vs Engagement Level")

    # sns.catplot(data=data, kind="bar", x="Lesson_Type", y="Stress_Level")

    # drum is a little more stress reducing
    c = sns.catplot(data=data, kind="box", x="Instrument", y="Stress_Level")
    c.set(yticks = np.arange(1,10,1))
    c.set(xlabel="Instrument", ylabel="Stress Level (out of 10)", title="Instrument vs Stress Level")

    # sns.catplot(data=data, kind="swarm", x="Instrument", y="Stress_Level")

    # sns.relplot(data=data, x="Age", y="Stress_Level", hue="Gender")

    plt.show()

# Pt 2: linear regression

def part2():
    features = data[[
                    "Accuracy (%)",
                    "Rhythm (%)"
                    ]]

    y = data["Performance_Score"]

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.25) 
    
    # Splitting the data into training and testing data 
    linRgr = LinearRegression() 

    linRgr.fit(X_train,y_train)

    predictions = linRgr.predict(X_test) 

    r2 =  mean_squared_error(y_test, predictions)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, predictions, alpha=0.7, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)  # Ideal line
    plt.xlabel("Observed Values (y_test)")
    plt.ylabel("Predicted Values (y_pred)")
    plt.title(f"Predicted vs. Observed Values\nR² = {r2:.3f}")
    plt.grid(True)
    plt.show()


# pt 3: go through all the features 

def part3():
    # get all the features that an do this job
    linFeatures = data.drop(columns=["Student_ID", "Gender", "Class_Level", "Instrument", "Lesson_Type", "Timestamp", "Performance_Score"])
    y = data["Performance_Score"]

    results = np.array([])

    for i in range(len(linFeatures.columns)):
        X_train, X_test, y_train, y_test = train_test_split(linFeatures.iloc[:, 0:i + 1], y, test_size = 0.25) 

        linRgr = LinearRegression() 

        linRgr.fit(X_train,y_train)

        predictions = linRgr.predict(X_test) 
        r2 = mean_squared_error(y_test, predictions)
        results = np.append(results, r2)


    plt.figure(figsize=(8,6))
    plt.scatter(np.arange(len(linFeatures.columns)), results, alpha=0.7, color='blue')
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Model Complexity vs Model Accuracy")
    plt.grid(True)
    plt.show()


EDA()
# part2()
# part3()