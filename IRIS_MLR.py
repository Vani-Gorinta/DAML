import pandas as pd
import numpy as np

# Load the dataset
iris = pd.read_csv("IRIS.csv")

x = iris[['sepal_length', 'sepal_width']]
y = iris.petal_length

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

training = LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
print(y_pred)